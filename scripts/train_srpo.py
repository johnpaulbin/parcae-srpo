"""
SRPO training for recurrent-depth Gemma E2B.

Self-Reflective Policy Optimization (ICML 2026, arxiv 2604.02288):
  Correct samples → GRPO branch (sequence-level group-relative advantage)
  Failed samples  → SDPO branch (self-distillation from feedback-conditioned teacher)
  Entropy-aware dynamic weighting suppresses unreliable teacher predictions.

Parcae training recipe (arxiv 2604.12946):
  Variable depth sampling T ~ Poisson(μ_rec)
  Truncated BPTT through μ_bwd = ceil(T * bptt_ratio)
  LTI-stable injection ρ(A) < 1 guaranteed by construction.

Target hardware: 2× RTX 5090 (32GB each), CUDA 13.0.

Run: torchrun --nproc_per_node=2 train_srpo.py
"""

import os
import sys
import math
import time
import json
import random
import subprocess
from dataclasses import dataclass, field
from typing import Optional, Generator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

if os.environ.get("LOAD_BACKEND", "").strip().lower() == "unsloth":
    try:
        import unsloth as _unsloth  # noqa: F401
    except Exception as exc:
        print(
            "WARNING: Unsloth failed to import before Transformers; "
            "falling back to Transformers 4-bit loading. "
            f"Original error: {exc}",
            file=sys.stderr,
        )
        os.environ["LOAD_BACKEND"] = "transformers"

from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from parcae import RecurrentDepthGemma, RecurrentDepthConfig

# ── Configuration ──────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # ── Model ──
    model_name: str = "google/gemma-4-E2B-it"
    model_path: Optional[str] = None       # set to local cache path to skip download
    prelude_layers: int = 12               # E2B: 35 layers → 12 + 11 + 12 split
    n_recurrent_layers: int = 11
    coda_layers: int = 12
    lora_rank: int = 16
    loop_embedding_dim: int = 128
    load_backend: str = "transformers"     # transformers | unsloth
    load_in_4bit: bool = False
    max_seq_length: int = 2048
    fast_inference: bool = False
    use_activation_checkpointing: bool = True

    # ── Recurrent depth (Parcae) ──
    poisson_mean: int = 2                  # recurrent-depth sampling mean
    min_loops: int = 1
    max_loops: int = 8
    bptt_ratio: float = 0.5                # bwd depth = ceil(T * bptt_ratio)

    # ── SRPO algorithm ──
    group_size: int = 4                    # G completions per prompt
    max_prompt_tokens: int = 512
    max_response_tokens: int = 512
    gen_temperature: float = 1.2           # high enough to get mixed groups
    clip_epsilon: float = 0.2
    clip_epsilon_high: float = 0.28        # GSPO Clip-Higher
    kl_beta: float = 0.0
    entropy_weight: float = 0.01           # SRPO entropy-aware weighting
    grpo_forward_batch_size: int = 1       # memory guard for long code outputs
    sdpo_forward_batch_size: int = 1       # memory guard for teacher-correction KL
    max_train_sequence_tokens: int = 0     # 0 = train on full generated sequence

    # ── Optimization ──
    micro_batch_size: int = 2              # prompts per micro-batch (per GPU)
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # ── Training schedule ──
    total_steps: int = 1000
    save_every: int = 200
    eval_every: int = 50
    log_every: int = 10
    sample_log_every: int = 10
    sample_log_prompts: int = 1             # log all G completions for this many prompts
    sample_log_path: str = "runs/samples.jsonl"
    seed: int = 42

    # ── Dataset ──
    dataset: str = "humaneval_mbpp_mix"    # humaneval_mbpp_mix | mbpp | humaneval | bigcodebench | builtin
    max_prompts: int = 500

    # ── Distributed ──
    world_size: int = 2                    # set by torchrun


# ── Dataset ────────────────────────────────────────────────────────────

class CodeProblemDataset:
    """Streaming dataset of coding problems with verifiable unit tests."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self._items = self._load()

    def _load(self) -> list[dict]:
        try:
            from datasets import load_dataset

            name = self.cfg.dataset
            if name == "mbpp":
                ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
                items = []
                for item in ds:
                    # MBPP: test_list is a list of assertion strings, join them
                    tests = item["test_list"]
                    test_str = "\n".join(tests)
                    # Extract entry point from first test (e.g., 'assert funcname(...' -> 'funcname')
                    entry = tests[0].split("assert ")[1].split("(")[0] if tests else "solution"
                    # Build full prompt: description + function signature from code
                    func_sig = item["code"].split("\n")[0]  # first line is 'def funcname(args):'
                    full_prompt = f"{item['prompt']}\n{func_sig}"
                    items.append({
                        "prompt": full_prompt,
                        "test": test_str,
                        "entry": entry,
                    })
            elif name == "humaneval":
                ds = load_dataset("openai/openai_humaneval", split="test")
                items = []
                for item in ds:
                    entry = item.get("entry_point", "solution")
                    test = item.get("test", "")
                    if "def check(" in test:
                        test = f"{test}\ncheck({entry})"
                    items.append({
                        "prompt": item["prompt"],
                        "test": test,
                        "entry": entry,
                    })
            elif name in {"humaneval_mbpp_mix", "mix"}:
                he = load_dataset("openai/openai_humaneval", split="test")
                mbpp = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
                he_items = [
                    {
                        "prompt": item["prompt"],
                        "test": (
                            f"{item.get('test', '')}\ncheck({item.get('entry_point', 'solution')})"
                            if "def check(" in item.get("test", "")
                            else item.get("test", "")
                        ),
                        "entry": item.get("entry_point", "solution"),
                    }
                    for item in he
                ]
                mbpp_items = []
                for item in mbpp:
                    tests = item["test_list"]
                    test_str = "\n".join(tests)
                    entry = tests[0].split("assert ")[1].split("(")[0] if tests else "solution"
                    func_sig = item["code"].split("\n")[0]
                    mbpp_items.append({
                        "prompt": f"{item['prompt']}\n{func_sig}",
                        "test": test_str,
                        "entry": entry,
                    })
                rng = random.Random(self.cfg.seed)
                rng.shuffle(he_items)
                rng.shuffle(mbpp_items)
                n_he = min(len(he_items), max(1, int(self.cfg.max_prompts * 0.8)))
                n_mbpp = max(0, self.cfg.max_prompts - n_he)
                items = he_items[:n_he] + mbpp_items[:n_mbpp]
            elif name == "bigcodebench":
                ds = load_dataset("bigcode/bigcodebench", split="v0.1")
                items = [{"prompt": i["prompt"], "test": i.get("test",""), "entry": i.get("entry_point","solution")} for i in ds]
            elif name == "builtin":
                return _builtin_problems(self.cfg.max_prompts)
            else:
                raise ValueError(name)

            rng = random.Random(self.cfg.seed)
            rng.shuffle(items)
            return items[: self.cfg.max_prompts]
        except Exception:
            return _builtin_problems(self.cfg.max_prompts)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int) -> dict:
        return self._items[i]

    def shuffle(self):
        rng = random.Random(self.cfg.seed + int(time.time() * 1e6) % 10000)
        rng.shuffle(self._items)


def _builtin_problems(n: int) -> list[dict]:
    """Fallback: 50 diverse coding problems with multi-assert tests."""
    problems = [
        # ── Math / numbers ──
        {"prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n", "test": "assert add(2,3)==5; assert add(-1,1)==0; assert add(0,0)==0\n", "entry": "add"},
        {"prompt": "def factorial(n):\n    \"\"\"Return n! recursively.\"\"\"\n", "test": "assert factorial(0)==1; assert factorial(1)==1; assert factorial(5)==120; assert factorial(7)==5040\n", "entry": "factorial"},
        {"prompt": "def is_prime(n):\n    \"\"\"Return True if n is prime.\"\"\"\n", "test": "assert is_prime(2); assert is_prime(17); assert is_prime(97); assert not is_prime(1); assert not is_prime(4); assert not is_prime(100)\n", "entry": "is_prime"},
        {"prompt": "def gcd(a, b):\n    \"\"\"Return greatest common divisor.\"\"\"\n", "test": "assert gcd(48,18)==6; assert gcd(7,13)==1; assert gcd(100,10)==10; assert gcd(0,5)==5\n", "entry": "gcd"},
        {"prompt": "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number (0-indexed).\"\"\"\n", "test": "assert fibonacci(0)==0; assert fibonacci(1)==1; assert fibonacci(2)==1; assert fibonacci(10)==55; assert fibonacci(20)==6765\n", "entry": "fibonacci"},
        {"prompt": "def power(x, n):\n    \"\"\"Return x raised to power n (integer).\"\"\"\n", "test": "assert power(2,10)==1024; assert power(3,0)==1; assert power(5,3)==125; assert power(2,-1)==0.5\n", "entry": "power"},
        {"prompt": "def is_even(n):\n    \"\"\"Return True if n is even.\"\"\"\n", "test": "assert is_even(2); assert is_even(0); assert is_even(-4); assert not is_even(1); assert not is_even(99)\n", "entry": "is_even"},
        {"prompt": "def sum_of_digits(n):\n    \"\"\"Return sum of decimal digits of positive int n.\"\"\"\n", "test": "assert sum_of_digits(123)==6; assert sum_of_digits(0)==0; assert sum_of_digits(9999)==36\n", "entry": "sum_of_digits"},
        {"prompt": "def is_perfect_square(n):\n    \"\"\"Return True if n is a perfect square.\"\"\"\n", "test": "assert is_perfect_square(0); assert is_perfect_square(1); assert is_perfect_square(16); assert is_perfect_square(100); assert not is_perfect_square(2); assert not is_perfect_square(99)\n", "entry": "is_perfect_square"},
        {"prompt": "def lcm(a, b):\n    \"\"\"Return least common multiple of a and b.\"\"\"\n", "test": "assert lcm(4,6)==12; assert lcm(7,13)==91; assert lcm(1,99)==99; assert lcm(10,10)==10\n", "entry": "lcm"},

        # ── Strings ──
        {"prompt": "def is_palindrome(s):\n    \"\"\"Return True if s reads the same backward.\"\"\"\n", "test": "assert is_palindrome('racecar'); assert is_palindrome(''); assert is_palindrome('a'); assert not is_palindrome('hello'); assert not is_palindrome('ab')\n", "entry": "is_palindrome"},
        {"prompt": "def count_vowels(s):\n    \"\"\"Return number of vowels in s (case-insensitive).\"\"\"\n", "test": "assert count_vowels('hello')==2; assert count_vowels('HELLO')==2; assert count_vowels('xyz')==0; assert count_vowels('aeiou')==5\n", "entry": "count_vowels"},
        {"prompt": "def char_frequency(s):\n    \"\"\"Return dict of character frequencies (case-sensitive).\"\"\"\n", "test": "assert char_frequency('aba')=={'a':2,'b':1}; assert char_frequency('')=={}; assert char_frequency('zzz')=={'z':3}\n", "entry": "char_frequency"},
        {"prompt": "def anagram(s1, s2):\n    \"\"\"Return True if s1 and s2 are anagrams.\"\"\"\n", "test": "assert anagram('listen','silent'); assert anagram('',''); assert not anagram('hello','world'); assert not anagram('a','ab')\n", "entry": "anagram"},
        {"prompt": "def longest_common_prefix(strs):\n    \"\"\"Return longest common prefix of list of strings.\"\"\"\n", "test": "assert longest_common_prefix(['flower','flow','flight'])=='fl'; assert longest_common_prefix(['dog','racecar','car'])==''; assert longest_common_prefix(['a'])=='a'\n", "entry": "longest_common_prefix"},
        {"prompt": "def reverse_string(s):\n    \"\"\"Return reversed copy of string.\"\"\"\n", "test": "assert reverse_string('hello')=='olleh'; assert reverse_string('')==''; assert reverse_string('a')=='a'\n", "entry": "reverse_string"},
        {"prompt": "def capitalize_words(s):\n    \"\"\"Return string with first letter of each word capitalized.\"\"\"\n", "test": "assert capitalize_words('hello world')=='Hello World'; assert capitalize_words('a b c')=='A B C'; assert capitalize_words('')==''\n", "entry": "capitalize_words"},
        {"prompt": "def remove_vowels(s):\n    \"\"\"Return s with all vowels removed (case-insensitive).\"\"\"\n", "test": "assert remove_vowels('hello')=='hll'; assert remove_vowels('AEIOU')==''; assert remove_vowels('xyz')=='xyz'\n", "entry": "remove_vowels"},
        {"prompt": "def word_count(s):\n    \"\"\"Return number of words in s (split by whitespace).\"\"\"\n", "test": "assert word_count('hello world')==2; assert word_count('  a  b  ')==2; assert word_count('')==0\n", "entry": "word_count"},

        # ── Lists / arrays ──
        {"prompt": "def reverse_list(lst):\n    \"\"\"Return reversed copy of list.\"\"\"\n", "test": "assert reverse_list([1,2,3])==[3,2,1]; assert reverse_list([])==[]; assert reverse_list([1])==[1]; assert reverse_list(['a','b'])==['b','a']\n", "entry": "reverse_list"},
        {"prompt": "def binary_search(arr, x):\n    \"\"\"Return index of x in sorted arr, or -1.\"\"\"\n", "test": "assert binary_search([1,3,5,7],5)==2; assert binary_search([1,3,5,7],4)==-1; assert binary_search([],1)==-1; assert binary_search([1],1)==0\n", "entry": "binary_search"},
        {"prompt": "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n", "test": "assert merge_sorted([1,3],[2,4])==[1,2,3,4]; assert merge_sorted([],[1])==[1]; assert merge_sorted([5,6],[1,2])==[1,2,5,6]\n", "entry": "merge_sorted"},
        {"prompt": "def max_subarray(nums):\n    \"\"\"Kadane's algorithm: max subarray sum.\"\"\"\n", "test": "assert max_subarray([-2,1,-3,4,-1,2,1,-5,4])==6; assert max_subarray([1])==1; assert max_subarray([-1,-2,-3])==-1; assert max_subarray([5,4,-1,7,8])==23\n", "entry": "max_subarray"},
        {"prompt": "def two_sum(nums, target):\n    \"\"\"Return indices of two numbers summing to target.\"\"\"\n", "test": "assert set(two_sum([2,7,11,15],9))=={0,1}; assert set(two_sum([3,2,4],6))=={1,2}; assert set(two_sum([3,3],6))=={0,1}\n", "entry": "two_sum"},
        {"prompt": "def remove_duplicates(lst):\n    \"\"\"Return list with duplicates removed, preserving order.\"\"\"\n", "test": "assert remove_duplicates([1,2,2,3,1])==[1,2,3]; assert remove_duplicates([])==[]; assert remove_duplicates([1,1,1])==[1]\n", "entry": "remove_duplicates"},
        {"prompt": "def rotate_list(lst, k):\n    \"\"\"Rotate list right by k positions.\"\"\"\n", "test": "assert rotate_list([1,2,3,4,5],2)==[4,5,1,2,3]; assert rotate_list([1,2,3],0)==[1,2,3]; assert rotate_list([1,2,3],3)==[1,2,3]; assert rotate_list([1,2,3],1)==[3,1,2]\n", "entry": "rotate_list"},
        {"prompt": "def flatten(lst):\n    \"\"\"Flatten a nested list one level deep.\"\"\"\n", "test": "assert flatten([[1,2],[3,4]])==[1,2,3,4]; assert flatten([])==[]; assert flatten([[1],[],[2,3]])==[1,2,3]\n", "entry": "flatten"},
        {"prompt": "def sort_by_length(words):\n    \"\"\"Sort list of words by length ascending.\"\"\"\n", "test": "assert sort_by_length(['a','abc','ab'])==['a','ab','abc']; assert sort_by_length([])==[]; assert sort_by_length(['xyz','a'])==['a','xyz']\n", "entry": "sort_by_length"},
        {"prompt": "def list_product(nums):\n    \"\"\"Return product of all numbers in list.\"\"\"\n", "test": "assert list_product([1,2,3,4])==24; assert list_product([5])==5; assert list_product([])==1; assert list_product([0,1,2])==0\n", "entry": "list_product"},
        {"prompt": "def running_sum(nums):\n    \"\"\"Return list of running totals (prefix sums).\"\"\"\n", "test": "assert running_sum([1,2,3,4])==[1,3,6,10]; assert running_sum([1])==[1]; assert running_sum([])==[]\n", "entry": "running_sum"},
        {"prompt": "def find_min_max(nums):\n    \"\"\"Return (min, max) tuple from list. Assume non-empty.\"\"\"\n", "test": "assert find_min_max([3,1,4,1,5])==(1,5); assert find_min_max([7])==(7,7); assert find_min_max([-5,0,5])==(-5,5)\n", "entry": "find_min_max"},
        {"prompt": "def count_occurrences(lst, x):\n    \"\"\"Return number of times x appears in lst.\"\"\"\n", "test": "assert count_occurrences([1,2,2,3,2],2)==3; assert count_occurrences([],1)==0; assert count_occurrences([1,2,3],4)==0\n", "entry": "count_occurrences"},
        {"prompt": "def interleave(a, b):\n    \"\"\"Interleave two lists: [a0,b0,a1,b1,...]. Assume equal length.\"\"\"\n", "test": "assert interleave([1,2,3],[4,5,6])==[1,4,2,5,3,6]; assert interleave([],[])==[]; assert interleave(['a'],['b'])==['a','b']\n", "entry": "interleave"},
        {"prompt": "def chunk_list(lst, size):\n    \"\"\"Split lst into chunks of given size.\"\"\"\n", "test": "assert chunk_list([1,2,3,4,5],2)==[[1,2],[3,4],[5]]; assert chunk_list([1],3)==[[1]]; assert chunk_list([],1)==[]\n", "entry": "chunk_list"},

        # ── Dictionaries / sets ──
        {"prompt": "def merge_dicts(a, b):\n    \"\"\"Merge two dicts; b overwrites a on conflict.\"\"\"\n", "test": "assert merge_dicts({'a':1},{'b':2})=={'a':1,'b':2}; assert merge_dicts({'a':1},{'a':2})=={'a':2}; assert merge_dicts({},{})=={}\n", "entry": "merge_dicts"},
        {"prompt": "def invert_dict(d):\n    \"\"\"Invert dict: keys become values and vice versa. Assume unique values.\"\"\"\n", "test": "assert invert_dict({'a':1,'b':2})=={1:'a',2:'b'}; assert invert_dict({})=={}; assert invert_dict({'x':10})=={10:'x'}\n", "entry": "invert_dict"},
        {"prompt": "def set_union(a, b):\n    \"\"\"Return sorted list of elements in either set a or b.\"\"\"\n", "test": "assert set_union({1,2},{2,3})==[1,2,3]; assert set_union(set(),{1})==[1]; assert set_union({},set())==[]\n", "entry": "set_union"},
        {"prompt": "def set_intersection(a, b):\n    \"\"\"Return sorted list of elements in both sets.\"\"\"\n", "test": "assert set_intersection({1,2,3},{2,3,4})==[2,3]; assert set_intersection({1},{2})==[]; assert set_intersection(set(),{1})==[]\n", "entry": "set_intersection"},
        {"prompt": "def set_difference(a, b):\n    \"\"\"Return sorted list of elements in a but not in b.\"\"\"\n", "test": "assert set_difference({1,2,3},{2})==[1,3]; assert set_difference({1},{1})==[]; assert set_difference(set(),{1})==[]\n", "entry": "set_difference"},

        # ── Classes / OOP ──
        {"prompt": "class Counter:\n    \"\"\"Count from 0, incrementing by 1 each call.\"\"\"\n    def __init__(self):\n        self.n = 0\n    def next(self):\n", "test": "c=Counter(); assert c.next()==0; assert c.next()==1; assert c.next()==2\n", "entry": "Counter"},
        {"prompt": "class Stack:\n    \"\"\"Simple stack with push, pop, peek, is_empty.\"\"\"\n    def __init__(self):\n        self._items = []\n    def push(self, x):\n", "test": "s=Stack(); s.push(1); s.push(2); assert s.peek()==2; assert s.pop()==2; assert s.pop()==1; assert s.is_empty()\n", "entry": "Stack"},
        {"prompt": "class Queue:\n    \"\"\"Simple queue with enqueue, dequeue, peek, is_empty.\"\"\"\n    def __init__(self):\n        self._items = []\n    def enqueue(self, x):\n", "test": "q=Queue(); q.enqueue(1); q.enqueue(2); assert q.peek()==1; assert q.dequeue()==1; assert q.dequeue()==2; assert q.is_empty()\n", "entry": "Queue"},

        # ── Recursion ──
        {"prompt": "def sum_nested(lst):\n    \"\"\"Return sum of all integers in a nested list (any depth).\"\"\"\n", "test": "assert sum_nested([1,[2,[3,4]],5])==15; assert sum_nested([])==0; assert sum_nested([1,2,3])==6\n", "entry": "sum_nested"},
        {"prompt": "def tree_depth(obj):\n    \"\"\"Return max nesting depth of a list. A plain value has depth 0, [] has depth 1.\"\"\"\n", "test": "assert tree_depth([[]])==2; assert tree_depth([1,[2,[3]]])==3; assert tree_depth(5)==0; assert tree_depth([])==1\n", "entry": "tree_depth"},

        # ── Search / sort ──
        {"prompt": "def linear_search(arr, x):\n    \"\"\"Return first index of x in arr, or -1.\"\"\"\n", "test": "assert linear_search([5,3,1,4],3)==1; assert linear_search([5,3,1,4],6)==-1; assert linear_search([],1)==-1\n", "entry": "linear_search"},
        {"prompt": "def bubble_sort(arr):\n    \"\"\"Sort list in place using bubble sort. Return the sorted list.\"\"\"\n", "test": "assert bubble_sort([3,1,2])==[1,2,3]; assert bubble_sort([])==[]; assert bubble_sort([5,5,1])==[1,5,5]\n", "entry": "bubble_sort"},

        # ── String manipulation (harder) ──
        {"prompt": "def compress_string(s):\n    \"\"\"Basic run-length encoding: 'aaabb' -> 'a3b2'.\"\"\"\n", "test": "assert compress_string('aaabb')=='a3b2'; assert compress_string('')==''; assert compress_string('abc')=='a1b1c1'\n", "entry": "compress_string"},
        {"prompt": "def longest_word(sentence):\n    \"\"\"Return the longest word in a sentence (by length). On tie, return first.\"\"\"\n", "test": "assert longest_word('the quick brown fox')=='quick'; assert longest_word('a')=='a'; assert longest_word('')==''\n", "entry": "longest_word"},
        {"prompt": "def is_substring(s, sub):\n    \"\"\"Return True if sub appears in s (case-sensitive). Do NOT use 'in'.\"\"\"\n", "test": "assert is_substring('hello','ell'); assert is_substring('abc','abc'); assert not is_substring('abc','abcd'); assert not is_substring('','a')\n", "entry": "is_substring"},
        {"prompt": "def title_case(s):\n    \"\"\"Convert to title case: first letter upper, rest lower, per word.\"\"\"\n", "test": "assert title_case('hello world')=='Hello World'; assert title_case('HELLO')=='Hello'; assert title_case('a b c')=='A B C'\n", "entry": "title_case"},
    ]
    rng = random.Random(42)
    items = []
    while len(items) < n:
        items.append(rng.choice(problems))
    return items


# ── Reward ─────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    """Extract Python function from model completion. Robust to missing fences."""
    # Try markdown fences first
    if "```python" in text:
        parts = text.split("```python", 1)
        if len(parts) > 1:
            code = parts[1].split("```", 1)[0]
            if code.strip():
                return code.strip()
    if "```" in text:
        parts = text.split("```", 1)
        if len(parts) > 1:
            code = parts[1].split("```", 1)[0]
            if code.strip():
                return code.strip()
    # Fallback: return raw text (model may output code directly)
    return text.strip()


def verify(code: str, test: str, timeout: float = 10.0) -> tuple[float, str]:
    """Run code + tests in subprocess. Returns (reward 0/1, feedback)."""
    script_lines = [code, ""]
    script_lines.append("try:")
    for t in test.strip().split("\n"):
        if t.strip():
            script_lines.append(f"    {t}")
    script_lines.append("    print('__ALL_PASSED__')")
    script_lines.append("except Exception as exc:")
    script_lines.append("    import traceback")
    script_lines.append("    print('__FAILED__')")
    script_lines.append("    traceback.print_exc()")
    full_script = "\n".join(script_lines)
    try:
        r = subprocess.run([sys.executable, "-I", "-c", full_script], capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr
        if "__ALL_PASSED__" in out:
            return 1.0, "All tests passed."
        return 0.0, out.strip()[:600]
    except subprocess.TimeoutExpired:
        return 0.0, "Timed out."
    except Exception as e:
        return 0.0, str(e)[:600]


# ── SRPO losses ────────────────────────────────────────────────────────

def grpo_loss(
    log_probs: torch.Tensor,         # (B, L)
    log_probs_old: torch.Tensor,     # (B, L)
    rewards: torch.Tensor,           # (B,)
    response_mask: torch.Tensor,     # (B, L)
    epsilon: float,
    epsilon_high: float,
    group_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    GRPO branch: sequence-level group-relative advantage.  One scalar
    importance ratio per sequence (GSPO sequence-level formulation),
    clipped PPO-style with asymmetric Clip-Higher.

    B is the group size (all completions for one or more prompts).
    """
    B = rewards.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=rewards.device)

    # sequence-level log‑prob (mean over response tokens)
    n_tokens = response_mask.sum(dim=-1).clamp(min=1)       # (B,)
    seq_lp = (log_probs * response_mask).sum(dim=-1) / n_tokens
    seq_lp_old = (log_probs_old * response_mask).sum(dim=-1) / n_tokens

    # group-relative advantage.  Binary rewards must include failed and
    # successful completions from the same prompt; otherwise all-correct
    # subsets collapse to zero advantage.
    if group_ids is None:
        sigma = rewards.std(unbiased=False)
        A = (rewards - rewards.mean()) / (sigma + 1e-8)
        if sigma < 1e-8:
            A = torch.zeros_like(rewards)
    else:
        A = torch.zeros_like(rewards)
        for gid in torch.unique(group_ids):
            mask = group_ids == gid
            if mask.sum() < 2:
                continue
            group_rewards = rewards[mask]
            sigma = group_rewards.std(unbiased=False)
            if sigma < 1e-8:
                continue
            A[mask] = (group_rewards - group_rewards.mean()) / (sigma + 1e-8)

    # sequence-level importance ratio (length-normalized)
    rho = torch.exp(seq_lp - seq_lp_old)                          # (B,)

    # clipped surrogate (sequence level)
    rho_clip = torch.clamp(rho, 1.0 - epsilon, 1.0 + epsilon_high)
    surr = torch.min(rho * A, rho_clip * A)                       # (B,)

    # expand to token level: every token in sequence i gets the same
    # scalar surrogate weight
    per_token = surr.unsqueeze(-1) * response_mask                # (B, L)
    loss = -per_token.sum() / n_tokens.sum().clamp(min=1)
    return loss


def sdpo_loss(
    student_logits: torch.Tensor,    # (L, V) or (B, L, V)
    teacher_logits: torch.Tensor,    # (L, V) or (B, L, V)
    response_mask: torch.Tensor,     # (L,) or (B, L)
    entropy_weight: float,
) -> torch.Tensor:
    """
    SDPO branch: reverse KL with entropy-aware weighting.
    Supports batched (B, L, V) or single (L, V) inputs.
    Weight in (0, 1] per token; uncertain teacher predictions
    suppressed; confident ones emphasized.
    """
    if student_logits.dim() == 2:  # single sequence: (L, V)
        student_logits = student_logits.unsqueeze(0)   # (1, L, V)
        teacher_logits = teacher_logits.unsqueeze(0)
        response_mask = response_mask.unsqueeze(0)      # (1, L)
    active = response_mask > 0
    if not active.any():
        return student_logits.sum() * 0.0

    student_active = student_logits[active]
    teacher_active = teacher_logits[active]
    V = student_active.shape[-1]

    student_lp = F.log_softmax(student_active.float(), dim=-1).to(student_active.dtype)
    teacher_p = F.softmax(teacher_active.float(), dim=-1).to(student_active.dtype)

    # token-level reverse KL: KL(p_student || p_teacher)
    kl = F.kl_div(
        student_lp.float(), teacher_p.float(),
        reduction="none", log_target=False,
    ).sum(dim=-1).to(student_active.dtype)

    # entropy-aware weight
    teacher_log_p = F.log_softmax(teacher_active.float(), dim=-1).to(teacher_active.dtype)
    H = -(teacher_p.float() * teacher_log_p.float()).sum(dim=-1).to(teacher_active.dtype)
    H_max = math.log(V)
    w = 1.0 - entropy_weight * (H / H_max)

    weighted = kl * w
    return weighted.float().mean().to(weighted.dtype)


# ── Feedback ───────────────────────────────────────────────────────────

def build_feedback(
    failed_code: str,
    error: str,
    problem: str,
    correct_demos: list[str],
) -> str:
    """Build self-distillation feedback for a failed completion."""
    if correct_demos:
        demo = correct_demos[0]
        return (
            f"Your code:\n```python\n{failed_code}\n```\n\n"
            f"Error: {error}\n\n"
            f"Here is a working solution:\n```python\n{demo}\n```\n\n"
            f"Write a corrected version."
        )
    return (
        f"Your code:\n```python\n{failed_code}\n```\n\n"
        f"Error: {error}\n\nIdentify and fix the mistake."
    )


# ── Trainer ────────────────────────────────────────────────────────────

class SRPOTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(f"cuda:{self.local_rank}")

        if self.world_size > 1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)

        self._seed()
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_path or cfg.model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._build_model()
        self._build_optimizer()
        self.dataset = CodeProblemDataset(cfg)
        self.dataloader = DataLoader(self.dataset, batch_size=cfg.micro_batch_size, shuffle=True, collate_fn=lambda x: list(x))
        self.scaler = torch.amp.GradScaler('cuda')
        self.step = 0

    def _format_prompt(self, prompt: str) -> str:
        """Apply the model chat template when available."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return prompt

    def _seed(self):
        torch.manual_seed(self.cfg.seed + self.local_rank)
        np.random.seed(self.cfg.seed + self.local_rank)
        random.seed(self.cfg.seed + self.local_rank)

    def _build_model(self):
        rd = RecurrentDepthConfig(
            model_path=self.cfg.model_path or self.cfg.model_name,
            load_backend=self.cfg.load_backend,
            load_in_4bit=self.cfg.load_in_4bit,
            max_seq_length=self.cfg.max_seq_length,
            fast_inference=self.cfg.fast_inference,
            prelude_layers=self.cfg.prelude_layers,
            n_recurrent_layers=self.cfg.n_recurrent_layers,
            coda_layers=self.cfg.coda_layers,
            default_loops=self.cfg.poisson_mean,
            use_depth_lora=True,
            lora_rank=self.cfg.lora_rank,
            use_loop_embedding=True,
            loop_embedding_dim=self.cfg.loop_embedding_dim,
            use_activation_checkpointing=self.cfg.use_activation_checkpointing,
        )
        self.model = RecurrentDepthGemma(rd)
        self.model.load_pretrained()
        if self.model.quantized_backbone:
            self.model.move_trainable_modules(self.device)
        else:
            self.model.to(self.device)

        # freeze backbone, train injection + lora + loop emb
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.injection.train()
        for p in self.model.injection.parameters():
            p.requires_grad = True
        if self.model.depth_lora:
            self.model.depth_lora.train()
            for p in self.model.depth_lora.parameters():
                p.requires_grad = True

        # Parallel old-policy modules for GRPO importance sampling.
        # We swap MODULE REFERENCES (not tensor data) to avoid DDP inplace-op tracking.
        import copy
        self.injection_old = copy.deepcopy(self.model.injection)
        self.injection_old.to(self.device)
        for p in self.injection_old.parameters():
            p.requires_grad = False
        if self.model.depth_lora:
            self.depth_lora_old = copy.deepcopy(self.model.depth_lora)
            self.depth_lora_old.to(self.device)
            for p in self.depth_lora_old.parameters():
                p.requires_grad = False
        else:
            self.depth_lora_old = None

        # Wrap in DDP for multi-GPU.
        # Module-reference swaps (for old-policy log-probs) don't trigger DDP versioning
        # because they're Python object assignments, not tensor inplace ops.
        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=False
            )
            self._model_unwrapped = self.model.module
        else:
            self._model_unwrapped = self.model

        n_train = sum(p.numel() for p in self.trainable_params())
        n_total = sum(p.numel() for p in self._model_unwrapped.parameters())
        if self.local_rank == 0:
            print(f"Model loaded. {n_total/1e9:.2f}B total, {n_train:,} trainable")

    def trainable_params(self):
        """Generator over trainable parameters (works through DDP wrapper)."""
        m = self._model_unwrapped
        for p in m.injection.parameters():
            yield p
        if m.depth_lora:
            for p in m.depth_lora.parameters():
                yield p

    def _build_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.trainable_params(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

    # ── generation with log‑prob caching ───────────────────────────

    @torch.no_grad()
    def _generate(self, prompts: list[str], T: int) -> list[dict]:
        """Generate G completions per prompt.

        Generation is intentionally per prompt in the pure PyTorch path.  The
        recurrent KV implementation expects unpadded inputs; batching prompts
        of different lengths would otherwise train on pad-token context.
        """
        self._model_unwrapped._bptt_depth = None
        results = []
        G = self.cfg.group_size

        for b, prompt in enumerate(prompts):
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.max_prompt_tokens,
            ).to(self.device)
            prompt_ids = enc["input_ids"]
            prompt_len = prompt_ids.shape[1]

            for _ in range(G):
                gen_out = self._model_unwrapped.generate(
                    input_ids=prompt_ids,
                    max_new_tokens=self.cfg.max_response_tokens,
                    n_loops=T,
                    temperature=self.cfg.gen_temperature,
                    top_k=50,
                    return_logprobs=True,
                )
                if isinstance(gen_out, tuple):
                    full_ids, token_lps_batch = gen_out
                else:
                    full_ids = gen_out
                    token_lps_batch = None

                pl = prompt_len
                ids = full_ids[0]
                L = ids.shape[0]
                eos_mask = (ids[pl:] == self.tokenizer.eos_token_id)
                if eos_mask.any():
                    first_eos = eos_mask.nonzero(as_tuple=True)[0][0].item() + pl
                    ids = ids[:first_eos + 1]
                    L = ids.shape[0]

                text = self.tokenizer.decode(ids[pl:], skip_special_tokens=True)
                resp_mask = torch.zeros(L, device=self.device)
                resp_mask[pl:] = 1

                if token_lps_batch is not None:
                    gen_len = min(L - pl, token_lps_batch.shape[1])
                    token_lp = token_lps_batch[0, :gen_len]
                else:
                    token_lp = torch.zeros(0, device=self.device)

                results.append({
                    "text": text,
                    "full_ids": ids,
                    "token_lp": token_lp,
                    "resp_mask": resp_mask,
                    "prompt_len": pl,
                    "batch_idx": b,
                })
        return results

    def _training_view(
        self,
        ids: torch.Tensor,
        prompt_len: int,
    ) -> tuple[torch.Tensor, int]:
        """Return the token window used for backprop without changing logs.

        Generation and sample logging keep the full prompt/completion. This
        optional crop only bounds recurrent activation memory during GRPO/SDPO
        log-prob recomputation.
        """
        max_tokens = int(self.cfg.max_train_sequence_tokens or 0)
        if max_tokens <= 0 or ids.shape[0] <= max_tokens:
            return ids, prompt_len
        start = ids.shape[0] - max_tokens
        return ids[start:], max(0, prompt_len - start)

    def train_step(self, batch: list[dict], loss_scale: Optional[float] = None) -> dict:
        cfg = self.cfg
        G = cfg.group_size
        raw_prompts = [b["prompt"] for b in batch]
        prompts = [self._format_prompt(p) for p in raw_prompts]
        B = len(raw_prompts)

        # sample depth
        T = max(cfg.min_loops, min(cfg.max_loops, np.random.poisson(cfg.poisson_mean)))
        T_bwd = max(1, math.ceil(T * cfg.bptt_ratio))

        # Set BPTT depth on model; gradients only flow through last T_bwd iterations
        self._model_unwrapped._bptt_depth = T_bwd

        # snapshot old policy: save current trainable params into old-policy modules
        self._snapshot_old_policy()

        # generate + cache log‑probs
        if self.local_rank == 0:
            print(f"  [generate] T={T}, prompts={B}, G={G}...", flush=True)
        comps = self._generate(prompts, T)   # list of B*G dicts
        # Free GPU memory. Current-policy GRPO logprobs are recomputed with grad.
        for c in comps:
            c["full_ids"] = c["full_ids"].cpu()

        # verify
        for c in comps:
            b = c["batch_idx"]
            prob = batch[b]
            code = extract_code(c["text"])
            reward, fb = verify(code, prob["test"])
            c["reward"] = reward
            c["feedback"] = fb

        # ── build per‑prompt correct demonstrations ──
        correct_by_prompt = {b: [] for b in range(B)}
        for c in comps:
            if c["reward"] > 0:
                correct_by_prompt[c["batch_idx"]].append(c["text"])

        for c in comps:
            if c["reward"] <= 0:
                demos = correct_by_prompt[c["batch_idx"]]
                c["sdpo_feedback"] = build_feedback(c["text"], c["feedback"], raw_prompts[c["batch_idx"]], demos)

        samples = self._collect_samples(raw_prompts, prompts, comps)

        # --- GRPO branch: all samples, grouped by prompt ---
        correct = [c for c in comps if c["reward"] > 0]
        grpo_l = torch.tensor(0.0, device=self.device)
        policy_samples = comps
        if len(policy_samples) >= 2:
            for c in policy_samples:
                train_ids, train_pl = self._training_view(
                    c["full_ids"],
                    c["prompt_len"],
                )
                c["train_ids"] = train_ids
                c["train_prompt_len"] = train_pl
            L_max = max(c["train_ids"].shape[0] for c in policy_samples)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id or 0
            batch_ids = torch.full(
                (len(policy_samples), L_max),
                pad_id,
                dtype=torch.long,
                device=self.device,
            )
            batch_attn = torch.zeros(len(policy_samples), L_max, dtype=torch.long, device=self.device)
            batch_mask = torch.zeros(len(policy_samples), L_max, device=self.device)
            for j, c in enumerate(policy_samples):
                ids_view = c["train_ids"]
                L = ids_view.shape[0]
                PL = c["train_prompt_len"]
                batch_ids[j, :L] = ids_view.to(self.device)
                batch_attn[j, :L] = 1
                batch_mask[j, max(PL, 1):L] = 1
            rwd = torch.tensor([c["reward"] for c in policy_samples], device=self.device)
            group_ids = torch.tensor([c["batch_idx"] for c in policy_samples], device=self.device)

            # Current-policy log-probs must be recomputed with grad.  Chunking
            # keeps long code outputs viable on 40GB A100 Colab runtimes.
            cur_rows = []
            grpo_bs = max(1, cfg.grpo_forward_batch_size)
            for start in range(0, len(policy_samples), grpo_bs):
                end = min(start + grpo_bs, len(policy_samples))
                chunk_len = int(batch_attn[start:end].sum(dim=-1).max().item())
                ids_chunk = batch_ids[start:end, :chunk_len]
                attn_chunk = batch_attn[start:end, :chunk_len]
                logits_cur = self.model(
                    input_ids=ids_chunk,
                    attention_mask=attn_chunk,
                    n_loops=T,
                    return_logits=True,
                )
                chunk_rows = []
                for local_j, c in enumerate(policy_samples[start:end]):
                    L = c["train_ids"].shape[0]
                    PL = c["train_prompt_len"]
                    loss_start = max(PL, 1)
                    if L > loss_start:
                        gen_pos = torch.arange(loss_start, L, device=self.device)
                        selected = logits_cur[local_j, gen_pos - 1, :]
                        selected_lp = F.log_softmax(selected.float(), dim=-1).to(selected.dtype)
                        token_lp = selected_lp.gather(
                            -1,
                            batch_ids[start + local_j, gen_pos].unsqueeze(-1),
                        ).squeeze(-1)
                        row = torch.zeros(
                            L_max,
                            device=self.device,
                            dtype=token_lp.dtype,
                        ).scatter(0, gen_pos, token_lp)
                    else:
                        row = torch.zeros(L_max, device=self.device, dtype=logits_cur.dtype)
                    chunk_rows.append(row)
                cur_rows.append(torch.stack(chunk_rows))
                del logits_cur
            lp = torch.cat(cur_rows, dim=0)

            # Old-policy log-probs: forward with old-policy modules swapped in.
            # Uses context manager to guarantee restoration even on exception.
            old_rows = []
            with self._old_policy_ctx():
                with torch.no_grad():
                    for start in range(0, len(policy_samples), grpo_bs):
                        end = min(start + grpo_bs, len(policy_samples))
                        chunk_len = int(batch_attn[start:end].sum(dim=-1).max().item())
                        ids_chunk = batch_ids[start:end, :chunk_len]
                        attn_chunk = batch_attn[start:end, :chunk_len]
                        logits_old = self._model_unwrapped.forward(
                            input_ids=ids_chunk,
                            attention_mask=attn_chunk,
                            n_loops=T,
                            return_logits=True,
                        )
                        chunk_rows = []
                        for local_j, c in enumerate(policy_samples[start:end]):
                            L = c["train_ids"].shape[0]
                            PL = c["train_prompt_len"]
                            loss_start = max(PL, 1)
                            if L > loss_start:
                                gen_pos = torch.arange(loss_start, L, device=self.device)
                                selected = logits_old[local_j, gen_pos - 1, :]
                                selected_lp = F.log_softmax(selected.float(), dim=-1).to(selected.dtype)
                                token_lp = selected_lp.gather(
                                    -1,
                                    batch_ids[start + local_j, gen_pos].unsqueeze(-1),
                                ).squeeze(-1)
                                row = torch.zeros(
                                    L_max,
                                    device=self.device,
                                    dtype=token_lp.dtype,
                                ).scatter(0, gen_pos, token_lp)
                            else:
                                row = torch.zeros(L_max, device=self.device, dtype=logits_old.dtype)
                            chunk_rows.append(row)
                        old_rows.append(torch.stack(chunk_rows))
                        del logits_old
            lp_old = torch.cat(old_rows, dim=0)

            grpo_l = grpo_loss(
                lp,
                lp_old,
                rwd,
                batch_mask,
                cfg.clip_epsilon,
                cfg.clip_epsilon_high,
                group_ids=group_ids,
            )

        grpo_value = float(grpo_l.detach().item())
        if loss_scale is not None:
            if grpo_l.requires_grad:
                with torch.amp.autocast('cuda', enabled=False):
                    self.scaler.scale(grpo_l * loss_scale).backward()
            grpo_l = grpo_l.detach()
            if len(policy_samples) >= 2:
                del lp, lp_old, batch_ids, batch_attn, batch_mask, rwd, group_ids, cur_rows, old_rows
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # --- SDPO branch: failed samples (batched) ---
        failed = [c for c in comps if c["reward"] <= 0 and c.get("sdpo_feedback")]
        sdpo_l = torch.tensor(0.0, device=self.device)
        if failed:
            teacher_prompts = [
                self._format_prompt(f + "\n\nNow write the corrected code for:\n" + raw_prompts[c["batch_idx"]])
                for c, f in [(c, c["sdpo_feedback"]) for c in failed]
            ]
            # Teacher generates per prompt (old policy, no grad).  The pure
            # PyTorch recurrent generator expects unpadded prompts.
            # _old_policy_ctx is a zero-copy reference swap on the unwrapped
            # model (see definition at _old_policy_ctx). No state_dict copy,
            # no CUDA sync. Old-policy weights stored in same dtype (bf16)
            # as active model via load_state_dict in _snapshot_old_policy.
            teacher_ids = []
            tp_lens = []
            with self._old_policy_ctx():
                with torch.no_grad():
                    for teacher_prompt in teacher_prompts:
                        tp_enc = self.tokenizer(
                            teacher_prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=self.cfg.max_prompt_tokens,
                        ).to(self.device)
                        teacher_gen = self._model_unwrapped.generate(
                            input_ids=tp_enc["input_ids"],
                            max_new_tokens=self.cfg.max_response_tokens,
                            n_loops=T,
                            temperature=0.6,
                            top_k=50,
                        )
                        teacher_ids.append(teacher_gen[0])
                        tp_lens.append(tp_enc["input_ids"].shape[1])

            teacher_views = []
            tp_lens_view = []
            for ids, pl in zip(teacher_ids, tp_lens):
                ids_view, pl_view = self._training_view(ids, pl)
                teacher_views.append(ids_view)
                tp_lens_view.append(pl_view)

            TL = max(ids.shape[0] for ids in teacher_views)
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id or 0
            teacher_full_ids = torch.full(
                (len(teacher_views), TL),
                pad_id,
                dtype=torch.long,
                device=self.device,
            )
            teacher_attn = torch.zeros(len(teacher_views), TL, dtype=torch.long, device=self.device)
            for j, ids in enumerate(teacher_views):
                L = ids.shape[0]
                teacher_full_ids[j, :L] = ids.to(self.device)
                teacher_attn[j, :L] = 1

            sdpo_terms = []
            sdpo_den = torch.tensor(0.0, device=self.device)
            sdpo_num_value = 0.0
            sdpo_den_value = 0.0
            sdpo_bs = max(1, cfg.sdpo_forward_batch_size)
            total_resp_tokens = torch.tensor(
                sum(
                    max(0, int(teacher_attn[j].sum().item()) - max(int(tp_lens_view[j]), 1))
                    for j in range(len(failed))
                ),
                device=self.device,
                dtype=torch.float32,
            ).clamp_min(1.0)
            for start in range(0, len(failed), sdpo_bs):
                end = min(start + sdpo_bs, len(failed))
                chunk_len = int(teacher_attn[start:end].sum(dim=-1).max().item())
                ids_chunk = teacher_full_ids[start:end, :chunk_len]
                attn_chunk = teacher_attn[start:end, :chunk_len]

                # Student: current policy (DDP, grad enabled)
                stu_logits = self.model(
                    input_ids=ids_chunk,
                    attention_mask=attn_chunk,
                    n_loops=T,
                    return_logits=True,
                )
                # Teacher: old policy forward (no grad)
                with self._old_policy_ctx():
                    with torch.no_grad():
                        tea_logits = self._model_unwrapped.forward(
                            input_ids=ids_chunk,
                            attention_mask=attn_chunk,
                            n_loops=T,
                            return_logits=True,
                        )

                resp_mask = torch.zeros(end - start, chunk_len, device=self.device)
                for local_j, pl in enumerate(tp_lens_view[start:end]):
                    resp_mask[local_j, max(pl, 1):attn_chunk[local_j].sum().item()] = 1

                n_resp = resp_mask.sum()
                if n_resp > 0:
                    chunk_l = sdpo_loss(stu_logits, tea_logits, resp_mask, cfg.entropy_weight)
                    n_resp_value = float(n_resp.detach().item())
                    sdpo_num_value += float(chunk_l.detach().item()) * n_resp_value
                    sdpo_den_value += n_resp_value
                    if loss_scale is None:
                        sdpo_terms.append(chunk_l * n_resp)
                        sdpo_den = sdpo_den + n_resp
                    elif chunk_l.requires_grad:
                        chunk_weight = n_resp / total_resp_tokens
                        with torch.amp.autocast('cuda', enabled=False):
                            self.scaler.scale(chunk_l * chunk_weight * loss_scale).backward()
                    del chunk_l
                del stu_logits, tea_logits, resp_mask
                if loss_scale is not None and self.device.type == "cuda":
                    torch.cuda.empty_cache()

            if loss_scale is not None and sdpo_den_value > 0:
                sdpo_l = torch.tensor(sdpo_num_value / sdpo_den_value, device=self.device)
            elif sdpo_terms:
                sdpo_l = sum(sdpo_terms) / sdpo_den.clamp_min(1.0)

        sdpo_value = float(sdpo_l.detach().item())
        total_value = grpo_value + sdpo_value
        if loss_scale is None:
            total_loss = grpo_l + sdpo_l
            if not total_loss.requires_grad:
                total_loss = sum(p.sum() * 0.0 for p in self.trainable_params())
        else:
            total_loss = torch.tensor(total_value, device=self.device)

        metrics = {
            "total_loss": total_loss,
            "loss": total_value,
            "grpo_loss": grpo_value,
            "sdpo_loss": sdpo_value,
            "reward_mean": sum(c["reward"] for c in comps) / len(comps) if comps else 0,
            "T": T,
            "T_bwd": T_bwd,
            "rho": float(self._model_unwrapped.injection.compute_spectral_radius().detach().item()),
            "n_correct": len(correct),
            "n_failed": len(failed),
            "samples": samples,
        }
        return metrics

    def _collect_samples(
        self,
        raw_prompts: list[str],
        model_prompts: list[str],
        completions: list[dict],
    ) -> list[dict]:
        """Collect full, untruncated text samples for progress logging."""
        n_prompts = max(0, self.cfg.sample_log_prompts)
        if n_prompts == 0:
            return []

        prompt_ids = []
        for c in completions:
            batch_idx = c["batch_idx"]
            if batch_idx not in prompt_ids:
                if len(prompt_ids) >= n_prompts:
                    break
                prompt_ids.append(batch_idx)

        completion_counts: dict[int, int] = {}
        samples = []
        for c in completions:
            batch_idx = c["batch_idx"]
            completion_idx = completion_counts.get(batch_idx, 0)
            completion_counts[batch_idx] = completion_idx + 1
            if batch_idx not in prompt_ids:
                continue
            samples.append({
                "prompt_index": batch_idx,
                "completion_index": completion_idx,
                "prompt": raw_prompts[batch_idx],
                "model_prompt": model_prompts[batch_idx],
                "completion": c["text"],
                "reward": int(c.get("reward", 0)),
                "feedback": c.get("feedback", ""),
            })
        return samples

    def _log_samples(self, step_idx: int, metrics: dict):
        """Print and persist full prompt/completion text without truncation."""
        samples = metrics.get("samples") or []
        if not samples:
            return

        log_path = self.cfg.sample_log_path
        if log_path:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                for sample in samples:
                    record = {
                        "step": step_idx,
                        "T": metrics.get("T"),
                        "loss": metrics.get("loss"),
                        "grpo_loss": metrics.get("grpo_loss"),
                        "sdpo_loss": metrics.get("sdpo_loss"),
                        "reward_mean": metrics.get("reward_mean"),
                        "n_correct": metrics.get("n_correct"),
                        "n_failed": metrics.get("n_failed"),
                        "rho": metrics.get("rho"),
                        **sample,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        for sample in samples:
            loss_bits = (
                f"loss={metrics.get('loss', 0):.4f} "
                f"grpo={metrics.get('grpo_loss', 0):.4f} "
                f"sdpo={metrics.get('sdpo_loss', 0):.4f} "
                f"R_mean={metrics.get('reward_mean', 0):.4f}"
            )
            print(
                "\n".join([
                    "",
                    (
                        f"[sample step={step_idx} T={metrics.get('T')} "
                        f"prompt={sample['prompt_index']} "
                        f"completion={sample['completion_index']} "
                        f"reward={sample['reward']} {loss_bits}]"
                    ),
                    "--- PROMPT ---",
                    sample["prompt"],
                    "--- MODEL PROMPT (CHAT TEMPLATE APPLIED) ---",
                    sample["model_prompt"],
                    "--- MODEL COMPLETION ---",
                    sample["completion"],
                    "--- VERIFIER FEEDBACK ---",
                    sample["feedback"],
                    "--- END SAMPLE ---",
                ]),
                flush=True,
            )

    def _snapshot_old_policy(self):
        """Copy current trainable params into old-policy modules."""
        m = self._model_unwrapped
        self.injection_old.load_state_dict(m.injection.state_dict())
        if self.depth_lora_old is not None and m.depth_lora is not None:
            self.depth_lora_old.load_state_dict(m.depth_lora.state_dict())

    @contextlib.contextmanager
    def _old_policy_ctx(self):
        """Context manager: temporarily swap model to old-policy modules.

        Swaps injection and depth_lora on the unwrapped model for
        old-policy forward() calls. Guarantees restoration on exit,
        so DDP gradient sync always sees the current (trainable) modules.

        DDP-safe: only touches _model_unwrapped, not the DDP wrapper.
        Pattern derived from TRL's reference model context managers
        (unwrap_model_for_generation in trl/models/utils.py).
        """
        m = self._model_unwrapped
        saved_inj = m.injection
        saved_lora = m.depth_lora
        m.injection = self.injection_old
        if self.depth_lora_old is not None and saved_lora is not None:
            m.depth_lora = self.depth_lora_old
        try:
            yield
        finally:
            m.injection = saved_inj
            m.depth_lora = saved_lora

    # ── training loop ──────────────────────────────────────────────

    def train(self):
        cfg = self.cfg
        if self.local_rank == 0:
            print(f"{'='*60}")
            print(f"SRPO · Recurrent-Depth Gemma E2B · {cfg.total_steps} steps")
            print(f"G={cfg.group_size} · T~Poisson({cfg.poisson_mean}) · μ_bwd≈{cfg.bptt_ratio}T")
            print(f"Device: {self.device} · World: {self.world_size}")
            print(f"{'='*60}")

        self.optimizer.zero_grad()
        data_iter = iter(self.dataloader)
        time_hist = []

        for step_idx in range(cfg.total_steps):
            t0 = time.time()

            # accumulate gradients over micro-batches (no_sync until last)
            for acc in range(cfg.gradient_accumulation_steps):
                is_last = (acc == cfg.gradient_accumulation_steps - 1)
                sync_ctx = self.model.no_sync() if hasattr(self.model, 'no_sync') and not is_last and self.world_size > 1 else contextlib.nullcontext()
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.dataloader)
                    batch = next(data_iter)

                # forward (autocast bf16), then backward scaled by GA steps
                with sync_ctx:
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        metrics = self.train_step(
                            list(batch),
                            loss_scale=1.0 / cfg.gradient_accumulation_steps,
                        )

            # optimizer step
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.trainable_params(), cfg.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.step += 1

            dt = time.time() - t0
            time_hist.append(dt)

            # logging (rank 0 only)
            if self.local_rank == 0 and (step_idx % cfg.log_every == 0 or step_idx < 5):
                avg_t = sum(time_hist[-10:]) / len(time_hist[-10:])
                print(
                    f"step {step_idx:4d}/{cfg.total_steps} | "
                    f"loss={metrics.get('loss',0):.3f} | "
                    f"grpo={metrics.get('grpo_loss',0):.3f} | "
                    f"sdpo={metrics.get('sdpo_loss',0):.3f} | "
                    f"R̄={metrics.get('reward_mean',0):.3f} | "
                    f"T={metrics['T']} | "
                    f"ρ(A)={metrics['rho']:.4f} | "
                    f"{dt:.1f}s"
                )

            sample_due = cfg.sample_log_every > 0 and (
                step_idx % cfg.sample_log_every == 0 or step_idx < 5
            )
            if self.local_rank == 0 and sample_due:
                self._log_samples(step_idx, metrics)

            # eval
            if step_idx % cfg.eval_every == 0 and step_idx > 0:
                self._eval()

            # checkpoint
            if step_idx % cfg.save_every == 0 and step_idx > 0:
                self._save(step_idx)

        if self.local_rank == 0:
            print("Training complete.")

    def _eval(self):
        rho = self._model_unwrapped.injection.compute_spectral_radius()
        ok = rho < 1.0
        if self.local_rank == 0:
            print(f"  [eval] ρ(A)={rho:.6f} {'✓' if ok else '✗ UNSTABLE'}")

    def _save(self, step: int):
        if self.local_rank != 0:
            return
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "step": step,
            "trainable": {n: p.data.clone() for n, p in self._model_unwrapped.named_parameters() if p.requires_grad},
            "optimizer": self.optimizer.state_dict(),
            "config": self.cfg,
        }, f"checkpoints/step_{step}.pt")
        print(f"  [save] checkpoints/step_{step}.pt")

    @classmethod
    def resume(cls, checkpoint_path: str, cfg_override: Optional[TrainConfig] = None):
        """Resume training from a checkpoint saved by _save().

        Restores model weights, optimizer state (including momentum buffers),
        and training step.  The pretrained backbone is reloaded via
        load_pretrained(); only trainable injection/LoRA weights and
        optimizer state come from the checkpoint.

        Args:
            checkpoint_path: Path to a step_N.pt file.
            cfg_override:    Optional TrainConfig override (e.g., to change
                              total_steps for a longer run).
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_cfg = ckpt["config"]
        cfg = cfg_override if cfg_override is not None else saved_cfg

        trainer = cls(cfg)
        trainer.step = ckpt["step"]

        # Restore trainable parameters into the freshly-loaded model
        m = trainer._model_unwrapped
        for name, param in m.named_parameters():
            if name in ckpt["trainable"]:
                param.data.copy_(ckpt["trainable"][name].to(param.device))

        # Restore optimizer state (momentum / variance buffers)
        # Must happen AFTER parameter data is restored so IDs match.
        trainer.optimizer.load_state_dict(ckpt["optimizer"])

        if trainer.local_rank == 0:
            print(f"Resumed from {checkpoint_path} at step {trainer.step}")
        return trainer


# ── entry ──────────────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    ap.add_argument("--steps", type=int, default=None,
                    help="Override total_steps")
    ap.add_argument("--sample-log-every", type=int, default=None,
                    help="Log full prompt/completion samples every N steps")
    ap.add_argument("--sample-log-prompts", type=int, default=None,
                    help="Number of prompt groups to log; each group logs all G completions")
    ap.add_argument("--sample-log-path", type=str, default=None,
                    help="JSONL path for full prompt/completion sample logs")
    ap.add_argument("--no-sample-log", action="store_true",
                    help="Disable full prompt/completion sample logging")
    ap.add_argument("--grpo-forward-batch-size", type=int, default=None,
                    help="Chunk size for GRPO current/reference log-prob forwards")
    ap.add_argument("--sdpo-forward-batch-size", type=int, default=None,
                    help="Chunk size for SDPO student/teacher forwards")
    ap.add_argument("--load-backend", choices=["transformers", "unsloth"], default=None,
                    help="Model loading backend")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="Use 4-bit loading with the Unsloth backend")
    ap.add_argument("--max-seq-length", type=int, default=None,
                    help="Unsloth max sequence length")
    ap.add_argument("--max-train-sequence-tokens", type=int, default=None,
                    help="Backprop window for GRPO/SDPO recomputation; 0 disables")
    ap.add_argument("--no-activation-checkpointing", action="store_true",
                    help="Disable checkpointing through recurrent-depth forwards")
    args = ap.parse_args()

    def apply_overrides(cfg: TrainConfig) -> bool:
        changed = False
        if args.steps is not None:
            cfg.total_steps = args.steps
            changed = True
        if args.sample_log_every is not None:
            cfg.sample_log_every = args.sample_log_every
            changed = True
        if args.sample_log_prompts is not None:
            cfg.sample_log_prompts = args.sample_log_prompts
            changed = True
        if args.sample_log_path is not None:
            cfg.sample_log_path = args.sample_log_path
            changed = True
        if args.no_sample_log:
            cfg.sample_log_every = 0
            changed = True
        if args.grpo_forward_batch_size is not None:
            cfg.grpo_forward_batch_size = args.grpo_forward_batch_size
            changed = True
        if args.sdpo_forward_batch_size is not None:
            cfg.sdpo_forward_batch_size = args.sdpo_forward_batch_size
            changed = True
        if args.load_backend is not None:
            cfg.load_backend = args.load_backend
            changed = True
        if args.load_in_4bit:
            cfg.load_in_4bit = True
            changed = True
        if args.max_seq_length is not None:
            cfg.max_seq_length = args.max_seq_length
            changed = True
        if args.max_train_sequence_tokens is not None:
            cfg.max_train_sequence_tokens = args.max_train_sequence_tokens
            changed = True
        if args.no_activation_checkpointing:
            cfg.use_activation_checkpointing = False
            changed = True
        return changed

    if args.resume:
        cfg = None
        if any([
            args.steps is not None,
            args.sample_log_every is not None,
            args.sample_log_prompts is not None,
            args.sample_log_path is not None,
            args.no_sample_log,
            args.grpo_forward_batch_size is not None,
            args.sdpo_forward_batch_size is not None,
            args.load_backend is not None,
            args.load_in_4bit,
            args.max_seq_length is not None,
            args.max_train_sequence_tokens is not None,
            args.no_activation_checkpointing,
        ]):
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            cfg = ckpt["config"]
            apply_overrides(cfg)
            del ckpt
        trainer = SRPOTrainer.resume(args.resume, cfg_override=cfg)
    else:
        cfg = TrainConfig()
        apply_overrides(cfg)
        trainer = SRPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
