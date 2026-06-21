from scripts.train_unsloth_srpo import (
    build_thinking_system_prompt,
    code_execution_reward,
    code_shape_reward,
    completion_to_text,
    extract_code_from_completion,
    thinking_format_reward,
)


def test_thinking_prompt_contains_requested_loop_tags():
    prompt = build_thinking_system_prompt(3)

    assert "<think_loop_1>" in prompt
    assert "</think_loop_3>" in prompt
    assert "<answer>" in prompt
    assert "```python" in prompt


def test_completion_to_text_handles_chat_completion_shape():
    completion = [{"role": "assistant", "content": "hello"}]

    assert completion_to_text(completion) == "hello"


def test_extract_code_from_answer_block_prefers_final_answer():
    completion = """
<think_loop_1>maybe return the input</think_loop_1>
<answer>
```python
def identity(x):
    return x
```
</answer>
"""

    assert extract_code_from_completion(completion).strip() == "def identity(x):\n    return x"


def test_thinking_format_reward_scores_loop_answer_shape(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SRPO_THINKING_LOOPS", "2")
    good = """
<think_loop_1>inspect the signature</think_loop_1>
<think_loop_2>write direct code</think_loop_2>
<answer>
```python
def add(a, b):
    return a + b
```
</answer>
"""
    bad = "def add(a, b):\n    return a + b"

    good_score, bad_score = thinking_format_reward([good, bad])

    assert good_score > bad_score
    assert good_score <= 1.0


def test_code_shape_reward_prefers_answer_wrapped_function():
    good = "<answer>```python\ndef add(a, b):\n    return a + b\n```</answer>"
    bad = "return a + b"

    good_score, bad_score = code_shape_reward([good, bad])

    assert good_score > bad_score


def test_code_execution_reward_uses_repo_verifier():
    completion = "<answer>```python\ndef add(a, b):\n    return a + b\n```</answer>"
    rewards = code_execution_reward([completion], test=["assert add(2, 3) == 5"])

    assert rewards == [3.0]
