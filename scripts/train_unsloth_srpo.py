"""
Unsloth-backed SRPO-lite training for code reasoning.

This path does not use the recurrent-depth Gemma wrapper.  Instead, it uses
Unsloth + TRL GRPO on a 4-bit LoRA model and teaches explicit thinking loops
in the generated text:

<think_loop_1>...</think_loop_1>
<think_loop_2>...</think_loop_2>
<answer>
```python
...
```
</answer>

The verifier and dataset loader are reused from train_srpo.py.
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

from datasets import Dataset

try:
    from train_srpo import CodeProblemDataset, TrainConfig, extract_code, verify
except ModuleNotFoundError:
    from scripts.train_srpo import CodeProblemDataset, TrainConfig, extract_code, verify


DEFAULT_MODEL = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"


@dataclass
class UnslothSRPOConfig:
    model_name: str = DEFAULT_MODEL
    dataset: str = "humaneval_mbpp_mix"
    max_prompts: int = 500
    seed: int = 42

    max_seq_length: int = 2048
    max_prompt_length: int = 768
    max_completion_length: int = 768
    thinking_loops: int = 3

    lora_rank: int = 32
    lora_alpha: int = 32
    learning_rate: float = 5e-6
    max_steps: int = 300
    save_steps: int = 100
    logging_steps: int = 1

    num_generations: int = 4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    temperature: float = 1.2
    top_p: float = 0.95

    beta: float = 0.0
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    loss_type: str = "dapo"
    mask_truncated_completions: bool = False

    fast_inference: bool = False
    use_vllm: bool = False
    output_dir: str = "outputs/unsloth_srpo"
    lora_output_dir: str = "outputs/unsloth_srpo_lora"


def build_thinking_system_prompt(loops: int) -> str:
    loop_lines = "\n".join(
        f"<think_loop_{i}>brief private reasoning pass {i}</think_loop_{i}>"
        for i in range(1, loops + 1)
    )
    return (
        "You are a code reasoning model. Solve the task by doing a fixed "
        "number of internal thinking loops before writing final code.\n\n"
        "Use exactly this response shape:\n"
        f"{loop_lines}\n"
        "<answer>\n"
        "```python\n"
        "# final complete solution only\n"
        "```\n"
        "</answer>\n\n"
        "The answer block must contain complete executable Python code. "
        "Do not include tests in the answer block."
    )


def build_prompt(problem_prompt: str, loops: int) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_thinking_system_prompt(loops)},
        {"role": "user", "content": problem_prompt},
    ]


def build_dataset(cfg: UnslothSRPOConfig) -> Dataset:
    legacy_cfg = TrainConfig()
    legacy_cfg.dataset = cfg.dataset
    legacy_cfg.max_prompts = cfg.max_prompts
    legacy_cfg.seed = cfg.seed

    problems = CodeProblemDataset(legacy_cfg)
    rows = []
    for item in problems:
        rows.append({
            "prompt": build_prompt(item["prompt"], cfg.thinking_loops),
            "raw_prompt": item["prompt"],
            "test": item["test"],
            "entry": item.get("entry", "solution"),
        })
    return Dataset.from_list(rows)


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", completion))
    if isinstance(completion, list):
        if not completion:
            return ""
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(completion)


def extract_answer_block(text: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def extract_code_from_completion(completion: Any) -> str:
    return extract_code(extract_answer_block(completion_to_text(completion)))


def _as_list(value: Any, n: int, default: Any = None) -> list[Any]:
    if value is None:
        return [default] * n
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value] * n


def _thinking_loop_count() -> int:
    return max(1, int(os.environ.get("UNSLOTH_SRPO_THINKING_LOOPS", "3")))


def thinking_format_reward(completions: list[Any], **_: Any) -> list[float]:
    loops = _thinking_loop_count()
    rewards = []
    for completion in completions:
        text = completion_to_text(completion)
        score = 0.0

        answer = re.search(r"<answer>.*?</answer>", text, flags=re.DOTALL | re.IGNORECASE)
        if answer:
            score += 0.25
            if "```python" in answer.group(0).lower():
                score += 0.15

        previous_start = -1
        for i in range(1, loops + 1):
            pattern = rf"<think_loop_{i}>(.*?)</think_loop_{i}>"
            match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
            if not match:
                continue
            content = match.group(1).strip()
            score += 0.15
            if len(content) >= 8:
                score += 0.05
            if match.start() > previous_start:
                score += 0.025
                previous_start = match.start()

        rewards.append(min(score, 1.0))
    return rewards


def code_shape_reward(completions: list[Any], **_: Any) -> list[float]:
    rewards = []
    for completion in completions:
        code = extract_code_from_completion(completion)
        score = 0.0
        if code.strip():
            score += 0.1
        if re.search(r"^\s*(def|class)\s+\w+", code, flags=re.MULTILINE):
            score += 0.25
        if "<answer>" in completion_to_text(completion).lower():
            score += 0.15
        rewards.append(score)
    return rewards


def code_execution_reward(
    completions: list[Any],
    test: Any = None,
    tests: Any = None,
    **_: Any,
) -> list[float]:
    test_cases = _as_list(test if test is not None else tests, len(completions), "")
    rewards = []
    for completion, test_case in zip(completions, test_cases):
        code = extract_code_from_completion(completion)
        reward, _feedback = verify(code, str(test_case))
        rewards.append(3.0 * float(reward))
    return rewards


def filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_obj).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in parameters}


def load_unsloth_model(cfg: UnslothSRPOConfig):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=True,
        fast_inference=cfg.fast_inference,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=cfg.lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    return model, tokenizer


def build_grpo_config(cfg: UnslothSRPOConfig, bf16: bool):
    from trl import GRPOConfig

    kwargs = {
        "output_dir": cfg.output_dir,
        "learning_rate": cfg.learning_rate,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "weight_decay": 0.1,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_8bit",
        "logging_steps": cfg.logging_steps,
        "bf16": bf16,
        "fp16": not bf16,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "num_generations": cfg.num_generations,
        "max_prompt_length": cfg.max_prompt_length,
        "max_completion_length": cfg.max_completion_length,
        "max_steps": cfg.max_steps,
        "save_steps": cfg.save_steps,
        "max_grad_norm": 0.1,
        "report_to": "none",
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "beta": cfg.beta,
        "use_vllm": cfg.use_vllm,
        "epsilon": cfg.epsilon,
        "epsilon_high": cfg.epsilon_high,
        "loss_type": cfg.loss_type,
        "mask_truncated_completions": cfg.mask_truncated_completions,
    }
    return GRPOConfig(**filter_supported_kwargs(GRPOConfig, kwargs))


def build_trainer(model: Any, tokenizer: Any, args: Any, dataset: Dataset):
    from trl import GRPOTrainer

    kwargs = {
        "model": model,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "reward_funcs": [
            code_execution_reward,
            thinking_format_reward,
            code_shape_reward,
        ],
        "args": args,
        "train_dataset": dataset,
    }
    return GRPOTrainer(**filter_supported_kwargs(GRPOTrainer, kwargs))


def train(cfg: UnslothSRPOConfig):
    os.environ["UNSLOTH_SRPO_THINKING_LOOPS"] = str(cfg.thinking_loops)

    from unsloth import is_bfloat16_supported

    dataset = build_dataset(cfg)
    model, tokenizer = load_unsloth_model(cfg)
    args = build_grpo_config(cfg, bf16=is_bfloat16_supported())
    trainer = build_trainer(model, tokenizer, args, dataset)
    trainer.train()

    os.makedirs(os.path.dirname(cfg.lora_output_dir) or ".", exist_ok=True)
    if hasattr(model, "save_lora"):
        model.save_lora(cfg.lora_output_dir)
    else:
        model.save_pretrained(cfg.lora_output_dir)
        tokenizer.save_pretrained(cfg.lora_output_dir)
    print(f"Saved Unsloth LoRA adapter to {cfg.lora_output_dir}")


def str_to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_default(name: str, default: Any) -> Any:
    return os.environ.get(name, default)


def parse_args(argv: Iterable[str] | None = None) -> UnslothSRPOConfig:
    parser = argparse.ArgumentParser(description="Train SRPO-lite code reasoning with Unsloth GRPO.")
    parser.add_argument("--model-name", default=env_default("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--dataset", default=env_default("DATASET", "humaneval_mbpp_mix"))
    parser.add_argument("--max-prompts", type=int, default=int(env_default("MAX_PROMPTS", 500)))
    parser.add_argument("--max-seq-length", type=int, default=int(env_default("MAX_SEQ_LENGTH", 2048)))
    parser.add_argument("--max-prompt-length", type=int, default=int(env_default("MAX_PROMPT_LENGTH", 768)))
    parser.add_argument("--max-completion-length", type=int, default=int(env_default("MAX_COMPLETION_LENGTH", 768)))
    parser.add_argument("--thinking-loops", type=int, default=int(env_default("THINKING_LOOPS", 3)))
    parser.add_argument("--lora-rank", type=int, default=int(env_default("LORA_RANK", 32)))
    parser.add_argument("--lora-alpha", type=int, default=int(env_default("LORA_ALPHA", 32)))
    parser.add_argument("--learning-rate", type=float, default=float(env_default("LEARNING_RATE", 5e-6)))
    parser.add_argument("--max-steps", type=int, default=int(env_default("STEPS", 300)))
    parser.add_argument("--save-steps", type=int, default=int(env_default("SAVE_STEPS", 100)))
    parser.add_argument("--num-generations", type=int, default=int(env_default("NUM_GENERATIONS", 4)))
    parser.add_argument("--per-device-train-batch-size", type=int, default=int(env_default("PER_DEVICE_TRAIN_BATCH_SIZE", 4)))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=int(env_default("GRAD_ACCUM_STEPS", 4)))
    parser.add_argument("--temperature", type=float, default=float(env_default("TEMPERATURE", 1.2)))
    parser.add_argument("--top-p", type=float, default=float(env_default("TOP_P", 0.95)))
    parser.add_argument("--beta", type=float, default=float(env_default("BETA", 0.0)))
    parser.add_argument("--loss-type", default=env_default("LOSS_TYPE", "dapo"))
    parser.add_argument("--fast-inference", type=str_to_bool, default=str_to_bool(env_default("FAST_INFERENCE", "0")))
    parser.add_argument("--use-vllm", type=str_to_bool, default=str_to_bool(env_default("USE_VLLM", "0")))
    parser.add_argument("--output-dir", default=env_default("OUTPUT_DIR", "outputs/unsloth_srpo"))
    parser.add_argument("--lora-output-dir", default=env_default("LORA_OUTPUT_DIR", "outputs/unsloth_srpo_lora"))
    args = parser.parse_args(argv)
    return UnslothSRPOConfig(**vars(args))


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
