#!/usr/bin/env bash
# Colab A100 bootstrap for parcae-srpo.
#
# Run from the repository root on a Colab A100 runtime:
#   HF_TOKEN=hf_... STEPS=100 bash scripts/colab_a100_start.sh
#
# Useful overrides:
#   STEPS=25 SAMPLE_LOG_EVERY=1 bash scripts/colab_a100_start.sh
#   MAX_RESPONSE_TOKENS=768 GROUP_SIZE=4 bash scripts/colab_a100_start.sh

set -euo pipefail

usage() {
    cat <<'EOF'
Colab A100 startup script for parcae-srpo.

Required before training gated/private models:
  export HF_TOKEN=hf_...

Common overrides:
  STEPS=100                  training steps (default: 100)
  MODEL_NAME=...             Hugging Face model id (default: google/gemma-4-E2B-it)
  MODEL_PATH=/path/to/model   local model path, if already downloaded
  LOAD_BACKEND=unsloth       real SRPO loader: transformers or unsloth
  LOAD_IN_4BIT=1             use Unsloth 4-bit loading for the Gemma backbone
  MAX_SEQ_LENGTH=2048        Unsloth setup sequence length
  MAX_TRAIN_SEQUENCE_TOKENS=384
                              backprop window; full samples are still logged
  DATASET=humaneval_mbpp_mix dataset name
  MAX_PROMPTS=500            number of dataset prompts
  GROUP_SIZE=4               completions per prompt
  MAX_PROMPT_TOKENS=512      prompt token budget
  MAX_RESPONSE_TOKENS=512    completion token budget
  MICRO_BATCH_SIZE=1         prompts per step on the A100
  GRAD_ACCUM_STEPS=8         gradient accumulation steps
  GRPO_FORWARD_BATCH_SIZE=1  GRPO log-prob forward chunk size
  SDPO_FORWARD_BATCH_SIZE=1  SDPO KL forward chunk size
  MAX_LOOPS=6                max recurrent depth for A100 startup runs
  SAMPLE_LOG_EVERY=1         print full samples every N steps
  SAMPLE_LOG_PROMPTS=1       prompt groups to print per sample log
  SAMPLE_LOG_PATH=...        JSONL path for full sample logs
  USE_ACTIVATION_CHECKPOINTING=1
  RUN_TESTS=1                run focused tests before training
  PREDOWNLOAD=0              predownload model before training; defaults off for Unsloth
  SKIP_INSTALL=1             skip pip install

Example:
  HF_TOKEN=hf_... STEPS=50 SAMPLE_LOG_EVERY=1 bash scripts/colab_a100_start.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
MODEL_NAME="${MODEL_NAME:-google/gemma-4-E2B-it}"
LOAD_BACKEND="${LOAD_BACKEND:-unsloth}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
MAX_TRAIN_SEQUENCE_TOKENS="${MAX_TRAIN_SEQUENCE_TOKENS:-384}"
HF_HOME="${HF_HOME:-/content/hf-cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/content/hf-datasets-cache}"
SAMPLE_LOG_PATH="${SAMPLE_LOG_PATH:-runs/colab_a100_samples.jsonl}"

if [[ -z "${PREDOWNLOAD+x}" ]]; then
    if [[ "${LOAD_BACKEND}" == "unsloth" ]]; then
        PREDOWNLOAD=0
    else
        PREDOWNLOAD=1
    fi
fi

export HF_HOME
export HF_DATASETS_CACHE
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== parcae-srpo Colab A100 startup ==="
echo "Repo: ${REPO_ROOT}"
echo "Python: $(${PYTHON} --version)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. In Colab, select Runtime > Change runtime type > A100 GPU." >&2
    exit 1
fi

nvidia-smi
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "${GPU_NAME}" != *"A100"* ]]; then
    echo "WARNING: first GPU is '${GPU_NAME}', not A100. Continuing anyway."
fi

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    echo "=== Installing repo and training dependencies ==="
    "${PYTHON}" -m pip install -q --upgrade pip wheel "setuptools<82"
    if [[ "${LOAD_BACKEND}" == "unsloth" ]]; then
        "${PYTHON}" -m pip install -q -e ".[train,test,unsloth]" "huggingface_hub>=0.24"
    else
        "${PYTHON}" -m pip install -q -e ".[train,test]" "huggingface_hub>=0.24"
    fi
fi

# Unsloth requires Transformers <= 4.57.2. We pin to the highest allowed
# version because that is the best chance of native Gemma 4 support while
# keeping Unsloth importable. Override with TRANSFORMERS_VERSION if needed.
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.57.2}"

if [[ "${LOAD_BACKEND}" == "unsloth" ]]; then
    echo "=== Checking Transformers Gemma 4 support ==="
    TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION}" "${PYTHON}" - <<'PY'
import importlib.util
import os
import subprocess
import sys

# Highest Transformers version Unsloth currently allows.
MAX_TRANSFORMERS = os.environ.get("TRANSFORMERS_VERSION", "4.57.2")


def has_gemma4() -> bool:
    try:
        return importlib.util.find_spec("transformers.models.gemma4.modeling_gemma4") is not None
    except Exception:
        return False


def current_version():
    try:
        import transformers
        return transformers.__version__
    except Exception:
        return None


if has_gemma4():
    import transformers
    print(f"Transformers Gemma 4 support detected: {transformers.__version__}")
else:
    cur = current_version()
    print(
        f"Transformers local Gemma 4 module is missing (installed: {cur}). "
        f"Pinning transformers to {MAX_TRANSFORMERS} (max allowed by Unsloth) "
        "to try to obtain native Gemma 4 support..."
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         f"transformers=={MAX_TRANSFORMERS}"]
    )
    # Re-import after install to refresh the version.
    import importlib
    import transformers
    importlib.reload(transformers)
    if has_gemma4():
        print(f"Transformers Gemma 4 support detected after pin: {transformers.__version__}")
    else:
        print(
            f"ERROR: transformers {transformers.__version__} still has no native "
            "gemma4 module, and Unsloth requires transformers <= "
            f"{MAX_TRANSFORMERS}. Cannot load google/gemma-4-E2B-it with this "
            "combination.\n"
            "To proceed you can either:\n"
            "  - set LOAD_BACKEND=transformers and LOAD_IN_4BIT=1, then raise "
            "TRANSFORMERS_VERSION above 4.57.2, or\n"
            "  - use a model repo that ships trusted remote modeling code.",
            file=sys.stderr,
        )
        sys.exit(1)
PY
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "=== Logging in to Hugging Face from HF_TOKEN ==="
    "${PYTHON}" - <<'PY'
import os
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("Hugging Face token accepted.")
PY
else
    echo "WARNING: HF_TOKEN is not set. This is okay only if the model is public or already cached."
fi

if [[ "${PREDOWNLOAD}" == "1" && -z "${MODEL_PATH:-}" ]]; then
    echo "=== Predownloading model: ${MODEL_NAME} ==="
    MODEL_NAME="${MODEL_NAME}" "${PYTHON}" - <<'PY'
import os
from huggingface_hub import snapshot_download

model_name = os.environ["MODEL_NAME"]
path = snapshot_download(
    model_name,
    ignore_patterns=["*.md", ".gitattributes"],
)
print(f"Cached model at: {path}")
PY
fi

if [[ "${RUN_TESTS:-1}" == "1" ]]; then
    echo "=== Running focused tests ==="
    "${PYTHON}" -m pytest -p no:cacheprovider \
        tests/test_old_policy.py \
        tests/test_grpo_loss.py \
        tests/test_sample_logging.py \
        -q
fi

mkdir -p "$(dirname "${SAMPLE_LOG_PATH}")" checkpoints

echo "=== Starting A100 training ==="
echo "Full sample logs: ${SAMPLE_LOG_PATH}"
echo "Tip: in another Colab cell, run: !tail -n 80 ${SAMPLE_LOG_PATH}"

MODEL_NAME="${MODEL_NAME}" \
MODEL_PATH="${MODEL_PATH:-}" \
LOAD_BACKEND="${LOAD_BACKEND}" \
LOAD_IN_4BIT="${LOAD_IN_4BIT}" \
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH}" \
MAX_TRAIN_SEQUENCE_TOKENS="${MAX_TRAIN_SEQUENCE_TOKENS}" \
USE_ACTIVATION_CHECKPOINTING="${USE_ACTIVATION_CHECKPOINTING:-1}" \
DATASET="${DATASET:-humaneval_mbpp_mix}" \
MAX_PROMPTS="${MAX_PROMPTS:-500}" \
STEPS="${STEPS:-100}" \
GROUP_SIZE="${GROUP_SIZE:-4}" \
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-512}" \
MAX_RESPONSE_TOKENS="${MAX_RESPONSE_TOKENS:-512}" \
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}" \
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}" \
GRPO_FORWARD_BATCH_SIZE="${GRPO_FORWARD_BATCH_SIZE:-1}" \
SDPO_FORWARD_BATCH_SIZE="${SDPO_FORWARD_BATCH_SIZE:-1}" \
POISSON_MEAN="${POISSON_MEAN:-2}" \
MAX_LOOPS="${MAX_LOOPS:-6}" \
LEARNING_RATE="${LEARNING_RATE:-5e-4}" \
SAVE_EVERY="${SAVE_EVERY:-50}" \
EVAL_EVERY="${EVAL_EVERY:-25}" \
LOG_EVERY="${LOG_EVERY:-1}" \
SAMPLE_LOG_EVERY="${SAMPLE_LOG_EVERY:-1}" \
SAMPLE_LOG_PROMPTS="${SAMPLE_LOG_PROMPTS:-1}" \
SAMPLE_LOG_PATH="${SAMPLE_LOG_PATH}" \
"${PYTHON}" - <<'PY'
import os

from scripts.train_srpo import SRPOTrainer, TrainConfig


def int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


cfg = TrainConfig()
cfg.model_name = os.environ["MODEL_NAME"]
if os.environ.get("MODEL_PATH"):
    cfg.model_path = os.environ["MODEL_PATH"]
cfg.load_backend = os.environ["LOAD_BACKEND"]
cfg.load_in_4bit = bool_env("LOAD_IN_4BIT", cfg.load_in_4bit)
cfg.max_seq_length = int_env("MAX_SEQ_LENGTH", cfg.max_seq_length)
cfg.max_train_sequence_tokens = int_env("MAX_TRAIN_SEQUENCE_TOKENS", cfg.max_train_sequence_tokens)
cfg.use_activation_checkpointing = bool_env("USE_ACTIVATION_CHECKPOINTING", cfg.use_activation_checkpointing)
cfg.dataset = os.environ["DATASET"]
cfg.max_prompts = int_env("MAX_PROMPTS", cfg.max_prompts)
cfg.total_steps = int_env("STEPS", cfg.total_steps)
cfg.group_size = int_env("GROUP_SIZE", cfg.group_size)
cfg.max_prompt_tokens = int_env("MAX_PROMPT_TOKENS", cfg.max_prompt_tokens)
cfg.max_response_tokens = int_env("MAX_RESPONSE_TOKENS", cfg.max_response_tokens)
cfg.micro_batch_size = int_env("MICRO_BATCH_SIZE", cfg.micro_batch_size)
cfg.gradient_accumulation_steps = int_env("GRAD_ACCUM_STEPS", cfg.gradient_accumulation_steps)
cfg.grpo_forward_batch_size = int_env("GRPO_FORWARD_BATCH_SIZE", cfg.grpo_forward_batch_size)
cfg.sdpo_forward_batch_size = int_env("SDPO_FORWARD_BATCH_SIZE", cfg.sdpo_forward_batch_size)
cfg.poisson_mean = int_env("POISSON_MEAN", cfg.poisson_mean)
cfg.max_loops = int_env("MAX_LOOPS", cfg.max_loops)
cfg.learning_rate = float_env("LEARNING_RATE", cfg.learning_rate)
cfg.save_every = int_env("SAVE_EVERY", cfg.save_every)
cfg.eval_every = int_env("EVAL_EVERY", cfg.eval_every)
cfg.log_every = int_env("LOG_EVERY", cfg.log_every)
cfg.sample_log_every = int_env("SAMPLE_LOG_EVERY", cfg.sample_log_every)
cfg.sample_log_prompts = int_env("SAMPLE_LOG_PROMPTS", cfg.sample_log_prompts)
cfg.sample_log_path = os.environ["SAMPLE_LOG_PATH"]

print("Training config:")
for name in [
    "model_name",
    "model_path",
    "load_backend",
    "load_in_4bit",
    "max_seq_length",
    "use_activation_checkpointing",
    "dataset",
    "max_prompts",
    "total_steps",
    "group_size",
    "max_prompt_tokens",
    "max_response_tokens",
    "micro_batch_size",
    "gradient_accumulation_steps",
    "grpo_forward_batch_size",
    "sdpo_forward_batch_size",
    "max_train_sequence_tokens",
    "poisson_mean",
    "max_loops",
    "learning_rate",
    "sample_log_every",
    "sample_log_prompts",
    "sample_log_path",
]:
    print(f"  {name}={getattr(cfg, name)}")

trainer = SRPOTrainer(cfg)
trainer.train()
PY
