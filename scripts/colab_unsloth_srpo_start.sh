#!/usr/bin/env bash
# Colab A100 startup for the Unsloth SRPO-lite path.
#
# Run from the repository root:
#   STEPS=300 bash scripts/colab_unsloth_srpo_start.sh

set -euo pipefail

usage() {
    cat <<'EOF'
Unsloth SRPO-lite Colab launcher.

Common overrides:
  MODEL_NAME=unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit
  STEPS=300
  MAX_PROMPTS=500
  MAX_SEQ_LENGTH=2048
  MAX_COMPLETION_LENGTH=768
  THINKING_LOOPS=3
  NUM_GENERATIONS=4
  PER_DEVICE_TRAIN_BATCH_SIZE=4
  GRAD_ACCUM_STEPS=4
  TEMPERATURE=1.2
  USE_VLLM=0
  FAST_INFERENCE=0
  RUN_TESTS=1
  SKIP_INSTALL=1

Example:
  STEPS=300 MAX_COMPLETION_LENGTH=1024 bash scripts/colab_unsloth_srpo_start.sh
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
export HF_HOME="${HF_HOME:-/content/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/content/hf-datasets-cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "=== parcae-srpo Unsloth SRPO-lite startup ==="
echo "Repo: ${REPO_ROOT}"
echo "Python: $(${PYTHON} --version)"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found. Colab GPU runtime may not be enabled."
fi

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    echo "=== Installing Unsloth and repo helpers ==="
    "${PYTHON}" -m pip install -q --upgrade pip wheel "setuptools<82"
    "${PYTHON}" -m pip install -q -e ".[test]"
    "${PYTHON}" -m pip install -q "unsloth" "trl" "diffusers"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "=== Logging in to Hugging Face from HF_TOKEN ==="
    "${PYTHON}" - <<'PY'
import os
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("Hugging Face token accepted.")
PY
fi

if [[ "${RUN_TESTS:-1}" == "1" ]]; then
    echo "=== Running focused tests ==="
    "${PYTHON}" -m pytest -p no:cacheprovider \
        tests/test_grpo_loss.py \
        tests/test_sample_logging.py \
        tests/test_unsloth_srpo_helpers.py \
        -q
fi

echo "=== Starting Unsloth SRPO-lite training ==="
"${PYTHON}" scripts/train_unsloth_srpo.py \
    --model-name "${MODEL_NAME:-unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit}" \
    --dataset "${DATASET:-humaneval_mbpp_mix}" \
    --max-prompts "${MAX_PROMPTS:-500}" \
    --max-seq-length "${MAX_SEQ_LENGTH:-2048}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH:-768}" \
    --max-completion-length "${MAX_COMPLETION_LENGTH:-768}" \
    --thinking-loops "${THINKING_LOOPS:-3}" \
    --lora-rank "${LORA_RANK:-32}" \
    --lora-alpha "${LORA_ALPHA:-32}" \
    --learning-rate "${LEARNING_RATE:-5e-6}" \
    --max-steps "${STEPS:-300}" \
    --save-steps "${SAVE_STEPS:-100}" \
    --num-generations "${NUM_GENERATIONS:-4}" \
    --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-4}" \
    --gradient-accumulation-steps "${GRAD_ACCUM_STEPS:-4}" \
    --temperature "${TEMPERATURE:-1.2}" \
    --top-p "${TOP_P:-0.95}" \
    --beta "${BETA:-0.0}" \
    --loss-type "${LOSS_TYPE:-dapo}" \
    --fast-inference "${FAST_INFERENCE:-0}" \
    --use-vllm "${USE_VLLM:-0}" \
    --output-dir "${OUTPUT_DIR:-outputs/unsloth_srpo}" \
    --lora-output-dir "${LORA_OUTPUT_DIR:-outputs/unsloth_srpo_lora}"
