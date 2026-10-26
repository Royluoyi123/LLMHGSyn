#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to a Hugging Face model name or local path}"
INSTRUCTION_DIR="${INSTRUCTION_DIR:-outputs/instructions/warm_start/fold_1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/teacher_lora/warm_start/fold_1}"

python finetune/llm_main.py \
  --model "${MODEL}" \
  --instruction-dir "${INSTRUCTION_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --load-in-4bit \
  --bf16
