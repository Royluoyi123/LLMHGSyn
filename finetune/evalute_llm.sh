#!/usr/bin/env bash
set -euo pipefail

: "${MODEL:?Set MODEL to the base teacher model}"
ADAPTER="${ADAPTER:-outputs/teacher_lora/warm_start/fold_1}"
INSTRUCTIONS="${INSTRUCTIONS:-outputs/instructions/warm_start/fold_1/all.jsonl}"
OUTPUT="${OUTPUT:-outputs/teacher/warm_start/fold_1.npz}"

python finetune/extract_teacher.py \
  --model "${MODEL}" \
  --adapter "${ADAPTER}" \
  --instructions "${INSTRUCTIONS}" \
  --output "${OUTPUT}" \
  --load-in-4bit
