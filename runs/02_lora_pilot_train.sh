#!/bin/bash
# Train LoRA students (r64 on Qwen3-1.7B), one per training file. Used by 02_lora_pilot.sh and 03_lora_5.6k_21.6k.sh.
# Usage: runs/02_lora_pilot_train.sh arm1 arm2 ...   (arm = results/sft/<arm>.jsonl -> ckpt/<arm>)
cd "$(dirname "$0")/.."
ts() { date '+%F %T'; }
for arm in "$@"; do
  if [ -f ckpt/$arm/adapter_model.safetensors ]; then echo "$(ts) $arm already trained"; continue; fi
  echo "$(ts) TRAIN $arm"
  docker exec trainenv python -u src/train_sft.py --data results/sft/$arm.jsonl --out ckpt/$arm \
      --model Qwen/Qwen3-1.7B --epochs 1 --batch-tokens 8192 --accum 4 --no-grad-ckpt || echo "$(ts) TRAIN $arm FAILED"
done
echo "$(ts) PILOT TRAIN DONE: $*"
