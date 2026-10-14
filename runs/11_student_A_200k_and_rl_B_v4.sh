#!/bin/bash
# Student A + 200k, and RL reward v4 (2026-09-28). (1) Student A + 200k: student A trained one more pass on 199,997
# new answer-only puzzles (pool_collect2; no overlap with the test sets, A's puzzles or the RL puzzles), same
# settings. (2) RL for student B with reward v4 (--reward truth4, src/rewards.py: the written line is replayed move by
# move). Reports: reports/teacher_baseline.md, reports/rl_rewards_B.md.
# Resumable: re-running skips finished steps.
# Launch: (nohup runs/11_student_A_200k_and_rl_B_v4.sh >> results/11_student_A_200k_and_rl_B_v4.log 2>&1 &)
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ts() { date '+%F %T'; }
has_model() { ls $1/*.safetensors > /dev/null 2>&1; }
memgb() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
testset() { [ -f splits/dev.jsonl ] && echo splits/dev.jsonl || echo pilot_set.jsonl; }  # in case the file gets renamed
echo "$(ts) 11_student_A_200k_and_rl_B_v4 START"
if docker ps --format '{{.Names}}' | grep -qx gptoss; then docker stop gptoss > /dev/null && echo "$(ts) teacher stopped"; fi
docker rm -f student > /dev/null 2>&1
( minm=999; while ps -p $$ > /dev/null; do m=$(memgb); [ $m -lt $minm ] && { minm=$m; echo $minm > results/11_student_A_200k_and_rl_B_v4.minmem; }
    if [ $m -lt 8 ]; then echo "$(ts) WATCHDOG: MemAvailable $m GiB -> restarting trainenv"; docker restart trainenv > /dev/null; sleep 60; fi
    sleep 5; done ) &
serve() {  # serve <name> <dir relative to ckpt/>
  docker rm -f student > /dev/null 2>&1
  docker run -d --name student --gpus all --ipc=host -p 8001:8001 \
    --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
    --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset \
    -e HF_HUB_OFFLINE=1 -v ~/.cache/huggingface:/root/.cache/huggingface -v "$PWD/ckpt":/ckpt \
    nvcr.io/nvidia/vllm:26.05-py3 vllm serve /ckpt/$2 --served-model-name $1 --dtype bfloat16 \
    --port 8001 --gpu-memory-utilization 0.30 --max-model-len 8192 --max-num-seqs 64 > /dev/null
  for i in $(seq 1 120); do curl -sf localhost:8001/v1/models > /dev/null && return 0
    docker ps --filter name=student -q | grep -q . || break; sleep 5; done
  echo "$(ts) student server failed to start ($*)"; docker logs --tail 20 student 2>&1; return 1
}
evaluate() {  # evaluate <tag> <dir relative to ckpt/>
  if has_model ckpt/$2 && [ ! -s results/student_$1_nothink.jsonl ]; then
    echo "$(ts) EVAL $1"
    serve $1 $2 && $PY -u src/eval_student.py --model ckpt/$2 --served-name $1 --tag $1 --puzzles $(testset) \
        --max-tokens 1024 --temperature 0 --concurrency 32 | tail -n 2 || echo "$(ts) EVAL $1 FAILED"
  fi
}

# 1. A on 200k more puzzles
if has_model ckpt/path_A200k; then echo "$(ts) ckpt/path_A200k already trained -> skip"; else
  echo "$(ts) TRAIN A200k (from ckpt/path_A, 200k new puzzles, 1 pass)"
  docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trainenv python -u src/train_sft.py \
      --data results/sft/ans_p1l_200k.jsonl --model ckpt/path_A --out ckpt/path_A200k --full --epochs 1 --lr 1e-5 \
      --batch-tokens 8192 --accum 4 --no-grad-ckpt > results/path_A200k.log 2>&1 || echo "$(ts) TRAIN A200k FAILED (see results/path_A200k.log)"
  grep -aE "^saved|final loss|min" results/path_A200k.log | tail -n 3
fi
evaluate path_A200k path_A200k
docker rm -f student > /dev/null 2>&1

# 2. RL for B with reward v4
out=ckpt/rlT4_B
if has_model $out; then echo "$(ts) $out already trained -> skip"; else
  echo "$(ts) RL B (truth4 reward) from ckpt/path_B: 390 steps"
  docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trainenv python -u src/rl_grpo.py --model ckpt/path_B \
      --puzzles pool_collect1.jsonl --skip 40800 --out $out --max-steps 390 --lr 2e-6 --save-every 130 --reward truth4 \
      > results/rlT4_B.log 2>&1 || echo "$(ts) RL B truth4 FAILED (see results/rlT4_B.log)"
  grep -aE "^saved" results/rlT4_B.log | tail -n 4
fi
evaluate rlT4_B_step130 rlT4_B/step130
evaluate rlT4_B_step260 rlT4_B/step260
evaluate rlT4_B_final rlT4_B
docker rm -f student > /dev/null 2>&1

# 3. reports
$PY experiments/teacher_baseline_report.py | tail -n 12
$PY experiments/rl_rewards_B_report.py | tail -n 5
echo "$(ts) 11_student_A_200k_and_rl_B_v4 DONE (teacher left stopped; min MemAvailable $(cat results/11_student_A_200k_and_rl_B_v4.minmem 2>/dev/null) GiB)"
