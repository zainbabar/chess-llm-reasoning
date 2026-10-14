#!/bin/bash
# RL for student B with reward v1 (2026-09-27): identical to B's run in 07_rl_answer_only_reward.sh except the reward
# (--reward truth: +1 right move, +0.5 x correct share of the line, -0.5 for a false claim, -0.5 for an explanation
# under 25 words). Same start, puzzles and order. Report: reports/rl_rewards_B.md.
# Resumable: re-running skips finished steps.
# Launch: (nohup runs/08_rl_B_reward_v1.sh >> results/08_rl_B_reward_v1.log 2>&1 &)
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ts() { date '+%F %T'; }
has_model() { ls $1/*.safetensors > /dev/null 2>&1; }
memgb() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
STEPS=${STEPS:-390}; LR=${LR:-2e-6}; SKIP=${SKIP:-40800}; SAVE=${SAVE:-130}
echo "$(ts) RL_TRUTH START (B, reward truth, steps $STEPS, lr $LR, puzzles from row $((SKIP + 1)), checkpoints every $SAVE)"
if docker ps --format '{{.Names}}' | grep -qx gptoss; then docker stop gptoss > /dev/null && echo "$(ts) teacher stopped"; fi
docker rm -f student > /dev/null 2>&1

( minm=999; while ps -p $$ > /dev/null; do m=$(memgb); [ $m -lt $minm ] && { minm=$m; echo $minm > results/08_rl_B_reward_v1.minmem; }
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

out=ckpt/rlT_B
if has_model $out; then
  echo "$(ts) $out already trained -> skip"
else
  echo "$(ts) RL B (truth reward) from ckpt/path_B: $STEPS steps"
  docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trainenv python -u src/rl_grpo.py --model ckpt/path_B \
      --puzzles pool_collect1.jsonl --skip $SKIP --out $out --max-steps $STEPS --lr $LR --save-every $SAVE --reward truth \
      > results/rlT_B.log 2>&1 || echo "$(ts) RL B truth FAILED (see results/rlT_B.log)"
  grep -aE "^saved" results/rlT_B.log | tail -n 4
fi

for ck in step130 step260 final; do
  d=rlT_B/$ck; tag=rlT_B_$ck
  [ $ck = final ] && d=rlT_B
  if has_model ckpt/$d && [ ! -s results/student_${tag}_nothink.jsonl ]; then
    echo "$(ts) EVAL $tag"
    serve $tag $d && $PY -u src/eval_student.py --model ckpt/$d --served-name $tag --tag $tag --puzzles splits/dev.jsonl \
        --max-tokens 1024 --temperature 0 --concurrency 32 | tail -n 2 || echo "$(ts) EVAL $tag FAILED"
  fi
done
docker rm -f student > /dev/null 2>&1
$PY experiments/rl_rewards_B_report.py | tail -n 30
echo "$(ts) RL_TRUTH DONE (teacher left stopped; min MemAvailable $(cat results/08_rl_B_reward_v1.minmem 2>/dev/null) GiB)"
