#!/bin/bash
# Answer-first ablation and sampled decoding (2026-09-29). (1) Student B answer-first: B's exact texts reordered so
# the line and move come first, same recipe as A and B. (2) One sampled run (temperature 0.7, top-p 0.8, top-k 20) of
# A, B, A + 200k and B answer-first on the development set and fresh set 1. (3) The earlier RL models on fresh set 1
# (added after the protocol). Report: reports/answer_first_and_sampling.md.
# Resumable: re-running skips finished steps.
# Launch: (nohup runs/13_answer_first_and_sampling.sh >> results/13_answer_first_and_sampling.log 2>&1 &)
cd "$(dirname "$0")/.."
PY=.venv/bin/python
DEV=splits/dev.jsonl
FRESH=splits/fresh_1.jsonl
ts() { date '+%F %T'; }
has_model() { ls $1/*.safetensors > /dev/null 2>&1; }
memgb() { awk '/MemAvailable/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
done500() { [ -s $1 ] && [ $($PY -c "import json,sys; print(len({json.loads(l)['puzzle_id'] for l in open(sys.argv[1]) if json.loads(l).get('status') != 'error'}))" $1) -eq 500 ]; }

echo "$(ts) 13_answer_first_and_sampling START"
if docker ps --format '{{.Names}}' | grep -qx gptoss; then docker stop gptoss > /dev/null && echo "$(ts) teacher stopped"; fi
docker rm -f student > /dev/null 2>&1
( minm=999; while ps -p $$ > /dev/null; do m=$(memgb); [ $m -lt $minm ] && { minm=$m; echo $minm > results/13_answer_first_and_sampling.minmem; }
    if [ $m -lt 8 ]; then echo "$(ts) WATCHDOG: MemAvailable $m GiB -> restarting trainenv"; docker restart trainenv > /dev/null; sleep 60; fi
    sleep 5; done ) &

serve() {  # serve <served name> <model path inside the container>
  docker rm -f student > /dev/null 2>&1
  docker run -d --name student --gpus all --ipc=host -p 8001:8001 \
    --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
    --device /dev/nvidia-uvm-tools --device /dev/nvidia-modeset \
    -e HF_HUB_OFFLINE=1 -v ~/.cache/huggingface:/root/.cache/huggingface -v "$PWD/ckpt":/ckpt \
    nvcr.io/nvidia/vllm:26.05-py3 vllm serve $2 --served-model-name $1 --dtype bfloat16 \
    --port 8001 --gpu-memory-utilization 0.30 --max-model-len 8192 --max-num-seqs 64 > /dev/null
  for i in $(seq 1 120); do curl -sf localhost:8001/v1/models > /dev/null && return 0
    docker ps --filter name=student -q | grep -q . || break; sleep 5; done
  echo "$(ts) student server failed to start ($*)"; docker logs --tail 20 student 2>&1; return 1
}
run_eval() {  # run_eval <tag> <puzzles> <out> [extra eval_student args]   (the server must be up)
  local tag=$1 puzzles=$2 out=$3; shift 3
  if done500 $out; then echo "$(ts) $out already done -> skip"; return; fi
  $PY -u src/eval_student.py --model ckpt/$tag --served-name $tag --tag $tag --puzzles $puzzles --out $out \
      --max-tokens 1024 --concurrency 32 "$@" | tail -n 2 || echo "$(ts) EVAL $out FAILED"
}
all_done() { for f in "$@"; do done500 $f || return 1; done; return 0; }

# 1. train B with the answer first
if has_model ckpt/path_Baf; then echo "$(ts) ckpt/path_Baf already trained -> skip"; else
  echo "$(ts) TRAIN path_Baf (B's explanations, answer first; same recipe as A and B)"
  docker exec -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True trainenv python -u src/train_sft.py \
      --data results/sft/path_B_answerfirst.jsonl --out ckpt/path_Baf --model Qwen/Qwen3-1.7B \
      --full --epochs 2 --batch-tokens 8192 --accum 4 --no-grad-ckpt --save-epochs --attn flash_attention_2 \
      > results/path_Baf.log 2>&1 || echo "$(ts) TRAIN path_Baf FAILED (see results/path_Baf.log)"
  grep -aE "^saved|optimizer steps" results/path_Baf.log | tail -n 3
fi

# 1b. greedy evals of B-answer-first (final, pass 1) on both sets
for pair in "path_Baf path_Baf" "path_Baf_ep1 path_Baf/epoch1"; do
  set -- $pair
  outs="results/student_$1_nothink.jsonl results/fresh_student_$1.jsonl"
  if has_model ckpt/$2 && ! all_done $outs; then
    echo "$(ts) EVAL $1 (greedy)"
    if serve $1 /ckpt/$2; then
      run_eval $1 $DEV results/student_$1_nothink.jsonl --temperature 0
      run_eval $1 $FRESH results/fresh_student_$1.jsonl --temperature 0
    fi
  fi
done

# 2. sampled decoding (eval_student's default when --temperature is not given: 0.7 / top-p 0.8 / top-k 20)
for pair in "path_A path_A" "path_B path_B" "path_A200k path_A200k" "path_Baf path_Baf"; do
  set -- $pair
  outs="results/student_$1_sampled_nothink.jsonl results/fresh_student_$1_sampled.jsonl"
  if has_model ckpt/$2 && ! all_done $outs; then
    echo "$(ts) EVAL $1 (sampled)"
    if serve $1 /ckpt/$2; then
      run_eval $1 $DEV results/student_$1_sampled_nothink.jsonl
      run_eval $1 $FRESH results/fresh_student_$1_sampled.jsonl
    fi
  fi
done

# 3. fresh-set greedy evals of the earlier RL finals
for m in rlL_A rlL_B rlT_B rlT2_B rlT3_B; do
  out=results/fresh_student_${m}_final.jsonl
  if has_model ckpt/$m && ! done500 $out; then
    echo "$(ts) EVAL $m final on the fresh set"
    serve $m /ckpt/$m && run_eval $m $FRESH $out --temperature 0
  fi
done
docker rm -f student > /dev/null 2>&1
echo "$(ts) 13_answer_first_and_sampling DONE (teacher left stopped; min MemAvailable $(cat results/13_answer_first_and_sampling.minmem 2>/dev/null) GiB)"
