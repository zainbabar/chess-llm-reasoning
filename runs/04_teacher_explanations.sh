#!/bin/bash
# Teacher explanations for training (2026-09-25): gpt-oss-120b at low effort explains the verified Lichess solution
# "as if discovering" it, with python-chess facts in the prompt, for pool_collect1 puzzles 6,001-40,000 (the first
# 6,000 came from 03). Per 5k chunk: write, then filter with the claim checker (src/package_sft.py). 37,543 were kept.
# Resumable: re-running skips finished steps.
# Launch: (nohup runs/04_teacher_explanations.sh >> results/04_teacher_explanations.log 2>&1 &)
cd "$(dirname "$0")/.."
PY=.venv/bin/python
ts() { date '+%F %T'; }
START=${START:-6000}; END=${END:-40000}; CHUNK=${CHUNK:-5000}; CONC=${CONC:-48}
D=results/collect40k
mkdir -p $D
echo "$(ts) 04_teacher_explanations START (puzzles $((START+1))-$END, chunks of $CHUNK)"
curl -sf localhost:8000/v1/models >/dev/null || { echo "$(ts) teacher not running"; exit 1; }
for ((s = START; s < END; s += CHUNK)); do
  e=$(( s + CHUNK < END ? s + CHUNK : END ))
  part=$(printf "%05d" $s)
  f=$D/pool_$part.jsonl
  [ -s $f ] || sed -n "$((s + 1)),${e}p" pool_collect1.jsonl > $f
  a=$D/engine_analysis_$part.jsonl
  [ -s $a ] || (nohup $PY src/engine_analysis.py --puzzles $f --out $a --workers 8 > /dev/null 2>&1 &)
  echo "$(ts) chunk $part: puzzles $((s + 1))-$e"
  $PY -u src/write_traces.py --run c40k_$part --variant FDF --puzzles $f --effort low --concurrency $CONC --max-tokens 4096 \
      || echo "$(ts) chunk $part writer FAILED (re-run to resume)"
  $PY -u src/write_traces.py --run c40k_$part --variant FDF --puzzles $f --effort low --concurrency $CONC --max-tokens 4096 \
      > /dev/null 2>&1 || true   # retry any errored requests
  $PY src/package_sft.py --traces results/traces_c40k_${part}_FDF.jsonl --puzzles $f --out $D/sft_fdf_$part.jsonl \
      || echo "$(ts) chunk $part packaging FAILED"
done
cat $D/sft_fdf_?????.jsonl > $D/sft_fdf_all.jsonl
echo "$(ts) 04_teacher_explanations DONE: $(wc -l < $D/sft_fdf_all.jsonl) accepted (plus the 5.6k run's $(wc -l < results/night2/sft_fdf.jsonl))"
