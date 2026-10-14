#!/bin/bash
# Teacher retries on the 140-puzzle subset (2026-09-24): which puzzles does a second low-effort attempt, and then a
# medium-effort attempt, solve? Steps 2-4; step 1 is results/cascade_t1a_formatP1L.jsonl. Summary:
# experiments/retry_cascade_report.py. Used for data collection only (it needs the answer key).
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
R=results
echo "$(date '+%F %T') STEP 2: tier 1 attempt #2 (low + P1L, 140)"
$PY -u src/run_teacher.py --run cascade_t1b --formats P1L --puzzles pilot_subset140.jsonl --effort low --max-tokens 8192 --concurrency 32
$PY src/select_unsolved.py --puzzles pilot_subset140.jsonl --results $R/cascade_t1a_formatP1L.jsonl $R/cascade_t1b_formatP1L.jsonl --out $R/cascade_tier2_puzzles.jsonl
echo "$(date '+%F %T') STEP 3: tier 2 (medium + P1L on unsolved)"
$PY -u src/run_teacher.py --run cascade_t2 --formats P1L --puzzles $R/cascade_tier2_puzzles.jsonl --effort medium --max-tokens 32768 --concurrency 32
$PY src/select_unsolved.py --puzzles pilot_subset140.jsonl --results $R/cascade_t1a_formatP1L.jsonl $R/cascade_t1b_formatP1L.jsonl $R/cascade_t2_formatP1L.jsonl --out $R/cascade_tier3_puzzles.jsonl
echo "$(date '+%F %T') STEP 4: tier 3 (rationalization R1, low, on still-unsolved)"
$PY -u src/run_teacher.py --run cascade_t3 --formats R1 --puzzles $R/cascade_tier3_puzzles.jsonl --effort low --max-tokens 8192 --concurrency 32
echo "$(date '+%F %T') CASCADE DONE"
