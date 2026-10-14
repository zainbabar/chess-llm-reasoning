# Reports and graded outputs

Every number in the main README comes from one of the reports in this folder, or, for the fresh-set figures, from
`experiments/make_figures.py`; all of them are computed from the graded model answers in [`outputs/`](outputs/). Each report `<name>.md` is written by
`experiments/<name>_report.py` and copied here by `experiments/export_reports.py`; the LoRA reports come from
`students_lora_pilot_report.py` and `students_lora_scaling_report.py`. Sections titled "Notes" were written after
reading the outputs, and say so. A few side numbers in the README come from early exploratory runs and training logs
that aren't included; the README describes each with its setup where it cites it.

## Which report backs which claim

In the order of the README's findings:

| Claim in the README | Report |
|---|---|
| The teacher plays illegal moves from a raw position; the board description fixes that | `teacher_baseline.md` ("How the prompt changes the teacher") |
| The teacher at low vs medium effort; students A, B and A + 200k against both (development set) | `teacher_baseline.md` |
| Fresh test 1: every model on 500 untouched puzzles, the primary comparisons | `fresh_test.md` (protocol, fixed before the run: `fresh_test_protocol.md`) |
| Fresh test 2, the pooled 1,000-puzzle comparisons, and the second-seed copies of A + 200k, B answer-first and RL v4 | `replication_and_fresh_test_2.md` (protocol: `fresh_test_2_protocol.md`) |
| Five sampled runs per student, B's second seed (all four seed pairings), the Stockfish check of 2,052 wrong answers, the Stockfish check of written replies | `robustness_checks.md` |
| The untrained model at the students' settings, the teacher resampled, sound full lines | `baselines_and_line_soundness.md` |
| Answers-only accuracy rises with more data (1.9k → 5.6k → 21.6k → 37.5k); the early steps are within noise and the last also switched from LoRA to a full fine-tune | `students_lora_pilot.md`, `students_lora_scaling.md`, `student_A.md` |
| Training on teacher explanations made students worse at every size | `students_lora_pilot.md`, `students_lora_scaling.md`, `students_A_vs_B.md`, `fresh_test.md` |
| B answer-first; sampled vs greedy decoding; the RL models on fresh set 1 | `answer_first_and_sampling.md` (the pooled answer-first comparisons: `replication_and_fresh_test_2.md`) |
| RL with a right-move reward gave no significant gain by the end of the run (B gained at step 130, p = 0.019 uncorrected, then fell back) | `rl_answer_only_reward.md` |
| Five reward designs on student B, how each was reward-hacked, and each final checkpoint against A | `rl_rewards_B.md` |
| Test puzzles were kept out of training (ids, source games, positions) | `../splits/train_puzzles.jsonl.gz`, checked by `experiments/check_overlap.py` (also a test) |

The protocols are kept exactly as committed before each run, so they name the test sets by their original paths
(`test_set.jsonl`, `fresh_test_set.jsonl`, `fresh_test_set_2.jsonl`). The same files are now `splits/dev.jsonl`,
`splits/fresh_1.jsonl` and `splits/fresh_2.jsonl`, and their sha256 hashes still match the protocols.

## Terms used in the reports

- **Development set / fresh test sets**: `splits/dev.jsonl` (the original 500, used while developing the experiments)
  / `splits/fresh_1.jsonl` and `splits/fresh_2.jsonl` (500 untouched puzzles each, evaluated once).
- **Standard prompt** (`P1L` in the output files): the position, a python-chess description of the board (diagram,
  pieces, legal moves) and a request for the forcing line (`FINAL_LINE`) and the move (`FINAL_MOVE`).
- **Teacher explanations**: gpt-oss-120b at low effort writes "as if discovering" the Lichess solution, with
  python-chess facts about the line in its prompt; texts the claim checker catches in a false claim are dropped.
- **Legal line / full line right**: the written `FINAL_LINE` is legal from the puzzle position / matches the whole
  Lichess solution. **Sound full line** (Stockfish): legal, every solver move as good as Stockfish's best, every
  opponent move a real defense, and as long as the solution or ending in mate.
- **Claim-clean / move flag**: the claim checker found no false claim (piece on a square, "wins the X", mate) / the text
  names a move that is illegal or wrongly marked as check or mate.
- **RL rewards**: answer-only (outcome-only) = the right move; truth v1, strict v2, v3 and v4 also check the
  explanation and the line (definitions at the top of `rl_rewards_B.md`). Checkpoints after 130, 260 and 390 steps.
- **Pass 1**: the checkpoint after the first of two training passes.

## Outputs

`outputs/<set>/<model>.jsonl.gz` holds one model's answers on one test set (`dev`, `fresh_1`, `fresh_2`): exactly one
JSON record per puzzle, with the puzzle, the model's answer (`content`; the teacher's hidden reasoning is in
`reasoning`), the parsed move and line, the grade (`status`: correct / wrong / illegal / parse_fail), the line checks
and token usage. The prompt isn't stored; `build_prompt(puzzle, record["format"])` in `src/run_teacher.py` rebuilds it
exactly. For example:

```python
import gzip, json
rows = [json.loads(l) for l in gzip.open("reports/outputs/fresh_1/student_a_200k.jsonl.gz", "rt")]
print(sum(r["status"] == "correct" for r in rows), "of", len(rows))   # 314 of 500
```

File names: `teacher_low` and `teacher_medium` are gpt-oss-120b at each effort (`_resample2`, `_resample3`: the same
run repeated; on the development set also `teacher_low_fen_only` and `teacher_low_fen_legal_moves`, the first prompt
tests). `student_a`, `student_a_200k`, `student_b` and `student_b_answer_first` are the main students (`_seed1`: the
second-seed copy; `_pass1`: after one training pass; `_sample1` to `_sample5`: sampled instead of greedy).
`student_b_rl_<reward>_step<N>` and `student_a_rl_answer_reward_step<N>` are the RL checkpoints,
`lora_<size>_<training text>` the early LoRA students, then `untrained_qwen3_1.7b` and `stockfish_0.1s`.
`src/outputs.py` maps each file to the local run that produced it.
