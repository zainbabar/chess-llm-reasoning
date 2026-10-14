"""Where the graded model outputs live, and how to load them.

Each evaluation writes results/<run name>.jsonl locally (results/ isn't in the repo). The public copies are in
reports/outputs/<set>/<model>.jsonl.gz, one folder per puzzle set (dev, fresh_1, fresh_2) and one readable file name
per model. PUBLIC maps each local file to its public copy; load() reads the local file when it exists and the public
copy otherwise, so the reports and figures can be rebuilt from a fresh clone.
"""
import gzip
import json
from pathlib import Path

OUTPUTS_DIR = Path("reports/outputs")

# local run tag -> public model name. Students are evaluated as results/student_<tag>_nothink.jsonl (development set),
# results/fresh_student_<tag>.jsonl (fresh set 1) and results/fresh2_student_<tag>.jsonl (fresh set 2).
_STUDENTS = {
    "path_A": "student_a", "path_A_ep1": "student_a_pass1",
    "path_A_seed1": "student_a_seed1", "path_A_seed1_ep1": "student_a_seed1_pass1",
    "path_A200k": "student_a_200k", "path_A200k_seed1": "student_a_200k_seed1",
    "path_B": "student_b", "path_B_ep1": "student_b_pass1",
    "path_B_seed1": "student_b_seed1", "path_B_seed1_ep1": "student_b_seed1_pass1",
    "path_Baf": "student_b_answer_first", "path_Baf_ep1": "student_b_answer_first_pass1",
    "path_Baf_seed1": "student_b_answer_first_seed1", "path_Baf_seed1_ep1": "student_b_answer_first_seed1_pass1",
    # five sampled runs per student (temperature 0.7, top-p 0.8, top-k 20)
    **{f"path_{m}_sampled{'' if k == 1 else k}": f"{name}_sample{k}"
       for m, name in [("A", "student_a"), ("A200k", "student_a_200k"), ("B", "student_b"),
                       ("Baf", "student_b_answer_first")] for k in range(1, 6)},
    # RL checkpoints after 130, 260 and 390 steps (final)
    **{f"{run}_{ck}": f"{name}_{ck.replace('final', 'step390')}"
       for run, name in [("rlL_A", "student_a_rl_answer_reward"), ("rlL_B", "student_b_rl_answer_reward"),
                         ("rlT_B", "student_b_rl_v1"), ("rlT2_B", "student_b_rl_v2"), ("rlT3_B", "student_b_rl_v3"),
                         ("rlT4_B", "student_b_rl_v4"), ("rlT4_B_seed1", "student_b_rl_v4_seed1")]
       for ck in ("step130", "step260", "final")},
    # LoRA students from the early data-strategy tests
    "pilot_p_answer": "lora_1.9k_answers", "pilot_p_code": "lora_1.9k_code_explanations",
    "pilot_p_llm": "lora_1.9k_teacher_explanations",
    "pilot_night2_answer": "lora_5.6k_answers", "pilot_night2_llm": "lora_5.6k_teacher_explanations",
    "pilot_night2_code": "lora_5.6k_code_explanations", "pilot_night2_aux": "lora_5.6k_answers_board_tracking",
    "pilot_night2_scale": "lora_21.6k_answers",
}

PUBLIC = {
    # development set (splits/dev.jsonl)
    "results/pilot_formatA.jsonl": "dev/teacher_low_fen_only",
    "results/pilot_formatB.jsonl": "dev/teacher_low_fen_legal_moves",
    "results/test500_formatP1L.jsonl": "dev/teacher_low",
    "results/test500_med_formatP1L.jsonl": "dev/teacher_medium",
    "results/student_base_nothink_dev.jsonl": "dev/untrained_qwen3_1.7b",
    **{f"results/student_{tag}_nothink.jsonl": f"dev/{name}" for tag, name in _STUDENTS.items()},
    # fresh test set 1 (splits/fresh_1.jsonl)
    "results/fresh_low_formatP1L.jsonl": "fresh_1/teacher_low",
    "results/fresh_low_s2_formatP1L.jsonl": "fresh_1/teacher_low_resample2",
    "results/fresh_low_s3_formatP1L.jsonl": "fresh_1/teacher_low_resample3",
    "results/fresh_med_formatP1L.jsonl": "fresh_1/teacher_medium",
    "results/fresh_med_s2_formatP1L.jsonl": "fresh_1/teacher_medium_resample2",
    "results/fresh_stockfish.jsonl": "fresh_1/stockfish_0.1s",
    "results/fresh_student_base.jsonl": "fresh_1/untrained_qwen3_1.7b",
    **{f"results/fresh_student_{tag}.jsonl": f"fresh_1/{name}" for tag, name in _STUDENTS.items()},
    # fresh test set 2 (splits/fresh_2.jsonl)
    "results/fresh2_low_formatP1L.jsonl": "fresh_2/teacher_low",
    "results/fresh2_med_formatP1L.jsonl": "fresh_2/teacher_medium",
    "results/fresh2_stockfish.jsonl": "fresh_2/stockfish_0.1s",
    "results/fresh2_student_base.jsonl": "fresh_2/untrained_qwen3_1.7b",
    **{f"results/fresh2_student_{tag}.jsonl": f"fresh_2/{name}" for tag, name in _STUDENTS.items()},
}
SETS = {"dev": "splits/dev.jsonl", "fresh_1": "splits/fresh_1.jsonl", "fresh_2": "splits/fresh_2.jsonl"}


def public_path(path):
    """The committed copy of a local results/ file, or None if it isn't published."""
    name = PUBLIC.get(str(path))
    return OUTPUTS_DIR / f"{name}.jsonl.gz" if name else None


def load(path):
    """Graded outputs keyed by puzzle id, plus the number of records read (a retried puzzle keeps its last record;
    unresolved errors are dropped). Reads the local results/ file, or its committed copy in a fresh clone."""
    p, gz = Path(path), public_path(path)
    if not p.exists() and not (gz and gz.exists()):
        return None, 0
    R, n_lines = {}, 0
    for line in (open(p) if p.exists() else gzip.open(gz, "rt")):
        r = json.loads(line)
        n_lines += 1
        R[r["puzzle_id"]] = r
    return {k: v for k, v in R.items() if v.get("status") != "error"}, n_lines
