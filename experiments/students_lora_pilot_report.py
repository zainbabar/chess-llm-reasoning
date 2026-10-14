"""LoRA pilot (runs/02_lora_pilot.sh): same base (Qwen3-1.7B + LoRA), same ~1.9k training puzzles, different training
text (answer only / code-built explanation / teacher explanation), evaluated on the development set.

Prints a markdown report (also written to results/students_lora_pilot.md): accuracy, illegal moves, format,
full-line accuracy, puzzle rating with bootstrap CI, per-band accuracy, and paired comparisons (exact
McNemar test on the puzzles where two arms disagree).
"""
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from outputs import load as load_outputs  # noqa: E402
from paired_tests import mcnemar  # noqa: E402
from puzzle_rating import rate  # noqa: E402

ARMS = [("p_answer", "answer only"), ("p_code", "code-built explanation"), ("p_llm", "teacher explanation")]
BANDS = [f"{lo}-{lo + 200}" for lo in range(800, 2200, 200)]


def load(arm):
    R, _ = load_outputs(f"results/student_pilot_{arm}_nothink.jsonl")
    return R or {}


def main():
    data = {arm: load(arm) for arm, _ in ARMS}
    L = ["# LoRA pilot: three training texts on 1.9k puzzles", "",
         "Qwen3-1.7B + LoRA r64, 3 epochs, the same 1,892 training puzzles (rated 800–2200; set sft_1.9k in splits/train_puzzles.jsonl.gz),"
         " differing only in the training text. Test: the development set (splits/dev.jsonl, 500 puzzles), greedy decoding,"
         " the standard prompt. Zero-shot baseline: 1/140.", "",
         "| Arm | n | Correct | Illegal | No answer | Exact format | Legal line | Full line right | Mean tokens | Puzzle rating (95% CI) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, label in ARMS:
        R = list(data[arm].values())
        if not R:
            L.append(f"| {label} | (not run) |" + " |" * 8)
            continue
        c = Counter(r["status"] for r in R)
        rt = rate([{"rating": r["rating"], "status": r["status"]} for r in R])
        L.append(f"| {label} | {len(R)} | **{c['correct']} ({100 * c['correct'] / len(R):.1f}%)** | {c['illegal']} | "
                 f"{c['parse_fail'] + c['truncated']} | {sum(bool(r.get('final_line_strict')) for r in R)} | "
                 f"{sum(bool(r.get('line_legal')) for r in R)} | {sum(bool(r.get('full_line_correct')) for r in R)} | "
                 f"{st.mean((r.get('usage') or {}).get('completion_tokens', 0) for r in R):.0f} | "
                 f"{rt['rating']} ({rt['ci_low']}–{rt['ci_high']}) |")
    L += ["", "Per band (correct of ~71):", "", "| Arm | " + " | ".join(BANDS) + " |", "|---|" + "---|" * len(BANDS)]
    for arm, label in ARMS:
        R = list(data[arm].values())
        if R:
            L.append(f"| {label} | " + " | ".join(str(sum(r["status"] == "correct" for r in R if r["rating_band"] == b))
                                                for b in BANDS) + " |")
    L += ["", "Paired comparisons (same 500 puzzles; exact McNemar):", ""]
    for i in range(len(ARMS)):
        for j in range(i + 1, len(ARMS)):
            (a1, l1), (a2, l2) = ARMS[i], ARMS[j]
            ids = set(data[a1]) & set(data[a2])
            if not ids:
                continue
            only1 = sum(data[a1][k]["status"] == "correct" and data[a2][k]["status"] != "correct" for k in ids)
            only2 = sum(data[a2][k]["status"] == "correct" and data[a1][k]["status"] != "correct" for k in ids)
            L.append(f"- {l1} vs {l2}: only the first right on {only1}, only the second on {only2} "
                     f"(p = {mcnemar(only1, only2):.3f})")
    L += ["", "Reference: the teacher gpt-oss-120b, low effort, standard prompt: 45% on the 140-puzzle subset "
          "(rating ~1425); FEN + legal moves on these 500: 31.2%."]
    Path("results/students_lora_pilot.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
