"""Student A (answers only, full fine-tune) -> results/student_A.md.

Compares A (final, after 2 passes) and A after pass 1 with the teacher (gpt-oss-120b, low effort, one attempt) and
the answers-only LoRA students (5.6k and 21.6k puzzles), all on the same 500 development-set puzzles with the same prompt.
Paired comparisons: exact McNemar on the puzzles where the two disagree.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from outputs import load
from paired_tests import mcnemar
from puzzle_rating import rate

ARMS = [
    ("A: answers only, full FT, 37.5k × 2 passes", "results/student_path_A_nothink.jsonl", "ckpt/path_A"),
    ("A after pass 1", "results/student_path_A_ep1_nothink.jsonl", None),
    ("teacher gpt-oss-120b (low, 1 attempt)", "results/test500_formatP1L.jsonl", None),
    ("LoRA 21.6k: answers only, 1 pass", "results/student_pilot_night2_scale_nothink.jsonl", "ckpt/night2_scale"),
    ("LoRA 5.6k: answers only, 1 pass", "results/student_pilot_night2_answer_nothink.jsonl", "ckpt/night2_answer"),
]
PAIRS = [(0, 2), (0, 3), (1, 2), (1, 0), (1, 3), (0, 4)]


def main():
    test = [json.loads(l) for l in open("splits/dev.jsonl")]
    test_ids = {p["puzzle_id"] for p in test}
    band = {p["puzzle_id"]: p["rating_band"] for p in test}
    D, L = {}, ["# Student A (answers only, full fine-tune)", "",
                "500 development-set puzzles, same prompt for every model; greedy decoding for students.", "",
                "| Model | Correct | Rating (95% CI) | Illegal | Parse fail / truncated | Legal line | Full line right |",
                "|---|---|---|---|---|---|---|"]
    notes = []
    for i, (label, path, _) in enumerate(ARMS):
        R, n_lines = load(path)
        if not R:
            notes.append(f"- missing: {label} ({path})")
            continue
        D[i] = R
        rs = list(R.values())
        st = Counter(r["status"] for r in rs)
        rt = rate(rs)
        c = st["correct"]
        L.append(f"| {label} | **{c}/{len(rs)} ({100 * c / len(rs):.1f}%)** | {rt['rating']} ({rt['ci_low']}–{rt['ci_high']}) "
                 f"| {st['illegal']} | {st['parse_fail']} / {st['truncated']} | {sum(bool(r.get('line_legal')) for r in rs)} "
                 f"| {sum(bool(r.get('full_line_correct')) for r in rs)} |")
        extra = set(R) - test_ids
        if len(R) != 500 or extra or n_lines != len(R):
            notes.append(f"- {label}: {len(R)} puzzles graded ({n_lines} records, {len(extra)} not in the test set)")
    L += ["", "Paired comparisons (exact McNemar; only-first-right vs only-second-right):", ""]
    for a, b in PAIRS:
        if a in D and b in D:
            ids = set(D[a]) & set(D[b])
            x = sum(D[a][k]["status"] == "correct" and D[b][k]["status"] != "correct" for k in ids)
            y = sum(D[b][k]["status"] == "correct" and D[a][k]["status"] != "correct" for k in ids)
            L.append(f"- {ARMS[a][0]} vs {ARMS[b][0]}: {x} vs {y}, p = {mcnemar(x, y):.2g} ({len(ids)} puzzles)")
    bands = sorted({b for b in band.values()}, key=lambda b: int(b.split("-")[0]))
    L += ["", "Correct by rating band:", "", "| Model | " + " | ".join(bands) + " |", "|---|" + "---|" * len(bands)]
    for i in D:
        cnt = Counter(band[k] for k, r in D[i].items() if r["status"] == "correct" and k in band)
        tot = Counter(band[k] for k in D[i] if k in band)
        L.append(f"| {ARMS[i][0]} | " + " | ".join(f"{cnt[b]}/{tot[b]}" for b in bands) + " |")
    L += ["", "Training:", ""]
    for label, _, ck in ARMS:
        m = Path(ck, "train_meta.json") if ck else None
        if m and m.exists():
            d = json.load(open(m))
            tps = d["tokens"] * d["epochs"] / (d["minutes"] * 60)
            L.append(f"- {label}: {'full FT' if d.get('full') else 'LoRA r' + str(d.get('lora_r'))}, "
                     f"{d['examples']:,} examples × {d['epochs']:g} passes, lr {d['lr']:g}, {d['minutes']} min "
                     f"(~{tps:,.0f} tok/s), final loss {d['final_loss']:.3f}")
    if notes:
        L += ["", "Notes / sanity checks:", ""] + notes
    Path("results/student_A.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
