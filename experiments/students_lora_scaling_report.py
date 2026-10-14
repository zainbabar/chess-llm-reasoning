"""LoRA students on 5.6k and 21.6k puzzles (runs/03_lora_5.6k_21.6k.sh; env RUN, default night2, the run's output
tag) -> results/<RUN>/results.md.

All students: Qwen3-1.7B + LoRA r64, evaluated on the development set (greedy, standard prompt). Includes the
LoRA pilot arms (1.9k puzzles) and the teacher's single attempt on the same 500 (results/test500_formatP1L.jsonl).
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from paired_tests import mcnemar
from puzzle_rating import rate

RUN = os.environ.get("RUN", "night2")


def load(path):
    p = Path(path)
    if not p.exists():
        return None
    R = {}
    for l in open(p):
        r = json.loads(l)
        R[r["puzzle_id"]] = r  # last record wins (retries)
    return {k: v for k, v in R.items() if v.get("status") != "error"} or None


def train_n(arm):
    m = Path(f"ckpt/{arm}/train_meta.json")
    if m.exists():
        d = json.load(open(m))
        return f"{d['examples']:,} ex × {d['epochs']} ep ({d['minutes']} min)"
    return ""


ARMS = [
    ("teacher gpt-oss-120b, low effort, 1 attempt", "results/test500_formatP1L.jsonl", ""),
    ("LoRA 1.9k: answer only", "results/student_pilot_p_answer_nothink.jsonl", "p_answer"),
    ("LoRA 1.9k: code-built explanation", "results/student_pilot_p_code_nothink.jsonl", "p_code"),
    ("LoRA 1.9k: teacher explanation", "results/student_pilot_p_llm_nothink.jsonl", "p_llm"),
    (f"LoRA 5.6k: answer only", f"results/student_pilot_{RUN}_answer_nothink.jsonl", f"{RUN}_answer"),
    (f"LoRA 5.6k: teacher explanation", f"results/student_pilot_{RUN}_llm_nothink.jsonl", f"{RUN}_llm"),
    (f"LoRA 5.6k: answer + board tracking", f"results/student_pilot_{RUN}_aux_nothink.jsonl", f"{RUN}_aux"),
    (f"LoRA 5.6k: code-built explanation", f"results/student_pilot_{RUN}_code_nothink.jsonl", f"{RUN}_code"),
    (f"LoRA 21.6k: answer only", f"results/student_pilot_{RUN}_scale_nothink.jsonl", f"{RUN}_scale"),
]


def main():
    D = {}
    L = ["# LoRA students on 5.6k and 21.6k puzzles", "", "500 development-set puzzles, the same prompt for everyone.", "",
         "| Model | Training | Correct | Rating (95% CI) | Illegal | Legal line | Full line right |",
         "|---|---|---|---|---|---|---|"]
    for label, path, arm in ARMS:
        R = load(path)
        if not R:
            continue
        D[label] = R
        rs = list(R.values())
        rt = rate(rs)
        c = sum(r["status"] == "correct" for r in rs)
        L.append(f"| {label} | {train_n(arm) if arm else '—'} | **{c}/{len(rs)} ({100 * c / len(rs):.1f}%)** | "
                 f"{rt['rating']} ({rt['ci_low']}–{rt['ci_high']}) | {sum(r['status'] == 'illegal' for r in rs)} | "
                 f"{sum(bool(r.get('line_legal')) for r in rs)} | {sum(bool(r.get('full_line_correct')) for r in rs)} |")
    L += ["", "Paired comparisons (exact McNemar on the puzzles where the two disagree):", ""]
    pairs = [("LoRA 5.6k: answer only", "LoRA 5.6k: teacher explanation"),
             ("LoRA 5.6k: answer only", "LoRA 5.6k: answer + board tracking"),
             ("LoRA 5.6k: answer only", "LoRA 21.6k: answer only"),
             ("LoRA 5.6k: answer only", "LoRA 5.6k: code-built explanation"),
             ("LoRA 1.9k: code-built explanation", "LoRA 5.6k: code-built explanation"),
             ("LoRA 1.9k: answer only", "LoRA 5.6k: answer only"),
             ("teacher gpt-oss-120b, low effort, 1 attempt", "LoRA 5.6k: answer only"),
             ("teacher gpt-oss-120b, low effort, 1 attempt", "LoRA 21.6k: answer only"),
             ("LoRA 1.9k: teacher explanation", "LoRA 5.6k: teacher explanation")]
    for a, b in pairs:
        if a in D and b in D:
            ids = set(D[a]) & set(D[b])
            x = sum(D[a][k]["status"] == "correct" and D[b][k]["status"] != "correct" for k in ids)
            y = sum(D[b][k]["status"] == "correct" and D[a][k]["status"] != "correct" for k in ids)
            L.append(f"- {a} vs {b}: only first right {x}, only second right {y}, p = {mcnemar(x, y):.3f}")
    aux = []
    for arm in (f"{RUN}_answer", f"{RUN}_aux"):
        p = Path(f"results/{RUN}/auxeval_{arm}.jsonl")
        if p.exists():
            R = [json.loads(l) for l in open(p)]
            for k in ("board", "square"):
                S = [r for r in R if r["kind"] == k]
                if S:
                    label = {f"{RUN}_answer": "LoRA 5.6k answer only", f"{RUN}_aux": "LoRA 5.6k answer + board tracking"}[arm]
                    aux.append(f"- {label}, {k} tasks: {sum(r['ok'] for r in S)}/{len(S)} exactly right")
    if aux:
        L += ["", "Held-out board-tracking test (200 tasks from unseen puzzles):", ""] + aux
    Path(f"results/{RUN}").mkdir(exist_ok=True, parents=True)
    Path(f"results/{RUN}/results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
