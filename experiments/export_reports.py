"""Copy the reports behind the README's numbers, and the graded outputs they are computed from, into reports/
(the committed, public copy; results/ stays local). Re-run after regenerating any report.

Reports: results/<name>.md -> reports/<name>.md. Outputs: every local evaluation file listed in src/outputs.py PUBLIC
-> reports/outputs/<set>/<model>.jsonl.gz, exactly one record per puzzle of its set (a retried puzzle keeps its last
record; checked here). Each record has the puzzle, the model's reasoning and answer, the parsed move and line, and the
grading (status, line_legal, full_line_correct, ...). Three fields are dropped to keep the files small: raw_response
(the full API response) and raw_text repeat the reasoning and the answer, and the prompt is rebuilt exactly by
src/run_teacher.py build_prompt(puzzle, record["format"]) (every model got the same prompt for a puzzle).
"""
import gzip
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from outputs import PUBLIC, SETS, public_path  # noqa: E402

REPORTS = [  # public report name; source: results/<name>, except the LoRA scaling report (its run folder)
    "teacher_baseline.md", "students_lora_pilot.md", "students_lora_scaling.md", "student_A.md",
    "students_A_vs_B.md", "rl_answer_only_reward.md", "rl_rewards_B.md", "fresh_test.md",
    "answer_first_and_sampling.md", "robustness_checks.md", "baselines_and_line_soundness.md",
    "replication_and_fresh_test_2.md",
]
SOURCE = {"students_lora_scaling.md": "results/night2/results.md"}
DROP = ("raw_response", "raw_text", "prompt")


def main():
    out = Path("reports")
    for name in REPORTS:
        shutil.copyfile(SOURCE.get(name, f"results/{name}"), out / name)
    ids = {s: {json.loads(line)["puzzle_id"] for line in open(f)} for s, f in SETS.items()}
    total, n = 0, 0
    for src, name in PUBLIC.items():
        if not Path(src).exists():
            continue  # not every model was evaluated on every set
        last = {}
        for line in open(src):
            r = json.loads(line)
            last[r["puzzle_id"]] = r
        puzzle_set = name.split("/")[0]
        assert set(last) == ids[puzzle_set], f"{src}: {len(last)} puzzles, not the {len(ids[puzzle_set])} of {puzzle_set}"
        assert not any(r.get("status") == "error" for r in last.values()), f"{src}: unresolved errors"
        data = "".join(json.dumps({k: v for k, v in r.items() if k not in DROP}, ensure_ascii=False) + "\n"
                       for r in last.values()).encode()
        dst = public_path(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # leave the file alone when its content hasn't changed: gzip headers differ between writers (git churn)
        if not (dst.exists() and gzip.decompress(dst.read_bytes()) == data):
            with gzip.GzipFile(dst, "wb", compresslevel=9, mtime=0) as g:
                g.write(data)
        total += dst.stat().st_size
        n += 1
    print(f"reports/: {len(REPORTS)} reports, {n} output files ({total / 1e6:.1f} MB compressed)")


if __name__ == "__main__":
    main()
