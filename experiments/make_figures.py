"""Figures for the README -> figures/. Reads the graded outputs (results/, or the committed copies in reports/outputs/).

1. fresh_results.png: first move right on the 1,000 fresh puzzles (both fresh sets), with a 95% bootstrap interval,
   for the teacher (low / medium effort) and students A, A + 200k and B, annotated with output tokens per answer.
2. by_band.png: the same, per Lichess rating band (about 143 puzzles each).
3. rl_rewards.png: student B during RL under each of the five rewards (checkpoints 0 = after imitation, 130, 260, 390
   steps, development set): accuracy and the text and line checks that show how each reward was gamed.
4. example_sacrifice_*.svg / example_reward_hack.svg: two development-set puzzles drawn with python-chess.
5. social_preview.png: 1280x640 link-preview image for GitHub.
"""
import json
import sys
from pathlib import Path
from statistics import mean

import chess
import chess.svg
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # shared modules live in src/
from outputs import load
from rl_rewards_B_report import stats

OUT = Path("figures")
P = {json.loads(l)["puzzle_id"]: json.loads(l) for l in open("splits/dev.jsonl")}
BANDS = ["800-1000", "1000-1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000", "2000-2200"]
TEACHER_MED, STUDENT_A, STUDENT_B = ("results/test500_med_formatP1L.jsonl", "results/student_path_A_nothink.jsonl",
                                     "results/student_path_B_nothink.jsonl")
# Colors follow the model across figures. The students take slots 1-3 of the reference categorical palette (validated
# for colour-blind separation); the teacher is neutral grey; the five RL rewards take slots 4-8.
INK, MUTED = "#0b0b0b", "#52514e"
GREY, DARK, BLUE, ORANGE, AQUA = "#9e9e9e", "#424242", "#2a78d6", "#eb6834", "#1baf7a"
MODELS = [  # label, fresh set 1 output, fresh set 2 output, color
    ("Student A + 200k (1.7B)", "results/fresh_student_path_A200k.jsonl", "results/fresh2_student_path_A200k.jsonl", AQUA),
    ("Student A (1.7B)", "results/fresh_student_path_A.jsonl", "results/fresh2_student_path_A.jsonl", BLUE),
    ("gpt-oss-120b, medium effort", "results/fresh_med_formatP1L.jsonl", "results/fresh2_med_formatP1L.jsonl", DARK),
    ("Student B (1.7B)", "results/fresh_student_path_B.jsonl", "results/fresh2_student_path_B.jsonl", ORANGE),
    ("gpt-oss-120b, low effort", "results/fresh_low_formatP1L.jsonl", "results/fresh2_low_formatP1L.jsonl", GREY),
]
NOTES = {"Student A + 200k (1.7B)": "answers, ~240k puzzles", "Student A (1.7B)": "answers, 37.5k puzzles",
         "Student B (1.7B)": "teacher explanations, 37.5k", "gpt-oss-120b, medium effort": "zero-shot",
         "gpt-oss-120b, low effort": "zero-shot"}
REWARDS = [("answer only", "rlL_B", "#eda100", "o"), ("truth v1", "rlT_B", "#e87ba4", "s"),
           ("strict v2", "rlT2_B", "#008300", "^"), ("v3", "rlT3_B", "#6250d6", "D"), ("v4", "rlT4_B", "#e34948", "v")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK, "axes.edgecolor": "#b5b4ae"})


def solved(R):
    return sum(r["status"] == "correct" for r in R.values())


def approx(n):
    return f"{round(n, -2):,.0f}" if n >= 1000 else f"{n:.0f}"


def pooled(f1, f2):
    """Graded outputs on both fresh sets (1,000 puzzles), keyed by puzzle id."""
    R1, _ = load(f1)
    R2, _ = load(f2)
    assert len(R1) == len(R2) == 500, (f1, f2)
    return {**R1, **R2}


def score(R):
    """Percent right with a 95% bootstrap interval (puzzles resampled)."""
    x = np.array([r["status"] == "correct" for r in R.values()], dtype=float)
    boots = x[np.random.default_rng(0).integers(0, len(x), (10000, len(x)))].mean(axis=1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return 100 * x.mean(), lo, hi


def dot_rows(ax, size=1.0):
    """The five models as dots with 95% intervals, best at the top. Returns the y-tick labels."""
    labels = []
    for y, (label, f1, f2, color) in enumerate(reversed(MODELS)):
        R = pooled(f1, f2)
        pct, lo, hi = score(R)
        ax.errorbar(pct, y, xerr=[[pct - lo], [hi - pct]], fmt="o", color=color, ecolor=color, capsize=4 * size,
                    ms=8 * size, lw=2 * size, mec="white", mew=1.5 * size)
        labels.append((label, R, pct, hi))
    ax.set_yticks(range(len(labels)), [l[0] for l in labels])
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.grid(axis="x", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return labels


def fig_fresh_results():
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    for y, (label, R, pct, hi) in enumerate(dot_rows(ax)):
        toks = mean(r["usage"]["completion_tokens"] for r in R.values() if r.get("usage"))
        ax.annotate(f"{pct:.1f}%", (hi, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9.5,
                    weight="bold")
        ax.annotate(f"{NOTES[label]}, ~{approx(toks)} tokens per answer", (hi, y), xytext=(46, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(35, 70)
    ax.set_xticks(range(35, 71, 5))
    ax.set_xlabel("First move right on 1,000 puzzles no model had seen (%, 95% interval)")
    fig.subplots_adjust(left=0.235, right=0.6, bottom=0.17, top=0.97)  # the notes sit in the right margin
    fig.savefig(OUT / "fresh_results.png", dpi=180)


def fig_by_band():
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    x = np.arange(len(BANDS))
    markers = {"Student A + 200k (1.7B)": "o", "Student A (1.7B)": "s", "gpt-oss-120b, medium effort": "D",
               "gpt-oss-120b, low effort": "^"}
    for label, f1, f2, color in MODELS:
        if label not in markers:
            continue
        R = pooled(f1, f2)
        vals = []
        for b in BANDS:
            rows = [r for r in R.values() if r["rating_band"] == b]
            vals.append(100 * sum(r["status"] == "correct" for r in rows) / len(rows))
        ax.plot(x, vals, marker=markers[label], ls="-", color=color, lw=2, ms=6.5, mec="white", mew=1.2, label=label)
    ax.set_xticks(x, [b.replace("-", "–") for b in BANDS])
    ax.set_xlim(-0.3, len(BANDS) - 0.7)
    ax.set_xlabel("Lichess puzzle rating band (about 143 puzzles each, both fresh sets)")
    ax.set_ylabel("First move right (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "by_band.png", dpi=180)


def fig_rl():
    sft, _ = load(STUDENT_B)
    base = stats(sft, P)
    panels = [("Solved (of 500)", lambda s: s["correct"]),
              ("False checkmate claims\n(% of 356 non-mate puzzles)", lambda s: 100 * s["false_mate"][0] / s["false_mate"][1]),
              ("Checkable claims per explanation", lambda s: s["claims"]),
              ("Written line length (half-moves)", lambda s: s["line_len"]),
              ("Lines longer than the solution\n(of 500)", lambda s: s["padded"]),
              ("Lines with a legal opponent reply\n(of 500)", lambda s: s["replies"])]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    steps = [0, 130, 260, 390]
    for label, tag, color, marker in REWARDS:
        S = [base] + [stats(load(f"results/student_{tag}_{ck}_nothink.jsonl")[0], P) for ck in ("step130", "step260", "final")]
        for ax, (title, f) in zip(axes.flat, panels):
            ax.plot(steps, [f(s) for s in S], marker=marker, ls="-", color=color, label=label, lw=2, ms=6,
                    mec="white", mew=1)
    A, _ = load(STUDENT_A)
    T, _ = load(TEACHER_MED)
    ref = [axes[0, 0].axhline(solved(A), color=BLUE, ls="--", lw=1.2, label="student A (answers only)"),
           axes[0, 0].axhline(solved(T), color=DARK, ls=":", lw=1.4, label="teacher, medium effort")]
    axes[0, 0].legend(handles=ref, loc="lower center", frameon=False, fontsize=8)
    axes[0, 0].set_ylim(215, 300)
    for ax, (title, _) in zip(axes.flat, panels):
        ax.set_title(title, fontsize=9.5)
        ax.set_xticks(steps)
        ax.grid(color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("RL steps (0 = after imitation)")
    axes[0, 1].legend(title="reward", frameon=False, fontsize=8.5, title_fontsize=8.5)
    fig.suptitle("Student B under RL on the development set: each reward fixed one check and was gamed another way",
                 fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "rl_rewards.png", dpi=180)


def board_svg(name, puzzle_id, arrows=(), play=(), size=300):
    """Draw a test puzzle from the side to move, optionally after playing `play` (uci moves), with (uci, color) arrows."""
    board = chess.Board(P[puzzle_id]["fen"])
    side = board.turn
    for u in play:
        board.push_uci(u)
    kw = {"lastmove": board.peek()} if play else {}
    if board.is_check():
        kw["check"] = board.king(board.turn)
    svg = chess.svg.board(board, orientation=side, size=size, coordinates=True,
                          arrows=[chess.svg.Arrow(chess.parse_square(u[:2]), chess.parse_square(u[2:4]), color=c)
                                  for u, c in arrows], **kw)
    (OUT / name).write_text(svg)


def fig_boards():
    GOOD, BAD = "#15781bcc", "#d62728cc"
    # uURSu: the student plays the forced mate Qxd1+ Kxd1 Re1#; the medium teacher played Qf4+ (about level per Stockfish)
    board_svg("example_sacrifice_1.svg", "uURSu", arrows=[("f3d1", GOOD)])
    board_svg("example_sacrifice_2.svg", "uURSu", play=["f3d1", "c1d1", "e7e1"])
    # yl9Uw: v3's final model writes Rxf2 and then Rd3, a second move by its own side, as the "line"
    board_svg("example_reward_hack.svg", "yl9Uw", arrows=[("e2f2", GOOD), ("d1d3", BAD)])


def fig_social():
    """1280x640 image for GitHub's link preview: the headline plus the fresh-set results."""
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.text(0.05, 0.87, "A 1.7B model beats a 120B model on unseen chess puzzles", fontsize=25, weight="bold")
    fig.text(0.05, 0.795, "Trained on verified answers. Trained on the big model's explanations instead, it did worse.",
             fontsize=16, color=MUTED)
    ax = fig.add_axes([0.30, 0.15, 0.60, 0.55])
    for y, (label, R, pct, hi) in enumerate(dot_rows(ax, size=1.5)):
        ax.text(hi + 1.2, y, f"{pct:.1f}%", va="center", fontsize=16, weight="bold")
    ax.tick_params(axis="y", labelsize=15)
    ax.tick_params(axis="x", labelsize=13)
    ax.set_xlim(35, 75)
    ax.set_xlabel("First move right on 1,000 unseen Lichess puzzles (%, 95% interval)", fontsize=14)
    fig.text(0.05, 0.04, "github.com/zainbabar/chess-llm-reasoning", fontsize=13, color=MUTED)
    fig.savefig(OUT / "social_preview.png", dpi=100)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_fresh_results()
    fig_by_band()
    fig_rl()
    fig_boards()
    fig_social()
    print("wrote", ", ".join(str(p) for p in sorted(OUT.glob("*.*"))))
