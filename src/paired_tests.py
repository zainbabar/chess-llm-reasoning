"""Paired statistics used by every report: exact McNemar tests, paired bootstrap intervals and the Holm correction.

Two models are compared on the same puzzles. Only the puzzles where exactly one of them is right carry information, so
the McNemar test asks whether that split is more lopsided than coin flips would give.
"""
from math import comb

import numpy as np


def mcnemar(a, b):
    """Exact two-sided McNemar p-value from discordant counts a (only arm1 right) and b (only arm2 right)."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def pair(a, b):
    """McNemar counts and p, plus the gap in points with a 95% paired bootstrap interval (puzzles resampled together)."""
    ids = sorted(set(a) & set(b))
    x = sum(a[i]["status"] == "correct" and b[i]["status"] != "correct" for i in ids)
    y = sum(b[i]["status"] == "correct" and a[i]["status"] != "correct" for i in ids)
    d = np.array([(a[i]["status"] == "correct") - (b[i]["status"] == "correct") for i in ids], dtype=float)
    boots = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(axis=1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return f"{x} vs {y}, p = {mcnemar(x, y):.2g}; gap {100 * d.mean():+.1f} points (95% interval {lo:+.1f} to {hi:+.1f})"


def holm(ps):
    """Holm-adjusted p-values, in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj
