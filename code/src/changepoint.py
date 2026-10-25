"""Optimal piecewise-constant change-point detection via dynamic programming.

Partitions a 1-D series into k contiguous segments minimising the total
within-segment L2 (sum-of-squares) cost — the same objective as ruptures'
Dynp/L2 model, implemented directly (n=129 is tiny, so O(k n^2) is trivial).
Used to (a) locate the run-to-failure change-points and (b) test whether a
3-segment fit is meaningfully better than 2 (i.e. whether an 'Early' stage
is actually supported by the data).
"""
from __future__ import annotations
import numpy as np


def _seg_cost_tables(x: np.ndarray):
    n = len(x)
    ps = np.concatenate([[0.0], np.cumsum(x)])
    ps2 = np.concatenate([[0.0], np.cumsum(x * x)])
    return ps, ps2, n


def _cost(ps, ps2, i, j) -> float:
    """L2 cost of segment x[i:j] around its mean."""
    m = j - i
    if m <= 0:
        return 0.0
    s = ps[j] - ps[i]
    s2 = ps2[j] - ps2[i]
    return float(s2 - s * s / m)


def segment(x: np.ndarray, k: int):
    """Return (boundaries, total_cost) for the optimal k-segment partition.

    boundaries: list of (start, end) index pairs, end-exclusive, covering [0, n).
    """
    x = np.asarray(x, dtype=np.float64)
    ps, ps2, n = _seg_cost_tables(x)
    INF = np.inf
    dp = np.full((k + 1, n + 1), INF)
    back = np.zeros((k + 1, n + 1), dtype=int)
    dp[0, 0] = 0.0
    for kk in range(1, k + 1):
        for j in range(kk, n + 1):
            best, arg = INF, kk - 1
            for i in range(kk - 1, j):
                c = dp[kk - 1, i] + _cost(ps, ps2, i, j)
                if c < best:
                    best, arg = c, i
            dp[kk, j] = best
            back[kk, j] = arg
    bounds = []
    j = n
    for kk in range(k, 0, -1):
        i = back[kk, j]
        bounds.append((i, j))
        j = i
    bounds.reverse()
    return bounds, float(dp[k, n])


def total_ss(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sum((x - x.mean()) ** 2))


def elbow_analysis(x: np.ndarray, kmax: int = 4) -> dict:
    """Cost vs k and the variance-explained gain of each extra segment.

    Returns per-k boundaries, total cost, and the marginal fraction of total
    variance explained by going k-1 -> k. A large gain at k=2 (the failure
    cliff) followed by a small gain at k=3 is evidence that a third ('Early')
    stage is NOT well supported.
    """
    tss = total_ss(x)
    out = {"total_ss": tss, "k": {}}
    prev = tss
    for k in range(1, kmax + 1):
        bounds, cost = segment(x, k)
        gain = (prev - cost) / tss if tss > 0 else 0.0
        out["k"][k] = {
            "boundaries": bounds,
            "cost": cost,
            "var_explained": 1.0 - cost / tss if tss > 0 else 0.0,
            "marginal_gain": gain,
            "cut_indices": [b[0] for b in bounds[1:]],  # internal cut points
        }
        prev = cost
    return out
