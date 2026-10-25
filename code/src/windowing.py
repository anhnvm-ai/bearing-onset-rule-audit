"""Windowing: split each hourly file into fixed-length windows spread across the
recording, and compute vibration-only handcrafted features per window.

- Window length 8192 samples @ 25.6 kHz = 0.32 s (~9.5 shaft revolutions @ 1775 RPM).
- Window starts are spread with np.linspace across the whole 78 s burst (NOT the
  first-N consecutive windows), so they represent the full recording.
- Features are vibration-only on purpose: E1/E2 must NOT leak temperature or hour id.
"""
from __future__ import annotations
import numpy as np

from .features import _vib_stats

WINDOW = 8192
WINDOWS_PER_HOUR = 50


def window_starts(n: int, win: int = WINDOW, k: int = WINDOWS_PER_HOUR) -> np.ndarray:
    if n <= win:
        return np.array([0], dtype=np.int64)
    return np.linspace(0, n - win, num=k, dtype=np.int64)


def window_features(arr: np.ndarray, win: int = WINDOW, k: int = WINDOWS_PER_HOUR) -> list[dict]:
    """Return one feature dict per window (vibration X/Y/magnitude stats)."""
    vx, vy = arr[:, 0], arr[:, 1]
    n = arr.shape[0]
    out = []
    for wi, s in enumerate(window_starts(n, win, k)):
        wx = vx[s:s + win]
        wy = vy[s:s + win]
        wmag = np.sqrt(wx ** 2 + wy ** 2)
        f: dict = {}
        f.update(_vib_stats(wx, "vx"))
        f.update(_vib_stats(wy, "vy"))
        f.update(_vib_stats(wmag, "mag"))
        f["win_idx"] = wi
        f["start"] = int(s)
        f["n_nan"] = int(np.isnan(wx).sum() + np.isnan(wy).sum())
        out.append(f)
    return out
