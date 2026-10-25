"""Hour-level health-indicator features for degradation-stage EDA.

All statistics are NaN-aware (the raw data contains systematic 'NaN' blocks).
Vibration is processed at the fast time scale (per-file signal); temperature is
summarised at the hour/file scale (it barely moves within a 0.16 s window).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import hilbert
from scipy.fft import next_fast_len

from .data_loader import FS


def _interp_nan(a: np.ndarray) -> np.ndarray | None:
    """Linearly interpolate NaNs (short blocks); return None if all-NaN."""
    a = a.astype(np.float64, copy=True)
    m = np.isnan(a)
    if not m.any():
        return a
    if m.all():
        return None
    idx = np.arange(len(a))
    a[m] = np.interp(idx[m], idx[~m], a[~m])
    return a


def _vib_stats(sig: np.ndarray, prefix: str) -> dict:
    s = sig[~np.isnan(sig)]
    out = {}
    if s.size == 0:
        for k in ["rms", "std", "p2p", "crest", "kurtosis", "skew", "absmax"]:
            out[f"{prefix}_{k}"] = np.nan
        return out
    rms = float(np.sqrt(np.mean(s ** 2)))
    peak = float(np.max(np.abs(s)))
    out[f"{prefix}_rms"] = rms
    out[f"{prefix}_std"] = float(np.std(s))
    out[f"{prefix}_p2p"] = float(np.ptp(s))
    out[f"{prefix}_crest"] = peak / rms if rms > 0 else np.nan
    out[f"{prefix}_kurtosis"] = float(kurtosis(s, fisher=True, bias=False))
    out[f"{prefix}_skew"] = float(skew(s, bias=False))
    out[f"{prefix}_absmax"] = peak
    return out


def _envelope_kurtosis(mag: np.ndarray) -> float:
    """Kurtosis of the Hilbert envelope of the vibration magnitude.

    The gold-standard impulsiveness indicator for bearing faults. Computed on
    NaN-interpolated magnitude, FFT length padded to a fast size for speed.
    """
    m = _interp_nan(mag)
    if m is None:
        return np.nan
    n = len(m)
    env = np.abs(hilbert(m, N=next_fast_len(n)))[:n]
    return float(kurtosis(env, fisher=True, bias=False))


def hour_features(arr: np.ndarray) -> dict:
    """Compute all hour-level features for one file's (N, 4) array."""
    vx, vy, tb, ta = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    mag = np.sqrt(vx ** 2 + vy ** 2)

    feats: dict = {}
    for name, sig in [("vx", vx), ("vy", vy), ("mag", mag)]:
        feats.update(_vib_stats(sig, name))
    feats["mag_env_kurtosis"] = _envelope_kurtosis(mag)

    # Temperature (hour-level, NaN-aware)
    tbc, tac = tb[~np.isnan(tb)], ta[~np.isnan(ta)]
    feats["bearing_temp_mean"] = float(np.mean(tbc)) if tbc.size else np.nan
    feats["bearing_temp_std"] = float(np.std(tbc)) if tbc.size else np.nan
    feats["bearing_temp_min"] = float(np.min(tbc)) if tbc.size else np.nan
    feats["bearing_temp_max"] = float(np.max(tbc)) if tbc.size else np.nan
    feats["ambient_temp_mean"] = float(np.mean(tac)) if tac.size else np.nan
    feats["delta_temp_mean"] = (
        feats["bearing_temp_mean"] - feats["ambient_temp_mean"]
        if tbc.size and tac.size else np.nan
    )

    # Within-file bearing-temp slope (deg C per second) — a candidate feature
    tb_i = _interp_nan(tb)
    if tb_i is not None:
        t = np.arange(len(tb_i)) / FS
        slope = np.polyfit(t, tb_i, 1)[0]
        feats["bearing_temp_slope"] = float(slope)
    else:
        feats["bearing_temp_slope"] = np.nan

    # NaN accounting (data-quality gate)
    feats["nan_vx"] = int(np.isnan(vx).sum())
    feats["nan_vy"] = int(np.isnan(vy).sum())
    feats["nan_bearing"] = int(np.isnan(tb).sum())
    feats["nan_ambient"] = int(np.isnan(ta).sum())
    feats["n_samples"] = int(arr.shape[0])
    return feats
