"""Envelope analysis (High-Frequency Resonance Technique) for bearing fault detection.

Band-pass the signal around the structural resonance (2.5-4.5 kHz, from PSD), take the
Hilbert envelope, and compute the envelope spectrum. Bearing defects amplitude-modulate
the resonance, producing lines at the defect frequencies (BPFO/BPFI/BSF/FTF) in the
envelope spectrum — classically visible BEFORE broadband RMS/kurtosis rise.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.fft import rfft, rfftfreq, next_fast_len

RESONANCE_BAND = (2500.0, 4500.0)   # Hz — impact-excited resonance from PSD


def band_envelope(sig: np.ndarray, fs: float, band=RESONANCE_BAND) -> np.ndarray:
    sos = butter(4, list(band), btype="band", fs=fs, output="sos")
    filt = sosfiltfilt(sos, sig)
    n = len(filt)
    env = np.abs(hilbert(filt, N=next_fast_len(n)))[:n]
    return env


def envelope_spectrum(env: np.ndarray, fs: float):
    e = env - env.mean()
    n = len(e)
    nf = next_fast_len(n)
    S = np.abs(rfft(e, n=nf)) / n
    f = rfftfreq(nf, d=1.0 / fs)
    return f, S


def defect_index(f: np.ndarray, S: np.ndarray, fc: float, tol: float = 0.02,
                 nharm: int = 3, floor_band=(5.0, 1000.0)) -> float:
    """Peak-to-noise-floor ratio summed over the first `nharm` harmonics of fc.

    tol allows +/-2% slip. Floor = median envelope-spectrum amplitude in floor_band.
    A value >> 1 means a clear defect line; tracking it over time gives early warning.
    """
    fm = (f > floor_band[0]) & (f < floor_band[1])
    floor = float(np.median(S[fm])) + 1e-12
    total = 0.0
    for h in range(1, nharm + 1):
        target = fc * h
        w = (f > target * (1 - tol)) & (f < target * (1 + tol))
        if w.any():
            total += float(S[w].max())
    return total / (floor * nharm)
