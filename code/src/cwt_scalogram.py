"""CWT scalogram generation (amplitude-preserving) for the CWT-ResNet18 branch.

Continuous wavelet transform (Morlet) of the vibration magnitude over a log-frequency
grid covering the bearing defect frequencies through the resonance band and harmonics
(~10 Hz .. 8 kHz). Stored as RAW log-magnitude (log1p|CWT|), NOT per-image normalised,
so that E3 can compare per-image vs global/train-fold normalisation, and amplitude
(the Healthy<->Severe cue) is preserved. Output image is [n_scales x out_size].
"""
from __future__ import annotations
import numpy as np
import pywt
from scipy.signal import decimate

from src.data_loader import FS

WAVELET = "morl"
N_SCALES = 64
DECIM = 2                      # decimate to 12.8 kHz (Nyquist 6.4 kHz) — keeps resonance band
FS_EFF = FS // DECIM
FMIN, FMAX = 10.0, 6000.0      # covers defect freqs .. resonance band .. harmonics
OUT_SIZE = 64


def scales_for(fs: float = FS_EFF, n_scales: int = N_SCALES, wavelet: str = WAVELET,
               fmin: float = FMIN, fmax: float = FMAX):
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_scales)
    fc = pywt.central_frequency(wavelet)
    return fc * fs / freqs, freqs


_SCALES, _FREQS = scales_for()


def cwt_scalogram(mag: np.ndarray, out_size: int = OUT_SIZE) -> np.ndarray:
    """Return a [N_SCALES, out_size] float32 log-magnitude scalogram (unnormalised)."""
    if DECIM > 1:
        mag = decimate(mag, DECIM, ftype="fir")
    coef, _ = pywt.cwt(mag, _SCALES, WAVELET, sampling_period=1.0 / FS_EFF)
    S = np.log1p(np.abs(coef)).astype(np.float32)      # [n_scales, len]
    n = S.shape[1]
    if n != out_size:                                  # time-axis block-mean downsample
        g = n // out_size
        S = S[:, : g * out_size].reshape(S.shape[0], out_size, g).mean(axis=2)
    return S.astype(np.float32)
