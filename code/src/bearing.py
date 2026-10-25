"""NSK 6205 deep-groove ball bearing — geometry and characteristic defect frequencies.

Dataset: Mendeley "Ball Bearing Vibration and Temperature Run-to-Failure" (5hcdd3tdvb).
NSK 6205: ID 25 mm, OD 52 mm, W 15 mm. Standard 6205 internal geometry (identical to
the widely-cited CWRU 6205): Z = 9 balls, ball diameter d = 7.94 mm,
pitch diameter D = 39.04 mm, contact angle = 0 deg.

Defect-frequency multipliers (× shaft rate f_r): BPFO 3.585, BPFI 5.415, BSF 2.356, FTF 0.398.
Operating speed 1770-1780 RPM => f_r ~ 29.5-29.7 Hz.
"""
from __future__ import annotations
import numpy as np

Z = 9
BALL_D = 7.94       # mm
PITCH_D = 39.04     # mm
CONTACT_ANGLE = 0.0  # deg
NOMINAL_RPM = 1775.0


def shaft_rate(rpm: float = NOMINAL_RPM) -> float:
    return rpm / 60.0


def defect_frequencies(fr: float | None = None) -> dict:
    """Characteristic frequencies (Hz) given shaft rate fr (Hz)."""
    if fr is None:
        fr = shaft_rate()
    r = (BALL_D / PITCH_D) * np.cos(np.deg2rad(CONTACT_ANGLE))
    return {
        "shaft": fr,
        "FTF": 0.5 * (1.0 - r) * fr,
        "BSF": (PITCH_D / (2.0 * BALL_D)) * (1.0 - r ** 2) * fr,
        "BPFO": (Z / 2.0) * (1.0 - r) * fr,
        "BPFI": (Z / 2.0) * (1.0 + r) * fr,
    }
