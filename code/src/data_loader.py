"""Data loading for the bearing run-to-failure dataset.

Each raw file `LogFile_YYYY-MM-DD-HH-MM-SS.csv` is one hourly ~78 s burst,
2,000,001 rows, 4 columns, NO header, CRLF line endings:
    col0 = vibration X, col1 = vibration Y, col2 = bearing temp, col3 = ambient temp.
The literal token 'NaN' appears in blocks across ALL files and BOTH modalities;
pandas parses it to np.nan automatically (default na_values).
"""
from __future__ import annotations
import os
import re
import glob
from datetime import datetime
import numpy as np
import pandas as pd

FS = 25_600                      # Hz — corroborated by PSD (resonance 2.5-4.5 kHz, shaft ~29.7 Hz)
COLS = ["vx", "vy", "bearing_temp", "ambient_temp"]

# Absolute paths (Google Drive persists across Colab resets)
PROJECT_ROOT = "/content/drive/MyDrive/ATC2026/bearing_multimodal_atc"
RAW_DIR = "/content/drive/MyDrive/ATC2026/ATC2026"
RESULTS = os.path.join(PROJECT_ROOT, "results")

_TS_RE = re.compile(r"LogFile_(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.csv$")


def parse_timestamp(path: str) -> datetime:
    m = _TS_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse timestamp from {path}")
    y, mo, d, h, mi, s = map(int, m.groups())
    return datetime(y, mo, d, h, mi, s)


def list_files(raw_dir: str = RAW_DIR) -> pd.DataFrame:
    """Return a chronologically-sorted metadata table of all raw files."""
    paths = sorted(glob.glob(os.path.join(raw_dir, "LogFile_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No LogFile_*.csv in {raw_dir}")
    rows = []
    for p in paths:
        ts = parse_timestamp(p)
        rows.append({"file_name": os.path.basename(p), "path": p, "timestamp": ts})
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df["hour_id"] = np.arange(len(df))
    t0 = df["timestamp"].iloc[0]
    df["hours_from_start"] = (df["timestamp"] - t0).dt.total_seconds() / 3600.0
    return df


def load_file(path: str) -> np.ndarray:
    """Load one raw CSV as an (N, 4) float64 array; 'NaN' tokens become np.nan."""
    df = pd.read_csv(
        path, header=None, names=COLS, dtype=np.float64,
        na_values=["NaN", "nan", ""], engine="c",
    )
    return df.to_numpy()
