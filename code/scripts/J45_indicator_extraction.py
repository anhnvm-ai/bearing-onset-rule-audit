"""J45 -- per-record spectral and envelope indicators, for the datasets Section IV interprets.

Every indicator the paper uses is a time-domain statistic of one vibration channel, and Section VII says
so. This computes indicators of two other kinds from the same records, so that Section IV's measurement
can be repeated on them:

  spec_centroid     the centre of mass of the amplitude spectrum, in Hz
  band_power_fixed  the share of power in 2.5-4.5 kHz, the resonance band src/envelope.py derives from
                    the NSK PSD. Applying one band to every bearing is a choice, so it is reported beside
  band_power_auto   the share of power in the 2 kHz band above 1 kHz with the most power in that
                    trajectory's OWN baseline window, which fixes the band without looking at the
                    degraded part. The floor keeps the search in the range where the envelope
                    technique looks for a resonance, away from the shaft and harmonic lines
  env_rms           the root mean square of the Hilbert envelope of the band-passed signal
  env_kurtosis      the kurtosis of that envelope, the classical early-fault statistic

Spectral ENERGY is deliberately not among them: by Parseval it is the record's mean square, so it is the
square of the RMS the paper already uses, and reporting it as another indicator would be circular.

Sources. XJTU-SY reads the windowed samples of colab_bundle, the same array J7 reads. FEMTO has no such
cache, so its acceleration files are read directly and paired with the temperature files exactly as
J7c_femto_channel_test.py pairs them, which is what put FEMTO on the one-minute temperature grid in the
first place. IMS and NSK 6205 are not extracted: Section IV interprets XJTU-SY and FEMTO.

Outputs: results/tables/j45_indicators_xjtu.csv   one row per record
         results/tables/j45_indicators_femto.csv
         results/logs/j45_run.log

Usage: python scripts/J45_indicator_extraction.py [--limit N]     (N records per dataset, for a dry run)
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import next_fast_len, rfft, rfftfreq
from scipy.signal import butter, hilbert, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BUNDLE = ROOT / "colab_bundle"
FEMTO_RAW = Path(os.environ.get("BEARING_DATASETS", "datasets")) / "FEMTO/raw"

FS = {"xjtu": 25_600.0, "femto": 25_600.0}
BAND = (2500.0, 4500.0)        # src/envelope.py, from the NSK PSD
AUTO_WIDTH = 2000.0            # the width of the per-trajectory band
AUTO_FLOOR = 1000.0            # searched above this, where the envelope technique looks for a resonance;
                               # below it the spectrum is dominated by shaft and harmonic lines
BASE_FRAC = 0.20               # the baseline window, as everywhere else in this paper
BAR = "=" * 78

LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def spectrum(sig: np.ndarray, fs: float):
    n = len(sig)
    nf = next_fast_len(n)
    S = np.abs(rfft(sig - sig.mean(), n=nf))
    return rfftfreq(nf, d=1.0 / fs), S


def indicators(sig: np.ndarray, fs: float, band: tuple[float, float]) -> dict[str, float]:
    f, S = spectrum(sig, fs)
    P = S ** 2
    tot = float(P.sum()) or 1.0
    in_fixed = (f >= BAND[0]) & (f <= BAND[1])
    in_auto = (f >= band[0]) & (f <= band[1])
    sos = butter(4, list(BAND), btype="band", fs=fs, output="sos")
    filt = sosfiltfilt(sos, sig)
    env = np.abs(hilbert(filt, N=next_fast_len(len(filt))))[:len(filt)]
    z = (env - env.mean()) / max(env.std(), 1e-12)
    return {"spec_centroid": float((f * P).sum() / tot),
            "band_power_fixed": float(P[in_fixed].sum() / tot),
            "band_power_auto": float(P[in_auto].sum() / tot),
            "env_rms": float(np.sqrt((env ** 2).mean())),
            "env_kurtosis": float((z ** 4).mean())}


def auto_band(sigs: list[np.ndarray], fs: float) -> tuple[float, float]:
    """The AUTO_WIDTH-wide band with the most power in the baseline records of one trajectory."""
    acc = None
    for s in sigs:
        f, S = spectrum(s, fs)
        acc = S ** 2 if acc is None else acc + S ** 2
    step = f[1] - f[0]
    w = max(int(AUTO_WIDTH / step), 1)
    csum = np.concatenate([[0.0], np.cumsum(acc)])
    power = csum[w:] - csum[:-w]
    lo = int(AUTO_FLOOR / step)                      # only bands whose lower edge is above the floor
    i = lo + int(np.argmax(power[lo:]))
    return float(f[i]), float(f[i] + AUTO_WIDTH)


def do_xjtu(limit: int | None) -> pd.DataFrame:
    rec = pd.read_csv(BUNDLE / "xjtu" / "records.csv")
    W = np.load(BUNDLE / "xjtu" / "windows.npy", mmap_mode="r")
    n_win = len(W) // len(rec)
    log(f"  XJTU-SY: {len(rec)} records, {n_win} windows of {W.shape[-1]} samples each, fs {FS['xjtu']:.0f} Hz")
    rows = []
    for traj, g in rec.groupby("trajectory_id", sort=False):
        idx = g.index.to_numpy()
        nb = max(5, int(len(idx) * BASE_FRAC))
        base_sigs = []
        for i in idx[:nb]:
            blk = np.asarray(W[i * n_win:(i + 1) * n_win], dtype=np.float64)
            base_sigs.append(np.sqrt((blk ** 2).sum(axis=0)).ravel() if blk.shape[0] > 1
                             else np.abs(blk).ravel())
        band = auto_band(base_sigs, FS["xjtu"])
        for k, i in enumerate(idx):
            if limit and k >= limit:
                break
            blk = np.asarray(W[i * n_win:(i + 1) * n_win], dtype=np.float64)
            mag = np.sqrt((blk ** 2).sum(axis=1)) if blk.shape[1] > 1 else np.abs(blk[:, 0])
            per = [indicators(mag[j], FS["xjtu"], band) for j in range(mag.shape[0])]
            row = {"dataset": "xjtu", "trajectory": traj, "record_index": int(g.record_index.iloc[k]),
                   "band_lo": band[0], "band_hi": band[1]}
            row |= {k2: float(np.mean([p[k2] for p in per])) for k2 in per[0]}
            rows.append(row)
        log(f"    {traj:12} {len(idx):5} records   auto band {band[0]:7.0f}-{band[1]:7.0f} Hz")
    return pd.DataFrame(rows)


def read_femto(path: Path, col: int) -> np.ndarray:
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        delim = ";" if sample.count(";") > sample.count(",") else ","
        return np.array([r for r in csv.reader(fh, delimiter=delim) if len(r) > col], dtype=object)


def do_femto(limit: int | None) -> pd.DataFrame:
    best: dict[str, tuple[int, Path]] = {}
    for d in FEMTO_RAW.rglob("Bearing*"):
        if not d.is_dir() or not list(d.glob("temp_*.csv")):
            continue
        n = len(list(d.glob("acc_*.csv")))
        if d.name not in best or n > best[d.name][0]:
            best[d.name] = (n, d)
    log(f"  FEMTO: {len(best)} bearings with temperature files, paired as J7c pairs them")
    rows = []
    for b, (_, d) in sorted(best.items()):
        accs = sorted(d.glob("acc_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
        tmps = sorted(d.glob("temp_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
        n_t, n_a = len(tmps), len(accs)
        picks = [min(n_a - 1, int(round((k + 0.5) * n_a / n_t))) for k in range(n_t)]
        nb = max(5, int(n_t * BASE_FRAC))
        base_sigs = []
        for j in picks[:nb]:
            A = read_femto(accs[j], 5)
            base_sigs.append(np.sqrt(A[:, 4].astype(float) ** 2 + A[:, 5].astype(float) ** 2))
        band = auto_band(base_sigs, FS["femto"])
        for k, j in enumerate(picks):
            if limit and k >= limit:
                break
            A = read_femto(accs[j], 5)
            mag = np.sqrt(A[:, 4].astype(float) ** 2 + A[:, 5].astype(float) ** 2)
            row = {"dataset": "femto", "trajectory": b, "record_index": k,
                   "band_lo": band[0], "band_hi": band[1]} | indicators(mag, FS["femto"], band)
            rows.append(row)
        log(f"    {b:12} {n_t:5} records   auto band {band[0]:7.0f}-{band[1]:7.0f} Hz")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="records per trajectory, for a dry run")
    a = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)
    log(BAR)
    log("J45  SPECTRAL AND ENVELOPE INDICATORS".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}"
        + (f"  [dry run: {a.limit} records per trajectory]" if a.limit else ""))
    log(f"fixed band {BAND[0]:.0f}-{BAND[1]:.0f} Hz (src/envelope.py, from the NSK PSD); the automatic band is the "
        f"{AUTO_WIDTH:.0f} Hz window with the most power in each trajectory's own baseline")

    for ds, fn in (("xjtu", do_xjtu), ("femto", do_femto)):
        df = fn(a.limit)
        out = T / f"j45_indicators_{ds}.csv"
        if not a.limit:
            df.to_csv(out, index=False)
            log(f"  wrote {out.name}: {len(df)} records x {len(df.columns)} columns")
        else:
            log(f"  dry run: {len(df)} records, not written")

    log(f"\nfinished {datetime.now():%Y-%m-%d %H:%M:%S}")
    (LOGS / "j45_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
