"""J63 -- WP-E: known-change simulation with gradual and nonlinear degradation and varying noise (R3.5).

R3.5: a comment of the review reports, answered in the response letter.

The submitted simulation inserted a STEP into Gaussian AR(1) noise. Real bearing indicators drift and then grow
roughly exponentially, so the time of 'the' change is not defined for them. This script simulates

    h_t = mu + sigma g(t/n) + e_t,          s = max(0, (t/n - u0)/(1 - u0)),  u0 in {0.3, 0.5, 0.7}

    step (control)          g = delta * 1[t/n >= u0]
    linear ramp             g = delta * s
    two-phase exponential   g = delta * (exp(5 s) - 1)/(exp(5) - 1)

with Gaussian AR(1) noise at phi = 0.367 (the XJTU-SY reference) or heteroscedastic noise whose innovation sd grows
with the degradation (multiplied by 1 + g/delta), and an end-of-life size delta in {2, 6} baseline sd. The warm-up
scenario of the analysis plan is not simulated: Gate 1 rule G1-b drops it because J60 did not support H_w.

Every series is passed through the unchanged rules under P-retro and P-prospective, at canonical settings and at the
5% false-alarm calibration of J61. Reference times: t_d' = first record at which the noiseless g reaches d' in {1, 3}.
Reported per cell, protocol and rule: alarms before t_d', misses, delay after t_d' (records and life fraction), median
onset fraction u, and b1 - 1 (slope of log u on log n over the series of the cell, pooled over the 15 or 13 XJTU-SY
lengths). This tests directly the common-shape model of the submitted Eq. (8): with h_t = mu + sigma g(t/n) + e, a
rule that answers to a fixed size of change alarms at a fixed fraction of life (b1 - 1 = 0).

Pre-specified in paper/access_resub/ANALYSIS_PLAN_2026-10-24.md, Section 11; amendment A5.

Inputs : results/tables/j61_calibration.csv (multipliers), XJTU-SY trajectory lengths (fig1_fpt_matrix.csv)
Outputs: results/tables/j63_gradual.csv   one row per shape x u0 x noise x delta x protocol x rule
         results/logs/j63_run.log
Usage  : python scripts/J63_gradual_simulation.py          (400 series per length and cell)
         R=50 python scripts/J63_gradual_simulation.py     (quick run)
"""
from __future__ import annotations

import importlib.util
import itertools
import os
import sys
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
SEED = 63
R = int(os.environ.get("R", "400"))
PHI = 0.367
SHAPES = ("step", "ramp", "exp")
U0S = (0.3, 0.5, 0.7)
NOISES = ("ar1", "hetero")
DELTAS = (2.0, 6.0)
REFS = (1.0, 3.0)
PROTOCOLS = ("P-prospective", "P-retro")
CANON = {"3sig_rms": ("thr", 3.0), "ma3sig": ("ma", 3.0), "ma5sig": ("ma", 5.0), "ewma": ("ewma", 3.0),
         "cusum": ("cusum", 5.0)}
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shape_g(shape: str, n: int, u0: float, delta: float) -> np.ndarray:
    f = np.arange(n) / n
    s = np.maximum(0.0, (f - u0) / (1 - u0))
    if shape == "step":
        return delta * (f >= u0).astype(float)
    if shape == "ramp":
        return delta * s
    if shape == "exp":
        return delta * (np.exp(5 * s) - 1) / (np.exp(5) - 1)
    raise ValueError(shape)


def simulate_cell(rng: np.random.Generator, n_series: int, g: np.ndarray, noise: str, delta: float) -> np.ndarray:
    e = rng.standard_normal((n_series, len(g)))
    if noise == "hetero":
        e = e * (1 + g / delta)[None, :]
    a = np.sqrt(1 - PHI ** 2)
    e[:, 0] = e[:, 0] / a
    return lfilter([a], [1.0, -PHI], e, axis=1) + g[None, :]


def first_alarm(stat: np.ndarray, m: float, t0: int) -> np.ndarray:
    """First t >= t0 with stat > m at t, t+1, t+2; -1 if none."""
    n = stat.shape[1]
    if n - t0 < 3:
        return np.full(stat.shape[0], -1)
    hit = np.minimum(np.minimum(stat[:, t0:n - 2], stat[:, t0 + 1:n - 1]), stat[:, t0 + 2:n]) > m
    anyh = hit.any(1)
    return np.where(anyh, hit.argmax(1) + t0, -1)


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j63_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J63  GRADUAL AND NONLINEAR DEGRADATION, VARYING NOISE  (WP-E)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}   R = {R} series per length and cell   phi = {PHI}")
    if R < 400:
        log("NOTE: quick run (R < 400); the manuscript numbers need the default")

    j61 = load("j61", "J61_far_calibration.py")
    calib = pd.read_csv(T / "j61_calibration.csv")
    if not (calib.R == 4000).all():
        log("FAIL: J61 tables are from a quick run")
        return 1
    cal = {(r.protocol, r.rule): r.multiplier for r in calib[(calib.dataset == "xjtu") & (calib.noise_arm == "main")
                                                             & (calib.level == 0.05)].itertuples()}
    ft = pd.read_csv(T / "fig1_fpt_matrix.csv")
    lengths = sorted(ft[(ft.dataset == "xjtu")].drop_duplicates("trajectory").n_records.astype(int).tolist())
    rules = [(name, kind, m, "canonical") for name, (kind, m) in CANON.items()]
    log(f"XJTU-SY lengths: {lengths}")

    rows = []
    for shape, u0, noise, delta in itertools.product(SHAPES, U0S, NOISES, DELTAS):
        cell_seed = zlib.crc32(f"{SEED}|{shape}|{u0}|{noise}|{delta}".encode())
        rng = np.random.default_rng(cell_seed)
        acc = {(p, name, c): {"i": [], "n": [], "t1": [], "t3": []}
               for p in PROTOCOLS for name, _, _, c in rules + [(f"{r}_cal", None, None, "matched5")
                                                                 for r in ("thr", "ma", "ewma", "cusum")]}
        for n in lengths:
            g = shape_g(shape, n, u0, delta)
            H = simulate_cell(rng, R, g, noise, delta)
            tref = {d: (int(np.argmax(g >= d)) if (g >= d).any() else -1) for d in REFS}
            for proto in PROTOCOLS:
                if proto == "P-prospective" and n < 80:
                    continue
                W, t0, clean, min_keep = j61.protocol_params(proto, n)
                mu, sd = j61.phase1_vec(H, W, clean, min_keep)
                Z = (H - mu[:, None]) / sd[:, None]
                M = (j61.moving_avg5(H) - mu[:, None]) / sd[:, None]
                E = j61.ewma_std(H, mu, sd, 0.2, t0)
                C = j61.cusum_std(H, mu, sd, 0.5, t0)
                stat = {"thr": Z, "ma": M, "ewma": E, "cusum": C}
                todo = [(name, kind, m, "canonical") for name, kind, m, _ in rules] + \
                       [(f"{r}_cal", r, cal[(proto, r)], "matched5") for r in ("thr", "ma", "ewma", "cusum")]
                for name, kind, m, c in todo:
                    i = first_alarm(stat[kind], m, t0)
                    a = acc[(proto, name, c)]
                    a["i"].append(i); a["n"].append(np.full(R, n)); a["t1"].append(np.full(R, tref[1.0]))
                    a["t3"].append(np.full(R, tref[3.0]))
        for (proto, name, c), a in acc.items():
            i, n, t1, t3 = (np.concatenate(a[k]) for k in ("i", "n", "t1", "t3"))
            fired = i >= 0
            u = (i[fired] + 1) / n[fired]
            row = {"shape": shape, "u0": u0, "noise": noise, "delta": delta, "protocol": proto, "rule": name,
                   "calibration": c, "n_series": len(i), "share_never": float(1 - fired.mean()),
                   "median_u": float(np.median(u)) if fired.any() else np.nan}
            if fired.sum() >= 10 and np.ptp(n[fired]) > 0:
                x, y = np.log(n[fired]), np.log(u)
                xc = x - x.mean()
                row["slope_b1_minus_1"] = float((xc * (y - y.mean())).sum() / (xc * xc).sum())
            else:
                row["slope_b1_minus_1"] = np.nan
            for d, tr in ((1, t1), (3, t3)):
                if (tr >= 0).all():
                    before = fired & (i < tr)
                    after = fired & (i >= tr)
                    row[f"share_before_ref{d}"] = float(before.mean())
                    row[f"median_delay_ref{d}_records"] = float(np.median(i[after] - tr[after])) if after.any() else np.nan
                    row[f"median_delay_ref{d}_fraction"] = float(np.median((i[after] - tr[after]) / n[after])) if after.any() else np.nan
                else:
                    row[f"share_before_ref{d}"] = np.nan
                    row[f"median_delay_ref{d}_records"] = np.nan
                    row[f"median_delay_ref{d}_fraction"] = np.nan
            rows.append(row)
        log(f"  {shape:5s} u0 {u0:.1f} {noise:6s} delta {delta:.0f}: done")

    out = pd.DataFrame(rows)
    out.to_csv(T / "j63_gradual.csv", index=False)

    log(f"\n{BAR}\nSummary: b1 - 1 and alarms before t_1, median over u0 (AR(1) noise, delta = 6)\n{BAR}")
    s = out[(out.noise == "ar1") & (out.delta == 6.0)]
    for proto in PROTOCOLS:
        log(f"  {proto}")
        for shape in SHAPES:
            q = s[(s.protocol == proto) & (s["shape"] == shape)].groupby(["rule", "calibration"]).agg(
                slope=("slope_b1_minus_1", "median"), before=("share_before_ref1", "median"),
                never=("share_never", "median"), u=("median_u", "median")).reset_index()
            log(f"    {shape:5s} " + "; ".join(f"{r.rule} b1-1 {r.slope:+.2f}, early {r.before:.2f}, missed {r.never:.2f}"
                                             for r in q.itertuples()))
    log(f"\nwrote results/tables/j63_gradual.csv ({len(out)} rows)")
    log(f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
