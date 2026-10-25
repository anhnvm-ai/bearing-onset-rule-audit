"""J24 -- what each rule recovers when the change time is known, and what reproduces the ARL signature.

Two simulation studies, both applying the unchanged onset rules (vectorised copies from J20) to synthetic series
whose change time is known by construction. The noise is Gaussian AR(1) with the median detrended lag-1
autocorrelation of the real baselines, and the trajectory lengths are those of the real XJTU-SY bearings.

  S1  one change: a step (or ramp) of delta baseline standard deviations at a fixed fraction u of life.
      Reports, per rule: share of series that fire before the change, share that never fire, and the delay
      (i - tau) in records and in fractions of life. Answers "which rule recovers a known change time, and for
      which size of change".

  S2  two changes: a small early drift of delta1 followed by a large change of delta2 at fraction u2. The early
      drift is placed either at a fixed record count r1 (a warm-up that does not scale with the test) or at a
      fixed fraction u1 of life. Reports the slope of log(i + 1) on log n per rule. This isolates what the ARL
      signature needs: a chart tracks the early drift, so its index is length-independent only when that drift
      arrives at a length-independent time.

Outputs: results/tables/j24_sim_single_change.csv, j24_sim_two_change.csv; log results/logs/j24_run.log.
Usage: python scripts/J24_known_change_simulation.py   (R=400 series per cell; R=50 for a quick run)
"""
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
R = int(os.environ.get("R", "400"))
SEED = 20260915
RULES = ["3sig_rms", "ma3sig", "ma5sig", "ma2sig", "ewma", "cusum"]
FAMILY = {"3sig_rms": "detector", "ma3sig": "detector", "ma5sig": "detector", "ma2sig": "detector",
          "ewma": "control chart", "cusum": "control chart"}
_fh = None


def log(m=""):
    print(m, flush=True)
    _fh.write(m + "\n")


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ar1(rng, S, n, phi):
    e = rng.standard_normal((S, n))
    x = np.empty((S, n))
    x[:, 0] = e[:, 0]
    a = np.sqrt(max(1 - phi ** 2, 1e-12))
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + a * e[:, t]
    return x


def slope_log(idx, lens):
    """b1 of log(i + 1) ~ log n over simulated series (idx and lens aligned, only fired series)."""
    ok = idx >= 0
    if ok.sum() < 10 or np.ptp(lens[ok]) == 0:
        return np.nan
    x = np.log(lens[ok].astype(float))
    y = np.log(idx[ok].astype(float) + 1)
    X = np.column_stack([np.ones(len(x)), x])
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])



def crossing_times(j11):
    """S0 (real data): the record at which the indicator first sustains a crossing of mu + k sd, against length."""
    log("\nS0  real data: when does the indicator first sustain a crossing of mu + k sd, and does that time scale with length?")
    log(f"    {'dataset':7s} {'k':>4s} {'n fired':>7s} {'median index':>12s} {'median fraction':>15s} {'rho(index, n)':>14s}")
    rows = []
    data = j11.load_traj()
    for ds in ("xjtu", "femto"):
        for k in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
            idx, lens = [], []
            for tname, g in data[ds].groupby("trajectory", sort=True):
                h = g["rms"].to_numpy()
                n = len(h)
                nb = max(5, int(0.2 * n))
                mu, sd = h[:nb].mean(), max(h[:nb].std(), 1e-12)
                flag = (h > mu + k * sd)[None, :]
                i = int(j20_first_confirmed(flag)[0])
                if i >= 0:
                    idx.append(i)
                    lens.append(n)
            if len(idx) >= 4:
                rho = spearmanr(lens, idx).statistic
                rows.append({"dataset": ds, "k_sd": k, "n_fired": len(idx), "median_index": float(np.median(idx)),
                             "median_fraction": float(np.median(np.array(idx) / np.array(lens))), "rho_index_length": rho})
                log(f"    {ds:7s} {k:4.1f} {len(idx):7d} {np.median(idx):12.0f} {np.median(np.array(idx) / np.array(lens)):15.3f} {rho:+14.3f}")
    pd.DataFrame(rows).to_csv(T / "j24_crossing_times.csv", index=False)


def main():
    global _fh
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j24_run.log", "w", encoding="utf-8")
    j20 = load("j20", "J20_arl_mechanism.py")
    j11 = load("j11", "J11_arl_and_covariates.py")
    global j20_first_confirmed
    j20_first_confirmed = j20.first_confirmed
    base = pd.read_csv(T / "j22_baseline_autocorrelation.csv")
    phi = float(np.clip(base[base.dataset == "xjtu"].phi_detrended.median(), -0.99, 0.99))
    lens = sorted(pd.read_csv(T / "fig1_fpt_matrix.csv").query("dataset == 'xjtu'").drop_duplicates("trajectory").n_records)
    rng = np.random.default_rng(SEED)
    log(f"J24 known-change simulation, started {datetime.now():%Y-%m-%d %H:%M:%S}, R={R} per cell, seed {SEED}")
    crossing_times(j11)
    log(f"AR(1) phi = {phi:.3f} (median detrended baseline autocorrelation on XJTU-SY); "
        f"lengths = the {len(lens)} XJTU-SY trajectory lengths ({min(lens)}-{max(lens)} records)")

    # ---------------------------------------------------------------- S1
    log("\nS1  one change of delta standard deviations at fraction u of life")
    log(f"    {'shape':5s} {'delta':>5s} {'u':>4s} {'rule':9s} {'fire<tau':>8s} {'never':>6s} {'delay (rec)':>12s} {'delay (life)':>13s}")
    rows = []
    for shape in ("step", "ramp"):
        for delta in (1.0, 2.0, 3.0, 6.0):
            for u in (0.3, 0.5, 0.7):
                idx = {r: [] for r in RULES}
                taus, all_lens = [], []
                for n in lens:
                    tau = int(u * n)
                    mean = np.zeros(n)
                    if shape == "step":
                        mean[tau:] = delta
                    else:
                        mean[tau:] = delta * np.linspace(0, 1, n - tau)
                    H = ar1(rng, R, n, phi) + mean
                    for r in RULES:
                        idx[r].append(j20.onsets(H, r))
                    taus.append(np.full(R, tau))
                    all_lens.append(np.full(R, n))
                taus = np.concatenate(taus)
                all_lens = np.concatenate(all_lens)
                for r in RULES:
                    o = np.concatenate(idx[r])
                    fired = o >= 0
                    early = fired & (o < taus)
                    late = fired & (o >= taus)
                    d_rec = np.median(o[late] - taus[late]) if late.any() else np.nan
                    d_life = np.median((o[late] - taus[late]) / all_lens[late]) if late.any() else np.nan
                    rows.append({"shape": shape, "delta_sd": delta, "u": u, "rule": r, "family": FAMILY[r],
                                 "n_series": len(o), "share_fire_before_change": float(early.mean()),
                                 "share_never_fire": float((~fired).mean()), "median_delay_records": d_rec,
                                 "median_delay_life_fraction": d_life,
                                 "median_abs_error_life_fraction": float(np.median(np.abs(o[fired] - taus[fired]) / all_lens[fired])) if fired.any() else np.nan})
                    log(f"    {shape:5s} {delta:5.1f} {u:4.1f} {r:9s} {early.mean():8.2f} {(~fired).mean():6.2f} "
                        f"{d_rec if np.isfinite(d_rec) else float('nan'):12.1f} {d_life if np.isfinite(d_life) else float('nan'):13.3f}")
    pd.DataFrame(rows).to_csv(T / "j24_sim_single_change.csv", index=False)

    # ---------------------------------------------------------------- S2
    log("\nS2  an early small drift plus a late large change: what makes the fire index length-independent")
    log(f"    {'early drift':22s} {'rule':9s} {'slope log i ~ log n':>20s} {'median i':>9s} {'never':>6s}")
    rows2 = []
    for early_kind, early_at in (("fixed record count", 15), ("fixed record count", 30), ("fraction of life", 0.10), ("fraction of life", 0.20)):
        for delta1 in (1.5, 2.5):
            idx = {r: [] for r in RULES}
            all_lens = []
            for n in lens:
                t1 = int(early_at) if early_kind == "fixed record count" else int(early_at * n)
                t2 = int(0.7 * n)
                mean = np.zeros(n)
                mean[t1:] = delta1
                mean[t2:] = 6.0
                H = ar1(rng, R, n, phi) + mean
                for r in RULES:
                    idx[r].append(j20.onsets(H, r))
                all_lens.append(np.full(R, n))
            all_lens = np.concatenate(all_lens)
            for r in RULES:
                o = np.concatenate(idx[r])
                b1 = slope_log(o, all_lens)
                med = float(np.median(o[o >= 0])) if (o >= 0).any() else np.nan
                rows2.append({"early_drift_placement": early_kind, "early_drift_at": early_at, "delta1_sd": delta1,
                              "delta2_sd": 6.0, "u2": 0.7, "rule": r, "family": FAMILY[r],
                              "slope_log_index_on_log_n": b1, "median_index": med, "share_never_fire": float((o < 0).mean())})
                log(f"    {early_kind + ' ' + str(early_at) + ', d1=' + str(delta1):22s} {r:9s} {b1:20.3f} {med:9.0f} {(o < 0).mean():6.2f}")
    pd.DataFrame(rows2).to_csv(T / "j24_sim_two_change.csv", index=False)

    log("\nS3  the same two-change design under the raw baseline autocorrelation (trend left in the noise)")
    phi_raw = float(np.clip(base[base.dataset == "xjtu"].phi_raw.median(), -0.99, 0.99))
    log(f"    AR(1) phi = {phi_raw:.3f}")
    rows3 = []
    for early_kind, early_at in (("fixed record count", 15), ("fraction of life", 0.10)):
        for delta1 in (1.5, 2.5):
            idx = {r: [] for r in RULES}
            all_lens = []
            for n in lens:
                t1 = int(early_at) if early_kind == "fixed record count" else int(early_at * n)
                t2 = int(0.7 * n)
                mean = np.zeros(n)
                mean[t1:] = delta1
                mean[t2:] = 6.0
                H = ar1(rng, R, n, phi_raw) + mean
                for r in RULES:
                    idx[r].append(j20.onsets(H, r))
                all_lens.append(np.full(R, n))
            all_lens = np.concatenate(all_lens)
            for r in RULES:
                o = np.concatenate(idx[r])
                b1 = slope_log(o, all_lens)
                med = float(np.median(o[o >= 0])) if (o >= 0).any() else np.nan
                rows3.append({"noise_phi": phi_raw, "early_drift_placement": early_kind, "early_drift_at": early_at,
                              "delta1_sd": delta1, "rule": r, "family": FAMILY[r], "slope_log_index_on_log_n": b1,
                              "median_index": med, "share_never_fire": float((o < 0).mean())})
                log(f"    {early_kind + ' ' + str(early_at) + ', d1=' + str(delta1):24s} {r:9s} slope {b1:+.3f}  median i {med:6.0f}  never {(o < 0).mean():.2f}")
    pd.DataFrame(rows3).to_csv(T / "j24_sim_two_change_raw_phi.csv", index=False)
    log(f"\nwrote j24_crossing_times.csv, j24_sim_single_change.csv, j24_sim_two_change.csv, j24_sim_two_change_raw_phi.csv; finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
