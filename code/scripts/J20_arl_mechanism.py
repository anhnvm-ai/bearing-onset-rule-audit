"""J20 -- a quantitative mechanism for the ARL signature: can a bearing that never degrades produce the same onset?

For every trajectory the healthy baseline of the RMS indicator (the records the rule itself assumes healthy) is
summarised by its mean, standard deviation and lag-1 autocorrelation. Degradation-free series of the same length are
then simulated under three null processes, and the unchanged onset rules are applied to them exactly as to the data:

  iid        independent Gaussian records, the assumption behind the textbook chart limits
  ar1_raw    Gaussian AR(1) with the lag-1 autocorrelation of the raw baseline
  ar1_detr   Gaussian AR(1) with the lag-1 autocorrelation of the linearly detrended baseline (trend removed)

For each rule and trajectory the script reports p_null = P(a degradation-free series fires at or before the observed
onset). A rule whose observed onsets are typical under a no-degradation null (p_null large) is placing its onset by
its own in-control run length; a rule whose onsets are rare under every null (p_null small) is responding to change.
It also reports the closed-form EWMA variance inflation under AR(1),
    Var(z) / Var_iid(z) = (1 + phi (1 - lambda)) / (1 - phi (1 - lambda)),
and the in-control average run length of each rule under iid data and under each bearing's AR(1) null.

The rule implementations are vectorised copies of J11.fpt and are checked against it on all 330 real onsets first.

Outputs
    results/tables/j22_arl_mechanism_per_trajectory.csv   one row per dataset x trajectory x rule x null
    results/tables/j22_arl_mechanism_summary.csv          per dataset x rule x null
    results/tables/j22_baseline_autocorrelation.csv       per trajectory: phi raw, phi detrended, trend R2, EWMA inflation
    results/tables/j22_shift_at_alarm.csv                 per trajectory x rule: standardised indicator level at the alarm
    results/logs/j20_run.log

Usage
    python scripts/J20_arl_mechanism.py            (R=2000 simulated series per trajectory and null)
    R=200 python scripts/J20_arl_mechanism.py      (quick run)
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
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
R = int(os.environ.get("R", "2000"))
SEED = 20260914
RULES = ["3sig_rms", "ma3sig", "ma5sig", "ma2sig", "ewma", "cusum"]
FAMILY = {"3sig_rms": "detector", "ma3sig": "detector", "ma5sig": "detector", "ma2sig": "detector",
          "ewma": "control chart", "cusum": "control chart"}
LAM, L, K_CUSUM, H_CUSUM, CONFIRM, BASE_FRAC = 0.2, 3.0, 0.5, 5.0, 3, 0.20
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    _fh.write(m + "\n")


def load_j11():
    spec = importlib.util.spec_from_file_location("j11", ROOT / "scripts" / "J11_arl_and_covariates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def first_confirmed(flag: np.ndarray) -> np.ndarray:
    """flag: (S, n) bool. Index of the first record of the first run of CONFIRM Trues, -1 if none."""
    S, n = flag.shape
    run = np.zeros(S, int)
    out = np.full(S, -1)
    for t in range(n):
        run = np.where(flag[:, t], run + 1, 0)
        hit = (run >= CONFIRM) & (out < 0)
        out[hit] = t - CONFIRM + 1
    return out


def onsets(H: np.ndarray, rule: str) -> np.ndarray:
    """Vectorised J11.fpt for RMS-based rules. H: (S, n)."""
    S, n = H.shape
    nb = max(5, int(n * BASE_FRAC))
    mu = H[:, :nb].mean(axis=1, keepdims=True)
    sd = np.maximum(H[:, :nb].std(axis=1, keepdims=True), 1e-12)
    if rule == "3sig_rms":
        return first_confirmed(H > mu + 3 * sd)
    if rule.startswith("ma"):
        k = float(rule[2])
        c = np.cumsum(H, axis=1)
        ma = np.empty_like(H)
        ma[:, :5] = c[:, :5] / np.arange(1, 6)
        ma[:, 5:] = (c[:, 5:] - c[:, :-5]) / 5
        return first_confirmed(ma > mu + k * sd)
    if rule == "ewma":
        z = mu[:, 0].copy()
        Z = np.empty_like(H)
        for t in range(n):
            z = LAM * H[:, t] + (1 - LAM) * z
            Z[:, t] = z
        return first_confirmed(Z > mu + L * sd * np.sqrt(LAM / (2 - LAM)))
    if rule == "cusum":
        s = np.zeros(S)
        C = np.empty_like(H)
        for t in range(n):
            s = np.maximum(0.0, s + (H[:, t] - mu[:, 0]) - K_CUSUM * sd[:, 0])
            C[:, t] = s
        return first_confirmed(C > H_CUSUM * sd)
    raise ValueError(rule)


def ar1(rng, S, n, phi, mu, sd):
    e = rng.standard_normal((S, n))
    x = np.empty((S, n))
    x[:, 0] = e[:, 0]
    a = np.sqrt(max(1 - phi ** 2, 1e-12))
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + a * e[:, t]
    return mu + sd * x


def lag1(x: np.ndarray) -> float:
    x = x - x.mean()
    d = (x * x).sum()
    return float((x[:-1] * x[1:]).sum() / d) if d > 0 else 0.0


def main() -> int:
    global _fh
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j20_run.log", "w", encoding="utf-8")
    log(f"J20 ARL mechanism, started {datetime.now():%Y-%m-%d %H:%M:%S}, R={R} per trajectory x null, seed {SEED}")
    j11 = load_j11()
    data = j11.load_traj()
    ft = pd.read_csv(TABLES / "fig1_fpt_matrix.csv")

    # ---- 1. the vectorised rules reproduce J11.fpt on every real trajectory
    bad = 0
    for ds, d in data.items():
        for t, g in d.groupby("trajectory", sort=True):
            h = g["rms"].to_numpy()
            for rule in RULES:
                ref = j11.fpt(h, g["hikurt"].to_numpy(), rule)
                got = int(onsets(h[None, :], rule)[0])
                bad += (ref is None and got != -1) or (ref is not None and got != ref)
    log(f"1. vectorised rules vs J11.fpt on {sum(d.trajectory.nunique() for d in data.values())} trajectories x {len(RULES)} rules: {bad} mismatches")
    if bad:
        return 1

    # ---- 2. iid in-control run length of each rule (long series, no truncation effects)
    rng = np.random.default_rng(SEED)
    n_long = 5000
    Hi = rng.standard_normal((R, n_long))
    log("\n2. in-control run length under iid Gaussian records (series of 5,000, baseline = first 1,000):")
    arl_iid = {}
    for rule in RULES:
        o = onsets(Hi, rule)
        fired = o >= 0
        arl_iid[rule] = (float(np.median(np.where(fired, o, n_long))), float(fired.mean()))
        log(f"    {rule:9s} median run length {arl_iid[rule][0]:7.0f} records, fires within 5,000 in {100 * arl_iid[rule][1]:5.1f}%")

    # ---- 3. per-trajectory baselines and nulls
    base_rows, rows = [], []
    for ds in ("xjtu", "femto", "ims"):
        for t, g in data[ds].groupby("trajectory", sort=True):
            h = g["rms"].to_numpy()
            n = len(h)
            nb = max(5, int(n * BASE_FRAC))
            b = h[:nb]
            x = np.arange(nb)
            coef = np.polyfit(x, b, 1)
            resid = b - np.polyval(coef, x)
            phi_raw, phi_det = lag1(b), lag1(resid)
            r2 = float(np.corrcoef(x, b)[0, 1] ** 2)
            infl = (1 + phi_raw * (1 - LAM)) / (1 - phi_raw * (1 - LAM))
            base_rows.append({"dataset": ds, "trajectory": t, "n_records": n, "baseline_records": nb, "phi_raw": phi_raw,
                              "phi_detrended": phi_det, "trend_r2": r2, "ewma_variance_inflation": infl})
            obs = ft[(ft.dataset == ds) & (ft.trajectory == t)].set_index("rule")
            nulls = {"iid": rng.standard_normal((R, n)) * b.std() + b.mean(),
                     "ar1_raw": ar1(rng, R, n, np.clip(phi_raw, -0.99, 0.99), b.mean(), b.std()),
                     "ar1_detr": ar1(rng, R, n, np.clip(phi_det, -0.99, 0.99), b.mean(), b.std())}
            for null, Hs in nulls.items():
                for rule in RULES:
                    o = onsets(Hs, rule)
                    fired = o >= 0
                    i_obs = obs.loc[rule].fpt_index
                    p = float(((o >= 0) & (o <= i_obs)).mean()) if pd.notna(i_obs) else np.nan
                    rows.append({"dataset": ds, "trajectory": t, "rule": rule, "family": FAMILY[rule], "null": null,
                                 "n_records": n, "observed_index": i_obs, "null_fire_share": float(fired.mean()),
                                 "null_median_index": float(np.median(o[fired])) if fired.any() else np.nan,
                                 "p_null_fires_by_observed": p})
    per = pd.DataFrame(rows)
    base = pd.DataFrame(base_rows)
    per.to_csv(TABLES / "j22_arl_mechanism_per_trajectory.csv", index=False)
    base.to_csv(TABLES / "j22_baseline_autocorrelation.csv", index=False)

    log("\n3. baseline of the RMS indicator (median over trajectories):")
    for ds, g in base.groupby("dataset", sort=False):
        log(f"    {ds:6s} phi raw {g.phi_raw.median():+.3f}  phi detrended {g.phi_detrended.median():+.3f}  "
            f"trend R2 {g.trend_r2.median():.3f}  EWMA variance inflation {g.ewma_variance_inflation.median():.2f}x")

    summ = []
    log("\n4. p_null = P(degradation-free series fires at or before the observed onset); trajectories where the rule fired")
    log(f"    {'dataset':6s} {'rule':9s} {'null':9s} {'n':>3s} {'median p':>9s} {'p>0.5':>6s} {'p<0.05':>7s}  "
        f"{'rho(null median, obs)':>22s}")
    for (ds, rule, null), g in per.groupby(["dataset", "rule", "null"], sort=False):
        g = g.dropna(subset=["p_null_fires_by_observed"])
        if g.empty:
            continue
        ok = g.dropna(subset=["null_median_index"])
        rho = spearmanr(ok.null_median_index, ok.observed_index).statistic if len(ok) >= 4 else np.nan
        row = {"dataset": ds, "rule": rule, "family": FAMILY[rule], "null": null, "n_traj": len(g),
               "median_p": g.p_null_fires_by_observed.median(),
               "share_p_gt_050": (g.p_null_fires_by_observed > 0.5).mean(),
               "share_p_lt_005": (g.p_null_fires_by_observed < 0.05).mean(),
               "rho_null_median_vs_observed": rho}
        summ.append(row)
        log(f"    {ds:6s} {rule:9s} {null:9s} {len(g):3d} {row['median_p']:9.3f} {row['share_p_gt_050']:6.2f} "
            f"{row['share_p_lt_005']:7.2f}  {rho:+22.3f}")
    pd.DataFrame(summ).to_csv(TABLES / "j22_arl_mechanism_summary.csv", index=False)

    # ---- 5. what each rule responds to: the indicator level over its confirming run, in baseline standard deviations,
    #         and as a share of the indicator's total rise from baseline mean to end level
    shift = []
    for ds in ("xjtu", "femto", "ims"):
        for t, g in data[ds].groupby("trajectory", sort=True):
            h = g["rms"].to_numpy()
            n = len(h)
            nb = max(5, int(n * BASE_FRAC))
            mu, sd = h[:nb].mean(), max(h[:nb].std(), 1e-12)
            end = float(np.median(h[-3:]))
            obs = ft[(ft.dataset == ds) & (ft.trajectory == t)].set_index("rule")
            for rule in RULES:
                i = obs.loc[rule].fpt_index
                if pd.isna(i):
                    continue
                w = h[int(i):int(i) + CONFIRM]
                shift.append({"dataset": ds, "trajectory": t, "rule": rule, "family": FAMILY[rule], "n_records": n,
                              "onset_index": int(i), "inside_baseline": int(i) < nb,
                              "shift_in_baseline_sd": (w.mean() - mu) / sd,
                              "share_of_total_rise": (w.mean() - mu) / (end - mu) if end > mu else np.nan})
    sh = pd.DataFrame(shift)
    sh.to_csv(TABLES / "j22_shift_at_alarm.csv", index=False)
    log("\n5. indicator level over the confirming run at the alarm (median over trajectories where the rule fires)")
    log(f"    {'dataset':6s} {'rule':9s} {'n':>3s} {'shift (baseline sd)':>20s} {'share of total rise':>20s}")
    for (ds, rule), g in sh.groupby(["dataset", "rule"], sort=False):
        log(f"    {ds:6s} {rule:9s} {len(g):3d} {g.shift_in_baseline_sd.median():20.2f} {g.share_of_total_rise.median():20.3f}")
    log("\nwrote j22_arl_mechanism_per_trajectory, j22_arl_mechanism_summary, j22_baseline_autocorrelation, j22_shift_at_alarm")
    log(f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
