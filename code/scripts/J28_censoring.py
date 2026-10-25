"""E8 -- a trajectory on which a rule never fires is right-censored, not missing.

Run from bearing_multimodal_atc/:
    python scripts/J28_censoring.py

Why this exists
---------------
36 of the 330 rule-trajectory pairs never fire. Section 5.3 reports that fact and Section 6 asks a study to disclose
what it did with them, but every number in the paper is still computed over the pairs that DID fire. That is an
exclusion, and an exclusion is a choice with a direction: a rule that fails to fire is not saying "there is no
onset", it is saying "the onset is later than the end of this test". That is exactly right-censoring.

E4 showed why this matters beyond tidiness. In a simulation with no change at all, the charts still produced
rho(alarm index, length) of +0.68 and +0.51 rather than 0, because conditioning on "fired" truncates the alarm
index at n: a long test can show a late alarm, a short one cannot. Wherever the firing rate is below 100%, the
paper's headline correlation is biased upward by the same mechanism. On XJTU-SY the rules that carry the headline
numbers fire on all 15 bearings, so nothing is excluded there. On FEMTO no rule fires on all 13.

This script therefore does three things.

  A  Kaplan-Meier for the onset FRACTION, with a non-firing trajectory censored at 1.0. Gives a median onset that
     uses the non-firing trajectories as the information they are, and the size of the exclusion bias.
  B  A censoring-aware positive rate. The share of life labelled degraded is 1 - E[onset fraction], and E[onset
     fraction] under censoring is the area under the Kaplan-Meier curve on [0, 1]. This is the ORD-3 number
     computed without discarding any bearing.
  C  Sensitivity bounds on rho(alarm index, length). A censored pair satisfies index > n, so imputing index = n is
     the smallest value consistent with the censoring and gives a bound rather than a guess. If the conclusion
     survives both the exclusion and the bound, it does not rest on the treatment of non-firing.

Kaplan-Meier is implemented here rather than imported: lifelines is not a dependency of this project, and the
estimator is short enough to read.

Outputs
    results/tables/j28_km_onset.csv        KM median and restricted mean per dataset x rule, with the naive values
    results/tables/j28_rho_bounds.csv      rho excluded vs rho with censored pairs imputed at n
    results/logs/j28_run.log
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
RULES = ["3sig_rms", "ma3sig", "ma5sig", "ma2sig", "3sig_kurt", "ewma", "cusum"]
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


# ------------------------------------------------------------------ Kaplan-Meier
def km(times: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier survival for right-censored data.

    times  observed time (onset fraction for an alarm, censoring time otherwise)
    event  True if the onset was observed, False if the trajectory ended first
    returns the distinct event times and S(t) after each of them.
    """
    order = np.argsort(times, kind="mergesort")
    t, e = np.asarray(times, float)[order], np.asarray(event, bool)[order]
    n_at_risk = len(t)
    ts, S, s = [], [], 1.0
    i = 0
    while i < len(t):
        j = i
        while j < len(t) and t[j] == t[i]:
            j += 1
        d = int(e[i:j].sum())                      # events at this time
        if d:
            s *= 1.0 - d / n_at_risk
            ts.append(t[i]); S.append(s)
        n_at_risk -= (j - i)                       # everyone at this time leaves the risk set
        i = j
    return np.asarray(ts), np.asarray(S)


def km_median(ts: np.ndarray, S: np.ndarray) -> float:
    """Smallest t with S(t) <= 0.5; NaN if the curve never reaches 0.5 (more than half is censored)."""
    below = np.where(S <= 0.5)[0]
    return float(ts[below[0]]) if len(below) else np.nan


def km_rmst(ts: np.ndarray, S: np.ndarray, horizon: float = 1.0) -> float:
    """Area under the KM curve on [0, horizon] -- the restricted mean of the onset fraction."""
    if not len(ts):
        return horizon
    t = np.concatenate([[0.0], ts])
    s = np.concatenate([[1.0], S])
    keep = t <= horizon
    t, s = t[keep], s[keep]
    t = np.concatenate([t, [horizon]])
    s = np.concatenate([s, [s[-1]]])
    return float(np.sum(np.diff(t) * s[:-1]))


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j28_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J28  NON-FIRING IS RIGHT-CENSORING  (E8)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}")

    ft = pd.read_csv(T / "fig1_fpt_matrix.csv")
    ft = ft[ft.rule.isin(RULES)].copy()
    ft["fired"] = ft.fired.astype(bool)
    log(f"\n{len(ft)} rule-trajectory pairs over {ft.rule.nunique()} rules; "
        f"{int((~ft.fired).sum())} never fire")

    # ---------------- A + B
    log(f"\n{BAR}\nA/B  Kaplan-Meier for the onset fraction, non-firing censored at 1.0\n{BAR}")
    log("     naive = over the trajectories that fired, which is what the paper reports today")
    log("     KM    = using the non-firing trajectories as 'onset later than the end of the test'")
    log("     positive rate = 1 - E[onset fraction]; under censoring E[.] is the area under the KM curve")
    log("     NOTE  the KM median is the smallest t with S(t) <= 0.5. On an UNCENSORED sample of even size that is")
    log("           the lower of the two middle values, while the naive median averages them, so a row with zero")
    log("           censored trajectories can still show a median shift. Those shifts are a definition, not a bias;")
    log("           read the positive-rate columns, which agree exactly when nothing is censored.\n")
    rows = []
    for (ds, rule), g in ft.groupby(["dataset", "rule"], sort=False):
        n_traj, n_fired = len(g), int(g.fired.sum())
        if n_traj < 3:
            continue
        # observed time: the onset fraction when it fired, 1.0 (end of test) when it did not
        t = np.where(g.fired, g.fpt_fraction.to_numpy(float), 1.0)
        e = g.fired.to_numpy(bool)
        ts, S = km(t, e)
        naive_med = float(g.loc[g.fired, "fpt_fraction"].median()) if n_fired else np.nan
        naive_mean = float(g.loc[g.fired, "fpt_fraction"].mean()) if n_fired else np.nan
        rows.append({"dataset": ds, "rule": rule, "n_traj": n_traj, "n_fired": n_fired,
                     "censored": n_traj - n_fired, "censored_share": 1 - n_fired / n_traj,
                     "naive_median_fraction": naive_med, "km_median_fraction": km_median(ts, S),
                     "naive_mean_fraction": naive_mean, "km_rmst_fraction": km_rmst(ts, S),
                     "naive_pos_rate_traj": 1 - naive_mean if n_fired else np.nan,
                     "km_pos_rate": 1 - km_rmst(ts, S)})
    kmdf = pd.DataFrame(rows)
    kmdf.to_csv(T / "j28_km_onset.csv", index=False)

    for ds in ("xjtu", "femto", "ims", "nsk"):
        g = kmdf[kmdf.dataset == ds]
        if not len(g):
            continue
        log(f"  {ds}")
        log(f"    {'rule':10s} {'fired':>7s} {'cens':>5s} {'naive med':>10s} {'KM med':>8s} {'shift':>7s} "
            f"{'naive rate':>11s} {'KM rate':>8s} {'shift':>7s}")
        for _, r in g.iterrows():
            dm = r.km_median_fraction - r.naive_median_fraction
            dr = r.km_pos_rate - r.naive_pos_rate_traj
            log(f"    {r.rule:10s} {r.n_fired:3d}/{r.n_traj:<3d} {r.censored:5d} "
                f"{r.naive_median_fraction:10.3f} {r.km_median_fraction:8.3f} {dm:+7.3f} "
                f"{r.naive_pos_rate_traj:11.3f} {r.km_pos_rate:8.3f} {dr:+7.3f}")
        cen = g[g.censored > 0]
        if len(cen):
            log(f"    exclusion bias where anything is censored: median shift in the positive rate "
                f"{(cen.km_pos_rate - cen.naive_pos_rate_traj).median():+.3f} "
                f"(max {(cen.km_pos_rate - cen.naive_pos_rate_traj).abs().max():.3f})")
        log("")

    # ---------------- C
    log(f"{BAR}\nC  rho(alarm index, length): excluded, versus censored pairs held at their lower bound\n{BAR}")
    log("  a censored pair satisfies index > n, so index = n is the smallest value consistent with the data.")
    log("  the bound is not an estimate; it is the least favourable value the censoring allows.\n")
    log(f"  {'dataset':7s} {'rule':10s} {'fired':>7s} {'rho excluded':>13s} {'rho bounded':>12s} {'change':>8s}")
    brows = []
    for (ds, rule), g in ft.groupby(["dataset", "rule"], sort=False):
        if len(g) < 4:
            continue
        n = g.n_records.to_numpy(float)
        i_ex = g.fpt_index.to_numpy(float)
        ok = np.isfinite(i_ex)
        if ok.sum() < 4:
            continue
        rho_ex = float(spearmanr(n[ok], i_ex[ok])[0])
        i_bd = np.where(np.isfinite(i_ex), i_ex, n)          # censored -> its lower bound, the test length
        rho_bd = float(spearmanr(n, i_bd)[0])
        brows.append({"dataset": ds, "rule": rule, "family": g.family.iloc[0], "n_traj": len(g),
                      "n_fired": int(ok.sum()), "rho_excluded": rho_ex, "rho_bounded": rho_bd,
                      "change": rho_bd - rho_ex})
        mark = "" if int(ok.sum()) == len(g) else "   <- censored"
        log(f"  {ds:7s} {rule:10s} {int(ok.sum()):3d}/{len(g):<3d} {rho_ex:+13.3f} {rho_bd:+12.3f} "
            f"{rho_bd - rho_ex:+8.3f}{mark}")
    bd = pd.DataFrame(brows)
    bd.to_csv(T / "j28_rho_bounds.csv", index=False)

    log(f"\n{BAR}\nthe question that matters: does the family separation survive the bound?\n{BAR}")
    RMS3 = ["3sig_rms", "ma3sig", "ma5sig"]
    CH = ["ewma", "cusum"]
    log(f"  {'dataset':7s} {'':10s} {'detectors':>10s} {'charts':>9s} {'gap':>8s}")
    for ds in ("xjtu", "femto", "ims"):
        g = bd[bd.dataset == ds]
        if not len(g):
            continue
        for col, name in (("rho_excluded", "excluded"), ("rho_bounded", "bounded")):
            det = g[g.rule.isin(RMS3)][col].median()
            ch = g[g.rule.isin(CH)][col].median()
            log(f"  {ds:7s} {name:10s} {det:+10.3f} {ch:+9.3f} {det - ch:+8.3f}")
        full = g[g.n_fired == g.n_traj]
        log(f"  {'':7s} {'(complete-firing rules only: ' + str(len(full)) + ' of ' + str(len(g)) + ')':46s}")
        log("")
    log(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
