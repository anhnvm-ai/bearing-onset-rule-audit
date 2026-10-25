"""J57 -- non-firing trajectories: weighting x treatment, in place of the Kaplan-Meier paragraph (R1.5).

Reviewer 1, R1.5: a comment of the review reports, answered in the response letter.

Section V-B of the submitted manuscript compares two numbers that differ in two ways at once:
  * the positive rate of Eq. (2), pooled over the RECORDS of the trajectories on which a rule fires, and
  * the Kaplan-Meier "share of life labeled degraded" of J28, an average over BEARINGS that keeps the
    non-firing trajectories in the denominator.
Their difference therefore mixes a change of weighting (records -> bearings) with a change of treatment
(exclude the non-firing trajectory -> keep it, never degraded). This script separates the two, per dataset x
rule:

                         exclude non-firing                 label non-firing healthy
    record-pooled        sum_f (n - i*) / sum_f n           sum_f (n - i*) / sum_all n
    bearing-averaged     mean_f (n - i*)/n                  sum_f (n - i*)/n / N_all

(f = trajectories on which the rule fires, all = every trajectory of the dataset.)

It also checks the identity behind the Kaplan-Meier number. J28 censors every non-firing trajectory at u = 1.
An alarm needs c = 3 confirming records, so i* <= n - 3 and every observed onset fraction is below 1: all
censoring falls after the last event. The Kaplan-Meier estimator on [0, 1] is then the empirical survival
function in which a censored trajectory counts as 'not yet fired', its restricted mean on [0, 1] is
(1/N) [sum_f U_j + N_censored], and 1 - RMST = (1/N) sum_f (1 - U_j) -- the bearing-averaged positive rate with
non-firing trajectories labeled healthy. The script asserts this to 1e-9 on every cell J28 reports. For this
quantity Kaplan-Meier is therefore not a third treatment.

The prevalence-screen verdict of the submitted manuscript ([0.01, 0.50]) is reported for every cell of the 2 x 2,
because Section V-B states that labeling the non-firing trajectories healthy turns a failing label set into a
passing one.

Pre-specified in paper/access_resub/ANALYSIS_PLAN_2026-10-24.md, Section 13 item 1. Nothing is refitted.

Inputs : results/tables/fig1_fpt_matrix.csv   the 330 onsets of the submitted manuscript
         results/tables/j28_km_onset.csv      the Kaplan-Meier values of the submitted manuscript
Outputs: results/tables/j57_censoring_2x2.csv  one row per dataset x rule
         results/logs/j57_run.log
Usage  : python scripts/J57_censoring_decomposition.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
TOL = 1e-9                         # Section 13 item 1 of the analysis plan
SCREEN = (0.01, 0.50)              # the prevalence screen of the submitted Section III-C
DS_ORDER = ["xjtu", "femto", "ims", "nsk"]
RULE_ORDER = ["3sig_rms", "ma3sig", "ma5sig", "ma2sig", "3sig_kurt", "ewma", "cusum", "pct02", "pct05", "pct10"]
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


def cell(g: pd.DataFrame) -> dict:
    """The 2 x 2 for the trajectories of one dataset under one rule."""
    f = g[g.fired]
    n_all, n_f = g.n_records.to_numpy(float), f.n_records.to_numpy(float)
    pos_f = n_f - f.fpt_index.to_numpy(float)                 # records i*, ..., n-1 are labeled degraded
    share_f = pos_f / n_f
    out = {"n_traj": len(g), "n_fired": len(f), "n_nonfiring": len(g) - len(f), "coverage": len(f) / len(g),
           "records_all": int(n_all.sum()), "records_fired": int(n_f.sum())}
    if len(f):
        out.update(rate_record_exclude=pos_f.sum() / n_f.sum(),
                   rate_record_healthy=pos_f.sum() / n_all.sum(),
                   rate_bearing_exclude=share_f.mean(),
                   rate_bearing_healthy=share_f.sum() / len(g))
    else:
        out.update(rate_record_exclude=np.nan, rate_record_healthy=0.0,
                   rate_bearing_exclude=np.nan, rate_bearing_healthy=0.0)
    return out


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j57_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J57  NON-FIRING TRAJECTORIES: WEIGHTING x TREATMENT  (R1.5)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}")

    ft = pd.read_csv(T / "fig1_fpt_matrix.csv")
    ft["fired"] = ft.fired.astype(bool)
    km = pd.read_csv(T / "j28_km_onset.csv")
    assert len(ft) == 330, f"expected the 330 onsets of the submitted manuscript, found {len(ft)}"
    log(f"\n{len(ft)} rule-trajectory pairs; {int((~ft.fired).sum())} never fire")

    rows = []
    for ds in DS_ORDER:
        for rule in RULE_ORDER:
            g = ft[(ft.dataset == ds) & (ft.rule == rule)]
            if not len(g):
                continue
            r = {"dataset": ds, "rule": rule, "family": g.family.iloc[0], **cell(g)}
            k = km[(km.dataset == ds) & (km.rule == rule)]
            r["km_j28"] = float(k.km_pos_rate.iloc[0]) if len(k) else np.nan
            rows.append(r)
    out = pd.DataFrame(rows)

    # the four contrasts: each changes ONE thing
    out["effect_treatment_record"] = out.rate_record_exclude - out.rate_record_healthy
    out["effect_treatment_bearing"] = out.rate_bearing_exclude - out.rate_bearing_healthy
    out["effect_weighting_exclude"] = out.rate_record_exclude - out.rate_bearing_exclude
    out["effect_weighting_healthy"] = out.rate_record_healthy - out.rate_bearing_healthy
    out["km_minus_bearing_healthy"] = out.km_j28 - out.rate_bearing_healthy
    for c in ("record_exclude", "record_healthy", "bearing_exclude", "bearing_healthy"):
        v = out[f"rate_{c}"]
        out[f"screen_{c}"] = np.where(v.isna(), None, (v >= SCREEN[0]) & (v <= SCREEN[1]))

    # ---------------------------------------------------------------- the identity
    log(f"\n{BAR}\nA  The Kaplan-Meier value of J28 against the bearing-averaged, labeled-healthy rate\n{BAR}")
    log("  J28 censors a non-firing trajectory at u = 1. An alarm needs c = 3 confirming records, so i* <= n - 3")
    log("  and every observed onset fraction U = i*/n is below 1: all censoring falls after the last event.")
    log("  Kaplan-Meier on [0, 1] then reduces to the empirical survival function with the censored trajectories")
    log("  counted as 'not yet fired':  S(u) = #{j : U_j > u} / N  for u < 1.")
    log("  Its restricted mean on [0, 1] is  (1/N) [ sum_fired U_j + N_censored ],  so")
    log("      1 - RMST = (1/N) sum_fired (1 - U_j),")
    log("  which is the bearing-averaged positive rate with non-firing trajectories labeled healthy.")
    fired = ft[ft.fired]
    max_u = float((fired.fpt_index / fired.n_records).max())
    log(f"  largest observed onset fraction over the 294 firing pairs: {max_u:.4f} (< 1, as the argument needs)")
    chk = out[out.km_j28.notna()]
    worst = float(chk.km_minus_bearing_healthy.abs().max())
    log(f"  cells checked (J28 reports datasets with >= 3 trajectories): {len(chk)}; "
        f"largest |KM - bearing-averaged healthy| = {worst:.2e}")
    if not worst < TOL:
        log(f"  FAIL: the identity does not hold to {TOL:g}")
        return 1
    log(f"  PASS: equal to {TOL:g} in every cell. Kaplan-Meier is not a third treatment for this quantity.")
    if max_u >= 1:
        log("  FAIL: an observed onset fraction reaches 1; the argument above does not apply")
        return 1

    # ---------------------------------------------------------------- the 2 x 2
    log(f"\n{BAR}\nB  The 2 x 2 for every dataset x rule with a non-firing trajectory\n{BAR}")
    log("  rate_* = share of records labeled degraded;  rec = pooled over records (Eq. 2), brg = mean over bearings")
    log("  excl = non-firing trajectories excluded (submitted convention), heal = labeled healthy to the end")
    cen = out[out.n_nonfiring > 0]
    hdr = (f"  {'dataset':7s} {'rule':10s} {'fired':>6s}  {'rec/excl':>8s} {'rec/heal':>8s} "
           f"{'brg/excl':>8s} {'brg/heal':>8s}   {'KM(J28)':>7s}")
    log(hdr)
    for _, r in cen.iterrows():
        km_s = f"{r.km_j28:7.3f}" if np.isfinite(r.km_j28) else "    n/a"
        log(f"  {r.dataset:7s} {r.rule:10s} {r.n_fired:>2d}/{r.n_traj:<3d}  {r.rate_record_exclude:8.3f} "
            f"{r.rate_record_healthy:8.3f} {r.rate_bearing_exclude:8.3f} {r.rate_bearing_healthy:8.3f}   {km_s}")

    # ---------------------------------------------------------------- what moves what
    log(f"\n{BAR}\nC  What each change does, median over the cells with a non-firing trajectory\n{BAR}")
    log("  treatment = exclude minus healthy (weighting fixed);  weighting = record minus bearing (treatment fixed)")
    summ = []
    for ds in DS_ORDER:
        c = cen[(cen.dataset == ds) & (cen.n_fired > 0)]      # a rule that never fires has no 'exclude' cell
        if not len(c):
            if len(cen[cen.dataset == ds]):
                log(f"  {ds:6s} only rules that fire on no trajectory: no contrast to compute")
            continue
        s = {"dataset": ds, "cells": len(c)}
        for k in ("effect_treatment_record", "effect_treatment_bearing",
                  "effect_weighting_exclude", "effect_weighting_healthy"):
            s[k] = float(c[k].median())
        summ.append(s)
        log(f"  {ds:6s} ({len(c)} cells)  treatment: record {s['effect_treatment_record']:+.3f}, "
            f"bearing {s['effect_treatment_bearing']:+.3f}   weighting: exclude {s['effect_weighting_exclude']:+.3f}, "
            f"healthy {s['effect_weighting_healthy']:+.3f}")
    log("\n  The submitted Section V-B reported a fall of 0.102 (XJTU-SY) and 0.094 (FEMTO) 'when non-firing is")
    log("  treated as censoring'. By the identity of part A that number is the TREATMENT effect under BEARING")
    log("  weighting (column 'treatment: bearing' above), not a property of Kaplan-Meier.")

    # ---------------------------------------------------------------- screen verdicts
    log(f"\n{BAR}\nD  Prevalence screen [{SCREEN[0]:.2f}, {SCREEN[1]:.2f}] under each cell of the 2 x 2\n{BAR}")
    flips = 0
    for _, r in cen.iterrows():
        v = [r[f"screen_{c}"] for c in ("record_exclude", "record_healthy", "bearing_exclude", "bearing_healthy")]
        if len({x for x in v if x is not None}) > 1:
            flips += 1
            lab = ["pass" if x else ("fail" if x is not None else "n/a") for x in v]
            log(f"  {r.dataset:6s} {r.rule:10s} rec/excl {lab[0]}, rec/heal {lab[1]}, "
                f"brg/excl {lab[2]}, brg/heal {lab[3]}")
    log(f"  cells whose verdict depends on the weighting or the treatment: {flips} of {len(cen)}")

    out.to_csv(T / "j57_censoring_2x2.csv", index=False)
    log(f"\nwrote results/tables/j57_censoring_2x2.csv ({len(out)} rows)")
    log(f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
