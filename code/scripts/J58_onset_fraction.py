"""J58 -- onset fraction against test length, and the tables by operating condition (R2.3, R2.4).

R2.3: a comment of the review reports, answered in the response letter.
R2.4: a comment of the review reports, answered in the response letter.

The submitted Table 3 reports rho(i*, n). The alarm index is bounded by n, so rho(i*, n) is positive for any rule
that alarms at a roughly fixed fraction of life, and a rank correlation carries no effect size: pct05, whose onset
fraction moves only between 0.93 and 0.95 because of the floor in floor(n(1 - p)), still has rho(i*/n, n) near
+0.86 on XJTU-SY. The main statistic here is therefore the slope

    log u = b0 + (b1 - 1) log n + condition effects + e,          u = (i* + 1) / n,

fitted over the trajectories on which the rule fires. Its coefficient on log n is b1 - 1 of the submitted Eq. (5)
exactly, since log u = log(i* + 1) - log n: 0 means the onset FRACTION does not depend on the test length, -1 means
the alarm INDEX does not (the onset sits at a fixed record count). Spearman rho(u, n) is the rank version. rho(i*, n)
and b1 are kept only for continuity with the submitted manuscript, and are checked against it.

All numbers are on the P-retro onsets of the submitted manuscript (fig1_fpt_matrix.csv). The other protocols are
J60's.

Intervals: 95% percentile intervals over B = 20,000 bootstrap resamples of whole bearings, stratified by operating
condition (each condition resampled with replacement at its own size), seed 58. Bearings are resampled once per
dataset and the same resamples serve every rule; a rule's statistic uses the resampled bearings on which it fires.
A resample with fewer than four firing trajectories, a single trajectory length, or a rank-deficient design gives
no value; the share of valid resamples is reported. XJTU-SY and FEMTO only: IMS (four bearings) gets point values,
NSK 6205 (one trajectory) gets none.

Tables by condition (XJTU-SY 5/5/5 bearings, FEMTO 6/5/2): firing coverage, median onset fraction, median shift at
the alarm d (j22_shift_at_alarm.csv; RMS rules), onsets inside the baseline, and the positive rate of the firing
trajectories pooled over records and averaged over bearings. They are descriptive: five bearings per condition
carry no inference, and the log says so.

Pre-specified in paper/access_resub/ANALYSIS_PLAN_2026-10-24.md, Sections 6 and 13 item 2.

Inputs : results/tables/fig1_fpt_matrix.csv, results/tables/j22_shift_at_alarm.csv,
         results/tables/j23_arl_condition_adjusted.csv (consistency check only)
         scripts/J23_reviewer_analyses.py (condition_of and ols_slope, imported)
Outputs: results/tables/j58_onset_fraction.csv   one row per dataset x rule
         results/tables/j58_by_condition.csv     one row per dataset x condition x rule
         results/logs/j58_run.log
Usage  : python scripts/J58_onset_fraction.py          (B = 20,000)
         B=2000 python scripts/J58_onset_fraction.py   (quick run; not for the manuscript)
"""
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
SEED = 58
B = int(os.environ.get("B", "20000"))
MIN_FIRED = 4                          # as in J26, J36 and J41: no correlation over fewer than four trajectories
INTERVAL_DS = ("xjtu", "femto")
DS_ORDER = ["xjtu", "femto", "ims", "nsk"]
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO", "ims": "IMS", "nsk": "NSK 6205"}
RULE_ORDER = ["3sig_rms", "ma3sig", "ma5sig", "ma2sig", "3sig_kurt", "ewma", "cusum", "pct02", "pct05", "pct10"]

# rho(i*, n) of the submitted Table 3, three decimals; reproduced to 5e-4 before anything new is written
SUBMITTED_RHO = {
    "xjtu": {"3sig_rms": 0.894, "ma3sig": 0.894, "ma5sig": 0.889, "ma2sig": 0.429, "3sig_kurt": 0.304,
             "ewma": -0.201, "cusum": -0.111, "pct02": 1.0, "pct05": 1.0, "pct10": 1.0},
    "femto": {"3sig_rms": 0.817, "ma3sig": 0.783, "ma5sig": 0.883, "ma2sig": 0.283, "3sig_kurt": 0.886,
              "ewma": -0.515, "cusum": -0.610, "pct02": 1.0, "pct05": 1.0, "pct10": 1.0},
}
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


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of average ranks (what scipy.stats.spearmanr computes); NaN if undefined."""
    if len(a) < MIN_FIRED:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def stats_of(i_star: np.ndarray, n: np.ndarray, cond: np.ndarray, ols_slope) -> tuple[float, float]:
    """(b1 - 1, rho(u, n)) over one set of firing trajectories."""
    if len(n) < MIN_FIRED or np.ptp(n) == 0:
        return np.nan, np.nan
    u = (i_star + 1) / n
    return ols_slope(np.log(u), np.log(n), cond), spearman(u, n)


def ci(v: list[float]) -> tuple[float, float, float]:
    a = np.asarray([x for x in v if np.isfinite(x)])
    if len(a) < len(v) // 2:
        return np.nan, np.nan, len(a) / max(len(v), 1)
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi), len(a) / len(v)


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j58_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J58  ONSET FRACTION AGAINST TEST LENGTH, AND BY CONDITION  (R2.3, R2.4)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}   B = {B:,}   seed = {SEED}")
    if B < 20000:
        log("NOTE: B below 20,000 -- a quick run; the manuscript numbers need the default")

    j23 = load("j23", "J23_reviewer_analyses.py")
    ft = pd.read_csv(T / "fig1_fpt_matrix.csv")
    ft["fired"] = ft.fired.astype(bool)
    assert len(ft) == 330, f"expected the 330 onsets of the submitted manuscript, found {len(ft)}"
    sh = pd.read_csv(T / "j22_shift_at_alarm.csv")[["dataset", "trajectory", "rule", "shift_in_baseline_sd"]]
    ft = ft.merge(sh, on=["dataset", "trajectory", "rule"], how="left")
    ft["cond"] = [j23.condition_of(d, t) for d, t in zip(ft.dataset, ft.trajectory)]
    rng = np.random.default_rng(SEED)

    rows, by_cond = [], []
    for ds in DS_ORDER:
        d0 = ft[ft.dataset == ds]
        trajs = sorted(d0.trajectory.unique())
        meta = d0.drop_duplicates("trajectory").set_index("trajectory").loc[trajs]
        n_all, c_all = meta.n_records.to_numpy(float), meta.cond.to_numpy()
        # one set of bearing resamples per dataset, stratified by condition, shared by every rule
        idx = None
        if ds in INTERVAL_DS:
            levels = sorted(set(c_all))
            parts = []
            for k in levels:
                ix = np.where(c_all == k)[0]
                parts.append(ix[rng.integers(0, len(ix), size=(B, len(ix)))])
            idx = np.concatenate(parts, axis=1)
        for rule in RULE_ORDER:
            g = d0[d0.rule == rule].set_index("trajectory").loc[trajs]
            fired = g.fired.to_numpy()
            i_all = g.fpt_index.to_numpy(float)
            f = fired
            u = (i_all[f] + 1) / n_all[f]
            r = {"dataset": ds, "rule": rule, "family": g.family.iloc[0], "n_traj": len(trajs),
                 "n_fired": int(f.sum()), "coverage": float(f.mean()),
                 "median_u": float(np.median(u)) if f.any() else np.nan,
                 "min_u": float(u.min()) if f.any() else np.nan, "max_u": float(u.max()) if f.any() else np.nan}
            cond_f = c_all[f] if ds in INTERVAL_DS else np.array(["all"] * int(f.sum()))
            slope, rho_u = stats_of(i_all[f], n_all[f], cond_f, j23.ols_slope)
            r.update(slope_b1_minus_1=slope, b1_eq5=slope + 1 if np.isfinite(slope) else np.nan,
                     rho_u_n=rho_u, rho_istar_n=spearman(i_all[f], n_all[f]),
                     condition_effects=ds in INTERVAL_DS)
            if idx is not None:
                sl, rh = [], []
                for b in range(B):
                    sel = idx[b]
                    keep = fired[sel]
                    s2, r2 = stats_of(i_all[sel][keep], n_all[sel][keep], c_all[sel][keep], j23.ols_slope)
                    sl.append(s2); rh.append(r2)
                r["slope_lo"], r["slope_hi"], r["slope_valid_share"] = ci(sl)
                r["rho_u_lo"], r["rho_u_hi"], r["rho_u_valid_share"] = ci(rh)
            rows.append(r)

            if ds in INTERVAL_DS:                                 # the descriptive table by condition
                for k in sorted(set(c_all)):
                    m = c_all == k
                    gk = g[m]
                    fk = gk[gk.fired]
                    nk = fk.n_records.to_numpy(float)
                    pos = nk - fk.fpt_index.to_numpy(float)
                    by_cond.append({
                        "dataset": ds, "condition": k, "rule": rule, "family": g.family.iloc[0],
                        "n_traj": int(m.sum()), "n_fired": len(fk), "coverage": len(fk) / int(m.sum()),
                        "length_min": int(gk.n_records.min()), "length_max": int(gk.n_records.max()),
                        "records": int(gk.n_records.sum()),
                        "median_u": float(np.median((fk.fpt_index + 1) / fk.n_records)) if len(fk) else np.nan,
                        "median_shift_d": (float(fk.shift_in_baseline_sd.dropna().median())
                                           if fk.shift_in_baseline_sd.notna().any() else np.nan),
                        "inside_baseline": int(fk.inside_baseline.astype(bool).sum()),
                        "pos_rate_record": float(pos.sum() / nk.sum()) if len(fk) else np.nan,
                        "pos_rate_bearing": float(np.mean(pos / nk)) if len(fk) else np.nan})

    out = pd.DataFrame(rows)
    cond = pd.DataFrame(by_cond)

    # ---------------------------------------------------------------- consistency with the submitted manuscript
    log(f"\n{BAR}\nA  Consistency with the submitted manuscript (checked before anything is written)\n{BAR}")
    bad = 0
    for ds, exp in SUBMITTED_RHO.items():
        for rule, v in exp.items():
            got = float(out[(out.dataset == ds) & (out.rule == rule)].rho_istar_n.iloc[0])
            if not abs(got - v) <= 5e-4:
                bad += 1
                log(f"  MISMATCH rho(i*, n) {ds} {rule}: submitted {v:+.3f}, recomputed {got:+.4f}")
    log(f"  rho(i*, n) of the submitted Table 3: {20 - bad} of 20 values reproduced to 5e-4")
    f23 = T / "j23_arl_condition_adjusted.csv"
    if f23.exists():
        a = pd.read_csv(f23).merge(out, on=["dataset", "rule"])
        dev = float((a.b1_eq5 - a.slope_log_n_adjusted).abs().max())
        log(f"  b1 of Eq. (5) against J23 ({len(a)} dataset-rule cells): largest difference {dev:.2e}")
        if not dev < 1e-9:
            bad += 1
            log("  MISMATCH with J23")
    if bad:
        log("  FAIL: the recomputation does not reproduce the submitted numbers; nothing written")
        return 1
    log("  PASS")

    # ---------------------------------------------------------------- the main table
    log(f"\n{BAR}\nB  Onset fraction u = (i*+1)/n against test length (P-retro onsets)\n{BAR}")
    log("  b1 - 1: slope of log u on log n with condition effects (0: fraction independent of n; -1: index independent)")
    for ds in DS_ORDER:
        o = out[out.dataset == ds]
        log(f"\n  {DS_NAME[ds]}")
        log(f"    {'rule':10s} {'fired':>6s} {'median u':>9s}   {'b1 - 1':>7s} {'95% interval':>18s}   "
            f"{'rho(u,n)':>8s} {'95% interval':>18s}   {'rho(i*,n)':>9s}")
        for _, r in o.iterrows():
            def fmt(v, lo=None, hi=None):
                s = f"{v:+7.3f}" if np.isfinite(v) else "    n/a"
                if lo is not None:
                    s += (f"  [{lo:+.2f}, {hi:+.2f}]" if np.isfinite(lo) else " " * 16)
                return s
            lo_s, hi_s = r.get("slope_lo", np.nan), r.get("slope_hi", np.nan)
            lo_r, hi_r = r.get("rho_u_lo", np.nan), r.get("rho_u_hi", np.nan)
            log(f"    {r.rule:10s} {r.n_fired:>2d}/{r.n_traj:<3d} {r.median_u:9.3f}   "
                f"{fmt(r.slope_b1_minus_1, lo_s, hi_s):>26s}   {fmt(r.rho_u_n, lo_r, hi_r):>26s}   "
                f"{fmt(r.rho_istar_n):>9s}")
    if "slope_valid_share" in out:
        vs = out.slope_valid_share.dropna()
        log(f"\n  valid bootstrap share, b1 - 1: min {vs.min():.3f}; rho(u, n): min {out.rho_u_valid_share.dropna().min():.3f}")
    pc = out[(out.dataset == "xjtu") & out.rule.str.startswith("pct")]
    log("\n  NOTE on the percentile rules. Their onset fraction is fixed by construction, so b1 - 1 is ~0 "
        f"(XJTU-SY: {pc.slope_b1_minus_1.min():+.3f} to {pc.slope_b1_minus_1.max():+.3f}). Their rank correlation is")
    log("  not: the floor in floor(n(1-p)) makes u = (i*+1)/n FALL toward 1-p as n grows (here rho(u, n) from "
        f"{pc.rho_u_n.min():+.2f} to {pc.rho_u_n.max():+.2f}),")
    log("  while i*/n RISES toward 1-p (rho(i*/n, n) about +0.86 for pct05). A rank statistic follows a discretisation")
    log("  artefact of a few thousandths of life in either direction; the slope does not. This is why b1 - 1 is the")
    log("  main statistic of the revision (analysis plan, Section 6).")

    # ---------------------------------------------------------------- by condition
    log(f"\n{BAR}\nC  By operating condition (descriptive: 5/5/5 and 6/5/2 bearings carry no inference)\n{BAR}")
    for ds in INTERVAL_DS:
        c = cond[cond.dataset == ds]
        log(f"\n  {DS_NAME[ds]}")
        for k in sorted(c.condition.unique()):
            ck = c[c.condition == k]
            r0 = ck.iloc[0]
            log(f"    condition {k}: {r0.n_traj} bearings, lengths {r0.length_min}-{r0.length_max}, "
                f"{r0.records} records")
            for _, r in ck[ck.rule.isin(["3sig_rms", "ma5sig", "3sig_kurt", "ewma", "cusum", "pct05"])].iterrows():
                d_s = f"{r.median_shift_d:5.2f}" if np.isfinite(r.median_shift_d) else "  n/a"
                log(f"      {r.rule:10s} fired {r.n_fired}/{r.n_traj}  median u {r.median_u:5.3f}  d {d_s}  "
                    f"inside {r.inside_baseline}  pos rate rec {r.pos_rate_record:5.3f} brg {r.pos_rate_bearing:5.3f}")
    xr = cond[cond.dataset == "xjtu"].drop_duplicates("condition")
    log(f"\n  XJTU-SY records by condition: " + ", ".join(f"{k}: {v}" for k, v in zip(xr.condition, xr.records)) +
        f" (condition 3 holds {xr.records.iloc[-1] / xr.records.sum():.1%} of the records)")

    out.to_csv(T / "j58_onset_fraction.csv", index=False)
    cond.to_csv(T / "j58_by_condition.csv", index=False)
    log(f"\nwrote results/tables/j58_onset_fraction.csv ({len(out)} rows), j58_by_condition.csv ({len(cond)} rows)")
    log(f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
