"""J23 -- analyses requested by the two reviewers (2026-09-15), on the real trajectories.

  A  condition-adjusted ARL signature: log(i + 1) = b0 + b1 log n + condition effects, per rule, on XJTU-SY and FEMTO,
     with a condition-stratified bearing bootstrap for b1; within-condition Spearman rho(i, n); and a permutation test
     of the detector-minus-chart slope gap that shuffles trajectory lengths only within operating conditions.
  C  a retrospective changepoint reference: binary segmentation for changes in the mean of RMS (Gaussian cost,
     BIC-type penalty), onset = earliest upward change; its rho(i, n), inside-baseline count and shift at the change.
  D  second health indicator: the ARL signature and the shift at the alarm recomputed with peak-to-peak instead of RMS.
  E  sensitivity of the usability screen: verdicts of the C2 statements over p_min x p_max.
  F  label disagreement between rules: record-level Jaccard index of the degraded sets and Cohen's kappa over the
     trajectories on which both rules fire, and the per-trajectory spread of onset fractions across rules.
  G  rank stability of methods across labelling rules: Kendall tau between the method rankings (macro-F1) under
     each pair of usable rules, per dataset, and how often the winner changes.

Outputs (results/tables): j23_arl_condition_adjusted.csv, j23_arl_within_condition.csv, j23_arl_permutation.csv,
j23_changepoint_reference.csv, j23_second_indicator.csv, j23_usability_grid.csv, j23_label_agreement.csv,
j23_onset_spread.csv, j23_rank_stability.csv; log results/logs/j23_run.log.
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
B = 2000
SEED = 20260915
RMS3 = ["3sig_rms", "ma3sig", "ma5sig"]
CHARTS = ["ewma", "cusum"]
_fh = None


def log(m=""):
    print(m, flush=True)
    _fh.write(m + "\n")


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def condition_of(ds, traj):
    return traj.split("_")[0].replace("Bearing", "") if ds in ("xjtu", "femto") else "all"


def ols_slope(y, x, cond):
    """b1 of y ~ x + condition dummies (one dummy per condition beyond the first)."""
    levels = sorted(set(cond))
    X = [np.ones(len(x)), x] + [(np.asarray(cond) == c).astype(float) for c in levels[1:]]
    X = np.column_stack(X)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return np.nan
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1])


# ------------------------------------------------------------------ A
def part_a(ft, rng):
    log("A. condition-adjusted ARL signature")
    rows, within, perm_rows = [], [], []
    for ds in ("xjtu", "femto"):
        g0 = ft[(ft.dataset == ds) & ft.fired].copy()
        g0["cond"] = [condition_of(ds, t) for t in g0.trajectory]
        slopes = {}
        for rule in RMS3 + ["ma2sig", "3sig_kurt"] + CHARTS:
            g = g0[g0.rule == rule]
            y, x, c = np.log(g.fpt_index.to_numpy() + 1), np.log(g.n_records.to_numpy()), g.cond.to_numpy()
            b1 = ols_slope(y, x, c)
            b1_raw = ols_slope(y, x, np.array(["all"] * len(x)))
            boots = []
            for _ in range(B):
                idx = np.concatenate([rng.choice(np.where(c == k)[0], (c == k).sum(), replace=True) for k in sorted(set(c))])
                boots.append(ols_slope(y[idx], x[idx], c[idx]))
            boots = np.array([b for b in boots if np.isfinite(b)])
            lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) > B // 2 else (np.nan, np.nan))
            slopes[rule] = b1
            rows.append({"dataset": ds, "rule": rule, "n_traj": len(g), "n_conditions": len(set(c)),
                         "slope_log_n_adjusted": b1, "ci_lo": lo, "ci_hi": hi, "slope_log_n_unadjusted": b1_raw,
                         "valid_bootstrap_share": len(boots) / B})
            log(f"    {ds:6s} {rule:9s} n={len(g):2d}  b1 adjusted {b1:+.3f} [{lo:+.2f}, {hi:+.2f}]  unadjusted {b1_raw:+.3f}")
            for k in sorted(set(c)):
                m = c == k
                if m.sum() >= 3 and np.ptp(x[m]) > 0 and np.ptp(y[m]) > 0:
                    within.append({"dataset": ds, "rule": rule, "condition": k, "n_traj": int(m.sum()),
                                   "spearman_rho": spearmanr(x[m], y[m]).statistic})
        # permutation: shuffle lengths within condition, keep each bearing's onsets for all rules together
        traj = g0.drop_duplicates("trajectory")[["trajectory", "cond", "n_records"]].reset_index(drop=True)
        obs_gap = np.median([slopes[r] for r in RMS3]) - np.median([slopes[r] for r in CHARTS])
        null = []
        for _ in range(B):
            perm = traj.copy()
            for k in perm.cond.unique():
                m = perm.cond == k
                perm.loc[m, "n_records"] = rng.permutation(perm.loc[m, "n_records"].to_numpy())
            nmap = dict(zip(perm.trajectory, perm.n_records))
            s = {}
            for rule in RMS3 + CHARTS:
                g = g0[g0.rule == rule]
                s[rule] = ols_slope(np.log(g.fpt_index.to_numpy() + 1), np.log(g.trajectory.map(nmap).to_numpy().astype(float)), g.cond.to_numpy())
            null.append(np.median([s[r] for r in RMS3]) - np.median([s[r] for r in CHARTS]))
        null = np.array(null)
        p = float((np.sum(null >= obs_gap) + 1) / (len(null) + 1))
        perm_rows.append({"dataset": ds, "observed_gap": obs_gap, "null_median": float(np.median(null)),
                          "null_q95": float(np.percentile(null, 95)), "p_one_sided": p, "B": B})
        log(f"    {ds:6s} detector-minus-chart slope gap {obs_gap:+.3f}; within-condition permutation p = {p:.4f}")
    pd.DataFrame(rows).to_csv(T / "j23_arl_condition_adjusted.csv", index=False)
    pd.DataFrame(within).to_csv(T / "j23_arl_within_condition.csv", index=False)
    pd.DataFrame(perm_rows).to_csv(T / "j23_arl_permutation.csv", index=False)
    w = pd.DataFrame(within)
    for ds in ("xjtu", "femto"):
        q = w[w.dataset == ds]
        log(f"    {ds:6s} within-condition rho, median over conditions: " +
            ", ".join(f"{r} {q[q.rule == r].spearman_rho.median():+.2f} ({len(q[q.rule == r])} cond.)" for r in RMS3 + CHARTS))


# ------------------------------------------------------------------ C
def binseg_first_up(h, nb, min_size=5):
    """Binary segmentation for mean changes (Gaussian cost with pooled variance, BIC penalty); earliest upward change."""
    n = len(h)
    var = np.var(np.diff(h)) / 2 if n > 2 else np.var(h)
    var = max(var, 1e-12)
    pen = 2 * np.log(n) * var

    def best_split(a, b):
        seg = h[a:b]
        m = len(seg)
        if m < 2 * min_size:
            return None
        c = np.cumsum(seg)
        c2 = np.cumsum(seg ** 2)
        tot = c2[-1] - c[-1] ** 2 / m
        k = np.arange(min_size, m - min_size + 1)
        left = c2[k - 1] - c[k - 1] ** 2 / k
        right = (c2[-1] - c2[k - 1]) - (c[-1] - c[k - 1]) ** 2 / (m - k)
        gain = tot - (left + right)
        j = int(np.argmax(gain))
        return (a + int(k[j]), float(gain[j]))

    cps, stack = [], [(0, n)]
    while stack:
        a, b = stack.pop()
        s = best_split(a, b)
        if s and s[1] > pen:
            cps.append(s[0])
            stack += [(a, s[0]), (s[0], b)]
    cps = sorted(cps)
    bounds = [0] + cps + [n]
    for i, cp in enumerate(cps):
        if h[cp:bounds[i + 2]].mean() > h[bounds[i]:cp].mean():
            return cp
    return None


def part_c(data, ft):
    log("\nC. retrospective changepoint reference (binary segmentation on RMS)")
    rows = []
    for ds in ("xjtu", "femto", "ims"):
        for t, g in data[ds].groupby("trajectory", sort=True):
            h = g["rms"].to_numpy()
            n = len(h)
            nb = max(5, int(0.2 * n))
            cp = binseg_first_up(h, nb)
            mu, sd = h[:nb].mean(), max(h[:nb].std(), 1e-12)
            rows.append({"dataset": ds, "trajectory": t, "n_records": n, "fired": cp is not None,
                         "onset_index": cp, "onset_fraction": cp / n if cp is not None else np.nan,
                         "inside_baseline": cp is not None and cp < nb,
                         "shift_in_baseline_sd": (h[cp:cp + 3].mean() - mu) / sd if cp is not None else np.nan})
    cpd = pd.DataFrame(rows)
    cpd.to_csv(T / "j23_changepoint_reference.csv", index=False)
    for ds in ("xjtu", "femto"):
        q = cpd[(cpd.dataset == ds) & cpd.fired]
        rho = spearmanr(q.n_records, q.onset_index).statistic
        log(f"    {ds:6s} fires on {len(q)}/{int((cpd.dataset == ds).sum())}, rho(i, n) {rho:+.3f}, inside baseline {int(q.inside_baseline.sum())}, "
            f"median shift {q.shift_in_baseline_sd.median():.2f} sd, median onset fraction {q.onset_fraction.median():.3f}")


# ------------------------------------------------------------------ D
def p2p_series():
    out = {}
    for ds in ("xjtu", "ims"):
        rec = pd.read_csv(ROOT / "colab_bundle" / ds / "records.csv")
        f = pd.read_csv(T / f"j7_features_{ds}.csv")
        d = pd.DataFrame({"trajectory": rec.trajectory_id, "record_index": rec.record_index, "p2p": f["p2p"]})
        if ds == "ims":
            d = d[d.trajectory.isin({"IMS_1st_test_B3", "IMS_1st_test_B4", "IMS_2nd_test_B1", "IMS_3rd_test_B3"})]
        out[ds] = d.sort_values(["trajectory", "record_index"])
    fm = pd.read_csv(T / "j7c_femto_features.csv")
    out["femto"] = pd.DataFrame({"trajectory": fm.bearing, "record_index": fm.record_index, "p2p": fm.v_p2p}).sort_values(["trajectory", "record_index"])
    return out


def part_d(j20, j11, rng):
    log("\nD. second health indicator: peak-to-peak")
    rows = []
    for ds, d in p2p_series().items():
        for t, g in d.groupby("trajectory", sort=True):
            h = g["p2p"].to_numpy(float)
            n = len(h)
            nb = max(5, int(0.2 * n))
            mu, sd = h[:nb].mean(), max(h[:nb].std(), 1e-12)
            for rule in RMS3 + ["ma2sig"] + CHARTS:
                i = int(j20.onsets(h[None, :], rule)[0])
                rows.append({"dataset": ds, "trajectory": t, "rule": rule, "n_records": n, "fired": i >= 0,
                             "onset_index": i if i >= 0 else np.nan, "inside_baseline": 0 <= i < nb,
                             "shift_in_baseline_sd": (h[i:i + 3].mean() - mu) / sd if i >= 0 else np.nan})
    p = pd.DataFrame(rows)
    p.to_csv(T / "j23_second_indicator.csv", index=False)
    summ = []
    for ds in ("xjtu", "femto"):
        for rule in RMS3 + ["ma2sig"] + CHARTS:
            q = p[(p.dataset == ds) & (p.rule == rule) & p.fired]
            rho, pval, lo, hi, nn = j11.boot_spearman(q.n_records.to_numpy(float), q.onset_index.to_numpy(float), rng, 20000)
            summ.append({"dataset": ds, "rule": rule, "n_fired": len(q), "n_traj": int(((p.dataset == ds) & (p.rule == rule)).sum()),
                         "rho": rho, "ci_lo": lo, "ci_hi": hi, "inside_baseline": int(q.inside_baseline.sum()),
                         "median_shift_sd": q.shift_in_baseline_sd.median()})
            log(f"    {ds:6s} {rule:9s} fires {len(q):2d}  rho {rho:+.3f} [{lo:+.2f}, {hi:+.2f}]  inside {int(q.inside_baseline.sum())}  shift {q.shift_in_baseline_sd.median():.2f} sd")
    pd.DataFrame(summ).to_csv(T / "j23_second_indicator_summary.csv", index=False)


# ------------------------------------------------------------------ E
def part_e():
    log("\nE. usability screen sensitivity")
    sub = pd.read_csv(T / "j17_prevalence_by_subset.csv")
    full = sub[sub.subset == "all"].set_index(["dataset", "rule"]).pos_rate
    xc = sub[(sub.dataset == "xjtu") & (sub.rule == "3sig_rms") & sub.subset.str.startswith("cond")].set_index("subset").pos_rate
    # revision (C4). The upper bound is swept finely, not over four coarse values, because the honest
    # answer to "why 0.50?" is not "the conclusions never change" -- they do change, and where they change is
    # itself the argument. A screen that admits a label set calling two thirds of every bearing's life degraded
    # is not screening for a partition into health states, and the flip point locates that.
    SIGNAL_RULES = ("3sig_rms", "ma3sig", "ma5sig", "ma2sig", "3sig_kurt", "ewma", "cusum")
    PMINS = (0.005, 0.01, 0.02, 0.05)
    PMAXS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    rows = []
    for pmin in PMINS:
        for pmax in PMAXS:
            use = (full >= pmin) & (full <= pmax)
            flips = [r for r in SIGNAL_RULES if use.xs(r, level="rule").nunique() > 1]
            rows.append({"p_min": pmin, "p_max": pmax,
                         "rules_changing_verdict_across_datasets": len(flips),
                         "rules_changing_verdict": ",".join(flips),
                         "3sig_rms_fails_on": ",".join(sorted(use.xs("3sig_rms", level="rule")[lambda s: ~s].index)),
                         "charts_fail_on_all_datasets": bool((~use.xs("ewma", level="rule")).all() and (~use.xs("cusum", level="rule")).all()),
                         "xjtu_condition_verdict_differs": bool(((xc >= pmin) & (xc <= pmax)).nunique() > 1)})
    e = pd.DataFrame(rows)
    e.to_csv(T / "j23_usability_grid.csv", index=False)

    # For each statement the paper makes with the screen, the smallest upper bound at which it stops holding.
    # Reported rather than hidden: it is the sensitivity the reviewer asked for, and it is favourable.
    # Indexed by column name, not by itertuples position: "3sig_rms_fails_on" starts with a digit, so pandas
    # renames it to a positional "_N" that silently shifts whenever a column is added ahead of it.
    STATEMENTS = {
        "charts fail on all four datasets": lambda r: bool(r["charts_fail_on_all_datasets"]),
        "3sig_rms fails on IMS": lambda r: "ims" in (r["3sig_rms_fails_on"] or ""),
        "XJTU-SY condition changes the verdict": lambda r: bool(r["xjtu_condition_verdict_differs"]),
        "at least four rules change verdict across datasets":
            lambda r: r["rules_changing_verdict_across_datasets"] >= 4,
    }
    flip_rows = []
    for pmin in PMINS:
        g = e[e.p_min == pmin].sort_values("p_max")
        for name, holds in STATEMENTS.items():
            ok = [holds(r) for r in g.to_dict("records")]
            first_fail = next((pm for pm, v in zip(g.p_max, ok) if not v), None)
            flip_rows.append({"p_min": pmin, "statement": name,
                              "holds_at_p_max_050": bool(ok[list(g.p_max).index(0.50)]),
                              "largest_p_max_still_holding": float(max(
                                  [pm for pm, v in zip(g.p_max, ok) if v], default=float("nan"))),
                              "first_p_max_failing": float(first_fail) if first_fail is not None else float("nan"),
                              "cells_holding": int(sum(ok)), "cells": len(ok)})
    fl = pd.DataFrame(flip_rows)
    fl.to_csv(T / "j23_usability_flippoints.csv", index=False)

    log(f"    cells: {len(e)} ({len(PMINS)} lower x {len(PMAXS)} upper bounds); rules changing verdict across "
        f"datasets: min {e.rules_changing_verdict_across_datasets.min()}, max {e.rules_changing_verdict_across_datasets.max()}")
    log(f"    charts fail on all datasets in {int(e.charts_fail_on_all_datasets.sum())}/{len(e)} cells; "
        f"XJTU-SY condition verdict differs in {int(e.xjtu_condition_verdict_differs.sum())}/{len(e)}")
    log("    where each statement stops holding as the upper bound is raised:")
    for r in fl.itertuples():
        held = f"{r.largest_p_max_still_holding:.2f}" if r.largest_p_max_still_holding == r.largest_p_max_still_holding else "never"
        fails = f"{r.first_p_max_failing:.2f}" if r.first_p_max_failing == r.first_p_max_failing else "never in grid"
        log(f"      p_min {r.p_min:<5} {r.statement:52s} at 0.50 {'yes' if r.holds_at_p_max_050 else 'no ':3s}  "
            f"holds up to p_max {held}  first fails at {fails}  ({r.cells_holding}/{r.cells} cells)")


# ------------------------------------------------------------------ F
def part_f(ft):
    log("\nF. label disagreement between rules")
    rules = RMS3 + ["ma2sig", "3sig_kurt"] + CHARTS + ["pct02", "pct05", "pct10"]
    rows = []
    for ds in ("xjtu", "femto", "ims"):
        g = ft[ft.dataset == ds].set_index(["trajectory", "rule"])
        trajs = sorted(ft[ft.dataset == ds].trajectory.unique())
        for a, b in itertools.combinations(rules, 2):
            inter = union = n_rec = agree = pa = pb = 0
            tp = tn = fp = fn = 0
            used = 0
            for t in trajs:
                ra, rb = g.loc[(t, a)], g.loc[(t, b)]
                if not (ra.fired and rb.fired):
                    continue
                n = int(ra.n_records)
                ia, ib = int(ra.fpt_index), int(rb.fpt_index)
                ya = np.arange(n) >= ia
                yb = np.arange(n) >= ib
                tp += int((ya & yb).sum()); tn += int((~ya & ~yb).sum()); fp += int((ya & ~yb).sum()); fn += int((~ya & yb).sum())
                used += 1
            tot = tp + tn + fp + fn
            if used == 0:
                continue
            po = (tp + tn) / tot
            pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / tot ** 2
            kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
            jac = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
            rows.append({"dataset": ds, "rule_a": a, "rule_b": b, "n_traj": used, "n_records": tot,
                         "disagreement": 1 - po, "jaccard_degraded": jac, "cohen_kappa": kappa})
    la = pd.DataFrame(rows)
    la.to_csv(T / "j23_label_agreement.csv", index=False)
    fam = lambda r: "chart" if r in CHARTS else ("rms3" if r in RMS3 else ("pct" if r.startswith("pct") else "other"))  # noqa: E731
    la["pair"] = [" / ".join(sorted([fam(a), fam(b)])) for a, b in zip(la.rule_a, la.rule_b)]
    for ds in ("xjtu", "femto"):
        q = la[la.dataset == ds].groupby("pair")[["disagreement", "jaccard_degraded", "cohen_kappa"]].median()
        log(f"    {ds}:\n" + q.round(3).to_string())
    spread = []
    for (ds, t), g in ft[ft.fired & ~ft.rule.str.startswith("pct")].groupby(["dataset", "trajectory"]):
        f = g.fpt_fraction.to_numpy()
        if len(f) >= 3:
            spread.append({"dataset": ds, "trajectory": t, "n_rules": len(f), "q10": np.quantile(f, 0.1), "q90": np.quantile(f, 0.9),
                           "spread_q90_q10": np.quantile(f, 0.9) - np.quantile(f, 0.1), "range": f.max() - f.min()})
    sp = pd.DataFrame(spread)
    sp.to_csv(T / "j23_onset_spread.csv", index=False)
    for ds in ("xjtu", "femto"):
        q = sp[sp.dataset == ds]
        log(f"    {ds:6s} per-trajectory onset spread over detector and chart rules (q90-q10): median {q.spread_q90_q10.median():.3f}, range {q.spread_q90_q10.min():.3f}-{q.spread_q90_q10.max():.3f}")


# ------------------------------------------------------------------ G
def part_g():
    log("\nG. rank stability of methods across labelling rules")
    p = pd.read_csv(T / "j3_performance.csv")
    p = p[(p["mode"] == "fixed") & (~p.degenerate)]
    rows = []
    for ds, g in p.groupby("dataset"):
        wide = g.pivot_table(index="method", columns="rule", values="macro_f1")
        wide = wide.dropna(axis=0, how="any")
        rules = list(wide.columns)
        winners = wide.idxmax(axis=0)
        for a, b in itertools.combinations(rules, 2):
            tau = kendalltau(wide[a], wide[b]).statistic
            rows.append({"dataset": ds, "rule_a": a, "rule_b": b, "n_methods": len(wide), "kendall_tau": tau,
                         "winner_a": winners[a], "winner_b": winners[b], "same_winner": winners[a] == winners[b]})
    rs = pd.DataFrame(rows)
    rs.to_csv(T / "j23_rank_stability.csv", index=False)
    for ds, q in rs.groupby("dataset"):
        log(f"    {ds:6s} usable rules {q[['rule_a', 'rule_b']].stack().nunique()}, pairs {len(q)}, methods {q.n_methods.iloc[0]}: "
            f"Kendall tau median {q.kendall_tau.median():+.2f} (min {q.kendall_tau.min():+.2f}, max {q.kendall_tau.max():+.2f}); "
            f"same winner in {int(q.same_winner.sum())}/{len(q)} pairs")


def main():
    global _fh
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j23_run.log", "w", encoding="utf-8")
    log(f"J23 reviewer analyses, started {datetime.now():%Y-%m-%d %H:%M:%S}, B={B}, seed {SEED}")
    rng = np.random.default_rng(SEED)
    j11 = load("j11", "J11_arl_and_covariates.py")
    j20 = load("j20", "J20_arl_mechanism.py")
    ft = pd.read_csv(T / "fig1_fpt_matrix.csv")
    part_a(ft, rng)
    part_c(j11.load_traj(), ft)
    part_d(j20, j11, rng)
    part_e()
    part_f(ft)
    part_g()
    log(f"\nfinished {datetime.now():%Y-%m-%d %H:%M:%S}")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
