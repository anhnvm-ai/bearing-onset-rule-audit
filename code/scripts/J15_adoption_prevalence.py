"""J15 -- what adopting a rule does and does not fix: prevalence by bearing subset, and the family-choice displacement.

Reads only results/tables:
    j3_fpt_times.csv                     fire index per firing rule-trajectory pair (all four datasets)
    j3_performance.csv (mode == 'fixed') pooled positive rate per dataset x rule, used to validate the recomputation

Writes:
    results/tables/j17_prevalence_by_subset.csv   pooled positive rate and usability verdict per dataset x rule x subset
                                                   (all bearings, each operating condition, each leave-one-bearing-out set)
    results/tables/j17_per_bearing_rates.csv      per-bearing positive rate (n - i) / n per rule
    results/tables/j18_family_displacement.csv    per bearing: mean percentile onset minus mean RMS-detector onset
    results/logs/j15_run.log

Definitions match J3: records from the fire index onwards are positive; a trajectory on which a rule never fires is
excluded; a label set is usable when the pooled positive rate over retained records lies in [0.01, 0.50].
Nothing here is random, so there are no intervals: every quantity is a deterministic property of rule and data.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T, LOGS = ROOT / "results" / "tables", ROOT / "results" / "logs"
MIN_RATE, MAX_RATE = 0.01, 0.50
RULES = ["3sig_rms", "ma3sig", "ma5sig", "ma2sig", "3sig_kurt", "ewma", "cusum", "pct02", "pct05", "pct10"]
RMS_DET = ["3sig_rms", "ma3sig", "ma5sig"]
PCT = ["pct02", "pct05", "pct10"]
_fh = None


def log(m=""):
    print(m)
    if _fh:
        _fh.write(m + "\n")


def condition(ds, traj):
    if ds in ("xjtu", "femto"):
        return "cond" + traj.replace("Bearing", "")[0]
    return "all"


def pooled(g):
    """Pooled positive rate over the firing trajectories in g (one row per trajectory)."""
    return float((g.n_records - g.fpt_index).sum() / g.n_records.sum()) if len(g) else np.nan


def usable(rate):
    return bool(MIN_RATE <= rate <= MAX_RATE) if np.isfinite(rate) else None


def main():
    global _fh
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    _fh = open(LOGS / "j15_run.log", "w", encoding="utf-8")
    bar = "=" * 78
    log(bar)
    log("J15  ADOPTION: prevalence by bearing subset and family-choice displacement")
    log(bar)
    ft = pd.read_csv(T / "j3_fpt_times.csv")
    perf = pd.read_csv(T / "j3_performance.csv")
    fx = perf[perf["mode"] == "fixed"].groupby(["dataset", "rule"]).agg(pos_rate=("pos_rate", "first"),
                                                                         degenerate=("degenerate", "first"))
    alltraj = ft.groupby(["dataset", "trajectory"]).n_records.first().reset_index()

    # ---- validation against J3
    bad = 0
    for (ds, r), g in ft.groupby(["dataset", "rule"]):
        if (ds, r) not in fx.index:
            continue
        rate = pooled(g)
        if abs(rate - fx.loc[(ds, r)].pos_rate) > 1e-12 or usable(rate) == bool(fx.loc[(ds, r)].degenerate):
            bad += 1
            log(f"  MISMATCH {ds} {r}: recomputed {rate} vs J3 {fx.loc[(ds, r)].pos_rate}")
    log(f"validation: pooled rate and verdict recomputed from j3_fpt_times match j3_performance for "
        f"{len(fx) - bad}/{len(fx)} dataset-rule pairs")

    # ---- per-bearing rates
    pb = ft.assign(rate=(ft.n_records - ft.fpt_index) / ft.n_records,
                   condition=[condition(d, t) for d, t in zip(ft.dataset, ft.trajectory)])
    pb[["dataset", "trajectory", "condition", "rule", "n_records", "fpt_index", "rate"]].to_csv(
        T / "j17_per_bearing_rates.csv", index=False)

    # ---- subsets
    rows = []
    for ds in ("xjtu", "femto", "ims", "nsk"):
        trajs = alltraj[alltraj.dataset == ds]
        conds = sorted({condition(ds, t) for t in trajs.trajectory}) if ds in ("xjtu", "femto") else []
        for r in RULES:
            g = pb[(pb.dataset == ds) & (pb.rule == r)]
            if g.empty:
                continue
            full = pooled(g)
            base = {"dataset": ds, "rule": r}
            rows.append({**base, "subset": "all", "n_bearings": trajs.shape[0], "n_firing": len(g),
                         "n_records": int(g.n_records.sum()), "pos_rate": full, "usable": usable(full),
                         "per_bearing_min": g.rate.min(), "per_bearing_max": g.rate.max(),
                         "per_bearing_mean": g.rate.mean()})
            for c in conds:
                gc = g[g.condition == c]
                rc = pooled(gc)
                rows.append({**base, "subset": c, "n_bearings": int(sum(condition(ds, t) == c for t in trajs.trajectory)),
                             "n_firing": len(gc), "n_records": int(gc.n_records.sum()), "pos_rate": rc, "usable": usable(rc)})
            if len(trajs) >= 3:
                for t in trajs.trajectory:
                    gl = g[g.trajectory != t]
                    rl = pooled(gl)
                    rows.append({**base, "subset": f"without:{t}", "n_bearings": trajs.shape[0] - 1, "n_firing": len(gl),
                                 "n_records": int(gl.n_records.sum()), "pos_rate": rl, "usable": usable(rl)})
    sub = pd.DataFrame(rows)
    sub.to_csv(T / "j17_prevalence_by_subset.csv", index=False)

    log("\nPooled positive rate and verdict (U usable, D degenerate): all bearings | by condition | leave-one-bearing-out")
    for ds in ("xjtu", "femto", "ims", "nsk"):
        log(f"  {ds}")
        for r in RULES:
            s = sub[(sub.dataset == ds) & (sub.rule == r)]
            if s.empty:
                log(f"    {r:9s} never fires")
                continue
            a = s[s.subset == "all"].iloc[0]
            cond = s[s.subset.str.startswith("cond")]
            lobo = s[s.subset.str.startswith("without:")]
            flips = lobo[lobo.usable != a.usable]
            ctext = "  ".join((f"{c.subset}=n/a(0 firing)" if c.n_firing == 0 else
                               f"{c.subset}={c.pos_rate:.3f}{'U' if c.usable else 'D'}({c.n_firing}f)") for c in cond.itertuples()) or "-"
            ltext = (f"LOBO {lobo.pos_rate.min():.3f}-{lobo.pos_rate.max():.3f}, verdict flips {len(flips)}/{len(lobo)}"
                     + (f" ({', '.join(x.replace('without:', '') for x in flips.subset)})" if len(flips) else "")) if len(lobo) else "-"
            log(f"    {r:9s} all={a.pos_rate:.3f}{'U' if a.usable else 'D'} per-bearing {a.per_bearing_min:.3f}-"
                f"{a.per_bearing_max:.3f} (mean {a.per_bearing_mean:.3f}) | {ctext} | {ltext}")

    # ---- family-choice displacement
    disp = []
    for ds in ("xjtu", "femto", "ims", "nsk"):
        w = ft[ft.dataset == ds].pivot_table(index="trajectory", columns="rule", values="fpt_fraction")
        det = w[[c for c in RMS_DET if c in w.columns]]
        pct = w[PCT]
        for t in w.index:
            if det.loc[t].notna().sum() == 0:
                continue
            disp.append({"dataset": ds, "trajectory": t, "n_rms_detectors_firing": int(det.loc[t].notna().sum()),
                         "mean_detector_onset": float(det.loc[t].mean()), "mean_percentile_onset": float(pct.loc[t].mean()),
                         "displacement": float(pct.loc[t].mean() - det.loc[t].mean())})
    dd = pd.DataFrame(disp)
    dd.to_csv(T / "j18_family_displacement.csv", index=False)
    log("\nFamily-choice displacement: mean percentile onset minus mean RMS-detector onset, per bearing")
    for ds, g in dd.groupby("dataset", sort=False):
        log(f"  {ds:5s} n={len(g):2d}  median {g.displacement.median():.3f}  range {g.displacement.min():.3f}-"
            f"{g.displacement.max():.3f}  positive {int((g.displacement > 0).sum())}/{len(g)}")
    log("\nwrote results/tables/j17_prevalence_by_subset.csv, j17_per_bearing_rates.csv, j18_family_displacement.csv")
    log(bar)
    _fh.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
