"""J37 -- the downstream labeller-versus-model comparison of J35, read bearing by bearing (Phase 5).

J35 reports POOLED numbers: over the leave-one-bearing-out predictions of six regressors under four
labelling rules, changing the labeller moves the pooled RMSE 7.6 times as much as changing the model on
XJTU-SY and 5.1 times on FEMTO. A pooled figure can be carried by a few long bearings -- J32 showed that
the pooled RUL-target distance was -- so the comparison has to be repeated per bearing, with an interval
whose unit is the bearing. Nothing is refitted: every held-out prediction was already scored per bearing
by J35 (j35_per_bearing.csv), and this script only regroups those numbers.

Per bearing j and per scoring scale (RMSE, and skill against a constant baseline):
    label spread  LS_j = median over the 6 models of (max - min over the 4 labellers)   model held fixed
    model spread  MS_j = median over the 4 labellers of (max - min over the 6 models)   labeller held fixed
and the ratio LS_j / MS_j. Reported: the median over bearings with its interquartile range and a 95%
interval from 20,000 resamples of whole bearings, and the number of bearings on which LS_j > MS_j.

The reviewer's pairwise table is also produced: Delta_j = RMSE_j(labeller A) - RMSE_j(labeller B) for
the within-family pair (3sig_rms, ma5sig), the cross-family pair (3sig_rms, ewma) and the percentile
pair (3sig_rms, pct05), for every model, summarised the same way.

The skill score needs a constant baseline per bearing, which J35 computed only pooled. It is rebuilt
here under J35's own protocol -- the held-out bearing is predicted by the mean target over the records
of the other usable bearings -- and checked against J35's pooled baseline RMSE before it is used.

Inputs : results/tables/j35_per_bearing.csv        per dataset x labeller x model x bearing (J35)
         results/tables/j35_downstream_rul.csv     pooled rows, for the baseline check (J35)
         results/tables/fig1_fpt_matrix.csv        onsets, to rebuild the targets of Eq. (1)
Outputs: results/tables/j37_per_bearing.csv        per dataset x bearing: LS_j, MS_j, ratio, both scales
         results/tables/j37_per_bearing_summary.csv
         results/tables/j37_pair_delta.csv         per dataset x pair x model x bearing: Delta_j
         results/tables/j37_pair_summary.csv
         results/logs/j37_run.log

Run: python scripts/J37_downstream_per_bearing.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES, LOGS = ROOT / "results" / "tables", ROOT / "results" / "logs"
SEED, N_BOOT = 0, 20000
LABELLERS = ["3sig_rms", "ma5sig", "ewma", "pct05"]
PAIRS = [("3sig_rms", "ma5sig", "within RMS detectors"),
         ("3sig_rms", "ewma", "detector vs chart"),
         ("3sig_rms", "pct05", "detector vs percentile")]
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO"}
LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


def target(n, i):
    """Eq. (1), identical to J35.target."""
    t = np.arange(n, dtype=float)
    y = np.ones(n)
    after = t >= int(i)
    y[after] = (n - 1 - t[after]) / max(n - 1 - int(i), 1)
    return y


def summarise(v, rng):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    boot = np.median(v[rng.integers(0, len(v), size=(N_BOOT, len(v)))], axis=1)
    return {"n_bearings": len(v), "median": float(np.median(v)),
            "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
            "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5))}


def baselines(ds, bearings, ft):
    """Per-bearing constant-baseline MSE under J35's protocol, for every labeller."""
    g = ft[ft.dataset == ds].set_index(["trajectory", "rule"])
    out = {}
    for lab in LABELLERS:
        ys = {b: target(int(g.loc[(b, lab), "n_records"]), g.loc[(b, lab), "fpt_index"]) for b in bearings}
        tot_sum = sum(y.sum() for y in ys.values())
        tot_n = sum(len(y) for y in ys.values())
        for b in bearings:
            y = ys[b]
            mean_other = (tot_sum - y.sum()) / (tot_n - len(y))
            out[(lab, b)] = (float(((y - mean_other) ** 2).sum()), len(y))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rng = np.random.default_rng(SEED)
    pb = pd.read_csv(TABLES / "j35_per_bearing.csv")
    pooled = pd.read_csv(TABLES / "j35_downstream_rul.csv")
    ft = pd.read_csv(TABLES / "fig1_fpt_matrix.csv")
    models = list(dict.fromkeys(pb.model))

    log("J37 -- labeller versus model, bearing by bearing (from J35's leave-one-bearing-out scores)")
    log("=" * 100)
    per_rows, sum_rows, pair_rows, pair_sum = [], [], [], []

    for ds in ("xjtu", "femto"):
        d = pb[pb.dataset == ds].copy()
        bearings = sorted(d.bearing.unique())

        # ---- constant baseline per bearing, checked against J35's pooled value before use
        base = baselines(ds, bearings, ft)
        for lab in LABELLERS:
            sse = sum(base[(lab, b)][0] for b in bearings)
            cnt = sum(base[(lab, b)][1] for b in bearings)
            mine = np.sqrt(sse / cnt)
            theirs = pooled[(pooled.dataset == ds) & (pooled.labeller == lab)].rmse_baseline.iloc[0]
            if abs(mine - theirs) > 1e-9:
                raise SystemExit(f"{ds}/{lab}: rebuilt pooled baseline {mine:.12f} != J35 {theirs:.12f}")
        log(f"{DS_NAME[ds]}: {len(bearings)} bearings; rebuilt constant baseline matches J35's pooled value "
            f"for all 4 labellers (|diff| < 1e-9)")

        d["mse"] = d.rmse ** 2
        d["mse_base"] = [base[(r.labeller, r.bearing)][0] / base[(r.labeller, r.bearing)][1]
                         for r in d.itertuples()]
        d["skill"] = 1.0 - d.mse / d.mse_base

        for b in bearings:
            x = d[d.bearing == b]
            row = {"dataset": ds, "bearing": b, "n_records": int(x.n_records.iloc[0])}
            for scale in ("rmse", "skill"):
                ls = x.groupby("model")[scale].agg(lambda v: v.max() - v.min()).median()
                ms = x.groupby("labeller")[scale].agg(lambda v: v.max() - v.min()).median()
                row[f"label_spread_{scale}"] = float(ls)
                row[f"model_spread_{scale}"] = float(ms)
                row[f"ratio_{scale}"] = float(ls / ms) if ms > 0 else np.nan
            per_rows.append(row)

        p = pd.DataFrame([r for r in per_rows if r["dataset"] == ds])
        for scale in ("rmse", "skill"):
            for q in ("label_spread", "model_spread", "ratio"):
                sum_rows.append({"dataset": ds, "scale": scale, "quantity": q,
                                 **summarise(p[f"{q}_{scale}"], rng)})
            k = int((p[f"label_spread_{scale}"] > p[f"model_spread_{scale}"]).sum())
            sum_rows.append({"dataset": ds, "scale": scale, "quantity": "bearings_label_gt_model",
                             "n_bearings": len(p), "median": k})

        # ---- the reviewer's pairwise table
        for a, b_, kind in PAIRS:
            for m in models:
                xa = d[(d.labeller == a) & (d.model == m)].set_index("bearing").rmse
                xb = d[(d.labeller == b_) & (d.model == m)].set_index("bearing").rmse
                delta = (xa - xb).reindex(bearings)
                for bb, v in delta.items():
                    pair_rows.append({"dataset": ds, "labeller_a": a, "labeller_b": b_, "pair": kind,
                                      "model": m, "bearing": bb, "delta_rmse": float(v)})
                s = summarise(delta.values, rng)
                sa = summarise(np.abs(delta.values), rng)
                pair_sum.append({"dataset": ds, "labeller_a": a, "labeller_b": b_, "pair": kind, "model": m,
                                 **{f"delta_{k}": v for k, v in s.items() if k != "n_bearings"},
                                 **{f"absdelta_{k}": v for k, v in sa.items() if k != "n_bearings"},
                                 "n_bearings": s["n_bearings"]})

    per = pd.DataFrame(per_rows)
    summ = pd.DataFrame(sum_rows)
    pairs = pd.DataFrame(pair_rows)
    psum = pd.DataFrame(pair_sum)
    per.to_csv(TABLES / "j37_per_bearing.csv", index=False)
    summ.to_csv(TABLES / "j37_per_bearing_summary.csv", index=False)
    pairs.to_csv(TABLES / "j37_pair_delta.csv", index=False)
    psum.to_csv(TABLES / "j37_pair_summary.csv", index=False)

    for ds in ("xjtu", "femto"):
        s = summ[summ.dataset == ds]
        log("")
        log(f"{DS_NAME[ds]} -- per-bearing medians [IQR] (95% bearing-bootstrap CI of the median)")
        for scale in ("rmse", "skill"):
            t = s[s.scale == scale].set_index("quantity")
            f = lambda q: (f"{t.loc[q, 'median']:.3f} [{t.loc[q, 'q25']:.3f}, {t.loc[q, 'q75']:.3f}] "
                           f"({t.loc[q, 'ci_lo']:.3f}, {t.loc[q, 'ci_hi']:.3f})")
            log(f"  [{scale}] label spread {f('label_spread')}")
            log(f"  [{scale}] model spread {f('model_spread')}")
            log(f"  [{scale}] ratio        {f('ratio')}")
            log(f"  [{scale}] label spread > model spread on {int(t.loc['bearings_label_gt_model', 'median'])} "
                f"of {int(t.loc['bearings_label_gt_model', 'n_bearings'])} bearings")
        ps = psum[psum.dataset == ds]
        log(f"  pairwise Delta_j = RMSE_j(A) - RMSE_j(B), median over bearings, median |Delta_j| across the 6 models:")
        for (a, b_, kind), g in ps.groupby(["labeller_a", "labeller_b", "pair"], sort=False):
            log(f"    {kind:24s} {a} - {b_}: median Delta {g.delta_median.median():+.3f}, "
                f"median |Delta| {g.absdelta_median.median():.3f} "
                f"(range over models {g.absdelta_median.min():.3f}-{g.absdelta_median.max():.3f})")
    log("")
    log("written: j37_per_bearing.csv, j37_per_bearing_summary.csv, j37_pair_delta.csv, j37_pair_summary.csv")
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "j37_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
