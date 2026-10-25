"""J38 -- a bearing-bootstrap interval for the pooled labeller-versus-model factor of J35 (B1, 2026-09-21).

J35 reports, on pooled leave-one-bearing-out RMSE, that changing the labeller moves the error 7.6 times as much as
changing the model on XJTU-SY and 5.1 times on FEMTO:

    factor = median over the 6 models of (max - min over the 4 labellers)      label spread, model held fixed
           / median over the 4 labellers of (max - min over the 6 models)      model spread, labeller held fixed

That factor carried no interval. J37 gave an interval to a DIFFERENT quantity (the median over bearings of a
per-bearing ratio, 2.5 [1.6, 5.8]). This script gives one to the pooled factor itself, under the paper's
convention: 20,000 resamples of whole bearings, seed 0. In each resample the pooled RMSE of every
labeller x model cell is recomputed over the resampled bearings -- weighting records exactly as the pooled number
does -- then both spreads, then their ratio. Every cell of one resample uses the same bearings, so the ratio is
coherent. Nothing is refitted: the held-out predictions were scored once by J35, per bearing
(j35_per_bearing.csv), and a bearing's sum of squared errors is rmse^2 x n_records.

The same is done on the skill score against a constant baseline (J35's 3.8 and 4.2). The per-bearing baseline
is J37's rebuild of J35's protocol, which J37 checks against J35's pooled baseline before use.

Reported per dataset and scale: the point factor, the bootstrap median, the 2.5-97.5 percentile interval, and
the share of resamples in which the label spread is the larger of the two (the direction).

Inputs : results/tables/j35_per_bearing.csv, results/tables/j35_downstream_rul.csv, results/tables/j35_spread.csv,
         results/tables/fig1_fpt_matrix.csv (for the baseline, through J37)
Outputs: results/tables/j38_ratio_bootstrap.csv, results/logs/j38_run.log
Run    : python scripts/J38_downstream_ratio_bootstrap.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES, LOGS = ROOT / "results" / "tables", ROOT / "results" / "logs"
SEED, N_BOOT = 0, 20000            # the paper's convention for bearing bootstraps
LABELLERS = ["3sig_rms", "ma5sig", "ewma", "pct05"]
MODELS = ["lin:rms", "ridge:all", "knn:all", "gbr:all", "et:all", "rf:all"]
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO"}
LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


def load_j37():
    spec = importlib.util.spec_from_file_location("j37", ROOT / "scripts" / "J37_downstream_per_bearing.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def spreads(v):
    """v: (labellers, models, K) -> median label spread (K,), median model spread (K,)."""
    lab_sp = v.max(axis=0) - v.min(axis=0)          # (models, K): model fixed, labeller varied
    mod_sp = v.max(axis=1) - v.min(axis=1)          # (labellers, K): labeller fixed, model varied
    return np.median(lab_sp, axis=0), np.median(mod_sp, axis=0)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rng = np.random.default_rng(SEED)
    pb = pd.read_csv(TABLES / "j35_per_bearing.csv")
    pooled = pd.read_csv(TABLES / "j35_downstream_rul.csv")
    spread = pd.read_csv(TABLES / "j35_spread.csv")
    ft = pd.read_csv(TABLES / "fig1_fpt_matrix.csv")
    j37 = load_j37()

    log("J38 -- bearing-bootstrap interval for the pooled labeller/model factor of J35")
    log(f"{N_BOOT} resamples of whole bearings, seed {SEED}; no model refitted")
    log("=" * 100)
    rows = []
    for ds in ("xjtu", "femto"):
        d = pb[pb.dataset == ds]
        bearings = sorted(d.bearing.unique())
        B = len(bearings)
        # per-bearing sums of squared errors, (labellers, models, bearings), and record counts
        sse = np.zeros((len(LABELLERS), len(MODELS), B))
        cnt = None
        for li, lab in enumerate(LABELLERS):
            for mi, m in enumerate(MODELS):
                x = d[(d.labeller == lab) & (d.model == m)].set_index("bearing").reindex(bearings)
                if x.rmse.isna().any():
                    raise SystemExit(f"{ds}/{lab}/{m}: missing bearings")
                sse[li, mi] = x.rmse.to_numpy() ** 2 * x.n_records.to_numpy()
                c = x.n_records.to_numpy(dtype=float)
                if cnt is None:
                    cnt = c
                elif not np.array_equal(cnt, c):
                    raise SystemExit(f"{ds}: record counts differ between cells")

        # ---- check 1: the pooled RMSE rebuilt from the per-bearing rows is J35's pooled RMSE
        rm_point = np.sqrt(sse.sum(-1) / cnt.sum())
        worst = 0.0
        for li, lab in enumerate(LABELLERS):
            for mi, m in enumerate(MODELS):
                j = pooled[(pooled.dataset == ds) & (pooled.labeller == lab) & (pooled.model == m)].rmse.iloc[0]
                worst = max(worst, abs(rm_point[li, mi] - j))
        if worst > 1e-12:
            raise SystemExit(f"{ds}: rebuilt pooled RMSE differs from J35 by {worst:.3e}")

        # ---- constant-baseline SSE per bearing (J37's rebuild of J35's protocol), for the skill scale
        base = j37.baselines(ds, bearings, ft)
        bsse = np.array([[base[(lab, b)][0] for b in bearings] for lab in LABELLERS])     # (labellers, B)
        sk_point = 1.0 - sse.sum(-1) / bsse.sum(-1)[:, None]
        worst = 0.0
        for li, lab in enumerate(LABELLERS):
            for mi, m in enumerate(MODELS):
                j = pooled[(pooled.dataset == ds) & (pooled.labeller == lab) & (pooled.model == m)].skill.iloc[0]
                worst = max(worst, abs(sk_point[li, mi] - j))
        if worst > 1e-9:
            raise SystemExit(f"{ds}: rebuilt pooled skill differs from J35 by {worst:.3e}")
        log(f"{DS_NAME[ds]}: {B} bearings; pooled RMSE and skill rebuilt from the per-bearing rows match J35")

        # ---- one set of resamples per dataset, shared by every cell and both scales
        idx = rng.integers(0, B, size=(N_BOOT, B))
        C = np.zeros((N_BOOT, B))
        np.add.at(C, (np.arange(N_BOOT)[:, None], idx), 1.0)
        S = np.einsum("lmb,kb->lmk", sse, C)
        N = C @ cnt
        SB = np.einsum("lb,kb->lk", bsse, C)
        boot = {"rmse": np.sqrt(S / N), "skill": 1.0 - S / SB[:, None, :]}
        point = {"rmse": rm_point[:, :, None], "skill": sk_point[:, :, None]}

        for scale in ("rmse", "skill"):
            lp, mp = spreads(point[scale])
            f_point = float(lp[0] / mp[0])
            # ---- check 2: the point factor is J35's
            s = spread[(spread.dataset == ds) & (spread.metric == scale)]
            j35_f = s[s.held_fixed == "model"].spread.median() / s[s.held_fixed == "labeller"].spread.median()
            if abs(f_point - j35_f) > 1e-9:
                raise SystemExit(f"{ds}/{scale}: point factor {f_point} != J35 {j35_f}")
            lb, mb = spreads(boot[scale])
            f = lb / mb
            row = {"dataset": ds, "scale": scale, "n_bearings": B, "n_boot": N_BOOT, "seed": SEED,
                   "factor": f_point, "factor_boot_median": float(np.median(f)),
                   "factor_lo": float(np.percentile(f, 2.5)), "factor_hi": float(np.percentile(f, 97.5)),
                   "share_label_gt_model": float((lb > mb).mean()),
                   "share_factor_below_point": float((f < f_point).mean()),
                   "label_spread": float(lp[0]), "label_spread_lo": float(np.percentile(lb, 2.5)),
                   "label_spread_hi": float(np.percentile(lb, 97.5)),
                   "model_spread": float(mp[0]), "model_spread_lo": float(np.percentile(mb, 2.5)),
                   "model_spread_hi": float(np.percentile(mb, 97.5))}
            rows.append(row)
            log(f"  [{scale:5s}] factor {f_point:.2f}  bootstrap median {row['factor_boot_median']:.2f}  "
                f"95% [{row['factor_lo']:.2f}, {row['factor_hi']:.2f}]  "
                f"label > model in {100 * row['share_label_gt_model']:.1f}% of resamples; "
                f"{100 * row['share_factor_below_point']:.1f}% of resampled factors lie below the point value")
            log(f"           label spread {lp[0]:.3f} [{row['label_spread_lo']:.3f}, {row['label_spread_hi']:.3f}]"
                f"   model spread {mp[0]:.3f} [{row['model_spread_lo']:.3f}, {row['model_spread_hi']:.3f}]")
        log("")

    pd.DataFrame(rows).to_csv(TABLES / "j38_ratio_bootstrap.csv", index=False)
    log("written: j38_ratio_bootstrap.csv")
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "j38_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
