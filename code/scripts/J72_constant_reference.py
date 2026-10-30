"""J72 -- the error of constant predictions on each RUL scoring set (amendment A11; post hoc, descriptive).

The models trained under pct05 learn only the last 5% of each life. Over S^fix_j, the last 20% of each life, all of
their training records are scored, so their low C-common RMSE may say no more than a constant prediction near the end
of life would. This script computes, against the common reference r_t = (n - 1 - t)/(n - 1), the RMSE of

  pct05 constant  the mean of r_t over the training records of pct05 (t >= floor(0.95 n)) on the training bearings of
                  each fold, one bearing held out at a time as in J62
  best constant   the mean of r_t over the scored records of all bearings; no constant prediction scores lower

over S_j and S3_j of J62 (the last n_S and n_S3 records of each bearing, from j62_per_bearing.csv) and over S^fix_j of
J71 (t >= floor(0.8 n)), for every configuration and dataset. No model is involved. For comparison, the log lists the
C-common RMSE of the models trained under pct05 (median over the nine models; j62_common_reference, j71_cells).

Self-checks before anything is written: one length and one count per bearing in j62_per_bearing.csv, whatever the
labeler or model; the counts of S^fix_j equal n - floor(0.8 n).

Pre-specified in paper/access_resub/ANALYSIS_PLAN_AMENDMENTS.md, amendment A11.

Outputs: results/tables/j72_constant_reference.csv   config x dataset x set: both constants and their RMSE
         results/logs/j72_run.log
Usage  : python scripts/J72_constant_reference.py
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
PCT_P, FIX_FRACTION = 0.05, 0.8
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


def ref(n: int, t: np.ndarray) -> np.ndarray:
    return (n - 1 - t) / (n - 1)


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j72_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J72  CONSTANT PREDICTIONS ON EACH RUL SCORING SET  (A11, post hoc, descriptive)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}")

    pb = pd.read_csv(T / "j62_per_bearing.csv")
    a = pb[pb.variant == "a"]
    counts = a.groupby(["config", "dataset", "trajectory"])[["n_all", "n_S", "n_S3"]].nunique()
    if (counts != 1).any().any():
        log("  FAIL: lengths or scored counts differ between rows of the same bearing; nothing written")
        return 1
    bear = a.groupby(["config", "dataset", "trajectory"])[["n_all", "n_S", "n_S3"]].first().reset_index()
    cr = pd.read_csv(T / "j62_common_reference.csv")
    cells = pd.read_csv(T / "j71_cells.csv")

    rows = []
    for (cfg, ds), g in bear.groupby(["config", "dataset"], sort=False):
        n_of = dict(zip(g.trajectory, g.n_all))
        last = {"S": dict(zip(g.trajectory, g.n_S)), "S3": dict(zip(g.trajectory, g.n_S3)),
                "Sfix": {b: n - int(np.floor(FIX_FRACTION * n)) for b, n in n_of.items()}}
        # pct05 trains on t >= int(n (1 - p)), as J62 places its onset
        window = {b: ref(n, np.arange(int(n * (1 - PCT_P)), n)) for b, n in n_of.items()}
        for st in ("S", "S3", "Sfix"):
            scored = {b: ref(n, np.arange(n - last[st][b], n)) for b, n in n_of.items()}
            if st == "Sfix" and any(len(scored[b]) != n - int(np.floor(FIX_FRACTION * n)) for b, n in n_of.items()):
                log("  FAIL: S^fix counts; nothing written")
                return 1
            pooled = np.concatenate(list(scored.values()))
            best = float(pooled.mean())
            sse_pct, consts = 0.0, []
            for b in n_of:                                   # leave one bearing out, as J62
                c = float(np.concatenate([window[o] for o in n_of if o != b]).mean())
                consts.append(c)
                sse_pct += float(((scored[b] - c) ** 2).sum())
            rows.append({"config": cfg, "dataset": ds, "set": st, "n_bearings": len(n_of), "n_scored": len(pooled),
                         "pct05_constant_mean": float(np.mean(consts)),
                         "rmse_pct05_constant": float(np.sqrt(sse_pct / len(pooled))),
                         "best_constant": best, "rmse_best_constant": float(np.sqrt(((pooled - best) ** 2).mean()))})

    res = pd.DataFrame(rows)
    log(f"\n  self-check: one length and one count per bearing; S^fix counts equal n - floor(0.8 n): PASS")
    log(f"\n{BAR}\nRMSE against r_t: pct05 constant / best constant; models trained under pct05 (median of nine)\n{BAR}")
    a62 = cr[(cr.variant == "a") & (cr.labeler == "pct05")]
    a71 = cells[(cells.variant == "a") & (cells.labeler == "pct05")]
    for _, r in res.iterrows():
        if r.set == "Sfix":
            mod = a71[(a71.config == r.config) & (a71.dataset == r.dataset)].rmse_common_Sfix.median()
        else:
            mod = a62[(a62.config == r.config) & (a62.dataset == r.dataset)][f"rmse_common_{r.set}"].median()
        log(f"  {r.config:24s} {r.dataset:6s} {r.set:5s}  constant {r.pct05_constant_mean:.3f}: {r.rmse_pct05_constant:.3f}"
            f"   best constant {r.best_constant:.3f}: {r.rmse_best_constant:.3f}   pct05 models: {mod:.3f}")
    res.to_csv(T / "j72_constant_reference.csv", index=False)
    log(f"\nwrote j72_constant_reference.csv ({len(res)} rows)")
    log(f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
