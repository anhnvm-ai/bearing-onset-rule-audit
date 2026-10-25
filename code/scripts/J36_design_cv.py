"""J36 -- is the response-size curve an explanation or a restatement of the parameterisation? (C1, revision).

Section 4.3 regresses the Spearman correlation rho(alarm record, trajectory length) on the measured median shift at
each design's alarm, over 67 designs, and reports R2 = 0.788 for the size term against 0.077 for the family label.
Two objections are made to that number, and they need different answers.

OBJECTION 1 -- "this is tautological". A design's parameters fix the shift at which it alarms, so of course a design
that fires at a larger shift fires later. The reply cannot be another in-sample fit. It has to be a prediction made
where the fit could fail: fit the curve on ONE family and predict the OTHER. If a quadratic in log(shift) estimated
from twelve RMS thresholds predicts the behaviour of forty-five EWMA and CUSUM charts -- designs with different
statistics, different update equations and different parameters -- then the curve is not a restatement of any one
family's parameterisation. Three held-out splits are run:

        fit on 12 detectors                  -> predict 45 EWMA/CUSUM charts
        fit on 45 EWMA/CUSUM charts          -> predict 12 detectors
        fit on those 57 conventional designs -> predict the 10 AR-corrected and adaptive-limit charts

The third is the strictest: those ten designs are a class the fit has never seen, added in the revision as the
"untested alternatives" of Section 7.

OBJECTION 2 -- "67 designs are not 67 independent observations". They are not. Every design is applied to the SAME
bearings, so the regression has 67 rows but only 15 independent units on XJTU-SY and 13 on FEMTO. An in-sample R2
computed over designs has no interval, and quoting it as though it did overstates the evidence. This script
resamples BEARINGS, recomputes every design's rho and median shift on the resampled bearings, refits both models,
and reports the bootstrap distribution of R2_size and R2_family. The unit of uncertainty is then the bearing, which
is the unit the rest of the paper uses.

A third reading is added because it is nearly free: leave-one-DESIGN-out cross-validation, which says how much of
the 0.788 is in-sample optimism from fitting three coefficients to 67 points.

DEFINITIONS are taken unchanged from J19_export_numbers.py so the numbers are comparable with Section 4.3:
    size model    rho ~ [1, log(shift), log(shift)^2]
    family model  rho ~ [1, 1{family == "detector"}]
    shift is clipped below at 1e-3 before the log, as there.

WHAT THIS DOES NOT SHOW. A curve that predicts across families is evidence against the tautology reading, not
evidence of a causal mechanism. The claim it supports is that realized response magnitude ORGANIZES the duration
scaling better than the family label does, not that response magnitude causes it. The rule-free crossing analysis
of Section 4.3 and the known-change simulation of Section 4.5 carry the part of the argument that no regression
over rule designs can carry, because neither uses a rule at all.

Inputs : results/tables/j26_design_sweep.csv            one row per design x dataset (aggregates)
         results/tables/j26_design_sweep_per_traj.csv   one row per design x trajectory (written by J26)
Outputs: results/tables/j36_design_cv.csv          leave-one-design-out and leave-one-family-out results
         results/tables/j36_cross_family_pred.csv  every held-out design with its observed and predicted rho
         results/tables/j36_design_bootstrap.csv   bearing-bootstrap distribution of R2_size and R2_family
         results/logs/j36_run.log

Run: python scripts/J36_design_cv.py [n_bootstrap]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 100

SEED = 0
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
MIN_FIRED = 4                 # a correlation over fewer than four firing trajectories is not computed, as in J26
DATASETS = ["xjtu", "femto"]
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO"}

# the three held-out splits, named by the family values J26 writes
DETECTOR = ["detector"]
CONVENTIONAL_CHART = ["control chart"]
NOVEL_CHART = ["chart, AR-corrected", "chart, adaptive limits"]

LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def design_matrix(shift: np.ndarray, is_detector: np.ndarray, kind: str) -> np.ndarray:
    """Exactly the two models of J19_export_numbers.py, so R2 is comparable with Section 4.3."""
    lx = np.log(np.clip(shift, 1e-3, None))
    if kind == "size":
        return np.column_stack([np.ones(len(lx)), lx, lx ** 2])
    if kind == "family":
        return np.column_stack([np.ones(len(lx)), is_detector.astype(float)])
    if kind == "size+family":
        return np.column_stack([np.ones(len(lx)), lx, lx ** 2, is_detector.astype(float)])
    raise ValueError(kind)


def r2_in_sample(y: np.ndarray, X: np.ndarray) -> float:
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    sst = float((y - y.mean()) @ (y - y.mean()))
    return float("nan") if sst <= 0 else float(1 - (r @ r) / sst)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho, written out so it can be called a million times inside the bootstrap."""
    if len(x) < MIN_FIRED:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d > 0 else np.nan


# ------------------------------------------------------------------ (a) + (b) held-out prediction
def held_out(y: np.ndarray, X: np.ndarray, train: np.ndarray, test: np.ndarray):
    """Fit on `train`, predict `test`. Returns (prediction, dict of scores).

    Two R2 are reported because they answer different questions when the held-out set is a different family:
      r2_own    1 - SSres / SS about the HELD-OUT mean -- does the curve explain variation WITHIN the new family?
      r2_train  1 - SSres / SS about the TRAINING mean -- does it beat carrying the old family's average over?
    r2_own is the strict one. r2_train is the one that is easy to look good on, and is reported so that it cannot
    be quoted alone.
    """
    b = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
    pred = X[test] @ b
    obs = y[test]
    sse = float(((obs - pred) ** 2).sum())
    ss_own = float(((obs - obs.mean()) ** 2).sum())
    ss_train = float(((obs - y[train].mean()) ** 2).sum())
    base = y[train].mean()
    return pred, {
        "n_train": int(train.sum()), "n_test": int(test.sum()),
        "median_abs_error": float(np.median(np.abs(obs - pred))),
        "median_abs_error_baseline": float(np.median(np.abs(obs - base))),
        "r2_own": float("nan") if ss_own <= 0 else 1 - sse / ss_own,
        "r2_train": float("nan") if ss_train <= 0 else 1 - sse / ss_train,
        # added 2026-09-21 (Phase 3B): mean absolute error and RMSE, each against the carry-the-training-
        # mean baseline, so the held-out comparison is reported on the usual scales and not only as a median
        "mae": float(np.mean(np.abs(obs - pred))),
        "mae_baseline": float(np.mean(np.abs(obs - base))),
        "rmse": float(np.sqrt(np.mean((obs - pred) ** 2))),
        "rmse_baseline": float(np.sqrt(np.mean((obs - base) ** 2))),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    LOGS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    sw = pd.read_csv(TABLES / "j26_design_sweep.csv")
    pt_path = TABLES / "j26_design_sweep_per_traj.csv"
    if not pt_path.exists():
        raise SystemExit("j26_design_sweep_per_traj.csv is missing -- run scripts/J26_design_sweep.py first")
    pt = pd.read_csv(pt_path)

    log(BAR)
    log("J36  cross-family validation and a bearing-level interval for the response-size curve".center(100))
    log(BAR)
    log(f"models as in J19: size = [1, log(shift), log(shift)^2]; family = [1, 1(detector)]")
    log(f"bootstrap resamples: {N_BOOT:,} (unit = bearing)")

    cv_rows, pred_rows, boot_rows = [], [], []

    for ds in DATASETS:
        g = sw[(sw.dataset == ds) & sw.median_shift_sd.notna() & sw.rho_index_length.notna()].copy()
        g = g.reset_index(drop=True)
        y = g.rho_index_length.to_numpy(float)
        shift = g.median_shift_sd.to_numpy(float)
        isdet = (g.family == "detector").to_numpy()
        fam = g.family.to_numpy()

        log(f"\n{BAR}\n{DS_NAME[ds]}: {len(g)} designs with a measurable response size\n{BAR}")
        for f in dict.fromkeys(fam):
            log(f"    {f:26s} {int((fam == f).sum()):3d} designs")

        # ---- in-sample, to reproduce Section 4.3 and anchor everything below
        for kind in ("size", "family", "size+family"):
            r2 = r2_in_sample(y, design_matrix(shift, isdet, kind))
            cv_rows.append({"dataset": ds, "analysis": "in-sample", "model": kind, "split": "all designs",
                            "n_train": len(g), "n_test": len(g), "r2_own": r2, "r2_train": np.nan,
                            "median_abs_error": np.nan, "median_abs_error_baseline": np.nan})
            log(f"    in-sample R2 [{kind:12s}] {r2:+.3f}")

        # ---- (a) leave-one-design-out
        log(f"\n  (a) leave-one-design-out cross-validation")
        for kind in ("size", "family"):
            X = design_matrix(shift, isdet, kind)
            pred = np.empty(len(y))
            for i in range(len(y)):
                tr = np.ones(len(y), bool)
                tr[i] = False
                b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
                pred[i] = X[i] @ b
            sse = float(((y - pred) ** 2).sum())
            sst = float(((y - y.mean()) ** 2).sum())
            r2cv = 1 - sse / sst
            # the baseline for a left-out design is the mean of the OTHER designs, as the fit's own protocol
            loo_mean = (y.sum() - y) / (len(y) - 1)
            cv_rows.append({"dataset": ds, "analysis": "leave-one-design-out", "model": kind,
                            "split": "all designs", "n_train": len(g) - 1, "n_test": len(g),
                            "r2_own": r2cv, "r2_train": np.nan,
                            "median_abs_error": float(np.median(np.abs(y - pred))),
                            "median_abs_error_baseline": float(np.median(np.abs(y - y.mean()))),
                            "mae": float(np.mean(np.abs(y - pred))),
                            "mae_baseline": float(np.mean(np.abs(y - loo_mean))),
                            "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
                            "rmse_baseline": float(np.sqrt(np.mean((y - loo_mean) ** 2)))})
            log(f"      {kind:12s} CV-R2 {r2cv:+.3f}   median |error| {np.median(np.abs(y - pred)):.3f}   "
                f"(in-sample {r2_in_sample(y, X):+.3f})")

        # ---- (b) leave-one-family-out
        log(f"\n  (b) leave-one-family-out: fit on one family, predict another")
        splits = [
            ("detectors -> conventional charts", np.isin(fam, DETECTOR), np.isin(fam, CONVENTIONAL_CHART)),
            ("conventional charts -> detectors", np.isin(fam, CONVENTIONAL_CHART), np.isin(fam, DETECTOR)),
            ("conventional -> novel charts",
             np.isin(fam, DETECTOR + CONVENTIONAL_CHART), np.isin(fam, NOVEL_CHART)),
        ]
        X = design_matrix(shift, isdet, "size")
        for name, tr, te in splits:
            if tr.sum() < 4 or te.sum() < 3:
                log(f"      {name:36s} skipped (train {int(tr.sum())}, test {int(te.sum())})")
                continue
            pred, sc = held_out(y, X, tr, te)
            cv_rows.append({"dataset": ds, "analysis": "leave-one-family-out", "model": "size",
                            "split": name, **sc})
            log(f"      {name:36s} train {sc['n_train']:2d} test {sc['n_test']:2d} | "
                f"median |err| {sc['median_abs_error']:.3f} (baseline {sc['median_abs_error_baseline']:.3f}) | "
                f"R2 vs own mean {sc['r2_own']:+.3f} | vs train mean {sc['r2_train']:+.3f}")
            for j, k in enumerate(np.where(te)[0]):
                pred_rows.append({"dataset": ds, "split": name, "design": g.design.iloc[k],
                                  "family": fam[k], "median_shift_sd": shift[k],
                                  "rho_observed": y[k], "rho_predicted": float(pred[j]),
                                  "abs_error": float(abs(y[k] - pred[j]))})

        # ---- (c) bearing-block bootstrap
        log(f"\n  (c) bearing bootstrap: the 67 designs share bearings, so the unit is the bearing")
        p = pt[pt.dataset == ds]
        designs = list(dict.fromkeys(p.design))
        trajs = list(dict.fromkeys(p.trajectory))
        nT = len(trajs)
        t_idx = {t: i for i, t in enumerate(trajs)}
        d_idx = {d: i for i, d in enumerate(designs)}
        ALARM = np.full((len(designs), nT), np.nan)
        SHIFT = np.full((len(designs), nT), np.nan)
        NREC = np.zeros(nT)
        for r in p.itertuples():
            i, j = d_idx[r.design], t_idx[r.trajectory]
            NREC[j] = r.n_records
            if r.fired:
                ALARM[i, j] = r.alarm_index
                SHIFT[i, j] = r.shift_sd
        FAMILY_OF = {d: p[p.design == d].family.iloc[0] for d in designs}
        IS_DET = np.array([FAMILY_OF[d] == "detector" for d in designs])

        log(f"      {len(designs)} designs x {nT} bearings")
        got = 0
        for _ in range(N_BOOT):
            s = rng.integers(0, nT, nT)
            nvals = NREC[s]
            rho_b = np.full(len(designs), np.nan)
            shift_b = np.full(len(designs), np.nan)
            A, S = ALARM[:, s], SHIFT[:, s]
            for i in range(len(designs)):
                m = np.isfinite(A[i])
                if m.sum() >= MIN_FIRED:
                    rho_b[i] = spearman(nvals[m], A[i][m])
                    shift_b[i] = np.nanmedian(S[i][m])
            ok = np.isfinite(rho_b) & np.isfinite(shift_b)
            if ok.sum() < 10 or len(set(IS_DET[ok])) < 2:
                continue
            yb, sb, db = rho_b[ok], shift_b[ok], IS_DET[ok]
            boot_rows.append({"dataset": ds, "n_designs": int(ok.sum()),
                              "r2_size": r2_in_sample(yb, design_matrix(sb, db, "size")),
                              "r2_family": r2_in_sample(yb, design_matrix(sb, db, "family")),
                              "r2_size_family": r2_in_sample(yb, design_matrix(sb, db, "size+family"))})
            got += 1
        log(f"      {got:,} of {N_BOOT:,} resamples usable "
            f"(a resample is dropped when fewer than 10 designs still fire on 4+ bearings)")

    cv = pd.DataFrame(cv_rows)
    cv.to_csv(TABLES / "j36_design_cv.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(TABLES / "j36_cross_family_pred.csv", index=False)
    bt = pd.DataFrame(boot_rows)
    bt.to_csv(TABLES / "j36_design_bootstrap.csv", index=False)

    log(f"\n{BAR}\nbearing-bootstrap intervals for the design-level fit\n{BAR}")
    for ds in DATASETS:
        b = bt[bt.dataset == ds]
        if not len(b):
            continue
        log(f"\n  {DS_NAME[ds]}  ({len(b):,} resamples)")
        for col, nm in (("r2_size", "R2 size"), ("r2_family", "R2 family"), ("r2_size_family", "R2 size+family")):
            v = b[col].dropna()
            log(f"    {nm:16s} median {v.median():+.3f}   95% interval "
                f"[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]")
        share = float((b.r2_size > b.r2_family).mean())
        log(f"    the size term beats the family label in {share:.1%} of resamples")

    log(f"\n{BAR}")
    log("written: j36_design_cv.csv, j36_cross_family_pred.csv, j36_design_bootstrap.csv")
    (LOGS / "j36_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
