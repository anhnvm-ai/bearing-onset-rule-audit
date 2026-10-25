"""J43 -- does the response-size curve predict designs measured on bearings it was not fitted on?

Section IV-C holds designs out of the fit: leave one design out, leave one family out. Every one of those
folds trains and tests on the same bearings, so none of them answers the generalisation a reader means
when the unit of the claim is the bearing, which is what Section IV-B argues it is.

This script holds BEARINGS out. The bearings of a dataset are split in half at random; each design's
correlation rho(i, n) and median shift are recomputed inside each half, the curve is fitted on one half
and applied to the design measurements of the other, and the out-of-sample R^2 is recorded. Repeating
over many random splits gives the distribution of what the curve transfers to bearings it has not seen.

Two halves are needed because rho is itself a correlation across bearings: a single held-out bearing has
no rho. With 15 XJTU-SY bearings a half holds seven or eight, above the four this paper requires anywhere.

The comparison the paper cares about is between explanations, so the same split is scored for the size
term and for the family label, and the two are reported side by side. A design that does not fire on at
least four bearings of a half has no rho there and sits out that split, which is stated per split.

Inputs : results/tables/j26_design_sweep_per_traj.csv   one row per design x trajectory (alarm, shift)
Outputs: results/tables/j43_heldout_bearings.csv        one row per dataset x split x model
         results/tables/j43_heldout_summary.csv         one row per dataset x model
         results/logs/j43_run.log

Usage: python scripts/J43_heldout_bearings.py [N_SPLITS]        (default 2000)
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"

SEED = 43
N_SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
MIN_FIRED = 4          # as in J26, J36 and J41: no correlation over fewer than four trajectories
MIN_DESIGNS = 10       # a fit over fewer designs than this is not attempted, as in J36
DATASETS = ["xjtu", "femto"]
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO"}
BAR = "=" * 78

LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < MIN_FIRED:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d > 0 else np.nan


def measure(ALARM: np.ndarray, SHIFT: np.ndarray, NREC: np.ndarray, cols: np.ndarray):
    """Each design's rho and median shift, computed on the given bearings only."""
    n = ALARM.shape[0]
    rho = np.full(n, np.nan)
    shift = np.full(n, np.nan)
    a, s, nrec = ALARM[:, cols], SHIFT[:, cols], NREC[cols]
    for i in range(n):
        m = np.isfinite(a[i])
        if m.sum() >= MIN_FIRED:
            rho[i] = spearman(nrec[m], a[i][m])
            shift[i] = np.nanmedian(s[i][m])
    return rho, shift


def design_matrix(shift: np.ndarray, is_det: np.ndarray, kind: str) -> np.ndarray:
    lx = np.log(np.clip(shift, 1e-3, None))
    if kind == "size":
        return np.column_stack([np.ones(len(lx)), lx, lx ** 2])
    if kind == "family":
        return np.column_stack([np.ones(len(lx)), is_det.astype(float)])
    raise ValueError(kind)


def oos_r2(y_tr, X_tr, y_te, X_te) -> float:
    b = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
    pred = X_te @ b
    sst = float(((y_te - y_te.mean()) ** 2).sum())
    return 1 - float(((y_te - pred) ** 2).sum()) / sst if sst > 0 else np.nan


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    log(BAR)
    log("J43  HELD-OUT BEARINGS".center(78))
    log("does the curve transfer to designs measured on bearings it was not fitted on?".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S} | seed {SEED} (its own stream) | {N_SPLITS:,} random halves")

    pt = pd.read_csv(TABLES / "j26_design_sweep_per_traj.csv")
    rng = np.random.default_rng(SEED)
    rows = []

    for ds in DATASETS:
        p = pt[pt.dataset == ds]
        designs = list(dict.fromkeys(p.design))
        trajs = list(dict.fromkeys(p.trajectory))
        d_idx = {d: i for i, d in enumerate(designs)}
        t_idx = {t: i for i, t in enumerate(trajs)}
        ALARM = np.full((len(designs), len(trajs)), np.nan)
        SHIFT = np.full((len(designs), len(trajs)), np.nan)
        NREC = np.zeros(len(trajs))
        for r in p.itertuples(index=False):
            i, j = d_idx[r.design], t_idx[r.trajectory]
            NREC[j] = r.n_records
            if r.fired:
                ALARM[i, j] = r.alarm_index
                SHIFT[i, j] = r.shift_sd
        fam = {d: p[p.design == d].family.iloc[0] for d in designs}
        IS_DET = np.array([fam[d] == "detector" for d in designs])
        nT = len(trajs)
        half = nT // 2
        log(f"\n{BAR}\n{DS_NAME[ds]}: {len(designs)} designs on {nT} bearings, split {half}/{nT - half}\n{BAR}")

        got = 0
        for _ in range(N_SPLITS):
            perm = rng.permutation(nT)
            A, B = perm[:half], perm[half:]
            rho_a, sh_a = measure(ALARM, SHIFT, NREC, A)
            rho_b, sh_b = measure(ALARM, SHIFT, NREC, B)
            ok = np.isfinite(rho_a) & np.isfinite(sh_a) & np.isfinite(rho_b) & np.isfinite(sh_b)
            if ok.sum() < MIN_DESIGNS or len(set(IS_DET[ok])) < 2:
                continue
            # how far the same design's rho agrees between the two halves: the ceiling for any predictor
            rel = float(np.corrcoef(rho_a[ok], rho_b[ok])[0, 1])
            for model in ("size", "family"):
                # fit on one half's measurements, predict the other half's, and the reverse
                r_ab = oos_r2(rho_a[ok], design_matrix(sh_a[ok], IS_DET[ok], model),
                              rho_b[ok], design_matrix(sh_b[ok], IS_DET[ok], model))
                r_ba = oos_r2(rho_b[ok], design_matrix(sh_b[ok], IS_DET[ok], model),
                              rho_a[ok], design_matrix(sh_a[ok], IS_DET[ok], model))
                rows.append({"dataset": ds, "split": got, "model": model, "n_designs": int(ok.sum()),
                             "r2_oos": float(np.nanmean([r_ab, r_ba])), "r2_a_to_b": r_ab, "r2_b_to_a": r_ba,
                             "split_half_reliability": rel})
            got += 1
        log(f"  {got:,} of {N_SPLITS:,} splits usable "
            f"(a split is dropped when fewer than {MIN_DESIGNS} designs still fire on {MIN_FIRED}+ bearings of both halves)")

    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "j43_heldout_bearings.csv", index=False)

    log(f"\n{BAR}\nout-of-sample R^2 on the held-out half of the bearings\n{BAR}")
    summ = []
    for ds in DATASETS:
        d = df[df.dataset == ds]
        log(f"\n  {DS_NAME[ds]}  ({d.split.nunique():,} splits)")
        for model in ("size", "family"):
            v = d[d.model == model].r2_oos.to_numpy(float)
            v = v[np.isfinite(v)]
            log(f"    {model:7} median {np.median(v):+.3f}   95% [{np.percentile(v, 2.5):+.3f}, "
                f"{np.percentile(v, 97.5):+.3f}]   above zero in {100 * (v > 0).mean():.1f}% of splits")
            summ.append({"dataset": ds, "model": model, "n_splits": int(len(v)),
                         "median": float(np.median(v)), "lo": float(np.percentile(v, 2.5)),
                         "hi": float(np.percentile(v, 97.5)), "share_above_zero": float((v > 0).mean()),
                         "median_designs": float(d[d.model == model].n_designs.median())})
        rel = d[d.model == "size"].split_half_reliability.to_numpy(float)
        rel = rel[np.isfinite(rel)]
        log(f"    the same design's $\\rho$ agrees between the two halves at r = {np.median(rel):+.3f} "
            f"[{np.percentile(rel, 2.5):+.3f}, {np.percentile(rel, 97.5):+.3f}], so no predictor can explain "
            f"more than about {np.median(rel) ** 2:.2f} of the held-out variation")
        a = d[d.model == "size"].set_index("split").r2_oos
        b = d[d.model == "family"].set_index("split").r2_oos
        both = pd.concat([a, b], axis=1, keys=["size", "family"]).dropna()
        log(f"    the size term predicts better than the family label in "
            f"{100 * (both['size'] > both.family).mean():.1f}% of splits")
        summ[-1]["share_size_better"] = float((both["size"] > both.family).mean())
        for s in summ[-2:]:
            s["split_half_reliability_median"] = float(np.median(rel))
            s["ceiling_r2"] = float(np.median(rel) ** 2)

    pd.DataFrame(summ).to_csv(TABLES / "j43_heldout_summary.csv", index=False)
    log(f"\nwrote j43_heldout_bearings.csv, j43_heldout_summary.csv; finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    (LOGS / "j43_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
