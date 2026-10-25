"""J42 -- is the labeller-versus-model comparison an artefact of the two pools being different sizes?

Section V-F compares two spreads: the range of the prediction error over four labellers, and its range
over six models. A range grows with the number of levels it is taken over, so a ratio of two ranges taken
over pools of different size is not, on its own, a comparison of two effects. This script recomputes the
comparison under statistics that do not depend on the pool size, and reports them beside the range.

  range                  max - min, what Section V-F reports (J38)
  range / d2(n)          the same, divided by the expected range of n draws from a standard normal
                         (d2 = 2.059 for four levels, 2.534 for six): the textbook correction
  sd of main effects     the standard deviation of the labeller effects against that of the model effects
  mad of main effects    the same with the mean absolute deviation, which is less sensitive to one level
  variance component     sigma_L and sigma_M from the two-way decomposition of the bearing-averaged
                         matrix, with the interaction as the error term

The bearing bootstrap resamples whole bearings, as everywhere else in the paper, with its own generator:
J38 and J36 each run one stream, and drawing from theirs would move numbers already published.

Inputs : results/tables/j35_per_bearing.csv   one RMSE and record count per dataset x labeller x model x bearing
Outputs: results/tables/j42_dispersion.csv    one row per dataset x statistic, with a 95% interval
         results/logs/j42_run.log

Usage: python scripts/J42_dispersion_statistics.py [N_BOOT]        (default 20000, as in J38)
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"

SEED = 42
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
DATASETS = ["xjtu", "femto"]
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO"}
BAR = "=" * 78

# E[range of n independent standard normal draws]; the control-chart d2 constant
D2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847}

LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def pooled(cube: np.ndarray, nrec: np.ndarray) -> np.ndarray:
    """The RMSE of each labeller x model cell over the records of the bearings, as Section V-F pools it."""
    w = nrec / nrec.sum()
    return np.sqrt((cube ** 2 * w).sum(axis=2))


def statistics(Y: np.ndarray) -> dict[str, tuple[float, float]]:
    """Labeller and model dispersion of the bearing-averaged matrix Y[labeller, model]."""
    nL, nM = Y.shape
    grand = Y.mean()
    alpha = Y.mean(axis=1) - grand          # labeller main effects
    beta = Y.mean(axis=0) - grand           # model main effects

    # the statistic Section V-F uses: the median over one axis of the range across the other
    rng_lab = float(np.median(Y.max(axis=0) - Y.min(axis=0)))   # per model, across labellers
    rng_mod = float(np.median(Y.max(axis=1) - Y.min(axis=1)))   # per labeller, across models

    # the two-way decomposition of the same matrix, interaction as the error term
    resid = Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + grand
    ms_e = float((resid ** 2).sum() / ((nL - 1) * (nM - 1)))
    ms_l = float(nM * (alpha ** 2).sum() / (nL - 1))
    ms_m = float(nL * (beta ** 2).sum() / (nM - 1))
    var_l = max((ms_l - ms_e) / nM, 0.0)
    var_m = max((ms_m - ms_e) / nL, 0.0)

    return {
        "range": (rng_lab, rng_mod),
        "range_over_d2": (rng_lab / D2[nL], rng_mod / D2[nM]),
        "sd_main_effects": (float(alpha.std(ddof=1)), float(beta.std(ddof=1))),
        "mad_main_effects": (float(np.abs(alpha).mean()), float(np.abs(beta).mean())),
        "variance_component": (float(np.sqrt(var_l)), float(np.sqrt(var_m))),
    }


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    log(BAR)
    log("J42  POOL-SIZE-INVARIANT DISPERSION".center(78))
    log("does the labeller-versus-model comparison survive a statistic that does not grow with the pool?".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S} | seed {SEED} (its own stream) | {N_BOOT:,} resamples")

    pb = pd.read_csv(TABLES / "j35_per_bearing.csv")
    rng = np.random.default_rng(SEED)
    rows = []

    for ds in DATASETS:
        d = pb[pb.dataset == ds]
        labs = sorted(d.labeller.unique())
        mods = sorted(d.model.unique())
        bears = sorted(d.bearing.unique())
        cube = np.full((len(labs), len(mods), len(bears)), np.nan)
        nrec = np.zeros(len(bears))
        idx = {("l", v): i for i, v in enumerate(labs)} | {("m", v): i for i, v in enumerate(mods)} \
            | {("b", v): i for i, v in enumerate(bears)}
        for r in d.itertuples(index=False):
            cube[idx[("l", r.labeller)], idx[("m", r.model)], idx[("b", r.bearing)]] = r.rmse
            nrec[idx[("b", r.bearing)]] = r.n_records
        if np.isnan(cube).any():
            raise SystemExit(f"{ds}: the labeller x model x bearing layout is not complete")

        log(f"\n{BAR}\n{DS_NAME[ds]}: {len(labs)} labellers x {len(mods)} models x {len(bears)} bearings\n{BAR}")
        log(f"  the range penalises the smaller pool: E[range] is {D2[len(labs)]:.3f} sd over {len(labs)} levels "
            f"and {D2[len(mods)]:.3f} over {len(mods)}, so a ratio of raw ranges understates the labeller axis "
            f"by a factor of {D2[len(mods)] / D2[len(labs)]:.2f} when the two effects are equal.")

        point = statistics(pooled(cube, nrec))
        boot: dict[str, list[float]] = {k: [] for k in point}
        for _ in range(N_BOOT):
            sel = rng.integers(0, len(bears), len(bears))
            st = statistics(pooled(cube[:, :, sel], nrec[sel]))
            for k, (a, b) in st.items():
                boot[k].append(a / b if b > 0 else np.nan)

        log(f"\n  {'statistic':22} {'labeller':>9} {'model':>9} {'ratio':>7}   95% interval over resamples")
        for k, (a, b) in point.items():
            v = np.array([x for x in boot[k] if np.isfinite(x)], dtype=float)
            lo, hi = (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)
            ratio = a / b if b > 0 else np.nan
            shown = f"{ratio:7.1f}" if np.isfinite(ratio) else "      -"
            note = "" if np.isfinite(ratio) else "   (the model component is not separable from the interaction)"
            log(f"  {k:22} {a:9.3f} {b:9.3f} {shown}   [{lo:.1f}, {hi:.1f}]   "
                f"(median {np.median(v):.1f}, larger in {100 * (v > 1).mean():.1f}% of resamples){note}")
            rows.append({"dataset": ds, "statistic": k, "n_labellers": len(labs), "n_models": len(mods),
                         "n_bearings": len(bears), "labeller": a, "model": b, "ratio": ratio,
                         "ratio_boot_median": float(np.median(v)), "ratio_lo": float(lo), "ratio_hi": float(hi),
                         "share_ratio_gt_1": float((v > 1).mean()), "n_boot": int(len(v))})

    pd.DataFrame(rows).to_csv(TABLES / "j42_dispersion.csv", index=False)
    log(f"\nwrote j42_dispersion.csv; finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    (LOGS / "j42_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
