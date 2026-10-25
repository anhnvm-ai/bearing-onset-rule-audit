"""J48 -- the pool-size-corrected dispersion of J42, read on the pool EXP-4 extended (J47).

J42 asked whether the labeller-versus-model comparison of Section V-F survives a statistic that does not
grow with the pool, and answered it on the six regressors of J35. J47 then widened the pool to nine
models -- the six regressors, the two-stage fit of J39 and the GRU and TCN of J47 -- and the raw ratio of
ranges fell, on FEMTO to below one on the skill scale. A raw range grows with the number of levels it is
taken over, so that fall is exactly the artefact J42 exists to remove, and the question it answers has to
be asked again of the wider pool rather than settled by the number that moved.

Nothing here is new statistics. `statistics()` and `pooled()` are imported from J42 so that there is one
implementation, and the six-model pool is recomputed alongside the nine-model one so that the two rows
can be read against each other and against what J42 already published.

  pool "six"    the six regressors of J35, which reproduces j42_dispersion.csv exactly
  pool "nine"   those, plus the two-stage exponential fit of J39 (variant `exp`) and the GRU and TCN of
                J47 at their median seed, in the main arm -- the same cells Table S12 reports

The bootstrap resamples whole bearings, as everywhere else in the paper, from its own stream (seed 48):
J36, J38 and J42 each run one, and drawing from theirs would move numbers already published.

Inputs : results/tables/j35_per_bearing.csv          six regressors
         results/tables/j39_two_stage_per_bearing.csv  the two-stage fit
         results/tables/j47_per_bearing.csv          the two sequence models
Outputs: results/tables/j48_dispersion_extended.csv  one row per dataset x pool x statistic
         results/logs/j48_run.log

Usage: python scripts/J48_pool_extended_dispersion.py [N_BOOT]      (default 20000, as J38 and J42)
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES, LOGS = ROOT / "results" / "tables", ROOT / "results" / "logs"
SEED = 48
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
DATASETS = ["xjtu", "femto"]
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO"}
BAR = "=" * 78
LOG: list[str] = []


def log(s: str = "") -> None:
    print(s)
    LOG.append(s)


def load_j42():
    spec = importlib.util.spec_from_file_location("j42", ROOT / "scripts" / "J42_dispersion_statistics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def frames() -> pd.DataFrame:
    """One long table, dataset x labeller x model x bearing, over all nine models."""
    six = pd.read_csv(TABLES / "j35_per_bearing.csv")[["dataset", "labeller", "model", "bearing",
                                                       "n_records", "rmse"]]
    ts = pd.read_csv(TABLES / "j39_two_stage_per_bearing.csv").query("variant == 'exp'").copy()
    ts["model"] = "two_stage:exp"
    ts = ts[["dataset", "labeller", "model", "bearing", "n_records", "rmse"]]
    seq = pd.read_csv(TABLES / "j47_per_bearing.csv").query("arm == 'main'").copy()
    seq = seq[["dataset", "labeller", "model", "bearing", "n_records", "rmse"]]
    return pd.concat([six, ts, seq], ignore_index=True)


def cube_of(d: pd.DataFrame, models: list[str]):
    labs = sorted(d.labeller.unique())
    bears = sorted(d.bearing.unique())
    cube = np.full((len(labs), len(models), len(bears)), np.nan)
    nrec = np.zeros(len(bears))
    li = {v: i for i, v in enumerate(labs)}
    mi = {v: i for i, v in enumerate(models)}
    bi = {v: i for i, v in enumerate(bears)}
    for r in d[d.model.isin(models)].itertuples(index=False):
        cube[li[r.labeller], mi[r.model], bi[r.bearing]] = r.rmse
        nrec[bi[r.bearing]] = r.n_records
    if np.isnan(cube).any():
        missing = [(labs[i], models[j], bears[k]) for i, j, k in zip(*np.where(np.isnan(cube)))][:5]
        raise SystemExit(f"the layout is not complete; first gaps: {missing}")
    return labs, bears, cube, nrec


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    LOGS.mkdir(parents=True, exist_ok=True)
    J42 = load_j42()
    D2 = dict(J42.D2)
    D2.setdefault(9, 2.970)          # E[range of nine standard normal draws]
    log(BAR)
    log("J48  POOL-SIZE-CORRECTED DISPERSION ON THE EXTENDED POOL".center(78))
    log("the raw ratio falls when models are added; does the corrected one?".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S} | seed {SEED} (its own stream) | {N_BOOT:,} resamples")

    long = frames()
    j42 = pd.read_csv(TABLES / "j42_dispersion.csv")
    rng = np.random.default_rng(SEED)
    rows = []

    for ds in DATASETS:
        d = long[long.dataset == ds]
        six = sorted(m for m in d.model.unique() if ":" in m and not m.startswith("two_stage"))
        nine = six + ["two_stage:exp", "gru", "tcn"]
        for tag, models in (("six", six), ("nine", nine)):
            labs, bears, cube, nrec = cube_of(d, models)
            log(f"\n{BAR}\n{DS_NAME[ds]}, pool '{tag}': {len(labs)} labellers x {len(models)} models x "
                f"{len(bears)} bearings\n{BAR}")
            log(f"  E[range] is {D2[len(labs)]:.3f} sd over {len(labs)} levels and {D2[len(models)]:.3f} over "
                f"{len(models)}, so a ratio of raw ranges understates the labeller axis by a factor of "
                f"{D2[len(models)] / D2[len(labs)]:.2f} when the two effects are equal")
            J42.D2 = D2                      # the nine-level constant, for this call
            point = J42.statistics(J42.pooled(cube, nrec))
            boot: dict[str, list[float]] = {k: [] for k in point}
            for _ in range(N_BOOT):
                sel = rng.integers(0, len(bears), len(bears))
                st = J42.statistics(J42.pooled(cube[:, :, sel], nrec[sel]))
                for k, (a, b) in st.items():
                    boot[k].append(a / b if b > 0 else np.nan)

            log(f"\n  {'statistic':22} {'labeller':>9} {'model':>9} {'ratio':>7}   95% interval over resamples")
            for k, (a, b) in point.items():
                v = np.array([x for x in boot[k] if np.isfinite(x)], dtype=float)
                lo, hi = (np.percentile(v, 2.5), np.percentile(v, 97.5)) if len(v) else (np.nan, np.nan)
                ratio = a / b if b > 0 else np.nan
                shown = f"{ratio:7.2f}" if np.isfinite(ratio) else "      -"
                log(f"  {k:22} {a:9.3f} {b:9.3f} {shown}   [{lo:.2f}, {hi:.2f}]   "
                    f"(larger in {100 * (v > 1).mean():.1f}% of resamples)")
                row = {"dataset": ds, "pool": tag, "statistic": k, "n_labellers": len(labs),
                       "n_models": len(models), "n_bearings": len(bears), "labeller": a, "model": b,
                       "ratio": ratio, "ratio_boot_median": float(np.median(v)), "ratio_lo": float(lo),
                       "ratio_hi": float(hi), "share_ratio_gt_1": float((v > 1).mean()), "n_boot": int(len(v))}
                if tag == "six":
                    ref = j42[(j42.dataset == ds) & (j42.statistic == k)].iloc[0]
                    if abs(row["ratio"] - ref.ratio) > 5e-9 and np.isfinite(ratio):
                        raise SystemExit(f"{ds}/{k}: the six-model pool gives {row['ratio']:.6f} but J42 "
                                         f"published {ref.ratio:.6f}")
                rows.append(row)
        log(f"\n  the six-model rows reproduce j42_dispersion.csv exactly, so the two pools are like for like")

    pd.DataFrame(rows).to_csv(TABLES / "j48_dispersion_extended.csv", index=False)
    log(f"\nwrote j48_dispersion_extended.csv; finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    (LOGS / "j48_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
