"""J69 -- Amendment A6: size of the labeler effect against the size of the model effect on the common reference.

Gate 1 (2026-10-25) approved abstract phrase 3A: on the common retrospective RUL reference, the model changes how well
remaining life is ranked more than the labeler does. The pre-specified plan has no analysis for that comparison, so
amendment A6 (paper/access_resub/ANALYSIS_PLAN_AMENDMENTS.md) adds one, labeled post hoc:

  cell value     C-rank: mean over bearings of the concordance on the scored records; C-common: RMSE pooled over the
                 scored records of all bearings (both exactly as in J62)
  main effects   a labeler: its cell values averaged over the models; a model: averaged over the labelers
  size           mean absolute difference between the main effects of two levels, over all pairs (Gini mean
                 difference); unlike a range or a population SD, its expectation does not depend on the number of levels
  statistic      R = size(labeler) / size(model); R < 1 means that the model matters more
  pools          (i) the nine models; (ii) without the two-stage model; (iii) leave-one-model-out, largest R
  inference      95% percentile intervals from B = 20,000 bearing resamples stratified by condition (J60 strat_index),
                 seed 69, one resample set per configuration x dataset, shared by S, S3, both metrics and both pools
  decision rule  phrase 3A keeps its model-versus-labeler clause only if, for C-rank, the upper bound of R is below 1
                 in all 16 cells (4 configurations x 2 datasets x S and S3), for pools (i) and (ii); no abstract claim
                 is made from C-common RMSE whatever the result

Self-checks before anything is written: the labeler main-effect differences reproduce diff_mean_over_models of
j62_main_effect.csv within 1e-9; the bearing counts match j62_common_reference.csv; the pooled values of the
deterministic models (and the C-common RMSE on S of GRU and TCN, whose median seed is chosen by it) reproduce
j62_common_reference.csv within 1e-9.

Inputs : results/tables/j62_per_bearing.csv, j62_main_effect.csv, j62_common_reference.csv
Outputs: results/tables/j69_labeler_model_contrast.csv, results/logs/j69_run.log ("_quick" suffix when B != 20,000)
Usage  : python scripts/J69_labeler_model_contrast.py
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
SEED = 69
NINE = ["lin:rms", "ridge:all", "knn:all", "gbr:all", "et:all", "rf:all", "two-stage", "gru", "tcn"]
SEQ = ["gru", "tcn"]
LABELERS = ["3sig_rms", "ma5sig", "ewma", "pct05"]
SIGNAL_LABS = ["3sig_rms", "ma5sig", "ewma"]
SETS = {"S": (LABELERS, "sse_com_S", "n_S", "conc_S"), "S3": (SIGNAL_LABS, "sse_com_S3", "n_S3", "conc_S3")}
REF_COL = {("S", "C-common RMSE"): "rmse_common_S", ("S3", "C-common RMSE"): "rmse_common_S3",
           ("S", "C-rank concordance"): "conc_S_mean", ("S3", "C-rank concordance"): "conc_S3_mean"}
POOLS = {"nine": NINE, "no two-stage": [m for m in NINE if m != "two-stage"]}
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO"}
TOL = 1e-9
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


def gini_md(v: np.ndarray) -> np.ndarray:
    """Mean absolute difference over all pairs of levels (levels on axis 0, any trailing axes)."""
    i, j = np.triu_indices(v.shape[0], 1)
    return np.abs(v[i] - v[j]).mean(axis=0)


def cell_values(SSE: np.ndarray, CNT: np.ndarray, CON: np.ndarray, idx: np.ndarray | None = None) -> dict:
    """Labeler x model (x resample) tables of J62: pooled C-common RMSE and mean C-rank concordance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if idx is None:
            return {"C-common RMSE": np.sqrt(SSE.sum(-1) / CNT.sum()), "C-rank concordance": np.nanmean(CON, axis=-1)}
        return {"C-common RMSE": np.sqrt(SSE[..., idx].sum(-1) / CNT[idx].sum(-1)),
                "C-rank concordance": np.nanmean(CON[..., idx], axis=-1)}


def ratio(tab: np.ndarray, cols: list[int]) -> np.ndarray:
    """R = size(labeler) / size(model) for a labeler x model (x B) table restricted to the given model columns."""
    t = tab[:, cols]
    return gini_md(t.mean(axis=1)) / gini_md(t.mean(axis=0))


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    j60 = load("j60", "J60_baseline_protocols.py")
    B = j60.B
    suffix = "" if B == 20000 else "_quick"
    _fh = open(LOGS / f"j69_run{suffix}.log", "w", encoding="utf-8")
    log(BAR)
    log("J69  LABELER EFFECT AGAINST MODEL EFFECT ON THE COMMON REFERENCE  (A6)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}; B = {B} bearing resamples stratified by condition, seed {SEED}")
    if B != 20000:
        log("  note: B differs from the 20,000 of amendment A6; this run is not the reported analysis")

    pb = pd.read_csv(T / "j62_per_bearing.csv")
    pb = pb[(pb.variant == "a") & pb.model.isin(NINE)]
    me = pd.read_csv(T / "j62_main_effect.csv")
    cr = pd.read_csv(T / "j62_common_reference.csv")
    cr = cr[cr.variant == "a"]

    rng = np.random.default_rng(SEED)
    rows, worst_me, worst_cr, bad_counts = [], 0.0, 0.0, []
    for (cfg, ds), g in pb.groupby(["config", "dataset"], sort=False):
        U = list(dict.fromkeys(g.trajectory))                       # J62's bearing order
        n_ref = int(cr[(cr.config == cfg) & (cr.dataset == ds)].n_bearings.iloc[0])
        if len(U) != n_ref:
            bad_counts.append(f"{cfg} {ds}: {len(U)} bearings against {n_ref}")
        cond = g.drop_duplicates("trajectory").set_index("trajectory").loc[U, "condition"].to_numpy()
        idx = j60.strat_index(cond, rng)                            # shared by S, S3, both metrics and both pools
        cell = g.set_index(["labeler", "model", "trajectory"]).sort_index()
        log(f"\n{BAR}\n{cfg}  {DS_NAME[ds]}  ({len(U)} bearings)\n{BAR}")
        for set_name, (labs, sse_k, n_k, conc_k) in SETS.items():
            SSE = np.array([[[cell.loc[(l, m, b), sse_k] for b in U] for m in NINE] for l in labs], float)
            CON = np.array([[[cell.loc[(l, m, b), conc_k] for b in U] for m in NINE] for l in labs], float)
            N = np.array([[[cell.loc[(l, m, b), n_k] for b in U] for m in NINE] for l in labs], float)
            if not (N == N[0, 0]).all():
                log(f"  FAIL: the scored record counts of {set_name} differ across labelers or models")
                return 1
            CNT = N[0, 0]
            pt, bt = cell_values(SSE, CNT, CON), cell_values(SSE, CNT, CON, idx)
            for metric in pt:
                # self-check 1: the labeler main-effect differences of J62
                mean_l = pt[metric].mean(axis=1)
                for ia, ib in itertools.combinations(range(len(labs)), 2):
                    r = me[(me.config == cfg) & (me.dataset == ds) & (me.set == set_name) & (me.metric == metric)
                           & (me.labeler_a == labs[ia]) & (me.labeler_b == labs[ib])]
                    worst_me = max(worst_me, abs(float(mean_l[ia] - mean_l[ib]) - float(r.diff_mean_over_models.iloc[0])))
                # self-check 2: pooled cell values against j62_common_reference
                for li, lab in enumerate(labs):
                    for mi, model in enumerate(NINE):
                        if model in SEQ and (set_name, metric) != ("S", "C-common RMSE"):
                            continue                                 # median seed chosen by C-common RMSE on S only
                        ref = cr[(cr.config == cfg) & (cr.dataset == ds) & (cr.labeler == lab) & (cr.model == model)]
                        worst_cr = max(worst_cr, abs(float(pt[metric][li, mi]) - float(ref[REF_COL[(set_name, metric)]].iloc[0])))
                # the contrast
                for pool, members in POOLS.items():
                    cols = [NINE.index(m) for m in members]
                    t_pt = pt[metric][:, cols]
                    s_lab, s_mod = float(gini_md(t_pt.mean(axis=1))), float(gini_md(t_pt.mean(axis=0)))
                    rb = ratio(bt[metric], cols)
                    ok = np.isfinite(rb)
                    lo, hi = np.percentile(rb[ok], [2.5, 97.5])
                    loo_max, loo_worst = np.nan, ""
                    if pool == "nine":
                        loo = {m: float(ratio(pt[metric], [NINE.index(x) for x in NINE if x != m])) for m in NINE}
                        loo_worst = max(loo, key=loo.get)
                        loo_max = loo[loo_worst]
                    rows.append({"config": cfg, "dataset": ds, "set": set_name, "metric": metric, "pool": pool,
                                 "n_bearings": len(U), "n_labelers": len(labs), "n_models": len(members),
                                 "size_labeler": s_lab, "size_model": s_mod, "ratio": s_lab / s_mod,
                                 "lo": float(lo), "hi": float(hi), "share_below_1": float((rb[ok] < 1).mean()),
                                 "n_nonfinite": int((~ok).sum()), "loo_max_ratio": loo_max, "loo_worst_drop": loo_worst})
                    log(f"  {set_name:2s} {metric:18s} {pool:12s}: labeler {s_lab:.4f}, model {s_mod:.4f}, "
                        f"R {s_lab / s_mod:.3f} [{lo:.3f}, {hi:.3f}], R < 1 in {(rb[ok] < 1).mean():.3f} of resamples"
                        + (f"; leave-one-out max {loo_max:.3f} (drop {loo_worst})" if pool == "nine" else ""))

    log(f"\n{BAR}\nSelf-checks\n{BAR}")
    log(f"  labeler main-effect differences against j62_main_effect.csv: largest |difference| {worst_me:.1e}")
    log(f"  pooled cell values against j62_common_reference.csv: largest |difference| {worst_cr:.1e}")
    log(f"  bearing counts against j62_common_reference.csv: {'all match' if not bad_counts else '; '.join(bad_counts)}")
    if worst_me > TOL or worst_cr > TOL or bad_counts:
        log("  FAIL: J69 does not reproduce J62; nothing written")
        return 1

    res = pd.DataFrame(rows)
    log(f"\n{BAR}\nDecision rule of A6 (C-rank, all 16 cells, pools 'nine' and 'no two-stage')\n{BAR}")
    rank = res[res.metric == "C-rank concordance"]
    fail = rank[rank.hi >= 1]
    for pool in POOLS:
        r = rank[rank.pool == pool]
        log(f"  {pool:12s}: upper bound below 1 in {int((r.hi < 1).sum())}/{len(r)} cells; "
            f"R from {r.ratio.min():.3f} to {r.ratio.max():.3f}; largest upper bound {r.hi.max():.3f}")
    if fail.empty:
        log("  -> the clause 'the model changes how well remaining life is ranked more than the labeler does' STAYS")
    else:
        log("  -> the clause is REMOVED from the abstract; cells with an upper bound of 1 or more:")
        for _, r in fail.iterrows():
            log(f"     {r.config} {DS_NAME[r.dataset]} {r.set} {r.pool}: R {r.ratio:.3f} [{r.lo:.3f}, {r.hi:.3f}]")
    com = res[res.metric == "C-common RMSE"]
    log(f"\n  C-common RMSE (reported, no abstract claim):")
    for (st, pool), r in com.groupby(["set", "pool"], sort=False):
        log(f"    {st:2s} {pool:12s}: R below 1 with upper bound below 1 in {int((r.hi < 1).sum())}/{len(r)} cells; "
            f"R above 1 with lower bound above 1 in {int((r.lo > 1).sum())}/{len(r)}; R from {r.ratio.min():.3f} to {r.ratio.max():.3f}")

    res.to_csv(T / f"j69_labeler_model_contrast{suffix}.csv", index=False)
    log(f"\nwrote j69_labeler_model_contrast{suffix}.csv ({len(res)} rows); finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
