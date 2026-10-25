"""J35 -- what the choice of onset rule does to a fitted prognostic model, not just to the target (C3, revision).

J32 measured the distance between two remaining-useful-life TARGETS built from two onset rules: a model that
reproduced one target exactly would be charged a root-mean-square 0.433 against the other on XJTU-SY. That is a
property of the labels alone. The obvious reviewer question is whether it survives a model: once a prognostic model
is actually fitted and scored under leave-one-bearing-out cross-validation, is the variation caused by changing the
LABELLER comparable to the variation caused by changing the MODEL?

This script answers that with the two quantities side by side:

    label-induced spread   = max - min of RMSE over labellers, with the model held fixed
    model-induced spread   = max - min of RMSE over models,    with the labeller held fixed

If the first is at least as large as the second, then a reported difference between two prognostic models is not
interpretable without knowing which onset rule built their target, which is the paper's Contribution 2 carried one
step downstream. If it is smaller, that is reported too -- it bounds how much of the evaluation the labeller moves.

Three further readings are produced:
  * Kendall tau between the model RANKINGS obtained under each pair of labellers. A study picks a winner; this says
    whether the winner is a property of the models or of the labelling convention.
  * Cross-label scoring: train against labeller A's target, score against labeller B's. This is the model-level
    version of J32's target distance -- what a model that learned A is charged when the reader assumes B.
  * Per-bearing errors, so every number carries a 95% bearing-bootstrap interval on the paper's convention.

DESIGN NOTES (the choices a reviewer will check)
  * Four labellers are used, chosen to separate two kinds of change:
        3sig_rms  RMS detector, the rule most studies name
        ma5sig    a SECOND RMS detector -- changing labeller WITHIN the family
        ewma      a control chart     -- changing labeller ACROSS families
        pct05     percentile of life  -- a rule whose boundary is known by construction (control)
    The within-family pair is the point of comparison: if the label-induced spread were an artefact of fitting
    models at all, it would be as large within the family as across it. J32 found 0.070 within against 0.433
    across on the targets; this script asks whether the models inherit that asymmetry.
  * Only bearings on which ALL four labellers fire enter the comparison, so every cell is scored on identical
    bearings and identical records. The bearings dropped for this reason are logged.
  * Features are the per-record vibration statistics only. The time index t and the trajectory length n are NOT
    features: the target of Eq. (1) is a deterministic function of (t, n, i*), so giving a model t/n would let it
    reconstruct the target and would make every model look identical. This matches the leakage-free protocol used
    for the classifiers in J3.
  * No deep learning. The four models are a single-feature linear fit, ridge, gradient boosting and a random
    forest -- deliberately ordinary, because the object of study is the label set and not the model.
  * Cross-validation holds out whole bearings (leave-one-bearing-out). No record of a held-out bearing is ever
    seen in training, so nothing leaks across the bearing boundary.
  * The bootstrap resamples BEARINGS over the leave-one-bearing-out predictions, which are computed once. No model
    is refitted inside the bootstrap, so the interval reflects which bearings were tested, not fitting noise.

Inputs : results/tables/fig1_fpt_matrix.csv       onsets of the ten rules on the 33 trajectories
         results/tables/j7_features_xjtu.csv      per-record vibration features, XJTU-SY (9,216 x 7)
         colab_bundle/xjtu/records.csv            trajectory id / condition / record index, XJTU-SY
         results/tables/j7c_femto_features.csv    per-record vibration features, FEMTO (3,018 x 5)
Outputs: results/tables/j35_downstream_rul.csv    one row per dataset x labeller x model: RMSE, MAE, CI
         results/tables/j35_per_bearing.csv       one row per dataset x labeller x model x bearing
         results/tables/j35_spread.csv            label-induced against model-induced spread
         results/tables/j35_ranking_kendall.csv   Kendall tau between model rankings, every labeller pair
         results/tables/j35_cross_label.csv       train on labeller A, score against labeller B
         results/logs/j35_run.log

Run: python scripts/J35_downstream_rul.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BUNDLE = ROOT / "colab_bundle"

SEED = 0
N_BOOT = 20000            # the paper's convention for bearing bootstraps
LABELLERS = ["3sig_rms", "ma5sig", "ewma", "pct05"]
LABELLER_KIND = {"3sig_rms": "RMS detector", "ma5sig": "RMS detector (second)",
                 "ewma": "control chart", "pct05": "percentile"}
DATASETS = ["xjtu", "femto"]
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO"}

LOG: list[str] = []


def log(s: str = "") -> None:
    print(s)
    LOG.append(s)


def models(n_features: int) -> dict:
    """Six ordinary regressors. Fresh instances per call so no state is shared between folds.

    The pool is deliberately spread from a one-feature linear fit to tree ensembles, and includes k-nearest
    neighbours, which is usually the weakest of the six. A model-induced spread measured over four similar
    models would understate the comparison this script makes, so the weak model is kept in.
    """
    return {
        "lin:rms": make_pipeline(StandardScaler(), LinearRegression()),       # single feature, set up by caller
        "ridge:all": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "knn:all": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15)),
        "gbr:all": GradientBoostingRegressor(random_state=SEED),
        "et:all": ExtraTreesRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
        "rf:all": RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1),
    }


MODEL_ORDER = ["lin:rms", "ridge:all", "knn:all", "gbr:all", "et:all", "rf:all"]


def target(n: int, i) -> np.ndarray | None:
    """Eq. (1) of the paper: 1 before the onset, falling linearly to 0 at the last record."""
    if i is None or (isinstance(i, float) and np.isnan(i)):
        return None
    i = int(i)
    t = np.arange(n, dtype=float)
    y = np.ones(n)
    after = t >= i
    y[after] = (n - 1 - t[after]) / max(n - 1 - i, 1)
    return y


def load_dataset(ds: str):
    """Return (trajectory_id array, feature matrix DataFrame) with rows in trajectory/record order."""
    if ds == "femto":
        d = pd.read_csv(TABLES / "j7c_femto_features.csv").sort_values(["bearing", "record_index"])
        d = d.reset_index(drop=True)
        traj = d.bearing.to_numpy()
        F = pd.DataFrame({"rms": d.v_rms, "kurtosis": d.v_kurtosis, "crest": d.v_crest,
                          "p2p": d.v_p2p, "absmax": d.v_absmax})
        return traj, F
    rec = pd.read_csv(BUNDLE / ds / "records.csv")
    F = pd.read_csv(TABLES / f"j7_features_{ds}.csv")
    if len(rec) != len(F):
        raise SystemExit(f"{ds}: records.csv has {len(rec)} rows but the feature table has {len(F)}")
    return rec.trajectory_id.to_numpy(), F


def boot_ci(per_bearing_sq: dict, bearings: list, rng: np.random.Generator):
    """95% bearing-bootstrap interval for a pooled RMSE.

    per_bearing_sq maps bearing -> (sum of squared errors, number of records). Resampling bearings and
    recomputing the pooled RMSE keeps the record weighting that the pooled number already has.
    """
    ssq = np.array([per_bearing_sq[b][0] for b in bearings], dtype=float)
    cnt = np.array([per_bearing_sq[b][1] for b in bearings], dtype=float)
    idx = rng.integers(0, len(bearings), size=(N_BOOT, len(bearings)))
    vals = np.sqrt(ssq[idx].sum(1) / cnt[idx].sum(1))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rng = np.random.default_rng(SEED)
    LOGS.mkdir(parents=True, exist_ok=True)

    ft = pd.read_csv(TABLES / "fig1_fpt_matrix.csv")

    log("J35 -- the choice of onset rule carried into a fitted prognostic model")
    log("=" * 104)
    log(f"labellers: {', '.join(f'{r} ({LABELLER_KIND[r]})' for r in LABELLERS)}")
    log(f"models   : {', '.join(MODEL_ORDER)}")
    log(f"protocol : leave-one-bearing-out; features are per-record vibration statistics only "
        f"(no time index -- it would reconstruct the target)")
    log("")

    rows, per_bearing_rows, cross_rows = [], [], []

    for ds in DATASETS:
        traj_all, F_all = load_dataset(ds)
        g = ft[ft.dataset == ds]

        # bearings on which every labeller fires, so all cells are scored on identical records
        fired = {r: set(g[(g.rule == r) & g.fired].trajectory) for r in LABELLERS}
        usable = sorted(set.intersection(*fired.values()))
        dropped = sorted(set(g.trajectory.unique()) - set(usable))

        keep = np.isin(traj_all, usable)
        traj, F = traj_all[keep], F_all.loc[keep].reset_index(drop=True)
        n_rec = int(keep.sum())

        log(f"{DS_NAME[ds]}: {len(usable)} of {g.trajectory.nunique()} bearings usable, {n_rec} records, "
            f"{F.shape[1]} features")
        if dropped:
            why = []
            for b in dropped:
                miss = [r for r in LABELLERS if b not in fired[r]]
                why.append(f"{b} (no onset under {', '.join(miss)})")
            log(f"  dropped: {'; '.join(why)}")

        # ---- targets: one per labeller, aligned to the kept records
        onset = {(r.rule, r.trajectory): (r.fpt_index if r.fired else None) for r in g.itertuples()}
        n_of = {r.trajectory: int(r.n_records) for r in g.itertuples()}
        Y = {}
        for lab in LABELLERS:
            parts = []
            for b in usable:
                y = target(n_of[b], onset[(lab, b)])
                if y is None:
                    raise SystemExit(f"{ds}/{lab}/{b}: no onset although the bearing was marked usable")
                parts.append(y)
            y_full = np.concatenate(parts)
            if len(y_full) != n_rec:
                raise SystemExit(f"{ds}/{lab}: target has {len(y_full)} records, features have {n_rec}")
            Y[lab] = y_full

        # ---- leave-one-bearing-out predictions, once per labeller x model
        X_all = F.to_numpy(dtype=float)
        X_rms = F[["rms"]].to_numpy(dtype=float)
        pred = {}
        for lab in LABELLERS:
            y = Y[lab]
            for mname in MODEL_ORDER:
                p = np.full(n_rec, np.nan)
                for b in usable:
                    te = traj == b
                    tr = ~te
                    X = X_rms if mname == "lin:rms" else X_all
                    m = models(X.shape[1])[mname]
                    m.fit(X[tr], y[tr])
                    p[te] = m.predict(X[te])
                if np.isnan(p).any():
                    raise SystemExit(f"{ds}/{lab}/{mname}: some records never held out")
                pred[(lab, mname)] = np.clip(p, 0.0, 1.0)

        # ---- a constant baseline under the same protocol, so RMSE can be read on a scale-free axis.
        # Different labellers give targets of very different difficulty: pct05 leaves y = 1 on about 95% of
        # records, so ANY predictor scores a small RMSE against it. Comparing raw RMSE across labellers would
        # therefore confound "the labeller changed what the model learned" with "the labeller changed how much
        # there is to get wrong". The baseline predicts the mean of y over the training bearings only, and the
        # skill score 1 - MSE_model / MSE_baseline removes that difference in target variance.
        base = {}
        for lab in LABELLERS:
            y = Y[lab]
            p = np.full(n_rec, np.nan)
            for b in usable:
                te = traj == b
                p[te] = y[~te].mean()
            base[lab] = p

        # ---- score each (labeller, model) against its OWN target
        for lab in LABELLERS:
            mse_base = float(((base[lab] - Y[lab]) ** 2).mean())
            for mname in MODEL_ORDER:
                p, y = pred[(lab, mname)], Y[lab]
                err = p - y
                pb = {}
                for b in usable:
                    te = traj == b
                    e = err[te]
                    pb[b] = (float((e ** 2).sum()), int(te.sum()))
                    per_bearing_rows.append({"dataset": ds, "labeller": lab, "model": mname, "bearing": b,
                                             "n_records": int(te.sum()),
                                             "rmse": float(np.sqrt((e ** 2).mean())),
                                             "mae": float(np.abs(e).mean())})
                rmse = float(np.sqrt((err ** 2).mean()))
                lo, hi = boot_ci(pb, usable, rng)
                rows.append({"dataset": ds, "labeller": lab, "labeller_kind": LABELLER_KIND[lab],
                             "model": mname, "n_bearings": len(usable), "n_records": n_rec,
                             "rmse": rmse, "rmse_lo": lo, "rmse_hi": hi,
                             "mae": float(np.abs(err).mean()),
                             "rmse_baseline": float(np.sqrt(mse_base)),
                             "skill": float(1.0 - (rmse ** 2) / mse_base),
                             "nrmse": float(rmse / Y[lab].std()),
                             "target_sd": float(Y[lab].std()),
                             "target_positive_share": float((y < 1.0).mean())})

        # ---- cross-label scoring: a model trained against A, read by someone who assumes B
        for a, b_ in itertools.permutations(LABELLERS, 2):
            for mname in MODEL_ORDER:
                e = pred[(a, mname)] - Y[b_]
                own = pred[(b_, mname)] - Y[b_]
                cross_rows.append({"dataset": ds, "trained_on": a, "scored_against": b_, "model": mname,
                                   "rmse_cross": float(np.sqrt((e ** 2).mean())),
                                   "rmse_own": float(np.sqrt((own ** 2).mean())),
                                   "penalty": float(np.sqrt((e ** 2).mean()) - np.sqrt((own ** 2).mean()))})
        log("")

    res = pd.DataFrame(rows)
    res.to_csv(TABLES / "j35_downstream_rul.csv", index=False)
    pd.DataFrame(per_bearing_rows).to_csv(TABLES / "j35_per_bearing.csv", index=False)
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(TABLES / "j35_cross_label.csv", index=False)

    # ---- the headline comparison: label-induced against model-induced spread
    spread_rows = []
    for ds in DATASETS:
        d = res[res.dataset == ds]
        for metric in ("rmse", "skill"):
            for mname in MODEL_ORDER:                   # model fixed, labeller varies
                v = d[d.model == mname][metric]
                spread_rows.append({"dataset": ds, "metric": metric, "held_fixed": "model", "name": mname,
                                    "n_varied": len(v), "min": float(v.min()), "max": float(v.max()),
                                    "spread": float(v.max() - v.min())})
            for lab in LABELLERS:                       # labeller fixed, model varies
                v = d[d.labeller == lab][metric]
                spread_rows.append({"dataset": ds, "metric": metric, "held_fixed": "labeller", "name": lab,
                                    "n_varied": len(v), "min": float(v.min()), "max": float(v.max()),
                                    "spread": float(v.max() - v.min())})
            # the within-family control: 3sig_rms against ma5sig, same family, both RMS detectors
            for mname in MODEL_ORDER:
                v = d[(d.model == mname) & (d.labeller.isin(["3sig_rms", "ma5sig"]))][metric]
                spread_rows.append({"dataset": ds, "metric": metric,
                                    "held_fixed": "model (within RMS detectors)", "name": mname,
                                    "n_varied": len(v), "min": float(v.min()), "max": float(v.max()),
                                    "spread": float(v.max() - v.min())})
    spread = pd.DataFrame(spread_rows)
    spread.to_csv(TABLES / "j35_spread.csv", index=False)

    # ---- Kendall tau between model rankings under each pair of labellers
    tau_rows = []
    for ds in DATASETS:
        d = res[res.dataset == ds]
        for a, b_ in itertools.combinations(LABELLERS, 2):
            ra = d[d.labeller == a].set_index("model").loc[MODEL_ORDER, "rmse"].to_numpy()
            rb = d[d.labeller == b_].set_index("model").loc[MODEL_ORDER, "rmse"].to_numpy()
            tau, p = kendalltau(ra, rb)
            tau_rows.append({"dataset": ds, "labeller_a": a, "labeller_b": b_,
                             "winner_a": MODEL_ORDER[int(np.argmin(ra))],
                             "winner_b": MODEL_ORDER[int(np.argmin(rb))],
                             "same_winner": MODEL_ORDER[int(np.argmin(ra))] == MODEL_ORDER[int(np.argmin(rb))],
                             "kendall_tau": float(tau), "p_value": float(p)})
    tau = pd.DataFrame(tau_rows)
    tau.to_csv(TABLES / "j35_ranking_kendall.csv", index=False)

    # ------------------------------------------------------------------ report
    for ds in DATASETS:
        d = res[res.dataset == ds]
        log("=" * 104)
        log(f"{DS_NAME[ds]} -- RMSE of the predicted remaining-useful-life target, leave-one-bearing-out")
        log(f"{'model':12s} " + "".join(f"{lab:>21s}" for lab in LABELLERS))
        for mname in MODEL_ORDER:
            cells = []
            for lab in LABELLERS:
                r = d[(d.model == mname) & (d.labeller == lab)].iloc[0]
                cells.append(f"{r.rmse:.3f} [{r.rmse_lo:.2f},{r.rmse_hi:.2f}]".rjust(21))
            log(f"{mname:12s} " + "".join(cells))

        log("")
        log(f"  target difficulty differs by labeller, so both the raw and the scale-free axis are shown:")
        for lab in LABELLERS:
            r = d[d.labeller == lab].iloc[0]
            log(f"    {lab:10s} target sd {r.target_sd:.3f}, records with y<1 {r.target_positive_share:.3f}, "
                f"constant-baseline RMSE {r.rmse_baseline:.3f}")
        log("")
        for metric, unit in (("rmse", "RMSE"), ("skill", "skill score")):
            s = spread[(spread.dataset == ds) & (spread.metric == metric)]
            lab_sp = s[s.held_fixed == "model"].spread
            mod_sp = s[s.held_fixed == "labeller"].spread
            win_sp = s[s.held_fixed == "model (within RMS detectors)"].spread
            log(f"  [{unit}]")
            log(f"    label-induced spread (model fixed, {len(LABELLERS)} labellers) : "
                f"median {lab_sp.median():.3f}  range {lab_sp.min():.3f}-{lab_sp.max():.3f}")
            log(f"    model-induced spread (labeller fixed, {len(MODEL_ORDER)} models) : "
                f"median {mod_sp.median():.3f}  range {mod_sp.min():.3f}-{mod_sp.max():.3f}")
            log(f"    within-family control (3sig_rms vs ma5sig only)  : "
                f"median {win_sp.median():.3f}  range {win_sp.min():.3f}-{win_sp.max():.3f}")
            ratio = lab_sp.median() / mod_sp.median() if mod_sp.median() > 0 else float("inf")
            log(f"    -> changing the labeller moves {unit} {ratio:.2f} times as much as changing the model")
        log("")
        log(f"  skill score of every cell (1 - MSE_model / MSE_constant-baseline, same protocol):")
        log(f"  {'model':12s} " + "".join(f"{lab:>12s}" for lab in LABELLERS))
        for mname in MODEL_ORDER:
            cells = [f"{d[(d.model == mname) & (d.labeller == lab)].iloc[0].skill:12.3f}" for lab in LABELLERS]
            log(f"  {mname:12s} " + "".join(cells))

        t = tau[tau.dataset == ds]
        log(f"  model rankings: same winner in {int(t.same_winner.sum())} of {len(t)} labeller pairs, "
            f"median Kendall tau {t.kendall_tau.median():+.2f}")
        cd = cross[cross.dataset == ds]
        worst = cd.loc[cd.penalty.idxmax()]
        log(f"  cross-label penalty: worst is {worst.model} trained on {worst.trained_on} and scored against "
            f"{worst.scored_against}, RMSE {worst.rmse_own:.3f} -> {worst.rmse_cross:.3f} "
            f"(+{worst.penalty:.3f})")
        log("")

    log("=" * 104)
    log("written: j35_downstream_rul.csv, j35_per_bearing.csv, j35_spread.csv, j35_ranking_kendall.csv, "
        "j35_cross_label.csv")
    (LOGS / "j35_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
