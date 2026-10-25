"""J7 (v2) -- Label-input circularity, measured per TRAJECTORY rather than per dataset.

Why v2 exists
-------------
v1 printed "TRIANGLE HOLDS". That verdict was wrong, for three compounding reasons, all
of them design faults in v1 rather than facts about the data:

  1. VERDICT LOGIC. v1 took a median GAP over the "signal-based" group whose members
     disagreed: XJTU at 5%/10% (+0.079, +0.086) behaved like IMS (+0.097, +0.070), not
     like NSK (-0.064). NSK dragged the group median negative and hid that H2 had failed.

  2. WRONG FEATURE FOR THE CRITERION. XJTU stops at "10x the normal maximum OF THAT
     BEARING" -- a RELATIVE criterion. v1 tested absolute RMS thresholds transferred
     across 15 bearings with different baselines. Normalising each trajectory by its own
     early-life baseline moves last-5% macro-F1 from 0.5498 to 0.6868.

  3. FATAL: 8 OF 12 IMS TRAJECTORIES NEVER FAILED. Each IMS test runs four bearings but
     only one or two fail; the NASA readme documents failures only in 1st_test B3 and B4,
     2nd_test B1, 3rd_test B3. v1 labelled "last x% of life" as terminal for all twelve,
     so two thirds of IMS positives sat on bearings that never degraded. The IMS number
     carried no information about circularity -- and the v1 verdict rested on it.

What v2 measures instead
------------------------
Circularity is not a property of a dataset. It is a property of a TRAJECTORY, and it is
governed by whether that trajectory's failure mode actually manifests in the criterion
channel. XJTU Bearing1_4 fails by CAGE FRACTURE and its RMS rises only 1.2x over its own
baseline, while the other fourteen bearings rise 5-20x. For that one bearing the label
boundary is not where vibration is extreme, so its label cannot be circular in vibration.

  D(trajectory) = end-of-life feature / own early-life baseline, on the criterion channel
  high D -> the criterion channel is extreme at the boundary -> circular
  low  D -> it is not

HONESTY NOTE -- this hypothesis is EXPLORATORY, not pre-registered. It was formed after
seeing v1's per-bearing ratios. It must therefore be confirmed on data not used to form
it: the FEMTO thermal channel, and the cage-fracture bearings specifically. Label it as
exploratory in the paper; do not present it as a prediction that was tested.

The IMS negative control
------------------------
The eight IMS bearings that never failed are a gift: a built-in negative control. A
detector that has learned real degradation should stay silent on them. Whatever it does
fire on there is pure label artefact -- it is detecting position-in-recording, not damage.

Usage
    python scripts/J7_circularity.py              # all datasets
    python scripts/J7_circularity.py nsk xjtu     # a subset
    python scripts/J7_circularity.py --check      # inputs only
    python scripts/J7_circularity.py --refresh    # ignore the cached feature tables
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import metrics_from_counts  # noqa: E402
from src.threshold import fit_threshold, predict_threshold  # noqa: E402

BUNDLE = ROOT / "colab_bundle"
TABLES = ROOT / "results" / "tables"
LIFE_PCTS = (0.02, 0.05, 0.10)
BAR = "=" * 78

CRITERION = {
    "nsk":  ("temp > 85 C AND vib > 9 m/s^2", "BOTH channels", "signal-based, absolute"),
    "xjtu": ("vibration > 10x own normal max", "vibration", "signal-based, RELATIVE"),
    "ims":  ("debris on magnetic plug (Qiu 2006)", "NONE recorded", "OFF-SIGNAL"),
}

XJTU_FAULT = {
    "Bearing1_1": "outer", "Bearing1_2": "outer", "Bearing1_3": "outer",
    "Bearing1_4": "CAGE", "Bearing1_5": "inner+outer",
    "Bearing2_1": "inner", "Bearing2_2": "outer", "Bearing2_3": "CAGE",
    "Bearing2_4": "outer", "Bearing2_5": "outer",
    "Bearing3_1": "outer", "Bearing3_2": "inner+ball+cage+outer", "Bearing3_3": "inner",
    "Bearing3_4": "inner", "Bearing3_5": "outer",
}
# NASA readme documents failures ONLY in these four; the other eight ran alongside.
IMS_FAULT = {
    "IMS_1st_test_B3": "inner", "IMS_1st_test_B4": "roller",
    "IMS_2nd_test_B1": "outer", "IMS_3rd_test_B3": "outer",
}


def head(step: str, title: str) -> None:
    print(f"\n{BAR}\nSTEP {step}  {title}\n{BAR}", flush=True)


def fault_of(ds: str, traj: str) -> str:
    if ds == "xjtu":
        return XJTU_FAULT.get(traj, "?")
    if ds == "ims":
        return IMS_FAULT.get(traj, "NEVER FAILED")
    return "thermal seizure"


# ---------------------------------------------------------------- features
def record_features(ds: str, rec: pd.DataFrame, n_win: int, refresh: bool) -> pd.DataFrame:
    cache = TABLES / f"j7_features_{ds}.csv"
    if cache.exists() and not refresh:
        print(f"      reusing cached features: {cache.name}")
        return pd.read_csv(cache)
    W = np.load(BUNDLE / ds / "windows.npy", mmap_mode="r")
    n = len(rec)
    cols = {k: np.empty(n) for k in ("rms", "std", "p2p", "crest", "kurtosis", "absmax", "skew")}
    for i in range(n):
        blk = np.asarray(W[i * n_win:(i + 1) * n_win], dtype=np.float64)
        mag = np.sqrt((blk ** 2).sum(axis=1)) if blk.shape[1] > 1 else np.abs(blk[:, 0])
        r = np.sqrt((mag ** 2).mean(axis=1))
        z = (mag - mag.mean(1, keepdims=True)) / np.maximum(mag.std(1, keepdims=True), 1e-12)
        cols["rms"][i] = r.mean()
        cols["std"][i] = mag.std(1).mean()
        cols["p2p"][i] = (mag.max(1) - mag.min(1)).mean()
        cols["crest"][i] = (mag.max(1) / np.maximum(r, 1e-12)).mean()
        cols["kurtosis"][i] = (z ** 4).mean(1).mean()
        cols["skew"][i] = (z ** 3).mean(1).mean()
        cols["absmax"][i] = mag.max(1).max()
        if n > 400 and i % max(1, n // 8) == 0:
            print(f"      record {i:5d}/{n}", flush=True)
    feat = pd.DataFrame(cols)
    if ds == "nsk":
        hf = pd.read_csv(TABLES / "hour_features.csv")
        if len(hf) == n:
            for c in ("bearing_temp_mean", "delta_temp_mean", "bearing_temp_slope"):
                feat[c] = hf[c].to_numpy()
    feat.to_csv(cache, index=False)
    return feat


def relativise(feat: pd.DataFrame, traj: np.ndarray) -> pd.DataFrame:
    """Divide every feature by that trajectory's own early-life baseline (median of first 10%)."""
    out = feat.copy()
    for t in np.unique(traj):
        m = traj == t
        for c in feat.columns:
            v = feat.loc[m, c].to_numpy(float)
            base = np.median(v[:max(3, len(v) // 10)])
            out.loc[m, c] = v / (base if abs(base) > 1e-9 else 1.0)
    return out


# ---------------------------------------------------------------- evaluation
def counts(y, p) -> dict:
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    m = metrics_from_counts(tp, fp, fn, tn)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "macro_f1": m["macro_f1"], "mcc": m["mcc"]}


def oof_pred_threshold(x, y, folds) -> np.ndarray:
    p = np.zeros_like(y)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        if y[tr].sum() in (0, len(tr)):
            continue
        p[te] = predict_threshold(x[te], fit_threshold(x[tr], y[tr], "balanced_accuracy"))
    return p


def oof_pred_forest(X, y, folds, seed=0) -> np.ndarray:
    p = np.zeros_like(y)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(np.unique(y[tr])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1,
                                     class_weight="balanced_subsample").fit(X[tr], y[tr])
        p[te] = clf.predict(X[te])
    return p


# ---------------------------------------------------------------- main
def main(which: list[str], check_only: bool, refresh: bool) -> int:
    print(BAR)
    print("J7 (v2)  LABEL-INPUT CIRCULARITY, PER TRAJECTORY".center(78))
    print(BAR)

    head("1", "Criteria, inventory, and trajectory census")
    for ds in which:
        c, ch, reg = CRITERION[ds]
        print(f"  {ds:5s} stop: {c:36s} criterion channel: {ch:15s} {reg}")
    meta = {}
    for ds in which:
        d = BUNDLE / ds
        if not (d / "windows.npy").exists():
            print(f"\n  {ds}: MISSING bundle -- run scripts/60_build_colab_bundle.py {ds}")
            return 1
        rec = pd.read_csv(d / "records.csv")
        W = np.load(d / "windows.npy", mmap_mode="r")
        meta[ds] = (rec, W.shape[0] // len(rec))
        n_failed = sum(1 for t in rec.trajectory_id.unique() if fault_of(ds, t) != "NEVER FAILED")
        note = ""
        if ds == "ims":
            note = f"  <-- only {n_failed} DOCUMENTED failures; {rec.trajectory_id.nunique() - n_failed} never failed"
        print(f"  {ds:5s} {len(rec):6d} records | {rec.trajectory_id.nunique():3d} trajectories{note}")
    if check_only:
        print("\n  --check: inputs OK.")
        return 0

    rows, traj_rows = [], []
    for ds in which:
        rec, n_win = meta[ds]
        traj = rec.trajectory_id.to_numpy()

        head("2", f"{ds.upper()} -- per-record features")
        feat_abs = record_features(ds, rec, n_win, refresh)
        feat_rel = relativise(feat_abs, traj)
        print(f"      {len(feat_abs.columns)} features, absolute and baseline-relative")

        head("3", f"{ds.upper()} -- per-trajectory degradation ratio D on the criterion channel")
        print(f"    {'trajectory':24s} {'n':>5s} {'baseline':>9s} {'end':>9s} {'D':>7s}  failure mode")
        D = {}
        for t in sorted(set(traj)):
            v = feat_abs.loc[traj == t, "rms"].to_numpy(float)
            base, end = np.median(v[:max(3, len(v) // 10)]), np.median(v[-3:])
            D[t] = end / base
            flag = "  <== flat" if D[t] < 2 else ""
            print(f"    {t:24s} {len(v):>5d} {base:>9.4f} {end:>9.4f} {D[t]:>6.1f}x  "
                  f"{fault_of(ds, t)}{flag}")

        # IMS: the documented-failure subset carries the analysis; the rest is the control
        use = np.ones(len(rec), bool)
        if ds == "ims":
            use = np.array([fault_of(ds, t) != "NEVER FAILED" for t in traj])
            print(f"\n    restricting the analysis to the {use.sum()} records of the "
                  f"{len(set(traj[use]))} DOCUMENTED failures;")
            print(f"    the other {(~use).sum()} records become the negative control in step 6.")

        head("4", f"{ds.upper()} -- CI and GAP, absolute vs baseline-relative features")
        sub_traj = traj[use]
        folds = ([np.where(sub_traj == t)[0] for t in np.unique(sub_traj)]
                 if len(np.unique(sub_traj)) > 1 else None)
        if folds is None:
            fa = pd.read_csv(ROOT / "data/metadata/fold_assignments.csv").sort_values("hour_id")
            folds = [np.where(fa.fold_id.to_numpy() == k)[0] for k in range(5)]
        print(f"    CV: {'leave-one-trajectory-out' if len(np.unique(sub_traj)) > 1 else 'frozen 5-fold record-grouped'}"
              f", {len(folds)} folds\n")
        print(f"    {'life%':>6s} {'features':>10s} {'best threshold':>22s} {'CI':>8s} {'RF':>8s} {'GAP':>8s}")
        for pct in LIFE_PCTS:
            life = rec.life_fraction.to_numpy()[use]
            y = (life > 1 - pct).astype(int)
            if y.sum() < 2:
                continue
            for tag, F in (("absolute", feat_abs[use]), ("relative", feat_rel[use])):
                best_c, best = None, None
                for c in F.columns:
                    r = counts(y, oof_pred_threshold(F[c].to_numpy(float), y, folds))
                    rows.append({"dataset": ds, "life_pct": pct, "featureset": tag,
                                 "method": f"thr:{c}", **r})
                    if best is None or r["macro_f1"] > best["macro_f1"]:
                        best_c, best = c, r
                rf = counts(y, oof_pred_forest(F.to_numpy(float), y, folds))
                rows.append({"dataset": ds, "life_pct": pct, "featureset": tag,
                             "method": "rf:all", **rf})
                print(f"    {pct:>6.0%} {tag:>10s} {best_c:>22s} {best['macro_f1']:>8.4f} "
                      f"{rf['macro_f1']:>8.4f} {rf['macro_f1'] - best['macro_f1']:>+8.4f}")

        # ------------------------------------------------ per-trajectory recoverability
        if len(np.unique(sub_traj)) > 1:
            head("5", f"{ds.upper()} -- per-trajectory recoverability vs D vs failure mode")
            life = rec.life_fraction.to_numpy()[use]
            y = (life > 1 - 0.05).astype(int)
            F = feat_rel[use]
            best_c = max(F.columns, key=lambda c: counts(y, oof_pred_threshold(
                F[c].to_numpy(float), y, folds))["macro_f1"])
            p = oof_pred_threshold(F[best_c].to_numpy(float), y, folds)
            print(f"    threshold on relative '{best_c}', last 5% of life, leave-one-out\n")
            print(f"    {'trajectory':24s} {'D':>6s} {'pos':>5s} {'recall':>7s} {'FP':>5s}  failure mode")
            for t in np.unique(sub_traj):
                m = sub_traj == t
                rc = (p[m][y[m] == 1] == 1).mean() if y[m].sum() else np.nan
                fp = int(((y[m] == 0) & (p[m] == 1)).sum())
                print(f"    {t:24s} {D[t]:>5.1f}x {int(y[m].sum()):>5d} {rc:>7.2f} {fp:>5d}  "
                      f"{fault_of(ds, t)}")
                traj_rows.append({"dataset": ds, "trajectory": t, "D": D[t], "recall": rc,
                                  "FP": fp, "fault": fault_of(ds, t)})

        # ------------------------------------------------ IMS negative control
        if ds == "ims":
            head("6", "IMS NEGATIVE CONTROL -- eight bearings that never failed")
            print("    A detector that learned real degradation should stay SILENT here.")
            print("    Anything it fires on is detecting position-in-recording, not damage.\n")
            yf = (rec.life_fraction.to_numpy()[use] > 0.95).astype(int)
            Ff = feat_rel[use]
            c = max(Ff.columns, key=lambda cc: counts(yf, oof_pred_threshold(
                Ff[cc].to_numpy(float), yf, folds))["macro_f1"])
            thr = fit_threshold(Ff[c].to_numpy(float), yf, "balanced_accuracy")
            ctrl = feat_rel[~use]
            ct = traj[~use]
            print(f"    threshold fitted on the 4 failed bearings: relative '{c}' >= {thr:.3f}\n")
            print(f"    {'trajectory':24s} {'D':>6s} {'alarms in last 5%':>19s} {'alarms overall':>16s}")
            for t in np.unique(ct):
                m = ct == t
                pr = predict_threshold(ctrl.loc[m, c].to_numpy(float), thr)
                lf = rec.life_fraction.to_numpy()[~use][m]
                tail = pr[lf > 0.95]
                print(f"    {t:24s} {D[t]:>5.1f}x {tail.mean():>18.2f} {pr.mean():>16.2f}")
                traj_rows.append({"dataset": "ims_control", "trajectory": t, "D": D[t],
                                  "recall": float(tail.mean()), "FP": int(pr.sum()),
                                  "fault": "NEVER FAILED"})

    res, tr = pd.DataFrame(rows), pd.DataFrame(traj_rows)

    head("7", "Verdict -- reported per dataset, with NO cross-group median")
    for ds in which:
        g = res[(res.dataset == ds) & (res.featureset == "relative")]
        if g.empty:
            continue
        ci = g[g.method.str.startswith("thr:")].groupby("life_pct").macro_f1.max()
        rf = g[g.method == "rf:all"].set_index("life_pct").macro_f1
        gap = (rf - ci).median()
        print(f"  {ds.upper():5s} criterion {CRITERION[ds][1]:15s} "
              f"CI(median) {ci.median():.4f}  GAP(median) {gap:+.4f}  -> "
              f"{'threshold already expresses the label (CIRCULAR)' if gap < 0.02 else 'a richer model adds real information'}")
    if not tr.empty:
        real = tr[tr.dataset != "ims_control"].dropna(subset=["recall"])
        if len(real) > 3:
            r = np.corrcoef(real.D, real.recall)[0, 1]
            print(f"\n  correlation between degradation ratio D and per-trajectory recall: "
                  f"r = {r:+.3f}  (n = {len(real)} trajectories)")
            print("  -> the mechanism claim is that circularity tracks D, and D tracks failure mode.")
            print("     EXPLORATORY: this hypothesis was formed after seeing v1, so it must be")
            print("     confirmed on data not used to form it (FEMTO thermal channel).")

    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "j7_circularity.csv", index=False)
    tr.to_csv(TABLES / "j7_per_trajectory.csv", index=False)
    print(f"\n{BAR}\n  WROTE  results/tables/j7_circularity.csv   ({len(res)} rows)"
          f"\n         results/tables/j7_per_trajectory.csv ({len(tr)} rows)\n{BAR}")
    return 0


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    raise SystemExit(main(a or ["nsk", "xjtu", "ims"], "--check" in sys.argv, "--refresh" in sys.argv))
