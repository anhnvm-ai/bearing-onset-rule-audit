"""J5 -- Witness-rate profiling across the whole trajectory.

Question this answers
---------------------
The conference paper found that hour 124, the transition record, is missed by every one
of the seven methods and by every seed. Two very different explanations fit that fact:

    (A) POOLING FAILURE   some windows in the burst do carry terminal evidence, but they
                          are a small minority, so averaging over 50 windows drowns them.
                          -> a different pooling rule (max / top-k / noisy-OR) would fix it,
                             and journal contribution C3 stands.

    (B) MODEL FAILURE     no window in the burst looks terminal to the model at all.
                          -> no pooling rule can help, and C3 is weak on this dataset.

J5 decides between them by computing the WITNESS RATE

    rho(r) = fraction of the windows inside record r that individually look terminal

under two independent definitions: a threshold on the signal, and the model's own
per-window probability. Everything runs on CPU from arrays already on disk.

Outputs
    results/tables/j5_witness_rate.csv   one row per (record, source)
    results/tables/j5_summary.csv        the diagnosis, per model/seed
    results/tables/j5_window_probs.csv   tidy per-window probabilities, reused by J6

Usage
    python scripts/J5_witness_rate.py            # full run
    python scripts/J5_witness_rate.py --check    # verify inputs only, compute nothing
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.threshold import threshold_oof  # noqa: E402

PROC = ROOT / "data" / "processed"
CKPT = ROOT / "results" / "checkpoints"
META = ROOT / "data" / "metadata"
TABLES = ROOT / "results" / "tables"

N_WIN = 50
MODELS = ("cnn1d", "resnet", "fusion")
SEEDS = (0, 21, 42, 84, 123)
BAR = "=" * 78


def head(step: str, total: str, title: str) -> None:
    print(f"\n{BAR}\nSTEP {step}/{total}  {title}\n{BAR}", flush=True)


def profile_bar(v: float, width: int = 28) -> str:
    n = int(round(v * width))
    return "#" * n + "." * (width - n)


# --------------------------------------------------------------------------
def load_window_probs(folds: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    """(model, seed) -> per-hour window probabilities, shape [129, 50], h5 global runs."""
    out = {}
    for m in MODELS:
        for s in SEEDS:
            mat = np.full((len(folds), N_WIN), np.nan)
            complete = True
            for f in range(5):
                p = CKPT / f"e7_{m}_seed{s}_global_fold{f}_testprob.npy"
                if not p.exists():
                    complete = False
                    break
                w = np.load(p)
                hrs = np.sort(folds.loc[folds.fold_id == f, "hour_id"].to_numpy())
                if w.size != len(hrs) * N_WIN:
                    complete = False
                    break
                mat[hrs] = w.reshape(len(hrs), N_WIN)
            if complete:
                out[(m, s)] = mat
    return out


def main(check_only: bool = False) -> int:
    print(BAR)
    print("J5  WITNESS-RATE PROFILING".center(78))
    print("is the hour-124 miss a POOLING failure or a MODEL failure?".center(78))
    print(BAR)

    # ---------------------------------------------------------------- step 1
    head("1", "5", "Inputs")
    folds = pd.read_csv(META / "fold_assignments.csv").sort_values("hour_id").reset_index(drop=True)
    win = np.load(PROC / "windows_raw.npy", mmap_mode="r")
    n_rec = len(folds)
    print(f"  fold_assignments.csv   {n_rec} records, folds "
          f"{sorted(int(f) for f in folds.fold_id.unique())}")
    print(f"  windows_raw.npy        {win.shape}  ->  {win.shape[0] // n_rec} windows/record")
    print(f"  label_h5 positives     {int(folds.label_h5.sum())} records: "
          f"{folds.loc[folds.label_h5 == 1, 'hour_id'].tolist()}")

    probs = load_window_probs(folds)
    print(f"  window probabilities   {len(probs)} complete (model, seed) runs of "
          f"{len(MODELS) * len(SEEDS)} possible")
    for m in MODELS:
        got = sorted(s for (mm, s) in probs if mm == m)
        print(f"      {m:8s} seeds {got if got else '-- none complete --'}")
    if not probs:
        print("\n  ERROR: no complete run found; cannot answer the question.")
        return 1
    if check_only:
        print("\n  --check: inputs OK, nothing computed.")
        return 0

    # ---------------------------------------------------------------- step 2
    head("2", "5", "Per-window RMS of the vibration magnitude")
    rms = np.empty((n_rec, N_WIN))
    for i in range(n_rec):
        blk = np.asarray(win[i * N_WIN:(i + 1) * N_WIN], dtype=np.float64)
        mag = np.sqrt(blk[:, 0, :] ** 2 + blk[:, 1, :] ** 2)
        rms[i] = np.sqrt((mag ** 2).mean(axis=1))
        if i % 26 == 0 or i == n_rec - 1:
            print(f"    record {i:3d}/{n_rec - 1}   burst mean RMS = {rms[i].mean():.4f}", flush=True)

    # ---------------------------------------------------------------- step 3
    head("3", "5", "Threshold-based witness rate  rho_thr(r)")
    print("  Fold thresholds are fitted on TRAIN records only (same rule as the paper),")
    print("  then applied to the windows of the held-out records.\n")
    feat = pd.Series(rms.mean(axis=1), index=folds.hour_id.to_numpy())
    y = pd.Series(folds.label_h5.to_numpy(int), index=folds.hour_id.to_numpy())
    fid = pd.Series(folds.fold_id.to_numpy(int), index=folds.hour_id.to_numpy())
    oof = threshold_oof(feat, y, fid, metric="balanced_accuracy").set_index("hour_id")
    thr = oof["fold_threshold"]
    for f in range(5):
        h = folds.loc[folds.fold_id == f, "hour_id"].iloc[0]
        print(f"    fold {f}  threshold = {thr.loc[h]:.4f}")

    rho_thr = np.array([(rms[h] > thr.loc[h]).mean() for h in folds.hour_id])

    print("\n  rho_thr profile over the whole trajectory (every 8th record, then the tail):")
    show = list(range(0, 118, 8)) + list(range(118, n_rec))
    for h in show:
        lab = int(folds.label_h5[h])
        mark = "  <-- TERMINAL" if lab else ""
        print(f"    h{h:3d} [{lab}]  rho_thr={rho_thr[h]:.2f}  |{profile_bar(rho_thr[h])}|{mark}")

    # ---------------------------------------------------------------- step 4
    head("4", "5", "Model-based witness rate  rho_model(r)")
    print("  rho_model = fraction of the 50 windows whose OWN probability exceeds 0.5.")
    print("  mean_prob = what the paper's mean pooling actually decides on.\n")
    rows = []
    for (m, s), mat in sorted(probs.items()):
        rho = (mat > 0.5).mean(axis=1)
        mean_p = mat.mean(axis=1)
        for h in range(n_rec):
            rows.append({"hour_id": h, "label_h5": int(folds.label_h5[h]), "fold_id": int(folds.fold_id[h]),
                         "model": m, "seed": s, "rho_model": rho[h], "mean_prob": mean_p[h],
                         "max_prob": mat[h].max(), "top5_prob": np.sort(mat[h])[-5:].mean(),
                         "pred_mean_pool": int(mean_p[h] > 0.5)})
    wr = pd.DataFrame(rows)

    print("  At the transition record h124:\n")
    print(f"    {'model':9s} {'seed':>4s} {'rho_model':>10s} {'mean_prob':>10s} {'max_prob':>9s} "
          f"{'top5':>7s}  {'mean-pool says':>14s}")
    t = wr[wr.hour_id == 124].sort_values(["model", "seed"])
    for _, r in t.iterrows():
        print(f"    {r.model:9s} {r.seed:>4d} {r.rho_model:>10.2f} {r.mean_prob:>10.3f} "
              f"{r.max_prob:>9.3f} {r.top5_prob:>7.3f}  {'TERMINAL' if r.pred_mean_pool else 'healthy':>14s}")

    # ---------------------------------------------------------------- step 5
    head("5", "5", "Diagnosis")
    t124 = wr[wr.hour_id == 124]
    n_runs = len(t124)
    missed = t124[t124.pred_mean_pool == 0]
    has_witness = missed[missed.rho_model > 0]
    would_fix_max = missed[missed.max_prob > 0.5]
    would_fix_top5 = missed[missed.top5_prob > 0.5]

    print(f"  runs examined                              {n_runs}")
    print(f"  runs where mean pooling MISSES h124        {len(missed)}")
    print(f"  ... of those, runs with >=1 witness window {len(has_witness)}")
    print(f"  ... that MAX pooling would have caught     {len(would_fix_max)}")
    print(f"  ... that TOP-5 pooling would have caught   {len(would_fix_top5)}")
    if len(missed):
        print(f"\n  median witness rate among missed runs      {missed.rho_model.median():.3f}")
        print(f"  median max window probability              {missed.max_prob.median():.3f}")
        print(f"  median mean-pooled probability             {missed.mean_prob.median():.3f}")

    print()
    if len(missed) and len(has_witness) / len(missed) >= 0.5:
        print("  >>> POOLING FAILURE.  The evidence is present in a minority of windows and")
        print("      mean pooling dilutes it below the decision threshold.")
        print("      Journal contribution C3 is supported on this dataset. Proceed to J6.")
    elif len(missed) == 0:
        print("  >>> NO MISS in these runs -- re-check which horizon/runs were loaded.")
    else:
        print("  >>> MODEL FAILURE.  Most missed runs have no witness window at all, so no")
        print("      pooling rule can recover h124. C3 must be re-scoped or moved to a")
        print("      dataset with longer records (Ferrara, 15 windows/record).")

    # false-alarm side: healthy records that a max rule would wrongly flag
    healthy = wr[wr.label_h5 == 0]
    fa_mean = int((healthy.pred_mean_pool == 1).sum())
    fa_max = int((healthy.max_prob > 0.5).sum())
    fa_top5 = int((healthy.top5_prob > 0.5).sum())
    print(f"\n  COST CHECK on {len(healthy)} healthy record-runs -- false alarms:")
    print(f"      mean pooling {fa_mean:5d}      max pooling {fa_max:5d}      "
          f"top-5 pooling {fa_top5:5d}")
    print("      (max pooling trading many false alarms for one recovered record would be a")
    print("       bad trade; J6 quantifies this properly with lead-time vs false-alarm curves.)")

    # ---------------------------------------------------------------- save
    TABLES.mkdir(parents=True, exist_ok=True)
    prof = pd.DataFrame({"hour_id": folds.hour_id, "label_h5": folds.label_h5,
                         "fold_id": folds.fold_id, "burst_mean_rms": rms.mean(axis=1),
                         "fold_threshold": [thr.loc[h] for h in folds.hour_id],
                         "rho_thr": rho_thr})
    prof.to_csv(TABLES / "j5_witness_rate.csv", index=False)
    wr.to_csv(TABLES / "j5_summary.csv", index=False)
    tidy = pd.concat([pd.DataFrame({"hour_id": np.repeat(np.arange(n_rec), N_WIN),
                                    "win_idx": np.tile(np.arange(N_WIN), n_rec),
                                    "model": m, "seed": s, "prob": mat.ravel()})
                      for (m, s), mat in sorted(probs.items())], ignore_index=True)
    tidy.to_csv(TABLES / "j5_window_probs.csv", index=False)

    print(f"\n{BAR}\n  WROTE")
    for f in ("j5_witness_rate.csv", "j5_summary.csv", "j5_window_probs.csv"):
        print(f"    results/tables/{f}   ({(TABLES / f).stat().st_size / 1048576:.1f} MB)")
    print(f"{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(check_only="--check" in sys.argv))
