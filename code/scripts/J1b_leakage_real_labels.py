"""J1b -- Delta_leakage on REAL labels, in the same unit as every other decomposition term.

Why this replaces J1 in the C4 table
------------------------------------
J1 measured leakage with RANDOM labels. That is the right control for "can a model memorise
which record a window came from", but it is NOT the same quantity as the other four terms in
the decomposition, which are all spreads of REAL record-level macro-F1. Putting a
random-label gap next to a real-performance spread and taking their ratio compares two
different kinds of number, and the headline claim -- that leakage is the smallest term --
would not survive a careful reviewer.

J1b measures the same thing the other terms measure: how much the reported record-level
macro-F1 moves when one choice is varied, on the real task.

The controlled comparison
-------------------------
Everything is held fixed except the one thing under test. For a given repeat:

    1. Choose the TEST RECORDS (30%), by whichever grouping the protocol specifies.
    2. Inside each test record, split its windows into a LEAK half and an EVAL half.
    3. S0_leaky  trains on: train-record windows  +  the LEAK half of the test records.
       S1_clean  trains on: train-record windows  only.
    4. BOTH are evaluated on exactly the same thing: the EVAL half of the same test records.

So S0 and S1 share the test records, the evaluation windows, the labels and the model. The
only difference is whether the model was shown other windows from the records it is graded
on -- which is precisely what `train_test_split` on a windowed signal does. The gap is
leakage and nothing else.

    S0 - S1   within-record leakage        (same test records, same evaluation windows)
    S1 - S2   adjacent-record correlation  (random records vs whole trajectories held out)
    S2 - S3   operating-condition shift    (whole conditions held out)

LABELLING
    Fixed at pct05 -- the last 5% of each trajectory's life. Chosen because it is the only
    rule that is non-degenerate on all three datasets AND gives a comparable positive rate
    (4-5%) on each, so the cross-dataset numbers mean the same thing. 3sig_rms is run as a
    sensitivity check; it is the most-cited statistical rule but its positive rate varies
    from 4% (NSK) to 28% (XJTU), which makes the datasets not directly comparable under it.

Outputs
    results/tables/j1b_leakage_real.csv     one row per (dataset, rule, protocol, repeat)
    results/logs/j1b_run.log                full training log, every fit recorded

Usage
    python scripts/J1b_leakage_real_labels.py
    python scripts/J1b_leakage_real_labels.py --check
    REPEATS=5 python scripts/J1b_leakage_real_labels.py      # quick pass
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
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
LOGS = ROOT / "results" / "logs"
REPEATS = int(os.environ.get("REPEATS", "20"))
TEST_FRAC, LEAK_FRAC = 0.30, 0.50
N_TREES = 200
RULES = ["pct05", "3sig_rms"]
PROTOCOLS = ["S0_leaky", "S1_clean", "S2_trajectory", "S3_condition"]
IMS_FAILED = {"IMS_1st_test_B3", "IMS_1st_test_B4", "IMS_2nd_test_B1", "IMS_3rd_test_B3"}
BAR = "=" * 78

_log_fh = None


def log(msg: str = "", to_screen: bool = True) -> None:
    if to_screen:
        print(msg, flush=True)
    if _log_fh:
        _log_fh.write(msg + "\n")
        _log_fh.flush()


def head(step: str, title: str) -> None:
    log(f"\n{BAR}\nSTEP {step}  {title}\n{BAR}")


# ---------------------------------------------------------------- labels
def label_pct(life: np.ndarray, pct: float) -> np.ndarray:
    return (life > 1 - pct).astype(int)


def label_3sig(rms: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """mean + 3 sd of the first 20% of each trajectory, confirmed over 3 records."""
    y = np.zeros(len(rms), int)
    for t in np.unique(traj):
        m = np.where(traj == t)[0]
        h = rms[m]
        nb = max(5, int(len(h) * 0.20))
        mu, sd = h[:nb].mean(), max(h[:nb].std(), 1e-12)
        flag, run = h > mu + 3 * sd, 0
        for i, f in enumerate(flag):
            run = run + 1 if f else 0
            if run >= 3:
                y[m[i - 2:]] = 1
                break
    return y


# ---------------------------------------------------------------- splits
def choose_test_records(proto: str, n_rec: int, traj: np.ndarray, cond: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray | None:
    """Boolean mask over RECORDS marking the held-out set."""
    if proto in ("S0_leaky", "S1_clean"):
        m = np.zeros(n_rec, bool)
        m[rng.choice(n_rec, max(1, int(n_rec * TEST_FRAC)), replace=False)] = True
        return m
    key = traj if proto == "S2_trajectory" else cond
    u = np.unique(key)
    if len(u) < 2:
        return None
    k = max(1, int(round(len(u) * TEST_FRAC)))
    return np.isin(key, rng.choice(u, k, replace=False))


def run_one(proto: str, F: np.ndarray, y_rec: np.ndarray, traj: np.ndarray, cond: np.ndarray,
            rng: np.random.Generator, seed: int) -> dict | None:
    """One repeat. F is [n_records, n_windows, n_features]."""
    n_rec, n_win, _ = F.shape
    te_rec = choose_test_records(proto, n_rec, traj, cond, rng)
    if te_rec is None or te_rec.sum() == 0 or (~te_rec).sum() == 0:
        return None
    if len(np.unique(y_rec[~te_rec])) < 2:
        return None

    # inside each test record: LEAK half (may enter training) and EVAL half (never does)
    n_leak = max(1, int(n_win * LEAK_FRAC)) if n_win > 1 else 0
    leak_idx = np.arange(n_leak)
    eval_idx = np.arange(n_leak, n_win)
    if len(eval_idx) == 0:
        return None

    Xtr = [F[~te_rec].reshape(-1, F.shape[2])]
    ytr = [np.repeat(y_rec[~te_rec], n_win)]
    if proto == "S0_leaky" and n_leak > 0:
        Xtr.append(F[te_rec][:, leak_idx].reshape(-1, F.shape[2]))
        ytr.append(np.repeat(y_rec[te_rec], n_leak))
    Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)

    Xte = F[te_rec][:, eval_idx].reshape(-1, F.shape[2])
    yte_rec = y_rec[te_rec]

    t0 = time.time()
    clf = RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=-1,
                                 class_weight="balanced_subsample").fit(Xtr, ytr)
    fit_s = time.time() - t0

    # window probabilities -> record score (mean pooling) -> threshold fitted on TRAIN records
    p_te = clf.predict_proba(Xte)[:, 1].reshape(te_rec.sum(), len(eval_idx)).mean(axis=1)
    p_tr = clf.predict_proba(F[~te_rec].reshape(-1, F.shape[2]))[:, 1] \
        .reshape((~te_rec).sum(), n_win).mean(axis=1)
    thr = fit_threshold(p_tr, y_rec[~te_rec], "balanced_accuracy")
    pred = predict_threshold(p_te, thr)

    tp = int(((yte_rec == 1) & (pred == 1)).sum()); fp = int(((yte_rec == 0) & (pred == 1)).sum())
    fn = int(((yte_rec == 1) & (pred == 0)).sum()); tn = int(((yte_rec == 0) & (pred == 0)).sum())
    m = metrics_from_counts(tp, fp, fn, tn)
    return {"n_train_win": len(ytr), "n_test_rec": int(te_rec.sum()),
            "n_eval_win": Xte.shape[0], "pos_train": int(y_rec[~te_rec].sum()),
            "pos_test": int(yte_rec.sum()), "threshold": float(thr), "fit_s": fit_s,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "macro_f1": m["macro_f1"], "mcc": m["mcc"]}


def main(check_only: bool) -> int:
    global _log_fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOGS / "j1b_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J1b  DELTA_LEAKAGE ON REAL LABELS".center(78))
    log("same test records, same evaluation windows -- only the training set differs".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S} | repeats={REPEATS} | "
        f"test_frac={TEST_FRAC} leak_frac={LEAK_FRAC} | RF {N_TREES} trees")

    head("1", "Inputs")
    meta = {}
    for ds in ("nsk", "xjtu", "ims"):
        wf = TABLES / f"j1_winfeat_{ds}.npy"
        rc = BUNDLE / ds / "records.csv"
        if not wf.exists() or not rc.exists():
            log(f"  {ds}: MISSING {wf.name if not wf.exists() else rc.name} "
                f"-- run scripts/J1_leakage_ladder.py first (it caches the window features)")
            return 1
        rec = pd.read_csv(rc)
        keep = rec.trajectory_id.isin(IMS_FAILED).to_numpy() if ds == "ims" else np.ones(len(rec), bool)
        F = np.load(wf)[keep]
        sub = rec.loc[keep].reset_index(drop=True)
        meta[ds] = (F, sub)
        n_leak = max(1, int(F.shape[1] * LEAK_FRAC)) if F.shape[1] > 1 else 0
        log(f"  {ds:5s} {F.shape[0]:6d} records x {F.shape[1]:2d} windows x {F.shape[2]} features"
            f" | {sub.trajectory_id.nunique():2d} traj | {sub.condition_id.nunique()} cond"
            f" | leak {n_leak}/eval {F.shape[1] - n_leak} windows per test record")
    if check_only:
        log("\n  --check: inputs OK.")
        return 0

    rows = []
    for ds in ("nsk", "xjtu", "ims"):
        F, sub = meta[ds]
        traj, cond = sub.trajectory_id.to_numpy(), sub.condition_id.to_numpy()
        life = sub.life_fraction.to_numpy()
        rms_rec = F[:, :, 0].mean(axis=1)

        for rule in RULES:
            y_rec = label_pct(life, 0.05) if rule == "pct05" else label_3sig(rms_rec, traj)
            rate = y_rec.mean()
            head("2", f"{ds.upper()} / {rule} -- {int(y_rec.sum())} positive records "
                      f"({rate:.1%})")
            if y_rec.sum() < 3 or rate > 0.5:
                log(f"    skipped: positive rate {rate:.1%} outside the usable range")
                continue

            for proto in PROTOCOLS:
                scores, first = [], True
                log(f"\n    --- {proto} ---")
                for r in range(REPEATS):
                    rng = np.random.default_rng(5000 + r)
                    out = run_one(proto, F, y_rec, traj, cond, rng, seed=r)
                    if out is None:
                        log(f"      not applicable to this dataset")
                        break
                    scores.append(out["macro_f1"])
                    rows.append({"dataset": ds, "rule": rule, "protocol": proto,
                                 "repeat": r, **out})
                    line = (f"      rep {r:2d} | train {out['n_train_win']:7d} win "
                            f"(pos rec {out['pos_train']:4d}) | test {out['n_test_rec']:5d} rec, "
                            f"{out['n_eval_win']:6d} eval win (pos {out['pos_test']:3d}) | "
                            f"thr {out['threshold']:.3f} | TP{out['TP']:3d} FP{out['FP']:4d} "
                            f"FN{out['FN']:3d} | F1 {out['macro_f1']:.4f} | {out['fit_s']:5.1f}s")
                    log(line, to_screen=(first or r == REPEATS - 1))
                    first = False
                if scores:
                    s = np.array(scores)
                    log(f"      => median {np.median(s):.4f}  [{s.min():.4f}, {s.max():.4f}]"
                        f"  over {len(s)} repeats")

    res = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "j1b_leakage_real.csv", index=False)

    head("3", "Delta_leakage, decomposed along the ladder")
    log("  S0-S1 is leakage and nothing else: identical test records, identical evaluation")
    log("  windows, identical labels. Only the training set differs.\n")
    log(f"  {'dataset':7s} {'rule':10s} {'S0':>8s} {'S1':>8s} {'S2':>8s} {'S3':>8s} "
        f"{'S0-S1':>8s} {'S1-S2':>8s} {'S2-S3':>8s}")
    summary = []
    for (ds, rule), g in res.groupby(["dataset", "rule"]):
        med = {p: float(g[g.protocol == p].macro_f1.median()) if (g.protocol == p).any() else np.nan
               for p in PROTOCOLS}
        d01 = med["S0_leaky"] - med["S1_clean"]
        d12 = med["S1_clean"] - med["S2_trajectory"]
        d23 = med["S2_trajectory"] - med["S3_condition"]
        summary.append({"dataset": ds, "rule": rule, **{f"median_{p}": med[p] for p in PROTOCOLS},
                        "leakage_S0_S1": d01, "adjacency_S1_S2": d12, "condition_S2_S3": d23})
        f = lambda v: "     n/a" if not np.isfinite(v) else f"{v:8.4f}"
        log(f"  {ds:7s} {rule:10s} {f(med['S0_leaky'])} {f(med['S1_clean'])} "
            f"{f(med['S2_trajectory'])} {f(med['S3_condition'])} "
            f"{f(d01)} {f(d12)} {f(d23)}")
    sm = pd.DataFrame(summary)
    sm.to_csv(TABLES / "j1b_summary.csv", index=False)

    head("4", "What goes into the C4 table")
    log("  Delta_leakage for the decomposition is the S0-S1 column above: real labels,")
    log("  real record-level macro-F1, same unit as boundary / aggregation / normalization /")
    log("  optimization. Use the pct05 row -- it is the only rule with a comparable positive")
    log("  rate on all three datasets.\n")
    for _, r in sm[sm.rule == "pct05"].iterrows():
        log(f"    {r.dataset:7s} Delta_leakage = {r.leakage_S0_S1:+.4f}   "
            f"(adjacency {r.adjacency_S1_S2:+.4f}, condition {r.condition_S2_S3:+.4f})")
    log("\n  Compare with the random-label version measured by J1, which is a DIFFERENT")
    log("  quantity and must not be mixed into the same column:")
    j1 = TABLES / "j1_leakage_ladder.csv"
    if j1.exists():
        d = pd.read_csv(j1)
        for ds, g in d.groupby("dataset"):
            s0 = g[g.protocol == "S0_window_random"].macro_f1.to_numpy()
            tgt = "S2_trajectory" if (g.protocol == "S2_trajectory").any() else "S1_record_grouped"
            sx = g[g.protocol == tgt].macro_f1.to_numpy()
            n = min(len(s0), len(sx))
            if n:
                log(f"    {ds:7s} random-label gap = {np.median(s0[:n] - sx[:n]):+.4f}   "
                    f"(memorisation capacity, NOT performance inflation)")

    log(f"\n{BAR}")
    log(f"  WROTE  results/tables/j1b_leakage_real.csv ({len(res)} rows)")
    log(f"         results/tables/j1b_summary.csv      ({len(sm)} rows)")
    log(f"         results/logs/j1b_run.log            (full training log)")
    log(f"  finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(BAR)
    _log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--check" in sys.argv))
