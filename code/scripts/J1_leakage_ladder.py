"""J1 -- how much score can a model manufacture from labels that carry no signal?

The negative control, extended from one trajectory to three datasets
--------------------------------------------------------------------
Every record is given a RANDOM class label, drawn independently of the measurements. No
learnable relationship exists, so an honest protocol must return chance. Anything above
chance is the model memorising which record a window came from -- the windows of one record
share amplitude, noise floor and spectral shape, so once a record's windows appear on both
sides of the split, its arbitrary label is recoverable.

The conference measured this on NSK alone (optimism gap 0.319, positive in 100/100
permutations). J1 walks the full split ladder on three datasets:

    S0  window-random     windows shuffled, so one record lands on both sides   <- the control
    S1  record-grouped    whole records held out; adjacent records still correlated
    S2  trajectory-wise   whole bearings held out
    S3  condition-wise    whole operating conditions held out

S0 is what a paper does when it windows the signal and then calls train_test_split. The gap
S0 - S2 is the score such a paper reports without any signal being present at all.

Chance level is 1/3: labels are drawn from three classes, matching the conference control.

COST CONTROL
    Each permutation uses ONE split per protocol rather than 5-fold CV -- for a permutation
    test the extra folds buy precision that 100 permutations already provide, at 5x the cost.
    NSK runs 100 permutations; XJTU and IMS run PERM_BIG because they are 4-6x larger.

Usage
    python scripts/J1_leakage_ladder.py
    python scripts/J1_leakage_ladder.py --check
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import multiclass_macro_f1  # noqa: E402

BUNDLE = ROOT / "colab_bundle"
TABLES = ROOT / "results" / "tables"
N_CLASS, PERM_NSK, PERM_BIG, TEST_FRAC = 3, 100, 30, 0.30
BAR = "=" * 78
FEATS = ["rms", "kurtosis", "crest", "p2p", "absmax", "std", "skew"]
IMS_FAILED = {"IMS_1st_test_B3", "IMS_1st_test_B4", "IMS_2nd_test_B1", "IMS_3rd_test_B3"}


def head(step: str, title: str) -> None:
    print(f"\n{BAR}\nSTEP {step}  {title}\n{BAR}", flush=True)


def window_features(ds: str, n_rec: int, n_win: int) -> np.ndarray:
    """[n_records, n_windows, 7] per-window statistics of the vibration magnitude. Cached."""
    cache = TABLES / f"j1_winfeat_{ds}.npy"
    if cache.exists():
        print(f"      reusing {cache.name}")
        return np.load(cache)
    W = np.load(BUNDLE / ds / "windows.npy", mmap_mode="r")
    out = np.empty((n_rec, n_win, len(FEATS)))
    for i in range(n_rec):
        blk = np.asarray(W[i * n_win:(i + 1) * n_win], dtype=np.float64)
        mag = np.sqrt((blk ** 2).sum(axis=1)) if blk.shape[1] > 1 else np.abs(blk[:, 0])
        r = np.sqrt((mag ** 2).mean(axis=1))
        sd = mag.std(axis=1)
        z = (mag - mag.mean(axis=1, keepdims=True)) / np.maximum(sd[:, None], 1e-12)
        out[i, :, 0] = r
        out[i, :, 1] = (z ** 4).mean(axis=1)
        out[i, :, 2] = mag.max(axis=1) / np.maximum(r, 1e-12)
        out[i, :, 3] = mag.max(axis=1) - mag.min(axis=1)
        out[i, :, 4] = mag.max(axis=1)
        out[i, :, 5] = sd
        out[i, :, 6] = (z ** 3).mean(axis=1)
        if n_rec > 2000 and i % max(1, n_rec // 6) == 0:
            print(f"      record {i:5d}/{n_rec}", flush=True)
    np.save(cache, out)
    return out


def split_mask(protocol: str, n_rec: int, n_win: int, traj: np.ndarray,
               cond: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
    """Boolean test mask over WINDOWS, or None when the protocol does not apply."""
    n = n_rec * n_win
    if protocol == "S0_window_random":
        m = np.zeros(n, bool)
        m[rng.choice(n, int(n * TEST_FRAC), replace=False)] = True
        return m
    if protocol == "S1_record_grouped":
        te = rng.choice(n_rec, int(n_rec * TEST_FRAC), replace=False)
        m = np.zeros(n_rec, bool); m[te] = True
        return np.repeat(m, n_win)
    key = traj if protocol == "S2_trajectory" else cond
    u = np.unique(key)
    if len(u) < 2:
        return None
    k = max(1, int(round(len(u) * TEST_FRAC)))
    held = rng.choice(u, k, replace=False)
    return np.repeat(np.isin(key, held), n_win)


PROTOCOLS = ["S0_window_random", "S1_record_grouped", "S2_trajectory", "S3_condition"]


def main(check_only: bool) -> int:
    print(BAR)
    print("J1  RANDOM-LABEL NEGATIVE CONTROL ACROSS THE SPLIT LADDER".center(78))
    print(f"labels carry no signal, so anything above {1/N_CLASS:.3f} is memorisation".center(78))
    print(BAR)

    head("1", "Inputs")
    meta = {}
    for ds in ("nsk", "xjtu", "ims"):
        d = BUNDLE / ds
        if not (d / "windows.npy").exists():
            print(f"  {ds}: MISSING bundle")
            return 1
        rec = pd.read_csv(d / "records.csv")
        W = np.load(d / "windows.npy", mmap_mode="r")
        n_win = W.shape[0] // len(rec)
        keep = rec.trajectory_id.isin(IMS_FAILED).to_numpy() if ds == "ims" else np.ones(len(rec), bool)
        nperm = PERM_NSK if ds == "nsk" else PERM_BIG
        meta[ds] = (rec, n_win, keep, nperm)
        print(f"  {ds:5s} {int(keep.sum()):6d} records x {n_win:2d} windows | "
              f"{rec.loc[keep].trajectory_id.nunique():2d} trajectories | "
              f"{rec.loc[keep].condition_id.nunique()} conditions | {nperm} permutations")
    if check_only:
        print("\n  --check: inputs OK.")
        return 0

    rows = []
    for ds in ("nsk", "xjtu", "ims"):
        rec, n_win, keep, nperm = meta[ds]
        head("2", f"{ds.upper()} -- per-window features")
        F = window_features(ds, len(rec), n_win)[keep]
        sub = rec.loc[keep].reset_index(drop=True)
        traj, cond = sub.trajectory_id.to_numpy(), sub.condition_id.to_numpy()
        n_rec = len(sub)
        X = F.reshape(n_rec * n_win, len(FEATS))
        print(f"      {X.shape[0]} windows x {X.shape[1]} features")

        head("3", f"{ds.upper()} -- {nperm} permutations of random record labels")
        print(f"    {'protocol':20s} {'macro-F1 median':>16s} {'5-95 pct':>18s} {'vs chance':>11s}")
        per_proto = {}
        for proto in PROTOCOLS:
            scores = []
            for p in range(nperm):
                rng = np.random.default_rng(1000 + p)
                y_rec = rng.integers(0, N_CLASS, n_rec)
                y = np.repeat(y_rec, n_win)
                te = split_mask(proto, n_rec, n_win, traj, cond, rng)
                if te is None:
                    break
                tr = ~te
                if len(np.unique(y[tr])) < 2 or te.sum() == 0:
                    continue
                clf = RandomForestClassifier(n_estimators=100, random_state=p, n_jobs=-1)
                clf.fit(X[tr], y[tr])
                scores.append(multiclass_macro_f1(y[te], clf.predict(X[te])))
            if not scores:
                print(f"    {proto:20s}    -- not applicable to this dataset")
                continue
            s = np.array(scores)
            per_proto[proto] = s
            print(f"    {proto:20s} {np.median(s):>16.4f} "
                  f"{f'[{np.percentile(s, 5):.3f}, {np.percentile(s, 95):.3f}]':>18s} "
                  f"{np.median(s) - 1 / N_CLASS:>+11.4f}")
            for v in s:
                rows.append({"dataset": ds, "protocol": proto, "macro_f1": float(v)})

        head("4", f"{ds.upper()} -- optimism gap: what a naive split reports from nothing")
        base = per_proto.get("S0_window_random")
        if base is not None:
            for proto in PROTOCOLS[1:]:
                o = per_proto.get(proto)
                if o is None:
                    continue
                m = min(len(base), len(o))
                gap = base[:m] - o[:m]
                print(f"    S0 - {proto:20s} median {np.median(gap):+.4f} "
                      f"[{np.percentile(gap, 5):+.3f}, {np.percentile(gap, 95):+.3f}]  "
                      f"positive in {int((gap > 0).sum())}/{m}")

    res = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "j1_leakage_ladder.csv", index=False)
    print(f"\n{BAR}\n  WROTE  results/tables/j1_leakage_ladder.csv ({len(res)} rows)\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--check" in sys.argv))
