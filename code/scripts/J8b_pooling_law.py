"""J8b -- does the pooling law k* ~ rho * n hold outside NSK, and is it more than the stop rule?

Two questions, both answerable on CPU
-------------------------------------
J6 found, on NSK's single trajectory, that the best number of windows to pool tracked the
witness rate measured independently in J5: rho ~ 0.06 of 50 windows -> k* ~ 3, and detection
of the transition record fell away monotonically as k grew (max 9/9, top3 8/9, top5 5/9,
top10 1/9, mean 0/9). That is a mechanism, but it rests on one bearing.

  Q1  THE LAW.  Across trajectories that each have their own boundary record and their own
      witness rate, does the best k track rho * n? XJTU gives 15 bearings and IMS gives 4
      documented failures, so for the first time there is a sample rather than an anecdote.

  Q2  THE OBJECTION.  A reviewer will say: "of course max pooling wins -- your label is
      derived from a stopping rule that is itself a threshold on peak amplitude, so you have
      re-implemented the stop rule and called it a method." The control is direct: compare
      when the max-pool detector fires against when the DATASET'S OWN STOPPING CRITERION
      would fire. If the detector fires no earlier, the objection stands. If it fires
      earlier, it carries information the criterion does not.

No model is involved. Window scores are plain RMS, so anything found here is a property of
the pooling geometry rather than of one architecture -- which is a stronger claim than the
model-based version in J6, not a weaker one.

HONEST LIMIT ON BAG SIZE
    windows per record: NSK 50, XJTU 4, IMS 2. The k-sweep therefore spans 1..50 on NSK but
    only 1..4 and 1..2 elsewhere. That is enough to test the DIRECTION of the law (smaller k
    better when rho is small) and the per-trajectory correlation, but NOT its functional form.
    Ferrara (15 windows/record) is the only remaining public dataset with a bag large enough
    to test the form; it is not downloaded.

Usage
    python scripts/J8b_pooling_law.py                # nsk xjtu ims
    python scripts/J8b_pooling_law.py xjtu           # a subset
    python scripts/J8b_pooling_law.py --check
    python scripts/J8b_pooling_law.py --refresh      # recompute cached window RMS
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import metrics_from_counts  # noqa: E402
from src.threshold import fit_threshold, predict_threshold  # noqa: E402

BUNDLE = ROOT / "colab_bundle"
TABLES = ROOT / "results" / "tables"
LIFE_PCT = 0.05                    # one pre-specified horizon; J3 sweeps the FPT rules
BAR = "=" * 78

# Each dataset's own documented stopping rule, expressed on the peak of the vibration
# magnitude relative to that trajectory's own early-life peak.
STOP_RULE = {
    "nsk":  ("temp > 85 C AND vib > 9 m/s^2", None),   # not a pure relative-peak rule
    "xjtu": ("peak > 10x own normal maximum", 10.0),
    "ims":  ("debris on magnetic plug (off-signal)", None),
}


def head(step: str, title: str) -> None:
    print(f"\n{BAR}\nSTEP {step}  {title}\n{BAR}", flush=True)


def window_rms(ds: str, n_rec: int, n_win: int, refresh: bool) -> np.ndarray:
    """[n_records, n_windows] RMS of the vibration magnitude. Cached."""
    cache = TABLES / f"j8b_winrms_{ds}.npy"
    if cache.exists() and not refresh:
        print(f"      reusing {cache.name}")
        return np.load(cache)
    W = np.load(BUNDLE / ds / "windows.npy", mmap_mode="r")
    out = np.empty((n_rec, n_win))
    for i in range(n_rec):
        blk = np.asarray(W[i * n_win:(i + 1) * n_win], dtype=np.float64)
        mag = np.sqrt((blk ** 2).sum(axis=1)) if blk.shape[1] > 1 else np.abs(blk[:, 0])
        out[i] = np.sqrt((mag ** 2).mean(axis=1))
        if n_rec > 2000 and i % max(1, n_rec // 6) == 0:
            print(f"      record {i:5d}/{n_rec}", flush=True)
    np.save(cache, out)
    return out


def window_peak(ds: str, n_rec: int, n_win: int) -> np.ndarray:
    """Per-record peak of the magnitude -- the quantity the stop rules are written on."""
    cache = TABLES / f"j8b_winpeak_{ds}.npy"
    if cache.exists():
        return np.load(cache)
    W = np.load(BUNDLE / ds / "windows.npy", mmap_mode="r")
    out = np.empty(n_rec)
    for i in range(n_rec):
        blk = np.asarray(W[i * n_win:(i + 1) * n_win], dtype=np.float64)
        mag = np.sqrt((blk ** 2).sum(axis=1)) if blk.shape[1] > 1 else np.abs(blk[:, 0])
        out[i] = mag.max()
    np.save(cache, out)
    return out


def pool(mat: np.ndarray, k: int) -> np.ndarray:
    """Mean of the k largest window scores; k = n_win reproduces mean pooling."""
    k = min(k, mat.shape[1])
    return np.sort(mat, axis=1)[:, -k:].mean(axis=1)


def oof_pred(score: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    p = np.zeros_like(y)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        if y[tr].sum() in (0, len(tr)):
            continue
        p[te] = predict_threshold(score[te], fit_threshold(score[tr], y[tr], "balanced_accuracy"))
    return p


def counts(y, p) -> dict:
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    m = metrics_from_counts(tp, fp, fn, tn)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "macro_f1": m["macro_f1"], "mcc": m["mcc"]}


IMS_FAILED = {"IMS_1st_test_B3", "IMS_1st_test_B4", "IMS_2nd_test_B1", "IMS_3rd_test_B3"}


def main(which: list[str], check_only: bool, refresh: bool) -> int:
    print(BAR)
    print("J8b  DOES THE POOLING LAW k* ~ rho*n SURVIVE OUTSIDE NSK?".center(78))
    print("window scores are plain RMS -- no model, so this is pooling geometry".center(78))
    print(BAR)

    head("1", "Inventory and bag sizes")
    meta = {}
    for ds in which:
        d = BUNDLE / ds
        if not (d / "windows.npy").exists():
            print(f"  {ds}: MISSING bundle -- run scripts/60_build_colab_bundle.py {ds}")
            return 1
        rec = pd.read_csv(d / "records.csv")
        W = np.load(d / "windows.npy", mmap_mode="r")
        n_win = W.shape[0] // len(rec)
        if ds == "ims":                                   # only documented failures carry labels
            keep = rec.trajectory_id.isin(IMS_FAILED).to_numpy()
        else:
            keep = np.ones(len(rec), bool)
        meta[ds] = (rec, n_win, keep)
        ks = [k for k in (1, 2, 3, 4, 5, 10, 25, 50) if k <= n_win]
        print(f"  {ds:5s} {len(rec):6d} records x {n_win:2d} windows | "
              f"{rec.loc[keep].trajectory_id.nunique():2d} trajectories used | k sweep {ks}")
        if ds == "ims":
            print(f"        (restricted to the {len(IMS_FAILED)} documented failures; the other "
                  f"8 bearings never failed -- see README C5)")
    if check_only:
        print("\n  --check: inputs OK.")
        return 0

    rows, law_rows = [], []
    for ds in which:
        rec, n_win, keep = meta[ds]
        head("2", f"{ds.upper()} -- per-window RMS")
        R = window_rms(ds, len(rec), n_win, refresh)[keep]
        sub = rec.loc[keep].reset_index(drop=True)
        traj = sub.trajectory_id.to_numpy()
        y = (sub.life_fraction.to_numpy() > 1 - LIFE_PCT).astype(int)
        folds = ([np.where(traj == t)[0] for t in np.unique(traj)] if len(np.unique(traj)) > 1
                 else [np.where(pd.read_csv(ROOT / "data/metadata/fold_assignments.csv")
                                .sort_values("hour_id").fold_id.to_numpy() == k)[0] for k in range(5)])
        print(f"      {len(sub)} records, {len(folds)} folds, {int(y.sum())} positives "
              f"at last {LIFE_PCT:.0%} of life")

        head("3", f"{ds.upper()} -- witness rate at each trajectory's boundary record")
        # window-level threshold fitted on TRAINING trajectories only
        print(f"    {'trajectory':24s} {'boundary':>9s} {'rho':>6s} {'rho*n':>6s}  windows above thr")
        rho_of = {}
        for te in folds:
            tr = np.setdiff1d(np.arange(len(y)), te)
            wy = np.repeat(y[tr], n_win)
            if wy.sum() in (0, len(wy)):
                continue
            wthr = fit_threshold(R[tr].ravel(), wy, "balanced_accuracy")
            t = traj[te][0]
            pos = np.where(y[te] == 1)[0]
            if len(pos) == 0:
                continue
            b = pos[0]                                    # first terminal record = the boundary
            rho = float((R[te][b] > wthr).mean())
            rho_of[t] = rho
            print(f"    {t:24s} {int(sub.record_index.to_numpy()[te][b]):>9d} {rho:>6.2f} "
                  f"{rho * n_win:>6.2f}  {int(rho * n_win)}/{n_win}")

        head("4", f"{ds.upper()} -- k sweep, thresholds fitted per k on training folds")
        ks = [k for k in (1, 2, 3, 4, 5, 10, 25, 50) if k <= n_win]
        print(f"    {'k':>4s} {'rule':>10s} {'macroF1':>9s} {'MCC':>8s} {'TP':>5s} {'FP':>6s} "
              f"{'boundary records caught':>24s}")
        best_k_overall, best_f1 = None, -1
        for k in ks:
            sc = pool(R, k)
            p = oof_pred(sc, y, folds)
            c = counts(y, p)
            nb = nb_tot = 0
            for te in folds:
                pos = np.where(y[te] == 1)[0]
                if len(pos):
                    nb_tot += 1
                    nb += int(p[te][pos[0]] == 1)
            name = "max" if k == 1 else ("mean" if k == n_win else f"top{k}")
            rows.append({"dataset": ds, "k": k, "rule": name, "boundary_caught": nb,
                         "boundary_total": nb_tot, **c})
            if c["macro_f1"] > best_f1:
                best_f1, best_k_overall = c["macro_f1"], k
            print(f"    {k:>4d} {name:>10s} {c['macro_f1']:>9.4f} {c['mcc']:>8.4f} "
                  f"{c['TP']:>5d} {c['FP']:>6d} {f'{nb}/{nb_tot}':>24s}")
        print(f"\n    best k overall for {ds}: {best_k_overall}  "
              f"(median rho*n = {np.median([r * n_win for r in rho_of.values()]):.2f})")

        # ---- per-trajectory: the k that first catches that trajectory's boundary record
        for te in folds:
            pos = np.where(y[te] == 1)[0]
            if len(pos) == 0:
                continue
            t, b = traj[te][0], pos[0]
            kstar = np.nan
            for k in ks:
                sc = pool(R, k)
                tr = np.setdiff1d(np.arange(len(y)), te)
                if y[tr].sum() in (0, len(tr)):
                    continue
                thr = fit_threshold(sc[tr], y[tr], "balanced_accuracy")
                if predict_threshold(sc[te][b:b + 1], thr)[0] == 1:
                    kstar = k
                    break
            law_rows.append({"dataset": ds, "trajectory": t, "n_win": n_win,
                             "rho": rho_of.get(t, np.nan),
                             "rho_n": rho_of.get(t, np.nan) * n_win, "k_star": kstar})

        # ---- Q2 control: does max pooling fire before the dataset's own stop rule?
        _, mult = STOP_RULE[ds]
        if mult is not None:
            head("5", f"{ds.upper()} -- control: max pooling vs the dataset's OWN stop rule")
            print(f"    stop rule: {STOP_RULE[ds][0]}\n")
            # The comparison is only meaningful at a CONTROLLED false-alarm rate. A detector
            # tuned for balanced accuracy fires on ~29% of healthy XJTU records, and a detector
            # that fires everywhere trivially fires "early": that first version of this control
            # reported +45 records of warning, all of it bought with false alarms. Here the
            # threshold is instead the (1-alpha) quantile of the TRAINING trajectories' healthy
            # scores, and the alarm must be SUSTAINED to the end of the run (a lone early spike
            # is a false alarm, not a detection) -- the same rule J6 used for lead time.
            pk = window_peak(ds, len(rec), n_win)[keep]
            sc = pool(R, 1)

            def sustained(p: np.ndarray):
                if len(p) == 0 or p[-1] == 0:
                    return None
                r = len(p) - 1
                while r > 0 and p[r - 1] == 1:
                    r -= 1
                return int(r)

            for alpha in (0.05, 0.01):
                print(f"    --- operating point: alpha = {alpha:.0%} false alarms on training healthy "
                      f"records ---")
                print(f"    {'trajectory':24s} {'stop':>7s} {'sustained':>11s} {'earlier':>9s} {'FA':>6s}")
                gains, fas = [], []
                for te in folds:
                    t = traj[te][0]
                    v = pk[te]
                    base = np.median(v[:max(3, len(v) // 10)])
                    fire_stop = np.where(v > mult * base)[0]
                    a = int(fire_stop[0]) if len(fire_stop) else len(v)
                    tr = np.setdiff1d(np.arange(len(y)), te)
                    if y[tr].sum() in (0, len(tr)):
                        continue
                    thr = float(np.quantile(sc[tr][y[tr] == 0], 1 - alpha))
                    pp = (sc[te] >= thr).astype(int)
                    s = sustained(pp)
                    fa = float((pp[y[te] == 0] == 1).mean())
                    fas.append(fa)
                    gains.append((a - s) if s is not None else np.nan)
                    print(f"    {t:24s} {a:>7d} "
                          f"{(str(s) if s is not None else 'none'):>11s} "
                          f"{(format(a - s, '+d') if s is not None else '--'):>9s} {fa:>6.2f}")
                g = np.array(gains, float)
                n_ok = int(np.nansum(g > 0))
                print(f"    -> actual median FA {np.median(fas):.3f} | median lead "
                      f"{np.nanmedian(g):+.1f} records | {n_ok}/{len(g)} earlier | "
                      f"{int(np.isnan(g).sum())} with no sustained alarm\n")
                rows.append({"dataset": ds, "k": 1, "rule": f"control_alpha{alpha}",
                             "boundary_caught": n_ok, "boundary_total": len(g),
                             "TP": -1, "FP": -1, "FN": -1, "TN": -1,
                             "macro_f1": float(np.nanmedian(g)), "mcc": float(np.median(fas))})
            print("    READ THIS: the lead shrinks sharply as the false-alarm rate is controlled")
            print("    (XJTU: +45 records at FA 0.29, +9 at FA 0.02, +0.5 at FA 0.00). At a")
            print("    defensible operating point max pooling fires at essentially the same time")
            print("    as the documented stopping criterion, so the reviewer objection largely")
            print("    STANDS: report the lead WITH its false-alarm rate, never on its own.")

    res, law = pd.DataFrame(rows), pd.DataFrame(law_rows)

    head("6", "Verdict on the law  k* ~ rho * n")
    ok = law.dropna(subset=["rho_n", "k_star"])
    for ds in which:
        g = ok[ok.dataset == ds]
        if len(g) < 2:
            continue
        print(f"  {ds:5s} n={len(g):2d} trajectories | median rho*n {g.rho_n.median():.2f} "
              f"vs median k* {g.k_star.median():.1f}")
    multi = ok[ok.dataset != "nsk"]
    if len(multi) >= 5 and multi.rho_n.std() > 1e-9 and multi.k_star.std() > 1e-9:
        r = float(np.corrcoef(multi.rho_n, multi.k_star)[0, 1])
        print(f"\n  correlation(rho*n, k*) over {len(multi)} multi-bearing trajectories: r = {r:+.3f}")
        if r > 0.4:
            print("  >>> THE LAW TRAVELS. k* tracks rho*n outside NSK, with plain RMS scores and")
            print("      no model. C3 generalises from one trajectory to a sample.")
        else:
            print("  >>> THE LAW DOES NOT TRAVEL. k* does not track rho*n here. Report C3 as an")
            print("      NSK-specific mechanism and say so; do not claim generality.")
    else:
        print("\n  Too few trajectories, or k*/rho*n has no spread at these bag sizes, to")
        print("  correlate. With 4 and 2 windows per record the sweep cannot resolve the law --")
        print("  this is the bag-size limit stated at the top of the file, not a null result.")

    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "j8b_pooling_sweep.csv", index=False)
    law.to_csv(TABLES / "j8b_law.csv", index=False)
    print(f"\n{BAR}\n  WROTE  results/tables/j8b_pooling_sweep.csv ({len(res)} rows)"
          f"\n         results/tables/j8b_law.csv            ({len(law)} rows)\n{BAR}")
    return 0


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    raise SystemExit(main(a or ["nsk", "xjtu", "ims"], "--check" in sys.argv, "--refresh" in sys.argv))
