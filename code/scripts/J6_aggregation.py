"""J6 -- Aggregation-aware decision rules for weakly-labelled run-to-failure records.

J5 established that hour 124 is a POOLING failure: every model on every seed assigned
probability 1.000 to at least one window of that burst, while mean pooling over the 50
windows produced a median of 0.096 and therefore called the record healthy.

J6 asks the follow-up question properly:

    Does a different pooling rule recover the transition record at an acceptable
    false-alarm cost -- when every rule is given a FAIRLY FITTED decision threshold?

The fairness point matters and is the reason J5's numbers were only indicative. mean,
max and top-k are different statistics with different distributions; comparing all of
them against a fixed 0.5 cut favours whichever rule happens to sit highest. Here every
rule gets its own threshold, swept on the TRAINING records of each fold and applied to
the held-out records -- the same protocol the paper used for the RMS threshold.

Reported per (model, seed, horizon, rule):
    macro-F1, MCC, TP/FP/FN/TN
    detection lead time  = hours between the first SUSTAINED alarm and the end of the run
    false-alarm rate     = fraction of pre-terminal records raising a non-sustained alarm

Lead time is the operationally meaningful quantity: recovering the transition record is
worth an extra hour of warning, which macro-F1 expresses only indirectly.

Pre-specified primary rule: top-5 (top 10% of a 50-window burst). The k-sweep is reported
as a sensitivity analysis, NOT tuned per fold -- see the note printed in step 6.

Usage
    python scripts/J6_aggregation.py            # full run
    python scripts/J6_aggregation.py --check    # verify inputs only
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import metrics_from_counts  # noqa: E402
from src.threshold import threshold_oof  # noqa: E402

CKPT = ROOT / "results" / "checkpoints"
META = ROOT / "data" / "metadata"
TABLES = ROOT / "results" / "tables"

N_WIN, N_REC = 50, 129
MODELS = ("cnn1d", "resnet", "fusion")
SEEDS = (0, 21, 42, 84, 123)
HORIZONS = ("h3", "h5", "h8")
PRIMARY = "top5"
BAR = "=" * 78


def head(step: str, title: str) -> None:
    print(f"\n{BAR}\nSTEP {step}/6  {title}\n{BAR}", flush=True)


# ---------------------------------------------------------------- pooling rules
def pool(mat: np.ndarray, rule: str) -> np.ndarray:
    """mat [n_records, n_windows] -> one score per record."""
    if rule == "mean":
        return mat.mean(axis=1)
    if rule == "max":
        return mat.max(axis=1)
    if rule.startswith("top"):
        k = int(rule[3:])
        return np.sort(mat, axis=1)[:, -k:].mean(axis=1)
    if rule == "noisyor":
        p = np.clip(mat, 1e-6, 1 - 1e-6)
        return 1.0 - np.exp(np.log1p(-p).sum(axis=1))
    raise ValueError(rule)


RULES = ("mean", "max", "top3", "top5", "top10", "noisyor")


# ---------------------------------------------------------------- loading
def load_probs(folds: pd.DataFrame) -> dict[tuple[str, str, int], np.ndarray]:
    """(horizon, model, seed) -> [129, 50] window probabilities, only complete runs."""
    out = {}
    for hz in HORIZONS:
        for m in MODELS:
            for s in SEEDS:
                # h5 came from the convergence audit (e7_*), h3/h8 from the horizon grid (e12_*)
                fmt = (f"e7_{m}_seed{s}_global_fold%d_testprob.npy" if hz == "h5"
                       else f"e12_{m}_{hz}_seed{s}_fold%d_testprob.npy")
                mat, ok = np.full((N_REC, N_WIN), np.nan), True
                for f in range(5):
                    p = CKPT / (fmt % f)
                    if not p.exists():
                        ok = False
                        break
                    w = np.load(p)
                    hrs = np.sort(folds.loc[folds.fold_id == f, "hour_id"].to_numpy())
                    if w.size != len(hrs) * N_WIN:
                        ok = False
                        break
                    mat[hrs] = w.reshape(len(hrs), N_WIN)
                if ok:
                    out[(hz, m, s)] = mat
    return out


# ---------------------------------------------------------------- lead time
def lead_time(pred: np.ndarray) -> int:
    """Hours between the first SUSTAINED alarm and the end of the run (records are hourly).

    Sustained = every record from that point to the end is an alarm. A lone early spike is
    a false alarm, not a detection, so it must not be rewarded with a huge lead time.
    """
    if pred[-1] == 0:
        return 0
    r = len(pred) - 1
    while r > 0 and pred[r - 1] == 1:
        r -= 1
    return int(len(pred) - 1 - r)


def main(check_only: bool = False) -> int:
    print(BAR)
    print("J6  AGGREGATION-AWARE DECISION RULES".center(78))
    print("does a better pooling rule recover the transition record, fairly compared?".center(78))
    print(BAR)

    # ---------------------------------------------------------------- 1
    head("1", "Inputs and run coverage")
    folds = pd.read_csv(META / "fold_assignments.csv").sort_values("hour_id").reset_index(drop=True)
    probs = load_probs(folds)
    print(f"  fold_assignments.csv   {len(folds)} records, 5 folds")
    for hz in HORIZONS:
        got = [(m, s) for (h, m, s) in probs if h == hz]
        n_pos = int(folds[f"label_{hz}"].sum())
        first = int(folds.loc[folds[f"label_{hz}"] == 1, "hour_id"].min())
        print(f"  {hz}:  {len(got):2d}/15 complete runs | {n_pos} terminal records, "
              f"starting at h{first} | ideal lead time = {128 - first} h")
        for m in MODELS:
            ss = sorted(s for (mm, s) in got if mm == m)
            print(f"        {m:8s} seeds {ss if ss else '-- none --'}")
    if not probs:
        print("\n  ERROR: no complete run found.")
        return 1
    if check_only:
        print("\n  --check: inputs OK, nothing computed.")
        return 0

    # ---------------------------------------------------------------- 2
    head("2", "Pooling rules and a saturation check")
    print("  mean     average of all 50 window probabilities   (what the paper used)")
    print("  max      single most confident window")
    print("  topK     mean of the K most confident windows     (K = 3, 5, 10)")
    print("  noisyor  1 - prod(1 - p_i)\n")
    demo = probs[sorted(probs)[0]]
    print(f"  score distribution on {sorted(probs)[0]}  (129 records):")
    print(f"    {'rule':9s} {'min':>8s} {'median':>8s} {'max':>8s}  {'% of records >0.99':>19s}")
    for r in RULES:
        v = pool(demo, r)
        print(f"    {r:9s} {v.min():>8.4f} {np.median(v):>8.4f} {v.max():>8.4f}  "
              f"{100 * (v > 0.99).mean():>18.1f}%")
    sat = pool(demo, "noisyor")
    if (sat > 0.99).mean() > 0.5:
        print("\n  NOTE: noisy-OR saturates -- with 50 windows, a single confident instance")
        print("        drives the product to 1. It is unusable at this bag size, which is")
        print("        itself a reportable finding about bag-size / pooling-rule matching.")

    # ---------------------------------------------------------------- 3
    head("3", "Per-rule thresholds fitted on TRAINING folds  <-- the fix over J5")
    print("  Each rule gets its own threshold, swept on train records only, then applied")
    print("  to held-out records. J5 compared every rule at a fixed 0.5 cut, which is not")
    print("  a fair comparison between statistics with different distributions.\n")
    y = pd.Series(folds["label_h5"].to_numpy(int), index=folds.hour_id.to_numpy())
    fid = pd.Series(folds.fold_id.to_numpy(int), index=folds.hour_id.to_numpy())
    ex = probs[sorted(probs)[0]]
    print(f"    {'rule':9s} " + "  ".join(f"fold{f}" for f in range(5)))
    for r in RULES:
        s = pd.Series(pool(ex, r), index=folds.hour_id.to_numpy())
        o = threshold_oof(s, y, fid, metric="balanced_accuracy")
        t = o.groupby("fold_id")["fold_threshold"].first()
        print(f"    {r:9s} " + "  ".join(f"{t[f]:.3f}" for f in range(5)))
    print("\n  (thresholds above are for the first run only, as an illustration;")
    print("   every run gets its own)")

    # ---------------------------------------------------------------- 4
    head("4", "Metrics for every (horizon, model, seed, rule)")
    rows = []
    for (hz, m, s), mat in sorted(probs.items()):
        yy = pd.Series(folds[f"label_{hz}"].to_numpy(int), index=folds.hour_id.to_numpy())
        first_term = int(folds.loc[folds[f"label_{hz}"] == 1, "hour_id"].min())
        for r in RULES:
            sc = pd.Series(pool(mat, r), index=folds.hour_id.to_numpy())
            o = threshold_oof(sc, yy, fid, metric="balanced_accuracy").sort_values("hour_id")
            t_, p_ = o["true_label"].to_numpy(int), o["pred_label"].to_numpy(int)
            tp = int(((t_ == 1) & (p_ == 1)).sum()); fp = int(((t_ == 0) & (p_ == 1)).sum())
            fn = int(((t_ == 1) & (p_ == 0)).sum()); tn = int(((t_ == 0) & (p_ == 0)).sum())
            mt = metrics_from_counts(tp, fp, fn, tn)
            lt = lead_time(p_)
            pre = p_[:first_term]
            rows.append({"horizon": hz, "model": m, "seed": s, "rule": r,
                         "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                         "macro_f1": mt["macro_f1"], "mcc": mt["mcc"],
                         "lead_time_h": lt, "ideal_lead_h": 128 - first_term,
                         "false_alarm_rate": float(pre.mean()) if len(pre) else 0.0,
                         "caught_transition": int(p_[first_term] == 1)})
    res = pd.DataFrame(rows)

    for hz in HORIZONS:
        g = res[res.horizon == hz]
        if g.empty:
            continue
        n = g.groupby("rule").size().iloc[0]
        print(f"\n  {hz}  ({n} runs)")
        print(f"    {'rule':9s} {'macroF1 med':>12s} {'[min,max]':>16s} {'MCC med':>8s} "
              f"{'lead h':>7s} {'FA rate':>8s} {'caught transition':>18s}")
        for r in RULES:
            q = g[g.rule == r]
            print(f"    {r:9s} {q.macro_f1.median():>12.4f} "
                  f"{'[%.3f, %.3f]' % (q.macro_f1.min(), q.macro_f1.max()):>16s} "
                  f"{q.mcc.median():>8.4f} {q.lead_time_h.median():>7.1f} "
                  f"{q.false_alarm_rate.median():>8.3f} "
                  f"{'%d/%d' % (q.caught_transition.sum(), len(q)):>18s}")
        print(f"    {'(ideal)':9s} {1.0:>12.4f} {'':>16s} {1.0:>8.4f} "
              f"{g.ideal_lead_h.iloc[0]:>7.1f} {0.0:>8.3f} {'%d/%d' % (len(q), len(q)):>18s}")

    # ---------------------------------------------------------------- 5
    head("5", "Lead time vs false alarms -- the operational trade")
    print("  Extra warning bought, and what it costs, relative to mean pooling.\n")
    print(f"    {'horizon':8s} {'rule':9s} {'d lead (h)':>11s} {'d FA rate':>10s} "
          f"{'d macroF1':>10s}  verdict")
    for hz in HORIZONS:
        g = res[res.horizon == hz]
        if g.empty:
            continue
        base = g[g.rule == "mean"]
        b_lead, b_fa, b_f1 = base.lead_time_h.median(), base.false_alarm_rate.median(), base.macro_f1.median()
        for r in RULES:
            if r == "mean":
                continue
            q = g[g.rule == r]
            dl = q.lead_time_h.median() - b_lead
            dfa = q.false_alarm_rate.median() - b_fa
            df1 = q.macro_f1.median() - b_f1
            if dl > 0 and dfa <= 0.01 and df1 >= 0:
                v = "BUYS WARNING, ~FREE"
            elif dl > 0 and df1 >= 0:
                v = "buys warning, some cost"
            elif dl > 0:
                v = "buys warning, F1 drops"
            elif dl == 0 and df1 > 0:
                v = "no extra warning, better F1"
            else:
                v = "no gain"
            print(f"    {hz:8s} {r:9s} {dl:>+11.1f} {dfa:>+10.3f} {df1:>+10.4f}  {v}")

    # ---------------------------------------------------------------- 6
    head("6", "Verdict")
    p = res[(res.rule == PRIMARY)]
    m0 = res[(res.rule == "mean")]
    print(f"  Pre-specified primary rule: {PRIMARY} (top 10% of a 50-window burst)\n")
    print(f"    {'horizon':8s} {'mean pooling':>28s} {'top-5 pooling':>28s}")
    for hz in HORIZONS:
        a, b = m0[m0.horizon == hz], p[p.horizon == hz]
        if a.empty:
            continue
        print(f"    {hz:8s} "
              f"{'F1 %.4f  lead %.1fh  FA %.3f' % (a.macro_f1.median(), a.lead_time_h.median(), a.false_alarm_rate.median()):>28s} "
              f"{'F1 %.4f  lead %.1fh  FA %.3f' % (b.macro_f1.median(), b.lead_time_h.median(), b.false_alarm_rate.median()):>28s}")

    print("\n  k-sweep (sensitivity, NOT tuned per fold -- k was pre-specified at 5):")
    for hz in HORIZONS:
        g = res[res.horizon == hz]
        if g.empty:
            continue
        line = "  ".join(f"{r}={g[g.rule == r].macro_f1.median():.3f}"
                         for r in ("max", "top3", "top5", "top10", "mean"))
        print(f"    {hz}:  {line}")
    print("\n  Because k was fixed in advance rather than fitted inside each fold, the")
    print("  k-sweep carries a mild optimism. Report the pre-specified k as the headline")
    print("  and the sweep as sensitivity; say so explicitly in the paper.")

    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "j6_aggregation.csv", index=False)
    print(f"\n{BAR}\n  WROTE  results/tables/j6_aggregation.csv  ({len(res)} rows)\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(check_only="--check" in sys.argv))
