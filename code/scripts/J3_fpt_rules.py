"""J3 -- label the same data with EIGHT PUBLISHED First-Prediction-Time rules.

Why this replaces the h3/h5/h8 sweep
------------------------------------
C2 currently rests on three horizons -- h3, h5, h8 -- that we chose ourselves. A reviewer
will ask why those numbers, and "because we picked them" ends the argument. The fix is not
a denser sweep of our own parameter; it is to stop using our parameter at all and label the
data with rules the literature already treats as standard. Then the claim changes from

    "results are sensitive to a horizon we chose"          (interesting)
to
    "two rules, both published, both treated as standard, applied to the same trajectory,
     put the boundary in different places and reverse the ranking of methods"   (indicting)

Each rule takes a health indicator per trajectory, decides a First Prediction Time T, and
labels every record from T onward as degraded. J3 emits the boundaries and the resulting
method performances; J4 turns them into the reversal analysis.

The rules
    3sig_rms        mean + 3 sd of the healthy baseline, on RMS          (Lei et al., the common one)
    3sig_kurt       same, on kurtosis
    ewma            EWMA control chart, lambda 0.2, L 3
    cusum           one-sided CUSUM, k = 0.5 sd, decision interval H = 5 sd
    ma2sig/ma3sig/ma5sig   moving average (w = 5) crossing mean + k sd
    pct02/pct05/pct10      last 2 / 5 / 10 percent of life (the convention used in J7/J8b)

PROTOCOL CHOICES, stated because they change the answer
    * The baseline is the first 20% of each trajectory, ASSUMED healthy. Every statistical
      rule needs such a window, and its length is itself an unexamined convention.
    * A crossing must persist for CONFIRM consecutive records to count. Without confirmation
      the statistical rules fire on single noise spikes and the comparison is meaningless.
    * Rules are applied PER TRAJECTORY, so each bearing gets its own baseline and boundary.
    * When a rule never fires on a trajectory, that trajectory is reported as "no FPT" and
      excluded from that rule's evaluation rather than silently given a default boundary.
      How often a rule fails to fire is itself a result.

Usage
    python scripts/J3_fpt_rules.py                # nsk xjtu ims
    python scripts/J3_fpt_rules.py nsk xjtu
    python scripts/J3_fpt_rules.py --check
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
BASE_FRAC = 0.20          # first 20% assumed healthy
MIN_RATE, MAX_RATE = 0.01, 0.50   # a usable label set lies between these positive rates
CONFIRM = 3               # consecutive records needed to confirm a crossing
BAR = "=" * 78

VIB = ["rms", "std", "p2p", "crest", "kurtosis", "absmax", "skew"]
TMP = ["bearing_temp_mean", "delta_temp_mean", "bearing_temp_slope"]


def head(step: str, title: str) -> None:
    print(f"\n{BAR}\nSTEP {step}  {title}\n{BAR}", flush=True)


def _first_confirmed(flag: np.ndarray) -> int | None:
    """First index where `flag` stays True for CONFIRM consecutive records."""
    run = 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        if run >= CONFIRM:
            return i - CONFIRM + 1
    return None


def fpt(hi_rms: np.ndarray, hi_kurt: np.ndarray, rule: str) -> int | None:
    """First Prediction Time for one trajectory, or None if the rule never fires."""
    n = len(hi_rms)
    if rule.startswith("pct"):
        return int(n * (1 - int(rule[3:]) / 100.0))
    h = hi_kurt if rule == "3sig_kurt" else hi_rms
    nb = max(5, int(n * BASE_FRAC))
    mu, sd = float(np.mean(h[:nb])), float(np.std(h[:nb]))
    if sd < 1e-12:
        sd = 1e-12
    if rule in ("3sig_rms", "3sig_kurt"):
        return _first_confirmed(h > mu + 3 * sd)
    if rule == "ewma":
        lam, L = 0.2, 3.0
        z, out = mu, np.empty(n)
        for i, v in enumerate(h):
            z = lam * v + (1 - lam) * z
            out[i] = z
        lim = mu + L * sd * np.sqrt(lam / (2 - lam))
        return _first_confirmed(out > lim)
    if rule == "cusum":
        k, H = 0.5 * sd, 5.0 * sd
        s, out = 0.0, np.empty(n)
        for i, v in enumerate(h):
            s = max(0.0, s + (v - mu) - k)
            out[i] = s
        return _first_confirmed(out > H)
    if rule.startswith("ma"):
        kk = float(rule[2])
        w = 5
        ma = pd.Series(h).rolling(w, min_periods=1).mean().to_numpy()
        return _first_confirmed(ma > mu + kk * sd)
    raise ValueError(rule)


RULES = ["3sig_rms", "3sig_kurt", "ewma", "cusum", "ma2sig", "ma3sig", "ma5sig",
         "pct02", "pct05", "pct10"]

IMS_FAILED = {"IMS_1st_test_B3", "IMS_1st_test_B4", "IMS_2nd_test_B1", "IMS_3rd_test_B3"}


def counts(y, p) -> dict:
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    m = metrics_from_counts(tp, fp, fn, tn)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "macro_f1": m["macro_f1"], "mcc": m["mcc"]}


def oof_thr(x, y, folds):
    p = np.zeros_like(y)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        if y[tr].sum() in (0, len(tr)):
            continue
        p[te] = predict_threshold(x[te], fit_threshold(x[tr], y[tr], "balanced_accuracy"))
    return p


def oof_rf(X, y, folds):
    p = np.zeros_like(y)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(np.unique(y[tr])) < 2:
            continue
        p[te] = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1,
                                       class_weight="balanced_subsample").fit(X[tr], y[tr]).predict(X[te])
    return p


def relativise(F: pd.DataFrame, traj: np.ndarray) -> pd.DataFrame:
    out = F.copy()
    for t in np.unique(traj):
        m = traj == t
        for c in F.columns:
            v = F.loc[m, c].to_numpy(float)
            b = np.median(v[:max(3, len(v) // 10)])
            out.loc[m, c] = v / (b if abs(b) > 1e-9 else 1.0)
    return out


def _score_rule(ds, rule, mode, y, Frel, cols, folds, excluded, n_dropped, perf_rows):
    """Score every method on one rule under one exclusion mode, appending to perf_rows.

    A rule that calls most of the trajectory degraded has not produced a health-state label at
    all, and comparing methods on it measures nothing. EWMA and CUSUM do exactly this (up to
    97% positive on XJTU) because they are designed to flag the ONSET of a small persistent
    shift, not to PARTITION a monotone trajectory -- using them as class labellers is a
    category error the literature makes routinely. Rules outside [MIN_RATE, MAX_RATE] are still
    evaluated but flagged degenerate, and J4 excludes them from the ranking analysis.

    Under mode="fixed" the rate is computed on retained records only, so the screen is applied
    to the rate the rule actually induces rather than one diluted by its own failures.
    """
    rate = y.sum() / len(y)
    degenerate = not (MIN_RATE <= rate <= MAX_RATE)
    if y.sum() < 3 or y.sum() == len(y):
        if mode == "fixed":
            print(f"    {rule:10s} {int(y.sum()):>6d}   -- too few positives, skipped")
        return
    common = {"dataset": ds, "rule": rule, "mode": mode, "n_records": len(y),
              "n_pos": int(y.sum()), "pos_rate": rate, "degenerate": degenerate,
              "excluded_trajectories": ";".join(excluded), "n_excluded_records": int(n_dropped)}
    scores = {}
    for c in cols:
        r = counts(y, oof_thr(Frel[c].to_numpy(float), y, folds))
        scores[f"thr:{c}"] = r["macro_f1"]
        perf_rows.append({**common, "method": f"thr:{c}", **r})
    r = counts(y, oof_rf(Frel.to_numpy(float), y, folds))
    scores["rf:all"] = r["macro_f1"]
    perf_rows.append({**common, "method": "rf:all", **r})

    best = max(scores, key=scores.get)
    methods = [f"thr:{c}" for c in cols] + ["rf:all"]
    line = "".join(f"{scores[m]:>9.3f}" for m in methods)
    tag = "  <== DEGENERATE" if degenerate else ""
    ex = f"  excl {len(excluded)} traj" if excluded and mode == "fixed" else ""
    print(f"    {rule:10s} {mode:7s} {int(y.sum()):>6d} ({rate:>4.0%}) {line}   "
          f"best={best.replace('thr:', '')}{tag}{ex}")


def merge_write(df: pd.DataFrame, path: Path, which: list[str]) -> int:
    """Write df, PRESERVING rows for datasets this run did not compute.

    Previously a run limited to one dataset overwrote the whole file, silently deleting the
    other datasets' rows. Anything downstream (J4, J9, J11) then read a truncated table.
    """
    if path.exists():
        try:
            old = pd.read_csv(path)
            if "dataset" in old.columns:
                keep = old[~old.dataset.isin(which)]
                if len(keep):
                    df = pd.concat([keep, df], ignore_index=True)
        except Exception:
            pass
    df.to_csv(path, index=False)
    return len(df)


def main(which: list[str], check_only: bool) -> int:
    print(BAR)
    print("J3  EIGHT PUBLISHED FPT RULES ON THE SAME DATA".center(78))
    print("stop sweeping our own horizon; use the rules the field already treats as standard".center(78))
    print(BAR)

    head("1", "Inputs")
    meta = {}
    for ds in which:
        if ds == "femto":
            # FEMTO was never built into the Colab bundle and its feature cache uses different
            # column names; adapt it here rather than leave the dataset out of J3 entirely.
            f = TABLES / "j7c_femto_features.csv"
            if not f.exists():
                print(f"  femto: MISSING {f.name} -- run scripts/J7c_femto_channel_test.py first")
                return 1
            d = pd.read_csv(f).sort_values(["bearing", "record_index"]).reset_index(drop=True)
            rec = pd.DataFrame({"trajectory_id": d.bearing.to_numpy(),
                                "condition_id": d.condition.to_numpy(),
                                "record_index": d.record_index.to_numpy(),
                                "life_fraction": d.life_fraction.to_numpy()})
            # Vibration only. Thermal is deliberately excluded: Bearing2_6's thermal channel is
            # corrupt (J7c), and mapping it in would silently drop a bearing the rest of the
            # analysis keeps.
            F = pd.DataFrame({"rms": d.v_rms.to_numpy(), "kurtosis": d.v_kurtosis.to_numpy(),
                              "crest": d.v_crest.to_numpy(), "p2p": d.v_p2p.to_numpy(),
                              "absmax": d.v_absmax.to_numpy()})
            meta[ds] = (rec, F, np.ones(len(rec), bool))
            print(f"  {ds:5s} {len(rec):6d} records | {rec.trajectory_id.nunique():2d} "
                  f"trajectories | {len(F.columns)} features (vibration only, no std/skew)")
            continue
        f = TABLES / f"j7_features_{ds}.csv"
        r = BUNDLE / ds / "records.csv"
        if not f.exists() or not r.exists():
            print(f"  {ds}: MISSING {f.name if not f.exists() else r.name} "
                  f"-- run scripts/J7_circularity.py (caches features) first")
            return 1
        rec, F = pd.read_csv(r), pd.read_csv(f)
        keep = (rec.trajectory_id.isin(IMS_FAILED).to_numpy() if ds == "ims"
                else np.ones(len(rec), bool))
        meta[ds] = (rec, F, keep)
        if ds != "femto":
            print(f"  {ds:5s} {int(keep.sum()):6d} records | "
                  f"{rec.loc[keep].trajectory_id.nunique():2d} trajectories | "
                  f"{len([c for c in F.columns])} features")
    print(f"\n  baseline = first {BASE_FRAC:.0%} of each trajectory (assumed healthy)")
    print(f"  a crossing must persist {CONFIRM} consecutive records to count")
    print(f"  {len(RULES)} rules: {RULES}")
    if check_only:
        print("\n  --check: inputs OK.")
        return 0

    fpt_rows, perf_rows = [], []
    for ds in which:
        rec, F, keep = meta[ds]
        sub = rec.loc[keep].reset_index(drop=True)
        Fk = F.loc[keep].reset_index(drop=True)
        traj = sub.trajectory_id.to_numpy()
        cols = [c for c in VIB + TMP if c in Fk.columns]
        Frel = relativise(Fk[cols], traj)
        folds = ([np.where(traj == t)[0] for t in np.unique(traj)] if len(np.unique(traj)) > 1
                 else [np.where(pd.read_csv(ROOT / "data/metadata/fold_assignments.csv")
                                .sort_values("hour_id").fold_id.to_numpy() == k)[0] for k in range(5)])

        head("2", f"{ds.upper()} -- where each rule puts the boundary (fraction of life)")
        print(f"    {'rule':10s} " + "".join(f"{t.replace('Bearing', 'B').replace('IMS_', '').replace('_test', ''):>9s}"
                                             for t in np.unique(traj)[:8]) + "   fired")
        labels = {}
        for rule in RULES:
            y = np.zeros(len(sub), int)
            fired_mask = np.zeros(len(sub), bool)   # records of trajectories the rule fired on
            excluded = []
            fr, nfire = [], 0
            for t in np.unique(traj):
                m = np.where(traj == t)[0]
                i = fpt(Fk["rms"].to_numpy()[m], Fk["kurtosis"].to_numpy()[m], rule)
                if i is None:
                    # The rule did not locate an onset on this trajectory. That is NOT a claim
                    # that the bearing stayed healthy, so these records must not enter the label
                    # set as negatives -- see E1 in the paper plan.
                    fr.append(np.nan)
                    excluded.append(str(t))
                    continue
                nfire += 1
                fired_mask[m] = True
                y[m[i:]] = 1
                fr.append(i / len(m))
                fpt_rows.append({"dataset": ds, "trajectory": t, "rule": rule,
                                 "fpt_index": int(i), "n_records": len(m),
                                 "fpt_fraction": i / len(m)})
            labels[rule] = (y, fired_mask, excluded)
            shown = "".join(("      n/a" if np.isnan(v) else f"{v:>9.2f}") for v in fr[:8])
            print(f"    {rule:10s} {shown}   {nfire}/{len(np.unique(traj))}")

        head("3", f"{ds.upper()} -- method performance under each rule (leave-one-trajectory-out)")
        methods = [f"thr:{c}" for c in cols] + ["rf:all"]
        print(f"    {'rule':10s} {'mode':7s} {'pos':>6s} {'rate':>6s} "
              + "".join(f"{m.replace('thr:', '')[:8]:>9s}" for m in methods))
        for rule in RULES:
            y_all, fired_mask, excluded = labels[rule]
            # legacy keeps every record (the defect); fixed drops non-firing trajectories
            for mode in ("legacy", "fixed"):
                sel = (np.ones(len(sub), bool) if mode == "legacy" else fired_mask)
                idx = np.where(sel)[0]
                if len(idx) == 0:
                    continue
                y = y_all[idx]
                Fr = Frel.iloc[idx].reset_index(drop=True)
                traj_r = traj[idx]
                if len(np.unique(traj_r)) > 1:
                    folds_r = [np.where(traj_r == t)[0] for t in np.unique(traj_r)]
                else:
                    fa = (pd.read_csv(ROOT / "data/metadata/fold_assignments.csv")
                          .sort_values("hour_id").fold_id.to_numpy())
                    fa_r = fa[idx]
                    folds_r = [np.where(fa_r == k)[0] for k in range(5)]
                _score_rule(ds, rule, mode, y, Fr, cols, folds_r, excluded,
                            len(sub) - len(idx), perf_rows)
            continue

    ft, pf = pd.DataFrame(fpt_rows), pd.DataFrame(perf_rows)

    head("4", "How far apart do the published rules put the boundary?")
    for ds in which:
        g = ft[ft.dataset == ds]
        if g.empty:
            continue
        st = g.groupby("trajectory").fpt_fraction.agg(["min", "max", "count"])
        st["spread"] = st["max"] - st["min"]
        print(f"  {ds:5s} median spread of FPT across rules, within a trajectory: "
              f"{st.spread.median():.2f} of life  (worst {st.spread.max():.2f})")
        never = g.groupby("rule").size()
        n_traj = ft[ft.dataset == ds].trajectory.nunique()
        miss = {r: n_traj - int(never.get(r, 0)) for r in RULES if n_traj - int(never.get(r, 0)) > 0}
        if miss:
            print(f"        rules that failed to fire on some trajectory: {miss}")

    head("5", "E1 -- what the exclusion defect was doing to the conclusions")
    demo = []
    if not pf.empty and "mode" in pf.columns:
        print(f"    {'dataset':7s} {'rule':10s} {'rate legacy':>12s} {'rate fixed':>11s} "
              f"{'screen':>16s} {'winner legacy':>14s} {'winner fixed':>13s} {'margin fix':>11s}")
        for (ds, rule), g in pf.groupby(["dataset", "rule"], sort=True):
            L, Fx = g[g["mode"] == "legacy"], g[g["mode"] == "fixed"]
            if L.empty or Fx.empty:
                continue
            wl = L.loc[L.macro_f1.idxmax(), "method"]
            wf = Fx.loc[Fx.macro_f1.idxmax(), "method"]
            top2 = Fx.macro_f1.nlargest(2).to_numpy()
            marg = float(top2[0] - top2[1]) if len(top2) > 1 else np.nan
            dl, df_ = bool(L.degenerate.iloc[0]), bool(Fx.degenerate.iloc[0])
            screen = ("unchanged" if dl == df_ else
                      ("NOW DEGENERATE" if df_ else "now usable"))
            demo.append({"dataset": ds, "rule": rule,
                         "pos_rate_legacy": float(L.pos_rate.iloc[0]),
                         "pos_rate_fixed": float(Fx.pos_rate.iloc[0]),
                         "degenerate_legacy": dl, "degenerate_fixed": df_,
                         "screen_change": screen,
                         "winner_legacy": wl, "winner_fixed": wf,
                         "winner_changed": wl != wf,
                         "margin_fixed": marg,
                         "excluded_trajectories": Fx.excluded_trajectories.iloc[0],
                         "n_excluded_records": int(Fx.n_excluded_records.iloc[0])})
            flag = "  <<<" if (screen != "unchanged" or wl != wf) else ""
            print(f"    {ds:7s} {rule:10s} {L.pos_rate.iloc[0]:>12.4f} "
                  f"{Fx.pos_rate.iloc[0]:>11.4f} {screen:>16s} "
                  f"{wl.replace('thr:', ''):>14s} {wf.replace('thr:', ''):>13s} "
                  f"{marg:>11.4f}{flag}")
        dm = pd.DataFrame(demo)
        if len(dm):
            print()
            print(f"    rules whose degeneracy screen flipped : "
                  f"{int((dm.screen_change != 'unchanged').sum())}/{len(dm)}")
            print(f"    rules whose winning method changed    : "
                  f"{int(dm.winner_changed.sum())}/{len(dm)}")
            ties = dm[(dm.margin_fixed < 0.01) & dm.margin_fixed.notna()]
            print(f"    winners that are ties (<0.01) after fix: {len(ties)}")
            merge_write(dm, TABLES / "j14_defect_demo.csv", which)

    TABLES.mkdir(parents=True, exist_ok=True)
    n_ft = merge_write(ft, TABLES / "j3_fpt_times.csv", which)
    n_pf = merge_write(pf, TABLES / "j3_performance.csv", which)
    print(f"\n{BAR}\n  WROTE  results/tables/j3_fpt_times.csv   ({n_ft} rows, merged)"
          f"\n         results/tables/j3_performance.csv ({n_pf} rows, legacy+fixed)"
          f"\n         results/tables/j14_defect_demo.csv ({len(demo)} rows)"
          f"\n\n  NOTE  j3_performance.csv now carries a 'mode' column. Downstream consumers"
          f"\n        MUST filter mode=='fixed'; J4/J9 were written before this column existed."
          f"\n\n  Next: python scripts/J4_ranking_reversal.py\n{BAR}")
    return 0


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    raise SystemExit(main(a or ["nsk", "xjtu", "ims", "femto"], "--check" in sys.argv))
