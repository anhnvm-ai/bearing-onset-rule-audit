"""J10 -- trajectory-level bootstrap confidence intervals.

Why the resampling unit is the TRAJECTORY
-----------------------------------------
The conference paper conceded that it could not put an interval on anything: with one
bearing, the 129 hourly records are serially dependent observations of a single physical
object, so resampling records would treat correlated measurements as independent and
produce intervals that are far too narrow. That concession was correct and it is the main
thing the multi-bearing datasets now fix.

Here the unit of resampling is the bearing. Resampling bearings with replacement respects
the only independence the data actually offers, and the resulting interval answers the
question a reviewer cares about: *if the experiment had drawn a different set of bearings
from the same population, how different would the number be?*

What is bootstrapped, and what is not
    DONE HERE -- everything measured on the EVALUATION side, where out-of-fold predictions
    already exist and a resample costs nothing: per-method macro-F1, the boundary term
    (spread of the best method across published rules), and whether a rule's winner is
    separable from its runner-up at all.

    NOT HERE -- the leakage terms from J1b. Those vary the TRAINING set, so a bootstrap
    would require refitting inside every resample. J1b's 20 repeats already give a
    distribution over random splits and its percentiles are reported instead; that is a
    different interval (over splits, not over bearings) and is labelled as such.

NSK cannot be bootstrapped at all -- one trajectory, nothing to resample. The script says
so rather than quietly resampling its records and inventing an interval.

Outputs
    results/tables/j10_bootstrap_ci.csv     per (dataset, rule, method): median + 95% CI
    results/tables/j10_winner_test.csv      per (dataset, rule): winner vs runner-up
    results/logs/j10_run.log

Usage
    python scripts/J10_bootstrap_ci.py
    python scripts/J10_bootstrap_ci.py --check
    B=500 python scripts/J10_bootstrap_ci.py      # quick pass
"""
from __future__ import annotations

import os
import sys
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
B = int(os.environ.get("B", "2000"))
LIFE_PCT = 0.05
RULES = ["pct05", "3sig_rms", "3sig_kurt", "ma3sig", "ma5sig", "pct02", "pct10"]
IMS_FAILED = {"IMS_1st_test_B3", "IMS_1st_test_B4", "IMS_2nd_test_B1", "IMS_3rd_test_B3"}
# Bearing2_6: thermal channel corrupt, see J7c. FEMTO_EXCLUDE overrides the set for sensitivity runs
# (Appendix A: Bearing2_7 passes J7c's median check but exceeds 150 C on 36.8% of its records).
FEMTO_BAD = set(filter(None, os.environ.get("FEMTO_EXCLUDE", "Bearing2_6").split(",")))
# DATASETS restricts the run; TAG writes j10<TAG>_*.csv and j10<TAG>_run.log so the canonical outputs stay untouched.
DATASETS = tuple(filter(None, os.environ.get("DATASETS", "nsk,xjtu,ims,femto").split(",")))
TAG = os.environ.get("TAG", "")
BAR = "=" * 78
_fh = None


# A long run must not die on a formatting character: when stdout is redirected Windows falls
# back to cp1252, which cannot encode many of the glyphs that look fine in a terminal.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def log(m: str = "", screen: bool = True) -> None:
    if screen:
        print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


def head(s: str, t: str) -> None:
    log(f"\n{BAR}\nSTEP {s}  {t}\n{BAR}")


# ---------------------------------------------------------------- labels
def fpt_label(rms: np.ndarray, kurt: np.ndarray, life: np.ndarray,
              traj: np.ndarray, rule: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Labels, plus a mask of records the rule is entitled to speak about.

    A trajectory on which the rule never confirms a crossing is NOT evidence that the bearing
    stayed healthy -- the rule simply failed to locate an onset. Leaving those records in as
    negatives hands every method a block of free true negatives and, in this script, feeds
    them into the bootstrap. They are excluded (E1); percentile rules always fire by
    construction, so nothing is excluded for them.
    """
    if rule.startswith("pct"):
        y = (life > 1 - int(rule[3:]) / 100.0).astype(int)
        return y, np.ones(len(y), bool), []
    h = kurt if rule == "3sig_kurt" else rms
    k = 3.0 if rule.startswith("3sig") else float(rule[2])
    y = np.zeros(len(h), int)
    keep = np.zeros(len(h), bool)
    excluded: list[str] = []
    for t in np.unique(traj):
        m = np.where(traj == t)[0]
        v = h[m]
        nb = max(5, int(len(v) * 0.20))
        mu, sd = v[:nb].mean(), max(v[:nb].std(), 1e-12)
        s = pd.Series(v).rolling(5, min_periods=1).mean().to_numpy() if rule.startswith("ma") else v
        run, fired = 0, False
        for i, f in enumerate(s > mu + k * sd):
            run = run + 1 if f else 0
            if run >= 3:
                y[m[i - 2:]] = 1
                keep[m] = True
                fired = True
                break
        if not fired:
            excluded.append(str(t))
    return y, keep, excluded


# ---------------------------------------------------------------- OOF
def oof_all(F: pd.DataFrame, y: np.ndarray, folds: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Leave-one-trajectory-out predictions for every single-feature threshold and for RF."""
    out = {}
    for c in F.columns:
        p = np.zeros_like(y)
        x = F[c].to_numpy(float)
        for te in folds:
            tr = np.setdiff1d(np.arange(len(y)), te)
            if y[tr].sum() in (0, len(tr)):
                continue
            p[te] = predict_threshold(x[te], fit_threshold(x[tr], y[tr], "balanced_accuracy"))
        out[f"thr:{c}"] = p
    p = np.zeros_like(y)
    X = F.to_numpy(float)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(np.unique(y[tr])) < 2:
            continue
        p[te] = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1,
                                       class_weight="balanced_subsample").fit(X[tr], y[tr]).predict(X[te])
    out["rf:all"] = p
    return out


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    return metrics_from_counts(tp, fp, fn, tn)["macro_f1"]


def boot_indices(traj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample TRAJECTORIES with replacement; return the concatenated record indices."""
    u = np.unique(traj)
    picked = rng.choice(u, len(u), replace=True)
    return np.concatenate([np.where(traj == t)[0] for t in picked])


def relativise(F: pd.DataFrame, traj: np.ndarray) -> pd.DataFrame:
    out = F.copy()
    for t in np.unique(traj):
        m = traj == t
        for c in F.columns:
            v = F.loc[m, c].to_numpy(float)
            b = np.median(v[:max(3, len(v) // 10)])
            out.loc[m, c] = v / (b if abs(b) > 1e-9 else 1.0)
    return out


# ---------------------------------------------------------------- datasets
def load_dataset(ds: str):
    """-> (features DataFrame, trajectory array, life fraction, n_trajectories) or None."""
    if ds == "femto":
        f = TABLES / "j7c_femto_features.csv"
        if not f.exists():
            return None
        d = pd.read_csv(f)
        d = d[~d.bearing.isin(FEMTO_BAD)].reset_index(drop=True)
        cols = ["v_rms", "v_kurtosis", "v_crest", "v_p2p", "v_absmax", "t_median", "t_slope"]
        return d[cols], d.bearing.to_numpy(), d.life_fraction.to_numpy()
    f = TABLES / f"j7_features_{ds}.csv"
    r = BUNDLE / ds / "records.csv"
    if not f.exists() or not r.exists():
        return None
    rec, F = pd.read_csv(r), pd.read_csv(f)
    keep = rec.trajectory_id.isin(IMS_FAILED).to_numpy() if ds == "ims" else np.ones(len(rec), bool)
    cols = [c for c in F.columns if not c.startswith("Unnamed")]
    return (F.loc[keep, cols].reset_index(drop=True),
            rec.loc[keep, "trajectory_id"].to_numpy(),
            rec.loc[keep, "life_fraction"].to_numpy())


def main(check_only: bool) -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    # A --check run must never truncate the log of the last real run.
    _fh = None if check_only else open(LOGS / f"j10{TAG}_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J10  TRAJECTORY-LEVEL BOOTSTRAP".center(78))
    log("resample bearings, not records -- the only independence the data offers".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S} | B={B} resamples | labels pct05 + 6 rules")
    log(f"datasets {DATASETS} | FEMTO excluded {sorted(FEMTO_BAD)} | tag '{TAG}'")

    head("1", "Inputs and what can be bootstrapped")
    data = {}
    for ds in DATASETS:
        got = load_dataset(ds)
        if got is None:
            log(f"  {ds:6s} MISSING feature cache -- skipped")
            continue
        F, traj, life = got
        n_traj = len(np.unique(traj))
        data[ds] = (F, traj, life)
        note = ("CANNOT BOOTSTRAP -- one trajectory, nothing to resample" if n_traj < 2
                else f"{n_traj} trajectories -> resample with replacement")
        log(f"  {ds:6s} {len(F):6d} records | {len(F.columns)} features | {note}")
    if check_only:
        log("\n  --check: inputs OK.")
        return 0

    rows, win_rows = [], []
    for ds, (F, traj, life) in data.items():
        n_traj = len(np.unique(traj))
        if n_traj < 2:
            head("2", f"{ds.upper()} -- skipped")
            log("  One trajectory. Resampling its records would treat serially dependent")
            log("  observations of one bearing as independent and manufacture an interval")
            log("  that is far too narrow. The conference paper conceded this; J10 does not")
            log("  undo the concession. Report NSK with raw counts and no interval.")
            continue

        Frel = relativise(F, traj)
        folds = [np.where(traj == t)[0] for t in np.unique(traj)]
        rms = F.iloc[:, 0].to_numpy(float)
        kurt = F["kurtosis"].to_numpy(float) if "kurtosis" in F else F["v_kurtosis"].to_numpy(float)

        head("2", f"{ds.upper()} -- {n_traj} trajectories, {B} resamples")
        for rule in RULES:
            y_all, keep_r, excluded = fpt_label(rms, kurt, life, traj, rule)
            sel = np.where(keep_r)[0]
            if len(sel) == 0:
                log(f"    {rule:10s} never fired on any trajectory -- skipped")
                continue
            # E1: restrict labels, features, folds AND the bootstrap pool to the trajectories
            # this rule actually located an onset on.
            y = y_all[sel]
            Fr = Frel.iloc[sel].reset_index(drop=True)
            traj_r = traj[sel]
            folds_r = [np.where(traj_r == t)[0] for t in np.unique(traj_r)]
            n_traj_r = int(len(np.unique(traj_r)))
            rate = y.mean()
            if y.sum() < 3 or rate > 0.5:
                log(f"    {rule:10s} pos rate {rate:5.1%} -- degenerate or too few, skipped")
                continue
            if n_traj_r < 3:
                log(f"    {rule:10s} only {n_traj_r} trajectories left after exclusion -- "
                    f"skipped (a trajectory bootstrap needs more)")
                continue
            if excluded:
                log(f"    {rule:10s} excluded {len(excluded)} non-firing trajectories: "
                    f"{', '.join(excluded)}")
            preds = oof_all(Fr, y, folds_r)

            # bootstrap over trajectories, reusing the OOF predictions
            idx = [boot_indices(traj_r, np.random.default_rng(7000 + b)) for b in range(B)]
            boot = {m: np.array([macro_f1(y[i], preds[m][i]) for i in idx]) for m in preds}

            order = sorted(preds, key=lambda m: macro_f1(y, preds[m]), reverse=True)
            top, second = order[0], order[1]
            diff = boot[top] - boot[second]
            p_better = float((diff > 0).mean())
            log(f"\n    {rule:10s} pos {int(y.sum()):5d} ({rate:4.1%})   "
                f"{n_traj_r}/{n_traj} traj   winner {top}")
            log(f"      {'method':22s} {'median':>8s} {'95% CI':>20s} {'width':>7s}")
            for m in order[:4]:
                s = boot[m]
                lo, hi = np.percentile(s, [2.5, 97.5])
                log(f"      {m:22s} {np.median(s):>8.4f} "
                    f"{f'[{lo:.3f}, {hi:.3f}]':>20s} {hi - lo:>7.3f}")
                rows.append({"dataset": ds, "rule": rule, "method": m,
                             "point": macro_f1(y, preds[m]), "boot_median": float(np.median(s)),
                             "ci_lo": float(lo), "ci_hi": float(hi), "n_traj": n_traj_r,
                             "n_traj_total": n_traj, "n_records": int(len(y)),
                             "pos_rate": float(rate),
                             "excluded_trajectories": ";".join(excluded)})
            lo, hi = np.percentile(diff, [2.5, 97.5])
            verdict = ("separable" if lo > 0 else "NOT separable -- the two are indistinguishable")
            log(f"      winner - runner-up: {np.median(diff):+.4f} [{lo:+.3f}, {hi:+.3f}]  "
                f"P(win)={p_better:.2f}  -> {verdict}")
            win_rows.append({"dataset": ds, "rule": rule, "winner": top, "runner_up": second,
                             "delta_median": float(np.median(diff)), "delta_lo": float(lo),
                             "delta_hi": float(hi), "p_winner_better": p_better,
                             "separable": bool(lo > 0)})

    res, wins = pd.DataFrame(rows), pd.DataFrame(win_rows)

    head("3", "Does the rule-dependence claim of J4 survive the interval?")
    log("  J4 said different published rules crown different winners. That only means")
    log("  something if each winner is separable from its own runner-up.\n")
    if not wins.empty:
        for ds, g in wins.groupby("dataset"):
            sep = g[g.separable]
            log(f"  {ds:6s} {len(sep)}/{len(g)} rules have a winner separable from the runner-up "
                f"(95% CI excludes 0)")
            log(f"         winners on separable rules: {sorted(set(sep.winner)) if len(sep) else '--'}")
            if len(set(sep.winner)) > 1:
                log(f"         -> rule-dependence SURVIVES on {ds}: at least two different")
                log(f"            methods win, each by a margin the bootstrap supports.")
            elif len(sep):
                log(f"         -> only one distinct separable winner; rule-dependence NOT supported.")
            else:
                log(f"         -> no rule has a separable winner; report scores, not rankings.")

    head("4", "Boundary term with an interval")
    log("  Delta_boundary = spread of the best method across published rules, resampled")
    log("  over trajectories.\n")
    for ds, g in res.groupby("dataset"):
        best = g.sort_values("point", ascending=False).groupby("rule").first()
        if len(best) < 2:
            continue
        lo = float((best.ci_hi.min()))
        hi = float((best.ci_lo.max()))
        log(f"  {ds:6s} best-method point spread {best.point.max() - best.point.min():.4f} "
            f"across {len(best)} rules")
        log(f"         most pessimistic reading: worst rule's CI top {lo:.3f} vs best rule's "
            f"CI bottom {hi:.3f}  -> {'still separated' if hi > lo else 'CIs overlap'}")

    head("5", "What J10 does NOT cover")
    log("  The leakage terms vary the TRAINING set, so bootstrapping them needs refitting")
    log("  inside each resample. J1b's 20 repeats already give a spread over random splits;")
    log("  use those percentiles and label them as an interval over SPLITS, not over BEARINGS.")
    j1b = TABLES / "j1b_leakage_real.csv"
    if j1b.exists():
        d = pd.read_csv(j1b)
        d = d[d.rule == "pct05"]
        log("")
        for ds, g in d.groupby("dataset"):
            s0 = g[g.protocol == "S0_leaky"].macro_f1.to_numpy()
            s1 = g[g.protocol == "S1_clean"].macro_f1.to_numpy()
            if len(s0) and len(s1):
                n = min(len(s0), len(s1))
                gap = s0[:n] - s1[:n]
                log(f"    {ds:6s} leakage_window {np.median(gap):+.4f} "
                    f"[{np.percentile(gap, 5):+.3f}, {np.percentile(gap, 95):+.3f}] over "
                    f"{n} split repeats")

    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / f"j10{TAG}_bootstrap_ci.csv", index=False)
    wins.to_csv(TABLES / f"j10{TAG}_winner_test.csv", index=False)
    log(f"\n{BAR}")
    log(f"  WROTE  results/tables/j10_bootstrap_ci.csv ({len(res)} rows)")
    log(f"         results/tables/j10_winner_test.csv  ({len(wins)} rows)")
    log(f"         results/logs/j10_run.log")
    log(f"  finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(BAR)
    if _fh:
        _fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--check" in sys.argv))
