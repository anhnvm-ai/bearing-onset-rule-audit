"""E2 done properly -- is the between-bearing duration slope an artefact of the baseline being 20% of n?

Run from bearing_multimodal_atc/:
    python scripts/J26c_baseline_confound.py

Why this exists
---------------
Section 4.2 reports a condition-adjusted slope of log(alarm index + 1) on log(n): +1.18 for the RMS detectors and
-0.13 for the charts on XJTU-SY. Every rule in J11 takes its baseline as max(5, int(0.2 n)), so the NUMBER of
baseline records is proportional to n by construction. Two consequences run in opposite directions:

  * a longer test gives a larger Phase-I sample, a better sigma-hat and therefore FEWER chance alarms;
  * a chart that alarms soon after the baseline window ends inherits the n-scaling of that window, because the
    window ends at record 0.2 n.

Either one could produce a slope that has nothing to do with the size of change the rule responds to. This script
repeats the Section 4.2 regression with the baseline fixed at a CONSTANT number of records, so that Phase-I size no
longer varies with test length, and reports every arm on the SAME set of bearings so the arms are comparable.

A first version of this comparison inside J26b compared a 13-bearing arm against a 7-bearing arm and was therefore
meaningless; that is why this script exists separately and prints the bearing set it used.

Outputs
    results/tables/j26c_fixed_baseline_slopes.csv
    results/logs/j26c_run.log
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "colab_bundle"
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
BASE_FRAC, CONFIRM, B = 0.20, 3, 2000
IMS_FAILED = {"IMS_1st_test_B3", "IMS_1st_test_B4", "IMS_2nd_test_B1", "IMS_3rd_test_B3"}
RMS3 = ["3sig_rms", "ma3sig", "ma5sig"]
CHARTS = ["ewma", "cusum"]
RULES = RMS3 + ["ma2sig"] + CHARTS
FAMILY = {**{r: "detector" for r in RMS3 + ["ma2sig"]}, **{r: "control chart" for r in CHARTS}}
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


def condition_of(ds: str, traj: str) -> str:
    return traj.split("_")[0].replace("Bearing", "") if ds in ("xjtu", "femto") else "all"


def load_traj() -> dict[str, pd.DataFrame]:
    out = {}
    for ds in ("xjtu", "ims"):
        f, r = TABLES / f"j7_features_{ds}.csv", BUNDLE / ds / "records.csv"
        F, rec = pd.read_csv(f), pd.read_csv(r)
        keep = rec.trajectory_id.isin(IMS_FAILED).to_numpy() if ds == "ims" else np.ones(len(rec), bool)
        out[ds] = pd.DataFrame({"trajectory": rec.loc[keep, "trajectory_id"].to_numpy(),
                                "rms": F.loc[keep, "rms"].to_numpy()})
    d = pd.read_csv(TABLES / "j7c_femto_features.csv")
    out["femto"] = pd.DataFrame({"trajectory": d.bearing.to_numpy(), "rms": d.v_rms.to_numpy()})
    return out


def first_confirmed(flag: np.ndarray) -> int | None:
    run = 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        if run >= CONFIRM:
            return i - CONFIRM + 1
    return None


def fpt(h: np.ndarray, rule: str, nb: int) -> int | None:
    n = len(h)
    mu, sd = float(np.mean(h[:nb])), max(float(np.std(h[:nb])), 1e-12)
    if rule == "3sig_rms":
        return first_confirmed(h > mu + 3 * sd)
    if rule == "ewma":
        lam, L = 0.2, 3.0
        z, out = mu, np.empty(n)
        for i, v in enumerate(h):
            z = lam * v + (1 - lam) * z
            out[i] = z
        return first_confirmed(out > mu + L * sd * np.sqrt(lam / (2 - lam)))
    if rule == "cusum":
        k, H = 0.5 * sd, 5.0 * sd
        s, out = 0.0, np.empty(n)
        for i, v in enumerate(h):
            s = max(0.0, s + (v - mu) - k)
            out[i] = s
        return first_confirmed(out > H)
    if rule.startswith("ma"):
        ma = pd.Series(h).rolling(5, min_periods=1).mean().to_numpy()
        return first_confirmed(ma > mu + float(rule[2]) * sd)
    raise ValueError(rule)


def ols_slope(y: np.ndarray, x: np.ndarray, c: np.ndarray) -> float:
    """Slope of y on x with a fixed effect per level of c -- identical to J23.ols_slope."""
    lev = sorted(set(c))
    X = np.column_stack([x] + [(c == k).astype(float) for k in lev])
    if len(y) <= X.shape[1]:
        return np.nan
    try:
        return float(np.linalg.lstsq(X, y, rcond=None)[0][0])
    except np.linalg.LinAlgError:
        return np.nan


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j26c_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J26c  IS THE DURATION SLOPE AN ARTEFACT OF A BASELINE THAT SCALES WITH n?".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S} | confirm {CONFIRM} | bootstrap B={B}")
    log("\nthe baseline in J11 is max(5, int(0.2 n)); every 'fixed' arm below uses a constant record count instead.")

    data = load_traj()
    rng = np.random.default_rng(2609)
    ARMS = [("prop", None), ("fixed", 10), ("fixed", 15), ("fixed", 20), ("fixed", 30)]
    rows = []
    for ds in ("xjtu", "femto", "ims"):
        series = [(t, g.rms.to_numpy(float)) for t, g in data[ds].groupby("trajectory", sort=False)]
        lens = np.array([len(h) for _, h in series])
        log(f"\n{BAR}\n{ds}: {len(series)} trajectories, lengths {lens.min()}-{lens.max()}\n{BAR}")
        for arm, nbf in ARMS:
            # a fixed baseline must never be capped, or it is not fixed: require n >= 4 nb
            keep = [(t, h) for t, h in series if arm == "prop" or len(h) >= 4 * nbf]
            if len(keep) < 6:
                log(f"  {arm}{nbf or ''}: only {len(keep)} trajectories with n >= 4 nb -- skipped")
                continue
            names = {t for t, _ in keep}
            recs = []
            for t, h in keep:
                nb = max(5, int(len(h) * BASE_FRAC)) if arm == "prop" else nbf
                for rule in RULES:
                    i = fpt(h, rule, nb)
                    recs.append({"trajectory": t, "rule": rule, "n": len(h), "nb": nb,
                                 "i": np.nan if i is None else i})
            R = pd.DataFrame(recs)
            R["cond"] = [condition_of(ds, t) for t in R.trajectory]
            for rule in RULES:
                g = R[(R.rule == rule) & R.i.notna()]
                if len(g) < 6:
                    continue
                y, x, c = np.log(g.i.to_numpy() + 1), np.log(g.n.to_numpy(float)), g.cond.to_numpy()
                b1 = ols_slope(y, x, c)
                boots = []
                for _ in range(B):
                    idx = np.concatenate([rng.choice(np.where(c == k)[0], (c == k).sum(), replace=True)
                                          for k in sorted(set(c))])
                    boots.append(ols_slope(y[idx], x[idx], c[idx]))
                boots = np.array([b for b in boots if np.isfinite(b)])
                lo, hi = (np.percentile(boots, [2.5, 97.5]) if len(boots) > B // 2 else (np.nan, np.nan))
                rows.append({"dataset": ds, "arm": f"{arm}{nbf or ''}", "baseline_records": nbf,
                             "rule": rule, "family": FAMILY[rule], "n_traj_arm": len(keep),
                             "n_fired": len(g), "slope": b1, "ci_lo": lo, "ci_hi": hi,
                             "bearing_set": "|".join(sorted(names))})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "j26c_fixed_baseline_slopes.csv", index=False)
    matched: list[dict] = []
    log(f"\nwrote j26c_fixed_baseline_slopes.csv ({len(out)} rows)")

    # ---------------- comparison on a MATCHED bearing set
    log(f"\n{BAR}\nslopes, each arm on its own bearing set (n_traj shown)\n{BAR}")
    for ds in ("xjtu", "femto", "ims"):
        g = out[out.dataset == ds]
        if not len(g):
            continue
        log(f"\n{ds}")
        log(f"  {'arm':9s} {'bearings':>8s} " + " ".join(f"{r:>10s}" for r in RULES) + f" {'det-chart':>10s}")
        for arm in g.arm.unique():
            a = g[g.arm == arm].set_index("rule")
            vals = [f"{a.loc[r].slope:+10.3f}" if r in a.index else f"{'--':>10s}" for r in RULES]
            det = a.loc[[r for r in RMS3 if r in a.index], "slope"].median() if any(r in a.index for r in RMS3) else np.nan
            ch = a.loc[[r for r in CHARTS if r in a.index], "slope"].median() if any(r in a.index for r in CHARTS) else np.nan
            log(f"  {arm:9s} {int(a.n_traj_arm.iloc[0]):8d} " + " ".join(vals) + f" {det - ch:+10.3f}")

    log(f"\n{BAR}\nthe same comparison restricted to the bearings every arm could use\n{BAR}")
    for ds in ("xjtu", "femto", "ims"):
        g = out[out.dataset == ds]
        if not len(g):
            continue
        sets = [set(s.split("|")) for s in g.groupby("arm").bearing_set.first()]
        common = set.intersection(*sets) if sets else set()
        log(f"\n{ds}: {len(common)} bearings common to all arms")
        if len(common) < 6:
            log("  too few for a matched comparison -- read the unmatched table above and the note in the log")
            continue
        series = [(t, h) for t, h in ((t, g2.rms.to_numpy(float))
                                      for t, g2 in data[ds].groupby("trajectory", sort=False)) if t in common]
        log(f"  {'arm':9s} " + " ".join(f"{r:>10s}" for r in RULES) + f" {'det-chart':>10s}")
        for arm, nbf in ARMS:
            recs = []
            for t, h in series:
                nb = max(5, int(len(h) * BASE_FRAC)) if arm == "prop" else nbf
                for rule in RULES:
                    i = fpt(h, rule, nb)
                    recs.append({"trajectory": t, "rule": rule, "n": len(h), "i": np.nan if i is None else i})
            R = pd.DataFrame(recs)
            R["cond"] = [condition_of(ds, t) for t in R.trajectory]
            vals, sl = [], {}
            for rule in RULES:
                gg = R[(R.rule == rule) & R.i.notna()]
                if len(gg) < 6:
                    vals.append(f"{'--':>10s}")
                    continue
                b1 = ols_slope(np.log(gg.i.to_numpy() + 1), np.log(gg.n.to_numpy(float)), gg.cond.to_numpy())
                sl[rule] = b1
                vals.append(f"{b1:+10.3f}")
            det = np.median([sl[r] for r in RMS3 if r in sl]) if any(r in sl for r in RMS3) else np.nan
            ch = np.median([sl[r] for r in CHARTS if r in sl]) if any(r in sl for r in CHARTS) else np.nan
            log(f"  {f'{arm}{nbf or chr(32)}':9s} " + " ".join(vals) + f" {det - ch:+10.3f}")
            # sigma-hat from this arm's baseline, relative to the 0.2 n baseline, on the same bearings
            ratio = []
            for t_, h in series:
                nbp = max(5, int(len(h) * BASE_FRAC))
                nb = nbp if arm == "prop" else nbf
                ratio.append(max(float(np.std(h[:nb])), 1e-12) / max(float(np.std(h[:nbp])), 1e-12))
            matched.append({"dataset": ds, "arm": f"{arm}{nbf or ''}", "baseline_records": nbf,
                            "n_bearings": len(series), "detectors": det, "charts": ch, "gap": det - ch,
                            "sigma_ratio_to_prop": float(np.median(ratio)),
                            **{f"slope_{r}": sl.get(r, np.nan) for r in RULES}})
    pd.DataFrame(matched).to_csv(TABLES / "j26c_matched_arms.csv", index=False)
    log(f"\nwrote j26c_matched_arms.csv ({len(matched)} rows: one per dataset x arm, same bearings in every arm)")
    log(f"\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
