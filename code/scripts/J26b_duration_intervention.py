"""E1 + E2 -- intervene on trajectory length instead of correlating with it.

Run from bearing_multimodal_atc/:
    python scripts/J26b_duration_intervention.py

Why this exists
---------------
Section 4.2 measures the duration scaling BETWEEN bearings: rho(alarm record, length) over 33 trajectories, with a
regression and a permutation test that hold operating condition fixed. A referee can still answer that longer
bearings differ from shorter ones in more ways than length, and that condition does not exhaust those ways.

E1 removes every between-bearing difference by construction. It takes one bearing and keeps every m-th record,
m in {1, 2, 3, 4}. The bearing, the load, the damage and the indicator trajectory in life fraction are unchanged;
only the number of records changes. The two families make opposite, paired predictions:

    detector (k sigma, k >= 3) : alarm INDEX scales as 1/m,  alarm LIFE FRACTION unchanged
    chart (EWMA, CUSUM)        : alarm INDEX unchanged,      alarm LIFE FRACTION scales as m

Reported as a within-bearing slope of log(alarm index) on log(n), directly comparable to the between-bearing
beta_1 of Section 4.2 (+1.18 detectors, -0.13 charts): the detector prediction is +1, the chart prediction is 0.

E2 removes a confound that is built into the rules themselves. The baseline is max(5, int(0.2 n)) everywhere in
J11 and in the J13 convention grid, so the NUMBER of baseline records is proportional to n by construction: a
longer test gives a larger Phase-I sample, a better sigma-hat and a chart that alarms less often by chance. E2
repeats the measurement with a baseline of a FIXED number of records, so that Phase-I quality no longer varies
with test length.

Honest caveat, reported in the output: decimation lowers the lag-1 autocorrelation of the record series
(phi -> approximately phi^m), which lengthens a chart's in-control run length a little. The realised phi is
reported at every m so that the reader can see how much of any chart movement it can explain.

Outputs
    results/tables/j26b_decimation.csv    one row per dataset x trajectory x rule x m x baseline mode
    results/tables/j26b_slopes.csv        within-bearing slope of log(index) on log(n), per rule and arm
    results/logs/j26b_run.log
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
BASE_FRAC, CONFIRM = 0.20, 3
MIN_AFTER = 25          # a decimated series shorter than this cannot carry a baseline and a confirmation run
IMS_FAILED = {"IMS_1st_test_B3", "IMS_1st_test_B4", "IMS_2nd_test_B1", "IMS_3rd_test_B3"}
RULES = ["3sig_rms", "ma3sig", "ma5sig", "ma2sig", "ewma", "cusum"]
FAMILY = {"3sig_rms": "detector", "ma3sig": "detector", "ma5sig": "detector", "ma2sig": "detector",
          "ewma": "control chart", "cusum": "control chart"}
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


def load_traj() -> dict[str, pd.DataFrame]:
    out = {}
    for ds in ("nsk", "xjtu", "ims"):
        f, r = TABLES / f"j7_features_{ds}.csv", BUNDLE / ds / "records.csv"
        if not f.exists() or not r.exists():
            continue
        F, rec = pd.read_csv(f), pd.read_csv(r)
        keep = rec.trajectory_id.isin(IMS_FAILED).to_numpy() if ds == "ims" else np.ones(len(rec), bool)
        out[ds] = pd.DataFrame({"trajectory": rec.loc[keep, "trajectory_id"].to_numpy(),
                                "rms": F.loc[keep, "rms"].to_numpy(),
                                "hikurt": F.loc[keep, "kurtosis"].to_numpy()})
    f = TABLES / "j7c_femto_features.csv"
    if f.exists():
        d = pd.read_csv(f)
        out["femto"] = pd.DataFrame({"trajectory": d.bearing.to_numpy(), "rms": d.v_rms.to_numpy(),
                                     "hikurt": d.v_kurtosis.to_numpy()})
    return out


def first_confirmed(flag: np.ndarray) -> int | None:
    run = 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        if run >= CONFIRM:
            return i - CONFIRM + 1
    return None


def fpt(h: np.ndarray, rule: str, nb: int) -> int | None:
    """Identical to J11.fpt except that the baseline length is an argument, not 0.2 n."""
    n = len(h)
    nb = min(max(5, nb), max(5, n // 2))
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


def ar1_phi(x: np.ndarray) -> float:
    if len(x) < 4:
        return np.nan
    t = np.arange(len(x), dtype=float)
    r = x - np.polyval(np.polyfit(t, x, 1), t)
    return np.nan if r.std() < 1e-12 else float(np.clip(np.corrcoef(r[:-1], r[1:])[0, 1], -0.99, 0.99))


def slope_within(g: pd.DataFrame) -> float:
    """Slope of log(alarm index) on log(n) inside one bearing, across the decimation factors."""
    d = g.dropna(subset=["onset_index"])
    d = d[d.onset_index > 0]
    if d.trajectory_length.nunique() < 3:
        return np.nan
    x, y = np.log(d.trajectory_length.to_numpy(float)), np.log(d.onset_index.to_numpy(float))
    return float(np.polyfit(x, y, 1)[0])


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j26b_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J26b  INTERVENE ON LENGTH  (E1 decimation + E2 fixed baseline)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S} | confirm {CONFIRM} | min records after decimation {MIN_AFTER}")

    data = load_traj()
    MS = (1, 2, 3, 4)
    ARMS = [("prop", None)] + [("fixed", nb) for nb in (10, 20, 30)]
    rows, dropped = [], 0
    for ds, d in data.items():
        for t, g in d.groupby("trajectory", sort=False):
            h0 = g.rms.to_numpy(float)
            for m in MS:
                h = h0[::m]
                n = len(h)
                if n < MIN_AFTER:
                    dropped += 1
                    continue
                nb_prop = max(5, int(n * BASE_FRAC))
                phi = ar1_phi(h[:nb_prop])
                for arm, nb_fixed in ARMS:
                    nb = nb_prop if arm == "prop" else nb_fixed
                    if arm == "fixed" and n < 3 * nb_fixed:
                        continue
                    nb_eff = min(max(5, nb), max(5, n // 2))
                    for rule in RULES:
                        i = fpt(h, rule, nb)
                        rows.append({"dataset": ds, "trajectory": t, "rule": rule, "family": FAMILY[rule],
                                     "m": m, "arm": arm, "baseline_records": nb_eff,
                                     "trajectory_length": n, "baseline_phi": phi,
                                     "onset_index": np.nan if i is None else i,
                                     "onset_fraction": np.nan if i is None else i / n,
                                     "fired": i is not None})
    dec = pd.DataFrame(rows)
    dec.to_csv(TABLES / "j26b_decimation.csv", index=False)
    log(f"\nwrote j26b_decimation.csv ({len(dec)} rows); {dropped} bearing-m cells dropped "
        f"(fewer than {MIN_AFTER} records after decimation)")

    log(f"\n{BAR}\n0. what decimation does to the autocorrelation (the caveat, stated first)\n{BAR}")
    log(f"    {'dataset':7s} " + " ".join(f"{'m=' + str(m):>9s}" for m in MS))
    for ds in data:
        p = dec[(dec.dataset == ds) & (dec.arm == "prop")].groupby("m").baseline_phi.median()
        log(f"    {ds:7s} " + " ".join(f"{p.get(m, np.nan):9.3f}" for m in MS))
    log("    a lower phi lengthens a chart's in-control run length, so a chart alarm index that grows with m is")
    log("    partly explained by this; a detector alarm index that FALLS with m is not.")

    # ---------------- E1: within-bearing slope
    log(f"\n{BAR}\n1. E1  within-bearing slope of log(alarm index) on log(n)\n{BAR}")
    log("    prediction: detector +1 (index tracks n), chart 0 (index is the in-control run length)")
    log(f"    between-bearing values in Section 4.2, for comparison: detectors +1.18, charts -0.13\n")
    sl = []
    for (ds, rule, arm), g in dec.groupby(["dataset", "rule", "arm"], sort=False):
        per = [slope_within(gg) for _, gg in g.groupby("trajectory", sort=False)]
        per = [v for v in per if np.isfinite(v)]
        if len(per) >= 3:
            sl.append({"dataset": ds, "rule": rule, "family": FAMILY[rule], "arm": arm, "n_bearings": len(per),
                       "slope_median": float(np.median(per)), "slope_q25": float(np.percentile(per, 25)),
                       "slope_q75": float(np.percentile(per, 75))})
    sldf = pd.DataFrame(sl)
    sldf.to_csv(TABLES / "j26b_slopes.csv", index=False)
    for ds in ("xjtu", "femto", "ims"):
        g = sldf[(sldf.dataset == ds) & (sldf.arm == "prop")]
        if not len(g):
            continue
        log(f"    {ds}  (baseline proportional, as in Section 4)")
        log(f"      {'rule':9s} {'family':14s} {'bearings':>8s} {'slope':>8s} {'IQR':>18s}")
        for _, r in g.iterrows():
            log(f"      {r.rule:9s} {r.family:14s} {r.n_bearings:8d} {r.slope_median:+8.3f} "
                f"[{r.slope_q25:+.3f}, {r.slope_q75:+.3f}]")
        det = g[g.family == "detector"].slope_median.median()
        ch = g[g.family == "control chart"].slope_median.median()
        log(f"      family medians: detectors {det:+.3f}  charts {ch:+.3f}  gap {det - ch:+.3f}\n")

    # ---------------- E2: does the separation survive a fixed baseline?
    log(f"{BAR}\n2. E2  the same, with a baseline of a FIXED number of records\n{BAR}")
    log("    the baseline in J11 is max(5, int(0.2 n)), so its SIZE is proportional to n by construction.")
    log("    if the separation is an artefact of Phase-I sample size, it must shrink here.\n")
    for ds in ("xjtu", "femto", "ims"):
        g = sldf[sldf.dataset == ds]
        if not len(g):
            continue
        log(f"    {ds}")
        log(f"      {'arm':12s} {'bearings':>8s} {'detectors':>10s} {'charts':>10s} {'gap':>8s}")
        for arm in ("prop", "fixed10", "fixed20", "fixed30"):
            a = arm if arm == "prop" else "fixed"
            sub = g[g.arm == a] if arm == "prop" else None
            if arm != "prop":
                nb = int(arm[5:])
                cells = dec[(dec.dataset == ds) & (dec.arm == "fixed") & (dec.baseline_records == nb)]
                if not len(cells):
                    continue
                sub = []
                for rule, gg in cells.groupby("rule", sort=False):
                    per = [slope_within(x) for _, x in gg.groupby("trajectory", sort=False)]
                    per = [v for v in per if np.isfinite(v)]
                    if len(per) >= 3:
                        sub.append({"rule": rule, "family": FAMILY[rule], "slope_median": float(np.median(per)),
                                    "n_bearings": len(per)})
                sub = pd.DataFrame(sub)
            if sub is None or not len(sub):
                continue
            det = sub[sub.family == "detector"].slope_median.median()
            ch = sub[sub.family == "control chart"].slope_median.median()
            log(f"      {arm:12s} {int(sub.n_bearings.max()):8d} {det:+10.3f} {ch:+10.3f} {det - ch:+8.3f}")
        log("")

    # ---------------- the paired prediction, stated as ratios
    log(f"{BAR}\n3. the paired prediction, as ratios to m = 1 (baseline proportional)\n{BAR}")
    log("    detector: index ratio -> 1/m and fraction ratio -> 1 | chart: index ratio -> 1 and fraction ratio -> m")
    base = dec[(dec.arm == "prop") & (dec.m == 1)].set_index(["dataset", "trajectory", "rule"])
    for ds in ("xjtu", "femto"):
        log(f"    {ds}")
        log(f"      {'family':14s} {'m':>2s} {'index ratio':>12s} {'1/m':>6s} {'fraction ratio':>15s} {'m':>4s} {'n':>4s}")
        for fam in ("detector", "control chart"):
            for m in MS[1:]:
                g = dec[(dec.dataset == ds) & (dec.arm == "prop") & (dec.m == m) & (dec.family == fam) & dec.fired]
                ir, fr = [], []
                for _, r in g.iterrows():
                    key = (r.dataset, r.trajectory, r.rule)
                    if key in base.index:
                        b = base.loc[key]
                        if np.isfinite(b.onset_index) and b.onset_index > 0:
                            ir.append(r.onset_index / b.onset_index)
                            fr.append(r.onset_fraction / b.onset_fraction)
                if ir:
                    log(f"      {fam:14s} {m:2d} {np.median(ir):12.3f} {1 / m:6.3f} {np.median(fr):15.3f} "
                        f"{m:4d} {len(ir):4d}")
        log("")
    log(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
