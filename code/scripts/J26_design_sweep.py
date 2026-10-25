"""E0 + E5 + E6 -- is the duration scaling a property of the rule FAMILY, or of the size of change it responds to?

Run from bearing_multimodal_atc/:
    python scripts/J26_design_sweep.py

Why this exists
---------------
Section 4 compares two control-chart designs tuned for small shifts against RMS thresholds tuned for large ones,
and finds that the chart onsets follow test duration while the detector onsets do not. A referee can answer that
the two families were tuned differently and that the result is a consequence of the tuning, not a finding.

This script tests that objection directly. It sweeps a grid of designs spanning both families, and for each design
measures two things on the same trajectories:

    x = median shift at the alarm, in baseline standard deviations   (the size of change the design responds to,
                                                                      MEASURED, not the nominal parameter)
    y = Spearman rho between the alarm record and the trajectory length

If every design falls on one curve y = f(x), then the family label carries no information once x is known, and the
paper's claim sharpens: what decides whether an onset follows test duration is the size of change the rule responds
to; the two conventional families merely sit on opposite sides of the transition.

E5 adds the two designs Section 7 names as untested alternatives, so that the limitation is measured rather than
conceded: an EWMA whose limits are widened by the autocorrelation inflation factor the paper already computes, and
a chart whose limits adapt to the bearing's own data (the design of Li, Wang and Shen 2024, 10.1088/1361-6501/ad646f).

E6 fits the curve on XJTU-SY only and predicts FEMTO, so the relation is tested out of sample rather than described.

Rules, baseline and confirmation are identical to J11_arl_and_covariates.py; only the design parameters change.
The two canonical designs of Section 4 (EWMA lambda=0.2 L=3; CUSUM k=0.5 h=5) appear in the grid and are flagged,
but nothing here replaces the canonical numbers.

Outputs
    results/tables/j26_design_sweep.csv   one row per design x dataset
    results/tables/j26_oos_femto.csv      out-of-sample prediction of rho on FEMTO from the XJTU-SY fit
    results/logs/j26_run.log
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
BUNDLE = ROOT / "colab_bundle"
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
BASE_FRAC, CONFIRM = 0.20, 3          # identical to J11
IMS_FAILED = {"IMS_1st_test_B3", "IMS_1st_test_B4", "IMS_2nd_test_B1", "IMS_3rd_test_B3"}
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


# ------------------------------------------------------------------ data (same as J11.load_traj)
def load_traj() -> dict[str, pd.DataFrame]:
    out = {}
    for ds in ("nsk", "xjtu", "ims"):
        f, r = TABLES / f"j7_features_{ds}.csv", BUNDLE / ds / "records.csv"
        if not f.exists() or not r.exists():
            continue
        F, rec = pd.read_csv(f), pd.read_csv(r)
        keep = rec.trajectory_id.isin(IMS_FAILED).to_numpy() if ds == "ims" else np.ones(len(rec), bool)
        out[ds] = pd.DataFrame({"trajectory": rec.loc[keep, "trajectory_id"].to_numpy(),
                                "rms": F.loc[keep, "rms"].to_numpy()})
    f = TABLES / "j7c_femto_features.csv"
    if f.exists():
        d = pd.read_csv(f)
        out["femto"] = pd.DataFrame({"trajectory": d.bearing.to_numpy(), "rms": d.v_rms.to_numpy()})
    return out


def first_confirmed(flag: np.ndarray) -> int | None:
    run = 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        if run >= CONFIRM:
            return i - CONFIRM + 1
    return None


# ------------------------------------------------------------------ the designs
def ar1_phi(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of the detrended baseline, as in J20."""
    if len(x) < 4:
        return 0.0
    t = np.arange(len(x), dtype=float)
    r = x - np.polyval(np.polyfit(t, x, 1), t)
    s = r.std()
    return 0.0 if s < 1e-12 else float(np.clip(np.corrcoef(r[:-1], r[1:])[0, 1], -0.99, 0.99))


def onset(h: np.ndarray, kind: str, p: dict) -> int | None:
    """Alarm record of one design on one indicator series. Baseline and confirmation as in J11."""
    n = len(h)
    nb = max(5, int(n * BASE_FRAC))
    b = h[:nb]
    mu, sd = float(b.mean()), max(float(b.std()), 1e-12)

    if kind == "threshold":                                   # RMS threshold at k sigma
        return first_confirmed(h > mu + p["k"] * sd)

    if kind == "ma":                                          # moving average of 5, threshold at k sigma
        ma = pd.Series(h).rolling(5, min_periods=1).mean().to_numpy()
        return first_confirmed(ma > mu + p["k"] * sd)

    if kind in ("ewma", "ewma_ar"):
        lam, L = p["lam"], p["L"]
        if kind == "ewma_ar":                                 # E5a: widen the limit by the AR(1) inflation factor
            phi = ar1_phi(b)
            infl = np.sqrt(max((1 + phi * (1 - lam)) / (1 - phi * (1 - lam)), 1e-9))
            L = L * infl
        z, out = mu, np.empty(n)
        for i, v in enumerate(h):
            z = lam * v + (1 - lam) * z
            out[i] = z
        return first_confirmed(out > mu + L * sd * np.sqrt(lam / (2 - lam)))

    if kind == "cusum":
        k, H = p["k"] * sd, p["h"] * sd
        s, out = 0.0, np.empty(n)
        for i, v in enumerate(h):
            s = max(0.0, s + (v - mu) - k)
            out[i] = s
        return first_confirmed(out > H)

    if kind == "adaptive":
        # E5b: limits recomputed from the bearing's own monitoring data as it accumulates, in the spirit of the
        # dynamic SPC of Li, Wang and Shen (2024). Only records already seen enter the limit, so the rule stays
        # causal; the limit can only widen, which is what "to prevent false alarms" means operationally.
        L = p["L"]
        run, lim_mu, lim_sd = 0, mu, sd
        for i in range(n):
            if i >= nb:
                w = h[:i]
                lim_mu = float(w.mean())
                lim_sd = max(float(w.std()), sd)              # never tighter than the Phase-I estimate
            run = run + 1 if h[i] > lim_mu + L * lim_sd else 0
            if run >= CONFIRM:
                return i - CONFIRM + 1
        return None

    raise ValueError(kind)


def build_grid() -> list[dict]:
    g = []
    for lam in (0.05, 0.1, 0.2, 0.4, 0.8):
        for L in (2.0, 3.0, 4.0, 5.0, 6.0):
            g.append({"design": f"ewma_l{lam}_L{L:g}", "kind": "ewma", "family": "control chart",
                      "p": {"lam": lam, "L": L}, "canonical": lam == 0.2 and L == 3.0})
    for k in (0.25, 0.5, 1.0, 2.0, 3.0):
        for h in (3.0, 5.0, 8.0, 12.0):
            g.append({"design": f"cusum_k{k:g}_h{h:g}", "kind": "cusum", "family": "control chart",
                      "p": {"k": k, "h": h}, "canonical": k == 0.5 and h == 5.0})
    for k in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
        g.append({"design": f"thr_{k:g}sd", "kind": "threshold", "family": "detector",
                  "p": {"k": k}, "canonical": k == 3.0})
    for k in (2.0, 3.0, 5.0):                                  # the ma* rules of Section 4, for continuity
        g.append({"design": f"ma5_{k:g}sd", "kind": "ma", "family": "detector", "p": {"k": k}, "canonical": True})
    for lam in (0.05, 0.1, 0.2, 0.4, 0.8):                     # E5a
        g.append({"design": f"ewmaAR_l{lam}_L3", "kind": "ewma_ar", "family": "chart, AR-corrected",
                  "p": {"lam": lam, "L": 3.0}, "canonical": False})
    for L in (2.0, 3.0, 4.0, 5.0, 6.0):                        # E5b
        g.append({"design": f"adapt_L{L:g}", "kind": "adaptive", "family": "chart, adaptive limits",
                  "p": {"L": L}, "canonical": False})
    return g


# ------------------------------------------------------------------ main
def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j26_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J26  DESIGN SWEEP  (E0 + E5 + E6)".center(78))
    log("is the duration scaling a property of the family, or of the response size?".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S} | baseline {BASE_FRAC:.0%} | confirm {CONFIRM}")

    data = load_traj()
    ref = pd.read_csv(TABLES / "fig1_fpt_matrix.csv")
    log(f"\ndatasets: " + ", ".join(f"{k} ({d.trajectory.nunique()})" for k, d in data.items()))
    n_ref = ref.groupby("dataset").trajectory.nunique().to_dict()
    log(f"reference (fig1_fpt_matrix): {n_ref}")
    for k, d in data.items():
        if d.trajectory.nunique() != n_ref.get(k):
            log(f"  !! {k}: {d.trajectory.nunique()} loaded vs {n_ref.get(k)} in the matrix")

    grid = build_grid()
    log(f"\n{len(grid)} designs: " + ", ".join(f"{f} {sum(g['family'] == f for g in grid)}"
                                               for f in dict.fromkeys(g["family"] for g in grid)))

    rows, per_traj = [], []
    for ds, d in data.items():
        series = [(t, g.rms.to_numpy(float)) for t, g in d.groupby("trajectory", sort=False)]
        if len(series) < 4:
            log(f"\n{ds}: {len(series)} trajectories, too few for a correlation -- shift only")
        for spec in grid:
            idx, lens, shifts = [], [], []
            for t, h in series:
                i = onset(h, spec["kind"], spec["p"])
                lens.append(len(h))
                # One row per design x trajectory, kept whether or not the design fired. The aggregates below
                # collapse these into one number per design, which is enough to describe the curve but not to put
                # an interval on it: all 67 designs are measured on the SAME bearings, so a regression over
                # designs has far fewer independent units than it has rows. J36_design_cv.py resamples bearings
                # from this table to give the design-level fit an interval whose unit is the bearing.
                rec = {"dataset": ds, "design": spec["design"], "kind": spec["kind"], "family": spec["family"],
                       "canonical": spec["canonical"], "trajectory": t, "n_records": len(h)}
                if i is None:
                    idx.append(np.nan)
                    per_traj.append({**rec, "fired": False, "alarm_index": np.nan, "shift_sd": np.nan})
                    continue
                idx.append(i)
                nb = max(5, int(len(h) * BASE_FRAC))
                mu, sd = float(h[:nb].mean()), max(float(h[:nb].std()), 1e-12)
                w = h[i:i + CONFIRM]
                shift = (float(w.mean()) - mu) / sd
                shifts.append(shift)
                per_traj.append({**rec, "fired": True, "alarm_index": int(i), "shift_sd": shift})
            a, b = np.asarray(idx, float), np.asarray(lens, float)
            ok = np.isfinite(a)
            rho = float(spearmanr(b[ok], a[ok])[0]) if ok.sum() >= 4 else np.nan
            rho_f = float(spearmanr(b[ok], a[ok] / b[ok])[0]) if ok.sum() >= 4 else np.nan
            nb_all = np.maximum(5, (b * BASE_FRAC).astype(int))
            rows.append({
                "dataset": ds, "design": spec["design"], "kind": spec["kind"], "family": spec["family"],
                "canonical": spec["canonical"], "n_traj": len(series), "n_fired": int(ok.sum()),
                "median_shift_sd": float(np.median(shifts)) if shifts else np.nan,
                "rho_index_length": rho, "rho_fraction_length": rho_f,
                "median_onset_fraction": float(np.nanmedian(a / b)) if ok.any() else np.nan,
                "inside_baseline": int(np.nansum(a[ok] < nb_all[ok])),
            })
    sw = pd.DataFrame(rows)
    sw.to_csv(TABLES / "j26_design_sweep.csv", index=False)
    log(f"\nwrote j26_design_sweep.csv  ({len(sw)} rows)")
    pt = pd.DataFrame(per_traj)
    pt.to_csv(TABLES / "j26_design_sweep_per_traj.csv", index=False)
    log(f"wrote j26_design_sweep_per_traj.csv  ({len(pt)} rows: design x trajectory, for J36)")

    # ---------------- the question
    log(f"\n{BAR}\n1. does rho depend on the family once the measured response size is known?\n{BAR}")
    for ds in ("xjtu", "femto"):
        g = sw[(sw.dataset == ds) & sw.median_shift_sd.notna() & sw.rho_index_length.notna()]
        if len(g) < 8:
            continue
        log(f"\n{ds}  ({len(g)} designs with a measurable response size)")
        log(f"    {'shift bin (sd)':>16s} {'n':>3s} {'rho median':>11s} {'families in the bin'}")
        bins = [(0, 1.5), (1.5, 2.5), (2.5, 4), (4, 7), (7, 1e9)]
        for lo, hi in bins:
            b = g[(g.median_shift_sd >= lo) & (g.median_shift_sd < hi)]
            if not len(b):
                continue
            fams = ", ".join(sorted({f.split(",")[0] for f in b.family}))
            log(f"    {f'[{lo:g}, {hi:g})' if hi < 1e9 else f'[{lo:g}, inf)':>16s} {len(b):3d} "
                f"{b.rho_index_length.median():11.3f}  {fams}")
        # within a bin that holds both families, is there a family effect left?
        for lo, hi in bins:
            b = g[(g.median_shift_sd >= lo) & (g.median_shift_sd < hi)]
            fam2 = {f.split(",")[0] for f in b.family}
            if len(b) >= 4 and len(fam2) > 1:
                ch = b[b.family.str.startswith("chart") | (b.family == "control chart")].rho_index_length
                de = b[b.family == "detector"].rho_index_length
                if len(ch) and len(de):
                    log(f"    overlap bin [{lo:g}, {hi:g}): charts {ch.median():+.3f} (n={len(ch)})  "
                        f"detectors {de.median():+.3f} (n={len(de)})  gap {ch.median() - de.median():+.3f}")

    log(f"\n{BAR}\n2. the two designs Section 7 names as untested (E5)\n{BAR}")
    for ds in ("xjtu", "femto"):
        g = sw[sw.dataset == ds]
        for fam in ("control chart", "chart, AR-corrected", "chart, adaptive limits"):
            b = g[g.family == fam]
            if len(b):
                log(f"    {ds:6s} {fam:24s} shift {b.median_shift_sd.median():5.2f} sd   "
                    f"rho {b.rho_index_length.median():+.3f}   fired "
                    f"{b.n_fired.sum()}/{b.n_traj.sum()}   inside baseline {int(b.inside_baseline.sum())}")

    # ---------------- E6: fit on XJTU, predict FEMTO
    log(f"\n{BAR}\n3. E6  fit the curve on XJTU-SY, predict FEMTO\n{BAR}")
    X = sw[(sw.dataset == "xjtu") & sw.median_shift_sd.notna() & sw.rho_index_length.notna()]
    F = sw[(sw.dataset == "femto") & sw.median_shift_sd.notna() & sw.rho_index_length.notna()]
    oos = pd.DataFrame()
    if len(X) >= 8 and len(F) >= 8:
        # a monotone saturating curve in log(shift): rho = a + b * logistic-free least squares on log x
        lx = np.log(np.clip(X.median_shift_sd.to_numpy(float), 1e-3, None))
        cf = np.polyfit(lx, X.rho_index_length.to_numpy(float), 2)
        pred = np.polyval(cf, np.log(np.clip(F.median_shift_sd.to_numpy(float), 1e-3, None)))
        obs = F.rho_index_length.to_numpy(float)
        oos = F[["design", "family", "median_shift_sd"]].copy()
        oos["rho_observed"], oos["rho_predicted"] = obs, pred
        oos["abs_error"] = np.abs(obs - pred)
        oos.to_csv(TABLES / "j26_oos_femto.csv", index=False)
        sst = float(((obs - obs.mean()) ** 2).sum())
        sse = float(((obs - pred) ** 2).sum())
        log(f"    fit on {len(X)} XJTU-SY designs, quadratic in log(shift)")
        log(f"    FEMTO: {len(F)} designs | median |error| {np.median(oos.abs_error):.3f} | "
            f"out-of-sample R2 {1 - sse / sst if sst > 0 else float('nan'):+.3f}")
        log(f"    baseline (predict the XJTU-SY mean rho for every FEMTO design): median |error| "
            f"{np.median(np.abs(obs - X.rho_index_length.mean())):.3f}")
        log("\n    worst five designs out of sample:")
        for _, r in oos.nlargest(5, "abs_error").iterrows():
            log(f"      {r.design:22s} shift {r.median_shift_sd:5.2f}  obs {r.rho_observed:+.3f}  "
                f"pred {r.rho_predicted:+.3f}  err {r.abs_error:.3f}")
    else:
        log("    not enough designs with a measurable response size on both datasets")

    log(f"\n{BAR}\nread j26_design_sweep.csv; the figure is drawn by J25 once the reading is settled.\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
