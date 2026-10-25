"""J16 -- persist the four groups of numbers that J11 and J13 only ever printed to their logs.

Section 4 of the journal manuscript quotes these, and until now verify_sec4.py had to parse
them out of results/logs. This script recomputes each one with the same code path and seeds,
and writes it to a table. On 2026-09-14 every value matched results/logs/archive_2026-09-14/.
Table prefix j19 because j16 is reserved for E7 (matched prevalence) and J15 wrote j17 and j18.

  (1) paired RMS vs kurtosis half-rise marker gap        J11 step 7d(b), rng seed 9003, B=20000
  (2) end-to-peak ratio of each health indicator          J11 step 7d(a)
  (3) percentile-rule discretisation drift and rho(f, n)  J11 step 5
  (4) firing coverage over the convention grid            J13 step 2

Outputs
    results/tables/j19_marker_pair_gap.csv
    results/tables/j19_indicator_end_peak.csv
    results/tables/j19_percentile_discretisation.csv
    results/tables/j19_grid_coverage.csv
    results/logs/j16_run.log

Usage
    python scripts/J16_persist_log_numbers.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
B = 20000  # the value J11 was run with (its log header says B=20000)


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


J11 = _load("j11", "J11_arl_and_covariates.py")
J13 = _load("j13", "J13_convention_sensitivity.py")
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    _fh.write(m + "\n")


def main() -> int:
    global _fh
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    _fh = open(LOGS / "j16_run.log", "w", encoding="utf-8")
    data = J11.load_traj()
    order = [ds for ds in ("xjtu", "ims", "femto") if ds in data]

    # (1) paired marker gap -- same loop order and seed as J11, so the bootstrap draws match
    log("(1) paired |RMS marker - kurtosis marker| at rise share 0.50")
    mk = {}
    for ds in order:
        for t, g in data[ds].groupby("trajectory", sort=True):
            mk[(ds, t, "rms")] = J11.markers(g["rms"].to_numpy(), 0.50)
            mk[(ds, t, "kurt")] = J11.markers(g["hikurt"].to_numpy(), 0.50)
    rng = np.random.default_rng(9003)
    rows = []
    for ds in order:
        tr = sorted({t for (d_, t, _) in mk if d_ == ds})
        a = np.array([mk[(ds, t, "rms")] for t in tr])
        b = np.array([mk[(ds, t, "kurt")] for t in tr])
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 4:
            continue
        a, b = a[ok], b[ok]
        diff = np.abs(a - b)
        bs = np.median(diff[rng.integers(0, len(diff), size=(B, len(diff)))], axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append({"dataset": ds, "n_bearings": len(a), "median_rms_marker": np.median(a),
                     "median_kurt_marker": np.median(b), "median_abs_diff": np.median(diff),
                     "ci_lo": lo, "ci_hi": hi, "share_diff_gt_0.25": (diff > 0.25).mean(), "B": B})
        log(f"    {ds:6s} n={len(a):2d}  median |diff| {np.median(diff):.3f} [{lo:.2f}, {hi:.2f}]")
    pd.DataFrame(rows).to_csv(TABLES / "j19_marker_pair_gap.csv", index=False)

    # (2) end-to-peak ratio
    log("\n(2) median (end - base) / (peak - base) per indicator")
    rows = []
    for ds in order:
        for name, col in (("rms", "rms"), ("kurt", "hikurt")):
            rr = []
            for _, g in data[ds].groupby("trajectory", sort=True):
                h = g[col].to_numpy()
                base = float(np.median(h[:max(3, len(h) // 10)]))
                end, pk = float(np.median(h[-3:])), float(np.max(h))
                if pk - base > 0:
                    rr.append((end - base) / (pk - base))
            if rr:
                rows.append({"dataset": ds, "indicator": name, "n_bearings": len(rr),
                             "median_end_over_peak": float(np.median(rr))})
                log(f"    {ds:6s} {name:5s} n={len(rr):2d}  {np.median(rr):.3f}")
    pd.DataFrame(rows).to_csv(TABLES / "j19_indicator_end_peak.csv", index=False)

    # (3) percentile discretisation
    log("\n(3) percentile rules: realised fraction int(n(1-p))/n")
    rows = []
    for ds in order:
        ns = np.array([len(g) for _, g in data[ds].groupby("trajectory", sort=True)])
        for rule in J11.PERCENTILE:
            f = np.array([int(n * (1 - int(rule[3:]) / 100.0)) / n for n in ns])
            if len(f) < 4:
                continue
            rho, p = spearmanr(ns, f)
            rows.append({"dataset": ds, "rule": rule, "n_bearings": len(f), "frac_min": f.min(),
                         "frac_max": f.max(), "drift": f.max() - f.min(), "rho_frac_length": rho, "p": p})
            log(f"    {ds:6s} {rule}  drift {f.max() - f.min():.4f}  rho {rho:+.3f}")
    pd.DataFrame(rows).to_csv(TABLES / "j19_percentile_discretisation.csv", index=False)

    # (4) grid firing coverage
    log("\n(4) firing coverage over the convention grid (all datasets, all ten rules)")
    data13 = J13.load_traj()
    possible = sum(d.trajectory.nunique() for d in data13.values()) * len(J13.FAMILY)
    rows = []
    for bf in J13.BASE_GRID:
        for cf in J13.CONFIRM_GRID:
            n_fire = len(J13.fpt_table(data13, bf, cf))
            rows.append({"base_frac": bf, "confirm": cf, "rule_trajectory_pairs": n_fire,
                         "possible_pairs": possible, "coverage": n_fire / possible})
            log(f"    base {bf:.2f} confirm {cf}  {n_fire:3d}/{possible}  {n_fire / possible:.0%}")
    pd.DataFrame(rows).to_csv(TABLES / "j19_grid_coverage.csv", index=False)
    log("\nwrote j19_marker_pair_gap, j19_indicator_end_peak, j19_percentile_discretisation, j19_grid_coverage")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
