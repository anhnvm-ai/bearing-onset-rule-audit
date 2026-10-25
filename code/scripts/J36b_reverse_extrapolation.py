"""J36b -- why the reverse cross-family fit (45 charts -> 12 detectors) scores a low R2 (Phase 3B).

J36 fits the response-size curve rho ~ [1, log(shift), log(shift)^2] on the 45 EWMA and CUSUM charts
and predicts the 12 threshold detectors. On XJTU-SY that gives R2 = 0.203 about the detectors' own mean,
far below the forward direction (0.653). A tempting explanation is that the detectors all sit in the
saturated high-rho region, leaving little variance to explain. The data refute it: the 12 detector
designs span rho from -0.04 to +0.92 on XJTU-SY, because the sweep includes 1-sigma and 1.5-sigma
thresholds that behave like charts.

The actual cause is extrapolation. The charts' measured shifts cover a bounded range; a quadratic fitted
inside it and evaluated beyond it bends upward and predicts rho above +1, which no correlation can
reach. This script splits the reverse-fit errors by whether each detector lies inside the charts' shift
range, which separates "the curve does not transfer" from "the curve was extrapolated".

Inputs : results/tables/j26_design_sweep.csv, results/tables/j36_cross_family_pred.csv
Output : results/tables/j36b_reverse_range.csv   per dataset x (inside | outside the chart shift range)
         results/logs/j36b_run.log
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T, LOGS = ROOT / "results" / "tables", ROOT / "results" / "logs"


def main():
    s = pd.read_csv(T / "j26_design_sweep.csv")
    p = pd.read_csv(T / "j36_cross_family_pred.csv")
    rows, log = [], []
    for ds in ("xjtu", "femto"):
        g = s[(s.dataset == ds) & s.rho_index_length.notna() & s.median_shift_sd.notna()]
        ch = g[g.family == "control chart"].median_shift_sd
        lo, hi = float(ch.min()), float(ch.max())
        r = p[(p.dataset == ds) & (p.split == "conventional charts -> detectors")].copy()
        r["inside"] = r.median_shift_sd.between(lo, hi)
        det = g[g.family == "detector"].rho_index_length
        log.append(f"{ds}: chart shift range [{lo:.2f}, {hi:.2f}] sd; detector rho spans "
                   f"{det.min():+.3f} to {det.max():+.3f} (sd {det.std():.3f})")
        for inside, sub in r.groupby("inside"):
            rows.append({"dataset": ds, "chart_shift_min": lo, "chart_shift_max": hi,
                         "detectors": "inside chart range" if inside else "outside chart range",
                         "n": len(sub), "median_abs_error": float(sub.abs_error.median()),
                         "max_rho_predicted": float(sub.rho_predicted.max()),
                         "n_predicted_above_1": int((sub.rho_predicted > 1).sum())})
            log.append(f"   {'inside ' if inside else 'outside'}: n={len(sub):2d}  median |err| "
                       f"{sub.abs_error.median():.3f}  predictions above +1: {int((sub.rho_predicted > 1).sum())}")
    out = pd.DataFrame(rows)
    out.to_csv(T / "j36b_reverse_range.csv", index=False)
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "j36b_run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print("\n".join(log))


if __name__ == "__main__":
    main()
