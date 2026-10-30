"""J70 -- the matched false-alarm calibration of J61 validated on independent null series (amendment A9, item 1).

J61 sets each multiplier by bisection on the critical multipliers of R = 4,000 simulated degradation-free series per
trajectory length, and reports the false-alarm rate (FAR) reached on those same series, which equals the target by
construction. This script keeps the 192 multipliers of j61_calibration.csv and measures the FAR that they give on
new series, simulated exactly as in J61 (same null models, lengths, protocols and c = 3) from other seeds.

  self-check   the J61 seed sequences [61, dataset, protocol, noise arm] with R = 4,000 must reproduce every multiplier
               and every achieved FAR of j61_calibration.csv through J61's bisection, to 1e-12
  validation   seed sequences [70, dataset, protocol, noise arm], R = 10,000 series per eligible trajectory length
  per row      FAR_val and its Monte Carlo SE at the calibrated multiplier, FAR_val +/- 1.96 SE_val, and
               z = (FAR_val - target) / sqrt(SE_cal^2 + SE_val^2), with SE_cal the mc_se of J61
  criterion    no |z| > 4 among the 192 calibrations, and at each level the median of FAR_val - target within
               +/-0.2 percentage points; also reported: the range of FAR_val per level and the number of calibrations
               within the +/-0.5-point tolerance of Section 8 of the plan

The in-control run lengths of J61 were computed on separate series (seed sequence [61, 99, dataset]) and are not
recomputed here.

Pre-specified in paper/access_resub/ANALYSIS_PLAN_AMENDMENTS.md, amendment A9 (item 1).

Outputs: results/tables/j70_far_validation.csv   one row per calibration (dataset x protocol x noise arm x rule x level)
         results/logs/j70_run.log
Usage  : python scripts/J70_far_validation.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
SEED = 70
R_CAL, R_VAL = 4000, 10_000
Z_MAX = 4.0                             # largest |z| allowed among the 192 calibrations (A9)
MEDIAN_TOL = 0.002                      # largest |median of FAR_val - target| per level (A9)
TOL_FAR = 0.005                         # tolerance of the achieved FAR in Section 8 of the plan (reported only)
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j70_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J70  MATCHED FALSE-ALARM CALIBRATION VALIDATED ON NEW SERIES  (A9, item 1)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}   R = {R_CAL:,} (self-check), {R_VAL:,} (validation)   seed = {SEED}")

    os.environ["R"] = str(R_CAL)        # J61 reads R at import; set again below before every simulation
    j61 = load("j61", "J61_far_calibration.py")
    j60, j11 = load("j60", "J60_baseline_protocols.py"), load("j11", "J11_arl_and_covariates.py")
    cal = pd.read_csv(T / "j61_calibration.csv")
    if not (cal.R == R_CAL).all():
        log("  FAIL: j61_calibration.csv is not from a full run (R = 4,000); nothing written")
        return 1
    raw = j11.load_traj()
    lengths = {ds: [len(g) for _, g in raw[ds].groupby("trajectory", sort=False)] for ds in j61.INTERVAL_DS}
    ac = pd.read_csv(T / "j22_baseline_autocorrelation.csv")
    noise_arms = {}
    for ds in j61.INTERVAL_DS:          # exactly as J61
        v = ac[ac.dataset == ds].phi_detrended
        noise_arms[ds] = {"main": ("gauss", float(v.median())), "t3": ("t3", float(v.median())),
                          "phi_q25": ("gauss", float(v.quantile(0.25))), "phi_q75": ("gauss", float(v.quantile(0.75)))}

    rows, worst = [], 0.0
    for ds in j61.INTERVAL_DS:
        for arm in j61.PROTOCOLS:
            lens = [n for n in lengths[ds] if j60.eligible(n, arm)]
            for na, (noise, phi) in noise_arms[ds].items():
                key = [j61.INTERVAL_DS.index(ds), j61.PROTOCOLS.index(arm), list(noise_arms[ds]).index(na)]
                j61.R = R_CAL
                ss = np.random.SeedSequence([j61.SEED, *key])
                crits = [j61.critical_values_chunked(child, n, phi, noise, arm)
                         for n, child in zip(lens, ss.spawn(len(lens)))]
                j61.R = R_VAL
                vs = np.random.SeedSequence([SEED, *key])
                vcrits = [j61.critical_values_chunked(child, n, phi, noise, arm)
                          for n, child in zip(lens, vs.spawn(len(lens)))]
                for rule in j61.CAL_RULES:
                    cl, vl = [c[rule] for c in crits], [c[rule] for c in vcrits]
                    for lev in j61.LEVELS:
                        ref = cal[(cal.dataset == ds) & (cal.protocol == arm) & (cal.noise_arm == na) &
                                  (cal.rule == rule) & np.isclose(cal.level, lev)]
                        if len(ref) != 1:
                            log(f"  FAIL: {len(ref)} rows of j61_calibration.csv for {ds} {arm} {na} {rule} {lev}")
                            return 1
                        ref = ref.iloc[0]
                        m, achieved = j61.calibrate(cl, lev)
                        worst = max(worst, abs(m - ref.multiplier), abs(achieved - ref.achieved_far))
                        far_v = j61.far(vl, float(ref.multiplier))
                        se_v = j61.far_se(vl, float(ref.multiplier))
                        z = (far_v - lev) / np.sqrt(float(ref.mc_se) ** 2 + se_v ** 2)
                        rows.append({"dataset": ds, "protocol": arm, "noise_arm": na, "noise": noise, "phi": phi,
                                     "rule": rule, "parameter": ref.parameter, "level": lev,
                                     "multiplier": float(ref.multiplier), "n_lengths": len(lens),
                                     "far_cal": float(ref.achieved_far), "se_cal": float(ref.mc_se), "R_cal": R_CAL,
                                     "far_val": far_v, "se_val": se_v, "lo": far_v - 1.96 * se_v,
                                     "hi": far_v + 1.96 * se_v, "R_val": R_VAL, "z": z,
                                     "within_tol": abs(far_v - lev) <= TOL_FAR})
                log(f"  {j61.DS_NAME[ds]:7s} {arm:13s} {na:7s} (phi {phi:.3f}, {len(lens)} lengths) done")
    log(f"\n  self-check: largest difference from j61_calibration.csv (multiplier, achieved FAR): {worst:.1e}")
    if not worst <= 1e-12:
        log("  FAIL: the J61 calibration is not reproduced; nothing written")
        return 1
    log("  PASS")

    res = pd.DataFrame(rows)
    log(f"\n{BAR}\nFalse-alarm rate on the new series at the calibrated multipliers ({len(res)} calibrations)\n{BAR}")
    ok = True
    for lev, g in res.groupby("level"):
        dev = g.far_val - lev
        med = float(dev.median())
        main = g[g.noise_arm == "main"]
        cover = float(((g.lo <= lev) & (lev <= g.hi)).mean())
        log(f"  level {lev:.0%}: FAR_val {g.far_val.min():.4f}-{g.far_val.max():.4f} (main null "
            f"{main.far_val.min():.4f}-{main.far_val.max():.4f}); median deviation {med * 100:+.3f} points; "
            f"largest |z| {g.z.abs().max():.2f}; within +/-0.5 points {int(g.within_tol.sum())}/{len(g)}; "
            f"95% intervals covering the target {cover:.2f}")
        ok &= abs(med) <= MEDIAN_TOL
    zmax = float(res.z.abs().max())
    ok &= zmax <= Z_MAX
    w = res.loc[res.z.abs().idxmax()]
    log(f"\n  largest |z| overall {zmax:.2f}: {w.dataset} {w.protocol} {w.noise_arm} {w.rule} level {w.level:.0%}, "
        f"FAR_val {w.far_val:.4f} (calibration series {w.far_cal:.4f})")
    log(f"  criterion of A9 (no |z| > {Z_MAX:g}; |median deviation| <= {MEDIAN_TOL * 100:.1f} points at each level): "
        + ("MET -- the calibration is validated" if ok else "NOT MET -- recalibration needed (further amendment)"))

    res.to_csv(T / "j70_far_validation.csv", index=False)
    log(f"\nwrote j70_far_validation.csv ({len(res)} rows)")
    log(f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
