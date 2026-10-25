"""E9 -- why a large-threshold onset scales with test duration and a small-threshold one does not.

Run from bearing_multimodal_atc/:
    python scripts/J29_scaling_law.py

The proposition
---------------
Suppose the health indicator of a trajectory of n records can be written

        h(i) = mu + sigma * g(i / n) + eps(i),        g increasing, g(0) = 0,

so that the degradation is parameterised in LIFE FRACTION: every bearing passes through the same shape, and only
the number of records differs. Then the first record at which h crosses mu + k sigma is

        i_k  ~  n * g^{-1}(k),

which is proportional to n for every k inside the range of g. A rule that alarms instead when a cumulative
statistic crosses a constant that does not involve g -- that is, a rule that alarms on noise rather than on the
signal -- alarms at its in-control run length, which does not involve n at all.

The two families of Section 4 sit on the two sides of that statement, and Section 4.3 already shows the empirical
signature: the record at which the indicator first crosses one standard deviation does not scale with length while
the crossing of three does.

What this script tests
----------------------
The proposition rests on an assumption that must be checked rather than assumed: that g is the SAME function for
every trajectory, so that g^{-1}(k) is a trajectory-independent constant. If that fails, the proposition is a
reference model and not a description of these data, and the paper must say so.

  1  Estimate g^{-1}(k) as the median over trajectories of (crossing record / n) at each k.
  2  Report the spread of that fraction across trajectories. A tight spread supports the common-shape assumption.
  3  Predict the crossing record as n * ghat^{-1}(k) and compare with the observed record, leave-one-out, so the
     trajectory being predicted does not contribute to the constant used to predict it.
  4  Report R^2 of that prediction against the null "the crossing record is a constant c_k", which is what a rule
     alarming on noise would give.

Outputs
    results/tables/j29_scaling_law.csv    per dataset x k: ghat, spread, LOO prediction error, R2 against both nulls
    results/logs/j29_run.log
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "colab_bundle"
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
BASE_FRAC, CONFIRM = 0.20, 3
KS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
IMS_FAILED = {"IMS_1st_test_B3", "IMS_1st_test_B4", "IMS_2nd_test_B1", "IMS_3rd_test_B3"}
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


def load_traj() -> dict[str, pd.DataFrame]:
    out = {}
    for ds in ("xjtu", "ims"):
        F = pd.read_csv(T / f"j7_features_{ds}.csv")
        rec = pd.read_csv(BUNDLE / ds / "records.csv")
        keep = rec.trajectory_id.isin(IMS_FAILED).to_numpy() if ds == "ims" else np.ones(len(rec), bool)
        out[ds] = pd.DataFrame({"trajectory": rec.loc[keep, "trajectory_id"].to_numpy(),
                                "rms": F.loc[keep, "rms"].to_numpy()})
    d = pd.read_csv(T / "j7c_femto_features.csv")
    out["femto"] = pd.DataFrame({"trajectory": d.bearing.to_numpy(), "rms": d.v_rms.to_numpy()})
    return out


def first_confirmed(flag: np.ndarray) -> int | None:
    run = 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        if run >= CONFIRM:
            return i - CONFIRM + 1
    return None


def crossings(h: np.ndarray) -> dict[float, int | None]:
    """First confirmed record at which the standardised indicator crosses k, for every k in KS."""
    n = len(h)
    nb = max(5, int(n * BASE_FRAC))
    mu, sd = float(h[:nb].mean()), max(float(h[:nb].std()), 1e-12)
    z = (h - mu) / sd
    return {k: first_confirmed(z > k) for k in KS}


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j29_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J29  IS THE COMMON-SHAPE ASSUMPTION BEHIND THE SCALING LAW TRUE?  (E9)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S} | baseline {BASE_FRAC:.0%} | confirm {CONFIRM}")
    log("\nmodel: h(i) = mu + sigma g(i/n) + eps(i), g the same for every trajectory")
    log("       => the first crossing of k sigma is at i_k ~ n ghat^{-1}(k), proportional to n")
    log("       the competing null is i_k = c_k, a constant record count, which is what a noise alarm gives\n")

    data = load_traj()
    rows = []
    for ds, d in data.items():
        series = [(t, g.rms.to_numpy(float)) for t, g in d.groupby("trajectory", sort=False)]
        if len(series) < 4:
            continue
        cr = {t: crossings(h) for t, h in series}
        lens = {t: len(h) for t, h in series}
        log(f"{BAR}\n{ds}: {len(series)} trajectories\n{BAR}")
        log(f"  {'k':>4s} {'crossed':>8s} {'ghat^-1(k)':>11s} {'IQR of frac':>20s} {'CV':>6s} "
            f"{'LOO R2 (n*ghat)':>16s} {'LOO R2 (const)':>15s}")
        for k in KS:
            names = [t for t, _ in series if cr[t][k] is not None]
            if len(names) < 4:
                log(f"  {k:4.1f} {len(names):3d}/{len(series):<4d}   -- fewer than four trajectories cross --")
                continue
            frac = np.array([cr[t][k] / lens[t] for t in names], float)
            idx = np.array([cr[t][k] for t in names], float)
            n = np.array([lens[t] for t in names], float)
            # leave-one-out: the constant used to predict a trajectory is estimated without it
            pred_scale, pred_const = np.empty(len(names)), np.empty(len(names))
            for j in range(len(names)):
                m = np.ones(len(names), bool)
                m[j] = False
                pred_scale[j] = n[j] * np.median(frac[m])     # proposition: i_k = n ghat^-1(k)
                pred_const[j] = np.median(idx[m])             # null: i_k = c_k, independent of n
            sst = float(((idx - idx.mean()) ** 2).sum())
            r2s = 1 - float(((idx - pred_scale) ** 2).sum()) / sst if sst > 0 else np.nan
            r2c = 1 - float(((idx - pred_const) ** 2).sum()) / sst if sst > 0 else np.nan
            cv = float(frac.std() / frac.mean()) if frac.mean() > 0 else np.nan
            q1, q3 = np.percentile(frac, [25, 75])
            rows.append({"dataset": ds, "k_sd": k, "n_crossed": len(names), "n_traj": len(series),
                         "ghat_inv": float(np.median(frac)), "frac_q25": q1, "frac_q75": q3, "frac_cv": cv,
                         "loo_r2_scaling": r2s, "loo_r2_constant": r2c,
                         "median_abs_err_scaling": float(np.median(np.abs(idx - pred_scale))),
                         "median_abs_err_constant": float(np.median(np.abs(idx - pred_const)))})
            log(f"  {k:4.1f} {len(names):3d}/{len(series):<4d} {np.median(frac):11.3f} "
                f"[{q1:8.3f}, {q3:8.3f}] {cv:6.2f} {r2s:16.3f} {r2c:15.3f}")
        log("")
    out = pd.DataFrame(rows)
    out.to_csv(T / "j29_scaling_law.csv", index=False)
    log(f"wrote j29_scaling_law.csv ({len(out)} rows)")

    log(f"\n{BAR}\nreading: where does the proposition beat the constant-record null?\n{BAR}")
    log(f"  {'dataset':7s} {'k':>4s} {'scaling wins by':>16s}  {'verdict'}")
    for _, r in out.iterrows():
        d = r.loo_r2_scaling - r.loo_r2_constant
        v = ("scaling" if d > 0.1 else ("constant" if d < -0.1 else "neither clearly"))
        log(f"  {r.dataset:7s} {r.k_sd:4.1f} {d:+16.3f}  {v}")
    log("\n  the proposition predicts that the crossover happens at the k where the indicator's rise leaves the")
    log("  noise band: below it a crossing is a noise event at a fixed record count, above it a signal event at a")
    log("  fixed life fraction. Section 4.3 measures that crossover on the same data.")
    log(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
