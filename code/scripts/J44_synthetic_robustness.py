"""J44 -- does the known-change reading survive noise that is not Gaussian AR(1) at one phi?

Section S5 simulates Gaussian AR(1) series at the median detrended baseline autocorrelation of XJTU-SY,
inserts a step or a ramp, and reports what each family recovers. Section VII concedes that real healthy
indicators may have heavier tails or longer memory. This widens that simulation in the two directions the
concession names, and adds the arm it lacks entirely: series with no change at all, which measure the
false alarms a rule raises when nothing happens.

  noise      Gaussian, Student-t with nu = 5, Student-t with nu = 3, each rescaled so that the marginal
             variance is 1 and delta keeps meaning "baseline standard deviations"
  phi        0.0, the measured reference, 0.6 and 0.9. The last is OUTSIDE the range these data show
             (the raw median is +0.763), and is reported as a stress test, not as a property of bearings
  shape      step and ramp, as Section S5 has them. Sigmoid and convex drifts are deliberately not run:
             they have no single change time, so a detection delay would depend on the reference chosen
  delta      0, 1, 2, 3, 6 baseline standard deviations; delta = 0 is the degradation-free arm
  metrics    false-alarm rate, share alarming before the change, detection probability after it, delay
             in records and in life fraction, and the size of change measured at the alarm

The rules are applied by J20's vectorised implementation, which J20 itself asserts is identical to the
scalar path used on the real bearings, so nothing here re-implements a rule. J24 is not touched: it draws
every cell from one generator, so an edit there would move every number of Table S1. This script seeds
each cell separately from a CRC of the cell's parameters, so its grid can be extended later without moving
what it already reported, and a re-run on any machine reproduces the table exactly.

Inputs : results/tables/j22_baseline_autocorrelation.csv   the measured phi
         results/tables/fig1_fpt_matrix.csv                the 15 XJTU-SY trajectory lengths
Outputs: results/tables/j44_robustness.csv                 one row per cell x rule
         results/logs/j44_run.log

Usage: python scripts/J44_synthetic_robustness.py            (R=400 series per cell, as J24)
       R=50 python scripts/J44_synthetic_robustness.py       a quick pass
"""
from __future__ import annotations

import importlib.util
import os
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"

R = int(os.environ.get("R", "400"))
SEED = 44
CONFIRM, BASE_FRAC = 3, 0.20          # the conventions of J11, J20 and J26
NOISE = [("gauss", np.nan), ("t5", 5.0), ("t3", 3.0)]
PHI_EXTRA = [0.0, 0.6, 0.9]           # the reference phi is read from the data and inserted among these
SHAPES = ["step", "ramp"]
DELTAS = [0.0, 1.0, 2.0, 3.0, 6.0]
US = [0.3, 0.5, 0.7]
BAR = "=" * 78

LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def series(rng, S: int, n: int, phi: float, nu: float) -> np.ndarray:
    """AR(1) with unit marginal variance; Student-t innovations when nu is finite."""
    if np.isfinite(nu):
        e = rng.standard_t(nu, size=(S, n)) / np.sqrt(nu / (nu - 2.0))
    else:
        e = rng.standard_normal((S, n))
    a = np.sqrt(max(1 - phi ** 2, 1e-12))
    x = np.empty((S, n))
    x[:, 0] = e[:, 0]
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + a * e[:, t]
    return x


def mean_profile(shape: str, n: int, tau: int, delta: float) -> np.ndarray:
    m = np.zeros(n)
    if delta == 0.0 or tau >= n:
        return m
    if shape == "step":
        m[tau:] = delta
    elif shape == "ramp":
        k = n - tau
        m[tau:] = delta * (np.arange(k) / max(k - 1, 1))
    else:
        raise ValueError(shape)
    return m


def shift_at_alarm(H: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """The level over the confirming records at the alarm, minus the baseline mean, in baseline sd."""
    S, n = H.shape
    nb = max(5, int(n * BASE_FRAC))
    mu = H[:, :nb].mean(axis=1)
    sd = np.maximum(H[:, :nb].std(axis=1), 1e-12)
    out = np.full(S, np.nan)
    for s in range(S):
        i = idx[s]
        if i >= 0:
            out[s] = (H[s, i:i + CONFIRM].mean() - mu[s]) / sd[s]
    return out


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    log(BAR)
    log("J44  SYNTHETIC ROBUSTNESS".center(78))
    log("heavier tails, longer memory, and the arm with no change at all".center(78))
    log(BAR)

    j20 = load("j20", "J20_arl_mechanism.py")
    rules = list(j20.RULES)
    base = pd.read_csv(T / "j22_baseline_autocorrelation.csv")
    phi_ref = float(np.clip(base[base.dataset == "xjtu"].phi_detrended.median(), -0.99, 0.99))
    phis = sorted(set(PHI_EXTRA + [round(phi_ref, 3)]))
    lens = sorted(pd.read_csv(T / "fig1_fpt_matrix.csv").query("dataset == 'xjtu'")
                  .drop_duplicates("trajectory").n_records)

    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S} | R={R} series per length per cell | seed {SEED}")
    log(f"phi grid {phis} (the reference {phi_ref:.3f} is the median detrended baseline autocorrelation of XJTU-SY;"
        f" the raw median is {float(base[base.dataset == 'xjtu'].phi_raw.median()):.3f}, so 0.9 is outside the range"
        f" these data show and is a stress test)")
    log(f"lengths: the {len(lens)} XJTU-SY trajectories, {min(lens)} to {max(lens)} records")
    log(f"rules: {', '.join(rules)}")

    rows = []
    cells = 0
    for nname, nu in NOISE:
        for phi in phis:
            for shape in SHAPES:
                for delta in DELTAS:
                    for u in (US if delta > 0 else [np.nan]):
                        if delta == 0.0 and shape != SHAPES[0]:
                            continue          # the degradation-free arm has no shape
                        cells += 1
                        # one generator per cell, so the grid can grow without moving what it reported.
                        # The key is a CRC rather than hash(): hash() of a str is salted per process, so
                        # that seed would have changed from run to run (fixed 2026-09-24).
                        key = zlib.crc32(f"{nname}|{phi:.3f}|{shape}|{delta:.1f}|"
                                         f"{0.0 if np.isnan(u) else u:.2f}".encode())
                        rng = np.random.default_rng([SEED, key])
                        acc = {r: {"fired": [], "before": [], "delay": [], "delay_frac": [], "shift": []}
                               for r in rules}
                        for n in lens:
                            tau = int(round((0.0 if np.isnan(u) else u) * n))
                            H = series(rng, R, n, phi, nu) + mean_profile(shape, n, tau, delta)
                            for r in rules:
                                idx = j20.onsets(H, r)
                                fired = idx >= 0
                                acc[r]["fired"].append(fired)
                                if delta > 0:
                                    acc[r]["before"].append(fired & (idx < tau))
                                    late = fired & (idx >= tau)
                                    acc[r]["delay"].append((idx - tau)[late].astype(float))
                                    acc[r]["delay_frac"].append(((idx - tau) / n)[late].astype(float))
                                acc[r]["shift"].append(shift_at_alarm(H, idx)[fired])
                        for r in rules:
                            fired = np.concatenate(acc[r]["fired"])
                            sh = np.concatenate(acc[r]["shift"]) if acc[r]["shift"] else np.array([])
                            row = {"noise": nname, "nu": nu, "phi": phi, "shape": shape if delta > 0 else "none",
                                   "delta_sd": delta, "u": u, "rule": r, "family": j20.FAMILY[r],
                                   "n_series": int(fired.size), "share_fired": float(fired.mean()),
                                   "median_shift_at_alarm": float(np.nanmedian(sh)) if sh.size else np.nan}
                            if delta > 0:
                                before = np.concatenate(acc[r]["before"])
                                d = np.concatenate(acc[r]["delay"]) if acc[r]["delay"] else np.array([])
                                df = np.concatenate(acc[r]["delay_frac"]) if acc[r]["delay_frac"] else np.array([])
                                row |= {"share_fire_before_change": float(before.mean()),
                                        "share_detect_after_change": float((fired.mean() - before.mean())),
                                        "share_never_fired": float(1 - fired.mean()),
                                        "median_delay_records": float(np.median(d)) if d.size else np.nan,
                                        "iqr_delay_records": float(np.percentile(d, 75) - np.percentile(d, 25)) if d.size else np.nan,
                                        "median_delay_life_fraction": float(np.median(df)) if df.size else np.nan}
                            else:
                                row |= {"false_alarm_rate": float(fired.mean())}
                            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(T / "j44_robustness.csv", index=False)
    log(f"\n{cells} cells x {len(rules)} rules = {len(df)} rows")

    # ---- the readings the manuscript will use
    log(f"\n{BAR}\nfalse alarms when nothing happens (delta = 0)\n{BAR}")
    z = df[df.delta_sd == 0]
    log(f"  {'noise':6} {'phi':>5}   " + "  ".join(f"{r:>9}" for r in rules))
    for (nname, phi), g in z.groupby(["noise", "phi"]):
        s = g.set_index("rule").false_alarm_rate
        log(f"  {nname:6} {phi:5.3f}   " + "  ".join(f"{s[r]:9.3f}" for r in rules))

    log(f"\n{BAR}\nalarms before the change, by family (step, pooled over u and the lengths)\n{BAR}")
    st = df[df["shape"] == "step"]
    for (nname, phi), g in st.groupby(["noise", "phi"]):
        by = g.groupby("family").share_fire_before_change.median()
        log(f"  {nname:6} phi {phi:5.3f}   charts {by.get('control chart', np.nan):.3f}   "
            f"detectors {by.get('detector', np.nan):.3f}")

    log(f"\n{BAR}\nthe reference cell against Table S1 (Gaussian, phi = {phi_ref:.3f}, step)\n{BAR}")
    ref = df[(df.noise == "gauss") & (np.isclose(df.phi, round(phi_ref, 3))) & (df["shape"] == "step")]
    s1 = pd.read_csv(T / "j24_sim_single_change.csv")
    s1 = s1[s1["shape"] == "step"]
    for r in rules:
        a = ref[ref.rule == r].share_fire_before_change.median()
        b = s1[s1.rule == r].share_fire_before_change.median()
        log(f"  {r:9} this run {a:.3f}   J24 {b:.3f}   difference {abs(a - b):.3f}")
    log("  (the two runs draw different streams, so they agree to Monte-Carlo error, not exactly)")

    log(f"\nwrote j44_robustness.csv; finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    (LOGS / "j44_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
