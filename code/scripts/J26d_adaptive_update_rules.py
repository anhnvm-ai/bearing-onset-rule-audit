"""J26d -- does the adaptive-limit result depend on how the limits are updated?

Run from bearing_multimodal_atc/:
    python scripts/J26d_adaptive_update_rules.py

Why this exists
---------------
Section 4.3 includes a chart whose limits adapt to the bearing's own data, "in the manner of" the dynamic control chart
of Li, Wang and Shen (Meas. Sci. Technol. 35, 2024). Their abstract says only that the limits are controlled
"dynamically ... based on the bearing's condition monitoring data to prevent false alarms"; the update rule itself
could not be obtained. J26 implemented one reading, an expanding window over every record seen so far, and it alarmed
at a median of about 20 baseline standard deviations. An expanding window absorbs the degradation it is meant to
detect, so that result may belong to the update rule rather than to adaptive limits as such.

This script applies four causal update rules that all fit the published description, each at L in {2, ..., 6}, to the
same trajectories, and asks two separate questions:

  1. how late does an adaptive chart alarm?          -> expected to depend on the update rule
  2. does it still lie on the curve of Section 4.3?  -> the claim the paper needs

Update rules (Phase-I mean and sd from the J11 baseline; a limit never tightens below the Phase-I sd; 3 confirmations)
    expanding   mean and sd of every record seen so far                        (the J26 reading)
    sliding     mean and sd of the last n_b records                            (forgets the healthy past)
    in-control  mean and sd of the records seen so far that did not exceed     (does not absorb the shift)
                the limit in force when they arrived
    ewma-limit  exponentially weighted mean and variance, weight 0.05          (a common "dynamic limit")

Outputs
    results/tables/j26d_adaptive_rules.csv    one row per dataset x update rule x L
    results/logs/j26d_run.log
Nothing from J26 is overwritten.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
T, LOGS = ROOT / "results" / "tables", ROOT / "results" / "logs"
spec = importlib.util.spec_from_file_location("j26", ROOT / "scripts" / "J26_design_sweep.py")
J26 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J26)
BASE_FRAC, CONFIRM = J26.BASE_FRAC, J26.CONFIRM
RULES = ("expanding", "sliding", "in-control", "ewma-limit")
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


def adaptive_onset(h: np.ndarray, rule: str, L: float) -> int | None:
    n = len(h)
    nb = max(5, int(n * BASE_FRAC))
    mu0, sd0 = float(h[:nb].mean()), max(float(h[:nb].std()), 1e-12)
    lim_mu, lim_sd = mu0, sd0
    kept = list(h[:nb])                        # in-control pool
    em, ev = mu0, sd0 ** 2                     # ewma-limit state
    run = 0
    for i in range(n):
        if i >= nb:
            if rule == "expanding":
                w = h[:i]
                lim_mu, lim_sd = float(w.mean()), max(float(w.std()), sd0)
            elif rule == "sliding":
                w = h[i - nb:i]
                lim_mu, lim_sd = float(w.mean()), max(float(w.std()), sd0)
            elif rule == "in-control":
                a = np.asarray(kept)
                lim_mu, lim_sd = float(a.mean()), max(float(a.std()), sd0)
            elif rule == "ewma-limit":
                lim_mu, lim_sd = em, max(float(np.sqrt(ev)), sd0)
        above = h[i] > lim_mu + L * lim_sd
        run = run + 1 if above else 0
        if run >= CONFIRM:
            return i - CONFIRM + 1
        if i >= nb:                            # update the state with record i after it has been judged
            if rule == "in-control" and not above:
                kept.append(h[i])
            if rule == "ewma-limit":
                d = h[i] - em
                em += 0.05 * d
                ev = 0.95 * (ev + 0.05 * d * d)
    return None


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j26d_run.log", "w", encoding="utf-8")
    log(f"J26d  adaptive-limit update rules, started {datetime.now():%Y-%m-%d %H:%M:%S}")
    data = J26.load_traj()
    sw = pd.read_csv(T / "j26_design_sweep.csv")
    sw = sw[sw.median_shift_sd.notna() & sw.rho_index_length.notna()]
    fits = {}
    for ds in ("xjtu", "femto"):
        g = sw[(sw.dataset == ds) & ~sw.family.str.startswith("chart, adaptive")]   # the curve without adaptive designs
        fits[ds] = np.polyfit(np.log(g.median_shift_sd.clip(lower=1e-3)), g.rho_index_length, 2)
    rows = []
    for ds in ("xjtu", "femto"):
        series = [(t, g.rms.to_numpy(float)) for t, g in data[ds].groupby("trajectory", sort=False)]
        for rule in RULES:
            for L in (2.0, 3.0, 4.0, 5.0, 6.0):
                idx, lens, shifts, inside = [], [], [], 0
                for _, h in series:
                    n = len(h)
                    nb = max(5, int(n * BASE_FRAC))
                    i = adaptive_onset(h, rule, L)
                    lens.append(n)
                    if i is None:
                        idx.append(np.nan)
                        continue
                    idx.append(i)
                    inside += int(i < nb)
                    mu, sd = float(h[:nb].mean()), max(float(h[:nb].std()), 1e-12)
                    shifts.append((float(h[i:i + CONFIRM].mean()) - mu) / sd)
                a, b = np.asarray(idx, float), np.asarray(lens, float)
                ok = np.isfinite(a)
                rho = float(spearmanr(b[ok], a[ok])[0]) if ok.sum() >= 4 else np.nan
                shift = float(np.median(shifts)) if shifts else np.nan
                pred = float(np.polyval(fits[ds], np.log(max(shift, 1e-3)))) if np.isfinite(shift) else np.nan
                rows.append({"dataset": ds, "update_rule": rule, "L": L, "n_traj": len(series), "n_fired": int(ok.sum()),
                             "median_shift_sd": shift, "rho_index_length": rho, "inside_baseline": inside,
                             "rho_predicted_by_curve": pred, "residual": rho - pred})
    out = pd.DataFrame(rows)
    out.to_csv(T / "j26d_adaptive_rules.csv", index=False)

    for ds in ("xjtu", "femto"):
        g = out[out.dataset == ds]
        others = sw[(sw.dataset == ds) & ~sw.family.str.startswith("chart, adaptive")]
        resid_other = others.rho_index_length - np.polyval(fits[ds], np.log(others.median_shift_sd.clip(lower=1e-3)))
        log(f"\n{ds}: residual spread of the non-adaptive designs about their own curve: "
            f"median |r| {resid_other.abs().median():.3f}, 90th pct {resid_other.abs().quantile(0.9):.3f}")
        log(f"  {'rule':11s} {'fired':>7s} {'shift sd (median over L)':>25s} {'rho':>7s} {'|residual|':>11s} {'inside':>7s}")
        for rule in RULES:
            r = g[g.update_rule == rule]
            log(f"  {rule:11s} {int(r.n_fired.sum()):3d}/{int(r.n_traj.sum()):<3d} "
                f"{r.median_shift_sd.median():12.2f} [{r.median_shift_sd.min():5.2f}, {r.median_shift_sd.max():6.2f}] "
                f"{r.rho_index_length.median():+7.3f} {r.residual.abs().median():11.3f} {int(r.inside_baseline.sum()):7d}")
        log(f"  all adaptive variants: median |residual| {g.residual.abs().median():.3f}, "
            f"share within the 90th pct of the non-adaptive spread {(g.residual.abs() <= resid_other.abs().quantile(0.9)).mean():.2f}")
    log("\nwrote j26d_adaptive_rules.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
