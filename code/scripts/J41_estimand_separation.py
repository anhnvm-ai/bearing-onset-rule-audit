"""J41 -- is the response-size relation a finding, or a change of variables?

Section 4 regresses rho(i, n), the correlation between a design's alarm index and the trajectory length,
on the size of change the design responds to. That size is READ OFF THE DATA at each design's own alarm
(J26_design_sweep.py:226-230), and the alarm index is bounded by n by construction, so a reviewer can
object that the two axes are two coordinates of one quantity: how late in life the design fires. The
paper's existing defence (sec4_families.tex:100) removes the named rule and the update equation but not
the thresholding, so it answers a weaker form of the objection.

Three questions, all answerable from tables already deposited:

  Q1  NOMINAL SHIFT. Each design's parameters imply a shift BEFORE any bearing is seen:
          thr_k       k                        the level must exceed k baseline sd
          ma5_k       k / sqrt(5)              the 5-record mean has sd sigma/sqrt(5)
          ewma(l,L)   L * sqrt(l/(2-l))        the steady-state control limit, in baseline sd
          ewmaAR      same form                (the AR correction widens the limit; same nominal family)
          cusum(k,h)  2k                       the classic tuning: reference k is tuned for a 2k shift
          adapt_L     L                        a widening threshold, nominally at L sd
      If this a-priori variable organises rho, the relation cannot be an artefact of measuring the shift
      at the alarm. Four conventions are fitted, so the answer does not rest on one choice.

  Q2  ONSET FRACTION. j26_design_sweep.csv already carries median_onset_fraction beside the two axes and
      no section of the paper uses it. If it explains rho as well as the shift does, the honest statement
      is that a graded design property -- equivalently "how large a change it waits for" or "how late it
      fires" -- organises the scaling, and the two readings are not separable on these data.

  Q3  CROSS-FAMILY PREDICTION with the a-priori variable: fit on the 12 RMS detectors, predict the 55
      control-chart designs, and the reverse.

Every R^2 carries a bearing-level interval. The 67 designs share bearings, so the unit is the bearing:
the resample is over trajectories and each design's rho, median shift and median onset fraction are
recomputed inside it, exactly as J36_design_cv.py:247-275 does.

This script has its OWN rng (SEED below). J36 runs one default_rng(0) across both datasets
(J36:147 feeding J36:258), so drawing from that stream would change every interval already published.

Inputs : results/tables/j26_design_sweep.csv          67 designs x 4 datasets, the published summaries
         results/tables/j26_design_sweep_per_traj.csv one row per design x trajectory (alarm, shift)
         scripts/J26_design_sweep.py                  build_grid(), for the parameters of each design
Outputs: results/tables/j41_estimand_r2.csv           point R^2 per dataset x model
         results/tables/j41_estimand_bootstrap.csv    one row per usable resample
         results/tables/j41_nominal_conventions.csv   R^2 under four nominal-shift conventions
         results/tables/j41_cross_family_nominal.csv  out-of-sample prediction across families
         results/logs/j41_run.log

Usage: python scripts/J41_estimand_separation.py [N_BOOT]      (default 20000, as in J36)
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"

SEED = 41
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
MIN_FIRED = 4                      # as in J26 and J36: no correlation over fewer than four trajectories
DATASETS = ["xjtu", "femto"]       # the two interval-bearing datasets of Section 4, as J36:66
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO", "ims": "IMS", "nsk": "NSK 6205"}
BAR = "=" * 78

# The nominal shift each design answers to, from its parameters alone. Convention A is the primary one;
# B, C and D vary one choice each so the reader can see what the answer rests on. D is included because
# it is WRONG on purpose: it drops the sqrt(lambda/(2-lambda)) factor that the EWMA limit actually has.
CONVENTIONS = {
    "A_primary": dict(cusum=2.0, ma=1 / np.sqrt(5.0), ewma_lambda=True,
                      note="cusum 2k, ma k/sqrt(5), ewma L*sqrt(l/(2-l))"),
    "B_cusum_k": dict(cusum=1.0, ma=1 / np.sqrt(5.0), ewma_lambda=True,
                      note="cusum k instead of 2k"),
    "C_ma_k": dict(cusum=2.0, ma=1.0, ewma_lambda=True,
                   note="ma k instead of k/sqrt(5)"),
    "D_ewma_L_only": dict(cusum=2.0, ma=1 / np.sqrt(5.0), ewma_lambda=False,
                          note="ewma L, dropping the lambda factor the limit has (a deliberate control)"),
}

LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


# ------------------------------------------------------------------ the two statistics, as J36 writes them
def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < MIN_FIRED:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    d = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d > 0 else np.nan


def r2_in_sample(y: np.ndarray, X: np.ndarray) -> float:
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    sst = float((y - y.mean()) @ (y - y.mean()))
    return float("nan") if sst <= 0 else float(1 - (r @ r) / sst)


def quad(v: np.ndarray, logged: bool) -> list[np.ndarray]:
    """The quadratic used throughout Section 4: log for a shift, raw for a fraction."""
    x = np.log(np.clip(v, 1e-3, None)) if logged else v
    return [x, x ** 2]


def dmat(n: int, *cols: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(n)] + list(cols))


def models(rho, shift, nominal, fraction, is_det) -> dict[str, float]:
    """Every model compared on the same designs, so the R^2 are comparable."""
    n = len(rho)
    ls, ln = quad(shift, True), quad(nominal, True)
    f = quad(fraction, False)
    return {
        "measured_shift": r2_in_sample(rho, dmat(n, *ls)),
        "nominal_shift": r2_in_sample(rho, dmat(n, *ln)),
        "onset_fraction": r2_in_sample(rho, dmat(n, *f)),
        "family": r2_in_sample(rho, dmat(n, is_det)),
        "nominal_plus_family": r2_in_sample(rho, dmat(n, *ln, is_det)),
        "fraction_plus_nominal": r2_in_sample(rho, dmat(n, *f, *ln)),
        "measured_plus_fraction": r2_in_sample(rho, dmat(n, *ls, *f)),
    }


# ------------------------------------------------------------------ nominal shift from the design grid
def nominal_shift(spec: dict, c: dict) -> float:
    kind, p = spec["kind"], spec["p"]
    if kind == "threshold":
        return p["k"]
    if kind == "ma":
        return p["k"] * c["ma"]
    if kind in ("ewma", "ewma_ar"):
        return p["L"] * (np.sqrt(p["lam"] / (2 - p["lam"])) if c["ewma_lambda"] else 1.0)
    if kind == "cusum":
        return c["cusum"] * p["k"]
    if kind == "adaptive":
        return p["L"]
    raise ValueError(kind)


def load_grid() -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location("j26", ROOT / "scripts" / "J26_design_sweep.py")
    j26 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(j26)
    g = j26.build_grid()
    # the design's kind is already in the sweep table; carrying it here would collide in the merge
    rows = [{"design": s["design"],
             **{f"nominal_{k}": nominal_shift(s, c) for k, c in CONVENTIONS.items()}} for s in g]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    log(BAR)
    log("J41  ESTIMAND SEPARATION".center(78))
    log("is the response-size relation a finding, or a change of variables?".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S} | seed {SEED} (its own stream) | {N_BOOT:,} resamples")

    grid = load_grid()
    sweep = pd.read_csv(TABLES / "j26_design_sweep.csv").merge(grid, on="design", how="left")
    per_traj = pd.read_csv(TABLES / "j26_design_sweep_per_traj.csv")
    if sweep.nominal_A_primary.isna().any():
        raise SystemExit("a design in j26_design_sweep.csv is not in build_grid()")
    log(f"\n{len(grid)} designs; nominal shift (convention A) from {grid.nominal_A_primary.min():.2f} "
        f"to {grid.nominal_A_primary.max():.2f} baseline sd, median {grid.nominal_A_primary.median():.2f}")

    rng = np.random.default_rng(SEED)
    point_rows, boot_rows, conv_rows, cf_rows = [], [], [], []

    for ds in DATASETS + ["ims"]:
        s = sweep[(sweep.dataset == ds) & sweep.rho_index_length.notna()
                  & sweep.median_shift_sd.notna() & sweep.median_onset_fraction.notna()].copy()
        if len(s) < 10:
            continue
        interpreted = ds in DATASETS
        rho = s.rho_index_length.to_numpy(float)
        shift = s.median_shift_sd.to_numpy(float)
        frac = s.median_onset_fraction.to_numpy(float)
        is_det = (s.family == "detector").to_numpy(float)

        log(f"\n{BAR}\n{DS_NAME[ds]}  ({len(s)} designs with a fitted rho)"
            f"{'' if interpreted else '   -- shown, not interpreted (as Section 4 treats IMS)'}\n{BAR}")

        # ---- (1) point R^2, every model on the same designs
        pt = models(rho, shift, s.nominal_A_primary.to_numpy(float), frac, is_det)
        for k, v in pt.items():
            point_rows.append({"dataset": ds, "model": k, "n_designs": len(s), "r2": v})
        log("  R^2 of rho(alarm index, trajectory length):")
        for k in ("measured_shift", "nominal_shift", "onset_fraction", "family"):
            tag = {"measured_shift": "  <- the number Section 4 reports",
                   "nominal_shift": "  <- from the design's parameters only, no bearing seen"}.get(k, "")
            log(f"    {k:22} {pt[k]:6.3f}{tag}")
        log(f"    {'nominal + family':22} {pt['nominal_plus_family']:6.3f}")
        log(f"    nominal adds this much on top of onset fraction: "
            f"{pt['fraction_plus_nominal'] - pt['onset_fraction']:+.3f}")
        log(f"    onset fraction adds this much on top of the measured shift: "
            f"{pt['measured_plus_fraction'] - pt['measured_shift']:+.3f}")
        lognom = np.log(np.clip(s.nominal_A_primary.to_numpy(float), 1e-3, None))
        log(f"    corr(log nominal, log measured) = {np.corrcoef(lognom, np.log(shift))[0, 1]:+.3f}   "
            f"corr(log measured, onset fraction) = {np.corrcoef(np.log(shift), frac)[0, 1]:+.3f}")
        if s.rho_fraction_length.notna().any():
            g = s.rho_fraction_length.dropna().to_numpy(float)
            log(f"    rho(onset FRACTION, length), the length-decoupled restatement J26 computes and the "
                f"paper never reports: median {np.median(g):+.3f}")

        # ---- (2) the four nominal conventions
        log("\n  the nominal shift under four conventions (D is a deliberate control: it drops the "
            "lambda factor the EWMA limit has):")
        for name, c in CONVENTIONS.items():
            v = r2_in_sample(rho, dmat(len(s), *quad(s[f"nominal_{name}"].to_numpy(float), True)))
            conv_rows.append({"dataset": ds, "convention": name, "note": c["note"], "r2": v})
            log(f"    {name:14} {v:6.3f}   {c['note']}")

        # ---- (3) cross-family prediction with the a-priori variable
        log("\n  out-of-sample prediction with the nominal shift, fitted on one family and applied to the other.")
        log("  The test sets follow Section IV-C: the conventional charts, the AR-corrected and adaptive")
        log("  designs it keeps apart as a class the fit has not seen, and the two together.")
        kinds = s.kind.to_numpy()
        CONV, NOVEL = ("ewma", "cusum"), ("ewma_ar", "adaptive")
        tests = [("detectors -> conventional charts", np.isin(kinds, CONV)),
                 ("detectors -> novel charts", np.isin(kinds, NOVEL)),
                 ("detectors -> charts", ~is_det.astype(bool))]
        det_m = is_det.astype(bool)
        for what, te_m in tests + [("charts -> detectors", det_m)]:
            tr_m = ~det_m if what == "charts -> detectors" else det_m
            if tr_m.sum() < 5 or te_m.sum() < 5:
                continue
            Xtr = dmat(int(tr_m.sum()), *quad(s.nominal_A_primary.to_numpy(float)[tr_m], True))
            Xte = dmat(int(te_m.sum()), *quad(s.nominal_A_primary.to_numpy(float)[te_m], True))
            b = np.linalg.lstsq(Xtr, rho[tr_m], rcond=None)[0]
            pred, y = Xte @ b, rho[te_m]
            sst = float(((y - y.mean()) ** 2).sum())
            oos = 1 - float(((y - pred) ** 2).sum()) / sst if sst > 0 else np.nan
            mae = float(np.median(np.abs(y - pred)))
            cf_rows.append({"dataset": ds, "direction": what, "n_train": int(tr_m.sum()),
                            "n_test": int(te_m.sum()), "r2_oos": oos, "median_abs_error": mae})
            log(f"    {what:34} n_train={int(tr_m.sum()):3}  n_test={int(te_m.sum()):3}  "
                f"R2_oos = {oos:+.3f}   median |error| = {mae:.3f}")

        if not interpreted:
            continue

        # ---- (4) bearing-level intervals: recompute every design's summaries inside the resample
        p = per_traj[per_traj.dataset == ds]
        designs = list(dict.fromkeys(p.design))
        trajs = list(dict.fromkeys(p.trajectory))
        nT = len(trajs)
        d_idx = {d: i for i, d in enumerate(designs)}
        t_idx = {t: i for i, t in enumerate(trajs)}
        ALARM = np.full((len(designs), nT), np.nan)
        SHIFT = np.full((len(designs), nT), np.nan)
        NREC = np.zeros(nT)
        for r in p.itertuples():
            i, j = d_idx[r.design], t_idx[r.trajectory]
            NREC[j] = r.n_records
            if r.fired:
                ALARM[i, j] = r.alarm_index
                SHIFT[i, j] = r.shift_sd
        fam = {d: p[p.design == d].family.iloc[0] for d in designs}
        IS_DET = np.array([fam[d] == "detector" for d in designs], dtype=float)
        NOM = grid.set_index("design").loc[designs, "nominal_A_primary"].to_numpy(float)
        log(f"\n  bearing bootstrap: {len(designs)} designs x {nT} bearings, "
            f"{N_BOOT:,} resamples of whole bearings")

        got = 0
        for _ in range(N_BOOT):
            sel = rng.integers(0, nT, nT)
            nvals = NREC[sel]
            A, S = ALARM[:, sel], SHIFT[:, sel]
            rho_b = np.full(len(designs), np.nan)
            shift_b = np.full(len(designs), np.nan)
            frac_b = np.full(len(designs), np.nan)
            for i in range(len(designs)):
                m = np.isfinite(A[i])
                if m.sum() >= MIN_FIRED:
                    rho_b[i] = spearman(nvals[m], A[i][m])
                    shift_b[i] = np.nanmedian(S[i][m])
                    frac_b[i] = np.nanmedian(A[i][m] / nvals[m])
            ok = np.isfinite(rho_b) & np.isfinite(shift_b) & np.isfinite(frac_b)
            if ok.sum() < 10 or len(set(IS_DET[ok])) < 2:
                continue
            r = models(rho_b[ok], shift_b[ok], NOM[ok], frac_b[ok], IS_DET[ok])
            boot_rows.append({"dataset": ds, "n_designs": int(ok.sum()), **r})
            got += 1
        log(f"    {got:,} of {N_BOOT:,} resamples usable")

    # ---- write and summarise
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(point_rows).to_csv(TABLES / "j41_estimand_r2.csv", index=False)
    pd.DataFrame(conv_rows).to_csv(TABLES / "j41_nominal_conventions.csv", index=False)
    pd.DataFrame(cf_rows).to_csv(TABLES / "j41_cross_family_nominal.csv", index=False)
    bt = pd.DataFrame(boot_rows)
    bt.to_csv(TABLES / "j41_estimand_bootstrap.csv", index=False)

    log(f"\n{BAR}\nbearing-bootstrap intervals\n{BAR}")
    for ds in DATASETS:
        b = bt[bt.dataset == ds]
        if not len(b):
            continue
        log(f"\n  {DS_NAME[ds]}  ({len(b):,} resamples)")
        for col in ("measured_shift", "nominal_shift", "onset_fraction", "family"):
            v = b[col].to_numpy(float)
            log(f"    R2 {col:16} median {np.median(v):.3f}   "
                f"95% [{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]")
        wins = float((b.nominal_shift > b.family).mean())
        log(f"    the a-priori nominal shift beats the family label in {wins * 100:.1f}% of resamples")
        gap = (b.onset_fraction - b.nominal_shift).to_numpy(float)
        log(f"    onset fraction minus nominal shift: median {np.median(gap):+.3f}   "
            f"95% [{np.percentile(gap, 2.5):+.3f}, {np.percentile(gap, 97.5):+.3f}]   "
            f"(fraction is the larger in {float((gap > 0).mean()) * 100:.1f}% of resamples)")
        add = (b.fraction_plus_nominal - b.onset_fraction).to_numpy(float)
        log(f"    nominal shift adds on top of onset fraction: median {np.median(add):+.3f}   "
            f"95% [{np.percentile(add, 2.5):+.3f}, {np.percentile(add, 97.5):+.3f}]")

    log(f"\nwrote j41_estimand_r2.csv, j41_estimand_bootstrap.csv, j41_nominal_conventions.csv, "
        f"j41_cross_family_nominal.csv; finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    (LOGS / "j41_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
