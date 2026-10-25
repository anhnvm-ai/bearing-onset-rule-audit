"""J61 -- WP-A: the RMS rules calibrated to a common false-alarm rate (R1.4, R3.4).

R1.4: a comment of the review reports, answered in the response letter.
R3.4: a comment of the review reports, answered in the response letter.

At their textbook settings the two families are not equally calibrated: on degradation-free AR(1) series the charts
alarm in about 59% of series and the detectors in about 0.5% (J44). This script calibrates one multiplier per rule
so that every rule has the same false-alarm rate, then repeats the comparisons of the manuscript at that rate.

  rules calibrated   threshold on RMS (k); five-record moving average of RMS (k; merges R3-R5);
                     EWMA with lambda = 0.2 (L); one-sided CUSUM with kappa = 0.5 sigma (H)
  protocols          P-prospective (primary) and P-retro (control), exactly as J60 implements them
  null model         Gaussian AR(1), unit variance, phi = dataset median of the detrended baseline lag-1
                     autocorrelation (j22_baseline_autocorrelation.csv: XJTU-SY 0.367, FEMTO 0.264);
                     sensitivity: Student-t(3) innovations rescaled to unit variance; phi at the 25th and 75th
                     percentiles of the per-trajectory values
  simulation         R = 4,000 series at every eligible trajectory length of the dataset; the full protocol is
                     applied to each series (Phase-I estimation and cleaning on that series; c = 3)
  FAR                probability of at least one alarm at t >= t0 on a no-change series, averaged with equal weight
                     over the trajectory lengths; primary 5%, sensitivity 1% and 10%; ARL0 as a secondary measure
                     on series of 10,000 records
  calibration        bisection on the exact Monte Carlo FAR function (amendment A1): for each series and rule the
                     critical multiplier m* (the largest multiplier at which it still alarms) is computed once, the
                     rule alarms at m iff m < m*, and FAR(m) is the share of series with m* > m
  design FAR         the FAR of each of the 67 designs at its own parameters, from the same series (for WP-F, J64)

Outcomes on the real trajectories at the calibrated multipliers, on the bearings eligible under P-prospective
(XJTU-SY 13, FEMTO 11), for both protocols: onset fraction u, b1 - 1 with a paired bearing-bootstrap interval
(B = 20,000, stratified by condition, seed 61), shift at the alarm d, positive rate, Cohen's kappa and disagreement
between chart and detector labels (in place of the submitted 68.9%), and the target distance D_j. The uncalibrated
rules (3sig_rms, ma3sig, ma5sig, ewma, cusum) are evaluated on the same bearings and resamples, so calibrated and
uncalibrated gaps are paired. The pre-specified reading of Section 8 of the analysis plan (with amendment A1) is
applied at the end.

Pre-specified in paper/access_resub/ANALYSIS_PLAN_2026-10-24.md, Section 8; amendment A1.

Outputs (results/tables/):
  j61_calibration.csv     dataset x protocol x noise arm x level x rule: multiplier, achieved FAR, Monte Carlo SE
  j61_arl0.csv            dataset x protocol x level x rule: ARL0 on 10,000-record series
  j61_design_far.csv      dataset x protocol x noise arm x design: FAR at the design's own parameters
  j61_matched_onsets.csv  protocol x noise arm x level x trajectory x rule: alarm, u, d (real trajectories)
  j61_matched_summary.csv protocol x noise arm x level x dataset x rule: coverage, u, b1 - 1 [CI], d, positive rate
  j61_matched_gap.csv     protocol x noise arm x level x dataset: gaps, intervals, kappa, D_j, d difference, reading
results/logs/j61_run.log
Usage: python scripts/J61_far_calibration.py                (R = 4,000, B = 20,000)
       R=500 B=500 python scripts/J61_far_calibration.py    (quick run; not for the manuscript)
"""
from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
SEED = 61
R = int(os.environ.get("R", "4000"))
B = int(os.environ.get("B", "20000"))
os.environ["B"] = str(B)                # J60 reads B at import; keep the two equal
LEVELS = (0.05, 0.01, 0.10)             # primary first
PROTOCOLS = ("P-prospective", "P-retro")
INTERVAL_DS = ("xjtu", "femto")
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO"}
ARL_LEN, R_ARL = 10_000, 1_000
MAIN_N0 = 20
LAM, KAPPA = 0.2, 0.5
EWMA_LAMS = (0.05, 0.1, 0.2, 0.4, 0.8)  # the EWMA grid of J26
CUSUM_KS = (0.25, 0.5, 1.0, 2.0, 3.0)   # the CUSUM grid of J26
CAL_RULES = ("thr", "ma", "ewma", "cusum")
CAL_SPEC = {"thr": ("threshold", "k"), "ma": ("ma", "k"), "ewma": ("ewma", "L"), "cusum": ("cusum", "H")}
DETS_CAL, CHARTS_CAL = ("thr", "ma"), ("ewma", "cusum")
RMS3, CHARTS = ("3sig_rms", "ma3sig", "ma5sig"), ("ewma", "cusum")
TOL_FAR = 0.005
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


# ------------------------------------------------------------------ simulation and vectorised protocol
def simulate(rng: np.random.Generator, n_series: int, n: int, phi: float, noise: str) -> np.ndarray:
    """Stationary AR(1) with unit marginal variance: x0 = e0, x_t = phi x_{t-1} + sqrt(1 - phi^2) e_t."""
    if noise == "t3":
        e = rng.standard_t(3, size=(n_series, n)) / np.sqrt(3.0)
    else:
        e = rng.standard_normal((n_series, n))
    a = np.sqrt(max(1 - phi ** 2, 1e-12))
    e[:, 0] = e[:, 0] / a
    return lfilter([a], [1.0, -phi], e, axis=1)


def phase1_vec(H: np.ndarray, W: int, clean: bool, min_keep: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised J60.phase1 with the classical estimator: (mu, sd) per series."""
    X = H[:, :W]
    keep = np.ones_like(X, bool)

    def est(k):
        cnt = k.sum(1)
        mu = (X * k).sum(1) / cnt
        var = (((X - mu[:, None]) ** 2) * k).sum(1) / cnt
        return mu, np.maximum(np.sqrt(var), 1e-12)

    if clean:
        for _ in range(3):
            mu, sd = est(keep)
            flag = keep & (np.abs(X - mu[:, None]) > 3.0 * sd[:, None])
            new_keep = keep & ~flag
            ok = flag.any(1) & (new_keep.sum(1) >= min_keep)
            if not ok.any():
                break
            keep[ok] = new_keep[ok]
    return est(keep)


def min3_max(Xs: np.ndarray, t0: int) -> np.ndarray:
    """Critical multiplier per series: max over t >= t0 of min(X_t, X_t+1, X_t+2)."""
    n = Xs.shape[1]
    if n - t0 < 3:
        return np.full(Xs.shape[0], -np.inf)
    return np.minimum(np.minimum(Xs[:, t0:n - 2], Xs[:, t0 + 1:n - 1]), Xs[:, t0 + 2:n]).max(1)


def moving_avg5(H: np.ndarray) -> np.ndarray:
    c = np.cumsum(H, axis=1)
    ma = np.empty_like(H)
    ma[:, :5] = c[:, :5] / np.arange(1, 6)
    ma[:, 5:] = (c[:, 5:] - c[:, :-5]) / 5
    return ma


def ewma_std(H, mu, sd, lam, t0):
    n = H.shape[1]
    out = np.full(H.shape, -np.inf)
    z = mu.copy()
    for t in range(t0, n):
        z = lam * H[:, t] + (1 - lam) * z
        out[:, t] = z
    return (out - mu[:, None]) / (sd[:, None] * np.sqrt(lam / (2 - lam)))


def cusum_std(H, mu, sd, kappa, t0):
    n = H.shape[1]
    out = np.full(H.shape, -np.inf)
    s = np.zeros(H.shape[0])
    for t in range(t0, n):
        s = np.maximum(0.0, s + (H[:, t] - mu) - kappa * sd)
        out[:, t] = s
    return out / sd[:, None]


def adaptive_std(H, mu, sd, W, t0):
    """J26 adaptive limits: Phase-I limits before W, expanding mean and sd (never below sd) from W on."""
    n = H.shape[1]
    out = (H - mu[:, None]) / sd[:, None]
    c1, c2 = np.cumsum(H, axis=1), np.cumsum(H * H, axis=1)
    t = np.arange(W, n)
    m = c1[:, t - 1] / t
    v = np.maximum(c2[:, t - 1] / t - m * m, 0.0)
    lim_sd = np.maximum(np.sqrt(v), sd[:, None])
    out[:, W:] = (H[:, W:] - m) / lim_sd
    out[:, :t0] = -np.inf
    return out


def ar1_phi_vec(X: np.ndarray) -> np.ndarray:
    """J26.ar1_phi per row: lag-1 autocorrelation of the linearly detrended window, clipped to [-0.99, 0.99]."""
    R_, w = X.shape
    if w < 4:
        return np.zeros(R_)
    t = np.arange(w, dtype=float)
    tc = t - t.mean()
    b = ((X - X.mean(1, keepdims=True)) * tc).sum(1) / (tc * tc).sum()
    a = X.mean(1) - b * t.mean()
    r = X - (a[:, None] + b[:, None] * t)
    s = r.std(1)
    r0, r1 = r[:, :-1], r[:, 1:]
    r0c, r1c = r0 - r0.mean(1, keepdims=True), r1 - r1.mean(1, keepdims=True)
    den = np.sqrt((r0c ** 2).sum(1) * (r1c ** 2).sum(1))
    phi = np.where(den > 0, (r0c * r1c).sum(1) / np.where(den > 0, den, 1), 0.0)
    return np.where(s < 1e-12, 0.0, np.clip(phi, -0.99, 0.99))


def protocol_params(arm: str, n: int) -> tuple[int, int, bool, int]:
    """(W, t0, clean, min_keep) of J60's arm for a series of length n."""
    if arm == "P-retro":
        W = max(5, int(n * 0.2))
        return W, 0, False, 0
    if arm == "P-prospective":
        return 20, 20, True, 15
    raise ValueError(arm)


def critical_values(H: np.ndarray, arm: str) -> dict:
    """Every critical multiplier needed for calibration and for the 67 design FARs, for one batch of series."""
    n = H.shape[1]
    W, t0, clean, min_keep = protocol_params(arm, n)
    mu, sd = phase1_vec(H, W, clean, min_keep)
    out = {}
    Z = (H - mu[:, None]) / sd[:, None]
    Zt = Z.copy(); Zt[:, :t0] = -np.inf
    out["thr"] = min3_max(Zt, t0)
    M = (moving_avg5(H) - mu[:, None]) / sd[:, None]
    M[:, :t0] = -np.inf
    out["ma"] = min3_max(M, t0)
    for lam in EWMA_LAMS:
        out[("ewma", lam)] = min3_max(ewma_std(H, mu, sd, lam, t0), t0)
    for k in CUSUM_KS:
        out[("cusum", k)] = min3_max(cusum_std(H, mu, sd, k, t0), t0)
    out["phi"] = ar1_phi_vec(H[:, :W])
    out["adaptive"] = min3_max(adaptive_std(H, mu, sd, W, t0), t0)
    out["ewma"], out["cusum"] = out[("ewma", LAM)], out[("cusum", KAPPA)]
    return out


CHUNK = 1000                            # series simulated at once (memory: 1,000 x 2,538 x 8 bytes per array)


def critical_values_chunked(seed_seq: np.random.SeedSequence, n: int, phi: float, noise: str, arm: str) -> dict:
    """critical_values over R series of length n, simulated in chunks; deterministic given the seed sequence."""
    parts = []
    sizes = [min(CHUNK, R - i) for i in range(0, R, CHUNK)]
    for size, child in zip(sizes, seed_seq.spawn(len(sizes))):
        parts.append(critical_values(simulate(np.random.default_rng(child), size, n, phi, noise), arm))
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def check_vectorised(j60, rng: np.random.Generator) -> int:
    """The critical-multiplier shortcut must give the same firing as J60's scalar rules, series by series."""
    S, n = 200, 300
    H = simulate(rng, S, n, 0.367, "gauss")
    bad = 0
    for arm in PROTOCOLS:
        c = critical_values(H, arm)
        W = protocol_params(arm, n)[0]
        tests = []
        for rule in CAL_RULES:
            m = float(np.median(c[rule][np.isfinite(c[rule])]))
            kind = CAL_SPEC[rule][0]
            p = {"thr": {"k": m}, "ma": {"k": m}, "ewma": {"lam": LAM, "L": m}, "cusum": {"k": KAPPA, "h": m}}[rule]
            tests.append((rule, kind, p, c[rule] > m))
        m = float(np.median(c[("cusum", 0.25)]))
        tests.append(("cusum k=0.25", "cusum", {"k": 0.25, "h": m}, c[("cusum", 0.25)] > m))
        infl = np.sqrt(np.maximum((1 + c["phi"] * (1 - LAM)) / (1 - c["phi"] * (1 - LAM)), 1e-9))
        tests.append(("ewma_ar L=3", "ewma_ar", {"lam": LAM, "L": 3.0}, c[("ewma", LAM)] > 3.0 * infl))
        m = float(np.median(c["adaptive"]))
        tests.append(("adaptive", "adaptive", {"L": m}, c["adaptive"] > m))
        for name, kind, p, vec in tests:
            sca = np.array([j60.onset({"rms": H[s], "kurt": H[s]}, kind, p, "rms", arm)["i_star"] is not None
                            for s in range(S)])
            mis = int((sca != vec).sum())
            bad += mis
            log(f"  {arm:13s} {name:13s} fired {int(sca.sum()):3d}/{S}; vectorised mismatches {mis}")
        phi_s = np.array([j60.ar1_phi(H[s, :W]) for s in range(S)])
        dphi = float(np.abs(phi_s - c["phi"]).max())
        log(f"  {arm:13s} detrended lag-1 phi, vectorised vs J60.ar1_phi: max difference {dphi:.1e}")
        bad += int(dphi > 1e-9)
    return bad


def far(crits: list[np.ndarray], m: float) -> float:
    return float(np.mean([np.mean(c > m) for c in crits]))


def far_se(crits: list[np.ndarray], m: float) -> float:
    p = np.array([np.mean(c > m) for c in crits])
    return float(np.sqrt(np.sum(p * (1 - p) / np.array([len(c) for c in crits]))) / len(crits))


def calibrate(crits: list[np.ndarray], alpha: float) -> tuple[float, float]:
    finite = np.concatenate([c[np.isfinite(c)] for c in crits])
    lo, hi = float(finite.min()) - 1.0, float(finite.max()) + 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if far(crits, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi, far(crits, hi)


def design_far(crits_by_len: list[dict], spec: dict) -> float:
    kind, p = spec["kind"], spec["p"]
    vals = []
    for c in crits_by_len:
        if kind == "threshold":
            vals.append(np.mean(c["thr"] > p["k"]))
        elif kind == "ma":
            vals.append(np.mean(c["ma"] > p["k"]))
        elif kind == "ewma":
            vals.append(np.mean(c[("ewma", p["lam"])] > p["L"]))
        elif kind == "ewma_ar":
            lam = p["lam"]
            infl = np.sqrt(np.maximum((1 + c["phi"] * (1 - lam)) / (1 - c["phi"] * (1 - lam)), 1e-9))
            vals.append(np.mean(c[("ewma", lam)] > p["L"] * infl))
        elif kind == "cusum":
            vals.append(np.mean(c[("cusum", p["k"])] > p["h"]))
        elif kind == "adaptive":
            vals.append(np.mean(c["adaptive"] > p["L"]))
        else:
            raise ValueError(kind)
    return float(np.mean(vals))


# ------------------------------------------------------------------ real-data helpers
def labels_kappa(onsets_a: dict, onsets_b: dict, n_of: dict, trajs: list[str]) -> tuple[float, float, int]:
    """Cohen's kappa and disagreement pooled over the records of the trajectories on which both rules fire (J23)."""
    tp = tn = fp = fn = used = 0
    for t in trajs:
        ia, ib = onsets_a.get(t), onsets_b.get(t)
        if ia is None or ib is None or not (np.isfinite(ia) and np.isfinite(ib)):
            continue
        n = n_of[t]
        ya, yb = np.arange(n) >= ia, np.arange(n) >= ib
        tp += int((ya & yb).sum()); tn += int((~ya & ~yb).sum())
        fp += int((ya & ~yb).sum()); fn += int((~ya & yb).sum())
        used += 1
    tot = tp + tn + fp + fn
    if not used:
        return np.nan, np.nan, 0
    po = (tp + tn) / tot
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / tot ** 2
    return ((po - pe) / (1 - pe) if pe < 1 else np.nan), 1 - po, used


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j61_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J61  RULES CALIBRATED TO A COMMON FALSE-ALARM RATE  (WP-A)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}   R = {R:,}   B = {B:,}   seed = {SEED}")
    if R < 4000 or B < 20000:
        log("NOTE: quick run (R < 4,000 or B < 20,000); the manuscript numbers need the defaults")

    j60, j23, j11, j26 = (load("j60", "J60_baseline_protocols.py"), load("j23", "J23_reviewer_analyses.py"),
                          load("j11", "J11_arl_and_covariates.py"), load("j26", "J26_design_sweep.py"))
    grid = j26.build_grid()
    raw = j11.load_traj()
    data = {ds: {t: {"rms": g.rms.to_numpy(float), "kurt": g.hikurt.to_numpy(float)}
                 for t, g in raw[ds].groupby("trajectory", sort=False)} for ds in INTERVAL_DS}
    ac = pd.read_csv(T / "j22_baseline_autocorrelation.csv")

    # ---------------------------------------------------------------- consistency: the submitted 68.9%
    log(f"\n{BAR}\nA  Consistency: chart-versus-RMS-detector label disagreement of the submitted manuscript\n{BAR}")
    la = pd.read_csv(T / "j23_label_agreement.csv")
    q = la[(la.dataset == "xjtu") & (((la.rule_a.isin(CHARTS)) & (la.rule_b.isin(RMS3))) |
                                     ((la.rule_b.isin(CHARTS)) & (la.rule_a.isin(RMS3))))]
    sub_dis = float(q.disagreement.median())
    trajs_all = list(data["xjtu"])
    n_of = {t: len(data["xjtu"][t]["rms"]) for t in trajs_all}
    ons = {}
    for r in RMS3 + CHARTS:
        kind, p, ind = j60.RULE_SPEC[r]
        ons[r] = {t: (lambda o: np.nan if o["i_star"] is None else float(o["i_star"]))(
            j60.onset(data["xjtu"][t], kind, p, ind, "P-retro")) for t in trajs_all}
    mine = np.median([labels_kappa(ons[c], ons[d], n_of, trajs_all)[1] for c in CHARTS for d in RMS3])
    log(f"  median disagreement, chart vs RMS detector, XJTU-SY, P-retro: J23 {sub_dis:.4f}, recomputed {mine:.4f}")
    if not abs(mine - sub_dis) < 1e-9:
        log("  FAIL: the label comparison does not reproduce J23; nothing written")
        return 1
    log("  the critical-multiplier shortcut against J60's scalar rules (200 simulated series, n = 300):")
    if check_vectorised(j60, np.random.default_rng(np.random.SeedSequence([SEED, 7]))):
        log("  FAIL: the vectorised implementation disagrees with J60; nothing written")
        return 1
    log("  PASS")

    # ---------------------------------------------------------------- simulate, calibrate, design FAR
    log(f"\n{BAR}\nB  Calibration on simulated degradation-free series\n{BAR}")
    noise_arms = {}
    for ds in INTERVAL_DS:
        v = ac[ac.dataset == ds].phi_detrended
        noise_arms[ds] = {"main": ("gauss", float(v.median())), "t3": ("t3", float(v.median())),
                          "phi_q25": ("gauss", float(v.quantile(0.25))), "phi_q75": ("gauss", float(v.quantile(0.75)))}
    cal_rows, dfar_rows, cal = [], [], {}
    for ds in INTERVAL_DS:
        lengths_all = [len(data[ds][t]["rms"]) for t in data[ds]]
        for arm in PROTOCOLS:
            lens = [n for n in lengths_all if j60.eligible(n, arm)]
            for na, (noise, phi) in noise_arms[ds].items():
                ss = np.random.SeedSequence([SEED, INTERVAL_DS.index(ds), PROTOCOLS.index(arm),
                                             list(noise_arms[ds]).index(na)])
                crits = [critical_values_chunked(child, n, phi, noise, arm)
                         for n, child in zip(lens, ss.spawn(len(lens)))]
                for rule in CAL_RULES:
                    cl = [c[rule] for c in crits]
                    canon = {"thr": 3.0, "ma": 3.0, "ewma": 3.0, "cusum": 5.0}[rule]
                    for lev in LEVELS:
                        m, achieved = calibrate(cl, lev)
                        se = far_se(cl, m)
                        cal[(ds, arm, na, lev, rule)] = m
                        cal_rows.append({"dataset": ds, "protocol": arm, "noise_arm": na, "noise": noise, "phi": phi,
                                         "level": lev, "rule": rule, "parameter": CAL_SPEC[rule][1],
                                         "multiplier": m, "achieved_far": achieved, "mc_se": se,
                                         "within_tolerance": abs(achieved - lev) <= TOL_FAR,
                                         "n_lengths": len(lens), "R": R,
                                         "canonical_multiplier": canon, "far_at_canonical": far(cl, canon)})
                for spec in grid:
                    dfar_rows.append({"dataset": ds, "protocol": arm, "noise_arm": na, "design": spec["design"],
                                      "kind": spec["kind"], "family": spec["family"], "canonical": spec["canonical"],
                                      "far": design_far(crits, spec)})
                log(f"  {DS_NAME[ds]:7s} {arm:13s} {na:7s} (phi {phi:.3f}, {len(lens)} lengths) done")
    calib, dfar = pd.DataFrame(cal_rows), pd.DataFrame(dfar_rows)
    bad = calib[~calib.within_tolerance]
    log(f"\n  calibrations within +/-{TOL_FAR:.3f} of the target: {len(calib) - len(bad)} of {len(calib)}")
    for _, r in bad.iterrows():
        log(f"    outside tolerance: {r.dataset} {r.protocol} {r.noise_arm} {r.rule} level {r.level}: achieved {r.achieved_far:.4f}")
    log("\n  main null model, multipliers (FAR at the canonical multiplier in brackets):")
    for ds in INTERVAL_DS:
        for arm in PROTOCOLS:
            c = calib[(calib.dataset == ds) & (calib.protocol == arm) & (calib.noise_arm == "main")]
            line = f"    {DS_NAME[ds]:7s} {arm:13s}"
            for rule in CAL_RULES:
                cr = c[c.rule == rule]
                line += (f"  {rule}: " + "/".join(f"{cr[cr.level == lv].multiplier.iloc[0]:.2f}" for lv in LEVELS)
                         + f" [{cr.far_at_canonical.iloc[0]:.3f}]")
            log(line)
    log("    (multiplier at FAR 5% / 1% / 10%; canonical: thr k=3, ma k=3, ewma L=3, cusum H=5)")

    # ---------------------------------------------------------------- ARL0 (secondary)
    log(f"\n{BAR}\nC  ARL0 on {ARL_LEN:,}-record series (secondary; amendment A1)\n{BAR}")
    arl_rows = []
    for ds in INTERVAL_DS:
        noise, phi = noise_arms[ds]["main"]
        nb_med = int(np.median([max(5, int(len(data[ds][t]["rms"]) * 0.2)) for t in data[ds]]))
        rng = np.random.default_rng(np.random.SeedSequence([SEED, 99, INTERVAL_DS.index(ds)]))
        H = simulate(rng, R_ARL, ARL_LEN, phi, noise)
        for arm in PROTOCOLS:
            if arm == "P-retro":
                W, t0, clean, min_keep = nb_med, 0, False, 0
            else:
                W, t0, clean, min_keep = 20, 20, True, 15
            mu, sd = phase1_vec(H, W, clean, min_keep)
            stats = {"thr": (H - mu[:, None]) / sd[:, None],
                     "ma": (moving_avg5(H) - mu[:, None]) / sd[:, None],
                     "ewma": ewma_std(H, mu, sd, LAM, t0), "cusum": cusum_std(H, mu, sd, KAPPA, t0)}
            for rule in CAL_RULES:
                X = stats[rule]
                X[:, :t0] = -np.inf
                a = np.minimum(np.minimum(X[:, t0:-2], X[:, t0 + 1:-1]), X[:, t0 + 2:])
                for lev in LEVELS:
                    m = cal[(ds, arm, "main", lev, rule)]
                    hit = a > m
                    anyh = hit.any(1)
                    rl = np.where(anyh, hit.argmax(1) + 1, ARL_LEN - t0)
                    cens = float(1 - anyh.mean())
                    arl_rows.append({"dataset": ds, "protocol": arm, "level": lev, "rule": rule, "multiplier": m,
                                     "phase1_window": W, "t0": t0, "arl0_censored_mean": float(rl.mean()),
                                     "median_run_length": float(np.median(rl)), "share_censored": cens,
                                     "reported": ("> 10,000" if cens > 0.5 else f"{rl.mean():.0f}")})
        log(f"  {DS_NAME[ds]}: P-retro window for ARL0 = {nb_med} records (median n_b), t0 = 0; P-prospective 20, t0 = 20")
    arl = pd.DataFrame(arl_rows)
    for ds in INTERVAL_DS:
        for arm in PROTOCOLS:
            a = arl[(arl.dataset == ds) & (arl.protocol == arm) & (arl.level == 0.05)]
            log(f"    {DS_NAME[ds]:7s} {arm:13s} ARL0 at FAR 5%: " +
                ", ".join(f"{r.rule} {r.reported}" for _, r in a.iterrows()))

    # ---------------------------------------------------------------- real trajectories at the calibrated multipliers
    log(f"\n{BAR}\nD  Real trajectories at the calibrated multipliers (main bearing set, n >= {4 * MAIN_N0})\n{BAR}")
    rng = np.random.default_rng(SEED)
    on_rows, sum_rows, gap_rows = [], [], []
    for ds in INTERVAL_DS:
        trajs = [t for t in data[ds] if len(data[ds][t]["rms"]) >= 4 * MAIN_N0]
        n_of = {t: len(data[ds][t]["rms"]) for t in trajs}
        cond = np.array([j23.condition_of(ds, t) for t in trajs])
        nn = np.array([n_of[t] for t in trajs], float)
        idx = j60.strat_index(cond, rng)                     # shared by every protocol, level and rule
        for arm in PROTOCOLS:
            # uncalibrated reference on the same bearings and resamples
            unc = {}
            for r in RMS3 + CHARTS:
                kind, p, ind = j60.RULE_SPEC[r]
                unc[r] = {t: (lambda o: np.nan if o["i_star"] is None else float(o["i_star"]))(
                    j60.onset(data[ds][t], kind, p, ind, arm)) for t in trajs}
            cfgs = [("main", lev) for lev in LEVELS] + [(na, 0.05) for na in ("t3", "phi_q25", "phi_q75")]
            un = None
            for na, lev in cfgs:
                ons, dsh = {}, {}
                for rule in CAL_RULES:
                    m = cal[(ds, arm, na, lev, rule)]
                    kind = CAL_SPEC[rule][0]
                    p = {"thr": {"k": m}, "ma": {"k": m}, "ewma": {"lam": LAM, "L": m}, "cusum": {"k": KAPPA, "h": m}}[rule]
                    ons[rule], dsh[rule] = {}, {}
                    for t in trajs:
                        o = j60.onset(data[ds][t], kind, p, "rms", arm)
                        i = np.nan if o["i_star"] is None else float(o["i_star"])
                        ons[rule][t], dsh[rule][t] = i, o["shift_d"] if np.isfinite(i) else np.nan
                        on_rows.append({"protocol": arm, "noise_arm": na, "level": lev, "dataset": ds, "trajectory": t,
                                        "condition": j23.condition_of(ds, t), "n_records": n_of[t], "rule": rule,
                                        "multiplier": m, "fired": bool(np.isfinite(i)), "i_star": i,
                                        "u": (i + 1) / n_of[t] if np.isfinite(i) else np.nan,
                                        "shift_d": dsh[rule][t]})
                # slopes with paired bootstrap, calibrated and uncalibrated
                def boot_slopes(onsets: dict) -> tuple[float, np.ndarray]:
                    ii = np.array([onsets[t] for t in trajs], float)
                    f = np.isfinite(ii)
                    s0, _ = j60.slope_rho(ii[f], nn[f], cond[f], False)
                    sb = np.empty(B)
                    for b in range(B):
                        sel = idx[b]
                        k = f[sel]
                        sb[b], _ = j60.slope_rho(ii[sel][k], nn[sel][k], cond[sel][k], False)
                    return s0, sb
                sl = {r: boot_slopes(ons[r]) for r in CAL_RULES}
                for rule in CAL_RULES:
                    ii = np.array([ons[rule][t] for t in trajs], float)
                    f = np.isfinite(ii)
                    pos = nn[f] - ii[f]
                    lo, hi, vs = j60.pct_ci(sl[rule][1])
                    sum_rows.append({"protocol": arm, "noise_arm": na, "level": lev, "dataset": ds, "rule": rule,
                                     "multiplier": cal[(ds, arm, na, lev, rule)], "n_bearings": len(trajs),
                                     "n_fired": int(f.sum()), "coverage": float(f.mean()),
                                     "median_u": float(np.median((ii[f] + 1) / nn[f])) if f.any() else np.nan,
                                     "slope_b1_minus_1": sl[rule][0], "slope_lo": lo, "slope_hi": hi,
                                     "median_d": float(np.nanmedian([dsh[rule][t] for t in trajs])) if f.any() else np.nan,
                                     "pos_rate_record": float(pos.sum() / nn[f].sum()) if f.any() else np.nan,
                                     "pos_rate_bearing": float(np.mean(pos / nn[f])) if f.any() else np.nan})
                det_med = np.median(np.vstack([sl[r][1] for r in DETS_CAL]), axis=0)
                cha_med = np.median(np.vstack([sl[r][1] for r in CHARTS_CAL]), axis=0)
                gap_b = det_med - cha_med
                gap0 = np.median([sl[r][0] for r in DETS_CAL]) - np.median([sl[r][0] for r in CHARTS_CAL])
                if un is None:                               # uncalibrated: once per dataset and protocol
                    un = {r: boot_slopes(unc[r]) for r in RMS3 + CHARTS}
                ugap_b = (np.median(np.vstack([un[r][1] for r in RMS3]), axis=0)
                          - np.median(np.vstack([un[r][1] for r in CHARTS]), axis=0))
                ugap0 = np.median([un[r][0] for r in RMS3]) - np.median([un[r][0] for r in CHARTS])
                g_lo, g_hi, _ = j60.pct_ci(gap_b)
                u_lo, u_hi, _ = j60.pct_ci(ugap_b)
                d_lo, d_hi, _ = j60.pct_ci(det_med)
                c_lo, c_hi, _ = j60.pct_ci(cha_med)
                overlap = bool(np.isfinite(d_lo) and np.isfinite(c_lo) and not (d_hi < c_lo or c_hi < d_lo))
                half = bool(np.isfinite(gap0) and np.isfinite(ugap0) and abs(gap0) < 0.5 * abs(ugap0))
                # shift d: calibrated detectors minus calibrated charts, median over firing (trajectory, rule) pairs
                def dmed(rules, sel=None):
                    tl = trajs if sel is None else [trajs[j] for j in sel]
                    v = [dsh[r][t] for r in rules for t in tl if np.isfinite(dsh[r][t])]
                    return float(np.median(v)) if v else np.nan
                ddiff0 = dmed(DETS_CAL) - dmed(CHARTS_CAL)
                ddiff_b = np.array([dmed(DETS_CAL, idx[b]) - dmed(CHARTS_CAL, idx[b]) for b in range(B)])
                dd_lo, dd_hi, _ = j60.pct_ci(ddiff_b)
                d_differs = bool(np.isfinite(dd_lo) and (dd_lo > 0 or dd_hi < 0))
                kap = [labels_kappa(ons[c], ons[d], n_of, trajs) for c in CHARTS_CAL for d in DETS_CAL]
                ukap = [labels_kappa(unc[c], unc[d], n_of, trajs) for c in CHARTS for d in RMS3]
                dj = []
                for c in CHARTS_CAL:
                    for d in DETS_CAL:
                        for t in trajs:
                            ya, yb = j60.target(n_of[t], ons[c][t]), j60.target(n_of[t], ons[d][t])
                            if ya is not None and yb is not None:
                                dj.append(float(np.sqrt(np.mean((ya - yb) ** 2))))
                reading = ("1: shrinks or disappears" if (overlap or half) else
                           ("2: persists, different d" if d_differs else "3: persists, similar d"))
                gap_rows.append({"protocol": arm, "noise_arm": na, "level": lev, "dataset": ds, "n_bearings": len(trajs),
                                 "gap_calibrated": float(gap0), "gap_cal_lo": g_lo, "gap_cal_hi": g_hi,
                                 "gap_uncalibrated": float(ugap0), "gap_unc_lo": u_lo, "gap_unc_hi": u_hi,
                                 "det_median_slope_lo": d_lo, "det_median_slope_hi": d_hi,
                                 "chart_median_slope_lo": c_lo, "chart_median_slope_hi": c_hi,
                                 "intervals_overlap": overlap, "less_than_half": half,
                                 "d_det_minus_chart": ddiff0, "d_diff_lo": dd_lo, "d_diff_hi": dd_hi, "d_differs": d_differs,
                                 "kappa_calibrated_median": float(np.nanmedian([k[0] for k in kap])),
                                 "disagreement_calibrated_median": float(np.nanmedian([k[1] for k in kap])),
                                 "kappa_uncalibrated_median": float(np.nanmedian([k[0] for k in ukap])),
                                 "disagreement_uncalibrated_median": float(np.nanmedian([k[1] for k in ukap])),
                                 "median_D_j_calibrated": float(np.median(dj)) if dj else np.nan,
                                 "reading": reading})
                log(f"  {DS_NAME[ds]:7s} {arm:13s} {na:7s} FAR {lev:.2f}: gap cal {gap0:+.2f} [{g_lo:+.2f}, {g_hi:+.2f}] vs "
                    f"uncal {ugap0:+.2f} [{u_lo:+.2f}, {u_hi:+.2f}]; d det-chart {ddiff0:+.2f} [{dd_lo:+.2f}, {dd_hi:+.2f}]; "
                    f"disagreement {np.nanmedian([k[1] for k in kap]):.3f} (uncal {np.nanmedian([k[1] for k in ukap]):.3f})"
                    f" -> reading {reading}")
    onsets_df, summary, gaps = pd.DataFrame(on_rows), pd.DataFrame(sum_rows), pd.DataFrame(gap_rows)

    log(f"\n{BAR}\nE  Calibrated rules at FAR 5%, main null model: b1 - 1 [95% interval], median u, median d\n{BAR}")
    for ds in INTERVAL_DS:
        for arm in PROTOCOLS:
            s = summary[(summary.dataset == ds) & (summary.protocol == arm) & (summary.noise_arm == "main") & (summary.level == 0.05)]
            log(f"  {DS_NAME[ds]:7s} {arm:13s} " + "; ".join(
                f"{r.rule} {r.slope_b1_minus_1:+.2f} [{r.slope_lo:+.2f},{r.slope_hi:+.2f}] u {r.median_u:.2f} d {r.median_d:.1f} "
                f"({r.n_fired}/{r.n_bearings})" for _, r in s.iterrows()))

    calib.to_csv(T / "j61_calibration.csv", index=False)
    arl.to_csv(T / "j61_arl0.csv", index=False)
    dfar.to_csv(T / "j61_design_far.csv", index=False)
    onsets_df.to_csv(T / "j61_matched_onsets.csv", index=False)
    summary.to_csv(T / "j61_matched_summary.csv", index=False)
    gaps.to_csv(T / "j61_matched_gap.csv", index=False)
    log(f"\nwrote j61_calibration.csv ({len(calib)}), j61_arl0.csv ({len(arl)}), j61_design_far.csv ({len(dfar)}), "
        f"j61_matched_onsets.csv ({len(onsets_df)}), j61_matched_summary.csv ({len(summary)}), j61_matched_gap.csv ({len(gaps)})")
    log(f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
