"""J60 -- WP-B: onset protocols (Phase I / Phase II), the protocol curve, baseline contamination and warm-up.

Reviewer comments: R1.3 (the 20%-of-life baseline is retrospective; alarms inside it), R2.6 (impact of baseline
alarms), R2.7 (Phase I versus Phase II), R3.2 (degradation inside the healthy baseline).

Every onset of the submitted manuscript was computed under one protocol, called P-retro here: mu and sigma
estimated on the first max(5, floor(0.2 n)) records (population sd), no cleaning, and an alarm search that starts
at record 0 -- inside the very records that define the healthy level, with a window whose length needs the total
life n. This script applies the unchanged rules under the protocols of Section 5 of the analysis plan. Each arm
changes ONE element of P-retro; P-prospective is the primary protocol of WP-A and WP-C.

  arm             Phase-I window       estimator            cleaning    alarm start t0
  P-retro         max(5, 0.2 n)        mean, population sd  none        0       submitted; control
  P-II            as P-retro           mean, population sd  none        n_b     changes the alarm start
  P-robust        as P-retro           median, 1.4826 MAD   none        0       changes the estimator
  P-iter          as P-retro           mean, population sd  iterative   n_b     Phase I as in SPC practice
  P-fixed(N0)     N0 = 10, 15, 20, 30  mean, population sd  none        N0      the protocol curve
  P-prospective   20                   mean, population sd  iterative   20      primary

Iterative cleaning: estimate, flag |h - mu| > 3 sigma, remove, repeat until nothing is flagged or for at most three
iterations; an iteration that would leave fewer than 15 records (P-iter: fewer than half of the window) is not
applied. A fixed window of N0 records is used only on trajectories with n >= 4 N0. EWMA and CUSUM start at record
t0 (initialised at mu and 0 at record t0 - 1); the moving average is causal over the whole series; the alarm index
is the first t >= t0 at which the alarm condition holds for t, t+1, t+2.

Before anything is written, P-retro must reproduce the 330 onsets of fig1_fpt_matrix.csv and the 2,211 design
alarms of j26_design_sweep_per_traj.csv exactly; otherwise the script stops.

Statistics (analysis plan, Section 6): u = (i* + 1)/n; b1 - 1 = slope of log u on log n with condition fixed effects
(computed by within-condition demeaning, which is the same estimator; checked against J23.ols_slope); 95% intervals
over B = 20,000 resamples of whole bearings stratified by condition (seed 60), resampled once per dataset and
bearing set and shared by every arm and rule, so that differences between arms are paired.

Bearing sets: "main" = trajectories eligible at N0 = 20 (XJTU-SY 13, FEMTO 11), used for every arm except
P-fixed(30), which is shown on its own eligible set (12, 9); "own" = every arm on its own eligible set (supplement).

Outputs (results/tables/):
  j60_onsets.csv            arm x trajectory x rule: window, Phase-I estimates, records removed, alarm, u, d
  j60_design_onsets.csv     67 designs x trajectory under P-retro and P-prospective (input of WP-F, J64)
  j60_protocol_curve.csv    arm x dataset x bearing set x rule: b1 - 1 and rho(u, n) with intervals, coverage
  j60_gap.csv               arm x dataset x bearing set: detector-minus-chart gap in b1 - 1; in-baseline shares
  j60_labels.csv            arm x dataset x rule: positive rate (record-pooled, bearing-averaged), coverage, and
                            arm x dataset x pair class: median D_j (main set)
  j60_contamination.csv     P-retro in-baseline onsets with >= 20 records before the alarm, baseline re-estimated
  j60_warmup.csv            per bearing: e10 and the chart alarm indices (the test of H_w is in the log)
results/figures/j60_warmup_first100.png   first 100 records of every bearing with a chart alarm before record 10
results/logs/j60_run.log

Pre-specified in paper/access_resub/ANALYSIS_PLAN_2026-10-24.md, Sections 3, 5, 6 and 7.
Usage: python scripts/J60_baseline_protocols.py          (B = 20,000)
       B=2000 python scripts/J60_baseline_protocols.py   (quick run; not for the manuscript)
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
SEED = 60
B = int(os.environ.get("B", "20000"))
CONFIRM = 3
BASE_FRAC = 0.20
CLEAN_K, CLEAN_MAX_ITER, CLEAN_MIN_FIXED = 3.0, 3, 15
MIN_FIRED = 4
MAIN_N0 = 20
INTERVAL_DS = ("xjtu", "femto")
DS_ORDER = ["xjtu", "femto", "ims", "nsk"]
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO", "ims": "IMS", "nsk": "NSK 6205"}

RULE_SPEC = {   # rule -> (kind, parameters, indicator); identical to J11.fpt
    "3sig_rms": ("threshold", {"k": 3.0}, "rms"),
    "3sig_kurt": ("threshold", {"k": 3.0}, "kurt"),
    "ma2sig": ("ma", {"k": 2.0}, "rms"),
    "ma3sig": ("ma", {"k": 3.0}, "rms"),
    "ma5sig": ("ma", {"k": 5.0}, "rms"),
    "ewma": ("ewma", {"lam": 0.2, "L": 3.0}, "rms"),
    "cusum": ("cusum", {"k": 0.5, "h": 5.0}, "rms"),
    "pct02": ("pct", {"p": 0.02}, None),
    "pct05": ("pct", {"p": 0.05}, None),
    "pct10": ("pct", {"p": 0.10}, None),
}
RULES = list(RULE_SPEC)
SIGNAL = [r for r in RULES if not r.startswith("pct")]
RMS3, CHARTS = ["3sig_rms", "ma3sig", "ma5sig"], ["ewma", "cusum"]

# arm -> (window: "frac" or N0, estimator, cleaning, alarm start: "zero" or "window")
ARMS = {
    "P-retro": ("frac", "classical", False, "zero"),
    "P-II": ("frac", "classical", False, "window"),
    "P-robust": ("frac", "robust", False, "zero"),
    "P-iter": ("frac", "classical", True, "window"),
    "P-fixed(10)": (10, "classical", False, "window"),
    "P-fixed(15)": (15, "classical", False, "window"),
    "P-fixed(20)": (20, "classical", False, "window"),
    "P-fixed(30)": (30, "classical", False, "window"),
    "P-prospective": (20, "classical", True, "window"),
}
CURVE_MAIN = [a for a in ARMS if a != "P-fixed(30)"]      # on the N0 = 20 set; P-fixed(30) on its own set
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


# ------------------------------------------------------------------ protocol primitives
def window_len(n: int, arm: str) -> int:
    w = ARMS[arm][0]
    return max(5, int(n * BASE_FRAC)) if w == "frac" else int(w)


def eligible(n: int, arm: str) -> bool:
    w = ARMS[arm][0]
    return True if w == "frac" else n >= 4 * int(w)


def estimate(x: np.ndarray, how: str) -> tuple[float, float]:
    if how == "robust":
        med = float(np.median(x))
        return med, max(1.4826 * float(np.median(np.abs(x - med))), 1e-12)
    return float(np.mean(x)), max(float(np.std(x)), 1e-12)


def phase1(h: np.ndarray, arm: str) -> tuple[float, float, int, int]:
    """(mu, sigma, window length W, records removed by cleaning) of one indicator series under one arm."""
    w_kind, how, clean, _ = ARMS[arm]
    W = window_len(len(h), arm)
    x = h[:W]
    if not clean:
        mu, sd = estimate(x, how)
        return mu, sd, W, 0
    min_keep = CLEAN_MIN_FIXED if w_kind != "frac" else math.ceil(W / 2)
    keep = np.ones(W, bool)
    for _ in range(CLEAN_MAX_ITER):
        mu, sd = estimate(x[keep], how)
        flag = keep & (np.abs(x - mu) > CLEAN_K * sd)
        if not flag.any():
            break
        new_keep = keep & ~flag
        if new_keep.sum() < min_keep:
            break
        keep = new_keep
    mu, sd = estimate(x[keep], how)
    return mu, sd, W, int(W - keep.sum())


def first_confirmed(flag: np.ndarray, t0: int) -> int | None:
    run = 0
    for i in range(t0, len(flag)):
        run = run + 1 if flag[i] else 0
        if run >= CONFIRM:
            return i - CONFIRM + 1
    return None


def ar1_phi(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of the linearly detrended window (J26.ar1_phi)."""
    if len(x) < 4:
        return 0.0
    t = np.arange(len(x), dtype=float)
    r = x - np.polyval(np.polyfit(t, x, 1), t)
    s = r.std()
    return 0.0 if s < 1e-12 else float(np.clip(np.corrcoef(r[:-1], r[1:])[0, 1], -0.99, 0.99))


def alarm(h: np.ndarray, kind: str, p: dict, mu: float, sd: float, W: int, t0: int) -> int | None:
    """Alarm index of one rule or design on one series, given the Phase-I estimates and the alarm start."""
    n = len(h)
    if kind == "threshold":
        return first_confirmed(h > mu + p["k"] * sd, t0)
    if kind == "ma":
        ma = pd.Series(h).rolling(5, min_periods=1).mean().to_numpy()
        return first_confirmed(ma > mu + p["k"] * sd, t0)
    if kind in ("ewma", "ewma_ar"):
        lam, L = p["lam"], p["L"]
        if kind == "ewma_ar":
            phi = ar1_phi(h[:W])
            L = L * np.sqrt(max((1 + phi * (1 - lam)) / (1 - phi * (1 - lam)), 1e-9))
        z, out = mu, np.full(n, -np.inf)
        for i in range(t0, n):
            z = lam * h[i] + (1 - lam) * z
            out[i] = z
        return first_confirmed(out > mu + L * sd * np.sqrt(lam / (2 - lam)), t0)
    if kind == "cusum":
        k, H = p["k"] * sd, p["h"] * sd
        s, out = 0.0, np.full(n, -np.inf)
        for i in range(t0, n):
            s = max(0.0, s + (h[i] - mu) - k)
            out[i] = s
        return first_confirmed(out > H, t0)
    if kind == "adaptive":
        L = p["L"]
        run, lim_mu, lim_sd = 0, mu, sd
        for i in range(n):
            if i >= W:
                w = h[:i]
                lim_mu = float(w.mean())
                lim_sd = max(float(w.std()), sd)
            if i < t0:
                continue
            run = run + 1 if h[i] > lim_mu + L * lim_sd else 0
            if run >= CONFIRM:
                return i - CONFIRM + 1
        return None
    raise ValueError(kind)


def onset(series: dict, kind: str, p: dict, ind: str | None, arm: str) -> dict:
    """Everything recorded for one rule or design on one trajectory under one arm."""
    h_rms = series["rms"]
    n = len(h_rms)
    if kind == "pct":
        return {"W": np.nan, "t0": np.nan, "mu": np.nan, "sd": np.nan, "removed": 0,
                "i_star": int(n * (1 - p["p"])), "shift_d": np.nan}
    h = series["kurt"] if ind == "kurt" else h_rms
    mu, sd, W, removed = phase1(h, arm)
    t0 = 0 if ARMS[arm][3] == "zero" else W
    i = alarm(h, kind, p, mu, sd, W, t0)
    d = np.nan
    if i is not None and ind == "rms":
        d = (float(h_rms[i:i + CONFIRM].mean()) - mu) / sd
    return {"W": W, "t0": t0, "mu": mu, "sd": sd, "removed": removed, "i_star": i, "shift_d": d}


# ------------------------------------------------------------------ statistics
def fe_slope(y: np.ndarray, x: np.ndarray, c: np.ndarray) -> float:
    """OLS slope of y on x with fixed effects of c (within-condition demeaning; equals J23.ols_slope)."""
    xm, ym = np.empty_like(x), np.empty_like(y)
    for k in np.unique(c):
        m = c == k
        xm[m], ym[m] = x[m] - x[m].mean(), y[m] - y[m].mean()
    den = float((xm * xm).sum())
    return float((xm * ym).sum() / den) if den > 1e-12 else np.nan


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < MIN_FIRED:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def slope_rho(i_star: np.ndarray, n: np.ndarray, c: np.ndarray, with_rho: bool = True) -> tuple[float, float]:
    if len(n) < MIN_FIRED or np.ptp(n) == 0:
        return np.nan, np.nan
    u = (i_star + 1) / n
    return fe_slope(np.log(u), np.log(n), c), (spearman(u, n) if with_rho else np.nan)


def pct_ci(v) -> tuple[float, float, float]:
    a = np.asarray(v, float)
    ok = a[np.isfinite(a)]
    if len(ok) < len(a) / 2:
        return np.nan, np.nan, len(ok) / max(len(a), 1)
    lo, hi = np.percentile(ok, [2.5, 97.5])
    return float(lo), float(hi), len(ok) / len(a)


def strat_index(c: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parts = []
    for k in sorted(set(c)):
        ix = np.where(c == k)[0]
        parts.append(ix[rng.integers(0, len(ix), size=(B, len(ix)))])
    return np.concatenate(parts, axis=1)


def target(n: int, i) -> np.ndarray | None:
    """RUL target of the submitted Eq. (3) (J32.target)."""
    if i is None or not np.isfinite(i):
        return None
    t = np.arange(n, dtype=float)
    y = np.ones(n)
    after = t >= int(i)
    y[after] = (n - 1 - t[after]) / max(n - 1 - int(i), 1)
    return y


# ------------------------------------------------------------------ main
def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j60_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J60  ONSET PROTOCOLS, PROTOCOL CURVE, CONTAMINATION, WARM-UP  (WP-B)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}   B = {B:,}   seed = {SEED}")
    if B < 20000:
        log("NOTE: B below 20,000 -- a quick run; the manuscript numbers need the default")

    j11, j23, j26 = load("j11", "J11_arl_and_covariates.py"), load("j23", "J23_reviewer_analyses.py"), \
        load("j26", "J26_design_sweep.py")
    raw = j11.load_traj()
    data = {}
    for ds in DS_ORDER:
        d = raw[ds]
        data[ds] = {t: {"rms": g.rms.to_numpy(float), "kurt": g.hikurt.to_numpy(float)}
                    for t, g in d.groupby("trajectory", sort=False)}
    log("\ntrajectories: " + ", ".join(f"{DS_NAME[ds]} {len(data[ds])}" for ds in DS_ORDER))

    # ---------------------------------------------------------------- onsets of the ten rules, every arm
    rows = []
    for arm in ARMS:
        for ds in DS_ORDER:
            for t, s in data[ds].items():
                n = len(s["rms"])
                ok = eligible(n, arm)
                for rule in RULES:
                    kind, p, ind = RULE_SPEC[rule]
                    rec = {"arm": arm, "dataset": ds, "trajectory": t, "condition": j23.condition_of(ds, t),
                           "n_records": n, "eligible": ok, "rule": rule}
                    if ok:
                        o = onset(s, kind, p, ind, arm)
                    else:
                        o = {"W": np.nan, "t0": np.nan, "mu": np.nan, "sd": np.nan, "removed": 0,
                             "i_star": None, "shift_d": np.nan}
                    rec.update(o)
                    rec["fired"] = ok and o["i_star"] is not None
                    rec["i_star"] = float(o["i_star"]) if rec["fired"] else np.nan
                    rec["u"] = (rec["i_star"] + 1) / n if rec["fired"] else np.nan
                    rec["inside_baseline"] = (bool(rec["fired"] and ARMS[arm][3] == "zero" and not rule.startswith("pct")
                                                   and rec["i_star"] < o["W"]))
                    rows.append(rec)
    on = pd.DataFrame(rows)

    # ---------------------------------------------------------------- reproduction checks
    log(f"\n{BAR}\nA  P-retro must reproduce the submitted onsets exactly\n{BAR}")
    ft = pd.read_csv(T / "fig1_fpt_matrix.csv")
    ft["fired"] = ft.fired.astype(bool)
    pr = on[on.arm == "P-retro"].merge(ft[["dataset", "trajectory", "rule", "fired", "fpt_index"]],
                                       on=["dataset", "trajectory", "rule"], suffixes=("", "_sub"))
    same = (pr.fired == pr.fired_sub) & ((~pr.fired) | (pr.i_star == pr.fpt_index))
    log(f"  ten rules: {int(same.sum())} of {len(ft)} onsets identical (merged {len(pr)})")
    if len(pr) != 330 or not same.all():
        log("  FAIL: P-retro does not reproduce fig1_fpt_matrix.csv; nothing written")
        log(pr[~same][["dataset", "trajectory", "rule", "fired", "i_star", "fired_sub", "fpt_index"]].to_string())
        return 1

    grid = j26.build_grid()
    drows = []
    for arm in ("P-retro", "P-prospective"):
        for ds in DS_ORDER:
            for t, s in data[ds].items():
                n = len(s["rms"])
                ok = eligible(n, arm)
                for spec in grid:
                    rec = {"arm": arm, "dataset": ds, "trajectory": t, "condition": j23.condition_of(ds, t),
                           "n_records": n, "eligible": ok, "design": spec["design"], "kind": spec["kind"],
                           "family": spec["family"], "canonical": spec["canonical"]}
                    o = onset(s, spec["kind"], spec["p"], "rms", arm) if ok else {"i_star": None, "shift_d": np.nan,
                                                                                 "W": np.nan, "t0": np.nan}
                    rec.update(fired=ok and o["i_star"] is not None, W=o["W"], t0=o["t0"])
                    rec["alarm_index"] = float(o["i_star"]) if rec["fired"] else np.nan
                    rec["shift_sd"] = o["shift_d"] if rec["fired"] else np.nan
                    drows.append(rec)
    des = pd.DataFrame(drows)
    j26t = pd.read_csv(T / "j26_design_sweep_per_traj.csv")
    j26t["fired"] = j26t.fired.astype(bool)
    dm = des[des.arm == "P-retro"].merge(j26t[["dataset", "design", "trajectory", "fired", "alarm_index", "shift_sd"]],
                                         on=["dataset", "design", "trajectory"], suffixes=("", "_sub"))
    same_d = (dm.fired == dm.fired_sub) & ((~dm.fired) | ((dm.alarm_index == dm.alarm_index_sub) &
                                                           ((dm.shift_sd - dm.shift_sd_sub).abs() < 1e-9)))
    log(f"  67 designs: {int(same_d.sum())} of {len(j26t)} design alarms identical, shift to 1e-9 (merged {len(dm)})")
    if len(dm) != len(j26t) or not same_d.all():
        log("  FAIL: P-retro does not reproduce j26_design_sweep_per_traj.csv; nothing written")
        log(dm[~same_d].head(20).to_string())
        return 1
    # the fast fixed-effects slope must equal J23.ols_slope
    g = pr[(pr.dataset == "xjtu") & (pr.rule == "3sig_rms")]
    y, x, c = np.log(g.u.to_numpy()), np.log(g.n_records.to_numpy(float)), g.condition.to_numpy()
    dev = abs(fe_slope(y, x, c) - j23.ols_slope(y, x, c))
    log(f"  within-condition slope against J23.ols_slope: difference {dev:.1e}")
    if not dev < 1e-9:
        log("  FAIL: slope estimator differs from J23")
        return 1
    log("  PASS")

    # ---------------------------------------------------------------- Phase-I estimates against P-retro
    log(f"\n{BAR}\nB  Phase-I estimates of RMS under each arm, relative to P-retro (median over bearings)\n{BAR}")
    rms_rows = on[on.rule == "3sig_rms"][["arm", "dataset", "trajectory", "eligible", "W", "mu", "sd", "removed"]]
    base = rms_rows[rms_rows.arm == "P-retro"].set_index(["dataset", "trajectory"])[["mu", "sd"]]
    rel = rms_rows.join(base, on=["dataset", "trajectory"], rsuffix="_retro")
    rel["mu_shift_in_retro_sd"] = (rel.mu - rel.mu_retro) / rel.sd_retro
    rel["sd_ratio"] = rel.sd / rel.sd_retro
    for ds in INTERVAL_DS:
        log(f"  {DS_NAME[ds]}")
        for arm in ARMS:
            r = rel[(rel.arm == arm) & (rel.dataset == ds) & rel.eligible]
            log(f"    {arm:14s} bearings {len(r):2d}  sd ratio {r.sd_ratio.median():5.2f} "
                f"[{r.sd_ratio.min():4.2f}-{r.sd_ratio.max():4.2f}]  mu shift {r.mu_shift_in_retro_sd.median():+5.2f} sd"
                f"  records removed (total) {int(r.removed.sum())}")

    # ---------------------------------------------------------------- protocol curve and gap
    log(f"\n{BAR}\nC  Protocol curve: b1 - 1 of each rule under each arm (paired bearing bootstrap)\n{BAR}")
    rng = np.random.default_rng(SEED)
    curve, gaps = [], []
    for ds in INTERVAL_DS:
        trajs = list(data[ds])
        n_of = {t: len(data[ds][t]["rms"]) for t in trajs}
        cond_of = {t: j23.condition_of(ds, t) for t in trajs}
        sets = {"main": [t for t in trajs if n_of[t] >= 4 * MAIN_N0]}
        for arm in ARMS:
            sets[f"own:{arm}"] = [t for t in trajs if eligible(n_of[t], arm)]
        # which arms are evaluated on which set
        plan = [("main", a) for a in CURVE_MAIN] + [(f"own:{a}", a) for a in ARMS]
        resamples = {}
        for set_name in dict.fromkeys(s for s, _ in plan):
            tl = sets[set_name]
            resamples[set_name] = strat_index(np.array([cond_of[t] for t in tl]), rng)
        for set_name, arm in plan:
            tl = sets[set_name]
            if not tl:
                continue
            nn = np.array([n_of[t] for t in tl], float)
            cc = np.array([cond_of[t] for t in tl])
            idx = resamples[set_name]
            with_rho = set_name == "main" or arm == "P-fixed(30)"
            sub = on[(on.arm == arm) & (on.dataset == ds)].set_index(["trajectory", "rule"])
            slopes_b = {}
            for rule in SIGNAL:
                ii = np.array([sub.loc[(t, rule), "i_star"] for t in tl], float)
                fired = np.isfinite(ii)
                s0, r0 = slope_rho(ii[fired], nn[fired], cc[fired], with_rho)
                sb, rb = np.empty(B), np.empty(B)
                for b in range(B):
                    sel = idx[b]
                    k = fired[sel]
                    sb[b], rb[b] = slope_rho(ii[sel][k], nn[sel][k], cc[sel][k], with_rho)
                slopes_b[rule] = sb
                lo, hi, vs = pct_ci(sb)
                rlo, rhi, _ = pct_ci(rb) if with_rho else (np.nan, np.nan, np.nan)
                curve.append({"dataset": ds, "bearing_set": set_name, "arm": arm, "rule": rule,
                              "n_bearings": len(tl), "n_fired": int(fired.sum()), "coverage": float(fired.mean()),
                              "slope_b1_minus_1": s0, "slope_lo": lo, "slope_hi": hi, "slope_valid_share": vs,
                              "rho_u_n": r0, "rho_lo": rlo, "rho_hi": rhi})
            # detector-minus-chart gap (median over RMS3 minus median over charts, as J23)
            pt = {r: [c for c in curve if c["dataset"] == ds and c["bearing_set"] == set_name and c["arm"] == arm
                      and c["rule"] == r][-1]["slope_b1_minus_1"] for r in RMS3 + CHARTS}
            gap0 = np.median([pt[r] for r in RMS3]) - np.median([pt[r] for r in CHARTS])
            gb = (np.median(np.vstack([slopes_b[r] for r in RMS3]), axis=0)
                  - np.median(np.vstack([slopes_b[r] for r in CHARTS]), axis=0))
            glo, ghi, gvs = pct_ci(gb)
            ins = {}
            if ARMS[arm][3] == "zero":
                for fam, rl in (("charts", CHARTS), ("rms_detectors", RMS3)):
                    vals = [bool(sub.loc[(t, r), "inside_baseline"]) for t in tl for r in rl
                            if np.isfinite(sub.loc[(t, r), "i_star"])]
                    ins[fam] = float(np.mean(vals)) if vals else np.nan
            gaps.append({"dataset": ds, "bearing_set": set_name, "arm": arm, "n_bearings": len(tl),
                         "gap_b1_minus_1": float(gap0), "gap_lo": glo, "gap_hi": ghi, "gap_valid_share": gvs,
                         "inside_share_charts": ins.get("charts", np.nan),
                         "inside_share_rms_detectors": ins.get("rms_detectors", np.nan)})
    curve, gaps = pd.DataFrame(curve), pd.DataFrame(gaps)

    for ds in INTERVAL_DS:
        log(f"\n  {DS_NAME[ds]}, main set ({len(sets_main := [t for t in data[ds] if len(data[ds][t]['rms']) >= 4 * MAIN_N0])} bearings)"
            f" except P-fixed(30) on its own set")
        log(f"    {'arm':14s}" + "".join(f"{r:>19s}" for r in ["3sig_rms", "ma5sig", "ewma", "cusum"])
            + f"{'gap (det - chart)':>26s}")
        for arm in ARMS:
            set_name = "main" if arm != "P-fixed(30)" else "own:P-fixed(30)"
            line = f"    {arm:14s}"
            for r in ["3sig_rms", "ma5sig", "ewma", "cusum"]:
                c = curve[(curve.dataset == ds) & (curve.bearing_set == set_name) & (curve.arm == arm) & (curve.rule == r)].iloc[0]
                line += f"  {c.slope_b1_minus_1:+5.2f} [{c.slope_lo:+5.2f},{c.slope_hi:+5.2f}]" if np.isfinite(c.slope_b1_minus_1) else f"{'n/a':>19s}"
            gr = gaps[(gaps.dataset == ds) & (gaps.bearing_set == set_name) & (gaps.arm == arm)].iloc[0]
            line += f"   {gr.gap_b1_minus_1:+5.2f} [{gr.gap_lo:+5.2f}, {gr.gap_hi:+5.2f}]"
            log(line)

    # ---------------------------------------------------------------- reading of the protocol curve (pre-specified)
    log(f"\n{BAR}\nD  Pre-specified reading of the protocol curve (Section 7 of the analysis plan)\n{BAR}")
    for ds in INTERVAL_DS:
        gm = gaps[(gaps.dataset == ds) & (gaps.bearing_set == "main")]
        gm = pd.concat([gm, gaps[(gaps.dataset == ds) & (gaps.bearing_set == "own:P-fixed(30)") & (gaps.arm == "P-fixed(30)")]])
        disjoint = [(a.arm, b.arm) for _, a in gm.iterrows() for _, b in gm.iterrows()
                    if a.arm < b.arm and np.isfinite(a.gap_lo) and np.isfinite(b.gap_lo)
                    and (a.gap_hi < b.gap_lo or b.gap_hi < a.gap_lo)]
        signs = set(np.sign(gm.gap_b1_minus_1.dropna()))
        material = bool(disjoint) or len(signs) > 1
        log(f"  {DS_NAME[ds]}: gap from {gm.gap_b1_minus_1.min():+.2f} ({gm.loc[gm.gap_b1_minus_1.idxmin(), 'arm']}) "
            f"to {gm.gap_b1_minus_1.max():+.2f} ({gm.loc[gm.gap_b1_minus_1.idxmax(), 'arm']}); "
            f"sign changes: {'yes' if len(signs) > 1 else 'no'}; arm pairs with disjoint intervals: {len(disjoint)}")
        for a, b in disjoint[:8]:
            log(f"      disjoint: {a} vs {b}")
        log(f"    -> {'MATERIAL CHANGE across arms: protocol-dependent onset-length relation (first reading)' if material else 'STABLE across arms (second reading)'}")
        for arm in ("P-retro", "P-robust"):
            r = gaps[(gaps.dataset == ds) & (gaps.bearing_set == "main") & (gaps.arm == arm)].iloc[0]
            log(f"    in-baseline share under {arm}: charts {r.inside_share_charts:.2f}, RMS detectors "
                f"{r.inside_share_rms_detectors:.2f}")

    # ---------------------------------------------------------------- labels under each arm
    log(f"\n{BAR}\nE  Labels under each arm: positive rate and target distance D_j (main set)\n{BAR}")
    lab_rows = []
    for ds in INTERVAL_DS:
        tl = [t for t in data[ds] if len(data[ds][t]["rms"]) >= 4 * MAIN_N0]
        for arm in ARMS:
            if arm == "P-fixed(30)":
                continue
            sub = on[(on.arm == arm) & (on.dataset == ds) & on.trajectory.isin(tl)]
            for rule in RULES:
                g2 = sub[(sub.rule == rule) & sub.fired]
                n2 = g2.n_records.to_numpy(float)
                pos = n2 - g2.i_star.to_numpy(float)
                lab_rows.append({"dataset": ds, "arm": arm, "kind": "positive_rate", "rule": rule,
                                 "n_bearings": len(tl), "n_fired": len(g2), "coverage": len(g2) / len(tl),
                                 "pos_rate_record": float(pos.sum() / n2.sum()) if len(g2) else np.nan,
                                 "pos_rate_bearing": float(np.mean(pos / n2)) if len(g2) else np.nan})
            piv = sub.pivot_table(index="trajectory", columns="rule", values="i_star", aggfunc="first")
            for cls, pairs in (("within RMS detectors", [(a, b) for i, a in enumerate(RMS3) for b in RMS3[i + 1:]]),
                               ("chart vs RMS detector", [(a, b) for a in CHARTS for b in RMS3])):
                dj = []
                for t in tl:
                    n = len(data[ds][t]["rms"])
                    for a, b in pairs:
                        ya, yb = target(n, piv.loc[t, a]), target(n, piv.loc[t, b])
                        if ya is not None and yb is not None:
                            dj.append(float(np.sqrt(np.mean((ya - yb) ** 2))))
                lab_rows.append({"dataset": ds, "arm": arm, "kind": "target_distance", "rule": cls,
                                 "n_bearings": len(tl), "n_fired": len(dj), "median_D_j": float(np.median(dj)) if dj else np.nan})
    labels = pd.DataFrame(lab_rows)
    for ds in INTERVAL_DS:
        log(f"  {DS_NAME[ds]}")
        log(f"    {'arm':14s} {'pos rate (rec): 3sig_rms':>25s} {'ewma':>7s} {'cusum':>7s}   "
            f"{'D_j within det':>14s} {'D_j chart-det':>14s}")
        for arm in [a for a in ARMS if a != "P-fixed(30)"]:
            L_ = labels[(labels.dataset == ds) & (labels.arm == arm)]
            pr_ = {r: L_[(L_.kind == "positive_rate") & (L_.rule == r)].pos_rate_record.iloc[0] for r in ("3sig_rms", "ewma", "cusum")}
            dj_ = {c: L_[(L_.kind == "target_distance") & (L_.rule == c)].median_D_j.iloc[0]
                   for c in ("within RMS detectors", "chart vs RMS detector")}
            log(f"    {arm:14s} {pr_['3sig_rms']:25.3f} {pr_['ewma']:7.3f} {pr_['cusum']:7.3f}   "
                f"{dj_['within RMS detectors']:14.3f} {dj_['chart vs RMS detector']:14.3f}")

    # ---------------------------------------------------------------- contamination diagnostic (descriptive)
    log(f"\n{BAR}\nF  Baseline contamination diagnostic (descriptive; P-retro onsets inside the baseline)\n{BAR}")
    con_rows, excluded = [], []
    retro = on[(on.arm == "P-retro") & on.inside_baseline]
    for _, r in retro.iterrows():
        s = data[r.dataset][r.trajectory]
        kind, p, ind = RULE_SPEC[r.rule]
        h = s["kurt"] if ind == "kurt" else s["rms"]
        i0 = int(r.i_star)
        if i0 < 20:
            excluded.append(f"{r.dataset}/{r.trajectory}/{r.rule} (alarm at record {i0})")
            continue
        mu2, sd2 = estimate(h[:i0], "classical")
        i2 = alarm(h, kind, p, mu2, sd2, i0, 0)
        n = len(h)
        y0, y2 = target(n, i0), target(n, i2)
        con_rows.append({"dataset": r.dataset, "trajectory": r.trajectory, "rule": r.rule, "n_records": n,
                         "W_retro": r.W, "i_star_retro": i0, "mu_retro": r.mu, "sd_retro": r.sd,
                         "mu_pre_alarm": mu2, "sd_pre_alarm": sd2, "sd_ratio": sd2 / r.sd,
                         "mu_shift_in_retro_sd": (mu2 - r.mu) / r.sd,
                         "i_star_reestimated": np.nan if i2 is None else i2,
                         "pos_share_retro": (n - i0) / n, "pos_share_reestimated": np.nan if i2 is None else (n - i2) / n,
                         "D_j_retro_vs_reestimated": (float(np.sqrt(np.mean((y0 - y2) ** 2))) if y2 is not None else np.nan)})
    con = pd.DataFrame(con_rows)
    log(f"  in-baseline P-retro onsets: {len(retro)}; with >= 20 records before the alarm: {len(con)}; "
        f"excluded (fewer records): {len(excluded)}")
    for e in excluded:
        log(f"    excluded: {e}")
    if len(con):
        for _, r in con.iterrows():
            i2 = "never" if not np.isfinite(r.i_star_reestimated) else f"{int(r.i_star_reestimated)}"
            log(f"    {r.dataset}/{r.trajectory} {r.rule}: alarm {r.i_star_retro} -> {i2}; sd ratio {r.sd_ratio:.2f}, "
                f"mu shift {r.mu_shift_in_retro_sd:+.2f} sd; D_j {r.D_j_retro_vs_reestimated:.3f}")

    # ---------------------------------------------------------------- warm-up hypothesis (pre-specified)
    log(f"\n{BAR}\nG  Warm-up hypothesis H_w: early chart alarms because the first records lie above the P-retro mean\n{BAR}")
    wrows = []
    for ds in INTERVAL_DS:
        for t, s in data[ds].items():
            h = s["rms"]
            r0 = on[(on.arm == "P-retro") & (on.dataset == ds) & (on.trajectory == t)].set_index("rule")
            mu, sd = r0.loc["3sig_rms", "mu"], r0.loc["3sig_rms", "sd"]
            wrows.append({"dataset": ds, "trajectory": t, "condition": j23.condition_of(ds, t), "n_records": len(h),
                          "e10": (float(h[:10].mean()) - mu) / sd,
                          "i_ewma": r0.loc["ewma", "i_star"], "i_cusum": r0.loc["cusum", "i_star"]})
    wu = pd.DataFrame(wrows)
    cond1, res = {}, []
    for chart in CHARTS:
        per_ds = {}
        for ds in INTERVAL_DS:
            w = wu[wu.dataset == ds]
            ii = w[f"i_{chart}"].to_numpy(float)
            e = w.e10.to_numpy(float)
            c = w.condition.to_numpy()
            f = np.isfinite(ii)
            rho0 = spearman(e[f], ii[f])
            idx = strat_index(c, rng)
            rb = np.array([spearman(e[s][f[s]], ii[s][f[s]]) for s in idx])
            lo, hi, _ = pct_ci(rb)
            per_ds[ds] = (rho0, lo, hi)
            res.append({"chart": chart, "dataset": ds, "rho_e10_istar": rho0, "lo": lo, "hi": hi, "n": int(f.sum())})
            log(f"  {chart:5s} {DS_NAME[ds]:8s} rho(e10, i*) = {rho0:+.3f} [{lo:+.2f}, {hi:+.2f}]  (n = {int(f.sum())})")
        strict = [ds for ds in INTERVAL_DS if per_ds[ds][0] < 0 and np.isfinite(per_ds[ds][2]) and per_ds[ds][2] < 0]
        cond1[chart] = bool(strict) and all(per_ds[ds][0] < 0 for ds in INTERVAL_DS)
    early = []
    for ds in INTERVAL_DS:
        w = wu[wu.dataset == ds]
        for chart in CHARTS:
            m = w[f"i_{chart}"] < 0.05 * w.n_records
            early += list(w.loc[m, "e10"] > 1)
    share = float(np.mean(early)) if early else np.nan
    c1, c2 = all(cond1.values()), (share >= 2 / 3 if early else False)
    log(f"  condition 1 (negative rho, interval excluding 0 on >= 1 dataset, negative on the other): "
        f"ewma {'met' if cond1['ewma'] else 'not met'}, cusum {'met' if cond1['cusum'] else 'not met'}")
    log(f"  condition 2: chart alarms at i* < 0.05 n: {len(early)}; share on bearings with e10 > 1: {share:.2f} "
        f"(needs >= 0.67) -> {'met' if c2 else 'not met'}")
    log(f"  -> H_w {'SUPPORTED' if (c1 and c2) else 'NOT SUPPORTED'} under the pre-specified criteria")

    # first 100 records of every bearing with a chart alarm before record 10
    early_b = wu[(wu.i_ewma < 10) | (wu.i_cusum < 10)]
    if len(early_b):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = len(early_b)
        cols = 3
        fig, axes = plt.subplots(math.ceil(k / cols), cols, figsize=(10, 2.4 * math.ceil(k / cols)), squeeze=False)
        for ax, (_, r) in zip(axes.ravel(), early_b.iterrows()):
            h = data[r.dataset][r.trajectory]["rms"]
            r0 = on[(on.arm == "P-retro") & (on.dataset == r.dataset) & (on.trajectory == r.trajectory) & (on.rule == "3sig_rms")].iloc[0]
            m = min(100, len(h))
            ax.plot(np.arange(m), h[:m], lw=0.9, color="0.2")
            ax.axhline(r0.mu, color="tab:blue", lw=0.8, label="P-retro baseline mean")
            ax.axhline(r0.mu + r0.sd, color="tab:blue", lw=0.6, ls=":")
            for chart, col in (("ewma", "tab:red"), ("cusum", "tab:orange")):
                i = r[f"i_{chart}"]
                if np.isfinite(i) and i < m:
                    ax.axvline(i, color=col, lw=0.9, ls="--", label=f"{chart} alarm")
            ax.set_title(f"{DS_NAME[r.dataset]} {r.trajectory} (n={r.n_records}, e10={r.e10:+.1f})", fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in axes.ravel()[k:]:
            ax.axis("off")
        axes.ravel()[0].legend(fontsize=6, loc="upper right")
        fig.supxlabel("record", fontsize=8)
        fig.supylabel("RMS", fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGS / "j60_warmup_first100.png", dpi=150)
        plt.close(fig)
        log(f"  plotted {k} bearings with a chart alarm before record 10: results/figures/j60_warmup_first100.png")

    # ---------------------------------------------------------------- write
    on.to_csv(T / "j60_onsets.csv", index=False)
    des.to_csv(T / "j60_design_onsets.csv", index=False)
    curve.to_csv(T / "j60_protocol_curve.csv", index=False)
    gaps.to_csv(T / "j60_gap.csv", index=False)
    labels.to_csv(T / "j60_labels.csv", index=False)
    con.to_csv(T / "j60_contamination.csv", index=False)
    wu.to_csv(T / "j60_warmup.csv", index=False)
    pd.DataFrame(res).to_csv(T / "j60_warmup_test.csv", index=False)
    log(f"\nwrote j60_onsets.csv ({len(on)}), j60_design_onsets.csv ({len(des)}), j60_protocol_curve.csv ({len(curve)}), "
        f"j60_gap.csv ({len(gaps)}), j60_labels.csv ({len(labels)}), j60_contamination.csv ({len(con)}), "
        f"j60_warmup.csv ({len(wu)}), j60_warmup_test.csv ({len(res)})")
    log(f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
