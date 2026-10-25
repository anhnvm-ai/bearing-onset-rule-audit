"""J62 -- WP-C: remaining-useful-life evaluation on a common retrospective reference (R1.1, R1.2).

R1.1: a comment of the review reports, answered in the response letter.
R1.2: a comment of the review reports, answered in the response letter.

The submitted comparison scored every model against its OWN labeler's target, so a difference between labelers was a
difference between targets. This script adds two evaluations in which every labeler is scored on the same thing:

  C-rank    models trained on each labeler's own target (submitted Eq. (3)); within each bearing, the rank agreement
            (Spearman, Harrell's concordance) between the prediction and the number of records left until the test
            stopped, n - 1 - t, over the common record set S_j (main) and over all records (secondary)
  C-common  the labeler only chooses the training records (t >= i* of each training bearing); every model learns the
            same target, the common retrospective RUL reference r_t = (n - 1 - t)/(n - 1), and is scored on S_j.
            r_t uses the total life n, so it is valid for offline evaluation only

S_j = {t >= the latest onset of the labelers on bearing j}. With pct05 among the labelers S_j is about the last 5% of
life, so a variant S3_j uses only the three signal-based labelers.

Labelers: 3sig_rms, ma5sig, ewma, pct05 at their canonical settings, under P-retro and P-prospective (J60); the rules
calibrated to 5% FAR by J61 as a sensitivity analysis (amendment A2, item 8). Models: the six regressors of J35, the
two-stage exponential model of J39, GRU and TCN of J47 (seeds 0, 1, 2). Inputs: (a) per-record statistics as J35;
(b) (a) + elapsed record count t; (c) (a) + history features over 16 records for the tree models (amendment A2,
item 5); (d) the same over 64 records, kept at Gate 1 (rule G1-c: WP-C did not give the third reading).
Leave-one-bearing-out within each dataset, on the bearings where all four labelers fire.

Model comparison under the common reference (Section 9 of the analysis plan): for each pair of labelers and pair of
models, the paired difference in C-common RMSE over B = 20,000 bearing resamples stratified by condition (seed 62);
a significant reversal needs both differences significant after Holm correction over the 2 x 36 tests of the labeler
pair. Also: how often the best model differs between two labelers, Kendall tau, and the labeler main effect
(amendment A2, item 2). The pre-specified reading is applied at the end.

Checks before anything is written: P-retro onsets of the four labelers equal the submitted ones; the six regressors
on their own targets reproduce J35's pooled RMSE; the two-stage model rewritten through T_hat reproduces J39; GRU and
TCN are compared with J47 (torch 2.12 there, the installed version here) and the difference is reported.

Pre-specified in paper/access_resub/ANALYSIS_PLAN_2026-10-24.md, Section 9; amendments A2 and A3.

Outputs: results/tables/j62_oof_predictions.parquet   every out-of-fold prediction (one row per record and cell)
         results/tables/j62_common_reference.csv      config x dataset x labeler x model x variant: all metrics
         results/tables/j62_per_bearing.csv           the same per bearing
         results/tables/j62_rankings.csv              config x dataset x set x metric x labeler pair: reversals,
                                                       winner changes, Kendall tau
         results/tables/j62_main_effect.csv           config x dataset x set x metric x labeler pair
         results/cache_j62/                           one file per training job (lets an interrupted run resume)
         results/logs/j62_run.log
Usage  : python scripts/J62_common_reference_rul.py                 (B = 20,000, WORKERS = 14)
         QUICK=1 EPOCHS=1 B=200 python scripts/J62_common_reference_rul.py   (FEMTO, P-retro only; code check)
"""
from __future__ import annotations

# torch must be imported before scikit-learn on this machine (WinError 1114 otherwise; see J59)
import torch  # noqa: F401  (must come first)

import hashlib
import importlib.util
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
SEED = 62
B = int(os.environ.get("B", "20000"))
WORKERS = int(os.environ.get("WORKERS", "14"))
QUICK = os.environ.get("QUICK", "0") == "1"
EPOCHS = int(os.environ.get("EPOCHS", "30"))   # J47 reads the same variable; part of the cache key of GRU/TCN
CACHE = ROOT / "results" / ("cache_j62_quick" if QUICK else "cache_j62")
LABELERS = ["3sig_rms", "ma5sig", "ewma", "pct05"]
SIGNAL_LABS = ["3sig_rms", "ma5sig", "ewma"]
REG = ["lin:rms", "ridge:all", "knn:all", "gbr:all", "et:all", "rf:all"]
TREES = ["gbr:all", "et:all", "rf:all"]
SEQ = ["gru", "tcn"]
NINE = REG + ["two-stage"] + SEQ
SEEDS = (0, 1, 2)
HIST_W = 16
DATASETS = ("femto",) if QUICK else ("xjtu", "femto")
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO"}
ALPHA = 0.05
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


def variants_of(model: str) -> tuple[str, ...]:
    if model in TREES:
        return ("a", "b", "c", "d")          # d: history window W = 64, kept at Gate 1 (rule G1-c)
    if model in REG or model in SEQ:
        return ("a", "b")
    return ("a",)


# ------------------------------------------------------------------ data (identical in main and workers)
_DATA: dict = {}


def prepare(ds: str) -> dict:
    """Features in J35's record order, the within-bearing record index, and the history features."""
    if ds in _DATA:
        return _DATA[ds]
    j35 = load("j35", "J35_downstream_rul.py")
    traj, F = j35.load_dataset(ds)
    traj = np.asarray(traj)
    t_idx = np.zeros(len(traj), int)
    rms = F["rms"].to_numpy(float)

    def history(W: int) -> np.ndarray:
        """Trailing mean and OLS slope of RMS over records max(0, t - W + 1)..t, and cumulative RMS^2."""
        hist = np.zeros((len(traj), 3))
        for b in pd.unique(traj):
            m = np.flatnonzero(traj == b)
            x = rms[m]
            j = np.arange(len(m), dtype=float)
            c1, cj, cjj, cjy = (np.concatenate([[0.0], np.cumsum(v)]) for v in (x, j, j * j, j * x))
            hi = np.arange(1, len(m) + 1)
            lo = np.maximum(0, hi - W)
            w = (hi - lo).astype(float)
            sy, sj, sjj, sjy = c1[hi] - c1[lo], cj[hi] - cj[lo], cjj[hi] - cjj[lo], cjy[hi] - cjy[lo]
            den = w * sjj - sj * sj
            slope = np.where((w > 1) & (den > 0), (w * sjy - sj * sy) / np.where(den > 0, den, 1.0), 0.0)
            hist[m, 0] = sy / w
            hist[m, 1] = slope
            hist[m, 2] = np.cumsum(x * x)
        return hist

    for b in pd.unique(traj):
        m = np.flatnonzero(traj == b)
        t_idx[m] = np.arange(len(m))
    X_a = F.to_numpy(float)
    out = {"traj": traj, "feats": list(F.columns), "t": t_idx, "rms": rms, "X_a": X_a,
           "X_b": np.column_stack([X_a, t_idx.astype(float)]),
           "X_c": np.column_stack([X_a, history(HIST_W)]), "X_d": np.column_stack([X_a, history(64)])}
    _DATA[ds] = out
    return out


def design_matrix(d: dict, keep: np.ndarray, model: str, variant: str) -> np.ndarray:
    if model == "lin:rms":
        X = d["X_a"][:, [d["feats"].index("rms")]]
        if variant == "b":
            X = np.column_stack([X, d["t"].astype(float)])
    else:
        X = d[f"X_{variant}"]
    return X[keep]


# ------------------------------------------------------------------ worker
_W: dict = {}


def _init_worker() -> None:
    import torch as _t
    _W["j47"] = load("j47", "J47_sequence_models.py")
    _W["j35"] = load("j35", "J35_downstream_rul.py")
    _t.set_num_threads(1)


def run_job(job: dict) -> tuple[str, np.ndarray, float]:
    """Leave-one-bearing-out predictions of one (config, dataset, labeler, target, model, variant, seed) cell."""
    t0 = time.time()
    d = prepare(job["ds"])
    usable = job["usable"]
    keep = np.isin(d["traj"], usable)
    traj, tt = d["traj"][keep], d["t"][keep]
    n_rec = int(keep.sum())
    y = np.empty(n_rec)
    train_mask = np.ones(n_rec, bool)
    for b in usable:
        m = traj == b
        n = int(m.sum())
        i = job["onset"][b]
        if job["target"] == "own":
            y[m] = _W["j35"].target(n, i)
        else:
            y[m] = (n - 1 - tt[m]) / (n - 1)
            train_mask[m] = tt[m] >= i
    X = design_matrix(d, keep, job["model"], job["variant"])
    p = np.full(n_rec, np.nan)
    if job["model"] in SEQ:
        j47 = _W["j47"]
        Wall = j47.windows(X.astype(np.float32), traj)
        for fold, b in enumerate(usable):
            te = traj == b
            tr = (~te) & train_mask
            mu = Wall[tr].reshape(-1, X.shape[1]).mean(0)
            sd = Wall[tr].reshape(-1, X.shape[1]).std(0)
            sd[sd == 0] = 1.0
            p[te] = j47.fit_predict(job["model"], (Wall[tr] - mu) / sd, y[tr], (Wall[te] - mu) / sd,
                                    job["seed"] * 1000 + fold)
    else:
        for b in usable:
            te = traj == b
            tr = (~te) & train_mask
            est = _W["j35"].models(X.shape[1])[job["model"]]
            if job["model"] in ("et:all", "rf:all"):
                est.set_params(n_jobs=1)
            est.fit(X[tr], y[tr])
            p[te] = est.predict(X[te])
        p = np.clip(p, 0.0, 1.0)
    if np.isnan(p).any():
        raise RuntimeError(f"{job['key']}: a record was never held out")
    return job["key"], p.astype(np.float64), time.time() - t0


def job_key(job: dict) -> str:
    h = hashlib.sha1(json.dumps({"usable": job["usable"], "onset": job["onset"]}, sort_keys=True).encode()).hexdigest()[:10]
    ep = f"e{EPOCHS}" if job["model"] in SEQ else "e-"
    return "__".join([job["config"].replace("|", "-"), job["ds"], job["lab"], job["target"],
                      job["model"].replace(":", "-"), job["variant"], f"s{job['seed']}", ep, h])


# ------------------------------------------------------------------ two-stage model through T_hat
def two_stage_T(r: np.ndarray, i_star: int, theta: float, min_fit: int) -> np.ndarray:
    """Predicted last record T_hat at every record (J39's exponential fit); inf where it makes no prediction."""
    n = len(r)
    T_hat = np.full(n, np.inf)
    if i_star >= n:
        return T_hat
    x, lt = np.log(r[i_star:]), np.log(theta)
    u = np.arange(len(x), dtype=float)
    m = u + 1.0
    su, suu, sx, sux = np.cumsum(u), np.cumsum(u * u), np.cumsum(x), np.cumsum(u * x)
    den = m * suu - su * su
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (m * sux - su * sx) / den
        a = (sx - b * su) / m
        f_now = a + b * u
        u_hit = (lt - a) / b
    th = np.where(b > 0, i_star + np.maximum(u_hit, u), np.inf)        # u_hit < u: already past -> T_hat = t
    th = np.where(f_now >= lt, i_star + u, th)
    th = np.where(m < min_fit, np.inf, th)
    T_hat[i_star:] = th
    return T_hat


# ------------------------------------------------------------------ metrics
def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return np.nan
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def concordance(yhat_in_time_order: np.ndarray) -> float:
    """Share of pairs s < t with yhat_s > yhat_t (more life predicted earlier), ties one half."""
    v = np.asarray(yhat_in_time_order, float)
    m = len(v)
    if m < 2:
        return np.nan
    d = v[:, None] - v[None, :]
    iu = np.triu_indices(m, 1)
    dd = d[iu]
    return float(((dd > 0).sum() + 0.5 * (dd == 0).sum()) / len(dd))


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    k = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (k - rank) * p[i]))
        adj[i] = running
    return adj


def boot_p(delta: np.ndarray) -> float:
    a = delta[np.isfinite(delta)]
    if not len(a):
        return 1.0
    return float(min(1.0, 2 * max(min((a <= 0).mean(), (a >= 0).mean()), 1.0 / len(a))))


# ------------------------------------------------------------------ main
def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / ("j62_quick.log" if QUICK else "j62_run.log"), "w", encoding="utf-8")
    log(BAR)
    log("J62  RUL EVALUATION ON A COMMON RETROSPECTIVE REFERENCE  (WP-C)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}   B = {B:,}   workers = {WORKERS}   torch {torch.__version__}"
        + ("   QUICK RUN (not for the manuscript)" if QUICK else ""))

    j60, j23, j35, j39 = (load("j60", "J60_baseline_protocols.py"), load("j23", "J23_reviewer_analyses.py"),
                          load("j35", "J35_downstream_rul.py"), load("j39", "J39_two_stage_rul.py"))
    configs = [("P-retro", "canonical")] if QUICK else [("P-retro", "canonical"), ("P-prospective", "canonical")]
    calf = T / "j61_calibration.csv"
    if not QUICK and calf.exists():
        c = pd.read_csv(calf)
        if (c.R == 4000).all():
            configs += [("P-prospective", "matched5"), ("P-retro", "matched5")]
            log("  J61 full calibration found: the calibrated labelers are included as a sensitivity analysis")
        else:
            log("  J61 calibration table is from a quick run: calibrated labelers skipped")
    cal = pd.read_csv(calf) if calf.exists() else None

    # ---------------------------------------------------------------- onsets of the labelers
    def labeler_spec(lab: str, arm: str, kind: str, ds: str):
        if lab == "pct05":
            return ("pct", {"p": 0.05})
        if kind == "canonical":
            k, p, _ = j60.RULE_SPEC[lab]
            return (k, p)
        rule = {"3sig_rms": "thr", "ma5sig": "ma", "ewma": "ewma"}[lab]
        m = float(cal[(cal.dataset == ds) & (cal.protocol == arm) & (cal.noise_arm == "main") &
                      (cal.level == 0.05) & (cal.rule == rule)].multiplier.iloc[0])
        return {"thr": ("threshold", {"k": m}), "ma": ("ma", {"k": m}), "ewma": ("ewma", {"lam": 0.2, "L": m})}[rule]

    onsets, usable, smask = {}, {}, {}
    for arm, kind in configs:
        cfg = f"{arm}|{kind}"
        for ds in DATASETS:
            d = prepare(ds)
            per = {}
            for b in pd.unique(d["traj"]):
                h = d["rms"][d["traj"] == b]
                for lab in LABELERS:
                    k, p = labeler_spec(lab, arm, kind, ds)
                    if not j60.eligible(len(h), arm):
                        per[(lab, b)] = None
                        continue
                    if k == "pct":
                        per[(lab, b)] = int(len(h) * (1 - p["p"]))
                    else:
                        o = j60.onset({"rms": h, "kurt": h}, k, p, "rms", arm)
                        per[(lab, b)] = o["i_star"]
            onsets[(cfg, ds)] = per
            ok = sorted(b for b in pd.unique(d["traj"]) if all(per[(lab, b)] is not None for lab in LABELERS))
            usable[(cfg, ds)] = ok
            log(f"  {cfg:24s} {DS_NAME[ds]:7s}: {len(ok)} usable bearings "
                f"(dropped: {', '.join(b for b in pd.unique(d['traj']) if b not in ok) or 'none'})")

    # check: P-retro canonical onsets equal the submitted ones
    ft = pd.read_csv(T / "fig1_fpt_matrix.csv")
    ft["fired"] = ft.fired.astype(bool)
    bad = 0
    for ds in DATASETS:
        for r in ft[(ft.dataset == ds) & ft.rule.isin(LABELERS)].itertuples():
            mine = onsets[("P-retro|canonical", ds)][(r.rule, r.trajectory)]
            sub = int(r.fpt_index) if r.fired else None
            bad += int(mine != sub)
    log(f"  P-retro canonical onsets against fig1_fpt_matrix.csv: {bad} mismatches")
    if bad:
        log("  FAIL: nothing written")
        return 1

    # ---------------------------------------------------------------- jobs
    jobs = []
    for arm, kind in configs:
        cfg = f"{arm}|{kind}"
        for ds in DATASETS:
            for lab in LABELERS:
                for target in ("own", "common"):
                    for model in REG + SEQ:
                        for variant in variants_of(model):
                            for seed in (SEEDS if model in SEQ else (0,)):
                                job = {"config": cfg, "ds": ds, "lab": lab, "target": target, "model": model,
                                       "variant": variant, "seed": seed, "usable": usable[(cfg, ds)],
                                       "onset": {b: int(onsets[(cfg, ds)][(lab, b)]) for b in usable[(cfg, ds)]}}
                                job["key"] = job_key(job)
                                jobs.append(job)
    todo = [j for j in jobs if not (CACHE / f"{j['key']}.npy").exists()]
    todo.sort(key=lambda j: (j["model"] not in SEQ, j["ds"] != "xjtu"))      # long jobs first
    log(f"\n  training jobs: {len(jobs)} ({len(jobs) - len(todo)} cached, {len(todo)} to run)")
    if todo:
        t_start = time.time()
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx, initializer=_init_worker) as ex:
            futs = [ex.submit(run_job, j) for j in todo]
            for k, f in enumerate(as_completed(futs), 1):
                key, p, secs = f.result()
                np.save(CACHE / f"{key}.npy", p)
                if k % 25 == 0 or k == len(todo):
                    log(f"    {k}/{len(todo)} jobs done, {(time.time() - t_start) / 60:.1f} min")
    pred = {j["key"]: np.load(CACHE / f"{j['key']}.npy") for j in jobs}

    # ---------------------------------------------------------------- evaluation
    rows, pb_rows, oof = [], [], []
    rng = np.random.default_rng(SEED)
    rank_rows, me_rows, rep_rows = [], [], []
    for arm, kind in configs:
        cfg = f"{arm}|{kind}"
        for ds in DATASETS:
            d = prepare(ds)
            U = usable[(cfg, ds)]
            keep = np.isin(d["traj"], U)
            traj, tt = d["traj"][keep], d["t"][keep]
            rms = d["rms"][keep]
            n_of = {b: int((traj == b).sum()) for b in U}
            on = onsets[(cfg, ds)]
            S = {b: tt[traj == b] >= max(on[(lab, b)] for lab in LABELERS) for b in U}
            S3 = {b: tt[traj == b] >= max(on[(lab, b)] for lab in SIGNAL_LABS) for b in U}
            rem = {b: (n_of[b] - 1 - tt[traj == b]).astype(float) for b in U}
            ref = {b: rem[b] / (n_of[b] - 1) for b in U}
            cond = np.array([j23.condition_of(ds, b) for b in U])

            # two-stage predictions (main process), with the check against J39
            W_of = {b: j60.window_len(n_of[b], arm) for b in U}
            scaled = {b: rms[traj == b] / rms[traj == b][:W_of[b]].mean() for b in U}
            last = {b: scaled[b][-1] for b in U}
            ts = {}
            worst = 0.0
            for lab in LABELERS:
                for b in U:
                    theta = float(np.median([last[o] for o in U if o != b]))
                    Th = two_stage_T(scaled[b], int(on[(lab, b)]), theta, j39.MIN_FIT)
                    i0 = int(on[(lab, b)])
                    t = tt[traj == b].astype(float)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        own = np.where(t < i0, 1.0, np.where(np.isinf(Th), 1.0, (Th - t) / np.maximum(Th - i0, 1e-12)))
                        com = np.where(np.isinf(Th), 1.0, (Th - t) / np.maximum(Th, 1e-12))
                    own, com = np.clip(own, 0, 1), np.clip(com, 0, 1)
                    if arm == "P-retro":
                        ref39 = j39.two_stage(scaled[b], i0, theta, "exp")
                        worst = max(worst, float(np.abs(ref39 - own).max()))
                    ts[(lab, b)] = (own, com, np.where(np.isinf(Th), 1e12, Th - t))
            if arm == "P-retro":
                log(f"  {cfg} {DS_NAME[ds]}: two-stage through T_hat against J39.two_stage: max difference {worst:.1e}")
                if worst > 1e-9:
                    log("  FAIL: two-stage rewrite differs from J39; nothing written")
                    return 1

            # every cell: metrics per bearing and pooled
            cells = []
            for lab in LABELERS:
                for model in NINE:
                    for variant in variants_of(model):
                        seeds = SEEDS if model in SEQ else (0,)
                        per_seed = {}
                        for seed in seeds:
                            if model == "two-stage":
                                own = np.concatenate([ts[(lab, b)][0] for b in U])
                                com = np.concatenate([ts[(lab, b)][1] for b in U])
                                remp = {b: ts[(lab, b)][2] for b in U}
                            else:
                                base = {"config": cfg, "ds": ds, "lab": lab, "model": model, "variant": variant,
                                        "seed": seed, "usable": U,
                                        "onset": {b: int(on[(lab, b)]) for b in U}}
                                own = pred[job_key({**base, "target": "own"})]
                                com = pred[job_key({**base, "target": "common"})]
                                remp = None
                            y_own = np.concatenate([j35.target(n_of[b], on[(lab, b)]) for b in U])
                            pbm = {}
                            off = 0
                            for b in U:
                                n = n_of[b]
                                sl = slice(off, off + n)
                                off += n
                                yo, yc = own[sl], com[sl]
                                score = remp[b] if remp is not None else yo
                                pbm[b] = {
                                    "sse_own": float(((yo - y_own[sl]) ** 2).sum()), "n_all": n,
                                    "sse_com_S": float(((yc - ref[b]) ** 2)[S[b]].sum()), "n_S": int(S[b].sum()),
                                    "sse_com_S3": float(((yc - ref[b]) ** 2)[S3[b]].sum()), "n_S3": int(S3[b].sum()),
                                    "spearman_S": spearman(score[S[b]], rem[b][S[b]]),
                                    "spearman_all": spearman(score, rem[b]),
                                    "conc_S": concordance(score[S[b]]), "conc_S3": concordance(score[S3[b]])}
                            pooled = {
                                "rmse_own": np.sqrt(sum(v["sse_own"] for v in pbm.values()) / sum(v["n_all"] for v in pbm.values())),
                                "rmse_common_S": np.sqrt(sum(v["sse_com_S"] for v in pbm.values()) / sum(v["n_S"] for v in pbm.values())),
                                "rmse_common_S3": np.sqrt(sum(v["sse_com_S3"] for v in pbm.values()) / sum(v["n_S3"] for v in pbm.values())),
                                "conc_S_mean": float(np.nanmean([v["conc_S"] for v in pbm.values()])),
                                "conc_S3_mean": float(np.nanmean([v["conc_S3"] for v in pbm.values()])),
                                "spearman_S_median": float(np.nanmedian([v["spearman_S"] for v in pbm.values()])),
                                "spearman_all_median": float(np.nanmedian([v["spearman_all"] for v in pbm.values()]))}
                            per_seed[seed] = (pooled, pbm, own, com)
                            oof.append(pd.DataFrame({"config": cfg, "dataset": ds, "labeler": lab, "model": model,
                                                     "variant": variant, "seed": seed, "trajectory": traj,
                                                     "t": tt, "pred_own": own.astype(np.float32),
                                                     "pred_common": com.astype(np.float32)}))
                        # cell value: median over seeds; per-bearing values of the median seed by pooled C-common RMSE
                        if len(seeds) > 1:
                            vals = {s: per_seed[s][0]["rmse_common_S"] for s in seeds}
                            mid = sorted(seeds, key=lambda s: vals[s])[1]
                            vals_o = {s: per_seed[s][0]["rmse_own"] for s in seeds}
                            mid_o = sorted(seeds, key=lambda s: vals_o[s])[1]
                            pooled = {k: float(np.median([per_seed[s][0][k] for s in seeds])) for k in per_seed[seeds[0]][0]}
                            pbm = per_seed[mid][1]
                            pbm_own = per_seed[mid_o][1]
                        else:
                            mid = mid_o = seeds[0]
                            pooled, pbm = per_seed[seeds[0]][0], per_seed[seeds[0]][1]
                            pbm_own = pbm
                        cells.append({"lab": lab, "model": model, "variant": variant, "pooled": pooled,
                                      "pbm": pbm, "pbm_own": pbm_own, "median_seed_common": mid, "median_seed_own": mid_o})
                        rows.append({"config": cfg, "dataset": ds, "labeler": lab, "model": model, "variant": variant,
                                     "n_bearings": len(U), "median_seed_common": mid, "median_seed_own": mid_o,
                                     **{k: float(v) for k, v in pooled.items()}})
                        for b in U:
                            pb_rows.append({"config": cfg, "dataset": ds, "labeler": lab, "model": model,
                                            "variant": variant, "trajectory": b, "condition": j23.condition_of(ds, b),
                                            **pbm[b], "sse_own_median_seed": pbm_own[b]["sse_own"]})

            # ------------------------------------------------ comparisons (variant a, nine models)
            ca = {(c["lab"], c["model"]): c for c in cells if c["variant"] == "a"}
            idx = j60.strat_index(cond, rng)                  # shared by every comparison of this config x dataset
            for set_name, sse_k, n_k, conc_k in (("S", "sse_com_S", "n_S", "conc_S"), ("S3", "sse_com_S3", "n_S3", "conc_S3")):
                labs = LABELERS if set_name == "S" else SIGNAL_LABS
                SSE = np.array([[[ca[(l, m)]["pbm"][b][sse_k] for b in U] for m in NINE] for l in labs])
                CNT = np.array([ca[(labs[0], NINE[0])]["pbm"][b][n_k] for b in U], float)
                CON = np.array([[[ca[(l, m)]["pbm"][b][conc_k] for b in U] for m in NINE] for l in labs])
                rm_pt = np.sqrt(SSE.sum(-1) / CNT.sum())                       # (labs, models)
                rm_bt = np.sqrt(SSE[..., idx].sum(-1) / CNT[idx].sum(-1))     # (labs, models, B)
                co_pt = np.nanmean(CON, axis=-1)
                co_bt = np.nanmean(CON[..., idx], axis=-1)
                for metric, pt, bt, better in (("C-common RMSE", rm_pt, rm_bt, "lower"),
                                               ("C-rank concordance", co_pt, co_bt, "higher")):
                    sign = 1.0 if better == "lower" else -1.0               # make "smaller is better" for deltas
                    for ia, ib in itertools.combinations(range(len(labs)), 2):
                        tests = []
                        for x, y in itertools.combinations(range(len(NINE)), 2):
                            for li in (ia, ib):
                                dlt = sign * (bt[li, x] - bt[li, y])
                                tests.append((x, y, li, sign * (pt[li, x] - pt[li, y]), boot_p(dlt)))
                        padj = holm(np.array([t[4] for t in tests]))
                        sig = {(t[0], t[1], t[2]): (t[3], padj[k] < ALPHA) for k, t in enumerate(tests)}
                        revs = []
                        for x, y in itertools.combinations(range(len(NINE)), 2):
                            da, sa = sig[(x, y, ia)]
                            db, sb_ = sig[(x, y, ib)]
                            if sa and sb_ and np.sign(da) != np.sign(db):
                                revs.append(f"{NINE[x]} vs {NINE[y]}")
                        best_a = (bt[ia].argmin(0) if better == "lower" else bt[ia].argmax(0))
                        best_b = (bt[ib].argmin(0) if better == "lower" else bt[ib].argmax(0))
                        tau = kendalltau(pt[ia], pt[ib]).statistic
                        rank_rows.append({"config": cfg, "dataset": ds, "set": set_name, "metric": metric,
                                          "labeler_a": labs[ia], "labeler_b": labs[ib],
                                          "best_a": NINE[int(np.argmin(pt[ia]) if better == 'lower' else np.argmax(pt[ia]))],
                                          "best_b": NINE[int(np.argmin(pt[ib]) if better == 'lower' else np.argmax(pt[ib]))],
                                          "share_best_differs": float((best_a != best_b).mean()), "kendall_tau": float(tau),
                                          "n_significant_reversals": len(revs), "reversals": "; ".join(revs)})
                    # labeler main effect: difference in the mean over the nine models
                    mean_l = bt.mean(axis=1)                                 # (labs, B)
                    mean_p = pt.mean(axis=1)
                    for ia, ib in itertools.combinations(range(len(labs)), 2):
                        dlt = mean_l[ia] - mean_l[ib]
                        lo, hi = np.percentile(dlt[np.isfinite(dlt)], [2.5, 97.5])
                        me_rows.append({"config": cfg, "dataset": ds, "set": set_name, "metric": metric,
                                        "labeler_a": labs[ia], "labeler_b": labs[ib],
                                        "diff_mean_over_models": float(mean_p[ia] - mean_p[ib]), "lo": float(lo), "hi": float(hi),
                                        "excludes_0": bool(lo > 0 or hi < 0)})

            # ------------------------------------------------ the pre-specified reading (S, all four labelers)
            rr = pd.DataFrame([r for r in rank_rows if r["config"] == cfg and r["dataset"] == ds and r["set"] == "S"])
            mm = pd.DataFrame([r for r in me_rows if r["config"] == cfg and r["dataset"] == ds and r["set"] == "S"])
            n_rev = int(rr[rr.metric == "C-common RMSE"].n_significant_reversals.sum())
            me = bool(mm.excludes_0.any())
            reading = ("1: reversal under the common reference" if n_rev else
                       ("2: labeler main effect, no reversal" if me else "3: neither"))
            rep_rows.append({"config": cfg, "dataset": ds, "n_reversals_common": n_rev,
                             "main_effect_present": me, "reading": reading})
            log(f"\n  {cfg} {DS_NAME[ds]} ({len(U)} bearings): significant reversals under C-common: {n_rev}; "
                f"labeler main effect present: {me} -> reading {reading}")
            for _, r in rr[rr.metric == "C-common RMSE"].iterrows():
                log(f"    {r.labeler_a:8s} vs {r.labeler_b:8s}: best {r.best_a} / {r.best_b}; best differs in "
                    f"{r.share_best_differs:.2f} of resamples; tau {r.kendall_tau:+.2f}; reversals: {r.reversals or 'none'}")

    res = pd.DataFrame(rows)

    # ---------------------------------------------------------------- replication of J35 and J47
    log(f"\n{BAR}\nReplication on the submitted protocol (P-retro, canonical labelers, input (a), own targets)\n{BAR}")
    r35 = pd.read_csv(T / "j35_downstream_rul.csv")
    mine = res[(res.config == "P-retro|canonical") & (res.variant == "a") & res.model.isin(REG)]
    m35 = mine.merge(r35, left_on=["dataset", "labeler", "model"], right_on=["dataset", "labeller", "model"])
    dev35 = float((m35.rmse_own - m35.rmse).abs().max()) if len(m35) else np.nan
    log(f"  six regressors against J35 ({len(m35)} cells): largest |RMSE difference| {dev35:.2e}")
    if not QUICK and not dev35 < 1e-9:
        log("  FAIL: the regressors do not reproduce J35; nothing written")
        return 1
    r47 = pd.read_csv(T / "j47_sequence_models.csv")
    r47 = r47[r47.arm == "main"]
    m47 = res[(res.config == "P-retro|canonical") & (res.variant == "a") & res.model.isin(SEQ)].merge(
        r47, left_on=["dataset", "labeler", "model"], right_on=["dataset", "labeller", "model"])
    for _, r in m47.iterrows():
        log(f"    {r.dataset:6s} {r.labeler:9s} {r.model}: J47 (torch 2.12) {r.rmse:.4f}, here (torch {torch.__version__}) "
            f"{r.rmse_own:.4f}, difference {r.rmse_own - r.rmse:+.4f}")
    if len(m47):
        log(f"  GRU/TCN: largest |difference| {float((m47.rmse_own - m47.rmse).abs().max()):.4f}")

    # ---------------------------------------------------------------- summaries
    log(f"\n{BAR}\nC-common RMSE on S_j (variant a) and C-rank concordance, by labeler (median over the nine models)\n{BAR}")
    for (cfg, ds), g in res[res.variant == "a"].groupby(["config", "dataset"], sort=False):
        line = f"  {cfg:24s} {DS_NAME[ds]:7s}"
        for lab in LABELERS:
            q = g[g.labeler == lab]
            line += f"  {lab}: {q.rmse_common_S.median():.3f} / {q.conc_S_mean.median():.3f}"
        log(line)
    log("\n  input variants (median over labelers and models of C-common RMSE on S_j): ")
    for (cfg, ds), g in res.groupby(["config", "dataset"], sort=False):
        log(f"    {cfg:24s} {DS_NAME[ds]:7s} " + ", ".join(
            f"{v}: {g[g.variant == v].rmse_common_S.median():.3f}" for v in ("a", "b", "c", "d")))
    log("    (a: per-record statistics; b: + elapsed record count; c, d: + history over 16, 64 records, tree models)")

    res.to_csv(T / "j62_common_reference.csv", index=False)
    pd.DataFrame(pb_rows).to_csv(T / "j62_per_bearing.csv", index=False)
    pd.DataFrame(rank_rows).to_csv(T / "j62_rankings.csv", index=False)
    pd.DataFrame(me_rows).to_csv(T / "j62_main_effect.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(T / "j62_readings.csv", index=False)
    pd.concat(oof, ignore_index=True).to_parquet(T / ("j62_oof_predictions_quick.parquet" if QUICK else "j62_oof_predictions.parquet"),
                                                  index=False)
    log(f"\nwrote j62_common_reference.csv ({len(res)}), j62_per_bearing.csv ({len(pb_rows)}), "
        f"j62_rankings.csv ({len(rank_rows)}), j62_main_effect.csv ({len(me_rows)}), j62_readings.csv ({len(rep_rows)}), "
        f"j62_oof_predictions parquet")
    log(f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
