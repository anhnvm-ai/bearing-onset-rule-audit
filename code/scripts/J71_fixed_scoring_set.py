"""J71 -- RUL model comparison on a scoring set that does not depend on the labelers, and with the elapsed record
count as input (amendment A9, items 2 and 3; post hoc).

J62 scores every model on S_j = {t >= the latest onset of the four labelers}; with pct05 among the labelers, S_j is
about the last 5% of life, which is also the training window of pct05. Its variant S3_j uses the three signal-based
labelers. This script re-scores the cached out-of-fold predictions of J62; no model is retrained.

  item 2   S^fix_j = {t >= floor(0.8 n_j)}, the last 20% of the records of bearing j, with all six labeler pairs, the
           nine models with input variant (a), C-common RMSE (primary) and C-rank concordance (secondary): reversals,
           best models, Kendall tau and labeler main effects exactly as in J62.
           Alignment of each labeler with S_j, S3_j and S^fix_j, pooled over bearings: coverage, the share of scored
           records with t >= i*_L,j (inside the training window that labeler L selects on bearing j), and
           specialization, the share of the records with t >= i*_L,j that are scored.
  item 3   the reversal test of J62 over S_j and S3_j with input variant (b) (per-record statistics plus the elapsed
           record count t) for the eight models that have it, and variant (a) for the two-stage model.

Everything else is J62's: configurations, bearings, onsets, cached predictions (results/cache_j62), the median seed of
GRU and TCN (by pooled C-common RMSE over S_j), B = 20,000 resamples (seed 62, drawn in J62's order, hence identical),
the bootstrap p-value and Holm's correction over the 2 x 36 tests of a labeler pair.

Self-check before anything is written: the same code over S_j and S3_j reproduces j62_common_reference.csv (pooled
values of variants (a) and (b), to 1e-9), j62_rankings.csv and j62_main_effect.csv.

Readings fixed in A9 (C-common RMSE, primary configurations P-retro and P-prospective with conventional labelers):
  F1  a significant reversal on S^fix between two signal-based labelers
  F2  significant reversals on S^fix only in pairs that include pct05
  F3  no significant reversal on S^fix
  variant (b): at least one significant reversal over S_j, or none

Pre-specified in paper/access_resub/ANALYSIS_PLAN_AMENDMENTS.md, amendment A9 (items 2 and 3).

Outputs: results/tables/j71_rankings.csv     analysis x config x dataset x set x metric x labeler pair
         results/tables/j71_main_effect.csv  analysis x config x dataset x set x metric x labeler pair
         results/tables/j71_cells.csv        config x dataset x labeler x model x variant: pooled values on every set
         results/tables/j71_alignment.csv    config x dataset x set x labeler: coverage and specialization
         results/tables/j71_readings.csv     the readings of A9
         results/logs/j71_run.log
Usage  : python scripts/J71_fixed_scoring_set.py
"""
from __future__ import annotations

# torch must be imported before scikit-learn on this machine (WinError 1114 otherwise; see J59); J62 imports both
import torch  # noqa: F401  (must come first)

import importlib.util
import itertools
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
FIX_FRACTION = 0.8                      # S^fix_j = {t >= floor(0.8 n_j)} (A9)
SETS = ("S", "S3", "Sfix")
PRIMARY = ("P-retro|canonical", "P-prospective|canonical")
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


def compare(j62, cells: dict, labs: list[str], set_name: str, U: list[str], idx: np.ndarray, models: list[str]):
    """J62's model comparison (reversals, best model, Kendall tau, labeler main effects) on one scoring set."""
    SSE = np.array([[[cells[(l, m)]["pbm"][b][f"sse_{set_name}"] for b in U] for m in models] for l in labs])
    CNT = np.array([cells[(labs[0], models[0])]["pbm"][b][f"n_{set_name}"] for b in U], float)
    CON = np.array([[[cells[(l, m)]["pbm"][b][f"conc_{set_name}"] for b in U] for m in models] for l in labs])
    rm_pt = np.sqrt(SSE.sum(-1) / CNT.sum())
    rm_bt = np.sqrt(SSE[..., idx].sum(-1) / CNT[idx].sum(-1))
    co_pt = np.nanmean(CON, axis=-1)
    co_bt = np.nanmean(CON[..., idx], axis=-1)
    rank_rows, me_rows = [], []
    for metric, pt, bt, better in (("C-common RMSE", rm_pt, rm_bt, "lower"),
                                   ("C-rank concordance", co_pt, co_bt, "higher")):
        sign = 1.0 if better == "lower" else -1.0
        for ia, ib in itertools.combinations(range(len(labs)), 2):
            tests = []
            for x, y in itertools.combinations(range(len(models)), 2):
                for li in (ia, ib):
                    dlt = sign * (bt[li, x] - bt[li, y])
                    tests.append((x, y, li, sign * (pt[li, x] - pt[li, y]), j62.boot_p(dlt)))
            padj = j62.holm(np.array([t[4] for t in tests]))
            sig = {(t[0], t[1], t[2]): (t[3], padj[k] < j62.ALPHA) for k, t in enumerate(tests)}
            revs = []
            for x, y in itertools.combinations(range(len(models)), 2):
                da, sa = sig[(x, y, ia)]
                db, sb_ = sig[(x, y, ib)]
                if sa and sb_ and np.sign(da) != np.sign(db):
                    revs.append(f"{models[x]} vs {models[y]}")
            best_a = (bt[ia].argmin(0) if better == "lower" else bt[ia].argmax(0))
            best_b = (bt[ib].argmin(0) if better == "lower" else bt[ib].argmax(0))
            tau = kendalltau(pt[ia], pt[ib]).statistic
            rank_rows.append({"set": set_name, "metric": metric, "labeler_a": labs[ia], "labeler_b": labs[ib],
                              "best_a": models[int(np.argmin(pt[ia]) if better == "lower" else np.argmax(pt[ia]))],
                              "best_b": models[int(np.argmin(pt[ib]) if better == "lower" else np.argmax(pt[ib]))],
                              "share_best_differs": float((best_a != best_b).mean()), "kendall_tau": float(tau),
                              "n_significant_reversals": len(revs), "reversals": "; ".join(revs)})
        mean_l = bt.mean(axis=1)
        mean_p = pt.mean(axis=1)
        for ia, ib in itertools.combinations(range(len(labs)), 2):
            dlt = mean_l[ia] - mean_l[ib]
            lo, hi = np.percentile(dlt[np.isfinite(dlt)], [2.5, 97.5])
            me_rows.append({"set": set_name, "metric": metric, "labeler_a": labs[ia], "labeler_b": labs[ib],
                            "diff_mean_over_models": float(mean_p[ia] - mean_p[ib]), "lo": float(lo), "hi": float(hi),
                            "excludes_0": bool(lo > 0 or hi < 0)})
    return rank_rows, me_rows


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j71_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J71  RUL COMPARISON ON A FIXED SCORING SET AND WITH TIME INPUTS  (A9, post hoc)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}")

    j62 = load("j62", "J62_common_reference_rul.py")
    if j62.QUICK or j62.CACHE.name != "cache_j62" or j62.EPOCHS != 30 or j62.B != 20000:
        log("  FAIL: J62 is not in its full configuration (QUICK, EPOCHS or B set in the environment)")
        return 1
    j60, j23, j35, j39 = (load("j60", "J60_baseline_protocols.py"), load("j23", "J23_reviewer_analyses.py"),
                          load("j35", "J35_downstream_rul.py"), load("j39", "J39_two_stage_rul.py"))
    LABELERS, SIGNAL_LABS, NINE, SEQ, SEEDS = j62.LABELERS, j62.SIGNAL_LABS, j62.NINE, j62.SEQ, j62.SEEDS
    configs = [("P-retro", "canonical"), ("P-prospective", "canonical"), ("P-prospective", "matched5"),
               ("P-retro", "matched5")]                                          # J62's order
    cal = pd.read_csv(T / "j61_calibration.csv")

    def labeler_spec(lab: str, arm: str, kind: str, ds: str):                   # as in J62
        if lab == "pct05":
            return ("pct", {"p": 0.05})
        if kind == "canonical":
            k, p, _ = j60.RULE_SPEC[lab]
            return (k, p)
        rule = {"3sig_rms": "thr", "ma5sig": "ma", "ewma": "ewma"}[lab]
        m = float(cal[(cal.dataset == ds) & (cal.protocol == arm) & (cal.noise_arm == "main") &
                      (cal.level == 0.05) & (cal.rule == rule)].multiplier.iloc[0])
        return {"thr": ("threshold", {"k": m}), "ma": ("ma", {"k": m}), "ewma": ("ewma", {"lam": 0.2, "L": m})}[rule]

    onsets, usable = {}, {}
    for arm, kind in configs:
        cfg = f"{arm}|{kind}"
        for ds in j62.DATASETS:
            d = j62.prepare(ds)
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
                        per[(lab, b)] = j60.onset({"rms": h, "kurt": h}, k, p, "rms", arm)["i_star"]
            onsets[(cfg, ds)] = per
            usable[(cfg, ds)] = sorted(b for b in pd.unique(d["traj"])
                                       if all(per[(lab, b)] is not None for lab in LABELERS))

    ref62 = pd.read_csv(T / "j62_common_reference.csv")
    rk62 = pd.read_csv(T / "j62_rankings.csv")
    me62 = pd.read_csv(T / "j62_main_effect.csv")
    rng = np.random.default_rng(j62.SEED)                                         # J62's resample stream
    cell_rows, rank_rows, me_rows, al_rows = [], [], [], []
    worst_cell = 0.0
    for arm, kind in configs:
        cfg = f"{arm}|{kind}"
        for ds in j62.DATASETS:
            d = j62.prepare(ds)
            U = usable[(cfg, ds)]
            keep = np.isin(d["traj"], U)
            traj, tt, rms = d["traj"][keep], d["t"][keep], d["rms"][keep]
            n_of = {b: int((traj == b).sum()) for b in U}
            on = onsets[(cfg, ds)]
            masks = {"S": {b: tt[traj == b] >= max(on[(lab, b)] for lab in LABELERS) for b in U},
                     "S3": {b: tt[traj == b] >= max(on[(lab, b)] for lab in SIGNAL_LABS) for b in U},
                     "Sfix": {b: tt[traj == b] >= int(np.floor(FIX_FRACTION * n_of[b])) for b in U}}
            rem = {b: (n_of[b] - 1 - tt[traj == b]).astype(float) for b in U}
            ref = {b: rem[b] / (n_of[b] - 1) for b in U}
            cond = np.array([j23.condition_of(ds, b) for b in U])

            # alignment of each labeler's training window with each scored set
            for lab in LABELERS:
                win = {b: tt[traj == b] >= on[(lab, b)] for b in U}
                for s in SETS:
                    both = sum(int((masks[s][b] & win[b]).sum()) for b in U)
                    al_rows.append({"config": cfg, "dataset": ds, "set": s, "labeler": lab,
                                    "coverage": both / sum(int(masks[s][b].sum()) for b in U),
                                    "specialization": both / sum(int(win[b].sum()) for b in U),
                                    "scored_records": sum(int(masks[s][b].sum()) for b in U),
                                    "window_records": sum(int(win[b].sum()) for b in U)})

            # two-stage predictions, as in J62
            W_of = {b: j60.window_len(n_of[b], arm) for b in U}
            scaled = {b: rms[traj == b] / rms[traj == b][:W_of[b]].mean() for b in U}
            last = {b: scaled[b][-1] for b in U}
            ts = {}
            for lab in LABELERS:
                for b in U:
                    theta = float(np.median([last[o] for o in U if o != b]))
                    Th = j62.two_stage_T(scaled[b], int(on[(lab, b)]), theta, j39.MIN_FIT)
                    i0 = int(on[(lab, b)])
                    t = tt[traj == b].astype(float)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        own = np.where(t < i0, 1.0, np.where(np.isinf(Th), 1.0, (Th - t) / np.maximum(Th - i0, 1e-12)))
                        com = np.where(np.isinf(Th), 1.0, (Th - t) / np.maximum(Th, 1e-12))
                    ts[(lab, b)] = (np.clip(own, 0, 1), np.clip(com, 0, 1), np.where(np.isinf(Th), 1e12, Th - t))

            # cells of variants (a) and (b): per-bearing values on every set; median seed as in J62
            cells = {}
            for lab in LABELERS:
                for model in NINE:
                    for variant in [v for v in j62.variants_of(model) if v in ("a", "b")]:
                        seeds = SEEDS if model in SEQ else (0,)
                        per_seed = {}
                        for seed in seeds:
                            if model == "two-stage":
                                own = np.concatenate([ts[(lab, b)][0] for b in U])
                                com = np.concatenate([ts[(lab, b)][1] for b in U])
                                remp = {b: ts[(lab, b)][2] for b in U}
                            else:
                                base = {"config": cfg, "ds": ds, "lab": lab, "model": model, "variant": variant,
                                        "seed": seed, "usable": U, "onset": {b: int(on[(lab, b)]) for b in U}}
                                own = np.load(j62.CACHE / f"{j62.job_key({**base, 'target': 'own'})}.npy")
                                com = np.load(j62.CACHE / f"{j62.job_key({**base, 'target': 'common'})}.npy")
                                remp = None
                            pbm, off = {}, 0
                            for b in U:
                                sl = slice(off, off + n_of[b])
                                off += n_of[b]
                                yo, yc = own[sl], com[sl]
                                score = remp[b] if remp is not None else yo
                                e2 = (yc - ref[b]) ** 2
                                pbm[b] = {}
                                for s in SETS:
                                    mk = masks[s][b]
                                    pbm[b][f"sse_{s}"] = float(e2[mk].sum())
                                    pbm[b][f"n_{s}"] = int(mk.sum())
                                    pbm[b][f"conc_{s}"] = j62.concordance(score[mk])
                            pooled = {}
                            for s in SETS:
                                pooled[f"rmse_common_{s}"] = float(np.sqrt(sum(v[f"sse_{s}"] for v in pbm.values()) /
                                                                           sum(v[f"n_{s}"] for v in pbm.values())))
                                pooled[f"conc_{s}_mean"] = float(np.nanmean([v[f"conc_{s}"] for v in pbm.values()]))
                            per_seed[seed] = (pooled, pbm)
                        if len(seeds) > 1:
                            vals = {s: per_seed[s][0]["rmse_common_S"] for s in seeds}
                            mid = sorted(seeds, key=lambda s: vals[s])[1]
                            pooled = {k: float(np.median([per_seed[s][0][k] for s in seeds])) for k in per_seed[seeds[0]][0]}
                            pbm = per_seed[mid][1]
                        else:
                            pooled, pbm = per_seed[seeds[0]]
                        cells[(lab, model, variant)] = {"pooled": pooled, "pbm": pbm}
                        cell_rows.append({"config": cfg, "dataset": ds, "labeler": lab, "model": model,
                                          "variant": variant, "n_bearings": len(U), **pooled})
                        r = ref62[(ref62.config == cfg) & (ref62.dataset == ds) & (ref62.labeler == lab) &
                                  (ref62.model == model) & (ref62.variant == variant)]
                        for k62, k71 in (("rmse_common_S", "rmse_common_S"), ("rmse_common_S3", "rmse_common_S3"),
                                         ("conc_S_mean", "conc_S_mean"), ("conc_S3_mean", "conc_S3_mean")):
                            worst_cell = max(worst_cell, abs(float(r[k62].iloc[0]) - pooled[k71]))

            # comparisons on J62's resamples
            idx = j60.strat_index(cond, rng)
            ca = {(l, m): cells[(l, m, "a")] for l in LABELERS for m in NINE}
            cb = {(l, m): cells[(l, m, "b" if "b" in j62.variants_of(m) else "a")] for l in LABELERS for m in NINE}
            for analysis, cs, set_name, labs in (("replication", ca, "S", LABELERS), ("replication", ca, "S3", SIGNAL_LABS),
                                                 ("fixed set", ca, "Sfix", LABELERS),
                                                 ("variant b", cb, "S", LABELERS), ("variant b", cb, "S3", SIGNAL_LABS)):
                rr, mm = compare(j62, cs, labs, set_name, U, idx, NINE)
                for x in rr:
                    rank_rows.append({"analysis": analysis, "config": cfg, "dataset": ds, **x})
                for x in mm:
                    me_rows.append({"analysis": analysis, "config": cfg, "dataset": ds, **x})
            log(f"  {cfg:24s} {j62.DS_NAME[ds]:7s} ({len(U)} bearings) done")

    # ---------------------------------------------------------------- self-checks against J62
    rk, me = pd.DataFrame(rank_rows), pd.DataFrame(me_rows)
    log(f"\n  self-check: pooled values against j62_common_reference.csv (variants a, b): largest |difference| {worst_cell:.1e}")
    rep = rk[rk.analysis == "replication"].drop(columns="analysis")
    keys = ["config", "dataset", "set", "metric", "labeler_a", "labeler_b"]
    m = rk62.merge(rep, on=keys, suffixes=("_62", "_71"), how="outer", indicator=True)
    bad_rank = int((m._merge != "both").sum())
    for col in ("best_a", "best_b", "n_significant_reversals"):
        bad_rank += int((m[f"{col}_62"] != m[f"{col}_71"]).sum())
    bad_rank += int((m.reversals_62.fillna("") != m.reversals_71.fillna("")).sum())
    num_rank = float(max((m.share_best_differs_62 - m.share_best_differs_71).abs().max(),
                         (m.kendall_tau_62 - m.kendall_tau_71).abs().max()))
    mrep = me[me.analysis == "replication"].drop(columns="analysis")
    mm_ = me62.merge(mrep, on=keys, suffixes=("_62", "_71"), how="outer", indicator=True)
    bad_me = int((mm_._merge != "both").sum()) + int((mm_.excludes_0_62 != mm_.excludes_0_71).sum())
    num_me = float(max((mm_[f"{c}_62"] - mm_[f"{c}_71"]).abs().max() for c in ("diff_mean_over_models", "lo", "hi")))
    log(f"  self-check: j62_rankings.csv ({len(rk62)} rows): {bad_rank} mismatches, largest numeric difference {num_rank:.1e}")
    log(f"  self-check: j62_main_effect.csv ({len(me62)} rows): {bad_me} mismatches, largest numeric difference {num_me:.1e}")
    if worst_cell > 1e-9 or bad_rank or num_rank > 1e-12 or bad_me or num_me > 1e-9:
        log("  FAIL: J62 is not reproduced; nothing written")
        return 1
    log("  PASS")

    # ---------------------------------------------------------------- results and the readings of A9
    log(f"\n{BAR}\nSignificant reversals (C-common RMSE / C-rank concordance), per configuration and dataset\n{BAR}")
    new = rk[rk.analysis != "replication"]
    for (analysis, set_name), g in new.groupby(["analysis", "set"], sort=False):
        log(f"\n  {analysis}, set {set_name}:")
        for (cfg, ds), h in g.groupby(["config", "dataset"], sort=False):
            c = h[h.metric == "C-common RMSE"]
            r = h[h.metric == "C-rank concordance"]
            log(f"    {cfg:24s} {j62.DS_NAME[ds]:7s}: {int(c.n_significant_reversals.sum())} / "
                f"{int(r.n_significant_reversals.sum())}; median tau (C-common) {c.kendall_tau.median():+.2f}")
            for _, x in h[h.n_significant_reversals > 0].iterrows():
                log(f"      {x.metric:18s} {x.labeler_a} vs {x.labeler_b}: {x.reversals}")

    fx = new[(new.analysis == "fixed set") & (new.metric == "C-common RMSE")]
    signal_pair = ~fx.labeler_a.eq("pct05") & ~fx.labeler_b.eq("pct05")
    prim = fx.config.isin(PRIMARY)
    n_sig_signal = int(fx[prim & signal_pair].n_significant_reversals.sum())
    n_sig_pct = int(fx[prim & ~signal_pair].n_significant_reversals.sum())
    reading_f = "F1" if n_sig_signal else ("F2" if n_sig_pct else "F3")
    vb = new[(new.analysis == "variant b") & (new.metric == "C-common RMSE")]
    n_vb = int(vb[vb.config.isin(PRIMARY) & (vb.set == "S")].n_significant_reversals.sum())
    reading_v = "at least one reversal" if n_vb else "no reversal"
    readings = pd.DataFrame([
        {"item": "2 (S^fix)", "reading": reading_f,
         "primary_signal_pairs": n_sig_signal, "primary_pct05_pairs": n_sig_pct,
         "sensitivity_signal_pairs": int(fx[~prim & signal_pair].n_significant_reversals.sum()),
         "sensitivity_pct05_pairs": int(fx[~prim & ~signal_pair].n_significant_reversals.sum())},
        {"item": "3 (variant b)", "reading": reading_v, "primary_S": n_vb,
         "primary_S3": int(vb[vb.config.isin(PRIMARY) & (vb.set == "S3")].n_significant_reversals.sum()),
         "sensitivity_S": int(vb[~vb.config.isin(PRIMARY) & (vb.set == "S")].n_significant_reversals.sum()),
         "sensitivity_S3": int(vb[~vb.config.isin(PRIMARY) & (vb.set == "S3")].n_significant_reversals.sum())}])
    log(f"\n{BAR}\nReadings of A9 (C-common RMSE, primary configurations)\n{BAR}")
    log(f"  item 2, S^fix: reversals between signal-based labelers {n_sig_signal}, in pairs with pct05 {n_sig_pct} "
        f"-> {reading_f}")
    log(f"  item 3, variant (b) over S_j: {n_vb} reversals -> {reading_v}")

    al = pd.DataFrame(al_rows)
    log(f"\n{BAR}\nAlignment with the scored set (coverage / specialization), median over configurations and datasets\n{BAR}")
    for s in SETS:
        g = al[al.set == s].groupby("labeler")[["coverage", "specialization"]].median()
        log(f"  {s:5s} " + "  ".join(f"{lab}: {g.loc[lab, 'coverage']:.2f} / {g.loc[lab, 'specialization']:.2f}"
                                    for lab in LABELERS))

    cells_df = pd.DataFrame(cell_rows)
    log(f"\n  C-common RMSE over S^fix (variant a), median over the nine models, by labeler:")
    for (cfg, ds), g in cells_df[cells_df.variant == "a"].groupby(["config", "dataset"], sort=False):
        log(f"    {cfg:24s} {j62.DS_NAME[ds]:7s} " + "  ".join(
            f"{lab}: {g[g.labeler == lab].rmse_common_Sfix.median():.3f}" for lab in LABELERS))

    new.to_csv(T / "j71_rankings.csv", index=False)
    me[me.analysis != "replication"].to_csv(T / "j71_main_effect.csv", index=False)
    cells_df.to_csv(T / "j71_cells.csv", index=False)
    al.to_csv(T / "j71_alignment.csv", index=False)
    readings.to_csv(T / "j71_readings.csv", index=False)
    log(f"\nwrote j71_rankings.csv ({len(new)}), j71_main_effect.csv, j71_cells.csv ({len(cells_df)}), "
        f"j71_alignment.csv ({len(al)}), j71_readings.csv")
    log(f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
