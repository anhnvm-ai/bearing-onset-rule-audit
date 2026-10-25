"""J64 -- WP-F: what predicts how a design's onset fraction scales with test length (R2.2, with R1.7).

R2.2: a comment of the review reports, answered in the response letter.
R1.7: a comment of the review reports, answered in the response letter.

The submitted Section IV regressed each design's duration statistic on the shift d MEASURED at its own alarms
(R^2 = 0.788 on XJTU-SY). d and the alarm are two readings of one event, so this script uses only predictors that are
known before any bearing is seen, and treats d as a descriptive mediator:

  outcome  b1 - 1 of the design: slope of log u, u = (i* + 1)/n, on log n with condition fixed effects, over the
           bearings of the set on which the design fires (>= 4)
  M0       family label of J26 (detector; control chart; AR-corrected chart; adaptive-limit chart), as dummies
  M1       nominal shift of J41 (convention A), log and its square
  M2       false-alarm rate of the design at its own parameters (J61, main null model), logit and its square
  M3       M2 + family
  M4       M1 + M2 (optional)

Reported: in-sample R^2; its distribution over B = 20,000 bearing resamples stratified by condition (seed 64), each
resample recomputing all 67 design outcomes from the per-trajectory alarms before the fit, with the design FAR held
fixed; Delta R^2 (M3 - M2) with its interval; out-of-sample R^2 over every distinct stratified half-split of the
bearings (or 2,000 random ones where more exist), each scored in both directions, with the split-half reliability of
the outcome as the ceiling. The measured shift d and the onset fraction are shown as descriptive mediators.

Bearing sets (amendment A4): both protocols on the bearings eligible under P-prospective (primary); P-retro on all
bearings (continuity with the submitted manuscript).

Pre-specified in paper/access_resub/ANALYSIS_PLAN_2026-10-24.md, Section 10; amendment A4.

Inputs : results/tables/j60_design_onsets.csv, results/tables/j61_design_far.csv, results/tables/j61_calibration.csv
Outputs: results/tables/j64_design_outcomes.csv   protocol x dataset x set x design
         results/tables/j64_models.csv            protocol x dataset x set x model: R^2 in and out of sample
         results/tables/j64_splits.csv            every split: out-of-sample R^2 per model, reliability
         results/logs/j64_run.log
Usage  : python scripts/J64_design_parameters.py          (B = 20,000)
         B=500 python scripts/J64_design_parameters.py    (quick run)
"""
from __future__ import annotations

import importlib.util
import itertools
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
SEED = 64
B = int(os.environ.get("B", "20000"))
os.environ["B"] = str(B)
MIN_FIRED, MIN_DESIGNS, MAX_SPLITS = 4, 10, 2000
DATASETS = ("xjtu", "femto")
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO"}
FAMILIES = ["detector", "control chart", "chart, AR-corrected", "chart, adaptive limits"]
MODELS = ["M0", "M1", "M2", "M3", "M4"]
SETS = [("P-prospective", "main"), ("P-retro", "main"), ("P-retro", "all")]
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


def design_matrix(model: str, lnom, lfar, fam_idx) -> np.ndarray:
    one = np.ones(len(lnom))
    dummies = [(fam_idx == k).astype(float) for k in range(1, len(FAMILIES))]
    cols = {"M0": [one] + dummies,
            "M1": [one, lnom, lnom ** 2],
            "M2": [one, lfar, lfar ** 2],
            "M3": [one, lfar, lfar ** 2] + dummies,
            "M4": [one, lnom, lnom ** 2, lfar, lfar ** 2]}[model]
    return np.column_stack(cols)


def r2_in(y, X) -> float:
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    sst = float(((y - y.mean()) ** 2).sum())
    return 1 - float(((y - X @ b) ** 2).sum()) / sst if sst > 0 else np.nan


def r2_oos(y_tr, X_tr, y_te, X_te) -> float:
    b = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
    sst = float(((y_te - y_te.mean()) ** 2).sum())
    return 1 - float(((y_te - X_te @ b) ** 2).sum()) / sst if sst > 0 else np.nan


def stratified_splits(cond: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Boolean masks of half A: every distinct unordered stratified split, or MAX_SPLITS random ones."""
    levels = sorted(set(cond))
    per = []
    for k in levels:
        ix = np.where(cond == k)[0]
        sizes = {len(ix) // 2, (len(ix) + 1) // 2}
        per.append([c for s in sorted(sizes) for c in itertools.combinations(ix, s)])
    total_ordered = int(np.prod([len(p) for p in per]))
    n = len(cond)
    if total_ordered // 2 <= MAX_SPLITS:
        out = []
        for combo in itertools.product(*per):
            A = np.zeros(n, bool)
            for c in combo:
                A[list(c)] = True
            if A[0]:                                  # one representative of each unordered pair
                out.append(A)
        return out
    out = []
    for _ in range(MAX_SPLITS):
        A = np.zeros(n, bool)
        for p in per:
            A[list(p[rng.integers(len(p))])] = True
        out.append(A)
    return out


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j64_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J64  DESIGN PARAMETERS AND ONSET SCALING  (WP-F)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}   B = {B:,}   seed = {SEED}")
    if B < 20000:
        log("NOTE: B below 20,000 -- a quick run; the manuscript numbers need the default")

    j60, j26, j41 = load("j60", "J60_baseline_protocols.py"), load("j26", "J26_design_sweep.py"), \
        load("j41", "J41_estimand_separation.py")
    grid = j26.build_grid()
    nominal = {s["design"]: j41.nominal_shift(s, j41.CONVENTIONS["A_primary"]) for s in grid}
    family = {s["design"]: s["family"] for s in grid}
    designs = [s["design"] for s in grid]
    don = pd.read_csv(T / "j60_design_onsets.csv")
    don["fired"] = don.fired.astype(bool)
    dfar = pd.read_csv(T / "j61_design_far.csv")
    dfar = dfar[dfar.noise_arm == "main"]
    calib = pd.read_csv(T / "j61_calibration.csv")
    if not (calib.R == 4000).all():
        log("FAIL: J61 tables are from a quick run; run J61 with R = 4,000 first")
        return 1
    rng = np.random.default_rng(SEED)

    out_rows, mod_rows, split_rows = [], [], []
    for proto, set_name in SETS:
        for ds in DATASETS:
            d = don[(don.arm == proto) & (don.dataset == ds)]
            trajs = list(dict.fromkeys(d.trajectory))
            nrec = d.drop_duplicates("trajectory").set_index("trajectory").n_records
            if set_name == "main":
                trajs = [t for t in trajs if nrec[t] >= 80]
            cond = d.drop_duplicates("trajectory").set_index("trajectory").condition.loc[trajs].to_numpy()
            n = nrec.loc[trajs].to_numpy(float)
            A = np.full((len(designs), len(trajs)), np.nan)
            S = np.full((len(designs), len(trajs)), np.nan)
            piv = d[d.trajectory.isin(trajs)].set_index(["design", "trajectory"])
            for i, des in enumerate(designs):
                for j, t in enumerate(trajs):
                    r = piv.loc[(des, t)]
                    if bool(r.fired):
                        A[i, j], S[i, j] = r.alarm_index, r.shift_sd
            R_tot = int(calib[(calib.dataset == ds) & (calib.protocol == proto)].assign(
                tot=lambda c: c.R * c.n_lengths).tot.iloc[0])
            far = dfar[(dfar.dataset == ds) & (dfar.protocol == proto)].set_index("design").far.loc[designs].to_numpy()
            farc = np.clip(far, 1 / (2 * R_tot), 1 - 1 / (2 * R_tot))
            lfar = np.log(farc / (1 - farc))
            lnom = np.log(np.array([nominal[x] for x in designs]))
            fam_idx = np.array([FAMILIES.index(family[x]) for x in designs])

            def outcomes(cols: np.ndarray) -> np.ndarray:
                y = np.full(len(designs), np.nan)
                for i in range(len(designs)):
                    a = A[i, cols]
                    f = np.isfinite(a)
                    if f.sum() >= MIN_FIRED and np.ptp(n[cols][f]) > 0:
                        y[i] = j60.fe_slope(np.log((a[f] + 1) / n[cols][f]), np.log(n[cols][f]), cond[cols][f])
                return y

            allc = np.arange(len(trajs))
            y0 = outcomes(allc)
            ok0 = np.isfinite(y0)
            med_u = np.array([np.nanmedian((A[i] + 1) / n) if np.isfinite(A[i]).any() else np.nan for i in range(len(designs))])
            med_d = np.array([np.nanmedian(S[i]) if np.isfinite(S[i]).any() else np.nan for i in range(len(designs))])
            for i, des in enumerate(designs):
                out_rows.append({"protocol": proto, "set": set_name, "dataset": ds, "design": des, "family": family[des],
                                 "n_bearings": len(trajs), "n_fired": int(np.isfinite(A[i]).sum()),
                                 "outcome_b1_minus_1": y0[i], "median_u": med_u[i], "median_shift_d": med_d[i],
                                 "nominal_shift": nominal[des], "far": far[i], "logit_far": lfar[i]})

            # in-sample, with the mediators for reference
            ins = {m: r2_in(y0[ok0], design_matrix(m, lnom[ok0], lfar[ok0], fam_idx[ok0])) for m in MODELS}
            okd = ok0 & np.isfinite(med_d) & (med_d > 0)
            ld = np.log(med_d[okd])
            ins_size = r2_in(y0[okd], np.column_stack([np.ones(okd.sum()), ld, ld ** 2]))
            oku = ok0 & np.isfinite(med_u)
            ins_frac = r2_in(y0[oku], np.column_stack([np.ones(oku.sum()), med_u[oku], med_u[oku] ** 2]))

            # bootstrap over bearings (stratified), all 67 outcomes recomputed per resample
            idx = j60.strat_index(cond, rng)
            boot = {m: np.full(B, np.nan) for m in MODELS}
            for b in range(B):
                yb = outcomes(idx[b])
                okb = np.isfinite(yb)
                if okb.sum() < MIN_DESIGNS:
                    continue
                for m in MODELS:
                    boot[m][b] = r2_in(yb[okb], design_matrix(m, lnom[okb], lfar[okb], fam_idx[okb]))
            delta = boot["M3"] - boot["M2"]

            # out of sample over stratified half-splits
            splits = stratified_splits(cond, rng)
            oos = {m: [] for m in MODELS}
            rels = []
            for k, Amask in enumerate(splits):
                ya, yb2 = outcomes(np.where(Amask)[0]), outcomes(np.where(~Amask)[0])
                ok = np.isfinite(ya) & np.isfinite(yb2)
                if ok.sum() < MIN_DESIGNS or len(set(fam_idx[ok])) < 2:
                    continue
                rel = float(np.corrcoef(ya[ok], yb2[ok])[0, 1])
                rels.append(rel)
                row = {"protocol": proto, "set": set_name, "dataset": ds, "split": k, "n_designs": int(ok.sum()),
                       "reliability": rel}
                for m in MODELS:
                    Xa = design_matrix(m, lnom[ok], lfar[ok], fam_idx[ok])
                    v = float(np.nanmean([r2_oos(ya[ok], Xa, yb2[ok], Xa), r2_oos(yb2[ok], Xa, ya[ok], Xa)]))
                    oos[m].append(v)
                    row[f"r2_oos_{m}"] = v
                split_rows.append(row)
            share_m2_beats_m0 = float(np.mean(np.array(oos["M2"]) > np.array(oos["M0"]))) if oos["M2"] else np.nan

            def q(v, p):
                v = np.asarray(v, float)
                v = v[np.isfinite(v)]
                return float(np.percentile(v, p)) if len(v) else np.nan

            for m in MODELS:
                mod_rows.append({"protocol": proto, "set": set_name, "dataset": ds, "model": m,
                                 "n_bearings": len(trajs), "n_designs": int(ok0.sum()), "r2_in": ins[m],
                                 "r2_boot_median": q(boot[m], 50), "r2_boot_lo": q(boot[m], 2.5), "r2_boot_hi": q(boot[m], 97.5),
                                 "r2_oos_median": q(oos[m], 50), "r2_oos_q25": q(oos[m], 25), "r2_oos_q75": q(oos[m], 75),
                                 "n_splits": len(oos[m]), "total_splits": len(splits)})
            dlo, dhi = q(delta, 2.5), q(delta, 97.5)
            rel_med = float(np.median(rels)) if rels else np.nan
            r1 = bool(np.isfinite(dlo) and dlo <= 0 <= dhi and share_m2_beats_m0 >= 0.75)
            r2 = bool(np.isfinite(dlo) and dlo > 0)
            r3 = all(q(oos[m], 50) < 0 for m in ["M0", "M1", "M2", "M3"])
            mod_rows.append({"protocol": proto, "set": set_name, "dataset": ds, "model": "M3-M2",
                             "n_bearings": len(trajs), "n_designs": int(ok0.sum()), "r2_in": ins["M3"] - ins["M2"],
                             "r2_boot_median": q(delta, 50), "r2_boot_lo": dlo, "r2_boot_hi": dhi,
                             "share_m2_beats_m0_oos": share_m2_beats_m0, "reliability_median": rel_med,
                             "ceiling_r2": rel_med ** 2 if np.isfinite(rel_med) else np.nan,
                             "r2_in_size_measured_d": ins_size, "r2_in_onset_fraction": ins_frac,
                             "reading_1": r1, "reading_2": r2, "reading_3": r3})
            log(f"\n  {proto:13s} {set_name:4s} {DS_NAME[ds]:7s}: {len(trajs)} bearings, {int(ok0.sum())} designs with an outcome, "
                f"{len(oos['M0'])} of {len(splits)} splits usable")
            for m in MODELS:
                r = mod_rows[-1 - (len(MODELS) - MODELS.index(m))]
                log(f"    {m}: R2 in {r['r2_in']:+.3f}, boot {r['r2_boot_median']:+.3f} [{r['r2_boot_lo']:+.3f}, "
                    f"{r['r2_boot_hi']:+.3f}], out of sample median {r['r2_oos_median']:+.3f} "
                    f"[q25 {r['r2_oos_q25']:+.3f}, q75 {r['r2_oos_q75']:+.3f}]")
            log(f"    Delta R2 (M3 - M2): in {ins['M3'] - ins['M2']:+.3f}, boot {q(delta, 50):+.3f} [{dlo:+.3f}, {dhi:+.3f}]; "
                f"M2 beats M0 out of sample in {share_m2_beats_m0:.2f} of splits")
            log(f"    split-half reliability median {rel_med:.3f} (ceiling R2 {rel_med ** 2 if np.isfinite(rel_med) else np.nan:.3f}); "
                f"mediators in sample: measured d {ins_size:.3f}, onset fraction {ins_frac:.3f}")
            log(f"    readings: (1) calibration beats family {r1}; (2) family adds beyond FAR {r2}; "
                f"(3) every model below 0 out of sample {r3}")

    pd.DataFrame(out_rows).to_csv(T / "j64_design_outcomes.csv", index=False)
    pd.DataFrame(mod_rows).to_csv(T / "j64_models.csv", index=False)
    pd.DataFrame(split_rows).to_csv(T / "j64_splits.csv", index=False)
    log(f"\nwrote j64_design_outcomes.csv ({len(out_rows)}), j64_models.csv ({len(mod_rows)}), j64_splits.csv ({len(split_rows)})")
    log(f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
