"""J39 -- a two-stage prognostic model of the form used in [2] and [9], scored under J35's protocol (B2, 2026-09-21).

The objection J35 invites: six ordinary regressors on per-record statistics are not prognostic models. The model
family the paper cites for the two-stage split is. Li et al. [2] (IEEE TIE 2015) fit an exponential degradation
model after the first prediction time; Wang et al. [9] (IEEE T-Rel 2020, the XJTU-SY paper) detect the start of
degradation with the method of [2], fit exponential degradation models, and estimate the RUL "by extrapolating the
degradation curve to reach a specified failure threshold" (excerpt kept in workflow_verification/sources/novelty/).
This script scores the plain form of that model -- no particle filter as in [2], no relevance-vector representation
as in [9] -- under exactly J35's protocol: the same 14 XJTU-SY and 9 FEMTO bearings, the same four labellers, the
same target (Eq. (1) of J35, Eq. (2) of the IEEE version), leave-one-bearing-out.

THE MODEL, for a held-out bearing j under labeller L with onset i*:
  * stage 1, t < i*        : y_hat = 1 (healthy; no prediction is made before the first prediction time)
  * stage 2, t >= i*       : fit a trend to the RMS of records i*..t (the records seen so far after the onset),
                             extrapolate it to the failure threshold theta, and take the first hitting record T_hat
                             as the estimate of the last record n-1. The target of Eq. (1) with n-1 replaced by
                             T_hat is y_hat = (T_hat - t) / (T_hat - i*), clipped to [0, 1].
      - 'exp': log RMS linear in time (an exponential trend, the form of [2] and [9]);
      - 'lin': RMS linear in time (the fallback, used if the exponential form is unstable on short bearings).
      - fewer than MIN_FIT records since the onset: y_hat = 1 (nothing to extrapolate yet);
        fitted value at t already at or above theta: y_hat = 0; non-positive slope: y_hat = 1 (never hits).
  * RMS is divided by its mean over the baseline window that every onset rule uses (fig1_fpt_matrix
    baseline_records, about the first 20% of records; identical for all rules on a trajectory), so bearings
    of different operating conditions share a scale.
  * theta, the only quantity learned across bearings: the median, over the OTHER usable bearings, of the last
    record's scaled RMS (the level at which their tests were stopped). Leave-one-bearing-out, like every J35 model.
  MIN_FIT = 5 was fixed before any result was seen and is not tuned.

SCORING, as J35: pooled RMSE and MAE over every record of the usable bearings, the skill score against J35's
constant baseline (the mean target of the other bearings), and a 95% bearing-bootstrap interval. The comparison
the paper states -- does the labeller move this model's error more than the choice between this model and the
six regressors? -- is read three ways: the two-stage label spread against the seven-model spread with the labeller
fixed, the seven-model factor as J35 computes it, and both over the same 20,000 bearing resamples as J38 (seed 0,
drawn in the same order, so the resamples are identical).

Stability of the exponential form (the user's rule: if unstable on short bearings, use the linear one):
reported as the share of consecutive stage-2 predictions that jump by more than 0.5, overall and on the
shortest third of bearings, for both variants.

Inputs : as J35 (features, records, fig1_fpt_matrix.csv), plus j35_downstream_rul.csv and j35_per_bearing.csv
Outputs: results/tables/j39_two_stage.csv            dataset x variant x labeller: RMSE, MAE, skill, CI
         results/tables/j39_two_stage_per_bearing.csv
         results/tables/j39_two_stage_compare.csv     the labeller-versus-model comparison with the six regressors
         results/logs/j39_run.log
Run    : python scripts/J39_two_stage_rul.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES, LOGS = ROOT / "results" / "tables", ROOT / "results" / "logs"
SEED, N_BOOT = 0, 20000
MIN_FIT = 5
VARIANTS = ("exp", "lin")
LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


def load_j35():
    spec = importlib.util.spec_from_file_location("j35", ROOT / "scripts" / "J35_downstream_rul.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def two_stage(r, i_star, theta, kind):
    """r: scaled RMS of one bearing in record order. Returns y_hat for every record."""
    n = len(r)
    yhat = np.ones(n)
    if i_star >= n:
        return yhat
    x = np.log(r[i_star:]) if kind == "exp" else r[i_star:].astype(float)
    lt = np.log(theta) if kind == "exp" else theta
    u = np.arange(len(x), dtype=float)                   # time since the onset, in records
    m = u + 1.0                                          # number of records fitted at each t
    su, suu = np.cumsum(u), np.cumsum(u * u)
    sx, sux = np.cumsum(x), np.cumsum(u * x)
    den = m * suu - su * su
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (m * sux - su * sx) / den
        a = (sx - b * su) / m
        f_now = a + b * u                                # fitted value at the current record
        u_hit = (lt - a) / b                             # first hitting time, records after the onset
        y2 = 1.0 - u / u_hit
    y2 = np.where(b > 0, y2, 1.0)
    y2 = np.where(f_now >= lt, 0.0, y2)
    y2 = np.where(m < MIN_FIT, 1.0, y2)
    yhat[i_star:] = np.clip(y2, 0.0, 1.0)
    if not np.isfinite(yhat).all():
        raise SystemExit("non-finite prediction")
    return yhat


def spreads(v):
    """v: (labellers, models, ...) -> median label spread, median model spread (as J35)."""
    lab_sp = v.max(axis=0) - v.min(axis=0)
    mod_sp = v.max(axis=1) - v.min(axis=1)
    return np.median(lab_sp, axis=0), np.median(mod_sp, axis=0)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    J35 = load_j35()
    LAB, MODELS = J35.LABELLERS, J35.MODEL_ORDER
    ft = pd.read_csv(TABLES / "fig1_fpt_matrix.csv")
    pooled = pd.read_csv(TABLES / "j35_downstream_rul.csv")
    pb35 = pd.read_csv(TABLES / "j35_per_bearing.csv")
    rng = np.random.default_rng(SEED)                    # same seed and draw order as J38 -> same resamples

    log("J39 -- two-stage model of the form of [2], [9], under J35's protocol")
    log(f"stage 1 y=1 before the onset; stage 2 trend on RMS after it, extrapolated to a threshold learned "
        f"leave-one-bearing-out; MIN_FIT={MIN_FIT}")
    log("=" * 104)
    rows, per_rows, cmp_rows = [], [], []
    for ds in J35.DATASETS:
        traj_all, F_all = J35.load_dataset(ds)
        g = ft[ft.dataset == ds]
        fired = {r: set(g[(g.rule == r) & g.fired].trajectory) for r in LAB}
        usable = sorted(set.intersection(*fired.values()))
        if sorted(pb35[pb35.dataset == ds].bearing.unique()) != usable:
            raise SystemExit(f"{ds}: usable bearings differ from J35's")
        info = g.drop_duplicates("trajectory").set_index("trajectory")
        onset = {(r.rule, r.trajectory): int(r.fpt_index) for r in g.itertuples() if r.fired}

        # RMS per bearing in record order, checked against the onset table's record counts
        rms_all = F_all["rms"].to_numpy(dtype=float)
        rms, scaled, last = {}, {}, {}
        for b in usable:
            x = rms_all[traj_all == b]
            if len(x) != int(info.loc[b, "n_records"]):
                raise SystemExit(f"{ds}/{b}: {len(x)} feature rows, onset table says {info.loc[b, 'n_records']}")
            base = x[: int(info.loc[b, "baseline_records"])].mean()
            rms[b], scaled[b] = x, x / base
            last[b] = scaled[b][-1]
        n_of = {b: len(rms[b]) for b in usable}
        log(f"{J35.DS_NAME[ds]}: {len(usable)} bearings, {sum(n_of.values())} records; last-record scaled RMS "
            f"median {np.median(list(last.values())):.2f} (range {min(last.values()):.2f}-{max(last.values()):.2f})")

        # targets and J35's constant baseline, per labeller
        Y = {lab: {b: J35.target(n_of[b], onset[(lab, b)]) for b in usable} for lab in LAB}
        base_sse = {}
        for lab in LAB:
            tot = sum(Y[lab][b].sum() for b in usable)
            cnt = sum(n_of[b] for b in usable)
            for b in usable:
                mean_other = (tot - Y[lab][b].sum()) / (cnt - n_of[b])
                base_sse[(lab, b)] = float(((Y[lab][b] - mean_other) ** 2).sum())
            j = pooled[(pooled.dataset == ds) & (pooled.labeller == lab)].rmse_baseline.iloc[0]
            mine = np.sqrt(sum(base_sse[(lab, b)] for b in usable) / cnt)
            if abs(mine - j) > 1e-9:
                raise SystemExit(f"{ds}/{lab}: constant baseline {mine} != J35 {j}")

        # predictions, leave-one-bearing-out threshold
        sse, sae = {}, {}
        post = {}          # (variant, labeller) -> signed errors on records at or after the onset
        flips = {v: [] for v in VARIANTS}
        short = set(sorted(usable, key=lambda b: n_of[b])[: max(1, len(usable) // 3)])
        for v in VARIANTS:
            for lab in LAB:
                for b in usable:
                    theta = float(np.median([last[o] for o in usable if o != b]))
                    i_star = onset[(lab, b)]
                    yh = two_stage(scaled[b], i_star, theta, v)
                    e = yh - Y[lab][b]
                    sse[(v, lab, b)] = float((e ** 2).sum())
                    sae[(v, lab, b)] = float(np.abs(e).sum())
                    post.setdefault((v, lab), []).append(e[i_star:])
                    per_rows.append({"dataset": ds, "variant": v, "labeller": lab, "bearing": b,
                                     "n_records": n_of[b], "onset": i_star, "theta": theta,
                                     "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean())})
                    d2 = np.abs(np.diff(yh[i_star + MIN_FIT - 1:])) if n_of[b] - i_star > MIN_FIT else np.array([])
                    flips[v].append((b in short, int((d2 > 0.5).sum()), len(d2)))
            for lab in LAB:
                s = sum(sse[(v, lab, b)] for b in usable)
                cnt = sum(n_of[b] for b in usable)
                bs = sum(base_sse[(lab, b)] for b in usable)
                ep = np.concatenate(post[(v, lab)])
                rows.append({"dataset": ds, "variant": v, "labeller": lab, "n_bearings": len(usable),
                             "n_records": cnt, "rmse": float(np.sqrt(s / cnt)),
                             "mae": float(sum(sae[(v, lab, b)] for b in usable) / cnt),
                             "skill": float(1.0 - s / bs),
                             # where the error sits: after the onset, predicting too long a life (> 0) or too short
                             "post_onset_records": int(len(ep)),
                             "post_onset_mean_signed_error": float(ep.mean()),
                             "post_onset_share_over_0p25": float((ep > 0.25).mean()),
                             "post_onset_share_under_0p25": float((ep < -0.25).mean()),
                             "share_of_sse_after_onset": float((ep ** 2).sum() / s) if s > 0 else float("nan")})
            tot_j = sum(f[1] for f in flips[v]); tot_n = sum(f[2] for f in flips[v])
            sh_j = sum(f[1] for f in flips[v] if f[0]); sh_n = sum(f[2] for f in flips[v] if f[0])
            log(f"  [{v}] jumps > 0.5 between consecutive stage-2 predictions: {tot_j} of {tot_n} "
                f"({100 * tot_j / max(tot_n, 1):.2f}%); on the shortest third of bearings {sh_j} of {sh_n} "
                f"({100 * sh_j / max(sh_n, 1):.2f}%)")
        # ---- comparison with the six regressors, point and over the J38 resamples
        B = len(usable)
        idx = rng.integers(0, B, size=(N_BOOT, B))
        C = np.zeros((N_BOOT, B))
        np.add.at(C, (np.arange(N_BOOT)[:, None], idx), 1.0)
        cnt_vec = np.array([n_of[b] for b in usable], dtype=float)
        sse6 = np.zeros((len(LAB), len(MODELS), B))
        for li, lab in enumerate(LAB):
            for mi, mname in enumerate(MODELS):
                x = pb35[(pb35.dataset == ds) & (pb35.labeller == lab) & (pb35.model == mname)]
                x = x.set_index("bearing").reindex(usable)
                sse6[li, mi] = x.rmse.to_numpy() ** 2 * x.n_records.to_numpy()
        for v in VARIANTS:
            sse2 = np.array([[sse[(v, lab, b)] for b in usable] for lab in LAB])[:, None, :]
            sse7 = np.concatenate([sse6, sse2], axis=1)                 # (labellers, 7 models, B)
            pt = np.sqrt(sse7.sum(-1) / cnt_vec.sum())                  # (labellers, 7)
            bt = np.sqrt(np.einsum("lmb,kb->lmk", sse7, C) / (C @ cnt_vec))
            two = pt[:, -1]
            ls2 = two.max() - two.min()                                  # two-stage fit, labeller varied
            ms7 = pt.max(axis=1) - pt.min(axis=1)                        # per labeller, over the 7 models
            dev = np.abs(pt[:, :-1] - pt[:, -1:])                        # |two-stage - regressor|, per labeller
            lp, mp = spreads(pt)
            ls2_b = bt[:, -1].max(axis=0) - bt[:, -1].min(axis=0)
            ms7_b = bt.max(axis=1) - bt.min(axis=1)                      # (labellers, K)
            lp_b, mp_b = spreads(bt)
            f7_b = lp_b / mp_b
            rec = {"dataset": ds, "variant": v,
                   "two_stage_label_spread": float(ls2),
                   "seven_model_spread_median": float(np.median(ms7)),
                   "seven_model_spread_max": float(ms7.max()),
                   "max_abs_two_stage_minus_regressor": float(dev.max()),
                   "factor7": float(lp / mp), "factor7_lo": float(np.percentile(f7_b, 2.5)),
                   "factor7_hi": float(np.percentile(f7_b, 97.5)),
                   "share_ls2_gt_median_ms7": float((ls2_b > np.median(ms7_b, axis=0)).mean()),
                   "share_ls2_gt_max_ms7": float((ls2_b > ms7_b.max(axis=0)).mean()),
                   "share_factor7_gt1": float((f7_b > 1).mean())}
            for li, lab in enumerate(LAB):
                rec[f"seven_model_spread_{lab}"] = float(ms7[li])
            cmp_rows.append(rec)

        d = pd.DataFrame([r for r in rows if r["dataset"] == ds])
        log("")
        log(f"  {'':22s}" + "".join(f"{lab:>18s}" for lab in LAB))
        for mname in MODELS:
            x = pooled[(pooled.dataset == ds) & (pooled.model == mname)].set_index("labeller")
            log(f"  {mname:22s}" + "".join(f"{x.loc[lab, 'rmse']:>9.3f} ({x.loc[lab, 'skill']:+.2f})" for lab in LAB))
        for v in VARIANTS:
            x = d[d.variant == v].set_index("labeller")
            log(f"  {'two-stage ' + v:22s}" + "".join(f"{x.loc[lab, 'rmse']:>9.3f} ({x.loc[lab, 'skill']:+.2f})"
                                                     for lab in LAB))
        for rec in [c for c in cmp_rows if c["dataset"] == ds]:
            log(f"  [{rec['variant']}] two-stage label spread {rec['two_stage_label_spread']:.3f}; "
                f"spread over the 7 models with the labeller fixed: "
                + ", ".join(f"{lab} {rec[f'seven_model_spread_{lab}']:.3f}" for lab in LAB)
                + f" (median {rec['seven_model_spread_median']:.3f}, max {rec['seven_model_spread_max']:.3f}); "
                f"largest |two-stage - regressor| {rec['max_abs_two_stage_minus_regressor']:.3f}")
            log(f"        seven-model factor {rec['factor7']:.2f} [{rec['factor7_lo']:.2f}, {rec['factor7_hi']:.2f}], "
                f"> 1 in {100 * rec['share_factor7_gt1']:.2f}% of resamples; two-stage label spread above the "
                f"median seven-model spread in {100 * rec['share_ls2_gt_median_ms7']:.2f}% and above the largest "
                f"in {100 * rec['share_ls2_gt_max_ms7']:.2f}%")
        log("")

    pd.DataFrame(rows).to_csv(TABLES / "j39_two_stage.csv", index=False)
    pd.DataFrame(per_rows).to_csv(TABLES / "j39_two_stage_per_bearing.csv", index=False)
    pd.DataFrame(cmp_rows).to_csv(TABLES / "j39_two_stage_compare.csv", index=False)
    log("written: j39_two_stage.csv, j39_two_stage_per_bearing.csv, j39_two_stage_compare.csv")
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "j39_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
