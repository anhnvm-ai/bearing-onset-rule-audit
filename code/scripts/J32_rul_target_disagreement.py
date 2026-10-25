"""J32 -- how far apart the remaining-useful-life targets built from different onset rules are (N1, revision).

A two-stage prognostic model is trained on a piecewise-linear target: 1 before the first prediction time and falling
linearly to 0 at the last record,
    y_t = 1                          for t < i*
    y_t = (n - 1 - t) / (n - 1 - i*)  for t >= i*        (t = 0, ..., n - 1)
the form used by Kim and Park (kim2025antiselfhealing). Every rule gives its own i*, and so its own target. For every
pair of rules and every trajectory on which both fire, this script compares the two targets record by record. The
comparison needs no model: a model that reproduced one rule's target perfectly would score this error against the other.

revision (C2): the pooled per-pair numbers are weighted by trajectory length, so on XJTU-SY, where three
bearings hold 71% of the records, a reader cannot tell whether a pooled 0.433 describes the bearings or the pool.
The same quantity is therefore also kept per bearing, D_j = sqrt(mean_t (y_a - y_b)^2), and each comparison class
is summarised by the median, the interquartile range and a 95% interval over 20,000 resamples of whole bearings.

Inputs : results/tables/fig1_fpt_matrix.csv (onsets of the ten rules on the 33 trajectories)
Outputs: results/tables/j32_rul_target_pairs.csv       one row per dataset and rule pair, pooled over records
         results/tables/j32_rul_target_summary.csv     median over the pairs of each comparison class
         results/tables/j32_rul_target_spread.csv      per trajectory: spread of the target across the signal rules
         results/tables/j32_rul_target_per_traj.csv    D_j for every dataset x rule pair x trajectory   (C2)
         results/tables/j32_rul_target_bearing_ci.csv  per class: median, IQR and bearing-bootstrap CI  (C2)
         results/tables/j32_rul_target_duration.csv    D_j against trajectory length, per class          (C2)
         results/figures/fig9_rul_target.pdf/.png
         results/logs/j32_run.log
"""
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("j14", ROOT / "scripts" / "J14_journal_figures.py")
J14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J14)
plt = J14.plt
T, LOGS = J14.TABLES, J14.LOGS
RMS3, BETWEEN, CHARTS, PCT = J14.RMS3, J14.BETWEEN, ["ewma", "cusum"], J14.PCT
ORDER = RMS3 + BETWEEN + CHARTS + PCT
SIGNAL = RMS3 + BETWEEN + CHARTS
LOG = []


def log(s=""):
    print(s)
    LOG.append(s)


def target(n, i):
    t = np.arange(n, dtype=float)
    y = np.ones(n)
    if i is None:
        return None
    i = int(i)
    after = t >= i
    y[after] = (n - 1 - t[after]) / max(n - 1 - i, 1)
    return y


def cls(a, b):
    s = {a, b}
    if s <= set(RMS3):
        return "within RMS detectors"
    if s <= set(CHARTS):
        return "within charts"
    if s <= set(PCT):
        return "within percentile"
    if (a in CHARTS and b in RMS3) or (b in CHARTS and a in RMS3):
        return "chart vs RMS detector"
    if (a in PCT) != (b in PCT):
        return "percentile vs signal rule"
    return "other"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ft = pd.read_csv(T / "fig1_fpt_matrix.csv")
    tg = {}
    for r in ft.itertuples():
        tg[(r.dataset, r.trajectory, r.rule)] = target(int(r.n_records), r.fpt_index if r.fired else None)

    rows, traj_rows = [], []
    for ds, g in ft.groupby("dataset", sort=False):
        trajs = sorted(g.trajectory.unique())
        for a, b in itertools.combinations(ORDER, 2):
            d_all, both, excl = [], 0, 0
            for tr in trajs:
                ya, yb = tg[(ds, tr, a)], tg[(ds, tr, b)]
                if ya is None or yb is None:
                    excl += 1
                    continue
                both += 1
                d_all.append(np.abs(ya - yb))
                # revision (C2): the pooled rmse below is one number per rule pair, weighted by trajectory
                # length, so a reader cannot tell whether it is carried by the longest bearings. Keep the same
                # quantity per bearing, D_j = sqrt(mean_t (y_a - y_b)^2), so the pair can be described by a
                # distribution over bearings with an interval rather than by a single pooled point.
                traj_rows.append({"dataset": ds, "trajectory": tr, "rule_a": a, "rule_b": b, "class": cls(a, b),
                                 "n_records": int(len(ya)),
                                 "rmse": float(np.sqrt(((ya - yb) ** 2).mean())),
                                 "mean_abs_diff": float(np.abs(ya - yb).mean()),
                                 "share_gt_025": float((np.abs(ya - yb) > 0.25).mean())})
            if not d_all:
                continue
            d = np.concatenate(d_all)
            per_traj = [x.mean() for x in d_all]
            rows.append({"dataset": ds, "rule_a": a, "rule_b": b, "class": cls(a, b), "n_traj_both": both,
                         "n_traj_excluded": excl, "n_records": len(d), "mean_abs_diff": d.mean(),
                         "rmse": float(np.sqrt((d ** 2).mean())), "share_gt_010": float((d > 0.10).mean()),
                         "share_gt_025": float((d > 0.25).mean()), "median_traj_mean_abs": float(np.median(per_traj))})
    pairs = pd.DataFrame(rows)
    pairs.to_csv(T / "j32_rul_target_pairs.csv", index=False)

    summ = (pairs.groupby(["dataset", "class"])
            .agg(n_pairs=("rule_a", "size"), mean_abs_diff=("mean_abs_diff", "median"), rmse=("rmse", "median"),
                 share_gt_010=("share_gt_010", "median"), share_gt_025=("share_gt_025", "median"),
                 n_traj_excluded=("n_traj_excluded", "sum"))
            .reset_index())
    summ.to_csv(T / "j32_rul_target_summary.csv", index=False)

    spread = []
    for (ds, tr), g in ft.groupby(["dataset", "trajectory"]):
        ys = [tg[(ds, tr, r)] for r in SIGNAL if tg[(ds, tr, r)] is not None]
        if len(ys) < 2:
            continue
        Y = np.vstack(ys)
        rng = Y.max(0) - Y.min(0)
        spread.append({"dataset": ds, "trajectory": tr, "n_records": Y.shape[1], "n_signal_rules_fired": len(ys),
                       "mean_range": float(rng.mean()), "share_range_gt_025": float((rng > 0.25).mean()),
                       "share_range_gt_050": float((rng > 0.50).mean())})
    spread = pd.DataFrame(spread)
    spread.to_csv(T / "j32_rul_target_spread.csv", index=False)

    # ---- revision (C2): the same comparison read over bearings instead of over pooled records.
    # For each comparison class, every bearing contributes the median over the rule pairs in that class of its own
    # D_j. The distribution of those per-bearing numbers is then summarised by its median, its interquartile range
    # and a 95% interval from 20,000 resamples of whole bearings -- the paper's convention everywhere else.
    ptj = pd.DataFrame(traj_rows)
    ptj.to_csv(T / "j32_rul_target_per_traj.csv", index=False)
    boot_rng = np.random.default_rng(0)
    ci_rows = []
    for (ds, cl), gg in ptj.groupby(["dataset", "class"]):
        per_bearing = gg.groupby("trajectory").rmse.median()
        v = per_bearing.to_numpy(float)
        if len(v) < 2:
            continue
        boot = np.median(v[boot_rng.integers(0, len(v), size=(20000, len(v)))], axis=1)
        ci_rows.append({"dataset": ds, "class": cl, "n_bearings": len(v),
                        "n_pairs": int(gg.groupby(["rule_a", "rule_b"]).ngroups),
                        "median": float(np.median(v)),
                        "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
                        "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
                        "ci_lo": float(np.percentile(boot, 2.5)), "ci_hi": float(np.percentile(boot, 97.5)),
                        "min": float(v.min()), "max": float(v.max())})
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(T / "j32_rul_target_bearing_ci.csv", index=False)

    # ---- revision (C2): why the pooled number and the per-bearing median differ, and why that is a
    # prediction rather than an awkwardness. Section 4.2 finds that a detector's alarm record grows with the
    # length of the test while a chart's does not. Two targets built from rules on opposite sides of that
    # transition must therefore drift further apart the longer the test ran, and the pooled figure, which is
    # weighted by trajectory length, must exceed the median bearing. Both are checked here: D_j against n.
    dur_rows = []
    for (ds, cl), gg in ptj.groupby(["dataset", "class"]):
        b = gg.groupby("trajectory").agg(D=("rmse", "median"), n=("n_records", "first"))
        if len(b) < 4:
            continue
        nv, dv = b.n.to_numpy(float), b.D.to_numpy(float)
        rho = float(spearmanr(nv, dv)[0])
        idx = boot_rng.integers(0, len(b), size=(2000, len(b)))
        bs = np.array([spearmanr(nv[i], dv[i])[0] for i in idx])
        bs = bs[np.isfinite(bs)]
        long3 = b.nlargest(3, "n")
        dur_rows.append({"dataset": ds, "class": cl, "n_bearings": len(b),
                         "rho_n_D": rho,
                         "rho_lo": float(np.percentile(bs, 2.5)) if len(bs) else np.nan,
                         "rho_hi": float(np.percentile(bs, 97.5)) if len(bs) else np.nan,
                         "median_D_three_longest": float(long3.D.median()),
                         "median_D_rest": float(b.nsmallest(len(b) - 3, "n").D.median()),
                         "record_share_three_longest": float(long3.n.sum() / b.n.sum())})
    dur = pd.DataFrame(dur_rows)
    dur.to_csv(T / "j32_rul_target_duration.csv", index=False)

    log("J32 -- RUL-target disagreement between onset rules")
    log(f"pairs of rules x datasets: {len(pairs)}; rule-trajectory pairs that never fire: {int((~ft.fired).sum())}")
    for ds in ("xjtu", "femto", "ims"):
        log(f"\n{ds}")
        for r in summ[summ.dataset == ds].itertuples():
            log(f"  {r._2:28s} pairs {r.n_pairs:2d}  mean|d| {r.mean_abs_diff:.3f}  rmse {r.rmse:.3f}  "
                f">0.10 {r.share_gt_010:.3f}  >0.25 {r.share_gt_025:.3f}  excluded traj {r.n_traj_excluded}")
        s = spread[spread.dataset == ds]
        log(f"  spread over signal rules: median over bearings of mean range {s.mean_range.median():.3f}, "
            f"share of records with range > 0.25 {s.share_range_gt_025.median():.3f}, > 0.50 {s.share_range_gt_050.median():.3f}")
        c = ci[ci.dataset == ds]
        if len(c):
            log(f"  per-bearing D_j = sqrt(mean_t (y_a - y_b)^2), median over the pairs of the class:")
            for r in c.sort_values("median", ascending=False).itertuples():
                log(f"    {r._2:28s} bearings {r.n_bearings:2d}  pairs {r.n_pairs:2d}  "
                    f"median {r.median:.3f}  IQR [{r.q25:.3f}, {r.q75:.3f}]  "
                    f"95% CI [{r.ci_lo:.3f}, {r.ci_hi:.3f}]  range {r.min:.3f}-{r.max:.3f}")
    (LOGS / "j32_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")

    # ---- figure: one bearing's targets, and the pairwise RMS difference on XJTU-SY and FEMTO
    # 2026-09-22: 0.30 in taller, axes raised, so that the part labels (a), (b) fit centered BELOW each part
    fig = plt.figure(figsize=(J14.W, 3.05), facecolor=J14.SURFACE)
    ax0 = fig.add_axes([0.075, 0.30, 0.27, 0.60])
    ex_ds, ex_tr = "xjtu", "Bearing1_1"
    g = ft[(ft.dataset == ex_ds) & (ft.trajectory == ex_tr)]
    n = int(g.n_records.iloc[0])
    x = np.arange(n) / (n - 1)
    for rule, ls in (("3sig_rms", "-"), ("ma5sig", "--"), ("ewma", "-"), ("cusum", "--"), ("pct10", ":")):
        y = tg[(ex_ds, ex_tr, rule)]
        if y is None:
            continue
        ax0.plot(x, y, ls=ls, lw=1.0, color=J14.FAM_COL[J14.FAMILY[rule]], label=rule)
    ax0.set_xlabel("fraction of life")
    ax0.set_ylabel("RUL target")
    ax0.set_title(f"XJTU-SY {ex_tr.replace('_', '-')}", color=J14.INK, pad=3)
    ax0.legend(frameon=False, fontsize=5.5, loc="lower left", handlelength=2.2)
    for s in ("top", "right"):
        ax0.spines[s].set_visible(False)
    im = None
    for k, (ds, label) in enumerate((("xjtu", "XJTU-SY"), ("femto", "FEMTO"))):
        ax = fig.add_axes([0.42 + k * 0.245, 0.30, 0.22, 0.60])
        M = np.full((len(ORDER), len(ORDER)), np.nan)
        for r in pairs[pairs.dataset == ds].itertuples():
            i, j = ORDER.index(r.rule_a), ORDER.index(r.rule_b)
            M[i, j] = M[j, i] = r.rmse
        np.fill_diagonal(M, 0.0)
        im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=0.5)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER, rotation=60, ha="right", fontsize=5.0)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels(ORDER if k == 0 else [""] * len(ORDER), fontsize=5.0)
        ax.tick_params(length=0, colors=J14.INK2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(label, color=J14.INK, pad=3)
        for kk in (2.5, 4.5, 6.5):
            ax.axhline(kk, color=J14.SURFACE, lw=1.2)
            ax.axvline(kk, color=J14.SURFACE, lw=1.2)
    cax = fig.add_axes([0.895, 0.36, 0.012, 0.45])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("RMS difference of targets", color=J14.INK2, fontsize=6, labelpad=2)
    cb.ax.tick_params(labelsize=5.5, colors=J14.INK2, length=1.5, width=0.4)
    # IEEE captions call the parts of a multipart figure (a), (b) (Editorial Style Manual, "Figures"); the IEEE
    # Access template asks for the labels "centered below each subfigure in 8 point Times New Roman" (2026-09-22)
    lab = dict(fontsize=8, family="Times New Roman", color="black", ha="center", va="bottom")
    fig.text(0.075 + 0.27 / 2, 0.012, "(a)", **lab)
    fig.text((0.42 + 0.665 + 0.22) / 2, 0.012, "(b)", **lab)
    J14.save(fig, "fig9_rul_target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
