"""J49 -- two supplementary figures the revision asks for: the censoring treatments and the LOBO split.

  Fig. S5, censoring. A trajectory on which a rule never fires can be handled three ways, and the paper
    reports the consequences of all three in different places. They are collected here.
      A  exclude it, and pool the positive rate over the firing trajectories only
      B  label it healthy and keep it, which is what a study that does not say otherwise does
      C  treat it as right-censored and estimate the onset distribution by Kaplan-Meier
    A and B are record-pooled and differ only in the denominator, so they belong on one axis; C is an
    estimate over trajectories, so it is drawn against A on the same trajectory footing rather than
    mixed with the record-pooled pair. Forcing all three onto one axis would compare three different
    estimands. A is checked on both footings against j17_prevalence_by_subset.csv, which computes
    it independently; B has no second source, being A with the non-firing records back in the
    denominator.

  Fig. S6, the LOBO split. The downstream protocol holds out whole bearings, and the labeler changes the
    target of every record but not the split. The figure says that in one picture, because every
    downstream number in the paper depends on it and no figure showed it.

Inputs : results/tables/fig1_fpt_matrix.csv          onsets, record counts and firing flags
         results/tables/j28_km_onset.csv             the Kaplan-Meier estimates
         results/tables/j17_prevalence_by_subset.csv the independent check on B
Outputs: results/figures/fig8_censoring_abc.pdf + .png
         results/figures/fig9_lobo_schematic.pdf + .png
         results/tables/j49_censoring_abc.csv        the numbers drawn, so the gate can read them
         results/logs/j49_run.log

Usage: python scripts/J49_supplementary_figures.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import Rectangle             # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES, FIGS, LOGS = ROOT / "results" / "tables", ROOT / "results" / "figures", ROOT / "results" / "logs"

FAM_COL = {"detector": "#1B6773", "control chart": "#B85C18", "percentile": "#7A8D98"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL, SURFACE = "#e1e0d9", "#c3c2b7", "#f0efec", "#ffffff"
W, DPI = 5.4, 300
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
RULES = ["3sig_rms", "ma3sig", "ma5sig", "ma2sig", "3sig_kurt", "ewma", "cusum", "pct02", "pct05", "pct10"]
LOG: list[str] = []


def log(s: str = "") -> None:
    print(s)
    LOG.append(s)


def tidy(ax, grid_axis=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, length=2, width=0.5, pad=1.5)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    log(f"  wrote results/figures/{name}.pdf + .png")


def abc_table() -> pd.DataFrame:
    """A, B and C for every dataset and rule, each from the table that defines it."""
    ft = pd.read_csv(TABLES / "fig1_fpt_matrix.csv")
    km = pd.read_csv(TABLES / "j28_km_onset.csv").set_index(["dataset", "rule"])
    prev = pd.read_csv(TABLES / "j17_prevalence_by_subset.csv").query("subset == 'all'") \
        .set_index(["dataset", "rule"])
    rows = []
    for ds in ("xjtu", "femto", "ims"):
        g = ft[ft.dataset == ds]
        for rule in RULES:
            r = g[g.rule == rule]
            if not len(r):
                continue
            fired = r[r.fired]
            pos = (r.n_records - r.fpt_index).where(r.fired, 0.0)
            a = float(pos[r.fired].sum() / fired.n_records.sum()) if len(fired) else np.nan
            b = float(pos.sum() / r.n_records.sum())
            row = {"dataset": ds, "rule": rule, "n_traj": int(len(r)), "n_fired": int(len(fired)),
                   "a_exclude_pooled": a, "b_healthy_pooled": b}
            if (ds, rule) in km.index:
                row["a_exclude_traj"] = float(km.loc[(ds, rule)].naive_pos_rate_traj)
                row["c_censored_traj"] = float(km.loc[(ds, rule)].km_pos_rate)
            if (ds, rule) in prev.index:                      # two independent computations of A
                row["a_check_j17"] = float(prev.loc[(ds, rule)].pos_rate)
                row["a_traj_check_j17"] = float(prev.loc[(ds, rule)].per_bearing_mean)
            rows.append(row)
    d = pd.DataFrame(rows)
    for mine, theirs, what in (("a_exclude_pooled", "a_check_j17", "record-pooled"),
                               ("a_exclude_traj", "a_traj_check_j17", "per-trajectory")):
        g = d.dropna(subset=[mine, theirs])
        worst = float((g[mine] - g[theirs]).abs().max())
        log(f"  A ({what}) recomputed here against j17_prevalence_by_subset.csv on {len(g)} cells: "
            f"max |difference| {worst:.2e}")
        if worst > 1e-9:
            raise SystemExit(f"A ({what}) does not agree with the prevalence table")
    return d


def fig_censoring(d: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), facecolor=SURFACE)
    ds_name = {"xjtu": "XJTU-SY", "femto": "FEMTO", "ims": "IMS"}
    mark = {"xjtu": "o", "femto": "s", "ims": "^"}

    ax = axes[0]
    for ds, m in mark.items():
        g = d[(d.dataset == ds) & d.b_healthy_pooled.notna()]
        ax.scatter(g.a_exclude_pooled, g.b_healthy_pooled, s=14, marker=m, facecolor="none",
                   edgecolor=FAM_COL["detector"], linewidth=0.8, label=ds_name[ds])
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("A: pooled over firing trajectories")
    ax.set_ylabel("B: non-firing labeled healthy")
    ax.set_title("Record-pooled positive rate", color=INK)
    ax.legend(frameon=False, loc="upper left")
    tidy(ax, "both")

    ax = axes[1]
    g = d.dropna(subset=["c_censored_traj"])
    for ds, m in mark.items():
        h = g[g.dataset == ds]
        ax.scatter(h.a_exclude_traj, h.c_censored_traj, s=14, marker=m, facecolor="none",
                   edgecolor=FAM_COL["control chart"], linewidth=0.8, label=ds_name[ds])
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("A: over firing trajectories")
    ax.set_ylabel("C: Kaplan--Meier")
    ax.set_title("Trajectory-based positive rate", color=INK)
    tidy(ax, "both")

    fig.tight_layout(pad=0.6)
    save(fig, "fig8_censoring_abc")


def fig_lobo() -> None:
    ft = pd.read_csv(TABLES / "fig1_fpt_matrix.csv")
    x = ft[ft.dataset == "xjtu"]
    # the bearings the downstream experiment uses: every one of its four labellers fires on them
    LAB = ["3sig_rms", "ma5sig", "ewma", "pct05"]
    fire = {r: set(x[(x.rule == r) & x.fired].trajectory) for r in LAB}
    bearings = sorted(set.intersection(*fire.values()))
    n = len(bearings)
    g = x[x.rule == "3sig_rms"]

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.15), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [2.05, 1.0]})
    ax = axes[0]
    for f in range(n):
        for b in range(n):
            held = (b == f)
            ax.add_patch(Rectangle((f, n - 1 - b), 1, 1,
                                   facecolor=FAM_COL["control chart"] if held else NEUTRAL,
                                   edgecolor="white", linewidth=0.5))
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xlabel(f"fold, one per bearing ({n} usable on XJTU-SY)")
    ax.set_ylabel("bearing")
    ax.set_xticks([0.5, n - 0.5])
    ax.set_xticklabels(["1", str(n)])
    ax.set_yticks([0.5, n - 0.5])
    ax.set_yticklabels([str(n), "1"])
    ax.set_title("Leave one bearing out: held out (dark), trained on (light)", color=INK)
    tidy(ax)

    ax = axes[1]
    b = bearings[0]
    n_rec = int(x[(x.rule == "3sig_rms") & (x.trajectory == b)].n_records.iloc[0])
    t = np.arange(n_rec) / (n_rec - 1)
    for rule, col in (("3sig_rms", FAM_COL["detector"]), ("ewma", FAM_COL["control chart"])):
        onset = float(x[(x.rule == rule) & (x.trajectory == b)].fpt_fraction.iloc[0])
        y = np.where(t < onset, 1.0, (1.0 - t) / max(1e-9, 1.0 - onset))
        ax.plot(t, y, color=col, linewidth=1.0, label=rule)   # matplotlib is not LaTeX; no escaping
        ax.axvline(onset, color=col, linewidth=0.6, linestyle=(0, (2, 2)))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel("fraction of life")
    ax.set_ylabel("target")
    ax.set_title("The labeler moves the target,\nnot the split", color=INK)
    ax.legend(frameon=False, loc="lower left")
    tidy(ax, "y")

    fig.tight_layout(pad=0.6)
    save(fig, "fig9_lobo_schematic")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    LOGS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    log("J49 -- the censoring figure and the leave-one-bearing-out schematic")
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}")
    d = abc_table()
    d.to_csv(TABLES / "j49_censoring_abc.csv", index=False)
    log(f"  wrote j49_censoring_abc.csv, {len(d)} dataset-rule cells")
    fig_censoring(d)
    fig_lobo()
    log(f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    (LOGS / "j49_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
