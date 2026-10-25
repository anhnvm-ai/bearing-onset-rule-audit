"""J25 -- figures for the reviewer-driven revision, drawn at text width in the J14 style.

  fig4_crossing_scale   the transition: rho(crossing time, trajectory length) against the crossing threshold k,
                        with the size of change at which each family alarms marked (j24_crossing_times, j22_shift_at_alarm)
  fig8_design_sweep    every design on one axis: measured response size against rho(alarm record, length)
  fig6_label_agreement  record-level agreement between the label sets of every pair of rules (j23_label_agreement)
  fig7_rank_heatmap     macro-F1 of every method under every usable rule, winner marked (j3_performance, mode=='fixed')
  fig0_estimand_ladder  schematic: physical initiation -> observable change -> size-dependent detectability ->
                        alarm time -> labelling boundary, with what each rule family delivers

Output: results/figures/<name>.pdf and .png
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("j14", ROOT / "scripts" / "J14_journal_figures.py")
J14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(J14)
plt = J14.plt
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

T = J14.TABLES
DS = [("xjtu", "XJTU-SY"), ("femto", "FEMTO")]
CHARTS, RMS3 = ["ewma", "cusum"], ["3sig_rms", "ma3sig", "ma5sig"]


def fig_crossing():
    cr = pd.read_csv(T / "j24_crossing_times.csv")
    sh = pd.read_csv(T / "j22_shift_at_alarm.csv")
    fig, axes = plt.subplots(1, 2, figsize=(J14.W, 2.3), facecolor=J14.SURFACE)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.34, wspace=0.26)
    for ax, (ds, label) in zip(axes, DS):
        J14.style_axes(ax, "y")
        g = cr[cr.dataset == ds]
        s = sh[sh.dataset == ds]
        cm = s[s.rule.isin(CHARTS)].shift_in_baseline_sd.median()
        dm = s[s.rule.isin(RMS3)].shift_in_baseline_sd.median()
        ax.axvspan(s[s.rule.isin(CHARTS)].shift_in_baseline_sd.quantile(0.25),
                   s[s.rule.isin(CHARTS)].shift_in_baseline_sd.quantile(0.75),
                   color=J14.FAM_COL["control chart"], alpha=0.12, lw=0)
        ax.axvspan(s[s.rule.isin(RMS3)].shift_in_baseline_sd.quantile(0.25),
                   min(s[s.rule.isin(RMS3)].shift_in_baseline_sd.quantile(0.75), 19.5),
                   color=J14.FAM_COL["detector"], alpha=0.12, lw=0)
        ax.axhline(0, color=J14.INK2, lw=0.6, zorder=1)
        ax.plot(g.k_sd, g.rho_index_length, "-o", ms=3.5, lw=1.1, color=J14.INK, zorder=3)
        ax.axvline(cm, color=J14.FAM_COL["control chart"], lw=0.9, ls="--", zorder=2)
        ax.axvline(dm, color=J14.FAM_COL["detector"], lw=0.9, ls="--", zorder=2)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 6, 12])
        ax.set_xticklabels(["1", "2", "3", "6", "12"])
        ax.minorticks_off()
        ax.set_xlim(0.85, 20)
        ax.set_ylim(-0.35, 1.05)
        ax.set_title(label, color=J14.INK, pad=3)
        ax.set_xlabel("crossing threshold $k$ (baseline sd)", color=J14.INK2, labelpad=1)
    axes[0].set_ylabel("$\\rho$(crossing record, length)", color=J14.INK2, labelpad=1)
    handles = [Line2D([], [], color=J14.INK, marker="o", ms=3.5, lw=1.1, label="crossing of $\\mu + k\\sigma$, sustained"),
               Line2D([], [], color=J14.FAM_COL["control chart"], lw=0.9, ls="--", label="median shift at a chart alarm"),
               Line2D([], [], color=J14.FAM_COL["detector"], lw=0.9, ls="--", label="median shift at an RMS-detector alarm")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.54, 0.0),
               handletextpad=0.4, columnspacing=1.2, labelcolor=J14.INK2)
    J14.save(fig, "fig4_crossing_scale")


def fig_design_sweep():
    """fig5 -- every design on one axis: what it responds to against whether its onset follows test duration."""
    sw = pd.read_csv(T / "j26_design_sweep.csv")
    sw = sw[sw.median_shift_sd.notna() & sw.rho_index_length.notna()]
    KIND = [("control chart", J14.FAM_COL["control chart"], "^", "control chart"),
            ("chart, AR-corrected", "#8C3B0F", "v", "chart, autocorrelation-corrected"),
            ("chart, adaptive limits", "#D89B6A", "D", "chart, adaptive limits"),
            ("detector", J14.FAM_COL["detector"], "o", "threshold detector")]
    fig, axes = plt.subplots(1, 2, figsize=(J14.W, 2.45), facecolor=J14.SURFACE)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.90, bottom=0.36, wspace=0.26)
    xj = sw[sw.dataset == "xjtu"]
    cf = np.polyfit(np.log(xj.median_shift_sd.clip(lower=1e-3)), xj.rho_index_length, 2)
    for ax, (ds, label) in zip(axes, DS):
        J14.style_axes(ax, "y")
        g = sw[sw.dataset == ds]
        ax.axhline(0, color=J14.INK2, lw=0.6, zorder=1)
        gx = np.linspace(np.log(max(g.median_shift_sd.min(), 0.6)), np.log(g.median_shift_sd.max()), 120)
        ax.plot(np.exp(gx), np.polyval(cf, gx), "-", lw=1.0, color=J14.MUTED, zorder=2,
                label="fit on XJTU-SY" if ds == "xjtu" else "XJTU-SY fit, applied to FEMTO")
        for fam, col, mk, _ in KIND:
            b = g[g.family == fam]
            if not len(b):
                continue
            ax.scatter(b.median_shift_sd, b.rho_index_length, s=13, marker=mk, facecolor=col,
                       edgecolor="none", alpha=0.85, zorder=3)
            c = b[b.canonical]
            if len(c):
                ax.scatter(c.median_shift_sd, c.rho_index_length, s=42, marker=mk, facecolor="none",
                           edgecolor=J14.INK, linewidth=0.8, zorder=4)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 3, 6, 12, 25])
        ax.set_xticklabels(["1", "2", "3", "6", "12", "25"])
        ax.minorticks_off()
        ax.set_ylim(-0.85, 1.05)
        ax.set_title(label, color=J14.INK, pad=3)
        ax.set_xlabel("median shift at the alarm (baseline sd)", color=J14.INK2, labelpad=1)
    axes[0].set_ylabel("$\\rho$(alarm record, length)", color=J14.INK2, labelpad=1)
    handles = [Line2D([], [], color=c, marker=m, ms=3.6, lw=0, label=lab) for _, c, m, lab in KIND]
    handles += [Line2D([], [], color=J14.INK, marker="o", ms=5.5, lw=0, mfc="none", mew=0.8,
                       label="design used in Section 4"),
                Line2D([], [], color=J14.MUTED, lw=1.0, label="quadratic fit in $\\log$(shift), XJTU-SY only")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.54, -0.01),
               handletextpad=0.4, columnspacing=1.0, labelcolor=J14.INK2)
    J14.save(fig, "fig8_design_sweep")


def fig_agreement():
    la = pd.read_csv(T / "j23_label_agreement.csv")
    order = RMS3 + ["ma2sig", "3sig_kurt"] + CHARTS + ["pct02", "pct05", "pct10"]
    fig, axes = plt.subplots(1, 2, figsize=(J14.W, 2.9), facecolor=J14.SURFACE)
    fig.subplots_adjust(left=0.16, right=0.88, top=0.90, bottom=0.26, wspace=0.30)
    im = None
    for ax, (ds, label) in zip(axes, DS):
        M = np.full((len(order), len(order)), np.nan)
        g = la[la.dataset == ds]
        for r in g.itertuples():
            i, j = order.index(r.rule_a), order.index(r.rule_b)
            M[i, j] = M[j, i] = r.cohen_kappa
        np.fill_diagonal(M, 1.0)
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order if ax is axes[0] else [""] * len(order), fontsize=5.5)
        ax.tick_params(length=0, colors=J14.INK2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(label, color=J14.INK, pad=3)
        for k in (2.5, 4.5, 6.5):
            ax.axhline(k, color=J14.SURFACE, lw=1.2)
            ax.axvline(k, color=J14.SURFACE, lw=1.2)
    cax = fig.add_axes([0.90, 0.30, 0.015, 0.55])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("Cohen's $\\kappa$ between record labels", color=J14.INK2, fontsize=6.5, labelpad=2)
    cb.ax.tick_params(labelsize=6, colors=J14.INK2, length=1.5, width=0.4)
    J14.save(fig, "fig6_label_agreement")


def fig_ranks():
    p = pd.read_csv(T / "j3_performance.csv")
    p = p[(p["mode"] == "fixed") & (~p.degenerate)]
    fig, axes = plt.subplots(1, 2, figsize=(J14.W, 2.6), facecolor=J14.SURFACE)
    fig.subplots_adjust(left=0.17, right=0.88, top=0.88, bottom=0.30, wspace=0.28)
    im = None
    for ax, (ds, label) in zip(axes, DS):
        g = p[p.dataset == ds]
        wide = g.pivot_table(index="method", columns="rule", values="macro_f1").dropna(axis=0, how="any")
        cols = [c for c in ["3sig_rms", "ma3sig", "ma5sig", "ma2sig", "3sig_kurt", "pct02", "pct05", "pct10"] if c in wide.columns]
        wide = wide[cols]
        im = ax.imshow(wide.to_numpy(), cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto")
        for j, c in enumerate(cols):
            i = int(np.argmax(wide[c].to_numpy()))
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=J14.INK, lw=1.0))
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticks(range(len(wide)))
        ax.set_yticklabels([m.replace("thr:", "").replace("rf:all", "random forest") for m in wide.index]
                           if ax is axes[0] else [""] * len(wide), fontsize=5.5)
        ax.tick_params(length=0, colors=J14.INK2)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(label, color=J14.INK, pad=3)
    cax = fig.add_axes([0.90, 0.34, 0.015, 0.50])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label("macro-F1", color=J14.INK2, fontsize=6.5, labelpad=2)
    cb.ax.tick_params(labelsize=6, colors=J14.INK2, length=1.5, width=0.4)
    fig.text(0.52, 0.02, "black outline: the method with the highest macro-F1 under that rule", ha="center",
             color=J14.INK2, fontsize=6.5)
    J14.save(fig, "fig7_rank_heatmap")


def fig_ladder():
    fig, ax = plt.subplots(figsize=(J14.W, 2.15), facecolor=J14.SURFACE)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.98, bottom=0.02)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")
    steps = [("physical fault\ninitiation", "not observable\nin these data", J14.MUTED),
             ("first observable\nchange", "small at first,\ngrows with damage", J14.INK2),
             ("change large enough\nfor a given rule", "set by the rule's\nresponse size", J14.INK2),
             ("alarm", "detection, after the\nrule's run length", J14.INK2),
             ("labeling\nboundary", "what the label set\nencodes", J14.INK)]   # American spelling, IEEE (2026-09-21)
    w, gap = 1.80, 0.13
    x = 0.12
    for i, (title, sub, col) in enumerate(steps):
        ax.add_patch(Rectangle((x, 2.65), w, 0.95, facecolor=J14.NEUTRAL, edgecolor=col, lw=0.8))
        ax.text(x + w / 2, 3.12, title, ha="center", va="center", fontsize=6.0, color=J14.INK)
        ax.text(x + w / 2, 2.50, sub, ha="center", va="top", fontsize=5.4, color=J14.INK2)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.01, 3.12), (x + w + gap - 0.01, 3.12), arrowstyle="-|>",
                                         mutation_scale=6, color=J14.MUTED, lw=0.8))
        x += w + gap
    ax.text(0.12, 1.55, "Where each rule family lands on this ladder", fontsize=6.4, color=J14.INK, fontweight="bold")
    rows = [("control chart (EWMA, CUSUM)", "alarms at a drift of about two baseline standard deviations,\noften inside the baseline window itself", J14.FAM_COL["control chart"]),
            ("RMS detector, $3\\sigma$ or stricter", "alarms at four standard deviations or more,\nlater in records on longer tests", J14.FAM_COL["detector"]),
            ("fixed percentile of life", "places the boundary by definition,\nindependently of the signal", J14.FAM_COL["percentile"])]
    y = 1.12
    for name, text, col in rows:
        ax.add_patch(Rectangle((0.12, y - 0.10), 0.10, 0.26, facecolor=col, lw=0))
        ax.text(0.34, y, name, fontsize=5.9, color=J14.INK, va="center")
        ax.text(3.30, y, text, fontsize=5.5, color=J14.INK2, va="center", linespacing=1.25)
        y -= 0.52
    J14.save(fig, "fig0_estimand_ladder")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    fig_crossing()
    fig_design_sweep()
    fig_agreement()
    fig_ranks()
    fig_ladder()
    return 0


if __name__ == "__main__":
    sys.exit(main())
