"""J21 -- Figure 3 of the two-contribution manuscript: the size of the change each rule responds to.

Reads results/tables/j22_shift_at_alarm.csv (J20) and draws, per dataset, the indicator level over each rule's
confirming run at the alarm, in baseline standard deviations, one marker per trajectory, with the median as a bar.
Hollow markers are onsets inside the rule's own baseline window. Style and width follow J14.

Output: results/figures/fig3_shift_at_alarm.pdf and .png
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

ORDER = ["3sig_rms", "ma3sig", "ma5sig", "ma2sig", "ewma", "cusum"]
PANELS = [("xjtu", "XJTU-SY"), ("femto", "FEMTO"), ("ims", "IMS")]


def main() -> int:
    sh = pd.read_csv(J14.TABLES / "j22_shift_at_alarm.csv")
    fig, axes = plt.subplots(1, 3, figsize=(J14.W, 2.35), sharey=True, facecolor=J14.SURFACE)
    fig.subplots_adjust(left=0.10, right=0.995, top=0.90, bottom=0.36, wspace=0.08)
    rng = np.random.default_rng(3)
    for ax, (ds, label) in zip(axes, PANELS):
        J14.style_axes(ax, "y")
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.axhline(3, color=J14.MUTED, lw=0.6, ls=":", zorder=1)
        for k, rule in enumerate(ORDER):
            g = sh[(sh.dataset == ds) & (sh.rule == rule)]
            col = J14.FAM_COL["control chart" if rule in ("ewma", "cusum") else "detector"]
            if g.empty:
                continue
            v = np.clip(g.shift_in_baseline_sd.to_numpy(), 0.3, None)
            xs = k + rng.uniform(-0.2, 0.2, len(g))
            inside = g.inside_baseline.to_numpy()
            ax.scatter(xs[~inside], v[~inside], s=9, color=col, edgecolor=J14.SURFACE, linewidth=0.3, zorder=3)
            ax.scatter(xs[inside], v[inside], s=9, facecolor=J14.SURFACE, edgecolor=col, linewidth=0.6, zorder=3)
            ax.plot([k - 0.3, k + 0.3], [np.median(g.shift_in_baseline_sd)] * 2, color=J14.INK, lw=1.0, zorder=4)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER, rotation=40, ha="right")
        ax.set_xlim(-0.6, len(ORDER) - 0.4)
        ax.set_ylim(0.3, 100)
        ax.set_yticks([0.5, 1, 2, 3, 5, 10, 30, 100])
        ax.set_yticklabels(["0.5", "1", "2", "3", "5", "10", "30", "100"])
        ax.set_title(label, color=J14.INK, pad=3)
    axes[0].set_ylabel("indicator shift at the alarm\n(baseline standard deviations)", color=J14.INK2, labelpad=1)
    handles = [Line2D([], [], ls="", marker="o", ms=3.5, color=J14.FAM_COL["detector"], label="detector rule"),
               Line2D([], [], ls="", marker="o", ms=3.5, color=J14.FAM_COL["control chart"], label="control chart"),
               Line2D([], [], ls="", marker="o", ms=3.5, markerfacecolor=J14.SURFACE, markeredgecolor=J14.INK2,
                      label="onset inside the baseline"),
               Line2D([], [], color=J14.INK, lw=1.0, label="median"),
               Line2D([], [], color=J14.MUTED, lw=0.6, ls=":", label="3 standard deviations")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.54, 0.0),
               handletextpad=0.3, columnspacing=1.0, labelcolor=J14.INK2)
    J14.save(fig, "fig3_shift_at_alarm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
