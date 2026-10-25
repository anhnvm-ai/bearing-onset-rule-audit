"""J67 -- the six figures of the resubmitted manuscript (paper/access_resub/figures/).

Style follows J54 and J52: the IEEE Access widths (text 177.53 mm, column 85.29 mm), Arial 8 pt, PDF with
fonttype 42 plus a 300 dpi PNG. Colour carries the rule family only: RMS detectors teal, control charts orange,
percentile rules a muted gray (a de-emphasised reference class). The teal is #008FA3 rather than J14's #1B6773,
which fails the chroma floor of the palette validator (dataviz skill, scripts/validate_palette.js); the pair
#008FA3 / #B85C18 passes every check on white (CVD dE 17.7, normal-vision dE 24.2, contrast >= 3:1).
Filled markers: conventional settings; open markers: calibrated at matched 5% FAR. Every plotted number comes from
results/tables with the summaries of J66 (J63: median over the 12 cells of each protocol x shape x rule).

    python scripts/J67_resub_figures.py          # all six figures
    python scripts/J67_resub_figures.py 2 5      # selected figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
OUT = ROOT / "paper" / "access_resub" / "figures"
TW, CW = 177.53 / 25.4, 85.29 / 25.4          # \textwidth and \columnwidth of ieeeaccess.cls, inches
DET, CHART, PCT = "#008FA3", "#B85C18", "#7A8D98"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL, SURFACE = "#e1e0d9", "#c3c2b7", "#f0efec", "#ffffff"
BASE = 8.0
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": BASE, "axes.titlesize": BASE, "axes.labelsize": BASE, "xtick.labelsize": BASE,
    "ytick.labelsize": BASE, "legend.fontsize": BASE, "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "mathtext.default": "it", "axes.titlecolor": INK, "axes.labelcolor": INK2})
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO"}
ARMS = ["P-retro", "P-robust", "P-II", "P-iter", "P-fixed(10)", "P-fixed(15)", "P-fixed(20)", "P-prospective"]
RMS_DET, CHARTS = ["3sig_rms", "ma3sig", "ma5sig"], ["ewma", "cusum"]
MODELS = ["lin:rms", "ridge:all", "knn:all", "gbr:all", "et:all", "rf:all", "two-stage", "gru", "tcn"]
MODEL_LABEL = ["linear (RMS)", "ridge", "k-NN", "boosting", "extra trees", "random forest", "two-stage", "GRU", "TCN"]
LABELER_STYLE = {"3sig_rms": (DET, "o"), "ma5sig": (DET, "s"), "ewma": (CHART, "^"), "pct05": (PCT, "D")}
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")      # log-axis ticks as 0.01, 0.1, 1 rather than powers of ten
CAL_LEGEND = [Line2D([], [], ls="", marker="o", ms=4, mfc=INK2, mec=INK2, label="conventional settings"),
              Line2D([], [], ls="", marker="o", ms=4, mfc=SURFACE, mec=INK2, mew=0.9, label="matched 5% false-alarm rate")]


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(T / f"{name}.csv")


def style(ax, grid: str | None = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, length=2, width=0.5, pad=1.5)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)


def panel(ax, letter: str, title: str) -> None:
    ax.set_title(rf"$\bf{{({letter})}}$ {title}", loc="left", pad=3)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote paper/access_resub/figures/{name}.pdf + .png")


def range_dot(ax, x, vals, color, cal, lw=1.0, ms=4.0) -> None:
    """A min-max range with a dot at the median: the summary of a group of rules."""
    v = np.asarray(vals, float)
    ax.plot([x, x], [v.min(), v.max()], color=color, lw=lw, solid_capstyle="butt", zorder=2)
    ax.plot(x, np.median(v), ls="", marker="o", ms=ms, mfc=SURFACE if cal else color, mec=color, mew=0.9, zorder=3)


# ------------------------------------------------------------------ Fig. 1: framework
def fig1() -> None:
    """Five boxes laid out on a 100 x 35 grid; every text line has a fixed position, so nothing can overflow."""
    fig = plt.figure(figsize=(TW, 2.45), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 35)
    ax.axis("off")
    y0, h, lh = 7.0, 26.0, 2.25
    widths, gap = [12.0, 15.0, 29.5, 16.0, 19.0], 2.0
    xs = [0.25]
    for w in widths[:-1]:
        xs.append(xs[-1] + w + gap)
    # (header, tag in the header, [(tag or None, text)], varied)
    boxes = [
        ("Data", None, [(None, "4 datasets,"), (None, "33 bearings"), (None, ""), (None, "RMS and"),
                        (None, "kurtosis"), (None, "per record")], False),
        ("Onset rule", 1, [(None, "10 rules and"), (None, "67 designs:"), (None, ""), (None, "RMS detectors,"),
                            (None, "control charts,"), (None, "percentile rules")], True),
        ("Implementation", None, [(2, "baseline protocol: length of"), (None, "Phase I, start of alarms"),
                                  (None, "(six protocols)"), (3, "false-alarm rate: conventional"),
                                  (None, "settings or matched at 5%"), (4, "non-firing trajectories:"),
                                  (None, "excluded or labeled healthy,"), (None, "record or bearing weights")], True),
        ("Labels", None, [(None, "onset fraction u,"), (None, r"its $\beta_1 - 1$ with length"), (None, ""),
                          (None, "positive rate,"), (None, "label agreement,"), (None, "RUL-target"),
                          (None, r"distance $D_j$")], False),
        ("RUL evaluation", 5, [(None, "target: labeler's own"), (None, "or a common"), (None, "reference;"),
                                (None, "scored records:"), (None, "S or S3"), (None, ""), (None, "9 models;"),
                                (None, "C-rank, C-common")], True),
    ]

    def tag(x, y, k):
        ax.add_patch(plt.Circle((x, y), 0.95, facecolor=INK, edgecolor="none", zorder=4))
        ax.text(x, y - 0.05, str(k), fontsize=BASE - 1.5, color=SURFACE, ha="center", va="center",
                fontweight="bold", zorder=5)

    for x, w, (head, htag, lines, varied) in zip(xs, widths, boxes):
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                    facecolor=SURFACE if varied else NEUTRAL, edgecolor=INK2 if varied else AXIS,
                                    linewidth=0.9 if varied else 0.6))
        top = y0 + h
        ax.text(x + 1.0, top - 2.2, head, fontsize=BASE, fontweight="bold", color=INK, va="center")
        if htag:
            tag(x + w - 1.6, top - 2.2, htag)
        indent = 3.0 if any(t for t, _ in lines) else 1.0
        for i, (t, text) in enumerate(lines):
            yy = top - 6.0 - i * lh
            if t:
                tag(x + 1.6, yy, t)
            ax.text(x + indent, yy, text, fontsize=BASE, color=INK2, va="center")
    for x, w, xn in zip(xs[:-1], widths[:-1], xs[1:]):
        ax.add_patch(FancyArrowPatch((x + w + 0.15, y0 + h / 2), (xn - 0.15, y0 + h / 2), arrowstyle="-|>",
                                     mutation_scale=6, color=INK2, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(50, 3.2, "Reported: the four items of ORD-4 (Table 5), with released code, labels, configurations "
            "and a dated pre-specified plan", fontsize=BASE, color=INK2, ha="center", va="center")
    save(fig, "fig1_framework")


# ------------------------------------------------------------------ Fig. 2: protocol x calibration
def fig2() -> None:
    gap, mg = read("j60_gap"), read("j61_matched_gap")
    g = gap[gap.bearing_set == "main"].set_index(["dataset", "arm"])
    m = mg[(mg.noise_arm == "main") & (mg.level == 0.05)].set_index(["dataset", "protocol"])
    ylo, yhi = -3.2, 4.8
    fig, axes = plt.subplots(1, 2, figsize=(TW, 2.8), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.84, bottom=0.36, wspace=0.08)
    for ax, ds, letter in zip(axes, ("xjtu", "femto"), "ab"):
        xs = list(range(len(ARMS))) + [len(ARMS) + 0.5, len(ARMS) + 1.5]
        pts = [(g.loc[(ds, a)].gap_b1_minus_1, g.loc[(ds, a)].gap_lo, g.loc[(ds, a)].gap_hi, False) for a in ARMS]
        pts += [(m.loc[(ds, p)].gap_calibrated, m.loc[(ds, p)].gap_cal_lo, m.loc[(ds, p)].gap_cal_hi, True)
                for p in ("P-retro", "P-prospective")]
        ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        ax.axvline(len(ARMS) - 0.25, color=AXIS, lw=0.6, zorder=1)
        for x, (v, lo, hi, cal) in zip(xs, pts):
            ax.plot([x, x], [max(lo, ylo), min(hi, yhi)], color=INK, lw=0.9, solid_capstyle="butt", zorder=2)
            for end, lim, mk in ((lo, ylo, "v"), (hi, yhi, "^")):
                if (end < lim and lim == ylo) or (end > lim and lim == yhi):
                    ax.plot(x, lim, ls="", marker=mk, ms=3, color=INK, zorder=3)
                    left = x < len(ARMS)                     # conventional arms: label left of the line
                    ax.text(x - 0.14 if left else x + 0.14, lim + (0.15 if lim == ylo else -0.15),
                            f"{end:+.1f}".replace("-", "−"), fontsize=BASE - 1, color=INK2,
                            va="bottom" if lim == ylo else "top", ha="right" if left else "left")
            ax.plot(x, v, ls="", marker="o", ms=4.2, mfc=SURFACE if cal else INK, mec=INK, mew=0.9, zorder=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(ARMS + ["P-retro", "P-prospective"], rotation=40, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, xs[-1] + 0.6)
        ax.set_ylim(ylo - 0.35, yhi + 1.1)
        ax.text((len(ARMS) - 1) / 2, yhi + 0.6, "conventional settings", ha="center", va="center", color=INK2)
        ax.text(len(ARMS) + 1.0, yhi + 0.6, "5% FAR", ha="center", va="center", color=INK2)
        n = int(g.loc[(ds, "P-retro")].n_bearings)
        panel(ax, letter, f"{DS_NAME[ds]}, {n} bearings")
        style(ax)
    axes[0].set_ylabel(r"detectors minus charts, $\beta_1 - 1$")
    fig.legend(handles=CAL_LEGEND + [Line2D([], [], color=INK, lw=0.9, label="95% bearing-bootstrap interval")],
               loc="lower center", ncol=3, frameon=False, handlelength=1.2, columnspacing=1.6, labelcolor=INK2,
               bbox_to_anchor=(0.53, 0.0))
    save(fig, "fig2_protocol_gap")


# ------------------------------------------------------------------ Fig. 3: onset fraction against length
def fig3() -> None:
    o = read("j60_onsets")
    x = o[(o.dataset == "xjtu") & o.fired.astype(bool)]
    panels = [("3sig_rms", "P-retro"), ("ewma", "P-retro"), ("3sig_rms", "P-prospective"), ("ewma", "P-prospective")]
    mk = {1: "o", 2: "s", 3: "^"}
    fig, axes = plt.subplots(2, 2, figsize=(CW, 3.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.93, bottom=0.21, hspace=0.32, wspace=0.10)
    nn = np.geomspace(35, 3000, 60)
    for ax, (rule, arm), letter in zip(axes.flat, panels, "abcd"):
        q = x[(x.rule == rule) & (x.arm == arm)]
        if arm == "P-prospective":
            q = q[q.eligible.astype(bool)]
        col = DET if rule == "3sig_rms" else CHART
        ax.plot(nn, 21 / nn, ls=":", color=MUTED, lw=0.7, zorder=1)
        for c, m_ in mk.items():
            s = q[q.condition.astype(str) == str(c)]           # j60_onsets stores the condition as text
            ax.scatter(s.n_records, s.u, marker=m_, s=16, facecolor=col, edgecolor=SURFACE, linewidth=0.4, zorder=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.004, 1.3)
        ax.xaxis.set_major_formatter(PLAIN)
        ax.yaxis.set_major_formatter(PLAIN)
        panel(ax, letter, f"{rule}, {arm}")
        style(ax, grid=None)
    axes[1, 0].set_xlabel("trajectory length n (records)")
    axes[1, 1].set_xlabel("trajectory length n (records)")
    axes[0, 0].set_ylabel("onset fraction u")
    axes[1, 0].set_ylabel("onset fraction u")
    axes[0, 0].text(40, 0.006, "dotted line:\nalarm at record 20", fontsize=BASE - 1, color=MUTED, va="bottom",
                    linespacing=1.1)
    handles = [Line2D([], [], ls="", marker=mk[c], ms=4.5, mfc=INK2, mec=INK2, label=lab)
               for c, lab in ((1, "condition 1"), (2, "condition 2"), (3, "condition 3"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, handletextpad=0.2, columnspacing=1.0,
               labelcolor=INK2, bbox_to_anchor=(0.57, 0.0))
    save(fig, "fig3_onset_fraction")


# ------------------------------------------------------------------ Fig. 4: simulation
def fig4() -> None:
    g = read("j63_gradual")
    g = g[g.protocol == "P-prospective"]
    med = g.groupby(["shape", "rule"]).agg(early=("share_before_ref1", "median"),
                                           slope=("slope_b1_minus_1", "median")).reset_index()
    groups = [("RMS detectors", RMS_DET, DET, False), ("control charts", CHARTS, CHART, False),
              ("detectors at 5% FAR", ["thr_cal", "ma_cal"], DET, True), ("charts at 5% FAR", ["ewma_cal", "cusum_cal"], CHART, True)]
    shapes = [("step", "step"), ("ramp", "linear ramp"), ("exp", "two-phase\nexponential")]
    fig, axes = plt.subplots(2, 1, figsize=(CW, 3.25), sharex=True)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.95, bottom=0.21, hspace=0.25)
    for ax, (col, ylab, scale), letter in zip(axes, (("early", r"alarms before $t_1$ (%)", 100),
                                                    ("slope", r"$\beta_1 - 1$", 1)), "ab"):
        for i, (shp, _) in enumerate(shapes):
            for j, (_, rules, color, cal) in enumerate(groups):
                v = med[(med["shape"] == shp) & med.rule.isin(rules)][col] * scale
                range_dot(ax, i + (j - 1.5) * 0.17, v, color, cal)
        if col == "slope":
            ax.axhline(0, color=INK2, lw=0.6, zorder=1)
        ax.set_ylabel(ylab)
        panel(ax, letter, "")
        style(ax)
    axes[1].set_xticks(range(len(shapes)))
    axes[1].set_xticklabels([lab for _, lab in shapes])
    axes[1].set_xlim(-0.5, len(shapes) - 0.5)
    handles = [Line2D([], [], ls="", marker="o", ms=4, mfc=SURFACE if cal else c, mec=c, mew=0.9, label=lab)
               for lab, _, c, cal in groups]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, handletextpad=0.2, columnspacing=1.0,
               labelcolor=INK2, bbox_to_anchor=(0.57, 0.0))
    save(fig, "fig4_simulation")


# ------------------------------------------------------------------ Fig. 5: consequences for the labels
def fig5() -> None:
    lab, mg = read("j60_labels"), read("j61_matched_gap")
    fig, axes = plt.subplots(1, 3, figsize=(TW, 2.45), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    fig.subplots_adjust(left=0.065, right=0.995, top=0.88, bottom=0.30, wspace=0.32)
    ax = axes[0]
    pr = lab[(lab.dataset == "xjtu") & (lab.kind == "positive_rate")].set_index(["arm", "rule"]).pos_rate_record
    for i, arm in enumerate(ARMS):
        range_dot(ax, i - 0.12, pr.loc[arm].loc[RMS_DET], DET, False)
        range_dot(ax, i + 0.12, pr.loc[arm].loc[CHARTS], CHART, False)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(ARMS, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("share of records labeled degraded")
    panel(ax, "a", "positive rate, XJTU-SY")
    ax.legend(handles=[Line2D([], [], color=DET, lw=1.0, marker="o", ms=4, label="RMS detectors (3)"),
                       Line2D([], [], color=CHART, lw=1.0, marker="o", ms=4, label="control charts (2)")],
              loc="upper right", frameon=False, handlelength=1.2, labelcolor=INK2, borderaxespad=0.1)
    style(ax)
    m = mg[(mg.noise_arm == "main") & (mg.level == 0.05)].set_index(["dataset", "protocol"])
    td = lab[(lab.kind == "target_distance") & (lab.rule == "chart vs RMS detector")].set_index(["dataset", "arm"]).median_D_j
    cats = [(ds, p) for ds in ("xjtu", "femto") for p in ("P-retro", "P-prospective")]
    names = [p for _, p in cats]
    for ax, letter, title, before, after in (
            (axes[1], "b", "chart-detector disagreement",
             [m.loc[c].disagreement_uncalibrated_median for c in cats], [m.loc[c].disagreement_calibrated_median for c in cats]),
            (axes[2], "c", r"RUL-target distance $D_j$",
             [td.loc[c] for c in cats], [m.loc[c].median_D_j_calibrated for c in cats])):
        for i, (b, a) in enumerate(zip(before, after)):
            ax.annotate("", xy=(i, a), xytext=(i, b), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8,
                                                                      mutation_scale=6, shrinkA=2.5, shrinkB=2.5))
            ax.plot(i, b, ls="", marker="o", ms=4.2, mfc=INK, mec=INK, zorder=3)
            ax.plot(i, a, ls="", marker="o", ms=4.2, mfc=SURFACE, mec=INK, mew=0.9, zorder=3)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(names, rotation=40, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.5, len(cats) - 0.5)
        top = 1.15 * max(before)
        ax.set_ylim(0, top)
        ax.axvline(1.5, color=AXIS, lw=0.6, zorder=1)
        for xc, ds in ((0.5, "xjtu"), (2.5, "femto")):
            ax.text(xc, 0.985 * top, DS_NAME[ds], ha="center", va="top", color=INK2)
        panel(ax, letter, title)
        style(ax)
    short = [Line2D([], [], ls="", marker="o", ms=4, mfc=INK2, mec=INK2, label="conventional"),
             Line2D([], [], ls="", marker="o", ms=4, mfc=SURFACE, mec=INK2, mew=0.9, label="5% FAR")]
    axes[2].legend(handles=short, loc="center right", frameon=False, handlelength=0.8, labelcolor=INK2,
                   borderaxespad=0.1)
    save(fig, "fig5_labels")


# ------------------------------------------------------------------ Fig. 6: RUL on the common reference
def fig6() -> None:
    o, cr = read("j60_onsets"), read("j62_common_reference")
    # two rows since the self-review of 2026-10-26: the C-rank panels moved to the supplementary (Table S22)
    fig = plt.figure(figsize=(TW, 3.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.05], hspace=0.55, wspace=0.12,
                          left=0.065, right=0.995, top=0.94, bottom=0.27)
    # (a) the targets on one bearing
    ax = fig.add_subplot(gs[0, :])
    b, arm = "Bearing1_3", "P-prospective"
    q = o[(o.dataset == "xjtu") & (o.trajectory == b) & (o.arm == arm)].set_index("rule")
    n = int(q.n_records.iloc[0])
    t = np.arange(n)
    onset = {r: int(q.loc[r, "i_star"]) for r in ("3sig_rms", "ma5sig", "ewma")}
    onset["pct05"] = int(np.floor(n * (1 - 0.05)))
    s_all, s_sig = max(onset.values()), max(onset[r] for r in ("3sig_rms", "ma5sig", "ewma"))
    ax.axvspan(s_sig, n - 1, color=NEUTRAL, lw=0, zorder=0)
    ax.axvspan(s_all, n - 1, color=GRID, lw=0, zorder=0)
    for r in ("ewma", "3sig_rms", "pct05"):
        col, _ = LABELER_STYLE[r]
        y = np.where(t < onset[r], 1.0, (n - 1 - t) / max(n - 1 - onset[r], 1))
        ax.plot(t, y, color=col, lw=1.1, zorder=3, label=f"{r} target, onset {onset[r]}")
    ax.plot(t, (n - 1 - t) / (n - 1), color=INK, lw=0.9, ls=(0, (4, 2)), zorder=4, label="common reference $r_t$")
    ax.text((s_sig + s_all) / 2, 1.02, "S3", ha="center", va="bottom", color=INK2)
    ax.text((s_all + n - 1) / 2, 1.02, "S", ha="center", va="bottom", color=INK2)
    ax.set_xlim(0, n - 1)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("record t", labelpad=1)
    ax.set_ylabel("RUL target")
    panel(ax, "a", f"XJTU-SY {b}, {n} records, {arm}")
    ax.legend(loc="lower left", frameon=False, ncol=1, handlelength=1.6, labelcolor=INK2, borderaxespad=0.1,
              labelspacing=0.25)
    style(ax)
    # (b), (c) the nine models under the four labelers, P-prospective, conventional settings, scoring set S
    a = cr[(cr.variant == "a") & (cr.config == "P-prospective|canonical")]
    specs = [(1, 0, "xjtu", "rmse_common_S", "C-common RMSE", "b"), (1, 1, "femto", "rmse_common_S", "C-common RMSE", "c")]
    for row, colm, ds, col, ylab, letter in specs:
        ax = fig.add_subplot(gs[row, colm])
        d = a[a.dataset == ds].set_index(["labeler", "model"])[col]
        for j, (lb, (c, mk)) in enumerate(LABELER_STYLE.items()):
            xs = np.arange(len(MODELS)) + (j - 1.5) * 0.15
            ax.scatter(xs, [d.loc[(lb, mo)] for mo in MODELS], marker=mk, s=14, facecolor=c, edgecolor=SURFACE,
                       linewidth=0.4, zorder=3, label=lb)
        if col.startswith("rmse"):
            ax.set_yscale("log")
            ax.set_ylim(0.008, 1.0)
            ax.yaxis.set_major_formatter(PLAIN)
        else:
            ax.set_ylim(0.3, 1.0)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODEL_LABEL if row == 1 else [], rotation=40, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        if colm == 0:
            ax.set_ylabel(ylab)
        nb = int(a[a.dataset == ds].n_bearings.iloc[0])
        panel(ax, letter, f"{DS_NAME[ds]}, {nb} bearings")
        style(ax)
    handles = [Line2D([], [], ls="", marker=mk, ms=4.5, mfc=c, mec=c, label=f"labeler {lb}")
               for lb, (c, mk) in LABELER_STYLE.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, handletextpad=0.2, columnspacing=1.6,
               labelcolor=INK2, bbox_to_anchor=(0.53, 0.0))
    save(fig, "fig6_rul")


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6}

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    which = [int(a) for a in sys.argv[1:]] or list(FIGS)
    for k in which:
        FIGS[k]()
