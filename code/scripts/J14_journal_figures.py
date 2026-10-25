"""J14 -- the six journal figures, drawn at the manuscript's text width from saved tables only.

Nothing is recomputed from raw data and no J1x analysis is rerun. Every figure is written as PDF (the file LaTeX
uses, since graphicx prefers .pdf) and PNG, 5.4 in wide, which is a preprint text width (390 pt), so
the figures are included at 1:1 and no label is smaller than 6 pt. There are no in-image titles: the LaTeX captions
carry the figure numbers.

    fig1_fpt_heatmap          j3_fpt_times.csv, j3_performance.csv (mode=='fixed')      -> fig1_fpt_matrix.csv
    fig2_arl_signature        j3_fpt_times.csv
    fig3_physics_test         j12_fpt_covariates.csv
    fig4_dispersion           j13_fpt_dispersion.csv (pairwise, non-degenerate subset)
    fig5_convention_grid      j15_rule_sensitivity.csv
    fig6_prevalence_circularity  j3_performance.csv (mode=='fixed')                      -> fig6a/fig6b CSVs

Note: J11, J12 and J13 still write their own PNG versions of fig2-fig5 when rerun; the manuscript uses the PDFs
written here.

Conventions shared with J3/J11: a rule "fires inside its own baseline" when its fire index is below
max(5, int(0.20 * n_records)); a rule is "degenerate" on a dataset when j3_performance flags it (induced positive
rate outside [0.01, 0.50]). Family colours are the ones used throughout the paper.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.colorbar import ColorbarBase  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES, FIGS, LOGS = ROOT / "results" / "tables", ROOT / "results" / "figures", ROOT / "results" / "logs"

# ------------------------------------------------------------------ style
FAM_COL = {"detector": "#1B6773", "control chart": "#B85C18", "percentile": "#7A8D98"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL, SURFACE = "#e1e0d9", "#c3c2b7", "#f0efec", "#ffffff"
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
W = 5.4          # figure width in inches = LaTeX \textwidth
DPI = 300
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

RULES = [("3sig_rms", "detector"), ("ma3sig", "detector"), ("ma5sig", "detector"),
         ("ma2sig", "detector"), ("3sig_kurt", "detector"),
         ("ewma", "control chart"), ("cusum", "control chart"),
         ("pct02", "percentile"), ("pct05", "percentile"), ("pct10", "percentile")]
FAMILY = dict(RULES)
FAMILY_TITLE = {"detector": "detector rules", "control chart": "control charts", "percentile": "percentile of life"}
DATASETS = [("xjtu", "XJTU-SY"), ("femto", "FEMTO"), ("ims", "IMS"), ("nsk", "NSK 6205")]
BASE_FRAC, MIN_BASE, MIN_TRAJ_FOR_RATE = 0.20, 5, 4
RMS3 = ["3sig_rms", "ma3sig", "ma5sig"]          # RMS detectors with thresholds of 3 sigma or more
BETWEEN = ["ma2sig", "3sig_kurt"]                # detector rules that fall between the families
RMS_DET4 = ["3sig_rms", "ma2sig", "ma3sig", "ma5sig"]
PCT = ["pct02", "pct05", "pct10"]
RMS_RULES, PCT_RULES = RMS3, PCT
METHOD_GROUPS = [("amplitude features", ["thr:rms", "thr:std", "thr:p2p", "thr:absmax"]),
                 ("shape features", ["thr:crest", "thr:kurtosis", "thr:skew"]),
                 ("all features", ["rf:all"])]
METHOD_NAME = {"thr:rms": "RMS (labelling statistic)", "thr:std": "standard deviation",
               "thr:p2p": "peak-to-peak", "thr:absmax": "absolute maximum", "thr:crest": "crest factor",
               "thr:kurtosis": "kurtosis", "thr:skew": "skewness", "rf:all": "random forest"}
_fh = None


def log(msg=""):
    print(msg)
    if _fh:
        _fh.write(msg + "\n")


def style_axes(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, length=2, width=0.5, pad=1.5)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    log(f"  wrote results/figures/{name}.pdf + .png")


def short_name(ds, traj):
    if ds == "ims":
        part = traj.replace("IMS_", "").split("_test_")
        return f"{part[0]} test, {part[1]}"
    if ds == "nsk":
        return traj.replace("_", " ")
    return traj


# ------------------------------------------------------------------ data builders (unchanged definitions)
def load():
    ft = pd.read_csv(TABLES / "j3_fpt_times.csv")
    perf = pd.read_csv(TABLES / "j3_performance.csv")
    return ft, perf[perf["mode"] == "fixed"].copy()


def build_fig1_matrix(ft, perf):
    lens = ft.groupby(["dataset", "trajectory"]).n_records.agg(["first", "nunique"])
    assert (lens["nunique"] == 1).all(), "n_records differs across rules for one trajectory"
    degen = perf.groupby(["dataset", "rule"]).degenerate.first()
    fired = ft.set_index(["dataset", "trajectory", "rule"])
    rows = []
    for ds, _ in DATASETS:
        trajs = lens.loc[ds]["first"].reset_index().sort_values(["first", "trajectory"])
        for traj, n in zip(trajs.trajectory, trajs["first"]):
            for rule, fam in RULES:
                key = (ds, traj, rule)
                has = key in fired.index
                idx = int(fired.loc[key, "fpt_index"]) if has else None
                frac = float(fired.loc[key, "fpt_fraction"]) if has else np.nan
                nb = max(MIN_BASE, int(n * BASE_FRAC))
                d = degen.get((ds, rule), None)
                rows.append({"dataset": ds, "trajectory": traj, "n_records": int(n), "rule": rule,
                             "family": fam, "fired": has, "fpt_index": idx, "fpt_fraction": frac,
                             "baseline_records": nb, "inside_baseline": bool(has and idx < nb),
                             "degenerate_on_dataset": None if d is None else bool(d)})
    return pd.DataFrame(rows)


def build_fig6a(perf):
    rows = []
    for ds, _ in DATASETS:
        g = perf[(perf.dataset == ds) & (~perf.degenerate)]
        for rule, gr in g.groupby("rule"):
            b1, b2 = gr.loc[gr.macro_f1.idxmax()], gr.loc[gr.mcc.idxmax()]
            rows.append({"dataset": ds, "rule": rule, "family": FAMILY[rule], "pos_rate": float(gr.pos_rate.iloc[0]),
                         "best_macro_f1": float(b1.macro_f1), "best_f1_method": b1.method,
                         "best_mcc": float(b2.mcc), "best_mcc_method": b2.method,
                         "n_records_scored": int(gr.n_records.iloc[0])})
    return pd.DataFrame(rows)


def build_fig6b(perf):
    rows = []
    for ds in ("xjtu", "femto"):
        w = perf[perf.dataset == ds].pivot_table(index="method", columns="rule", values="macro_f1")
        for group, methods in METHOD_GROUPS:
            for m in methods:
                if m not in w.index:
                    rows.append({"dataset": ds, "group": group, "method": m, "gain": np.nan,
                                 "mean_rms_rules": np.nan, "mean_pct_rules": np.nan})
                    continue
                a, b = float(w.loc[m, RMS_RULES].mean()), float(w.loc[m, PCT_RULES].mean())
                rows.append({"dataset": ds, "group": group, "method": m, "gain": a - b,
                             "mean_rms_rules": a, "mean_pct_rules": b})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Figure 1
def draw_fig1(mat):
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    norm = Normalize(0.0, 1.0)
    # Drawn at the two-column width it is printed at (\textwidth = 7.16 in), not at the 5.4 in of the
    # other figures: this one carries 33 row labels and ten rotated column labels, and printing it at
    # scale 1 is what keeps them at the size set here rather than at 0.76 of it (2026-09-25).
    FW = 7.16
    LEFT, RIGHT, TOP, BOTTOM = 1.62, 0.08, 0.74, 0.70
    ROW_H, HEAD_GAP, FAM_GAP = 0.135, 1.45, 0.30
    col_x, x = {}, 0.0
    for i, (rule, fam) in enumerate(RULES):
        if i and fam != RULES[i - 1][1]:
            x += FAM_GAP
        col_x[rule] = x
        x += 1.0
    grid_w = x
    cell_w = (FW - LEFT - RIGHT) / grid_w
    n_rows = mat.groupby(["dataset", "trajectory"]).ngroups
    grid_rows = n_rows + HEAD_GAP * len(DATASETS)
    H = TOP + grid_rows * ROW_H + BOTTOM
    fig = plt.figure(figsize=(FW, H), facecolor=SURFACE)
    ax = fig.add_axes([LEFT / FW, BOTTOM / H, grid_w * cell_w / FW, grid_rows * ROW_H / H])
    ax.set_xlim(0, grid_w)
    ax.set_ylim(grid_rows, 0)
    ax.axis("off")
    left_edge = -(LEFT - 0.04) / cell_w
    pad_x, pad_y, y = 0.04, 0.08, 0.0
    for ds, label in DATASETS:
        sub = mat[mat.dataset == ds]
        trajs = sub.drop_duplicates("trajectory")[["trajectory", "n_records"]]
        nb = len(trajs)
        ax.text(left_edge, y + HEAD_GAP - 0.25, f"{label} · {nb} bearing{'s' if nb > 1 else ''}", ha="left",
                va="bottom", fontsize=8.0, color=INK, fontweight="bold", clip_on=False)
        y += HEAD_GAP
        y0 = y
        for _, t in trajs.iterrows():
            for _, c in sub[sub.trajectory == t.trajectory].iterrows():
                cx = col_x[c.rule]
                rect = dict(xy=(cx + pad_x, y + pad_y), width=1 - 2 * pad_x, height=1 - 2 * pad_y, linewidth=0)
                if c.fired:
                    ax.add_patch(Rectangle(facecolor=cmap(norm(c.fpt_fraction)), **rect))
                    if c.inside_baseline:
                        ax.plot(cx + 0.5, y + 0.5, marker="o", ms=2.1, color=INK, mec=SURFACE, mew=0.35, zorder=4)
                else:
                    ax.add_patch(Rectangle(facecolor=NEUTRAL, edgecolor=MUTED, hatch="//////", **rect))
            ax.text(-0.1, y + 0.55, f"{short_name(ds, t.trajectory)} ({t.n_records:,})", ha="right", va="center",
                    fontsize=7.0, color=INK2, clip_on=False)
            y += 1
        for rule, _ in RULES:
            if sub[sub.rule == rule].degenerate_on_dataset.iloc[0] is True:
                ax.add_patch(Rectangle((col_x[rule] + 0.015, y0 + 0.02), 0.97, (y - y0) - 0.04, fill=False,
                                       edgecolor=INK, linewidth=0.9, zorder=5))
    fams = {}
    for rule, fam in RULES:
        fams.setdefault(fam, []).append(col_x[rule])
    strip_y = -(TOP - 0.20) / ROW_H
    for fam, xs in fams.items():
        x0, x1 = min(xs), max(xs) + 1
        ax.add_patch(Rectangle((x0 + 0.05, strip_y), (x1 - x0) - 0.10, 0.45, facecolor=FAM_COL[fam], linewidth=0,
                               clip_on=False))
        ax.text((x0 + x1) / 2, strip_y - 0.15, FAMILY_TITLE[fam], ha="center", va="bottom", fontsize=8.0, color=INK,
                clip_on=False)
    for rule, _ in RULES:
        ax.text(col_x[rule] + 0.35, -0.2, rule, rotation=45, ha="left", va="bottom", fontsize=7.5, color=INK,
                clip_on=False)
    cax = fig.add_axes([LEFT / FW, 0.40 / H, 2.05 / FW, 0.10 / H])
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal", ticks=[0, 0.2, 0.5, 0.8, 1.0])
    cb.outline.set_visible(False)
    cax.tick_params(labelsize=7.0, colors=INK2, length=1.5, width=0.4, pad=1)
    cax.set_xlabel("onset as fraction of life", fontsize=7.5, color=INK2, labelpad=1.5)
    handles = [Patch(facecolor=NEUTRAL, edgecolor=MUTED, hatch="//////", linewidth=0, label="rule never fired"),
               Line2D([], [], marker="o", ls="", ms=3.2, color=INK, mec=SURFACE, mew=0.35,
                      label="onset inside the rule's 20% baseline"),
               # "fails the screen" rather than a named screen: this figure is shared by the journal
               # manuscript (paper/journal, "usability screen") and the IEEE one (paper/ieee,
               # "prevalence screen"), and both captions read "fails the ... screen on that dataset".
               # The earlier "degenerate" matched neither.
               Patch(facecolor="none", edgecolor=INK, linewidth=0.9, label="label set fails the screen on this dataset")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=((LEFT + 2.35) / FW, 0.05 / H), frameon=False,
               fontsize=7.5, labelcolor=INK2, handlelength=1.4, handleheight=0.9, borderaxespad=0,
               labelspacing=0.35)
    save(fig, "fig1_fpt_heatmap")
    return H


# ------------------------------------------------------------------ Figure 2
def draw_fig2(ft):
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.05), facecolor=SURFACE)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.36, wspace=0.28)
    styles = [("RMS detectors, threshold 3σ or more", RMS3, dict(marker="o", s=9, facecolor=FAM_COL["detector"],
                                                                 edgecolor=SURFACE, linewidth=0.3)),
              ("ma2sig and 3sig_kurt", BETWEEN, dict(marker="o", s=9, facecolor="none", edgecolor=FAM_COL["detector"],
                                                     linewidth=0.6)),
              ("control charts", ["ewma", "cusum"], dict(marker="^", s=10, facecolor=FAM_COL["control chart"],
                                                         edgecolor=SURFACE, linewidth=0.3)),
              ("percentile rules", PCT, dict(marker="s", s=5, facecolor=FAM_COL["percentile"], edgecolor="none"))]
    for ax, (ds, label) in zip(axes, [("xjtu", "XJTU-SY"), ("femto", "FEMTO"), ("ims", "IMS")]):
        style_axes(ax, "both")
        g = ft[ft.dataset == ds]
        for _, rules, kw in styles:
            h = g[g.rule.isin(rules)]
            ax.scatter(h.n_records, np.maximum(h.fpt_index, 1), zorder=3, alpha=0.9, **kw)
        lo, hi = g.n_records.min() * 0.7, g.n_records.max() * 1.4
        ax.plot([lo, hi], [lo, hi], ls=":", color=MUTED, lw=0.7, zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.7, 5000)
        ax.set_title(f"{label} ({g.trajectory.nunique()} bearings)", color=INK, pad=3)
        ax.set_xlabel("trajectory length n (records)", color=INK2, labelpad=1)
    axes[0].set_ylabel("fire index i (records)", color=INK2, labelpad=1)
    handles = [Line2D([], [], ls="", marker=kw["marker"], ms=3.5,
                      markerfacecolor=kw["facecolor"] if kw["facecolor"] != "none" else "none",
                      markeredgecolor=kw.get("edgecolor") if kw["facecolor"] == "none" else kw["facecolor"],
                      label=name) for name, _, kw in styles]
    handles.append(Line2D([], [], ls=":", color=MUTED, lw=0.7, label="i = n"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.54, 0.0),
               handletextpad=0.3, columnspacing=1.2, labelcolor=INK2)
    save(fig, "fig2_arl_signature")


# ------------------------------------------------------------------ Figure 3
def draw_fig3():
    cov = pd.read_csv(TABLES / "j12_fpt_covariates.csv")
    order = ["3sig_rms", "ma3sig", "ma5sig", "ma2sig", "3sig_kurt", "ewma", "cusum"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.55), sharey=True, facecolor=SURFACE)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.90, bottom=0.30, wspace=0.08)
    shares = {0.25: -0.18, 0.5: 0.0, 0.75: 0.18}
    for ax, (ds, label) in zip(axes, [("xjtu", "XJTU-SY"), ("femto", "FEMTO")]):
        style_axes(ax, "x")
        ax.axvline(0, color=INK2, lw=0.6, zorder=1)
        for yi, rule in enumerate(order):
            g = cov[(cov.dataset == ds) & (cov.rule == rule)]
            col = FAM_COL[FAMILY[rule]]
            ax.plot([g.rho.min(), g.rho.max()], [yi, yi], color=col, alpha=0.35, lw=1.0, zorder=2)
            for r in g.itertuples():
                filled = r.indicator == "rms"
                ax.scatter(r.rho, yi + shares[r.rise_share], s=10, marker="o", zorder=3, linewidth=0.6,
                           facecolor=col if filled else SURFACE, edgecolor=col)
            ax.plot([g.rho.median()] * 2, [yi - 0.32, yi + 0.32], color=INK, lw=1.1, zorder=4)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_xticklabels(["-1", "-0.5", "0", "0.5", "1"])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.invert_yaxis()
        ax.set_title(label, color=INK, pad=3)
    fig.text(0.555, 0.175, "Spearman correlation between onset and rule-free marker", ha="center", va="top",
             color=INK2, fontsize=7)
    handles = [Line2D([], [], ls="", marker="o", ms=3.5, markerfacecolor=INK2, markeredgecolor=INK2, label="RMS marker"),
               Line2D([], [], ls="", marker="o", ms=3.5, markerfacecolor=SURFACE, markeredgecolor=INK2, label="kurtosis marker"),
               Line2D([], [], color=INK, lw=1.1, label="median over the six settings"),
               Patch(facecolor=FAM_COL["detector"], label="detector rule"),
               Patch(facecolor=FAM_COL["control chart"], label="control chart")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 0.0),
               handletextpad=0.35, columnspacing=1.2, handlelength=1.4, labelcolor=INK2)
    save(fig, "fig3_physics_test")


# ------------------------------------------------------------------ Figure 4
def pair_group(a, b):
    s = {a, b}
    if s <= set(RMS_DET4):
        return 0
    if "3sig_kurt" in s and (s - {"3sig_kurt"}) <= set(RMS_DET4):
        return 1
    if s <= set(PCT):
        return 3
    if s & set(PCT):
        return 2
    return None


def draw_fig4():
    d = pd.read_csv(TABLES / "j13_fpt_dispersion.csv")
    p = d[(d.kind == "pairwise") & (d.subset == "non_degenerate")].copy()
    p["group"] = [pair_group(a, b) for a, b in zip(p.rule_a, p.rule_b)]
    GROUPS = [("pairs of RMS detectors", FAM_COL["detector"], "o", FAM_COL["detector"]),
              ("kurtosis detector vs RMS detector", FAM_COL["detector"], "D", SURFACE),
              ("detector vs percentile rule", INK2, "o", INK2),
              ("pairs of percentile rules", FAM_COL["percentile"], "s", FAM_COL["percentile"])]
    fig, axes = plt.subplots(1, 4, figsize=(W, 2.35), sharey=True, facecolor=SURFACE)
    fig.subplots_adjust(left=0.10, right=0.995, top=0.90, bottom=0.25, wspace=0.08)
    rng = np.random.default_rng(4)
    for ax, (ds, label) in zip(axes, DATASETS):
        style_axes(ax, "y")
        ax.set_yscale("function", functions=(np.sqrt, np.square))
        g = p[(p.dataset == ds) & p.group.notna()]
        for grp, (_, col, marker, face) in enumerate(GROUPS):
            h = g[g.group == grp]
            if h.empty:
                ax.text(grp, 0.0015, "n/a", ha="center", va="bottom", fontsize=5.5, color=MUTED)
                continue
            xs = grp + rng.uniform(-0.22, 0.22, len(h))
            ok = h.ci_lo.notna().to_numpy()
            ax.vlines(xs[ok], h.ci_lo[ok], h.ci_hi[ok], color=col, alpha=0.22, lw=0.6, zorder=2)
            ax.scatter(xs, h["median"], s=11, marker=marker, facecolor=face, edgecolor=col, linewidth=0.6, zorder=3)
        ax.plot([2.6, 3.4], [0.05, 0.05], color=INK, lw=0.9, zorder=4)
        ax.set_xlim(-0.6, 3.6)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title(label, color=INK, pad=3)
    axes[0].set_yticks([0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5])
    axes[0].set_yticklabels(["0", "0.01", "0.05", "0.1", "0.2", "0.3", "0.5"])
    axes[0].set_ylim(0, 0.55)
    axes[0].set_ylabel("median |Δf| (fraction of life)", color=INK2, labelpad=1)
    handles = [Line2D([], [], ls="", marker=m, ms=3.8, markerfacecolor=f, markeredgecolor=c, label=n)
               for n, c, m, f in GROUPS]
    handles.append(Line2D([], [], color=INK, lw=0.9, label="exact value 0.050"))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 0.0),
               handletextpad=0.3, columnspacing=1.1, labelcolor=INK2)
    save(fig, "fig4_dispersion")


# ------------------------------------------------------------------ Figure 5
def draw_fig5():
    s = pd.read_csv(TABLES / "j15_rule_sensitivity.csv")
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.15), facecolor=SURFACE)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.34, wspace=0.40)
    mk = {1: "o", 3: "s", 5: "^"}
    ax = axes[0]
    style_axes(ax, "y")
    a = s[s.claim == "C-A_arl"]
    for fam, col in (("detector", FAM_COL["detector"]), ("control-chart", FAM_COL["control chart"])):
        q = a[a.family == fam].groupby(["base_frac", "confirm"]).value.median().reset_index()
        for cf in (1, 3, 5):
            qq = q[q.confirm == cf]
            ax.plot(qq.base_frac, qq.value, marker=mk[cf], ms=3, lw=0.9, color=col, alpha=0.9)
    ax.axhline(0, color=INK2, lw=0.6)
    # "duration signature" (2026-09-21, A2): the behavior of the alarm record with test length is named for
    # what it measures, not for the chart-design quantity (average run length) it was first compared with
    ax.set_title("duration signature", color=INK, pad=3)
    ax.set_ylabel("median ρ(index, length)", color=INK2, labelpad=1)
    ax = axes[1]
    style_axes(ax, "y")
    b = s[s.claim == "C-B_inside_baseline_gap"]
    for cf in (1, 3, 5):
        q = b[b.confirm == cf].sort_values("base_frac")
        ax.plot(q.base_frac, q.value, marker=mk[cf], ms=3, lw=0.9, color=INK2)
    ax.axhline(0.20, ls=":", color=MUTED, lw=0.8)
    ax.set_ylim(0, 0.7)
    ax.set_title("onset inside own baseline", color=INK, pad=3)
    ax.set_ylabel("chart minus detector share", color=INK2, labelpad=1)
    ax = axes[2]
    style_axes(ax, "y")
    for claim, col in (("C-C_within_family", FAM_COL["detector"]), ("C-C_between_family", FAM_COL["control chart"])):
        q = s[s.claim == claim].groupby(["base_frac", "confirm"]).value.median().reset_index()
        for cf in (1, 3, 5):
            qq = q[q.confirm == cf]
            ax.plot(qq.base_frac, qq.value, marker=mk[cf], ms=3, lw=0.9, color=col, alpha=0.9)
    ax.set_ylim(0, 0.75)
    ax.set_title("within vs between", color=INK, pad=3)
    ax.set_ylabel("median pairwise |Δf|", color=INK2, labelpad=1)
    for ax in axes:
        ax.axvline(0.20, ls=":", color=MUTED, lw=0.8, zorder=0)
        ax.set_xticks([0.1, 0.2, 0.3, 0.4])
        ax.set_xlabel("baseline fraction", color=INK2, labelpad=1)
    handles = [Line2D([], [], color=FAM_COL["detector"], lw=1.2, label="detector rules (left) / within detectors (right)"),
               Line2D([], [], color=FAM_COL["control chart"], lw=1.2, label="control charts (left) / detector vs chart (right)")]
    handles += [Line2D([], [], color=INK2, lw=0, marker=mk[cf], ms=3.5, label=f"confirmation {cf}") for cf in (1, 3, 5)]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.53, 0.0),
               handletextpad=0.4, columnspacing=1.0, labelcolor=INK2)
    save(fig, "fig5_convention_grid")


# ------------------------------------------------------------------ Figure 6
LABEL_OFFSETS = {
    ("xjtu", "pct02"): (3, 4), ("xjtu", "pct05"): (-3, 5), ("xjtu", "pct10"): (3, -7),
    ("xjtu", "ma5sig"): (-22, -2), ("xjtu", "ma3sig"): (-3, -8), ("xjtu", "3sig_rms"): (-10, 4),
    ("femto", "pct02"): (-3, -8), ("femto", "pct05"): (-5, 5), ("femto", "pct10"): (2, -7),
    ("femto", "ma5sig"): (-5, -8), ("femto", "3sig_rms"): (-11, 4), ("femto", "ma3sig"): (4, -2),
    ("femto", "3sig_kurt"): (-7, -8), ("femto", "ma2sig"): (-10, -8),
}
PANEL_TITLE = {"xjtu": "XJTU-SY · claim", "femto": "FEMTO · shown", "ims": "IMS · shown", "nsk": "NSK 6205 · shown"}
RHO_CORNER = {"xjtu": "lower right", "femto": "upper left", "ims": "upper left", "nsk": "lower right"}


def draw_fig6(a, b):
    fig = plt.figure(figsize=(W, 5.35), facecolor=SURFACE)
    gs_a = fig.add_gridspec(2, 4, left=0.085, right=0.99, top=0.915, bottom=0.53, hspace=0.12, wspace=0.10)
    gs_b = fig.add_gridspec(1, 2, left=0.25, right=0.99, top=0.40, bottom=0.075, wspace=0.07)
    stats = {}
    for ci, (ds, label) in enumerate(DATASETS):
        g = a[a.dataset == ds].sort_values("pos_rate")
        r1, r2 = spearmanr(g.pos_rate, g.best_macro_f1), spearmanr(g.pos_rate, g.best_mcc)
        stats[ds] = (r1.statistic, r1.pvalue, r2.statistic, r2.pvalue, len(g))
        for ri, (col, ylab, ylim) in enumerate((("best_macro_f1", "best macro-F1", (0.55, 1.04)),
                                                ("best_mcc", "best MCC", (0.2, 1.06)))):
            ax = fig.add_subplot(gs_a[ri, ci])
            style_axes(ax, "both")
            for fam, marker in (("detector", "o"), ("percentile", "s")):
                h = g[g.family == fam]
                ax.scatter(h.pos_rate, h[col], s=12, marker=marker, color=FAM_COL[fam], edgecolor=SURFACE,
                           linewidth=0.5, zorder=3)
            ax.set_xlim(0, 0.37)
            ax.set_ylim(*ylim)
            ax.set_xticks([0, 0.1, 0.2, 0.3])
            if ci:
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(ylab, color=INK2, labelpad=1)
            rho = r1.statistic if ri == 0 else r2.statistic
            corner = RHO_CORNER[ds]
            tx, ha = (0.97, "right") if "right" in corner else (0.04, "left")
            ty, va = (0.04, "bottom") if "lower" in corner else (0.96, "top")
            ax.text(tx, ty, f"ρ = {rho:+.2f} ({len(g)})", transform=ax.transAxes, ha=ha, va=va, fontsize=6,
                    color=INK if ds == "xjtu" else INK2, fontweight="bold" if ds == "xjtu" else "normal")
            if ri == 0:
                ax.tick_params(labelbottom=False)
                ax.set_title(PANEL_TITLE[ds], color=INK, pad=3)
                if ds in ("xjtu", "femto"):
                    for (xv, yv), names in g.groupby(["pos_rate", col]).rule.apply(lambda s: ", ".join(s)).items():
                        dx, dy = LABEL_OFFSETS.get((ds, names.split(", ")[0]), (3, 3))
                        ax.annotate(names, (xv, yv), xytext=(dx, dy), textcoords="offset points", fontsize=5,
                                    color=INK2)
            else:
                ax.set_xlabel("positive rate", color=INK2, labelpad=1)
    fig.text(0.01, 0.955, "A", fontsize=9, fontweight="bold", color=INK)
    fig.text(0.01, 0.445, "B", fontsize=9, fontweight="bold", color=INK)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.5, color=FAM_COL["detector"], label="detector rule"),
                        Line2D([], [], marker="s", ls="", ms=3.5, color=FAM_COL["percentile"], label="percentile rule")],
               loc="upper right", bbox_to_anchor=(0.99, 0.995), ncol=2, frameon=False, labelcolor=INK2,
               handletextpad=0.3, columnspacing=1.0)
    titles = {"xjtu": "XJTU-SY", "femto": "FEMTO"}
    for pi, ds in enumerate(("xjtu", "femto")):
        ax = fig.add_subplot(gs_b[0, pi])
        style_axes(ax, "x")
        g = b[b.dataset == ds]
        ypos, yl, heads, y = [], [], [], 0.0
        for group, methods in METHOD_GROUPS:
            heads.append((y - 0.8, group))
            for m in methods:
                row = g[g.method == m].iloc[0]
                ypos.append(y)
                yl.append(METHOD_NAME[m])
                if np.isfinite(row.gain):
                    emph = m == "thr:rms"
                    ax.barh(y, row.gain, height=0.62, color=FAM_COL["detector"] if emph else MUTED, linewidth=0, zorder=3)
                    ax.text(row.gain + 0.006, y, f"{row.gain:+.3f}", va="center", ha="left", fontsize=5.8,
                            color=INK if emph else INK2)
                else:
                    ax.text(0.006, y, "not in this feature set", va="center", ha="left", fontsize=5.5, color=MUTED,
                            style="italic")
                y += 1
            y += 0.85
        for hy, text in heads:
            ax.text(0.0, hy, text, ha="left", va="center", fontsize=5.8, color=INK, fontweight="bold")
        ax.set_yticks(ypos)
        ax.set_yticklabels(yl if pi == 0 else [""] * len(yl), fontsize=6)
        ax.set_ylim(y - 0.5, -1.4)
        ax.set_xlim(0, 0.39)
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.tick_params(axis="y", length=0)
        ax.set_title(titles[ds], color=INK, pad=3, loc="left")
    fig.text((0.25 + 0.99) / 2, 0.012, "macro-F1 gain, RMS-defined rules minus percentile rules (mean of three each)",
             ha="center", va="bottom", fontsize=6.5, color=INK2)
    save(fig, "fig6_prevalence_circularity")
    return stats


# ------------------------------------------------------------------ main
def main():
    global _fh
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    LOGS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j14_run.log", "w", encoding="utf-8")
    bar = "=" * 78
    log(bar)
    log("J14  JOURNAL FIGURES 1-6 at text width (5.4 in), from saved tables only")
    log(bar)
    ft, perf = load()
    mat = build_fig1_matrix(ft, perf)
    mat.to_csv(TABLES / "fig1_fpt_matrix.csv", index=False)
    log(f"Figure 1 matrix: {mat.groupby(['dataset', 'trajectory']).ngroups} trajectories x {len(RULES)} rules = {len(mat)} cells")
    arl = pd.read_csv(TABLES / "j11_arl_signature.csv")
    cnt = mat[mat.fired].groupby(["dataset", "rule"]).inside_baseline.sum()
    bad = [r for r in arl.itertuples() if int(cnt.get((r.dataset, r.rule), -1)) != int(r.n_inside_baseline)]
    log(f"  inside-baseline counts vs j11_arl_signature.csv: {len(arl) - len(bad)}/{len(arl)} rows agree")
    per = (mat[mat.fired & mat.dataset.isin(["xjtu", "femto", "ims"]) & (mat.family != "percentile")]
           .groupby(["dataset", "rule", "family"]).inside_baseline.agg(["mean", "size"]).reset_index())
    per = per[per["size"] >= MIN_TRAJ_FOR_RATE]
    log("  row-mean inside-baseline by family, J11 convention: "
        + ", ".join(f"{k} {v:.3f}" for k, v in per.groupby("family")["mean"].mean().items()))
    h1 = draw_fig1(mat)
    log(f"  figure 1 height {h1:.2f} in")
    draw_fig2(ft)
    draw_fig3()
    draw_fig4()
    draw_fig5()
    a, b = build_fig6a(perf), build_fig6b(perf)
    a.to_csv(TABLES / "fig6a_prevalence.csv", index=False)
    b.to_csv(TABLES / "fig6b_circularity_gain.csv", index=False)
    stats = draw_fig6(a, b)
    for ds, _ in DATASETS:
        r1, p1, r2, p2, n = stats[ds]
        log(f"  6A {ds:5s} n={n} rho(F1) {r1:+.3f} p={p1:.4f} | rho(MCC) {r2:+.3f} p={p2:.4f}")
    log(bar)
    _fh.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
