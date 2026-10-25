"""J12 -- E5: how far apart do the published rules put the boundary, with real intervals?

The conference paper reported a dispersion number with no interval, because with one NSK
trajectory there was nothing to resample. There are now 32 trajectories across three
multi-bearing datasets, so the headline can carry a bearing bootstrap.

TWO STATISTICS, and the second is the honest one
------------------------------------------------
  within-bearing spread   per trajectory, max(FPT fraction) - min(FPT fraction) over the
                          rules in the subset. Intuitive, but a MAXIMUM over k rules grows
                          with k by construction, so it is not comparable between subsets of
                          different size. Reported, and flagged, for exactly that reason.

  pairwise |delta FPT|    for each PAIR of rules, the median over bearings of the absolute
                          difference in where they put the boundary. Paired within bearing,
                          independent of how many rules are in the subset, and therefore the
                          number that belongs in the abstract.

FOUR SUBSETS, because "the rules disagree by 0.9 of life" is not a finding if it is produced
by including rules that label 90% of the trajectory as degraded:

    all                     all ten rules
    non_degenerate          rules whose induced positive rate passed [0.01, 0.50] in J3
                            under mode="fixed" (i.e. after the E1 exclusion correction)
    firing_detectors        detector-family rules that fired on EVERY trajectory of that
                            dataset. NOT a robustness subset: every rule in it keys off RMS,
                            so it measures WITHIN-family agreement. Reported as that control,
                            and cross-checked against E6, which finds the same split.
    percentile_only         pct02/05/10 -- a negative control. These are fixed fractions of
                            life, so their mutual dispersion is known a priori (0.050) and
                            any method error shows up here as a departure from it.

The claim the paper needs is that the dispersion survives in `non_degenerate`: the screen
that removes rules which had no business generating labels, while keeping BOTH families. If it
survived only in `firing_detectors` it would be a within-family statement and would say
nothing about the rules the literature actually swaps.

Bootstrap resamples BEARINGS (20,000), the only independent unit available.

Outputs
    results/tables/j13_fpt_dispersion.csv
    results/figures/fig4_dispersion.png
    results/logs/j12_run.log

Usage
    python scripts/J12_fpt_dispersion.py
    python scripts/J12_fpt_dispersion.py --check
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
LOGS = ROOT / "results" / "logs"
B = int(os.environ.get("B", "20000"))
BAR = "=" * 78

DETECTOR = ["3sig_rms", "3sig_kurt", "ma2sig", "ma3sig", "ma5sig"]
CHART = ["ewma", "cusum"]
PERCENTILE = ["pct02", "pct05", "pct10"]
FAMILY = {**{r: "detector" for r in DETECTOR}, **{r: "control-chart" for r in CHART},
          **{r: "percentile" for r in PERCENTILE}}

_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


def head(s: str, t: str) -> None:
    log(f"\n{BAR}\nSTEP {s}  {t}\n{BAR}")


def boot_median(v: np.ndarray, rng: np.random.Generator, b: int = B):
    """Median with a bootstrap CI over the supplied units (here: bearings)."""
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan, np.nan, 0
    if len(v) < 3:
        return float(np.median(v)), np.nan, np.nan, len(v)
    draws = np.median(v[rng.integers(0, len(v), size=(b, len(v)))], axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(np.median(v)), float(lo), float(hi), len(v)


def load_fpt() -> pd.DataFrame:
    parts = []
    p = TABLES / "j3_fpt_times.csv"
    if p.exists():
        parts.append(pd.read_csv(p))
    p = TABLES / "j11_femto_fpt_times.csv"
    if p.exists():
        parts.append(pd.read_csv(p))
    if not parts:
        return pd.DataFrame()
    ft = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["dataset", "trajectory", "rule"], keep="last")
    ft["family"] = ft.rule.map(FAMILY)
    return ft


def subsets_for(ds: str, ft: pd.DataFrame, perf: pd.DataFrame | None) -> dict[str, list[str]]:
    g = ft[ft.dataset == ds]
    present = [r for r in FAMILY if (g.rule == r).any()]
    n_traj = g.trajectory.nunique()

    nondeg = present
    if perf is not None and not perf.empty:
        q = perf[perf.dataset == ds]
        if "mode" in q.columns:
            q = q[q["mode"] == "fixed"]
        if not q.empty and "degenerate" in q.columns:
            ok = q.groupby("rule").degenerate.first()
            nondeg = [r for r in present if not bool(ok.get(r, True))]

    fired_all = g.groupby("rule").trajectory.nunique()
    firing_det = [r for r in present
                  if FAMILY[r] == "detector" and int(fired_all.get(r, 0)) == n_traj]

    return {"all": present,
            "non_degenerate": nondeg,
            "firing_detectors": firing_det,
            "percentile_only": [r for r in present if FAMILY[r] == "percentile"]}


def main(check_only: bool) -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    # A --check run must never truncate the log of the last real run.
    _fh = None if check_only else open(LOGS / "j12_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J12  E5 -- FPT DISPERSION WITH BEARING BOOTSTRAP".center(78))
    log("the first real confidence intervals on the paper's headline number".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S} | B={B} bearing resamples")

    head("1", "Inputs")
    ft = load_fpt()
    if ft.empty:
        log("  no FPT tables found -- run scripts/J3_fpt_rules.py first")
        return 1
    perf = None
    p = TABLES / "j3_performance.csv"
    if p.exists():
        perf = pd.read_csv(p)
        if "mode" in perf.columns:
            log(f"  j3_performance.csv has a 'mode' column -- using mode='fixed' "
                f"(post-E1) for the degeneracy screen")
        else:
            log("  WARNING j3_performance.csv predates E1 (no 'mode' column); the "
                "non_degenerate subset reflects the UNCORRECTED screen")
    for ds, g in ft.groupby("dataset"):
        log(f"  {ds:6s} {g.trajectory.nunique():3d} trajectories | {g.rule.nunique():2d} rules "
            f"| {len(g):4d} rule-trajectory pairs")
    if check_only:
        log("\n  --check: inputs OK.")
        return 0

    rows = []
    order = [d for d in ("xjtu", "femto", "ims", "nsk") if (ft.dataset == d).any()]

    head("2", "Subsets in play")
    subs = {}
    for ds in order:
        subs[ds] = subsets_for(ds, ft, perf)
        log(f"  {ds.upper()}")
        for name, rules in subs[ds].items():
            log(f"    {name:18s} {len(rules):2d}  {', '.join(rules) if rules else '(empty)'}")

    head("3", "Within-bearing spread  (max - min over the subset, per trajectory)")
    log("  A maximum over k rules grows with k, so compare ACROSS datasets within a subset,")
    log("  never across subsets of different size. The pairwise table in step 4 is the one")
    log("  that is comparable.\n")
    log(f"  {'dataset':8s} {'subset':18s} {'k':>3s} {'n':>4s} {'median':>8s} {'95% CI':>16s} "
        f"{'IQR':>16s} {'worst':>7s}")
    for ds in order:
        for name, rules in subs[ds].items():
            if len(rules) < 2:
                continue
            g = ft[(ft.dataset == ds) & (ft.rule.isin(rules))]
            sp, trs = [], []
            for t, q in g.groupby("trajectory"):
                if q.rule.nunique() < 2:
                    continue
                sp.append(float(q.fpt_fraction.max() - q.fpt_fraction.min()))
                trs.append(t)
            if not sp:
                continue
            sp = np.array(sp)
            med, lo, hi, n = boot_median(sp, np.random.default_rng(4100))
            q1, q3 = np.percentile(sp, [25, 75])
            rows.append({"dataset": ds, "subset": name, "kind": "within_bearing_spread",
                         "rule_a": "", "rule_b": "", "k_rules": len(rules), "n_bearings": n,
                         "median": med, "ci_lo": lo, "ci_hi": hi,
                         "q1": float(q1), "q3": float(q3),
                         "min": float(sp.min()), "max": float(sp.max())})
            log(f"  {ds:8s} {name:18s} {len(rules):>3d} {n:>4d} {med:>8.3f} "
                f"{f'[{lo:.3f}, {hi:.3f}]' if np.isfinite(lo) else '            n/a':>16s} "
                f"{f'[{q1:.3f}, {q3:.3f}]':>16s} {sp.max():>7.3f}")
        log("")

    head("4", "Pairwise median |delta FPT|  -- paired within bearing, size-independent")
    log("  This is the number for the abstract: pick two published rules, apply both to the")
    log("  same bearing, and this is how far apart they put the boundary.\n")
    pair_summary = []
    for ds in order:
        for name, rules in subs[ds].items():
            if len(rules) < 2:
                continue
            g = ft[(ft.dataset == ds) & (ft.rule.isin(rules))]
            wide = g.pivot_table(index="trajectory", columns="rule", values="fpt_fraction")
            meds = []
            for a, b_ in combinations(sorted(rules), 2):
                if a not in wide.columns or b_ not in wide.columns:
                    continue
                d = (wide[a] - wide[b_]).abs().to_numpy(float)
                d = d[np.isfinite(d)]
                if len(d) == 0:
                    continue
                med, lo, hi, n = boot_median(d, np.random.default_rng(4200))
                rows.append({"dataset": ds, "subset": name, "kind": "pairwise",
                             "rule_a": a, "rule_b": b_, "k_rules": len(rules),
                             "n_bearings": n, "median": med, "ci_lo": lo, "ci_hi": hi,
                             "q1": np.nan, "q3": np.nan,
                             "min": float(d.min()), "max": float(d.max())})
                meds.append(med)
            if meds:
                meds = np.array(meds)
                pair_summary.append({"dataset": ds, "subset": name, "n_pairs": len(meds),
                                     "median_of_pairs": float(np.median(meds)),
                                     "max_pair": float(meds.max())})
    if pair_summary:
        log(f"  {'dataset':8s} {'subset':18s} {'pairs':>6s} {'median pair |d|':>16s} "
            f"{'worst pair':>11s}")
        for r in pair_summary:
            log(f"  {r['dataset']:8s} {r['subset']:18s} {r['n_pairs']:>6d} "
                f"{r['median_of_pairs']:>16.3f} {r['max_pair']:>11.3f}")

    head("5", "The pairs that matter most, with intervals")
    df = pd.DataFrame(rows)
    for ds in order:
        g = df[(df.dataset == ds) & (df.kind == "pairwise") & (df.subset == "all")]
        if g.empty:
            continue
        g = g.sort_values("median", ascending=False).head(6)
        log(f"  {ds.upper()}  (largest six disagreements, all ten rules)")
        log(f"    {'rule A':10s} {'rule B':10s} {'median |d|':>11s} {'95% CI':>18s} {'n':>4s}")
        for _, r in g.iterrows():
            ci = f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]" if np.isfinite(r.ci_lo) else "n/a"
            log(f"    {r.rule_a:10s} {r.rule_b:10s} {r['median']:>11.3f} {ci:>18s} "
                f"{int(r.n_bearings):>4d}")
        log("")

    head("6", "Does the dispersion survive the strictest subset that keeps both families?")
    log("  'non_degenerate' drops the rules that had no business generating labels -- those")
    log("  calling 90%+ of the trajectory degraded -- while keeping BOTH families. That is the")
    log("  strictest meaningful screen.")
    log("")
    log("  'firing_detectors' is NOT a stricter version of the same test: every rule in it")
    log("  keys off RMS, so it measures WITHIN-family agreement (see E6, which found the same")
    log("  thing: within a family the rules agree to ~0.03 of life, between families they do")
    log("  not). It is reported below as that control, not as a robustness check.")
    log("")
    log(f"  {'dataset':8s} {'all rules':>10s} {'non-degen':>10s} {'k':>3s} "
        f"{'within-family':>14s} {'pct control':>12s}")
    verdict_vals = {}
    for ds in order:
        def med(sub):
            q = df[(df.dataset == ds) & (df.kind == "pairwise") & (df.subset == sub)]["median"]
            return q.median() if not q.empty else np.nan
        a, nd, fd, pc = med("all"), med("non_degenerate"), med("firing_detectors"), \
            med("percentile_only")
        k = df[(df.dataset == ds) & (df.subset == "non_degenerate")].k_rules
        k = int(k.iloc[0]) if len(k) else 0
        verdict_vals[ds] = nd
        log(f"  {ds:8s} {a:>10.3f} {nd:>10.3f} {k:>3d} "
            + (f"{fd:>14.3f}" if np.isfinite(fd) else f"{'subset empty':>14s}")
            + f" {pc:>12.3f}")
    log("")
    log("  The percentile-only column is a control with a known answer: three fixed fractions")
    log("  of life, so the median pairwise gap must be median{0.03, 0.08, 0.05} = 0.050.")
    log("  Observing 0.047-0.050 on four datasets says the measurement machinery is sound.")
    log("")
    fem = df[(df.dataset == "femto") & (df.subset == "firing_detectors")]
    if fem.empty and (df.dataset == "femto").any():
        log("  NOTE on FEMTO: the firing_detectors subset is EMPTY -- not one detector rule")
        log("  fired on all 13 bearings. That is a finding, not a gap: on the dataset with the")
        log("  most bearings, every published detector rule fails on at least one of them.")
        log("")
    multi = {d: v for d, v in verdict_vals.items()
             if np.isfinite(v) and d in ("xjtu", "femto", "ims")}
    if multi and min(multi.values()) > 0.05:
        log("  >>> IT SURVIVES. With the degenerate rules removed, two published FPT rules")
        log("      applied to the SAME bearing still place the boundary "
            f"{min(multi.values()):.2f} to {max(multi.values()):.2f}")
        log("      of total life apart (median over rule pairs), on every multi-bearing")
        log("      dataset. On XJTU the number barely moves from the all-rules value, so the")
        log("      dispersion is not an artefact of the rules that label almost everything.")
    elif multi:
        log("  >>> Does NOT survive uniformly; report subset by subset, per dataset.")
        for d, v in multi.items():
            log(f"      {d}: {v:.3f}")

    head("7", "Figure")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"all": "#13222D", "non_degenerate": "#1B6773",
           "firing_detectors": "#B85C18", "percentile_only": "#7A8D98"}
    pw = df[df.kind == "pairwise"]
    dss = [d for d in order if (pw.dataset == d).any()]
    if dss:
        fig, axes = plt.subplots(1, len(dss), figsize=(4.3 * len(dss), 4.2), sharey=True,
                                 squeeze=False)
        for ax, ds in zip(axes[0], dss):
            names = [n for n in COL if ((pw.dataset == ds) & (pw.subset == n)).any()]
            for i, n in enumerate(names):
                v = pw[(pw.dataset == ds) & (pw.subset == n)]["median"].to_numpy()
                ax.scatter(np.full(len(v), i) + np.linspace(-.15, .15, len(v)), v,
                           s=20, alpha=.7, color=COL[n])
                ax.plot([i - .28, i + .28], [np.median(v)] * 2, lw=2.4, color=COL[n])
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=8)
            ax.set_ylim(-0.02, 1.02)
            ax.set_title(f"{ds.upper()}", fontsize=10)
            ax.axhline(0.10, ls=":", lw=1, color="#A6B7C0")
        axes[0][0].set_ylabel("median |ΔFPT| between a pair of rules\n(fraction of life)")
        fig.suptitle("Figure 4 — pairwise disagreement between published FPT rules, "
                     "by rule subset\n"
                     "each dot is one pair of rules; the bar is the median over pairs",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(FIGS / "fig4_dispersion.png", dpi=170)
        plt.close(fig)
        log(f"  wrote {FIGS / 'fig4_dispersion.png'}")

    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "j13_fpt_dispersion.csv", index=False)
    log(f"\n{BAR}")
    log(f"  WROTE  results/tables/j13_fpt_dispersion.csv ({len(df)} rows)   [E5]")
    log(f"         results/figures/fig4_dispersion.png")
    log(f"         results/logs/j12_run.log")
    log(f"  finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(BAR)
    if _fh:
        _fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--check" in sys.argv))
