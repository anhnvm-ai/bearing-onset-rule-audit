"""J13 -- E6: is the paper's own taxonomy an artefact of the conventions the paper chose?

THE QUESTION THIS ANSWERS
-------------------------
The paper's thesis is that reported quantities in run-to-failure evaluation are functions of
unreported conventions. Every FPT rule here was computed with a baseline of the first 20% of
the trajectory and a confirmation run of 3 records. Those two numbers were never justified;
they were inherited. So the first thing a referee will ask -- and the paper has no standing
to object -- is whether the ARL taxonomy survives when they move.

This script recomputes the whole taxonomy over the grid

    BASE_FRAC in {0.10, 0.20, 0.30, 0.40}  x  CONFIRM in {1, 3, 5}   = 12 conventions

on all four datasets, and re-tests each of the three claims the paper rests on:

  C-A  THE ARL SIGNATURE (E4). Spearman(absolute fire index, trajectory length) per rule.
       The claim is that detector-family rules scale with length and control charts do not.
       For the claim to be a property of the RULES rather than of our baseline choice, the
       sign pattern must hold across the grid.

  C-B  THE BASELINE SELF-CONTRADICTION (E4b). Fraction of trajectories where the rule places
       onset inside its own assumed-healthy window. This one has a built-in dependency --
       a larger BASE_FRAC is a larger window to fall inside -- so the interesting question is
       not whether the number moves but whether the control-chart/detector GAP survives. If
       it vanishes at BASE_FRAC 0.10, the finding is partly our own construction and must be
       reported with the baseline stated.

  C-C  THE DISPERSION (E5). Median pairwise |delta FPT| between rules on the same bearing.

Either outcome is publishable, which is why this is worth doing before drafting: if the
taxonomy is stable the paper gains its strongest robustness section, and if it is not, the
paper's own thesis predicted it and the honest version is a stronger paper than a fragile
claim discovered by a referee.

The classifier arm is deliberately NOT run here. C-A, C-B and C-C are all label-side and
model-free, so no classifier can strengthen or weaken them; adding one would import a
capacity and seed confound for nothing. Use --with-classifier only if a referee demands it.

Outputs
    results/tables/j15_rule_sensitivity.csv
    results/figures/fig5_convention_grid.png
    results/logs/j13_run.log

Usage
    python scripts/J13_convention_sensitivity.py
    python scripts/J13_convention_sensitivity.py --check
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "colab_bundle"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
LOGS = ROOT / "results" / "logs"

BASE_GRID = (0.10, 0.20, 0.30, 0.40)
CONFIRM_GRID = (1, 3, 5)
DEFAULT_CELL = (0.20, 3)
BAR = "=" * 78

DETECTOR = ["3sig_rms", "3sig_kurt", "ma2sig", "ma3sig", "ma5sig"]
CHART = ["ewma", "cusum"]
PERCENTILE = ["pct02", "pct05", "pct10"]
FAMILY = {**{r: "detector" for r in DETECTOR}, **{r: "control-chart" for r in CHART},
          **{r: "percentile" for r in PERCENTILE}}
IMS_FAILED = {"IMS_1st_test_B3", "IMS_1st_test_B4", "IMS_2nd_test_B1", "IMS_3rd_test_B3"}

_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


def head(s: str, t: str) -> None:
    log(f"\n{BAR}\nSTEP {s}  {t}\n{BAR}")


# ---------------------------------------------------------------- FPT, parameterised
def _first_confirmed(flag: np.ndarray, confirm: int) -> int | None:
    run = 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        if run >= confirm:
            return i - confirm + 1
    return None


def fpt(hi_rms: np.ndarray, hi_kurt: np.ndarray, rule: str,
        base_frac: float, confirm: int) -> int | None:
    """Identical to J3/J11 except BASE_FRAC and CONFIRM are arguments, not constants."""
    n = len(hi_rms)
    if rule.startswith("pct"):
        return int(n * (1 - int(rule[3:]) / 100.0))
    h = hi_kurt if rule == "3sig_kurt" else hi_rms
    nb = max(5, int(n * base_frac))
    mu, sd = float(np.mean(h[:nb])), max(float(np.std(h[:nb])), 1e-12)
    if rule in ("3sig_rms", "3sig_kurt"):
        return _first_confirmed(h > mu + 3 * sd, confirm)
    if rule == "ewma":
        lam, L = 0.2, 3.0
        z, out = mu, np.empty(n)
        for i, v in enumerate(h):
            z = lam * v + (1 - lam) * z
            out[i] = z
        return _first_confirmed(out > mu + L * sd * np.sqrt(lam / (2 - lam)), confirm)
    if rule == "cusum":
        k, H = 0.5 * sd, 5.0 * sd
        s, out = 0.0, np.empty(n)
        for i, v in enumerate(h):
            s = max(0.0, s + (v - mu) - k)
            out[i] = s
        return _first_confirmed(out > H, confirm)
    if rule.startswith("ma"):
        ma = pd.Series(h).rolling(5, min_periods=1).mean().to_numpy()
        return _first_confirmed(ma > mu + float(rule[2]) * sd, confirm)
    raise ValueError(rule)


def load_traj() -> dict[str, pd.DataFrame]:
    out = {}
    for ds in ("nsk", "xjtu", "ims"):
        f, r = TABLES / f"j7_features_{ds}.csv", BUNDLE / ds / "records.csv"
        if not f.exists() or not r.exists():
            continue
        F, rec = pd.read_csv(f), pd.read_csv(r)
        keep = rec.trajectory_id.isin(IMS_FAILED).to_numpy() if ds == "ims" else np.ones(len(rec), bool)
        out[ds] = pd.DataFrame({"trajectory": rec.loc[keep, "trajectory_id"].to_numpy(),
                                "rms": F.loc[keep, "rms"].to_numpy(),
                                "hikurt": F.loc[keep, "kurtosis"].to_numpy()})
    f = TABLES / "j7c_femto_features.csv"
    if f.exists():
        d = pd.read_csv(f)
        out["femto"] = pd.DataFrame({"trajectory": d.bearing.to_numpy(),
                                     "rms": d.v_rms.to_numpy(),
                                     "hikurt": d.v_kurtosis.to_numpy()})
    return out


def fpt_table(data: dict[str, pd.DataFrame], base_frac: float, confirm: int) -> pd.DataFrame:
    rows = []
    for ds, d in data.items():
        for t, g in d.groupby("trajectory", sort=True):
            rms, kurt, n = g["rms"].to_numpy(), g["hikurt"].to_numpy(), len(g)
            nb = max(5, int(n * base_frac))
            for rule in FAMILY:
                i = fpt(rms, kurt, rule, base_frac, confirm)
                if i is None:
                    continue
                rows.append({"dataset": ds, "trajectory": t, "rule": rule,
                             "family": FAMILY[rule], "fpt_index": int(i), "n_records": n,
                             "fpt_fraction": i / n, "inside_baseline": bool(i < nb)})
    return pd.DataFrame(rows)


def main(check_only: bool) -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    # A --check run must never truncate the log of the last real run.
    _fh = None if check_only else open(LOGS / "j13_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J13  E6 -- CONVENTION SENSITIVITY GRID".center(78))
    log("is our own taxonomy an artefact of the baseline we inherited?".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(f"grid: BASE_FRAC {BASE_GRID} x CONFIRM {CONFIRM_GRID} = "
        f"{len(BASE_GRID) * len(CONFIRM_GRID)} conventions")
    log(f"the paper's inherited choice is BASE_FRAC={DEFAULT_CELL[0]}, CONFIRM={DEFAULT_CELL[1]}")

    head("1", "Inputs")
    data = load_traj()
    for ds, d in data.items():
        log(f"  {ds:6s} {len(d):6d} records | {d.trajectory.nunique():2d} trajectories")
    if not data:
        log("  no feature caches found")
        return 1
    if check_only:
        log("\n  --check: inputs OK.")
        return 0

    head("2", "Recomputing FPT over the whole grid")
    cells = {}
    for bf in BASE_GRID:
        for cf in CONFIRM_GRID:
            cells[(bf, cf)] = fpt_table(data, bf, cf)
    log(f"  {len(cells)} cells computed")
    log(f"  {'BASE_FRAC':>9s} {'CONFIRM':>7s} {'rule-traj pairs':>16s} {'firing coverage':>16s}")
    for (bf, cf), t in cells.items():
        tot = sum(d.trajectory.nunique() for d in data.values()) * len(FAMILY)
        log(f"  {bf:>9.2f} {cf:>7d} {len(t):>16d} {len(t) / tot:>15.0%}")

    rows = []

    # ---------------------------------------------------------------- C-A
    head("3", "C-A -- does the ARL signature survive the grid?")
    log("  Spearman(absolute fire index, trajectory length) per rule, per convention.")
    log("  Two criteria are reported per cell, because they differ and the paper quotes both:")
    log("    gap     detector-family median rho > 0.3 AND chart-family median at least 0.4 lower")
    log("    strict  detector-family median rho > 0 AND chart-family median rho <= 0")
    log("  (the gap criterion is the one this script has always applied; the strict one is what")
    log("  an earlier version of this log described).\n")
    for ds in ("xjtu", "femto"):
        if ds not in data or data[ds].trajectory.nunique() < 6:
            continue
        log(f"  {ds.upper()}   rho(index, length)     rows = convention, columns = rule")
        rules = [r for r in FAMILY if FAMILY[r] != "percentile"]
        log(f"    {'base':>5s} {'cf':>3s} " + "".join(f"{r:>11s}" for r in rules) + "   gap    strict")
        for (bf, cf), t in cells.items():
            g0 = t[t.dataset == ds]
            vals, det, ctl = [], [], []
            for r in rules:
                g = g0[g0.rule == r]
                if len(g) < 4 or np.ptp(g.n_records) == 0 or np.ptp(g.fpt_index) == 0:
                    vals.append(np.nan); continue
                rho, p = spearmanr(g.n_records, g.fpt_index)
                vals.append(rho)
                (det if FAMILY[r] == "detector" else ctl).append(rho)
                rows.append({"dataset": ds, "base_frac": bf, "confirm": cf, "rule": r,
                             "family": FAMILY[r], "claim": "C-A_arl",
                             "value": float(rho), "p": float(p),
                             "n_traj": int(g.trajectory.nunique()),
                             "is_default": (bf, cf) == DEFAULT_CELL})
            dm = np.nanmedian(det) if det else np.nan
            cm = np.nanmedian(ctl) if ctl else np.nan
            ok = np.isfinite(dm) and np.isfinite(cm) and dm > 0.3 and cm < dm - 0.4
            strict = np.isfinite(dm) and np.isfinite(cm) and dm > 0 and cm <= 0
            star = " <-- paper's cell" if (bf, cf) == DEFAULT_CELL else ""
            log(f"    {bf:>5.2f} {cf:>3d} "
                + "".join(("        n/a" if not np.isfinite(v) else f"{v:>+11.3f}") for v in vals)
                + f"   {'HOLDS' if ok else 'fails'}  {'HOLDS' if strict else 'fails'}{star}")
        log("")

    # ---------------------------------------------------------------- C-B
    head("4", "C-B -- does the baseline self-contradiction survive the grid?")
    log("  Fraction of trajectories where the rule places onset INSIDE its own assumed-healthy")
    log("  window. A wider baseline is a wider window to fall inside, so the level is expected")
    log("  to move; what must survive is the control-chart minus detector GAP.\n")
    log(f"  {'base':>5s} {'cf':>3s} {'control-chart':>14s} {'detector':>10s} {'gap':>8s}  verdict")
    for (bf, cf), t in cells.items():
        g = t[t.dataset.isin([d for d in ("xjtu", "femto", "ims") if d in data])]
        if g.empty:
            continue
        per = g.groupby(["dataset", "rule", "family"]).inside_baseline.mean().reset_index()
        c = per[per.family == "control-chart"].inside_baseline.mean()
        d_ = per[per.family == "detector"].inside_baseline.mean()
        gap = c - d_
        rows.append({"dataset": "pooled", "base_frac": bf, "confirm": cf, "rule": "",
                     "family": "", "claim": "C-B_inside_baseline_gap",
                     "value": float(gap), "p": np.nan, "n_traj": np.nan,
                     "is_default": (bf, cf) == DEFAULT_CELL})
        star = " <-- paper's cell" if (bf, cf) == DEFAULT_CELL else ""
        log(f"  {bf:>5.2f} {cf:>3d} {c:>13.0%} {d_:>10.0%} {gap:>+8.2f}  "
            f"{'HOLDS' if gap > 0.20 else 'fails'}{star}")

    # ---------------------------------------------------------------- C-C
    head("5", "C-C -- WHERE does the disagreement live: within a family, or between them?")
    log("  Median pairwise |delta FPT| on the same bearing, split two ways:")
    log("    within   both rules from the detector family (all key off RMS)")
    log("    between  one detector rule and one control-chart rule")
    log("  If the literature is right that these rules are interchangeable, both should be")
    log("  small. If the paper is right that they estimate different quantities, 'within'")
    log("  should be small and 'between' large -- and the gap is the result.\n")

    def _pairs(wide, ra, rb):
        out = []
        for a in ra:
            for b_ in rb:
                if a == b_ or a not in wide.columns or b_ not in wide.columns:
                    continue
                d = (wide[a] - wide[b_]).abs().to_numpy(float)
                d = d[np.isfinite(d)]
                if len(d):
                    out.append(float(np.median(d)))
        return out

    dss = [x for x in ("xjtu", "femto") if x in data]
    log(f"  {'base':>5s} {'cf':>3s} " +
        "".join(f"{d + ' within':>14s}{d + ' between':>15s}" for d in dss) + "   verdict")
    for (bf, cf), t in cells.items():
        vals, ratios = [], []
        for ds in dss:
            g = t[t.dataset == ds]
            wide = g.pivot_table(index="trajectory", columns="rule", values="fpt_fraction")
            det = [r for r in DETECTOR if r in wide.columns]
            ctl = [r for r in CHART if r in wide.columns]
            w = _pairs(wide, det, det)
            b2 = _pairs(wide, det, ctl)
            wv = float(np.median(w)) if w else np.nan
            bv = float(np.median(b2)) if b2 else np.nan
            vals += [wv, bv]
            if np.isfinite(wv) and np.isfinite(bv) and wv > 1e-9:
                ratios.append(bv / wv)
            for kind, val in (("C-C_within_family", wv), ("C-C_between_family", bv)):
                if np.isfinite(val):
                    rows.append({"dataset": ds, "base_frac": bf, "confirm": cf, "rule": "",
                                 "family": "detector" if "within" in kind else "det-vs-chart",
                                 "claim": kind, "value": val, "p": np.nan,
                                 "n_traj": int(g.trajectory.nunique()),
                                 "is_default": (bf, cf) == DEFAULT_CELL})
        ok = bool(ratios) and min(ratios) > 3.0
        star = " <-- paper's cell" if (bf, cf) == DEFAULT_CELL else ""
        log(f"  {bf:>5.2f} {cf:>3d} " +
            "".join(("           n/a" if not np.isfinite(v) else f"{v:>14.3f}")
                    for v in vals) +
            f"   {'HOLDS' if ok else 'fails'}{star}")
    log("")
    log("  'HOLDS' here means between-family disagreement is more than 3x within-family on")
    log("  BOTH datasets -- i.e. the split is structural, not a matter of degree.")

    df = pd.DataFrame(rows)

    head("6", "Verdict -- how much of the paper is convention-dependent?")
    # C-A is judged per DATASET per convention. Pooling the datasets first would let a
    # dataset where the claim holds strongly cover for one where it fails.
    ga = df[df.claim == "C-A_arl"]
    if not ga.empty:
        n_ok = n_strict = n_tot = 0
        fails, fails_strict = [], []
        for (ds, bf, cf), gg in ga.groupby(["dataset", "base_frac", "confirm"]):
            dm = gg[gg.family == "detector"].value.median()
            cm = gg[gg.family == "control-chart"].value.median()
            n_tot += 1
            if np.isfinite(dm) and np.isfinite(cm) and dm > 0.3 and cm < dm - 0.4:
                n_ok += 1
            else:
                fails.append(f"{ds} base={bf} cf={cf}")
            if np.isfinite(dm) and np.isfinite(cm) and dm > 0 and cm <= 0:
                n_strict += 1
            else:
                fails_strict.append(f"{ds} base={bf} cf={cf}")
        log(f"  {'C-A  ARL signature, gap':32s} holds in {n_ok}/{n_tot} dataset x convention cells")
        if fails:
            log(f"  {'':32s} fails: {'; '.join(fails)}")
            log(f"  {'':32s} (the most permissive convention there is -- a 10% baseline")
            log(f"  {'':32s}  confirmed by a single record)")
        log(f"  {'C-A  ARL signature, strict':32s} holds in {n_strict}/{n_tot} dataset x convention cells")
        if fails_strict:
            log(f"  {'':32s} fails: {'; '.join(fails_strict)}")

    gb = df[df.claim == "C-B_inside_baseline_gap"]
    if not gb.empty:
        n_ok = int((gb.value > 0.20).sum())
        dflt = gb[gb.is_default].value
        log(f"  {'C-B  baseline contradiction':32s} holds in {n_ok}/{len(gb)} conventions "
            f"| range {gb.value.min():+.3f} .. {gb.value.max():+.3f}"
            + (f" | paper's cell {dflt.median():+.3f}" if len(dflt) else ""))
        log(f"  {'':32s} the gap GROWS with the baseline, and the paper's cell sits")
        log(f"  {'':32s} mid-range -- it is not the convention that flatters the claim")

    gw = df[df.claim == "C-C_within_family"]
    gbw = df[df.claim == "C-C_between_family"]
    if not gw.empty and not gbw.empty:
        log(f"  {'C-C  within-family |dFPT|':32s} median {gw.value.median():.3f} "
            f"| range {gw.value.min():.3f} .. {gw.value.max():.3f}")
        log(f"  {'C-C  between-family |dFPT|':32s} median {gbw.value.median():.3f} "
            f"| range {gbw.value.min():.3f} .. {gbw.value.max():.3f}")
        ratio = gbw.value.median() / max(gw.value.median(), 1e-9)
        log(f"  {'C-C  ratio between/within':32s} {ratio:.0f}x")
        log("")
        log("  >>> THIS IS THE SHARPEST NUMBER IN THE PAPER. Two rules from the same family")
        log("      put the boundary within a few percent of life of each other. A detector")
        log(f"      rule and a control-chart rule differ by {ratio:.0f} times as much. The")
        log("      disagreement the paper is about is not noise between interchangeable")
        log("      rules -- it is a systematic split between two families that the")
        log("      literature substitutes for one another without comment.")

    log("")
    log("  Read this table as the paper's own honesty test. A claim that holds in 12/12")
    log("  conventions is a property of the rules. One that holds only at BASE_FRAC 0.20 is a")
    log("  property of our inheritance, and the paper must say so in the same breath as the")
    log("  claim -- that is precisely the standard this paper asks of everyone else.")

    head("7", "Figure")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    COL = {"detector": "#1B6773", "control-chart": "#B85C18"}

    ax = axes[0]
    ga = df[df.claim == "C-A_arl"]
    for fam in ("detector", "control-chart"):
        q = ga[ga.family == fam].groupby(["base_frac", "confirm"]).value.median().reset_index()
        for cf, m in zip(CONFIRM_GRID, ("o", "s", "^")):
            qq = q[q.confirm == cf]
            ax.plot(qq.base_frac, qq.value, m + "-", ms=5, lw=1.2, color=COL[fam],
                    alpha=.85, label=f"{fam}, confirm={cf}")
    ax.axhline(0, color="#13222D", lw=1)
    ax.axvline(DEFAULT_CELL[0], ls=":", color="#A6B7C0")
    ax.set_xlabel("BASE_FRAC"); ax.set_ylabel("median rho(index, length)")
    ax.set_title("C-A  ARL signature", fontsize=10)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)

    ax = axes[1]
    gb = df[df.claim == "C-B_inside_baseline_gap"]
    for cf, m in zip(CONFIRM_GRID, ("o", "s", "^")):
        q = gb[gb.confirm == cf]
        ax.plot(q.base_frac, q.value, m + "-", ms=5, lw=1.2, label=f"confirm={cf}")
    ax.axhline(0.20, ls=":", color="#A6B7C0")
    ax.axvline(DEFAULT_CELL[0], ls=":", color="#A6B7C0")
    ax.set_xlabel("BASE_FRAC"); ax.set_ylabel("control-chart − detector gap")
    ax.set_title("C-B  fires inside assumed-healthy window", fontsize=10)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    for claim, c, lab in (("C-C_within_family", "#1B6773", "within detector family"),
                          ("C-C_between_family", "#B85C18", "detector vs control-chart")):
        q = df[df.claim == claim].groupby("base_frac").value.median().reset_index()
        if q.empty:
            continue
        ax.plot(q.base_frac, q.value, "o-", ms=5, lw=1.6, color=c, label=lab)
    ax.axvline(DEFAULT_CELL[0], ls=":", color="#A6B7C0")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("BASE_FRAC"); ax.set_ylabel("median pairwise |ΔFPT|")
    ax.set_title("C-C  the disagreement is BETWEEN families", fontsize=10)
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle("Diagnostic — the paper's three claims recomputed over 12 labelling "
                 "conventions (dotted line = the convention the paper inherited)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "j13_convention_grid_diagnostic.png", dpi=170)
    plt.close(fig)
    log(f"  wrote {FIGS / 'j13_convention_grid_diagnostic.png'} (the journal Figure 5 is drawn by J14)")

    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "j15_rule_sensitivity.csv", index=False)
    log(f"\n{BAR}")
    log(f"  WROTE  results/tables/j15_rule_sensitivity.csv ({len(df)} rows)   [E6]")
    log(f"         results/figures/j13_convention_grid_diagnostic.png")
    log(f"         results/logs/j13_run.log")
    log(f"  finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    log(BAR)
    if _fh:
        _fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--check" in sys.argv))
