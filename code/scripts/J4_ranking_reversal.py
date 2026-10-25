"""J4 -- do published FPT rules reverse the ranking of methods on the same data?

Reads J3's output and answers the question C2 exists to answer:

    If two rules that the literature treats as interchangeable put the boundary in
    different places, does the CONCLUSION about which method is better change with them?

A sensitivity result would be "scores move". The indicting result is "the ORDER moves":
a paper using 3-sigma-on-RMS and a paper using EWMA, on identical data, would report
different winners and neither would be wrong by the standards of the field.

What is reported
    1. Best method per rule -- the headline table. If the winner changes, the ranking is
       rule-dependent.
    2. Kendall tau between the method rankings induced by every pair of rules. tau = 1 means
       the two rules agree completely on the ordering; tau <= 0 means they disagree as much
       as chance, or invert it.
    3. The specific REVERSED PAIRS: method A beats B under one published rule and B beats A
       under another, with both margins large enough to be reported as a finding.
    4. Boundary disagreement: how far apart the rules place the FPT, per trajectory.
    5. Modality reversal on NSK, the only dataset with a synchronised thermal channel.

Usage
    python scripts/J4_ranking_reversal.py
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
BAR = "=" * 78
MIN_MARGIN = 0.02          # a reversal counts only if both directions clear this margin


def head(step: str, title: str) -> None:
    print(f"\n{BAR}\nSTEP {step}  {title}\n{BAR}", flush=True)


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Rank correlation between two orderings of the same methods."""
    idx_a = {m: i for i, m in enumerate(a)}
    idx_b = {m: i for i, m in enumerate(b)}
    common = [m for m in a if m in idx_b]
    n = len(common)
    if n < 2:
        return float("nan")
    conc = disc = 0
    for x, y in combinations(common, 2):
        s = np.sign(idx_a[x] - idx_a[y]) * np.sign(idx_b[x] - idx_b[y])
        conc += s > 0
        disc += s < 0
    return (conc - disc) / (0.5 * n * (n - 1))


def main() -> int:
    print(BAR)
    print("J4  DOES THE PUBLISHED RULE YOU PICK CHANGE WHO WINS?".center(78))
    print(BAR)

    pf_path, ft_path = TABLES / "j3_performance.csv", TABLES / "j3_fpt_times.csv"
    if not pf_path.exists():
        print(f"\n  MISSING {pf_path.name} -- run scripts/J3_fpt_rules.py first.")
        return 1
    pf, ft = pd.read_csv(pf_path), pd.read_csv(ft_path)

    # Drop rules whose label set is degenerate. EWMA and CUSUM call 79-97% of every
    # trajectory degraded here, because a control chart is built to flag the ONSET of a small
    # persistent shift, not to PARTITION a monotone trajectory into healthy and faulty halves.
    # Ranking methods against such a label set measures nothing, and a reversal drawn from it
    # would be the first thing a reviewer struck out. Which rules fail this way is itself a
    # result and is printed below.
    if "degenerate" in pf.columns:
        bad = (pf[pf.degenerate].groupby("dataset").rule.unique().to_dict())
        if bad:
            print("\n  DEGENERATE label sets excluded (positive rate outside 1%-50%):")
            for d, rs in bad.items():
                rates = {r: round(float(pf[(pf.dataset == d) & (pf.rule == r)].pos_rate.iloc[0]), 2)
                         for r in rs}
                print(f"    {d:5s} {rates}")
        pf = pf[~pf.degenerate]
        ft = ft.merge(pf[["dataset", "rule"]].drop_duplicates(), on=["dataset", "rule"], how="inner")

    for ds in pf.dataset.unique():
        g = pf[pf.dataset == ds]
        rules = [r for r in g.rule.unique() if g[g.rule == r].shape[0] >= 3]
        if len(rules) < 2:
            continue

        head("1", f"{ds.upper()} -- who wins under each published rule")
        print(f"    {'rule':10s} {'n_pos':>6s} {'winner':>22s} {'macroF1':>9s} "
              f"{'runner-up':>22s} {'margin':>8s}")
        order = {}
        for r in rules:
            q = g[g.rule == r].sort_values("macro_f1", ascending=False)
            order[r] = q.method.tolist()
            top, second = q.iloc[0], q.iloc[1]
            print(f"    {r:10s} {int(top.n_pos):>6d} {top.method:>22s} {top.macro_f1:>9.4f} "
                  f"{second.method:>22s} {top.macro_f1 - second.macro_f1:>8.4f}")
        # A "winner" decided by a margin of ~0 is decided by sort order, not by the data.
        # With only 3-13 positive records (NSK, IMS) macro-F1 takes few discrete values and
        # ties at the top are common, so count only winners that clear MIN_MARGIN.
        winners = {r: order[r][0] for r in rules}
        clear = {}
        for r in rules:
            q = g[g.rule == r].sort_values("macro_f1", ascending=False)
            if q.iloc[0].macro_f1 - q.iloc[1].macro_f1 > MIN_MARGIN:
                clear[r] = q.iloc[0].method
        uniq_all = sorted(set(winners.values()))
        uniq_clear = sorted(set(clear.values()))
        print(f"\n    distinct winners across {len(rules)} published rules: {len(uniq_all)}  {uniq_all}")
        print(f"    ... of which decided by a margin > {MIN_MARGIN}: {len(uniq_clear)}  {uniq_clear}"
              f"   ({len(clear)}/{len(rules)} rules have a clear winner)")
        if len(uniq_clear) > 1:
            print("    -> the winner IS rule-dependent on this dataset, on clear margins.")
        elif len(uniq_all) > 1:
            print("    -> winners differ but mostly by ties; do NOT claim rule-dependence here.")

        head("2", f"{ds.upper()} -- Kendall tau between the rankings induced by each rule pair")
        print(f"    {'':10s}" + "".join(f"{r[:8]:>9s}" for r in rules))
        taus = {}
        for a in rules:
            row = ""
            for b in rules:
                t = 1.0 if a == b else kendall_tau(order[a], order[b])
                taus[(a, b)] = t
                row += f"{t:>9.2f}"
            print(f"    {a:10s}{row}")
        off = [v for (a, b), v in taus.items() if a != b]
        print(f"\n    median tau {np.median(off):+.2f} | min {np.min(off):+.2f}"
              f" | pairs with tau <= 0: {sum(1 for v in off if v <= 0) // 2}")

        head("3", f"{ds.upper()} -- explicit reversals (A beats B under one rule, B beats A under another)")
        piv = g.pivot_table(index="method", columns="rule", values="macro_f1")
        found = 0
        for m1, m2 in combinations(piv.index, 2):
            for r1, r2 in combinations(rules, 2):
                d1 = piv.loc[m1, r1] - piv.loc[m2, r1]
                d2 = piv.loc[m1, r2] - piv.loc[m2, r2]
                if np.isnan(d1) or np.isnan(d2):
                    continue
                if d1 > MIN_MARGIN and d2 < -MIN_MARGIN:
                    found += 1
                    if found <= 12:
                        print(f"    {m1:>20s} vs {m2:<20s}  {r1:>9s}: {d1:+.3f}   "
                              f"{r2:>9s}: {d2:+.3f}")
        print(f"\n    {found} reversed (method-pair, rule-pair) combinations with both margins "
              f"> {MIN_MARGIN}")
        if found == 0:
            print("    -> no reversal survives the margin: report the scores as sensitive but")
            print("       the ORDER as stable, which is a weaker but still reportable claim.")

        if ds == "nsk":
            head("4", "NSK -- modality reversal (the only dataset with a thermal channel)")
            gm = g.assign(ch=np.where(g.method.str.contains("temp"), "thermal", "vibration"))
            print(f"    {'rule':10s} {'best vibration':>16s} {'best thermal':>14s} {'margin':>9s}  leader")
            for r in rules:
                q = gm[gm.rule == r]
                v = q[q.ch == "vibration"].macro_f1.max()
                t = q[q.ch == "thermal"].macro_f1.max()
                if np.isnan(v) or np.isnan(t):
                    continue
                print(f"    {r:10s} {v:>16.4f} {t:>14.4f} {v - t:>+9.4f}  "
                      f"{'VIBRATION' if v > t else 'THERMAL'}")

    head("5", "How far apart do the rules place the boundary?")
    for ds in ft.dataset.unique():
        g = ft[ft.dataset == ds]
        st = g.groupby("trajectory").fpt_fraction.agg(["min", "max", "count"])
        st["spread"] = st["max"] - st["min"]
        print(f"  {ds:5s} median spread {st.spread.median():.2f} of life | worst {st.spread.max():.2f}"
              f" | {len(st)} trajectories")
        w = st.spread.idxmax()
        sub = g[g.trajectory == w].sort_values("fpt_fraction")
        print(f"        worst case {w}: " +
              "  ".join(f"{r.rule}={r.fpt_fraction:.2f}" for _, r in sub.iterrows()))

    head("6", "What to write")
    print("  The reportable claim is the strongest of these that the tables above support:")
    print("    (a) different published rules crown different winners        -> ranking is rule-dependent")
    print("    (b) rankings agree but scores move a lot                     -> sensitivity only")
    print("    (c) rules place the boundary far apart within one trajectory -> the boundary itself")
    print("        is unconstrained, independent of any downstream model")
    print()
    print("  (c) holds whenever the spread in step 5 is large, and it needs no model at all,")
    print("  so it is the claim least exposed to a reviewer arguing about architectures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
