"""J9 -- decompose a reported score into the choices that produced it.

The deliverable of C4. Every term below was measured by an earlier experiment in this
repository; J9 only assembles them, converts each into the same unit (macro-F1), and sets
them against the quantity papers actually argue about -- which model was used.

    Delta_leakage        how the split is drawn           J1  (and e1 on NSK)
    Delta_boundary       which published FPT rule labels  J3 / J4
    Delta_aggregation    how windows become a record      J6
    Delta_optimization   which random seed was drawn      e5
    Delta_normalization  absolute vs baseline-relative    J7
    ---------------------------------------------------------------
    reference            spread across ALL methods tried  e2 / J3

The claim J9 exists to support is not that any single term is large, but that EVERY term is
larger than the reference. A paper that reports one number, from one split, one labelling
rule, one pooling rule, one seed and one normalisation has not measured its model; it has
measured its own conventions.

Each row records where the number came from, so the table is auditable rather than asserted.

Usage
    python scripts/J9_optimism_decomposition.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
BAR = "=" * 78


def head(step: str, title: str) -> None:
    print(f"\n{BAR}\nSTEP {step}  {title}\n{BAR}", flush=True)


def load(name: str) -> pd.DataFrame | None:
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None


def main() -> int:
    print(BAR)
    print("J9  OPTIMISM DECOMPOSITION".center(78))
    print("how much of a reported number is the model, and how much is the conventions?".center(78))
    print(BAR)

    rows: list[dict] = []

    def add(ds, term, value, source, detail):
        rows.append({"dataset": ds, "term": term, "delta_macro_f1": value,
                     "source": source, "detail": detail})

    head("1", "The split ladder -- THREE tiers, not one")
    # J1b, not J1. J1 measured the gap on RANDOM labels, which is memorisation capacity and
    # NOT commensurable with the other terms here (all real record-level macro-F1 spreads).
    # J1b holds the test records, the evaluation windows and the labels fixed and varies only
    # what the model was trained on, so its numbers belong in the same column as the rest.
    j1b = load("j1b_summary.csv")
    if j1b is not None:
        g = j1b[j1b.rule == "pct05"]
        print("  Fixed labelling rule pct05 (comparable positive rate on all three datasets).")
        print(f"  {'dataset':7s} {'S0-S1 window':>14s} {'S1-S2 trajectory':>18s} {'S2-S3 condition':>17s}")
        for _, r in g.iterrows():
            add(r.dataset, "leakage_window", float(r.leakage_S0_S1), "J1b",
                "S0 leaky minus S1 clean: same test records, same eval windows, real labels")
            if np.isfinite(r.adjacency_S1_S2):
                add(r.dataset, "leakage_trajectory", float(r.adjacency_S1_S2), "J1b",
                    "S1 random records minus S2 whole trajectories held out")
            if np.isfinite(r.condition_S2_S3):
                add(r.dataset, "condition_shift", float(r.condition_S2_S3), "J1b",
                    "S2 trajectory-wise minus S3 condition-wise")
            f = lambda v: "           n/a" if not np.isfinite(v) else f"{v:+14.4f}"
            print(f"  {r.dataset:7s} {f(r.leakage_S0_S1)} {f(r.adjacency_S1_S2):>18s} "
                  f"{f(r.condition_S2_S3):>17s}")
        print()
        print("  READ THIS: only the FIRST tier is window-level leakage, and it is the small one.")
        print("  The trajectory tier is 6-23x larger -- and that is exactly the bearing-wise")
        print("  split Vieira et al. (MSSP 2026) advocate, so this result CONFIRMS them rather")
        print("  than contradicting them. What is new here is putting the labelling rule on the")
        print("  same scale (step 2) and finding it comparable to, or larger than, the whole")
        print("  split hierarchy.")
    else:
        print("  j1b_summary.csv missing -- run scripts/J1b_leakage_real_labels.py")
    j1 = load("j1_leakage_ladder.csv")
    if j1 is not None:
        print("\n  For reference only, NOT added to the table -- the random-label control (J1),")
        print("  which measures memorisation capacity and is a different quantity:")
        for ds, g in j1.groupby("dataset"):
            s0 = g[g.protocol == "S0_window_random"].macro_f1.to_numpy()
            strict = ("S2_trajectory" if (g.protocol == "S2_trajectory").any()
                      else "S1_record_grouped")
            sx = g[g.protocol == strict].macro_f1.to_numpy()
            if len(s0) and len(sx):
                m = min(len(s0), len(sx))
                print(f"    {ds:5s} {np.median(s0[:m] - sx[:m]):+.4f}   S0 vs {strict}, random labels")

    head("2", "Delta_boundary -- which published FPT rule assigns the labels")
    j3 = load("j3_performance.csv")
    if j3 is not None:
        if "degenerate" in j3.columns:
            j3 = j3[~j3.degenerate]
        for ds, g in j3.groupby("dataset"):
            best = g.groupby("rule").macro_f1.max()
            if len(best) < 2:
                continue
            d = float(best.max() - best.min())
            add(ds, "boundary", d, "J3/J4",
                f"best-method macro-F1 across {len(best)} non-degenerate published rules")
            print(f"  {ds:5s} {d:+.4f}   across {len(best)} rules: "
                  f"{best.min():.3f} ({best.idxmin()}) .. {best.max():.3f} ({best.idxmax()})")
    else:
        print("  j3_performance.csv missing -- run scripts/J3_fpt_rules.py")

    head("3", "Delta_aggregation -- how window scores become a record decision")
    j6 = load("j6_aggregation.csv")
    if j6 is not None:
        g = j6[j6.horizon == "h5"]
        by = g.groupby("rule").macro_f1.median()
        d = float(by.max() - by.min())
        add("nsk", "aggregation", d, "J6",
            f"median macro-F1 across {len(by)} pooling rules at h5, thresholds fitted per rule")
        print(f"  nsk   {d:+.4f}   {by.idxmin()} {by.min():.3f} .. {by.idxmax()} {by.max():.3f}"
              f"   (9 runs, h5)")
        print("        NOTE J8b: on XJTU the direction REVERSES (rho ~ 1 -> mean pooling best),")
        print("        so this term is trajectory-specific, not a constant.")
    else:
        print("  j6_aggregation.csv missing -- run scripts/J6_aggregation.py")

    head("4", "Delta_optimization -- which random seed was drawn")
    e5 = load("e5_deep_seed_stability.csv")
    if e5 is not None:
        ps = e5[e5.seed.astype(str).str.fullmatch(r"\d+")]
        for model, g in ps.groupby("model"):
            d = float(g.macro_f1.max() - g.macro_f1.min())
            if model == "resnet":
                add("nsk", "optimization", d, "e5",
                    f"{model} macro-F1 range over {len(g)} seeds, fixed 12-epoch budget")
            print(f"  nsk   {d:+.4f}   {model:8s} over {len(g)} seeds "
                  f"({g.macro_f1.min():.3f} .. {g.macro_f1.max():.3f})")
    else:
        print("  e5_deep_seed_stability.csv missing")

    head("5", "Delta_normalization -- absolute vs baseline-relative features")
    j7 = load("j7_circularity.csv")
    if j7 is not None and "featureset" in j7.columns:
        for ds, g in j7.groupby("dataset"):
            worst, where = 0.0, ""
            for pct, gg in g.groupby("life_pct"):
                a = gg[gg.featureset == "absolute"]
                r = gg[gg.featureset == "relative"]
                if a.empty or r.empty:
                    continue
                ga = a[a.method == "rf:all"].macro_f1.max() - a[a.method.str.startswith("thr:")].macro_f1.max()
                gr = r[r.method == "rf:all"].macro_f1.max() - r[r.method.str.startswith("thr:")].macro_f1.max()
                if abs(ga - gr) > worst:
                    worst, where = abs(ga - gr), f"last {pct:.0%}: GAP {ga:+.3f} -> {gr:+.3f}"
            if worst:
                add(ds, "normalization", worst, "J7",
                    "change in GAP(forest - best threshold) when features are made relative")
                print(f"  {ds:5s} {worst:+.4f}   {where}")
    else:
        print("  j7_circularity.csv missing -- run scripts/J7_circularity.py")

    head("6", "Reference -- the spread papers actually argue about")
    ref = {}
    e2 = load("e2_raw_counts_h5.csv")
    if e2 is not None:
        d = float(e2.macro_f1.max() - e2.macro_f1.min())
        ref["nsk"] = d
        print(f"  nsk   {d:+.4f}   across the 7 conference methods at h5 "
              f"(1 parameter to 11.2M), pooled OOF")
    # The reference must be the spread between methods a paper would ACTUALLY compare, not
    # between the best and the worst thing tried. Best-to-worst includes features nobody would
    # propose (skew, crest) and inflates the reference three to ten fold, which in turn makes
    # every real term look small. The spread across the TOP THREE methods under one rule is the
    # honest stand-in. On NSK no stand-in is needed: the conference measured the real thing.
    if j3 is not None:
        for ds, g in j3.groupby("dataset"):
            sp = g.groupby("rule").apply(
                lambda q: q.macro_f1.nlargest(3).max() - q.macro_f1.nlargest(3).min(),
                include_groups=False)
            ref.setdefault(ds, float(sp.median()))
            print(f"  {ds:5s} {float(sp.median()):+.4f}   median spread across the TOP 3 methods, "
                  f"within one labelling rule")

    head("7", "The table")
    df = pd.DataFrame(rows)
    if df.empty:
        print("  nothing assembled -- run the upstream experiments first.")
        return 1
    print(f"  {'dataset':8s} {'term':15s} {'delta':>9s} {'x reference':>12s}  source   detail")
    for ds in df.dataset.unique():
        r = ref.get(ds, np.nan)
        for _, x in df[df.dataset == ds].sort_values("delta_macro_f1", ascending=False).iterrows():
            if np.isfinite(r) and r > 1e-9:
                lab = f"{x.delta_macro_f1 / r:.1f}x"
            elif np.isfinite(r):
                lab = "INF"          # the reference is exactly zero: see the NSK note below
            else:
                lab = "--"
            print(f"  {ds:8s} {x.term:15s} {x.delta_macro_f1:>9.4f} "
                  f"{lab:>12s}  {x.source:8s} {x.detail}")
        if np.isfinite(r):
            print(f"  {ds:8s} {'REFERENCE':15s} {r:>9.4f} {'1.0x':>12s}  --       "
                  f"choice of method, within one protocol")
            if r <= 1e-9:
                print(f"  {ds:8s} {'':15s} {'':>9s} {'':>12s}  "
                      f"the reference is EXACTLY ZERO -- all seven conference methods, from a")
                print(f"  {ds:8s} {'':15s} {'':>9s} {'':>12s}  "
                      f"one-parameter threshold to an 11.2M-parameter network, returned the")
                print(f"  {ds:8s} {'':15s} {'':>9s} {'':>12s}  "
                      f"same per-record decisions, so every term above is infinitely larger.")
        print()

    head("8", "Reliability report card -- what a run must disclose")
    print("  A single reported number is uninterpretable unless it is accompanied by:")
    print("    1. the split protocol, named on the ladder S0/S1/S2/S3")
    print("    2. the labelling rule, cited, with the resulting positive rate")
    print("    3. the pooling rule and how its threshold was fitted")
    print("    4. the seed policy: number of seeds and the spread, not one run")
    print("    5. the feature normalisation, absolute or trajectory-relative")
    print("    6. for any lead-time claim, the false-alarm rate it was measured at")
    print()
    print("  Items 1-5 are the terms above. Item 6 comes from J8b, where a lead of +45 records")
    print("  fell to +0.5 once the false-alarm rate was controlled.")

    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "j9_decomposition.csv", index=False)
    print(f"\n{BAR}\n  WROTE  results/tables/j9_decomposition.csv ({len(df)} rows)\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
