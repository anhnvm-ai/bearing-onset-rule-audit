"""J17 -- release the per-rule onset labels and run the Section 7 reporting screen on them.

The screen is written so that anyone can run it on their own onsets: it needs one row per (dataset, trajectory, rule)
with the trajectory length and the onset index (empty when the rule never fired), and optionally the positive rate
over the records that were actually scored, for the implied-vs-scored check.

Outputs
    results/release/onset_labels.csv        33 trajectories x 10 rules, one row each (from fig1_fpt_matrix.csv)
    results/release/README.md               what the columns mean and how to expand onsets into record labels
    results/tables/j20_reporting_screen.csv screen items 1-4 and 6-7 per dataset x rule, plus the item-5 statistic
    results/tables/j20_scored_check.csv     implied vs scored positive rate, corrected and defective pipelines
    results/logs/j17_run.log

Usage
    python scripts/J17_release_and_screen.py
    python scripts/J17_release_and_screen.py --onsets my_onsets.csv [--scored my_scored.csv] --out my_screen.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
RELEASE = ROOT / "results" / "release"
LOGS = ROOT / "results" / "logs"
USABLE = (0.01, 0.50)
FAMILY = {"3sig_rms": "detector", "ma3sig": "detector", "ma5sig": "detector", "ma2sig": "detector",
          "3sig_kurt": "detector", "ewma": "control chart", "cusum": "control chart",
          "pct02": "fixed convention", "pct05": "fixed convention", "pct10": "fixed convention"}
INDICATOR = {r: ("kurtosis" if r == "3sig_kurt" else "none" if r.startswith("pct") else "RMS") for r in FAMILY}
BASE_FRAC, CONFIRM = 0.20, 3
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n")


def release(ft: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "dataset": ft.dataset, "trajectory": ft.trajectory, "n_records": ft.n_records, "rule": ft.rule,
        "family": ft.rule.map(FAMILY), "indicator": ft.rule.map(INDICATOR), "fired": ft.fired,
        "onset_index": ft.fpt_index.astype("Int64"), "onset_fraction": ft.fpt_fraction,
        "baseline_records": np.where(ft.rule.str.startswith("pct"), pd.NA, ft.baseline_records),
        "onset_inside_baseline": np.where(ft.rule.str.startswith("pct"), pd.NA, ft.inside_baseline),
        "baseline_fraction": np.where(ft.rule.str.startswith("pct"), np.nan, BASE_FRAC),
        "confirm_records": np.where(ft.rule.str.startswith("pct"), pd.NA, CONFIRM)})
    # revision 2026-09-21 (Phase 30): four columns a reader otherwise has to derive, appended after
    # the original ones so that any code reading the earlier columns by name or position keeps working.
    #   censored          the rule never fired: its onset is right-censored at the end of the test (T* > C).
    #   positive_records  records labelled degraded, n - onset; empty when censored (the trajectory carries
    #                     no labels under that rule, it is NOT labelled healthy).
    #   positive_rate     positive_records / n, the per-trajectory rate; the pooled rate of ORD-4 item 3 is
    #                     sum(positive_records) / sum(n) over the non-censored rows.
    #   response_size_sd  indicator level over the confirming records at the alarm minus the baseline mean, in
    #                     baseline standard deviations (ORD-4 item 1), from J20's j22_shift_at_alarm table.
    #                     Measured on RMS, so it is empty for 3sig_kurt (a kurtosis rule) and for the
    #                     percentile rules, which read no indicator.
    fired = ft.fired.astype(bool).to_numpy()
    idx = ft.fpt_index.to_numpy(dtype=float)
    n = ft.n_records.to_numpy(dtype=float)
    pos = np.where(fired, n - idx, np.nan)
    out["censored"] = ~fired
    out["positive_records"] = pd.array(np.where(fired, pos, np.nan), dtype="Int64")
    out["positive_rate"] = np.where(fired, pos / n, np.nan)
    sh = pd.read_csv(TABLES / "j22_shift_at_alarm.csv")[["dataset", "trajectory", "rule", "shift_in_baseline_sd"]]
    out = out.merge(sh, on=["dataset", "trajectory", "rule"], how="left", validate="one_to_one")
    out = out.rename(columns={"shift_in_baseline_sd": "response_size_sd"})
    return out


def screen(on: pd.DataFrame) -> pd.DataFrame:
    """The four ORD-4 disclosures per dataset x rule, plus the onset descriptives Section 4 uses.

    Columns are prefixed ord1_ to ord4_ after the four items of Table 5 in the paper. ORD-1 also asks
    for the median shift at the alarm in baseline standard deviations; that needs the indicator series
    and not only the onsets, so it is released separately as shift_at_alarm.csv (script J22).
    """
    rows = []
    for (ds, rule), g in on.groupby(["dataset", "rule"], sort=False):
        f = g[g.fired.astype(bool)]
        n_fire = len(f)
        pos = (f.n_records - f.onset_index.astype(float)).sum()
        rate = pos / f.n_records.sum() if n_fire else np.nan
        per = 1 - f.onset_fraction if n_fire else pd.Series(dtype=float)
        inside = f.onset_inside_baseline.dropna()
        rows.append({
            "dataset": ds, "rule": rule,
            "ord1_family": g.family.iloc[0],
            "ord2_trajectories": len(g), "ord2_fired": n_fire, "ord2_coverage": n_fire / len(g),
            "ord2_nonfiring_policy": "excluded",
            "ord3_pooled_rate": rate,
            "ord3_per_trajectory_min": per.min() if n_fire else np.nan,
            "ord3_per_trajectory_max": per.max() if n_fire else np.nan,
            "ord3_usable": bool(n_fire and USABLE[0] <= rate <= USABLE[1]),
            "onset_median": f.onset_fraction.median() if n_fire else np.nan,
            "onset_min": f.onset_fraction.min() if n_fire else np.nan,
            "onset_max": f.onset_fraction.max() if n_fire else np.nan,
            "onset_inside_baseline": int(inside.astype(bool).sum()) if len(inside) else pd.NA,
            "ord1_statistic": g.indicator.iloc[0],
            "ord4_baseline_fraction": g.baseline_fraction.iloc[0],
            "ord4_confirm_records": g.confirm_records.iloc[0],
            "ord4_within_validity_floor": (bool(g.baseline_fraction.iloc[0] >= 0.20 and g.confirm_records.iloc[0] >= 3)
                                            if pd.notna(g.baseline_fraction.iloc[0]) else pd.NA)})
    cols = ['dataset', 'rule', 'ord1_family', 'ord1_statistic', 'ord2_trajectories', 'ord2_fired', 'ord2_coverage', 'ord2_nonfiring_policy', 'ord3_pooled_rate', 'ord3_per_trajectory_min', 'ord3_per_trajectory_max', 'ord3_usable', 'ord4_baseline_fraction', 'ord4_confirm_records', 'ord4_within_validity_floor', 'onset_median', 'onset_min', 'onset_max', 'onset_inside_baseline']
    return pd.DataFrame(rows)[cols]


def scored_check(scr: pd.DataFrame, scored: pd.DataFrame, col: str) -> pd.DataFrame:
    """The implied-versus-scored identity of Section 6: the rate implied by the onsets must equal the
    rate over the scored records."""
    m = scr[["dataset", "rule", "ord3_pooled_rate"]].merge(scored[["dataset", "rule", col]], on=["dataset", "rule"])
    m = m.rename(columns={"ord3_pooled_rate": "implied_rate", col: "scored_rate"})
    m["agree"] = (m.implied_rate - m.scored_rate).abs() < 5e-4
    return m


def main() -> int:
    global _fh
    ap = argparse.ArgumentParser()
    ap.add_argument("--onsets")
    ap.add_argument("--scored")
    ap.add_argument("--out")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    if a.onsets:  # screen someone else's onsets
        on = pd.read_csv(a.onsets)
        for c, d in (("family", "unknown"), ("indicator", "unknown"), ("onset_inside_baseline", pd.NA),
                     ("baseline_fraction", np.nan), ("confirm_records", pd.NA)):
            if c not in on:
                on[c] = d
        if "onset_fraction" not in on:
            on["onset_fraction"] = on.onset_index / on.n_records
        scr = screen(on)
        scr.to_csv(a.out or "reporting_screen.csv", index=False)
        if a.scored:
            print(scored_check(scr, pd.read_csv(a.scored), "scored_rate").to_string(index=False))
        print(scr.to_string(index=False))
        return 0

    LOGS.mkdir(parents=True, exist_ok=True)
    RELEASE.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j17_run.log", "w", encoding="utf-8")
    ft = pd.read_csv(TABLES / "fig1_fpt_matrix.csv")
    on = release(ft)
    on.to_csv(RELEASE / "onset_labels.csv", index=False)
    log(f"released {len(on)} rows: {on.groupby(['dataset', 'trajectory']).ngroups} trajectories x {on.rule.nunique()} rules")

    scr = screen(on)
    scr.to_csv(TABLES / "j20_reporting_screen.csv", index=False)
    ref = pd.read_csv(TABLES / "j17_prevalence_by_subset.csv").query("subset == 'all'")
    cmp = scr.merge(ref, on=["dataset", "rule"])
    dr = (cmp.ord3_pooled_rate - cmp.pos_rate).abs().max()
    du = (cmp.ord3_usable != cmp.usable.astype(str).eq("True")).sum()
    log(f"pooled rates vs j17_prevalence_by_subset: max |diff| {dr:.2e}, usability disagreements {du} over {len(cmp)} pairs")

    demo = pd.read_csv(TABLES / "j14_defect_demo.csv")
    fixed = scored_check(scr, demo, "pos_rate_fixed").assign(pipeline="corrected")
    legacy = scored_check(scr, demo, "pos_rate_legacy").assign(pipeline="defective")
    chk = pd.concat([fixed, legacy], ignore_index=True)
    chk.to_csv(TABLES / "j20_scored_check.csv", index=False)
    log(f"item 6, corrected pipeline: {int(fixed.agree.sum())}/{len(fixed)} dataset-rule pairs agree")
    log(f"item 6, defective pipeline: {int(legacy.agree.sum())}/{len(legacy)} agree; disagreements:")
    for r in legacy[~legacy.agree].itertuples():
        log(f"    {r.dataset:6s} {r.rule:10s} implied {r.implied_rate:.3f}  scored {r.scored_rate:.3f}")

    (RELEASE / "README.md").write_text(
        "# Onset labels for 33 run-to-failure trajectories under 10 rules\n\n"
        "`onset_labels.csv` has one row per dataset, trajectory and rule.\n\n"
        "- `n_records`: records in the trajectory, in time order, indexed from 0.\n"
        "- `onset_index`: first record labelled degraded; empty when the rule never fired (`fired` is False).\n"
        "  Records `onset_index` to `n_records - 1` are degraded, earlier records healthy.\n"
        "- `onset_fraction`: `onset_index / n_records`.\n"
        "- `baseline_records`: records used for the healthy reference, max(5, floor(0.2 n)).\n"
        "- `onset_inside_baseline`: the onset falls inside that reference window.\n"
        "- `baseline_fraction`, `confirm_records`: the conventions used (0.20 and 3). Percentile rules use neither.\n"
        "- `censored`: the rule never fired, so its onset is right-censored at the end of the test (T* > C).\n"
        "- `positive_records`, `positive_rate`: records labelled degraded, and that count over `n_records`.\n"
        "  Empty when censored. The pooled rate (ORD-4 item 3) is sum(positive_records) / sum(n_records)\n"
        "  over the rows that are not censored.\n"
        "- `response_size_sd`: indicator level over the confirming records at the alarm minus the baseline\n"
        "  mean, in baseline standard deviations (ORD-4 item 1). Measured on RMS, so it is empty for\n"
        "  `3sig_kurt` (a kurtosis rule) and for the percentile rules, which read no indicator.\n\n"
        "A trajectory on which a rule never fires has no labels under that rule. Do not label it healthy:\n"
        "the paper's section on non-firing trajectories shows that this changes prevalence-screen verdicts\n"
        "and winning methods.\n\n"
        "Run the reporting screen on these or on your own onsets with\n"
        "`python scripts/J17_release_and_screen.py --onsets onset_labels.csv --out screen.csv`.\n",
        encoding="utf-8")
    log("wrote results/release/onset_labels.csv, README.md, j20_reporting_screen.csv, j20_scored_check.csv")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
