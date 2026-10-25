"""J19 -- export the headline numbers of the journal manuscript to paper/journal/generated_numbers.tex.

The abstract, the introduction and the conclusion repeat numbers that the body sections establish. Those repeated
numbers are written here as LaTeX macros computed from results/tables, so that the summary text cannot drift from
the tables. Numbers that appear once, in the section that establishes them, stay literal and are checked by that
section's verify script.

Usage
    python scripts/J19_export_numbers.py           write the file
    python scripts/J19_export_numbers.py --check   exit 1 if the file on disk differs from a fresh export
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
OUT = ROOT / "paper" / "journal" / "generated_numbers.tex"


def numbers() -> list[tuple[str, str, str]]:
    ft = pd.read_csv(T / "fig1_fpt_matrix.csv")
    e2 = pd.read_csv(T / "e2_raw_counts_h5.csv")
    sub = pd.read_csv(T / "j17_prevalence_by_subset.csv")
    shift = pd.read_csv(T / "j22_shift_at_alarm.csv")
    out = []

    def add(name, value, source):
        out.append((name, value, source))

    add("NTrajectories", str(ft.groupby(["dataset", "trajectory"]).ngroups), "fig1_fpt_matrix: dataset x trajectory")
    add("NRules", str(ft.rule.nunique()), "fig1_fpt_matrix: rules")
    add("NRuleTrajectoryPairs", str(len(ft)), "fig1_fpt_matrix: rows")
    add("NNeverFire", str(int((~ft.fired).sum())), "fig1_fpt_matrix: fired == False")
    cus = ft[(ft.dataset == "xjtu") & (ft.rule == "cusum")]
    add("CusumIndexMin", str(int(cus.fpt_index.min())), "XJTU-SY CUSUM fire index, min")
    add("CusumIndexMax", str(int(cus.fpt_index.max())), "XJTU-SY CUSUM fire index, max")
    add("XjtuLengthMin", str(cus.n_records.min()), "XJTU-SY trajectory length, min")
    # IEEE style: no separator in a four-digit number (2538); 2026-09-22, was f"{...:,}" -> "2,538"
    add("XjtuLengthMax", str(cus.n_records.max()), "XJTU-SY trajectory length, max")
    mb = ft[ft.dataset.isin(["xjtu", "femto", "ims"]) & ft.fired]
    cells = mb.groupby(["dataset", "rule", "family"]).inside_baseline.agg(["mean", "size"]).reset_index()
    cells = cells[cells["size"] >= 4]
    add("ChartInsideBaselinePct", f"{100 * cells[cells.family == 'control chart']['mean'].mean():.1f}",
        "mean over dataset x chart cells (>= 4 firing trajectories) of the share inside the baseline, XJTU-SY/FEMTO/IMS")
    add("DetectorInsideBaselinePct", f"{100 * cells[cells.family == 'detector']['mean'].mean():.1f}", "same, detector cells")
    r3 = sub[(sub.rule == "3sig_rms") & (sub.subset == "all")].pos_rate
    add("RmsRateMin", f"{r3.min():.3f}", "3sig_rms pooled positive rate, lowest dataset")
    add("RmsRateMax", f"{r3.max():.3f}", "3sig_rms pooled positive rate, highest dataset")
    xc = sub[(sub.dataset == "xjtu") & (sub.rule == "3sig_rms") & sub.subset.str.startswith("cond")].pos_rate
    add("XjtuConditionRateMin", f"{xc.min():.3f}", "3sig_rms pooled rate on one XJTU-SY condition, lowest")
    add("XjtuConditionRateMax", f"{xc.max():.3f}", "3sig_rms pooled rate on one XJTU-SY condition, highest")
    sx = shift[shift.dataset == "xjtu"]
    add("ChartShiftXjtu", f"{sx[sx.rule.isin(['ewma', 'cusum'])].shift_in_baseline_sd.median():.2f}",
        "j22_shift_at_alarm XJTU-SY: median shift at the alarm over EWMA and CUSUM onsets, baseline sd")
    add("DetectorShiftXjtu", f"{sx[sx.rule.isin(['3sig_rms', 'ma3sig', 'ma5sig'])].shift_in_baseline_sd.median():.2f}",
        "j22_shift_at_alarm XJTU-SY: median shift at the alarm over 3sig_rms, ma3sig, ma5sig onsets")
    ca = pd.read_csv(T / "j23_arl_condition_adjusted.csv").set_index(["dataset", "rule"])
    add("DetectorSlopeXjtu", f"{ca.loc[('xjtu', '3sig_rms')].slope_log_n_adjusted:+.2f}",
        "j23_arl_condition_adjusted: slope of log(i+1) on log n, 3sig_rms, condition-adjusted, XJTU-SY")
    add("ChartSlopeXjtu", f"{ca.loc[('xjtu', 'ewma')].slope_log_n_adjusted:+.2f}", "same for EWMA")
    cr = pd.read_csv(T / "j24_crossing_times.csv").set_index(["dataset", "k_sd"])
    add("CrossingRhoOneSd", f"{cr.loc[('xjtu', 1.0)].rho_index_length:+.3f}",
        "j24_crossing_times XJTU-SY: rho(record of first sustained crossing of mu+1sd, length)")
    add("CrossingRhoThreeSd", f"{cr.loc[('xjtu', 3.0)].rho_index_length:+.3f}", "same at three standard deviations")
    sim = pd.read_csv(T / "j24_sim_single_change.csv")
    st = sim[sim["shape"] == "step"].groupby(["delta_sd", "family"]).share_fire_before_change.median()
    add("SimChartPremature", f"{100 * st.xs('control chart', level='family').median():.0f}",
        "j24_sim_single_change: share of simulated series where a chart alarms before the known change, per cent")
    # ---- revision 2026-09-15: the design sweep, the length intervention and the baseline floor
    sw = pd.read_csv(T / "j26_design_sweep.csv")
    sw = sw[sw.median_shift_sd.notna() & sw.rho_index_length.notna()]
    add("SweepDesigns", str(sw[sw.dataset == "xjtu"].design.nunique()), "j26_design_sweep: rule designs applied unchanged")
    gx26 = sw[sw.dataset == "xjtu"]
    lx26 = np.log(gx26.median_shift_sd.clip(lower=1e-3).to_numpy(float))
    y26 = gx26.rho_index_length.to_numpy(float)
    d26 = (gx26.family == "detector").astype(float).to_numpy()
    r2 = {}
    for nm, X in (("size", np.column_stack([np.ones(len(y26)), lx26, lx26 ** 2])),
                  ("family", np.column_stack([np.ones(len(y26)), d26]))):
        b = np.linalg.lstsq(X, y26, rcond=None)[0]
        r = y26 - X @ b
        r2[nm] = 1 - (r @ r) / ((y26 - y26.mean()) @ (y26 - y26.mean()))
    add("SweepRSqSize", f"{r2['size']:.3f}", "j26_design_sweep XJTU-SY: R2 of rho on log(shift) and its square")
    add("SweepRSqFamily", f"{r2['family']:.3f}", "j26_design_sweep XJTU-SY: R2 of rho on the family label alone")
    sl26 = pd.read_csv(T / "j26b_slopes.csv").query("dataset == 'xjtu' and arm == 'prop'").set_index("rule")
    add("DecimSlopeDetector", f"{sl26.loc['3sig_rms'].slope_median:+.3f}",
        "j26b_slopes XJTU-SY: within-bearing slope of log(alarm index) on log n for 3sig_rms under decimation")
    dm26 = sl26[sl26.family == "detector"].slope_median.median()
    cm26 = sl26[sl26.family == "control chart"].slope_median.median()
    add("DecimGapXjtu", f"{dm26 - cm26:+.3f}", "j26b_slopes XJTU-SY: within-bearing detector-minus-chart slope gap")
    ma26 = pd.read_csv(T / "j26c_matched_arms.csv").set_index(["dataset", "arm"])
    add("SigmaRatioTenRecords", f"{ma26.loc[('xjtu', 'fixed10')].sigma_ratio_to_prop:.3f}",
        "j26c_matched_arms XJTU-SY: sigma-hat from a 10-record baseline over sigma-hat from the 20% window")
    km19 = pd.read_csv(T / "j28_km_onset.csv")
    cen19 = km19[(km19.dataset == "xjtu") & (km19.censored > 0)]
    add("CensoringRateShift", f"{(cen19.naive_pos_rate_traj - cen19.km_pos_rate).median():.3f}",
        "j28_km_onset XJTU-SY: median fall in the positive rate when non-firing is treated as censoring")
    cs30 = pd.read_csv(T / "j30_cross_study_summary.csv").set_index("dataset").loc["xjtu"]
    add("CrossStudyRateLi", f"{cs30.pos_rate_li_3sigma:.3f}",
        "j30_cross_study_summary XJTU-SY: pooled positive rate of the 3 sigma onsets published by Li et al. 2025")
    add("CrossStudyRateOurs", f"{cs30.pos_rate_our_3sig_rms:.3f}",
        "j30_cross_study_summary XJTU-SY: pooled positive rate of this study's 3sig_rms on the same 15 bearings")
    j32 = pd.read_csv(T / "j32_rul_target_summary.csv").set_index(["dataset", "class"])
    add("RulTargetRmseChartDetXjtu", f"{j32.loc[('xjtu', 'chart vs RMS detector')].rmse:.3f}",
        "j32_rul_target_summary XJTU-SY: median RMS difference of RUL targets, chart vs RMS detector")
    add("RulTargetRmseWithinDetXjtu", f"{j32.loc[('xjtu', 'within RMS detectors')].rmse:.3f}",
        "j32_rul_target_summary XJTU-SY: median RMS difference of RUL targets, between two RMS detectors")
    r = e2[["TP", "FP", "FN", "TN"]].drop_duplicates().iloc[0]
    add("NskTP", str(r.TP), "e2_raw_counts_h5 (identical for all seven models)")
    add("NskFN", str(r.FN), "e2_raw_counts_h5")
    add("NskTN", str(r.TN), "e2_raw_counts_h5")
    return out


def render() -> str:
    lines = ["% generated_numbers.tex -- written by scripts/J19_export_numbers.py; do not edit by hand.",
             "% Each macro is a number repeated in the abstract, Section 1 or Section 8. Use as \\Name{}."]
    for name, value, source in numbers():
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}  % {source}")
    return "\n".join(lines) + "\n"


def main() -> int:
    text = render()
    if "--check" in sys.argv:
        ok = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print(f"generated_numbers.tex {'is up to date' if ok else 'DIFFERS from a fresh export'}")
        return 0 if ok else 1
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
