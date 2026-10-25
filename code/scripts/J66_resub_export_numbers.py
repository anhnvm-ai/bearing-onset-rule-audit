"""J66 -- export the numbers of the resubmitted manuscript to paper/access_resub/.

Every number that the resubmission cites is written here as a LaTeX macro computed from results/tables, with the
table it came from in a comment, so that the text cannot drift from the analyses (plan, Section 12.4). Two table
bodies are written as well, because their cells are numbers: the rule panel of Table 2 and Table 4 (non-firing
trajectories).

Macro names start with "Rs" and contain letters only, so that they cannot collide with the macros of the submitted
version (generated_numbers.tex, revision_numbers.tex), which the working tree still loads.

Signed numbers and intervals carry an explicit sign ("+0.15", "[-0.07, +0.56]") and are meant for math mode, where
"-" is set as a minus sign: write $\\RsGapXjtuRetro$, not \\RsGapXjtuRetro.

Summary of the simulation (J63), used by the text and by Fig. 4: for each protocol, shape and rule, the median over
the 12 cells of that shape (three change points u0, two noise models, two end-of-life sizes delta); quoted ranges run
over the rules of a group and the three shapes. The plan fixes no summary for J63; this one is fixed here, before the
text is written.

Usage
    python scripts/J66_resub_export_numbers.py           write the files
    python scripts/J66_resub_export_numbers.py --check   exit 1 if a file on disk differs from a fresh export
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
OUT = ROOT / "paper" / "access_resub"
FILES = {"numbers": OUT / "resub_numbers.tex", "rules": OUT / "resub_tab_rules.tex",
         "nonfiring": OUT / "resub_tab_nonfiring.tex",
         "supp_configs": OUT / "supp_tab_configs.tex", "supp_models": OUT / "supp_tab_models.tex"}
OLD_MACROS = [OUT / "generated_numbers.tex", OUT / "revision_numbers.tex"]

DS = {"xjtu": "Xjtu", "femto": "Femto"}
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO", "ims": "IMS", "nsk": "NSK"}
ARM = {"P-retro": "Retro", "P-II": "PII", "P-prospective": "Prosp"}
PROT = {"P-retro": "Retro", "P-prospective": "Prosp"}
CFG = {"P-retro|canonical": "RetroCan", "P-prospective|canonical": "ProspCan",
       "P-prospective|matched5": "ProspCal", "P-retro|matched5": "RetroCal"}
CHARTS = ["ewma", "cusum"]
RMS_DET = ["3sig_rms", "ma3sig", "ma5sig"]
CAL_RULES = ["thr_cal", "ma_cal", "ewma_cal", "cusum_cal"]
CANONICAL_DESIGN = {"3sig_rms": "thr_3sd", "ma2sig": "ma5_2sd", "ma3sig": "ma5_3sd", "ma5sig": "ma5_5sd",
                    "ewma": "ewma_l0.2_L3", "cusum": "cusum_k0.5_h5"}
CAL_FAMILY = {"3sig_rms": ("thr", "k"), "ma2sig": ("ma", "k"), "ma3sig": ("ma", "k"), "ma5sig": ("ma", "k"),
              "ewma": ("ewma", "L"), "cusum": ("cusum", "H")}
RULE_ROWS = [  # group, rule, indicator, alarm condition (equations of Section III)
    ("Detector rules", "3sig_rms", "RMS", r"(\ref{eq:detector}), $k = 3$"),
    ("Detector rules", "3sig_kurt", "kurtosis", r"(\ref{eq:detector}), $k = 3$"),
    ("Detector rules", "ma2sig", "MA of RMS", r"(\ref{eq:detector}), $k = 2$"),
    ("Detector rules", "ma3sig", "MA of RMS", r"(\ref{eq:detector}), $k = 3$"),
    ("Detector rules", "ma5sig", "MA of RMS", r"(\ref{eq:detector}), $k = 5$"),
    ("Control charts", "ewma", "RMS", r"(\ref{eq:ewma}), $\lambda = 0.2$, $L = 3$"),
    ("Control charts", "cusum", "RMS", r"(\ref{eq:cusum}), $\kappa = 0.5\sigma$, $H = 5\sigma$"),
    ("Percentile rules", "pct02", "none", r"(\ref{eq:percentile}), $p = 0.02$"),
    ("Percentile rules", "pct05", "none", r"(\ref{eq:percentile}), $p = 0.05$"),
    ("Percentile rules", "pct10", "none", r"(\ref{eq:percentile}), $p = 0.10$"),
]
NCOL_RULES = 9


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(T / f"{name}.csv")


def _r(x, d: int) -> float:
    v = round(float(x), d)
    return 0.0 if v == 0 else v


def f(x, d: int = 2) -> str:
    return f"{_r(x, d):.{d}f}"


def s(x, d: int = 2) -> str:
    """Signed, except that a value rounding to zero is written without a sign."""
    v = _r(x, d)
    return f"{v:.{d}f}" if v == 0 else f"{v:+.{d}f}"


def ci(lo, hi, d: int = 2) -> str:
    return f"[{s(lo, d)}, {s(hi, d)}]"


def pct(x, d: int = 0) -> str:
    return f"{_r(100 * float(x), d):.{d}f}"


def sci(x: float) -> str:
    mant, exp = f"{x:.0e}".split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


def pv(p: float) -> str:
    """A p-value: two significant digits, or a power of ten below 0.001."""
    return f"${sci(p)}$" if p < 0.001 else f"{p:.2g}"


def tt(rule: str) -> str:
    return r"\texttt{" + rule.replace("_", r"\_") + "}"


class Macros:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, name: str, value, source: str) -> None:
        if not re.fullmatch(r"Rs[A-Za-z]+", name):
            raise ValueError(f"macro name must be Rs + letters: {name}")
        if any(n == name for n, _, _ in self.rows):
            raise ValueError(f"duplicate macro {name}")
        self.rows.append((name, str(value), source))

    def get(self, name: str) -> str:
        return next(v for n, v, _ in self.rows if n == name)


# ------------------------------------------------------------------ Section III: data and design
def data_design(M: Macros) -> None:
    j57, j59, cr = read("j57_censoring_2x2"), read("j59_rule_configurations"), read("j62_common_reference")
    pc, cal = read("j60_protocol_curve"), read("j61_calibration")
    one = j57[j57.rule == "3sig_rms"].set_index("dataset")
    M.add("RsTrajectories", int(one.n_traj.sum()), "j57: trajectories over the four datasets")
    for ds, nm in (("xjtu", "Xjtu"), ("femto", "Femto"), ("ims", "Ims"), ("nsk", "Nsk")):
        M.add(f"RsTraj{nm}", int(one.loc[ds, "n_traj"]), f"j57: trajectories, {DS_NAME[ds]}")
    M.add("RsRules", j57.rule.nunique(), "j57: onset rules")
    M.add("RsDesigns", int((j59.set == "design").sum()), "j59_rule_configurations: designs of the sweep")
    M.add("RsConfigs", len(j59), "j59_rule_configurations: rules plus designs")
    M.add("RsModels", cr.model.nunique(), "j62_common_reference: prediction models")
    M.add("RsResamples", r"20\,000", "B of J60-J65 and J69: bearing resamples stratified by condition")
    for ds in DS:
        n = pc[(pc.dataset == ds) & (pc.bearing_set == "main")].n_bearings.unique()
        assert len(n) == 1
        M.add(f"RsElig{DS[ds]}", int(n[0]), f"j60_protocol_curve: bearings with n >= 80, {DS_NAME[ds]}")
        phi = cal[(cal.dataset == ds) & (cal.noise_arm == "main")].phi.unique()
        assert len(phi) == 1
        M.add(f"RsPhi{DS[ds]}", f(phi[0], 3), f"j61_calibration: AR(1) coefficient of the null model, {DS_NAME[ds]}")
    ft = read("fig1_fpt_matrix").drop_duplicates(["dataset", "trajectory"])
    for ds, nm in (("xjtu", "Xjtu"), ("femto", "Femto"), ("ims", "Ims"), ("nsk", "Nsk")):
        n80 = int((ft[ft.dataset == ds].n_records >= 80).sum())
        if ds in DS:
            assert n80 == int(M.get(f"RsElig{nm}")), f"n >= 80 count differs between fig1_fpt_matrix and J60, {ds}"
        else:
            M.add(f"RsElig{nm}", n80, f"fig1_fpt_matrix: trajectories with n >= 80, {DS_NAME[ds]}")
    xl = ft[ft.dataset == "xjtu"].n_records
    M.add("RsXjtuLenMin", int(xl.min()), "fig1_fpt_matrix: shortest XJTU-SY trajectory, records")
    M.add("RsXjtuLenMax", int(xl.max()), "fig1_fpt_matrix: longest XJTU-SY trajectory, records")
    M.add("RsCalibrations", len(cal), "j61_calibration: calibrations run")
    M.add("RsCalibrationsOk", int(cal.within_tolerance.sum()), "j61_calibration: calibrations within tolerance")
    M.add("RsCalMaxErr", f(100 * (cal.achieved_far - cal.level).abs().max(), 2),
          "j61_calibration: largest |achieved minus target FAR|, percentage points")
    M.add("RsCalMaxSe", f(100 * cal.mc_se.max(), 2), "j61_calibration: largest Monte Carlo standard error of the achieved FAR, percentage points")


# ------------------------------------------------------------------ Section IV-A: baseline protocol
def protocol(M: Macros) -> None:
    gap, pc, bc, w = read("j60_gap"), read("j60_protocol_curve"), read("j58_by_condition"), read("j60_warmup")
    g = gap[gap.bearing_set == "main"].set_index(["dataset", "arm"])
    for ds in DS:
        M.add(f"RsInsideCharts{DS[ds]}", pct(g.loc[(ds, "P-retro"), "inside_share_charts"]),
              f"j60_gap: chart onsets inside the P-retro baseline, per cent, {DS_NAME[ds]}")
        M.add(f"RsInsideDet{DS[ds]}", pct(g.loc[(ds, "P-retro"), "inside_share_rms_detectors"]),
              f"j60_gap: RMS-detector onsets inside the P-retro baseline, per cent, {DS_NAME[ds]}")
        for arm, an in ARM.items():
            r = g.loc[(ds, arm)]
            M.add(f"RsGap{DS[ds]}{an}", s(r.gap_b1_minus_1),
                  f"j60_gap: median b1-1 of the RMS detectors minus that of the charts, {DS_NAME[ds]}, {arm}")
            M.add(f"RsGap{DS[ds]}{an}Ci", ci(r.gap_lo, r.gap_hi), "same, 95% bearing-bootstrap interval")
    for ds in DS:                     # the rest of the protocol curve (Fig. 2)
        for arm, an in (("P-robust", "Robust"), ("P-iter", "Iter"), ("P-fixed(10)", "FixTen"),
                        ("P-fixed(15)", "FixFifteen"), ("P-fixed(20)", "FixTwenty")):
            r = g.loc[(ds, arm)]
            M.add(f"RsGap{DS[ds]}{an}", s(r.gap_b1_minus_1), f"j60_gap: detector-minus-chart gap, {DS_NAME[ds]}, {arm}")
            M.add(f"RsGap{DS[ds]}{an}Ci", ci(r.gap_lo, r.gap_hi), "same, 95% interval")
    on = read("j60_onsets")
    for arm, an in (("P-retro", "Retro"), ("P-prospective", "Prosp")):
        q = on[(on.dataset == "xjtu") & (on.rule == "ewma") & (on.arm == arm) & on.eligible & on.fired]
        M.add(f"RsEwmaEarly{an}", int((q.u < 0.05).sum()),
              f"j60_onsets: XJTU-SY bearings with n >= 80 whose EWMA onset lies in the first 5% of life, {arm}")
    p = pc[pc.bearing_set == "main"].set_index(["dataset", "arm", "rule"])
    for rule, rn in (("3sig_rms", "Rms"), ("ewma", "Ewma"), ("cusum", "Cusum")):
        for ds in DS:
            for arm, an in ARM.items():
                r = p.loc[(ds, arm, rule)]
                M.add(f"RsSlope{rn}{DS[ds]}{an}", s(r.slope_b1_minus_1), f"j60_protocol_curve: b1-1, {rule}, {DS_NAME[ds]}, {arm}")
                M.add(f"RsSlope{rn}{DS[ds]}{an}Ci", ci(r.slope_lo, r.slope_hi), "same, 95% interval")
    al = pd.concat([w[w[c] < 0.05 * w.n_records] for c in ("i_ewma", "i_cusum")])
    on_all = read("j60_onsets")
    ins = on_all[(on_all.arm == "P-retro") & on_all.fired & on_all.inside_baseline.astype(bool) & on_all.rule.isin(SIGNAL7)]
    con = read("j60_contamination")
    M.add("RsContamOnsets", len(ins), "j60_onsets: P-retro onsets of the signal-based rules inside the baseline, all datasets")
    M.add("RsContamUsed", len(con), "j60_contamination: those with at least 20 records before the alarm")
    M.add("RsContamDropped", len(ins) - len(con), "same, with fewer than 20 records before the alarm")
    M.add("RsContamLarge", int((con.D_j_retro_vs_reestimated > 0.3).sum()), "j60_contamination: re-estimated targets with D_j above 0.3")
    ac = read("j22_baseline_autocorrelation")
    for ds in DS:
        q = ac[ac.dataset == ds]
        M.add(f"RsPhiRaw{DS[ds]}", f(q.phi_raw.median(), 3), f"j22_baseline_autocorrelation: median lag-1 autocorrelation of the baseline, {DS_NAME[ds]}")
        M.add(f"RsTrendHalf{DS[ds]}", int((q.trend_r2 > 0.5).sum()), f"same: trajectories on which a linear trend explains more than half of the baseline variance, {DS_NAME[ds]}")
    M.add("RsWarmupAlarms", len(al), "j60_warmup: chart alarms before record 0.05 n (H_w, condition 2)")
    M.add("RsWarmupShare", f((al.e10 > 1).mean()), "j60_warmup: share of those alarms on bearings with e10 > 1")
    M.add("RsWarmupNeeded", "0.67", "PRE Section 7: H_w needs at least two thirds")
    b = bc[(bc.dataset == "xjtu") & (bc.rule == "3sig_rms")].set_index("condition")
    for c, cn in ((1, "One"), (2, "Two"), (3, "Three")):
        M.add(f"RsRmsUCond{cn}", f(b.loc[c, "median_u"]), f"j58_by_condition: median onset fraction, 3sig_rms, XJTU-SY condition {c}")
    M.add("RsCondThreeRecordsPct", pct(b.loc[3, "records"] / b.records.sum(), 1),
          "j58_by_condition: XJTU-SY records in condition 3, per cent")


# ------------------------------------------------------------------ Section IV-B: false-alarm calibration
def calibration(M: Macros) -> None:
    df, cal, mg, arl = read("j61_design_far"), read("j61_calibration"), read("j61_matched_gap"), read("j61_arl0")
    can = df[(df.noise_arm == "main") & df.canonical & (df.dataset == "xjtu")]
    ch = can[can.design.isin([CANONICAL_DESIGN[r] for r in CHARTS])].far
    det = can[can.design.isin([CANONICAL_DESIGN[r] for r in RMS_DET])].far
    M.add("RsFarChartMin", f(ch.min()), "j61_design_far: FAR of EWMA and CUSUM at canonical settings, XJTU-SY, both protocols, lowest")
    M.add("RsFarChartMax", f(ch.max()), "same, highest")
    M.add("RsFarRmsDetMax", f(det.max(), 3), "j61_design_far: FAR of 3sig_rms, ma3sig, ma5sig at canonical settings, XJTU-SY, both protocols, highest")
    ma2 = can[(can.design == CANONICAL_DESIGN["ma2sig"]) & (can.protocol == "P-prospective")].far.iloc[0]
    M.add("RsFarMaTwo", f(ma2), "j61_design_far: FAR of ma2sig at canonical settings, XJTU-SY, P-prospective")
    m = cal[(cal.noise_arm == "main") & (cal.level == 0.05)].set_index(["dataset", "protocol", "rule"])
    for ds in DS:
        for prot, pn in PROT.items():
            for rule, rn, d in (("thr", "Thr", 2), ("ma", "Ma", 2), ("ewma", "Ewma", 2), ("cusum", "Cusum", 1)):
                M.add(f"RsMult{rn}{DS[ds]}{pn}", f(m.loc[(ds, prot, rule), "multiplier"], d),
                      f"j61_calibration: multiplier at 5% FAR, {rule}, {DS_NAME[ds]}, {prot}")
    g = mg[(mg.noise_arm == "main") & (mg.level == 0.05)].set_index(["dataset", "protocol"])
    for ds in DS:
        for prot, pn in PROT.items():
            r, k = g.loc[(ds, prot)], f"{DS[ds]}{pn}"
            M.add(f"RsCalGap{k}", s(r.gap_calibrated), f"j61_matched_gap: detector-minus-chart b1-1 at 5% FAR, {DS_NAME[ds]}, {prot}")
            M.add(f"RsCalGap{k}Ci", ci(r.gap_cal_lo, r.gap_cal_hi), "same, 95% interval")
            M.add(f"RsDisagreeUnc{k}", f(r.disagreement_uncalibrated_median),
                  f"j61_matched_gap: median chart-detector label disagreement at canonical settings, {DS_NAME[ds]}, {prot}")
            M.add(f"RsDisagreeCal{k}", f(r.disagreement_calibrated_median), "same, at 5% FAR")
            M.add(f"RsDjCal{k}", f(r.median_D_j_calibrated, 3), f"j61_matched_gap: median chart-detector D_j at 5% FAR, {DS_NAME[ds]}, {prot}")
    M.add("RsCalReadingOne", int(mg.reading.str.startswith("1").sum()), "j61_matched_gap: configurations with reading 1")
    M.add("RsCalCells", len(mg), "j61_matched_gap: configurations")
    a5 = arl[arl.level == 0.05]
    M.add("RsArlAbove", int((a5.reported == "> 10,000").sum()), "j61_arl0: calibrated rules with ARL0 above 10 000 records at 5% FAR")
    M.add("RsArlCells", len(a5), "j61_arl0: calibrated rules at 5% FAR")


# ------------------------------------------------------------------ Section IV-C: simulation
def simulation(M: Macros) -> None:
    g = read("j63_gradual")
    med = (g.groupby(["protocol", "shape", "rule"])
           .agg(early=("share_before_ref1", "median"), slope=("slope_b1_minus_1", "median"), n=("n_series", "size"))
           .reset_index())
    assert (med.n == 12).all(), "every protocol x shape x rule must have 12 cells"
    for prot, pn in PROT.items():
        for gn, rules in (("Chart", CHARTS), ("Cal", CAL_RULES), ("Det", RMS_DET)):
            q = med[(med.protocol == prot) & med.rule.isin(rules)]
            src = f"j63_gradual: median over 12 cells per shape; range over {', '.join(rules)} and the 3 shapes, {prot}"
            M.add(f"RsSim{gn}Early{pn}Min", pct(q.early.min()), src + "; alarms before t_1, per cent, lowest")
            M.add(f"RsSim{gn}Early{pn}Max", pct(q.early.max()), "same, highest")
            M.add(f"RsSim{gn}Slope{pn}Min", s(q.slope.min()), src + "; b1-1, lowest")
            M.add(f"RsSim{gn}Slope{pn}Max", s(q.slope.max()), "same, highest")
    x = g[(g["shape"] == "exp") & (g.delta == 6) & g.rule.isin(RMS_DET)]
    miss = x.groupby(["protocol", "rule"]).share_never.median()
    for prot, pn in PROT.items():
        M.add(f"RsSimDetMissExp{pn}", pct(miss.loc[prot].max()),
              f"j63_gradual: exponential degradation, delta = 6: largest median miss share of an RMS detector, per cent, {prot}")


# ------------------------------------------------------------------ Section IV-D: design parameters
def design(M: Macros) -> None:
    j = read("j64_models").set_index(["protocol", "set", "dataset", "model"])
    cells = {("P-prospective", "main", "xjtu"): "ProspXjtu", ("P-prospective", "main", "femto"): "ProspFemto",
             ("P-retro", "main", "xjtu"): "RetroXjtu", ("P-retro", "main", "femto"): "RetroFemto",
             ("P-retro", "all", "xjtu"): "RetroAllXjtu"}
    for (prot, st, ds), cn in cells.items():
        for model, mn in (("M0", "Family"), ("M1", "Shift"), ("M2", "Far"), ("M3", "FarFamily")):
            M.add(f"RsRsq{mn}{cn}", f(j.loc[(prot, st, ds, model), "r2_in"]), f"j64_models: in-sample R2 of {model}, {prot}, {st}, {DS_NAME[ds]}")
        d = j.loc[(prot, st, ds, "M3-M2")]
        M.add(f"RsDeltaRsq{cn}", s(d.r2_in), f"j64_models: R2 of M3 minus M2, {prot}, {st}, {DS_NAME[ds]}")
        M.add(f"RsDeltaRsq{cn}Ci", ci(d.r2_boot_lo, d.r2_boot_hi), "same, 95% interval")
        M.add(f"RsReliability{cn}", s(d.reliability_median), f"j64_models: median split-half reliability of design b1-1, {prot}, {st}, {DS_NAME[ds]}")
    jj = j.reset_index()
    main = jj[(jj.set == "main") & jj.model.isin(["M0", "M1", "M2", "M3", "M4"])]
    neg = main.groupby(["protocol", "dataset"]).r2_oos_median.max() < 0
    M.add("RsOosNegCells", int(neg.sum()), "j64_models: main cells where every model has a negative median out-of-sample R2")
    M.add("RsOosCells", len(neg), "j64_models: main cells")


# ------------------------------------------------------------------ Section V-A: labels
def labels(M: Macros) -> None:
    lab, j57 = read("j60_labels"), read("j57_censoring_2x2")
    x = lab[lab.dataset == "xjtu"]
    pr = x[x.kind == "positive_rate"].set_index(["arm", "rule"]).pos_rate_record
    for arm, an in ARM.items():
        v = pr.loc[arm].loc[CHARTS]
        M.add(f"RsPosChart{an}Min", f(v.min()), f"j60_labels: record positive rate of EWMA and CUSUM, XJTU-SY, {arm}, lower")
        M.add(f"RsPosChart{an}Max", f(v.max()), "same, higher")
    dv = pd.concat([pr.loc[arm].loc[RMS_DET] for arm in ARM])
    M.add("RsPosDetMin", f(dv.min()), "j60_labels: record positive rate of the RMS detectors, XJTU-SY, P-retro, P-II, P-prospective, lowest")
    M.add("RsPosDetMax", f(dv.max()), "same, highest")
    td = x[x.kind == "target_distance"].set_index(["arm", "rule"]).median_D_j
    for arm, an in ARM.items():
        M.add(f"RsDj{an}", f(td.loc[(arm, "chart vs RMS detector")], 3), f"j60_labels: median chart-detector D_j, XJTU-SY, {arm}")
    M.add("RsDjWithin", f(td.loc[("P-retro", "within RMS detectors")], 3), "j60_labels: median D_j within the RMS detectors, XJTU-SY, P-retro")
    M.add("RsPosDetFixTenMax", f(pr.loc["P-fixed(10)"].loc[RMS_DET].max()),
          "j60_labels: record positive rate of the RMS detectors, XJTU-SY, P-fixed(10), highest")
    fe = lab[(lab.dataset == "femto") & (lab.kind == "target_distance")].set_index(["arm", "rule"]).median_D_j
    for arm, an in PROT.items():
        M.add(f"RsDjFemto{an}", f(fe.loc[(arm, "chart vs RMS detector")], 3), f"j60_labels: median chart-detector D_j, FEMTO, {arm}")
    M.add("RsRuleTrajPairs", int(j57.n_traj.sum()), "j57: rule-trajectory pairs")
    M.add("RsNeverFire", int(j57.n_nonfiring.sum()), "j57: rule-trajectory pairs that never fire")
    M.add("RsNonfiringPairs", int((j57.n_nonfiring > 0).sum()), "j57: dataset-rule pairs with a non-firing trajectory")
    M.add("RsPairsFiring", int((j57.n_fired > 0).sum()), "j57: dataset-rule pairs in which the rule fires on at least one trajectory")
    M.add("RsDsRulePairs", len(j57), "j57: dataset-rule pairs")
    M.add("RsDefectPairs", int(((j57.n_fired > 0) & (j57.n_nonfiring > 0)).sum()),
          "j57: dataset-rule pairs in which the rule fires on some trajectories but not on all (rows of the defect table)")
    mr = read("j65_mixed_rul")
    nonconv = int((~mr.converged.astype(bool)).sum())
    M.add("RsMixedNonconv", nonconv, "j65_mixed_rul: RUL mixed models that did not converge")
    M.add("RsMixedNonconvText", {1: "One", 2: "Two", 3: "Three"}.get(nonconv, str(nonconv)), "same, as a word that can open a sentence")
    M.add("RsMixedModels", len(mr), "j65_mixed_rul: RUL mixed models")
    M.add("RsMixedSdModelLarger", int((mr.sd_model_effects > mr.sd_labeler_effects).sum()),
          "j65_mixed_rul: models in which the SD of the model main effects exceeds that of the labeler main effects")
    rk, cm = mr[mr.metric == "conc_S"], mr[mr.metric == "rmse_common_S"]
    M.add("RsMixedModelSigRank", int((rk.p_model < 0.05).sum()), "j65_mixed_rul: C-rank configurations with p < 0.05 for the model term")
    M.add("RsMixedCommonBothSig", int(((cm.p_model < 0.05) & (cm.p_labeler < 0.05)).sum()),
          "j65_mixed_rul: C-common RMSE configurations with p < 0.05 for both the labeler and the model term")
    km = j57.km_minus_bearing_healthy.abs().dropna()
    M.add("RsKmMaxDiff", sci(km.max()), "j57: largest |Kaplan-Meier minus bearing-weighted healthy rate|")
    M.add("RsKmCells", len(km), "j57: dataset-rule pairs compared")
    k = j57[(j57.dataset == "xjtu") & (j57.rule == "3sig_kurt")].iloc[0]
    for col, nm in (("rate_record_exclude", "RecExcl"), ("rate_record_healthy", "RecHealthy"),
                    ("rate_bearing_exclude", "BearExcl"), ("rate_bearing_healthy", "BearHealthy")):
        M.add(f"RsKurt{nm}", f(k[col], 3), f"j57: {col}, XJTU-SY, 3sig_kurt")
    nf = j57[(j57.n_nonfiring > 0) & (j57.n_fired > 0)]
    for ds in DS:
        M.add(f"RsTreatBear{DS[ds]}", f(nf[nf.dataset == ds].effect_treatment_bearing.median(), 3),
              f"j57: exclude minus healthy under bearing weights, median over pairs with a non-firing trajectory, {DS_NAME[ds]}")


# ------------------------------------------------------------------ Section V-B: RUL on the common reference
def rul(M: Macros) -> None:
    cr, rk, me = read("j62_common_reference"), read("j62_rankings"), read("j62_main_effect")
    c69, mx = read("j69_labeler_model_contrast"), read("j65_mixed_rul")
    a = cr[cr.variant == "a"]
    for cfg, cn in CFG.items():
        for ds in DS:
            M.add(f"RsRulBear{DS[ds]}{cn}", int(a[(a.config == cfg) & (a.dataset == ds)].n_bearings.iloc[0]),
                  f"j62_common_reference: bearings on which all four labelers fire, {DS_NAME[ds]}, {cfg}")
    cc = rk[rk.metric == "C-common RMSE"]
    sS, s3 = cc[cc.set == "S"], cc[cc.set == "S3"]
    assert sS[sS.dataset == "xjtu"].n_significant_reversals.sum() == 0
    for cfg, cn in CFG.items():
        M.add(f"RsRevFemto{cn}", int(sS[(sS.config == cfg) & (sS.dataset == "femto")].n_significant_reversals.sum()),
              f"j62_rankings: significant reversals under C-common, scoring set S, FEMTO, {cfg}")
    M.add("RsRevTotal", int(sS.n_significant_reversals.sum()), "j62_rankings: significant reversals, scoring set S, all configurations")
    hit = sS[sS.n_significant_reversals > 0]
    assert ((hit.labeler_a == "pct05") | (hit.labeler_b == "pct05")).all(), "a reversal under S without pct05"
    M.add("RsRevSthree", int(s3.n_significant_reversals.sum()), "j62_rankings: significant reversals, scoring set S3, all configurations")
    can = a[a.config.str.endswith("canonical")]
    eight = can[can.model != "two-stage"]
    pct05 = eight[eight.labeler == "pct05"].rmse_common_S
    other = eight[eight.labeler != "pct05"].rmse_common_S
    M.add("RsPctRmseMin", f(pct05.min(), 3), "j62_common_reference: C-common RMSE on S under pct05, eight models without two-stage, canonical configurations, lowest")
    M.add("RsPctRmseMax", f(pct05.max(), 3), "same, highest")
    M.add("RsOtherRmseMin", f(other.min()), "same, under the three signal-based labelers, lowest")
    M.add("RsOtherRmseMax", f(other.max()), "same, highest")
    ts = can[can.model == "two-stage"].rmse_common_S
    M.add("RsTwoStageRmseMin", f(ts.min()), "j62_common_reference: C-common RMSE on S of the two-stage model, canonical configurations, lowest")
    M.add("RsTwoStageRmseMax", f(ts.max()), "same, highest")
    worst = a.loc[a.groupby(["config", "dataset", "labeler"]).rmse_common_S.idxmax(), "model"]
    M.add("RsTwoStageWorst", int((worst == "two-stage").sum()), "j62_common_reference: labeler cells in which the two-stage model has the highest C-common RMSE on S")
    M.add("RsTwoStageCells", len(worst), "j62_common_reference: labeler cells")
    m3 = me[me.set == "S3"]
    ce = m3[m3.config.str.endswith("canonical") & (m3.metric == "C-common RMSE") & m3.excludes_0
            & ((m3.labeler_a == "ewma") | (m3.labeler_b == "ewma"))]
    M.add("RsEwmaEffectMin", f(ce.diff_mean_over_models.abs().min()), "j62_main_effect: S3, canonical, C-common RMSE, EWMA against a detector, intervals excluding 0, smallest |difference|")
    M.add("RsEwmaEffectMax", f(ce.diff_mean_over_models.abs().max()), "same, largest")
    cal3 = m3[m3.config.str.endswith("matched5")]
    M.add("RsSthreeCalXjtuSig", int(cal3[cal3.dataset == "xjtu"].excludes_0.sum()), "j62_main_effect: S3, 5% FAR, XJTU-SY, labeler contrasts whose interval excludes 0")
    fe = cal3[(cal3.dataset == "femto") & cal3.excludes_0]
    M.add("RsSthreeCalFemtoRmse", f(fe[fe.metric == "C-common RMSE"].diff_mean_over_models.abs().max(), 3),
          "j62_main_effect: S3, 5% FAR, FEMTO, largest |labeler difference| with an interval excluding 0, C-common RMSE")
    M.add("RsSthreeCalFemtoConc", f(fe[fe.metric == "C-rank concordance"].diff_mean_over_models.abs().max(), 3), "same, C-rank concordance")
    r = c69[c69.metric == "C-rank concordance"]
    nine, nts = r[r.pool == "nine"], r[r.pool == "no two-stage"]
    M.add("RsAsixRankMin", f(nine.ratio.min()), "j69: C-rank, labeler-to-model ratio of Gini mean differences, nine models, lowest")
    M.add("RsAsixRankMax", f(nine.ratio.max()), "same, highest")
    M.add("RsAsixRankHiMax", f(nine.hi.max()), "j69: same, largest upper 95% bound")
    M.add("RsAsixRankOk", int((nine.hi < 1).sum()), "j69: C-rank cells with the upper bound below 1, nine models")
    M.add("RsAsixRankOkNoTs", int((nts.hi < 1).sum()), "j69: same, without the two-stage model")
    M.add("RsAsixCells", len(nine), "j69: cells (4 configurations x 2 datasets x S, S3)")
    M.add("RsAsixRankNoTsMax", f(nts.ratio.max()), "j69: C-rank ratio without the two-stage model, highest")
    cS = c69[(c69.metric == "C-common RMSE") & (c69.set == "S") & (c69.pool == "no two-stage")]
    M.add("RsAsixCommonMin", f(cS.ratio.min(), 1), "j69: C-common RMSE, S, without the two-stage model, ratio, lowest")
    M.add("RsAsixCommonMax", f(cS.ratio.max(), 1), "same, highest")
    M.add("RsAsixCommonAbove", int((cS.lo > 1).sum()), "j69: same, cells with the lower bound above 1")
    M.add("RsAsixCommonCells", len(cS), "j69: C-common RMSE, cells per scoring set (4 configurations x 2 datasets)")
    cN = c69[(c69.metric == "C-common RMSE") & (c69.set == "S") & (c69.pool == "nine")]
    M.add("RsAsixCommonNineMin", f(cN.ratio.min()), "j69: C-common RMSE, S, nine models, ratio, lowest")
    M.add("RsAsixCommonNineMax", f(cN.ratio.max()), "same, highest")
    c3 = c69[(c69.metric == "C-common RMSE") & (c69.set == "S3")]
    c3n, c3t = c3[c3.pool == "nine"], c3[c3.pool == "no two-stage"]
    M.add("RsAsixCommonSthreeMin", f(c3n.ratio.min()), "j69: C-common RMSE, S3, nine models, ratio, lowest")
    M.add("RsAsixCommonSthreeMax", f(c3n.ratio.max()), "same, highest")
    M.add("RsAsixCommonSthreeIncl", int(((c3t.lo <= 1) & (c3t.hi >= 1)).sum()),
          "j69: C-common RMSE, S3, without the two-stage model, cells whose interval includes 1")
    cs = mx[mx.metric == "conc_S"]
    ns = cs[cs.p_labeler >= 0.05]
    M.add("RsMixedRankNs", len(ns), "j65_mixed_rul: configurations in which the labeler term on C-rank has p >= 0.05")
    M.add("RsMixedCfgs", len(cs), "j65_mixed_rul: configurations")
    M.add("RsMixedRankPMin", f(ns.p_labeler.min()), "j65_mixed_rul: those p-values, lowest")
    M.add("RsMixedRankPMax", f(ns.p_labeler.max()), "same, highest")
    for ds in DS:
        q = cr[cr.dataset == ds].groupby(["config", "variant"]).rmse_common_S.median().unstack()
        for v in "abc":
            M.add(f"RsVar{v.upper()}{DS[ds]}Min", f(q[v].min()),
                  f"j62_common_reference: median C-common RMSE on S over labelers and models, input variant ({v}), {DS_NAME[ds]}, lowest over configurations")
            M.add(f"RsVar{v.upper()}{DS[ds]}Max", f(q[v].max()), "same, highest")
        best = {"xjtu": "b", "femto": "c"}[ds]     # the variant that Section S8 names as lowest on this dataset
        M.add(f"RsVarBest{DS[ds]}", int((q.idxmin(axis=1) == best).sum()),
              f"j62_common_reference: configurations in which variant ({best}) has the lowest median, {DS_NAME[ds]}")
    M.add("RsVarCfgs", len(q), "j62_common_reference: configurations")


# ------------------------------------------------------------------ table bodies
def table_rules() -> str:
    df, cal, pc = read("j61_design_far"), read("j61_calibration"), read("j60_protocol_curve")
    far = df[(df.dataset == "xjtu") & (df.noise_arm == "main") & (df.protocol == "P-prospective") & df.canonical].set_index("design").far
    mult = cal[(cal.dataset == "xjtu") & (cal.noise_arm == "main") & (cal.level == 0.05)
               & (cal.protocol == "P-prospective")].set_index("rule").multiplier
    p = pc[pc.bearing_set == "main"].set_index(["dataset", "arm", "rule"]).slope_b1_minus_1
    lines = ["% resub_tab_rules.tex -- written by scripts/J66_resub_export_numbers.py; do not edit by hand.",
             "% Rule panel of Table 2. FAR and multiplier: XJTU-SY, P-prospective, Gaussian AR(1) null (j61_design_far,",
             "% j61_calibration). b1 - 1: bearings with n >= 80 (j60_protocol_curve). Signed numbers in math mode."]
    group = None
    for grp, rule, ind, cond in RULE_ROWS:
        if grp != group:
            lines.append(rf"\multicolumn{{{NCOL_RULES}}}{{@{{}}l}}{{\textbf{{{grp}}}}} \\")
            group = grp
        if rule in CANONICAL_DESIGN:
            v = far[CANONICAL_DESIGN[rule]]
            fa = "$<0.001$" if v < 0.0005 else f(v, 3)
        else:
            fa = "---"
        if rule in CAL_FAMILY:
            fam, sym = CAL_FAMILY[rule]
            mu = f"${sym} = {f(mult[fam], 1)}\\sigma$" if fam == "cusum" else f"${sym} = {f(mult[fam], 2)}$"
        else:
            mu = "---"
        sl = []
        for ds in DS:
            for arm in PROT:
                key = (ds, arm, rule)
                sl.append(f"${s(p[key])}$" if key in p.index else "---")
        lines.append(" & ".join([tt(rule), ind, cond, fa, mu] + sl) + r" \\")
    return "\n".join(lines) + "\n"


def table_nonfiring() -> str:
    j57 = read("j57_censoring_2x2")
    nf = j57[(j57.n_nonfiring > 0) & (j57.n_fired > 0)]
    lines = ["% resub_tab_nonfiring.tex -- written by scripts/J66_resub_export_numbers.py; do not edit by hand.",
             "% Table 4: dataset-rule pairs with a non-firing trajectory (j57_censoring_2x2). Positive rate with",
             "% non-firing trajectories excluded or labeled healthy, pooled over records or averaged over bearings."]
    for ds in ("xjtu", "femto", "ims"):
        for _, r in nf[nf.dataset == ds].iterrows():
            cells = [DS_NAME[ds], tt(r.rule), f"{int(r.n_fired)}/{int(r.n_traj)}",
                     f(r.rate_record_exclude, 3), f(r.rate_record_healthy, 3),
                     f(r.rate_bearing_exclude, 3), f(r.rate_bearing_healthy, 3)]
            lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ supplementary tables (Section S1)
KIND_LABEL = {"threshold": "threshold", "ma": "moving average", "ewma": "EWMA", "cusum": "CUSUM",
              "ewma_ar": "AR-corrected EWMA", "adaptive": "adaptive-limit chart",
              "detector": "detector", "control chart": "control chart", "percentile": "percentile"}


def _num(v) -> str:
    return f"{float(v):g}"


def config_parameters(r) -> str:
    k = r.kind
    if k in ("detector", "threshold"):
        return f"$k = {_num(r.k)}$"
    if k == "ma":
        return f"$k = {_num(r.k)}$, window {int(r.ma_window)}"
    if k == "percentile":
        return f"$p = {_num(r.percentile_p)}$"
    if k in ("ewma", "control chart") and pd.notna(r["lambda"]):
        extra = ", AR-corrected limit" if k == "ewma_ar" else ""
        return f"$\\lambda = {_num(r['lambda'])}$, $L = {_num(r.L)}${extra}"
    if k == "ewma_ar":
        return f"$\\lambda = {_num(r['lambda'])}$, $L = {_num(r.L)}$, AR-corrected limit"
    if k in ("cusum", "control chart"):
        return f"$\\kappa = {_num(r.kappa_sd)}\\sigma$, $H = {_num(r.H_sd)}\\sigma$"
    if k == "adaptive":
        return f"$L = {_num(r.L)}$, expanding-window limits"
    return "---"


def supp_table_configs() -> str:
    r = read("j59_rule_configurations")
    r = pd.concat([r[r.set == "rule"], r[r.set != "rule"]])          # the ten rules first, then the designs
    lines = ["% supp_tab_configs.tex -- written by scripts/J66_resub_export_numbers.py; do not edit by hand.",
             "% Table S1 rows: the ten rules and the 67 designs (j59_rule_configurations). Nominal shift: the shift, in",
             "% baseline standard deviations, to which a design responds by construction (definitions of J41)."]
    for _, x in r.iterrows():
        name = f"{x.id} {tt(x['name'])}" if x.set == "rule" else tt(x.id)
        fam = KIND_LABEL.get(x.family if x.set == "rule" else x.kind, x.kind)
        ind = x.indicator if isinstance(x.indicator, str) and x.indicator != "none" else "---"
        ind = "MA of RMS" if ind.startswith("five-record") else ind
        shift = f(x.nominal_shift_A) if pd.notna(x.nominal_shift_A) else "---"
        canon = "---" if x.set == "rule" else ("yes" if str(x.canonical).lower() == "true" else "")
        lines.append(" & ".join([name, fam, ind, config_parameters(x), shift, canon]) + r" \\")
    return "\n".join(lines) + "\n"


def supp_table_models() -> str:
    import json
    m = read("j59_model_settings").set_index("model")
    s = {k: json.loads(v) for k, v in m.settings.items()}
    rows = [
        ("Linear regression", "RMS", "ordinary least squares on standardized RMS"),
        ("Ridge regression", "per-record statistics", f"$\\alpha = {_num(s['ridge:all']['ridge__alpha'])}$, standardized inputs"),
        ("$k$-nearest neighbors", "per-record statistics",
         f"$k = {s['knn:all']['kneighborsregressor__n_neighbors']}$, {s['knn:all']['kneighborsregressor__weights']} weights, standardized inputs"),
        ("Gradient boosting", "per-record statistics",
         f"{s['gbr:all']['n_estimators']} trees, depth {s['gbr:all']['max_depth']}, learning rate {_num(s['gbr:all']['learning_rate'])}"),
        ("Extra trees", "per-record statistics", f"{s['et:all']['n_estimators']} trees, no bootstrap"),
        ("Random forest", "per-record statistics", f"{s['rf:all']['n_estimators']} trees, bootstrap"),
        ("Two-stage exponential", "RMS over its baseline mean",
         f"log RMS linear in time from the onset, extrapolated to the median end-of-life level of the other bearings; at least {s['two-stage (exp)']['min_records_to_fit']} records"),
        ("GRU", f"window of {s['gru']['window']} records",
         f"one GRU layer of {s['gru']['architecture']['gru']['hidden_size']} units; {s['gru']['epochs']} epochs, batch {s['gru']['batch']}, Adam ($10^{{-3}}$); median over seeds 0--2"),
        ("TCN", f"window of {s['tcn']['window']} records",
         f"three dilated convolutions (32 channels, kernel 3, dilations 1, 2, 4); training as GRU"),
    ]
    lines = ["% supp_tab_models.tex -- written by scripts/J66_resub_export_numbers.py; do not edit by hand.",
             "% Table S2 rows: the nine models (j59_model_settings, settings read from the code by J59)."]
    lines += [" & ".join(row) + r" \\" for row in rows]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ supplementary tables (Sections S2-S5)
SIGNAL7 = ["3sig_rms", "3sig_kurt", "ma2sig", "ma3sig", "ma5sig", "ewma", "cusum"]
ARMS8 = ["P-retro", "P-II", "P-robust", "P-iter", "P-fixed(10)", "P-fixed(15)", "P-fixed(20)", "P-prospective"]
CAL_LABEL = {"thr_cal": "threshold", "ma_cal": "moving average", "ewma_cal": "EWMA", "cusum_cal": "CUSUM"}


def _head(name: str, what: str) -> list[str]:
    return [f"% {name} -- written by scripts/J66_resub_export_numbers.py; do not edit by hand.", f"% {what}"]


def _ci_cell(v, lo, hi, d: int = 2) -> str:
    return f"${s(v, d)}$ ${ci(lo, hi, d)}$" if pd.notna(v) else "---"


def supp_protocurve() -> str:
    pc = read("j60_protocol_curve")
    p = pc[pc.bearing_set == "main"].set_index(["dataset", "arm", "rule"]).slope_b1_minus_1
    lines = _head("supp_tab_protocurve.tex", "Table rows: b1-1 for each rule and protocol, bearings with n >= 80 (j60_protocol_curve).")
    for ds in DS:
        lines.append(rf"\multicolumn{{9}}{{@{{}}l}}{{\textbf{{{DS_NAME[ds]}}}}} \\")
        for rule in SIGNAL7:
            cells = [tt(rule)] + [f"${s(p[(ds, a, rule)])}$" if (ds, a, rule) in p.index else "---" for a in ARMS8]
            lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def supp_gaps() -> str:
    g = read("j60_gap")
    lines = _head("supp_tab_gaps.tex", "Table rows: detector-minus-chart gap per protocol, on the common and on each protocol's own bearings (j60_gap).")
    for ds in DS:
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textbf{{{DS_NAME[ds]}}}}} \\")
        for arm in ARMS8 + ["P-fixed(30)"]:
            main = g[(g.dataset == ds) & (g.bearing_set == "main") & (g.arm == arm)]
            own = g[(g.dataset == ds) & (g.bearing_set == f"own:{arm}") & (g.arm == arm)]
            mr = main.iloc[0] if len(main) else None
            orow = own.iloc[0] if len(own) else None
            src = mr if mr is not None else orow
            ins_c = pct(src.inside_share_charts) if src is not None and pd.notna(src.inside_share_charts) else "---"
            ins_d = pct(src.inside_share_rms_detectors) if src is not None and pd.notna(src.inside_share_rms_detectors) else "---"
            cells = [arm,
                     str(int(mr.n_bearings)) if mr is not None else "---",
                     _ci_cell(mr.gap_b1_minus_1, mr.gap_lo, mr.gap_hi) if mr is not None else "---",
                     str(int(orow.n_bearings)) if orow is not None else "---",
                     _ci_cell(orow.gap_b1_minus_1, orow.gap_lo, orow.gap_hi) if orow is not None else "---",
                     ins_c, ins_d]
            lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def supp_contamination() -> str:
    c = read("j60_contamination")
    lines = _head("supp_tab_contamination.tex", "Table rows: P-retro onsets inside the baseline with at least 20 records before the alarm (j60_contamination).")
    for _, r in c.iterrows():
        cells = [DS_NAME[r.dataset], r.trajectory.replace("_", r"\_"), tt(r.rule), str(int(r.n_records)), str(int(r.W_retro)),
                 str(int(r.i_star_retro)), f(r.sd_ratio), f"${s(r.mu_shift_in_retro_sd)}$", str(int(r.i_star_reestimated)),
                 f(r.pos_share_retro), f(r.pos_share_reestimated), f(r.D_j_retro_vs_reestimated, 3)]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def supp_warmup() -> str:
    w = read("j60_warmup_test")
    lines = _head("supp_tab_warmup.tex", "Table rows: Spearman correlation between e10 and the chart onset, P-retro (j60_warmup_test).")
    for _, r in w.iterrows():
        lines.append(" & ".join([tt(r.chart), DS_NAME[r.dataset], str(int(r.n)), _ci_cell(r.rho_e10_istar, r.lo, r.hi)]) + r" \\")
    return "\n".join(lines) + "\n"


def supp_calibration() -> str:
    c = read("j61_calibration").set_index(["dataset", "protocol", "rule", "noise_arm", "level"])
    lines = _head("supp_tab_calibration.tex", "Table rows: multipliers and false-alarm rates of the calibrated rules (j61_calibration).")
    sym = {"thr": "k", "ma": "k", "ewma": "L", "cusum": "H"}
    for ds in DS:
        lines.append(rf"\multicolumn{{10}}{{@{{}}l}}{{\textbf{{{DS_NAME[ds]}}}}} \\")
        for prot in PROT:
            for rule in ("thr", "ma", "ewma", "cusum"):
                d = 1 if rule == "cusum" else 2
                base = c.loc[(ds, prot, rule, "main", 0.05)]
                cells = [prot if rule == "thr" else "", f"{CAL_LABEL[rule + '_cal']} (${sym[rule]}$)",
                         f(base.canonical_multiplier, d), f(base.far_at_canonical, 3)]
                cells += [f(c.loc[(ds, prot, rule, "main", lv), "multiplier"], d) for lv in (0.01, 0.05, 0.10)]
                cells += [f(c.loc[(ds, prot, rule, arm, 0.05), "multiplier"], d) for arm in ("t3", "phi_q25", "phi_q75")]
                lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def supp_arl() -> str:
    a = read("j61_arl0").set_index(["dataset", "protocol", "rule", "level"]).reported
    lines = _head("supp_tab_arl.tex", "Table rows: in-control average run length of the calibrated rules, records (j61_arl0).")
    for ds in DS:
        for prot in PROT:
            for rule in ("thr", "ma", "ewma", "cusum"):
                cells = [DS_NAME[ds] if (prot == "P-retro" and rule == "thr") else "", prot if rule == "thr" else "", CAL_LABEL[rule + "_cal"]]
                cells += [str(a[(ds, prot, rule, lv)]).replace(",", r"\,").replace(">", r"$>$") for lv in (0.01, 0.05, 0.10)]
                lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def supp_matched() -> str:
    m = read("j61_matched_gap")
    arm_label = {"main": "Gaussian", "t3": "t(3)", "phi_q25": r"$\phi$ lower quartile", "phi_q75": r"$\phi$ upper quartile"}
    lines = _head("supp_tab_matched.tex", "Table rows: chart-detector contrasts after calibration, every configuration (j61_matched_gap).")
    for _, r in m.iterrows():
        cells = [DS_NAME[r.dataset], r.protocol, arm_label[r.noise_arm], f"{int(round(100 * r.level))}\\%",
                 f"${s(r.gap_uncalibrated)}$", _ci_cell(r.gap_calibrated, r.gap_cal_lo, r.gap_cal_hi),
                 f"{f(r.disagreement_uncalibrated_median)} $\\to$ {f(r.disagreement_calibrated_median)}",
                 f(r.median_D_j_calibrated, 3)]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def supp_fraction() -> str:
    j = read("j58_onset_fraction")
    lines = _head("supp_tab_fraction.tex", "Table rows: onset fraction and scaling of the ten rules, P-retro, all bearings (j58_onset_fraction).")
    for ds in DS:
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textbf{{{DS_NAME[ds]}}}}} \\")
        for _, r in j[j.dataset == ds].iterrows():
            cells = [tt(r.rule), f"{int(r.n_fired)}/{int(r.n_traj)}", f(r.median_u),
                     _ci_cell(r.slope_b1_minus_1, r.slope_lo, r.slope_hi), _ci_cell(r.rho_u_n, r.rho_u_lo, r.rho_u_hi),
                     f"${s(r.rho_istar_n)}$" if pd.notna(r.rho_istar_n) else "---"]
            lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def supp_conditions() -> str:
    b = read("j58_by_condition")
    lines = _head("supp_tab_conditions.tex", "Table rows: per operating condition, P-retro, all bearings (j58_by_condition).")
    for ds in DS:
        for cond in sorted(b[b.dataset == ds].condition.unique()):
            q = b[(b.dataset == ds) & (b.condition == cond)]
            for i, (_, r) in enumerate(q.iterrows()):
                cells = [DS_NAME[ds] if i == 0 else "", str(cond) if i == 0 else "", tt(r.rule), f"{int(r.n_fired)}/{int(r.n_traj)}",
                         f"{int(r.length_min)}--{int(r.length_max)}" if i == 0 else "",
                         f(r.median_u) if pd.notna(r.median_u) else "---", f(r.median_shift_d) if pd.notna(r.median_shift_d) else "---",
                         str(int(r.inside_baseline)), f(r.pos_rate_record), f(r.pos_rate_bearing)]
                lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def supp_decimation() -> str:
    d = read("j26b_slopes")
    lines = _head("supp_tab_decimation.tex", "Table rows: within-bearing slope of log onset on log length when every m-th record is kept (j26b_slopes).")
    for ds in ("xjtu", "femto", "ims"):
        for arm, al in (("prop", "20\\% of the thinned length"), ("fixed", "fixed in records")):
            q = d[(d.dataset == ds) & (d.arm == arm)]
            for i, (_, r) in enumerate(q.iterrows()):
                cells = [DS_NAME[ds] if (i == 0 and arm == "prop") else "", al if i == 0 else "", tt(r.rule), str(int(r.n_bearings)),
                         f"${s(r.slope_median)}$", f"${ci(r.slope_q25, r.slope_q75)}$"]
                lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def supp_sim(protocol: str) -> str:
    g = read("j63_gradual")
    g = g[g.protocol == protocol]
    med = g.groupby(["shape", "rule"]).agg(early=("share_before_ref1", "median"), never=("share_never", "median"),
                                           slope=("slope_b1_minus_1", "median"))
    lines = _head(f"supp_tab_sim_{PROT[protocol].lower()}.tex", f"Table rows: simulation under {protocol}, median over the 12 cells of each shape (j63_gradual).")
    for grp, rules in (("Conventional settings", RMS_DET + CHARTS), ("Matched 5\\% false-alarm rate", CAL_RULES)):
        lines.append(rf"\multicolumn{{10}}{{@{{}}l}}{{\textbf{{{grp}}}}} \\")
        for rule in rules:
            cells = [tt(rule) if rule in RMS_DET + CHARTS else CAL_LABEL[rule]]
            for shp in ("step", "ramp", "exp"):
                r = med.loc[(shp, rule)]
                cells += [pct(r.early), pct(r.never), f"${s(r.slope)}$"]
            lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def supp_known_change() -> str:
    d = read("j24_sim_single_change")
    d = d[d["shape"] == "step"]
    lines = _head("supp_tab_known_change.tex", "Table rows: earlier step-change simulation, P-retro, median over the rules of a family and the change points (j24_sim_single_change).")
    for delta in sorted(d.delta_sd.unique()):
        cells = [f"$\\delta = {delta:g}$"]
        for rules in (CHARTS, RMS_DET):
            q = d[(d.delta_sd == delta) & d.rule.isin(rules)]
            cells += [f(q.share_fire_before_change.median(), 3), f(q.share_never_fire.median(), 3),
                      f"{q.median_delay_records.median():.0f}"]
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ supplementary tables (Sections S6-S9)
CELL_LABEL = {("P-prospective", "main"): "P-prospective, $n \\geq 80$", ("P-retro", "main"): "P-retro, $n \\geq 80$",
              ("P-retro", "all"): "P-retro, all bearings"}
MODEL_SHORT = {"lin:rms": "Linear", "ridge:all": "Ridge", "knn:all": "$k$-NN", "gbr:all": "Boosting", "et:all": "Extra trees",
               "rf:all": "Forest", "two-stage": "Two-stage", "gru": "GRU", "tcn": "TCN"}
CFG_LABEL = {"P-retro|canonical": "P-retro, conventional", "P-prospective|canonical": "P-prospective, conventional",
             "P-prospective|matched5": "P-prospective, 5\\% FAR", "P-retro|matched5": "P-retro, 5\\% FAR"}
NINE_M = list(MODEL_SHORT)


def supp_design_models() -> str:
    j = read("j64_models")
    lines = _head("supp_tab_design_models.tex", "Table rows: models M0-M4 of the design-parameter analysis (j64_models).")
    for (prot, st, ds), q in j.groupby(["protocol", "set", "dataset"], sort=False):
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\textbf{{{DS_NAME[ds]}, {CELL_LABEL[(prot, st)]}}}}} \\")
        for _, r in q.iterrows():
            if r.model == "M3-M2":
                cells = [r"$\Delta R^2$ (M3 $-$ M2)", f"${s(r.r2_in)}$", f"${ci(r.r2_boot_lo, r.r2_boot_hi)}$", "---",
                         f"${s(r.reliability_median)}$", f(r.ceiling_r2)]
            else:
                cells = [r.model, f(r.r2_in), f"${ci(r.r2_boot_lo, r.r2_boot_hi)}$".replace("+", ""),
                         f"${s(r.r2_oos_median)}$ ${ci(r.r2_oos_q25, r.r2_oos_q75)}$", "", ""]
            lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def supp_design_outcomes() -> str:
    d = read("j64_design_outcomes")
    d = d[d.set == "main"].set_index(["dataset", "protocol", "design"])
    designs = list(dict.fromkeys(read("j64_design_outcomes").design))
    lines = _head("supp_tab_design_outcomes.tex", "Table rows: false-alarm rate and onset scaling of each design, n >= 80 (j64_design_outcomes).")
    for dz in designs:
        r0 = d.loc[("xjtu", "P-prospective", dz)]
        cells = [tt(dz), KIND_LABEL.get(r0.family, r0.family), f(r0.nominal_shift),
                 f(d.loc[("xjtu", "P-prospective", dz), "far"], 3), f(d.loc[("xjtu", "P-retro", dz), "far"], 3)]
        for ds in DS:
            for prot in ("P-prospective", "P-retro"):
                key = (ds, prot, dz)
                v = d.loc[key, "outcome_b1_minus_1"] if key in d.index else float("nan")
                cells.append(f"${s(v)}$" if pd.notna(v) else "---")
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def supp_nonfiring_full() -> str:
    j = read("j57_censoring_2x2")
    yn = lambda v: "pass" if str(v).lower() == "true" else ("fail" if str(v).lower() == "false" else "---")
    lines = _head("supp_tab_nonfiring_full.tex", "Table rows: treatment x weighting for all dataset-rule pairs, with the prevalence screen (j57_censoring_2x2).")
    for _, r in j.iterrows():
        km = f(r.km_j28, 3) if pd.notna(r.km_j28) else "---"
        rates = [f(r[c], 3) if pd.notna(r[c]) else "---" for c in
                 ("rate_record_exclude", "rate_record_healthy", "rate_bearing_exclude", "rate_bearing_healthy")]
        screens = [yn(r[c]) for c in ("screen_record_exclude", "screen_record_healthy", "screen_bearing_exclude", "screen_bearing_healthy")]
        lines.append(" & ".join([DS_NAME[r.dataset], tt(r.rule), f"{int(r.n_fired)}/{int(r.n_traj)}"] + rates + [km] + screens) + r" \\")
    return "\n".join(lines) + "\n"


def _rul_grid(name: str, col: str, labs: list[str], d: int) -> str:
    a = read("j62_common_reference")
    a = a[a.variant == "a"].set_index(["config", "dataset", "labeler", "model"])[col]
    lines = _head(name, f"Table rows: {col} for every configuration, dataset, labeler and model, input variant (a) (j62_common_reference).")
    for cfg in CFG:
        for ds in DS:
            lines.append(rf"\multicolumn{{10}}{{@{{}}l}}{{\textbf{{{DS_NAME[ds]}, {CFG_LABEL[cfg]}}}}} \\")
            for lab in labs:
                lines.append(" & ".join([tt(lab)] + [f(a[(cfg, ds, lab, m)], d) for m in NINE_M]) + r" \\")
    return "\n".join(lines) + "\n"


def supp_rul_common() -> str:
    return _rul_grid("supp_tab_rul_common.tex", "rmse_common_S", ["3sig_rms", "ma5sig", "ewma", "pct05"], 3)


def supp_rul_common_s3() -> str:
    return _rul_grid("supp_tab_rul_common_s3.tex", "rmse_common_S3", ["3sig_rms", "ma5sig", "ewma"], 3)


def supp_rul_rank() -> str:
    return _rul_grid("supp_tab_rul_rank.tex", "conc_S_mean", ["3sig_rms", "ma5sig", "ewma", "pct05"], 2)


def supp_rul_own() -> str:
    a = read("j62_common_reference")
    a = a[(a.variant == "a") & (a.config == "P-retro|canonical")].set_index(["dataset", "labeler", "model"]).rmse_own
    lines = _head("supp_tab_rul_own.tex", "Table rows: RMSE on each labeler's own target, P-retro, conventional settings (j62_common_reference, rmse_own).")
    for ds in DS:
        lines.append(rf"\multicolumn{{10}}{{@{{}}l}}{{\textbf{{{DS_NAME[ds]}}}}} \\")
        for lab in ("3sig_rms", "ma5sig", "ewma", "pct05"):
            lines.append(" & ".join([tt(lab)] + [f(a[(ds, lab, m)], 3) for m in NINE_M]) + r" \\")
    return "\n".join(lines) + "\n"


def supp_reversals() -> str:
    r = read("j62_rankings")
    r = r[(r.metric == "C-common RMSE") & (r.n_significant_reversals > 0)]
    lines = _head("supp_tab_reversals.tex", "Table rows: significant reversals under C-common (j62_rankings).")
    for _, x in r.iterrows():
        pairs = "; ".join(" vs ".join(MODEL_SHORT[m.strip()] for m in p.split(" vs ")) for p in x.reversals.split(";"))
        lines.append(" & ".join([DS_NAME[x.dataset], CFG_LABEL[x.config], f"${x.set}$", f"{tt(x.labeler_a)} vs {tt(x.labeler_b)}",
                                 str(int(x.n_significant_reversals)), pairs]) + r" \\")
    return "\n".join(lines) + "\n"


def supp_main_effects() -> str:
    m = read("j62_main_effect")
    lines = _head("supp_tab_main_effects.tex", "Table rows: labeler main effects, difference in the mean over the nine models (j62_main_effect).")
    for _, x in m.iterrows():
        lines.append(" & ".join([DS_NAME[x.dataset], CFG_LABEL[x.config], f"${x.set}$", "C-common" if x.metric.startswith("C-common") else "C-rank",
                                 f"{tt(x.labeler_a)} $-$ {tt(x.labeler_b)}", f"${s(x.diff_mean_over_models, 3)}$",
                                 f"${ci(x.lo, x.hi, 3)}$", "yes" if x.excludes_0 else ""]) + r" \\")
    return "\n".join(lines) + "\n"


def supp_variants() -> str:
    a = read("j62_common_reference")
    q = a.groupby(["config", "dataset", "variant"]).rmse_common_S.median().unstack()
    lines = _head("supp_tab_variants.tex", "Table rows: median C-common RMSE on S over labelers and models, per input variant (j62_common_reference).")
    for cfg in CFG:
        for ds in DS:
            lines.append(" & ".join([CFG_LABEL[cfg], DS_NAME[ds]] + [f(q.loc[(cfg, ds), v], 3) for v in "abcd"]) + r" \\")
    return "\n".join(lines) + "\n"


def supp_asix() -> str:
    c = read("j69_labeler_model_contrast")
    lines = _head("supp_tab_asix.tex", "Table rows: amendment A6, ratio of labeler to model effects (j69_labeler_model_contrast).")
    for _, x in c.iterrows():
        loo = f"{f(x.loo_max_ratio)} ({MODEL_SHORT[x.loo_worst_drop]})" if pd.notna(x.loo_max_ratio) else ""
        lines.append(" & ".join([DS_NAME[x.dataset], CFG_LABEL[x.config], f"${x.set}$", "C-common" if x.metric.startswith("C-common") else "C-rank",
                                 "nine" if x.pool == "nine" else "eight", f(x.ratio), f"${ci(x.lo, x.hi)}$".replace("+", ""),
                                 f(x.share_below_1), loo]) + r" \\")
    return "\n".join(lines) + "\n"


def supp_mixed_onset() -> str:
    m = read("j65_mixed_onset")
    lines = _head("supp_tab_mixed_onset.tex", "Table rows: onset mixed model, detector-minus-chart difference in b1-1 (j65_mixed_onset).")
    for _, x in m.iterrows():
        lines.append(" & ".join([x.protocol, DS_NAME[x.dataset], str(int(x.n_obs)), str(int(x.n_bearings)), f"${s(x.chart_b1_minus_1)}$",
                                 _ci_cell(x.detector_minus_chart, x.lo, x.hi), pv(x.p), f(x.var_bearing, 3), f(x.var_residual, 3),
                                 "yes" if x.converged else "no"]) + r" \\")
    return "\n".join(lines) + "\n"


def supp_mixed_rul() -> str:
    m = read("j65_mixed_rul")
    lines = _head("supp_tab_mixed_rul.tex", "Table rows: RUL mixed models, Wald tests and effect sizes (j65_mixed_rul).")
    for _, x in m.iterrows():
        metric = "C-common RMSE" if x.metric.startswith("rmse") else "C-rank"
        lines.append(" & ".join([DS_NAME[x.dataset], CFG_LABEL[x.config], metric,
                                 f"{x.wald_labeler:.1f} ({int(x.df_labeler)})", pv(x.p_labeler),
                                 f"{x.wald_model:.1f} ({int(x.df_model)})", pv(x.p_model),
                                 f(x.sd_labeler_effects, 3), f(x.sd_model_effects, 3), "yes" if x.converged else "no"]) + r" \\")
    return "\n".join(lines) + "\n"


SUPP_TABLES_2 = {"supp_design_models": supp_design_models, "supp_design_outcomes": supp_design_outcomes,
                 "supp_nonfiring_full": supp_nonfiring_full, "supp_rul_common": supp_rul_common,
                 "supp_rul_common_s3": supp_rul_common_s3, "supp_rul_rank": supp_rul_rank, "supp_rul_own": supp_rul_own,
                 "supp_reversals": supp_reversals, "supp_main_effects": supp_main_effects, "supp_variants": supp_variants,
                 "supp_asix": supp_asix, "supp_mixed_onset": supp_mixed_onset, "supp_mixed_rul": supp_mixed_rul}

SUPP_TABLES = {"supp_protocurve": supp_protocurve, "supp_gaps": supp_gaps, "supp_contamination": supp_contamination,
               "supp_warmup": supp_warmup, "supp_calibration": supp_calibration, "supp_arl": supp_arl,
               "supp_matched": supp_matched, "supp_fraction": supp_fraction, "supp_conditions": supp_conditions,
               "supp_decimation": supp_decimation, "supp_sim_prosp": lambda: supp_sim("P-prospective"),
               "supp_sim_retro": lambda: supp_sim("P-retro"), "supp_known_change": supp_known_change,
               **SUPP_TABLES_2}
FILES.update({k: OUT / f"supp_tab_{k[5:]}.tex" for k in SUPP_TABLES})


# ------------------------------------------------------------------ self-checks against earlier outputs
def self_check(M: Macros) -> list[str]:
    """Values that earlier scripts printed in their logs or that the submitted paper reports."""
    expect = {"RsTrajectories": "33", "RsRules": "10", "RsDesigns": "67", "RsModels": "9", "RsWarmupAlarms": "21",
              "RsWarmupShare": "0.52", "RsInsideChartsXjtu": "62", "RsTreatBearXjtu": "0.102", "RsTreatBearFemto": "0.094",
              "RsNeverFire": "36", "RsRuleTrajPairs": "330", "RsDisagreeUncXjtuRetro": "0.70", "RsCalReadingOne": "24",
              "RsRevTotal": "14", "RsRevSthree": "1", "RsAsixRankOk": "16", "RsAsixRankOkNoTs": "13",
              "RsContamOnsets": "39", "RsContamUsed": "11", "RsPhiRawXjtu": "0.763", "RsPhiRawFemto": "0.579"}
    bad = [f"{k}: {M.get(k)} against {v}" for k, v in expect.items() if M.get(k) != v]
    # counts that the text reports as "all <total>"
    whole = [("RsTwoStageWorst", "RsTwoStageCells"), ("RsMixedModelSigRank", "RsMixedCfgs"),
             ("RsMixedCommonBothSig", "RsMixedCfgs"), ("RsMixedSdModelLarger", "RsMixedModels"),
             ("RsAsixRankOk", "RsAsixCells"), ("RsVarBestXjtu", "RsVarCfgs"), ("RsVarBestFemto", "RsVarCfgs")]
    return bad + [f"{a} = {M.get(a)}, but the text says all {M.get(b)}" for a, b in whole if M.get(a) != M.get(b)]


def build() -> dict[str, str]:
    M = Macros()
    for part in (data_design, protocol, calibration, simulation, design, labels, rul):
        part(M)
    old = set()
    for p in OLD_MACROS:
        if p.exists():
            old |= set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", p.read_text(encoding="utf-8")))
    clash = [n for n, _, _ in M.rows if n in old]
    if clash:
        raise SystemExit(f"FAIL: macro names already used by the submitted version: {clash}")
    bad = self_check(M)
    if bad:
        raise SystemExit("FAIL: J66 does not reproduce earlier outputs: " + "; ".join(bad))
    head = ["% resub_numbers.tex -- written by scripts/J66_resub_export_numbers.py; do not edit by hand.",
            "% Every number of the resubmission (Access-2026-51094), with the table it comes from. Use as \\Name{};",
            "% signed numbers and intervals belong in math mode."]
    body = [f"\\newcommand{{\\{n}}}{{{v}}}  % {src}" for n, v, src in M.rows]
    return {"numbers": "\n".join(head + body) + "\n", "rules": table_rules(), "nonfiring": table_nonfiring(),
            "supp_configs": supp_table_configs(), "supp_models": supp_table_models(),
            **{k: fn() for k, fn in SUPP_TABLES.items()}}


def main() -> int:
    check = "--check" in sys.argv[1:]
    out = build()
    if check:
        stale = [k for k, txt in out.items() if not FILES[k].exists() or FILES[k].read_text(encoding="utf-8") != txt]
        for k in stale:
            print(f"STALE: {FILES[k].relative_to(ROOT)} differs from a fresh export")
        if not stale:
            n = out["numbers"].count("\\newcommand")
            print(f"OK: {len(FILES)} files up to date ({n} macros)")
        return 1 if stale else 0
    for k, txt in out.items():
        FILES[k].write_text(txt, encoding="utf-8", newline="\n")
        print(f"wrote {FILES[k].relative_to(ROOT)}")
    print(f"{out['numbers'].count(chr(92) + 'newcommand')} macros")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
