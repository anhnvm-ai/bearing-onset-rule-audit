"""J65 -- WP-G: linear mixed models as a sensitivity analysis to the bearing-cluster bootstrap (R2.5).

R2.5: a comment of the review reports, answered in the response letter.

Every interval of the revision is a bearing-cluster bootstrap stratified by operating condition (the primary
inference). This script fits linear mixed models with a random intercept per bearing and operating condition as a
fixed effect (three levels, too few to be a random effect), with REML (statsmodels MixedLM):

  (i)   RUL, per bearing (J62, input variant (a), nine models):
            C-common RMSE on S_j  ~ labeler + model + condition + (1 | bearing)
            C-rank concordance    ~ labeler + model + condition + (1 | bearing)
        reported: Wald tests of the labeler and model terms, the standard deviation of the labeler main effects
        against that of the model main effects (a comparison that does not grow with the number of levels), and the
        variance components.
  (ii)  onset, per rule x bearing (J60, the three RMS detectors and the two charts, bearings eligible at N0 = 20):
            log u ~ log n * family + condition + (1 | bearing)
        reported: the log n x family interaction, i.e. the detector-minus-chart difference in b1 - 1, with its 95%
        interval, under P-retro and P-prospective. Gate 1 (rule G1-d) keeps this model because the onset results
        depend on the condition structure (J58: the onset fraction of 3sig_rms is 0.54 / 0.29 / 0.92 across the three
        XJTU-SY conditions).

Pre-specified in paper/access_resub/ANALYSIS_PLAN_2026-10-24.md, Section 12; Gate 1 rule G1-d.

Inputs : results/tables/j62_per_bearing.csv, results/tables/j60_onsets.csv
Outputs: results/tables/j65_mixed_rul.csv, results/tables/j65_mixed_onset.csv, results/logs/j65_run.log
Usage  : python scripts/J65_mixed_effects.py
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
NINE = ["lin:rms", "ridge:all", "knn:all", "gbr:all", "et:all", "rf:all", "two-stage", "gru", "tcn"]
RMS3, CHARTS = ["3sig_rms", "ma3sig", "ma5sig"], ["ewma", "cusum"]
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO"}
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


def wald(res, prefix: str) -> tuple[float, int, float]:
    """Joint Wald test that every coefficient whose name starts with prefix is zero."""
    from scipy.stats import chi2
    names = [n for n in res.fe_params.index if n.startswith(prefix)]
    if not names:
        return np.nan, 0, np.nan
    b = res.fe_params[names].to_numpy(float)
    V = res.cov_params().loc[names, names].to_numpy(float)     # fixed-effect block of the covariance
    stat = float(b @ np.linalg.solve(V, b))
    return stat, len(names), float(chi2.sf(stat, len(names)))


def main_effect_sd(res, prefix: str) -> float:
    """SD of the main effects of a factor (the reference level counted as 0)."""
    v = [0.0] + [float(res.fe_params[n]) for n in res.fe_params.index if n.startswith(prefix)]
    return float(np.std(v))


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j65_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J65  LINEAR MIXED MODELS: SENSITIVITY TO THE CLUSTER BOOTSTRAP  (WP-G)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}")

    # ---------------------------------------------------------------- (i) RUL
    pb = pd.read_csv(T / "j62_per_bearing.csv")
    pb = pb[(pb.variant == "a") & pb.model.isin(NINE)].copy()
    pb["rmse_common_S"] = np.sqrt(pb.sse_com_S / pb.n_S)
    rows = []
    log(f"\n{BAR}\n(i) RUL per bearing ~ labeler + model + condition + (1 | bearing)\n{BAR}")
    for (cfg, ds), g in pb.groupby(["config", "dataset"], sort=False):
        for metric in ("rmse_common_S", "conc_S"):
            d = g.dropna(subset=[metric]).copy()
            d["model_"] = d.model.str.replace(":", "_").str.replace("-", "_")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.mixedlm(f"{metric} ~ C(labeler) + C(model_) + C(condition)", d,
                                  groups=d["trajectory"]).fit(reml=True)
            wl, dfl, pl = wald(res, "C(labeler)")
            wm, dfm, pm = wald(res, "C(model_)")
            sdl, sdm = main_effect_sd(res, "C(labeler)"), main_effect_sd(res, "C(model_)")
            vb, ve = float(res.cov_re.iloc[0, 0]), float(res.scale)
            rows.append({"config": cfg, "dataset": ds, "metric": metric, "n_obs": len(d),
                         "n_bearings": d.trajectory.nunique(), "wald_labeler": wl, "df_labeler": dfl, "p_labeler": pl,
                         "wald_model": wm, "df_model": dfm, "p_model": pm, "sd_labeler_effects": sdl,
                         "sd_model_effects": sdm, "ratio_sd_labeler_to_model": sdl / sdm if sdm > 0 else np.nan,
                         "var_bearing": vb, "var_residual": ve, "converged": bool(res.converged)})
            log(f"  {cfg:24s} {DS_NAME[ds]:7s} {metric:14s}: labeler chi2 {wl:7.1f} (df {dfl}, p {pl:.2g}); "
                f"model chi2 {wm:7.1f} (df {dfm}, p {pm:.2g}); SD of effects labeler {sdl:.4f} vs model {sdm:.4f}; "
                f"bearing var {vb:.4g}, residual {ve:.4g}{'' if res.converged else '  (not converged)'}")
    pd.DataFrame(rows).to_csv(T / "j65_mixed_rul.csv", index=False)

    # ---------------------------------------------------------------- (ii) onset
    on = pd.read_csv(T / "j60_onsets.csv")
    on["fired"] = on.fired.astype(bool)
    rows2 = []
    log(f"\n{BAR}\n(ii) log u ~ log n * family + condition + (1 | bearing), RMS detectors and charts, n >= 80\n{BAR}")
    for arm in ("P-retro", "P-prospective"):
        for ds in ("xjtu", "femto"):
            d = on[(on.arm == arm) & (on.dataset == ds) & on.fired & (on.n_records >= 80) &
                   on.rule.isin(RMS3 + CHARTS)].copy()
            d["family"] = np.where(d.rule.isin(CHARTS), "chart", "detector")
            d["log_u"] = np.log(d.u)
            d["log_n"] = np.log(d.n_records.astype(float))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = smf.mixedlm("log_u ~ log_n * C(family, Treatment('chart')) + C(condition)", d,
                                  groups=d["trajectory"]).fit(reml=True)
            name = [n for n in res.fe_params.index if n.startswith("log_n:")][0]
            est, se = float(res.fe_params[name]), float(res.bse_fe[name])
            chart_slope = float(res.fe_params["log_n"])
            rows2.append({"protocol": arm, "dataset": ds, "n_obs": len(d), "n_bearings": d.trajectory.nunique(),
                          "chart_b1_minus_1": chart_slope, "detector_minus_chart": est,
                          "lo": est - 1.96 * se, "hi": est + 1.96 * se, "p": float(res.pvalues[name]),
                          "var_bearing": float(res.cov_re.iloc[0, 0]), "var_residual": float(res.scale),
                          "converged": bool(res.converged)})
            log(f"  {arm:13s} {DS_NAME[ds]:7s}: chart b1-1 {chart_slope:+.2f}; detector minus chart {est:+.2f} "
                f"[{est - 1.96 * se:+.2f}, {est + 1.96 * se:+.2f}], p {res.pvalues[name]:.2g} "
                f"({len(d)} rule-bearing rows, {d.trajectory.nunique()} bearings){'' if res.converged else '  (not converged)'}")
    pd.DataFrame(rows2).to_csv(T / "j65_mixed_onset.csv", index=False)
    log(f"\nwrote j65_mixed_rul.csv ({len(rows)}), j65_mixed_onset.csv ({len(rows2)})")
    log(f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
