"""J59 -- every rule and design configuration, and every model setting, as tables (R1.6).

R1.6: a comment of the review reports, answered in the response letter.

Nothing is computed from data. The tables are read off the code that produced the results, so they cannot drift
from it:
  * the 67 designs of the response-size sweep, from J26.build_grid(), one parameter per column, with the nominal
    shift of J41 (convention A, the primary one: k for a threshold, k/sqrt(5) for a five-record moving average,
    L*sqrt(lambda/(2-lambda)) for an EWMA, 2*kappa for a CUSUM, L for an adaptive limit) and the update rules of
    the two special classes as J26 implements them;
  * the ten rules of Table 2, with their parameters, indicator, the conventions every rule shares, and the design of
    the sweep that implements the same rule;
  * the models of the RUL comparison: the six regressors of J35, instantiated from J35.models() and read back with
    get_params() so the table is what the code runs; the two-stage model of J39; the GRU and TCN of J47,
    instantiated from J47.GRUHead / J47.TCNHead and read back from the torch modules.

The false-alarm rate of each design is added by J61 (WP-A), which computes it.

Pre-specified in paper/access_resub/ANALYSIS_PLAN_2026-10-24.md, Section 13 item 3.

Outputs: results/tables/j59_rule_configurations.csv   67 designs + 10 rules, one row each
         results/tables/j59_model_settings.csv        one row per model
         results/logs/j59_run.log
Usage  : python scripts/J59_configurations.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
BASELINE_RULE = "first max(5, floor(0.2 n)) records"
SD_CONVENTION = "population standard deviation (ddof = 0)"
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def design_rows(j26, j41) -> list[dict]:
    conv = j41.CONVENTIONS["A_primary"]
    rows = []
    for s in j26.build_grid():
        p, kind = s["p"], s["kind"]
        rows.append({
            "set": "design", "id": s["design"], "kind": kind, "family": s["family"], "indicator": "RMS",
            "k": p.get("k") if kind in ("threshold", "ma") else np.nan,
            "lambda": p.get("lam", np.nan),
            "L": p.get("L", np.nan),
            "kappa_sd": p["k"] if kind == "cusum" else np.nan,
            "H_sd": p["h"] if kind == "cusum" else np.nan,
            "ma_window": 5 if kind == "ma" else np.nan,
            "percentile_p": np.nan,
            "confirmation_c": j26.CONFIRM,
            "baseline_rule": BASELINE_RULE, "sd_convention": SD_CONVENTION,
            "chart_start": ("EWMA starts at the baseline mean" if kind in ("ewma", "ewma_ar") else
                            "CUSUM starts at 0" if kind == "cusum" else ""),
            "ar_correction": ("limit multiplied by sqrt((1+phi(1-lambda))/(1-phi(1-lambda))), phi = lag-1 "
                              "autocorrelation of the linearly detrended baseline" if kind == "ewma_ar" else ""),
            "adaptive_update": ("after the baseline, mean and sd of every record seen so far (expanding window); "
                                "sd never below the baseline sd" if kind == "adaptive" else ""),
            "nominal_shift_A": float(j41.nominal_shift(s, conv)),
            "canonical": bool(s["canonical"]),
        })
    return rows


# The ten rules of Table 2 of the submitted manuscript, and the design of the sweep that implements each one.
RULES = [
    ("R1", "3sig_rms", "detector", "RMS", dict(k=3.0), "thr_3sd"),
    ("R2", "3sig_kurt", "detector", "kurtosis", dict(k=3.0), ""),
    ("R3", "ma2sig", "detector", "five-record moving average of RMS", dict(k=2.0, ma_window=5), "ma5_2sd"),
    ("R4", "ma3sig", "detector", "five-record moving average of RMS", dict(k=3.0, ma_window=5), "ma5_3sd"),
    ("R5", "ma5sig", "detector", "five-record moving average of RMS", dict(k=5.0, ma_window=5), "ma5_5sd"),
    ("R6", "ewma", "control chart", "RMS", dict(lam=0.2, L=3.0), "ewma_l0.2_L3"),
    ("R7", "cusum", "control chart", "RMS", dict(kappa=0.5, H=5.0), "cusum_k0.5_h5"),
    ("R8", "pct02", "percentile", "none", dict(p=0.02), ""),
    ("R9", "pct05", "percentile", "none", dict(p=0.05), ""),
    ("R10", "pct10", "percentile", "none", dict(p=0.10), ""),
]


def rule_rows(designs: pd.DataFrame) -> list[dict]:
    rows = []
    for rid, name, fam, ind, p, design in RULES:
        if design:
            assert design in set(designs.id), f"{name}: design {design} not in the J26 grid"
        pct = fam == "percentile"
        rows.append({
            "set": "rule", "id": rid, "name": name, "kind": fam, "family": fam, "indicator": ind,
            "k": p.get("k", np.nan), "lambda": p.get("lam", np.nan), "L": p.get("L", np.nan),
            "kappa_sd": p.get("kappa", np.nan), "H_sd": p.get("H", np.nan), "ma_window": p.get("ma_window", np.nan),
            "percentile_p": p.get("p", np.nan),
            "confirmation_c": np.nan if pct else 3,
            "baseline_rule": "" if pct else BASELINE_RULE, "sd_convention": "" if pct else SD_CONVENTION,
            "chart_start": ("EWMA starts at the baseline mean" if name == "ewma" else
                            "CUSUM starts at 0" if name == "cusum" else ""),
            "onset_definition": ("i* = floor(n (1 - p))" if pct else
                                 "first record of c = 3 consecutive alarm records"),
            "implemented_by_design": design,
            "nominal_shift_A": (float(designs.set_index("id").loc[design, "nominal_shift_A"]) if design else np.nan),
        })
    return rows


def model_rows(j35, j39, j47) -> list[dict]:
    import torch  # noqa: F401  (J47 needs it; imported here so a missing torch fails with a clear message)
    rows = []
    feats = {"xjtu": "rms, std, p2p, crest, kurtosis, absmax, skew (j7_features_xjtu.csv)",
             "femto": "rms, kurtosis, crest, p2p, absmax (j7c_femto_features.csv)"}
    for name, est in j35.models(7).items():
        params = est.get_params(deep=True)
        keep = {k: v for k, v in params.items() if not k.startswith("steps") and not k.startswith("memory")
                and isinstance(v, (int, float, str, bool, type(None)))}
        rows.append({"model": name, "source": "J35.models", "family": "regressor (scikit-learn)",
                     "inputs": "rms only" if name == "lin:rms" else f"XJTU-SY: {feats['xjtu']}; FEMTO: {feats['femto']}",
                     "tuning": "none", "settings": json.dumps(keep, sort_keys=True, default=str)})
    rows.append({"model": "two-stage (exp)", "source": "J39", "family": "two-stage degradation model",
                 "inputs": "RMS divided by its baseline mean, records from the labeler's onset to t",
                 "tuning": "none",
                 "settings": json.dumps({"variant_used": "exp", "variants_available": list(j39.VARIANTS),
                                         "min_records_to_fit": j39.MIN_FIT,
                                         "trend": "log RMS linear in time, extrapolated to theta",
                                         "theta": "median over the other usable bearings of the last record's "
                                                  "scaled RMS",
                                         "before_onset": "y_hat = 1"}, sort_keys=True)})
    for name, cls in (("gru", j47.GRUHead), ("tcn", j47.TCNHead)):
        m = cls(7)
        arch = {}
        for sub_name, sub in m.named_modules():
            if sub.__class__.__name__ == "GRU":
                arch[sub_name] = {"type": "GRU", "hidden_size": sub.hidden_size, "num_layers": sub.num_layers}
            elif sub.__class__.__name__ == "Conv1d":
                arch[sub_name] = {"type": "Conv1d", "out_channels": sub.out_channels,
                                  "kernel_size": sub.kernel_size[0], "dilation": sub.dilation[0]}
            elif sub.__class__.__name__ == "Linear":
                arch[sub_name] = {"type": "Linear", "in": sub.in_features, "out": sub.out_features}
        n_par = int(sum(p.numel() for p in m.parameters()))
        rows.append({"model": name, "source": f"J47.{cls.__name__}", "family": "sequence model (PyTorch)",
                     "inputs": f"window of {j47.L} records of the J35 features, standardised on training bearings",
                     "tuning": "none (settings declared before any result, J47 docstring)",
                     "settings": json.dumps({"window": j47.L, "epochs": j47.EPOCHS, "batch": j47.BATCH,
                                             "learning_rate": j47.LR, "optimizer": "Adam", "loss": "MSE",
                                             "seeds": list(j47.SEEDS), "cell_value": "median over seeds",
                                             "parameters_xjtu": n_par, "architecture": arch}, sort_keys=True)})
    return rows


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j59_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J59  RULE, DESIGN AND MODEL CONFIGURATIONS  (R1.6)".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}")

    j26, j41 = load("j26", "J26_design_sweep.py"), load("j41", "J41_estimand_separation.py")
    designs = pd.DataFrame(design_rows(j26, j41))
    if len(designs) != 67:
        log(f"FAIL: J26.build_grid() returned {len(designs)} designs, not 67")
        return 1
    rules = pd.DataFrame(rule_rows(designs))
    conf = pd.concat([designs, rules], ignore_index=True, sort=False)

    log(f"\n{BAR}\nA  The 67 designs (J26.build_grid) and the ten rules\n{BAR}")
    for fam, g in designs.groupby("family", sort=False):
        log(f"  {fam:24s} {len(g):3d} designs; nominal shift {g.nominal_shift_A.min():.2f}-{g.nominal_shift_A.max():.2f} sd")
    log(f"  canonical designs (the rules of Table 2 inside the grid): "
        f"{', '.join(designs[designs.canonical].id)}")
    for _, r in rules.iterrows():
        link = f" = design {r.implemented_by_design}" if r.implemented_by_design else ""
        log(f"  {r.id:4s} {r['name']:10s} {r.family:13s} {r.indicator:34s}{link}")

    # J47 (torch) must be loaded before J35 (scikit-learn): on this Windows setup, importing torch after
    # sklearn.ensemble fails with WinError 1114 (c10.dll initialisation), while the reverse order works.
    j47 = load("j47", "J47_sequence_models.py")
    j35, j39 = load("j35", "J35_downstream_rul.py"), load("j39", "J39_two_stage_rul.py")
    models = pd.DataFrame(model_rows(j35, j39, j47))
    log(f"\n{BAR}\nB  Models of the RUL comparison (read back from the code)\n{BAR}")
    for _, r in models.iterrows():
        s = json.loads(r.settings)
        wanted = ("alpha", "n_neighbors", "n_estimators", "random_state", "window", "epochs", "batch",
                  "learning_rate", "seeds", "variant_used", "min_records_to_fit", "parameters_xjtu")
        # pipeline parameters carry the step name as a prefix (ridge__alpha); match on the last part
        short = {k.split("__")[-1]: v for k, v in s.items() if k.split("__")[-1] in wanted
                 and not (k.split("__")[-1] == "alpha" and r.model == "gbr:all")}
        log(f"  {r.model:16s} {r.source:16s} {json.dumps(short)}")

    conf.to_csv(T / "j59_rule_configurations.csv", index=False)
    models.to_csv(T / "j59_model_settings.csv", index=False)
    log(f"\nwrote results/tables/j59_rule_configurations.csv ({len(conf)} rows), "
        f"j59_model_settings.csv ({len(models)} rows)")
    log(f"finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
