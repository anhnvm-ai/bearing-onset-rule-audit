"""J30 -- the same named rule, applied by another study to the same bearings.

Run from bearing_multimodal_atc/:
    python scripts/J30_cross_study.py

Why this exists
---------------
Contribution 2 says that naming a rule does not identify the label set. Every piece of evidence for it so far comes
from inside this study: one implementation of each rule, applied across datasets, conditions and conventions. A
referee can answer that a second group applying "the 3 sigma method" would agree with this study.

Li et al. (RESS 259, 2025, 110923) publish, for all 15 XJTU-SY bearings (Table 16) and for PRONOSTIA/FEMTO
Bearing1_1-1_7 (Table 2), the onset of their CNN + "3/5" rule and of "the 3 sigma method". XJTU-SY records one
snapshot per minute and their table is in minutes, so on XJTU-SY both studies count the same records; the script
checks that the total lives agree bearing by bearing before comparing anything. FEMTO is reported but not used for
claims: their table counts 10-s snapshots and this study's FEMTO features sit on the coarser temperature grid, so the
two record axes differ (Appendix A).

Their paper does not state which indicator, reference window or confirmation count its 3 sigma method uses, which is
itself the point of ORD-4.

Inputs
    scripts/workflow_verification/sources/novelty/li2025stagedivision_tables.csv   Tables 2 and 16, transcribed
    results/tables/fig1_fpt_matrix.csv                                              this study's onsets
Outputs
    results/tables/j30_cross_study.csv       one row per bearing: both studies' onsets, fractions, differences
    results/tables/j30_cross_study_summary.csv
    results/logs/j30_run.log
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
SRC = ROOT / "scripts" / "workflow_verification" / "sources" / "novelty" / "li2025stagedivision_tables.csv"
BAR = "=" * 78
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j30_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J30  THE SAME NAMED RULE IN TWO STUDIES, ON THE SAME BEARINGS".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}")

    li = pd.read_csv(SRC)
    ft = pd.read_csv(T / "fig1_fpt_matrix.csv")
    rows, summ = [], []
    for ds in ("xjtu", "femto"):
        L = li[li.dataset == ds].copy()
        F = ft[ft.dataset == ds]
        n = F.drop_duplicates("trajectory").set_index("trajectory").n_records
        piv = F.pivot_table(index="trajectory", columns="rule", values="fpt_index", aggfunc="first")
        L = L[L.trajectory.isin(n.index)].copy()
        L["our_n"] = L.trajectory.map(n).astype(int)
        same_axis = bool((L.total_life == L.our_n).all())
        log(f"\n{ds}: {len(L)} bearings in both studies; record counts identical: {same_axis}")
        for r in ("3sig_rms", "ma3sig", "ma5sig", "ewma", "cusum"):
            L[f"our_{r}"] = L.trajectory.map(piv[r])
        L["li_3sigma_frac"] = L.onset_3sigma / L.total_life
        L["li_cnn_frac"] = L.ds_point_cnn_3of5 / L.total_life
        L["our_3sig_rms_frac"] = L.our_3sig_rms / L.our_n
        L["diff_frac"] = L.our_3sig_rms_frac - L.li_3sigma_frac
        L["same_record_axis"] = same_axis
        rows.append(L)
        ok = L.our_3sig_rms.notna()
        d = L.diff_frac[ok]
        pos_li = float((L.total_life - L.onset_3sigma)[ok].sum() / L.total_life[ok].sum())
        pos_our = float((L.our_n - L.our_3sig_rms)[ok].sum() / L.our_n[ok].sum())
        s = {"dataset": ds, "n_bearings": int(len(L)), "n_both_fire": int(ok.sum()), "same_record_axis": same_axis,
             "li_3sigma_median_frac": float(L.li_3sigma_frac.median()),
             "li_3sigma_min_frac": float(L.li_3sigma_frac.min()), "li_3sigma_max_frac": float(L.li_3sigma_frac.max()),
             "our_3sig_rms_median_frac": float(L.our_3sig_rms_frac[ok].median()),
             "our_3sig_rms_min_frac": float(L.our_3sig_rms_frac[ok].min()),
             "our_3sig_rms_max_frac": float(L.our_3sig_rms_frac[ok].max()),
             "abs_diff_median": float(d.abs().median()), "abs_diff_max": float(d.abs().max()),
             "n_apart_over_025": int((d.abs() > 0.25).sum()),
             "pos_rate_li_3sigma": pos_li, "pos_rate_our_3sig_rms": pos_our,
             "abs_diff_to_li_cnn_median": float((L.our_3sig_rms_frac - L.li_cnn_frac)[ok].abs().median()),
             "li_cnn_minus_3sigma_median": float((L.li_cnn_frac - L.li_3sigma_frac).median())}
        for lab, col, den in (("li_3sigma", "onset_3sigma", "total_life"), ("li_cnn", "ds_point_cnn_3of5", "total_life"),
                              ("our_ewma", "our_ewma", "our_n"), ("our_cusum", "our_cusum", "our_n")):
            m = L[col].notna()
            s[f"rho_{lab}"] = float(spearmanr(L[den][m], L[col][m])[0]) if m.sum() >= 4 else np.nan
        summ.append(s)
        log(f"  Li 3 sigma onset fraction   median {s['li_3sigma_median_frac']:.3f} "
            f"[{s['li_3sigma_min_frac']:.3f}, {s['li_3sigma_max_frac']:.3f}]")
        log(f"  our 3sig_rms onset fraction median {s['our_3sig_rms_median_frac']:.3f} "
            f"[{s['our_3sig_rms_min_frac']:.3f}, {s['our_3sig_rms_max_frac']:.3f}]")
        log(f"  |difference| median {s['abs_diff_median']:.3f}, max {s['abs_diff_max']:.3f}; "
            f"more than 0.25 of life apart on {s['n_apart_over_025']} of {s['n_both_fire']}")
        log(f"  pooled positive rate: Li 3 sigma {pos_li:.3f} | ours {pos_our:.3f}")
        log(f"  ours vs Li CNN + 3/5: |difference| median {s['abs_diff_to_li_cnn_median']:.3f}")
        log(f"  rho(onset record, n): Li 3 sigma {s['rho_li_3sigma']:+.3f} | Li CNN {s['rho_li_cnn']:+.3f} | "
            f"our EWMA {s['rho_our_ewma']:+.3f} | our CUSUM {s['rho_our_cusum']:+.3f}")
        if not same_axis:
            log("  NOTE different record axes -- reported, not used for any claim")

    pd.concat(rows).to_csv(T / "j30_cross_study.csv", index=False)
    pd.DataFrame(summ).to_csv(T / "j30_cross_study_summary.csv", index=False)
    log(f"\nwrote j30_cross_study.csv and j30_cross_study_summary.csv\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
