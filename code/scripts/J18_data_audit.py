"""J18 -- the data-quality audit behind Appendix A, recomputed from the raw public archives.

Every defect listed in the appendix is measured here from the files as distributed, so that none of them rests on a
note in the project README.

  A1  IMS: Readme file counts and end dates against the files on disk (set 3 in particular)
  A2  IMS: growth of RMS on all 12 bearings, failed and not failed, and where set 3's growth happens
  A3  XJTU-SY: the stopping criterion (max amplitude > 10 x normal-stage max) checked per bearing on raw files
  A4  FEMTO: delimiters per file type and bearing, temperature files present, temperature ranges, file lengths

Outputs
    results/tables/j21_ims_files.csv
    results/tables/j21_ims_growth.csv
    results/tables/j21_xjtu_amplitude.csv
    results/tables/j21_femto_files.csv
    results/logs/j18_run.log

Usage
    python scripts/J18_data_audit.py
"""
from __future__ import annotations

import sys
from datetime import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
DATA = Path(os.environ.get("BEARING_DATASETS", "datasets"))
IMS_DIRS = {"1st_test": DATA / "IMS/IMS/1st_test", "2nd_test": DATA / "IMS/IMS/2nd_test", "3rd_test": DATA / "IMS/IMS/4th_test/txt"}
# From "Readme Document for IMS Bearing Data.pdf", page 2, as distributed with the archive
IMS_README = {"1st_test": (2156, "2003.11.25.23.39.56"), "2nd_test": (984, "2004.02.19.06.22.39"), "3rd_test": (4448, "2004.04.04.19.01.57")}
IMS_FAILED = {"IMS_1st_test_B3", "IMS_1st_test_B4", "IMS_2nd_test_B1", "IMS_3rd_test_B3"}
XJTU = {"1": DATA / "XJTU-SY/XJTU-SY_Bearing_Datasets/35Hz12kN", "2": DATA / "XJTU-SY/XJTU-SY_Bearing_Datasets/37.5Hz11kN",
        "3": DATA / "XJTU-SY/XJTU-SY_Bearing_Datasets/40Hz10kN"}
FEMTO = DATA / "FEMTO/raw"
BASE_LO, BASE_HI = 0.10, 0.20   # the post-warm-up window used for growth ratios, as in J7c
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    _fh.write(m + "\n")


def growth(v: np.ndarray) -> float:
    """Median of the last three records over the median of the 10-20% of life window (J7c definition)."""
    lf = (np.arange(len(v)) + 1) / len(v)
    win = (lf >= BASE_LO) & (lf < BASE_HI)
    if win.sum() < 3:
        win = np.zeros(len(v), bool)
        win[:max(3, len(v) // 10)] = True
    return float(np.median(v[-3:]) / np.median(v[win]))


def audit_ims() -> None:
    log("A1  IMS files on disk against the Readme")
    rows = []
    for test, d in IMS_DIRS.items():
        names = sorted(p.name for p in d.iterdir() if p.is_file() and p.name[:4].isdigit())
        n_readme, end_readme = IMS_README[test]
        rows.append({"test": test, "files_readme": n_readme, "files_on_disk": len(names), "first_file": names[0],
                     "last_file_on_disk": names[-1], "readme_end": end_readme,
                     "files_after_readme_end": sum(n > end_readme for n in names)})
        log(f"    {test}: Readme {n_readme} files to {end_readme}; disk {len(names)} files {names[0]} .. {names[-1]}, "
            f"{rows[-1]['files_after_readme_end']} after the Readme end")
    pd.DataFrame(rows).to_csv(TABLES / "j21_ims_files.csv", index=False)

    log("\nA2  IMS RMS growth on all 12 bearings (end / 10-20% of life)")
    rec = pd.read_csv(ROOT / "colab_bundle/ims/records.csv")
    feat = pd.read_csv(TABLES / "j7_features_ims.csv")
    rec["rms"] = feat["rms"].to_numpy()
    readme3 = IMS_README["3rd_test"][0]
    rows = []
    for t, g in rec.groupby("trajectory_id", sort=True):
        v = g.sort_values("record_index").rms.to_numpy()
        row = {"trajectory": t, "failed_documented": t in IMS_FAILED, "n_records": len(v), "rms_growth": growth(v)}
        if t.startswith("IMS_3rd"):
            # records are every third file (Section 3), so record r is file 3r
            r_end = (readme3 - 1) // 3
            described = v[: r_end + 1]
            row["records_within_readme"] = r_end + 1
            row["max_rms_within_readme_over_baseline"] = float(described.max() / np.median(v[int(BASE_LO * len(v)):int(BASE_HI * len(v))]))
            row["max_rms_after_readme_over_baseline"] = float(v[r_end + 1:].max() / np.median(v[int(BASE_LO * len(v)):int(BASE_HI * len(v))]))
        rows.append(row)
        log(f"    {t:18s} failed={t in IMS_FAILED!s:5s} n={len(v):5d} growth {row['rms_growth']:.2f}x"
            + (f"  | max RMS/baseline within Readme {row['max_rms_within_readme_over_baseline']:.2f}x, after {row['max_rms_after_readme_over_baseline']:.2f}x"
               if "max_rms_within_readme_over_baseline" in row else ""))
    pd.DataFrame(rows).to_csv(TABLES / "j21_ims_growth.csv", index=False)


def audit_xjtu() -> None:
    log("\nA3  XJTU-SY stopping criterion on raw files: max |amplitude| of either axis")
    rows = []
    for cond, d in XJTU.items():
        for b in sorted(p for p in d.iterdir() if p.is_dir()):
            files = sorted(b.glob("*.csv"), key=lambda p: int(p.stem))
            amax = np.empty((len(files), 2))
            rms = np.empty(len(files))
            for k, f in enumerate(files):
                a = pd.read_csv(f, engine="c").to_numpy(float)
                amax[k] = np.abs(a).max(axis=0)
                rms[k] = float(np.sqrt((a ** 2).sum(axis=1).mean()))
            n = len(files)
            normal = amax[: max(1, int(BASE_HI * n))]      # first 20% of life taken as the normal stage
            ratio = amax[-1] / normal.max(axis=0)
            rows.append({"bearing": b.name, "condition": cond, "n_files": n,
                         "normal_stage_max_h": normal[:, 0].max(), "normal_stage_max_v": normal[:, 1].max(),
                         "final_max_h": amax[-1, 0], "final_max_v": amax[-1, 1],
                         "final_over_normal_h": ratio[0], "final_over_normal_v": ratio[1],
                         "final_over_normal_max_axis": ratio.max(),
                         "life_max_over_normal_max_axis": float((amax.max(axis=0) / normal.max(axis=0)).max()),
                         "rms_growth": growth(rms)})
            log(f"    {b.name:11s} n={n:5d}  final max / normal max: H {ratio[0]:5.2f}x  V {ratio[1]:5.2f}x   "
                f"RMS growth {rows[-1]['rms_growth']:.2f}x")
    pd.DataFrame(rows).to_csv(TABLES / "j21_xjtu_amplitude.csv", index=False)


def audit_femto() -> None:
    log("\nA4  FEMTO files: delimiters, temperature files, temperature range, file lengths")
    rows = []
    for split in ("train", "valid"):
        base = FEMTO / split
        if not base.exists():
            continue
        for b in sorted(base.rglob("Bearing*")):
            if not b.is_dir():
                continue
            accs, tmps = sorted(b.glob("acc_*.csv")), sorted(b.glob("temp_*.csv"))

            def sep(p):
                line = p.open().readline()
                return "," if line.count(",") > line.count(";") else ";"
            acc_seps = {sep(p) for p in accs}
            tmp_seps = {sep(p) for p in tmps}
            tmax, tmin, lens = np.nan, np.nan, []
            if tmps:
                vals = []
                for p in tmps:
                    s = sep(p)
                    t = pd.read_csv(p, header=None, sep=s, engine="c").to_numpy()
                    lens.append(len(t))
                    vals.append(t[:, 4].astype(float))
                allv = np.concatenate(vals)
                tmax, tmin = float(allv.max()), float(allv.min())
            rows.append({"split": split, "bearing": b.name, "path": str(b.relative_to(FEMTO)), "n_acc": len(accs),
                         "n_temp": len(tmps), "acc_delimiters": "".join(sorted(acc_seps)),
                         "temp_delimiters": "".join(sorted(tmp_seps)), "temp_min_C": tmin, "temp_max_C": tmax,
                         "temp_file_rows_min": min(lens) if lens else np.nan, "temp_file_rows_max": max(lens) if lens else np.nan})
            log(f"    {split:5s} {b.name:12s} acc {len(accs):5d} [{rows[-1]['acc_delimiters']}]  temp {len(tmps):4d} "
                f"[{rows[-1]['temp_delimiters']}]  T {tmin:7.1f}..{tmax:7.1f} C  rows/temp file "
                f"{rows[-1]['temp_file_rows_min']}..{rows[-1]['temp_file_rows_max']}")
    pd.DataFrame(rows).to_csv(TABLES / "j21_femto_files.csv", index=False)


def main() -> int:
    global _fh
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j18_run.log", "w", encoding="utf-8")
    log(f"J18 data audit, started {datetime.now():%Y-%m-%d %H:%M:%S}")
    audit_ims()
    audit_femto()
    audit_xjtu()
    log("\nwrote j21_ims_files, j21_ims_growth, j21_femto_files, j21_xjtu_amplitude")
    _fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
