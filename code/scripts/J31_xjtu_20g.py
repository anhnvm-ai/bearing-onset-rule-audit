"""J31 -- which stated end-of-test criterion do the XJTU-SY recordings satisfy?

Run from bearing_multimodal_atc/:
    python scripts/J31_xjtu_20g.py

Why this exists
---------------
The two papers that introduce XJTU-SY state different end-of-test criteria. The dataset tutorial (Lei et al. 2019)
says a test ran until the maximum amplitude of either axis exceeded ten times its maximum in the normal stage;
Appendix A.7 and J18 already show that this depends on an undefined normal stage. The hybrid-prognostics paper by
the same group (Wang et al., IEEE Trans. Reliability 69(1), 2020, p. 406) says instead that "the accelerated
degradation tests of bearings are stopped when the amplitude of the vibration signal is higher than 20 g".

This script reads the raw files, exactly as J18 does, and records for every bearing the maximum absolute amplitude
over both axes of every file, whether the recording ends above 20 g, and the first file at which it exceeds 20 g.
If the tests stopped at the first crossing, that file would be the last one or close to it. Nothing existing is
overwritten; J18's j21_xjtu_amplitude.csv is left as it is.

Output
    results/tables/j31_xjtu_20g.csv
    results/logs/j31_run.log
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES, LOGS = ROOT / "results" / "tables", ROOT / "results" / "logs"
DATA = Path(os.environ.get("BEARING_DATASETS", "datasets"))
XJTU = {"1": DATA / "XJTU-SY/XJTU-SY_Bearing_Datasets/35Hz12kN", "2": DATA / "XJTU-SY/XJTU-SY_Bearing_Datasets/37.5Hz11kN",
        "3": DATA / "XJTU-SY/XJTU-SY_Bearing_Datasets/40Hz10kN"}
G = 20.0


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    fh = open(LOGS / "j31_run.log", "w", encoding="utf-8")

    def log(m=""):
        print(m, flush=True)
        fh.write(m + "\n")

    log("J31  XJTU-SY: the 20 g end-of-test criterion of Wang et al. (2020) against the raw files")
    rows = []
    for cond, d in XJTU.items():
        for b in sorted(p for p in d.iterdir() if p.is_dir()):
            files = sorted(b.glob("*.csv"), key=lambda p: int(p.stem))
            amax = np.array([np.abs(pd.read_csv(f, engine="c").to_numpy(float)).max() for f in files])
            n = len(files)
            over = np.flatnonzero(amax > G)
            first = int(over[0]) + 1 if len(over) else None          # 1-based file number, as the files are named
            rows.append({"bearing": b.name, "condition": cond, "n_files": n, "final_max_both_axes": float(amax[-1]),
                         "ends_above_20g": bool(amax[-1] > G), "first_file_over_20g": first,
                         "files_after_first_crossing": (n - first) if first else None,
                         "first_crossing_fraction": (first / n) if first else None})
            r = rows[-1]
            log(f"  {b.name:11s} n={n:5d}  final max {r['final_max_both_axes']:6.2f} g  first > 20 g at file "
                f"{first} ({r['first_crossing_fraction']:.3f} of life), {r['files_after_first_crossing']} files after")
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "j31_xjtu_20g.csv", index=False)
    late = out[out.files_after_first_crossing <= 1]
    log(f"\nends above 20 g: {int(out.ends_above_20g.sum())} of {len(out)}")
    log(f"stopped within one file of the first 20 g crossing: {len(late)} of {len(out)}")
    log(f"first crossing before 0.9 of life: {int((out.first_crossing_fraction < 0.9).sum())} of {len(out)}")
    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
