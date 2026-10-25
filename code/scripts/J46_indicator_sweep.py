"""J46 -- does the response-size reading hold on indicators that are not time-domain statistics?

Section IV measures, over 67 rule designs, how much of the duration signature the size of change at a
design's alarm accounts for, and how much the family label accounts for. Every indicator behind that
measurement is a time-domain statistic of one vibration channel, and Section VII says the result is not
claimed beyond them. This repeats the whole measurement on each indicator separately, including the
spectral and envelope indicators J45 extracts, and reports every cell.

What is reported for each dataset and indicator:
  n_fired_designs   designs that fire on at least four trajectories, of the 67
  r2_size           R^2 of rho(i, n) on a quadratic in the log of the measured shift
  r2_family         R^2 of the same on the detector/chart label
  gap_rho           the detector-minus-chart difference in median rho, the statistic of Section S7
  direction_sd      how far the indicator moves by the end of life, in baseline standard deviations, median
                    over trajectories. It is negative for an indicator that FALLS as the bearing degrades,
                    which the ten rules cannot detect: every one of them fires on an upward crossing

An indicator on which the rules rarely fire is not a missing cell: it is a result, and is printed with
its firing count. The rules, the baseline of 20% and the confirmation count of 3 are J26's, imported
rather than re-implemented.

Inputs : results/tables/j7_features_xjtu.csv, j7c_femto_features.csv   the time-domain indicators
         results/tables/j45_indicators_{xjtu,femto}.csv                the spectral and envelope ones
         colab_bundle/xjtu/records.csv                                 the row order of the XJTU features
Outputs: results/tables/j46_indicator_sweep.csv   one row per dataset x indicator x design
         results/tables/j46_indicator_r2.csv      one row per dataset x indicator
         results/logs/j46_run.log

Usage: python scripts/J46_indicator_sweep.py
"""
from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BUNDLE = ROOT / "colab_bundle"

MIN_FIRED = 4
BAR = "=" * 78
KIND = {"rms": "amplitude", "std": "amplitude", "p2p": "amplitude", "absmax": "amplitude",
        "crest": "waveform shape", "kurtosis": "waveform shape", "skew": "waveform shape",
        "v_rms": "amplitude", "v_p2p": "amplitude", "v_absmax": "amplitude",
        "v_crest": "waveform shape", "v_kurtosis": "waveform shape",
        "spec_centroid": "spectrum", "band_power_fixed": "spectrum", "band_power_auto": "spectrum",
        "env_rms": "envelope", "env_kurtosis": "envelope"}

LOG: list[str] = []


def log(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(s)


def load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def direction(series: list[tuple[str, np.ndarray]], base_frac: float) -> float:
    """Median over trajectories of (mean of the last tenth of life - baseline mean) / baseline sd."""
    out = []
    for _, h in series:
        nb = max(5, int(len(h) * base_frac))
        tail = max(1, len(h) // 10)
        mu, sd = float(h[:nb].mean()), max(float(h[:nb].std()), 1e-12)
        out.append((float(h[-tail:].mean()) - mu) / sd)
    return float(np.median(out)) if out else np.nan


def r2(y: np.ndarray, X: np.ndarray) -> float:
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    sst = float((y - y.mean()) @ (y - y.mean()))
    return float("nan") if sst <= 0 else float(1 - (r @ r) / sst)


def series_by_indicator() -> dict[tuple[str, str], list[tuple[str, np.ndarray]]]:
    """Every (dataset, indicator) as a list of (trajectory, values in record order)."""
    out: dict[tuple[str, str], list[tuple[str, np.ndarray]]] = {}

    rec = pd.read_csv(BUNDLE / "xjtu" / "records.csv")
    f7 = pd.read_csv(T / "j7_features_xjtu.csv")
    if len(f7) != len(rec):
        raise SystemExit("the XJTU feature table and record index are not the same length")
    for col in ("rms", "std", "p2p", "absmax", "crest", "kurtosis", "skew"):
        d = pd.DataFrame({"trajectory": rec.trajectory_id.to_numpy(), "v": f7[col].to_numpy(float)})
        out[("xjtu", col)] = [(t, g.v.to_numpy(float)) for t, g in d.groupby("trajectory", sort=False)]

    fe = pd.read_csv(T / "j7c_femto_features.csv")
    for col in ("v_rms", "v_kurtosis", "v_crest", "v_p2p", "v_absmax"):
        out[("femto", col)] = [(t, g[col].to_numpy(float)) for t, g in fe.groupby("bearing", sort=False)]

    for ds in ("xjtu", "femto"):
        p = T / f"j45_indicators_{ds}.csv"
        if not p.exists():
            log(f"  !! {p.name} is missing: run J45 first")
            continue
        d = pd.read_csv(p)
        for col in ("spec_centroid", "band_power_fixed", "band_power_auto", "env_rms", "env_kurtosis"):
            out[(ds, col)] = [(t, g[col].to_numpy(float)) for t, g in d.groupby("trajectory", sort=False)]
    return out


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    log(BAR)
    log("J46  THE SWEEP, INDICATOR BY INDICATOR".center(78))
    log("is the response-size reading a property of the rules, or of tracking amplitude?".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S}")

    j26 = load("j26", "J26_design_sweep.py")
    grid = j26.build_grid()
    data = series_by_indicator()
    log(f"{len(grid)} designs x {len(data)} dataset-indicator pairs")

    rows, summary = [], []
    for (ds, ind), series in data.items():
        for spec in grid:
            idx, lens, shifts = [], [], []
            for t, h in series:
                i = j26.onset(h, spec["kind"], spec["p"])
                lens.append(len(h))
                if i is None:
                    idx.append(np.nan)
                    continue
                idx.append(i)
                nb = max(5, int(len(h) * j26.BASE_FRAC))
                mu, sd = float(h[:nb].mean()), max(float(h[:nb].std()), 1e-12)
                shifts.append((float(h[i:i + j26.CONFIRM].mean()) - mu) / sd)
            a, b = np.asarray(idx, float), np.asarray(lens, float)
            ok = np.isfinite(a)
            rows.append({"dataset": ds, "indicator": ind, "kind": KIND.get(ind, "?"),
                         "design": spec["design"], "family": spec["family"],
                         "n_traj": len(series), "n_fired": int(ok.sum()),
                         "rho_index_length": float(spearmanr(b[ok], a[ok])[0]) if ok.sum() >= MIN_FIRED else np.nan,
                         "median_shift_sd": float(np.median(shifts)) if shifts else np.nan})

        d = pd.DataFrame([r for r in rows if r["dataset"] == ds and r["indicator"] == ind])
        use = d[d.rho_index_length.notna() & d.median_shift_sd.notna()]
        n_use = len(use)
        if n_use >= 10 and use.family.nunique() > 1:
            y = use.rho_index_length.to_numpy(float)
            lx = np.log(np.clip(use.median_shift_sd.to_numpy(float), 1e-3, None))
            det = (use.family == "detector").to_numpy(float)
            r2s = r2(y, np.column_stack([np.ones(n_use), lx, lx ** 2]))
            r2f = r2(y, np.column_stack([np.ones(n_use), det]))
            gap = float(use[use.family == "detector"].rho_index_length.median()
                        - use[use.family != "detector"].rho_index_length.median())
        else:
            r2s = r2f = gap = np.nan
        dirn = direction(series, j26.BASE_FRAC)
        summary.append({"dataset": ds, "indicator": ind, "kind": KIND.get(ind, "?"),
                        "n_designs_fitted": n_use, "median_fired_trajectories": float(d.n_fired.median()),
                        "n_trajectories": int(d.n_traj.iloc[0]), "direction_sd": dirn,
                        "r2_size": r2s, "r2_family": r2f, "gap_rho": gap})
        log(f"  {ds:6} {ind:17} {KIND.get(ind, '?'):15} moves {dirn:+7.2f} sd   fitted {n_use:3}/67   "
            f"median fired {d.n_fired.median():5.1f}/{int(d.n_traj.iloc[0])}   "
            + (f"R2 size {r2s:.3f}   family {r2f:.3f}   gap {gap:+.3f}" if np.isfinite(r2s)
               else ("not fitted: the rules fire on too few" + (" (it falls, and every rule looks up)"
                                                               if dirn < 0 else ""))))

    pd.DataFrame(rows).to_csv(T / "j46_indicator_sweep.csv", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(T / "j46_indicator_r2.csv", index=False)

    log(f"\n{BAR}\nwhat the reading looks like by kind of indicator\n{BAR}")
    for ds in ("xjtu", "femto"):
        g = s[(s.dataset == ds) & s.r2_size.notna()]
        if not len(g):
            continue
        log(f"\n  {ds}")
        log(f"    indicators that rise: {int((s[(s.dataset == ds)].direction_sd > 0).sum())} of "
            f"{len(s[s.dataset == ds])}; the ten rules can only fire on those")
        for kind, gg in g.groupby("kind"):
            log(f"    {kind:15} {len(gg)} indicators   median R2 size {gg.r2_size.median():.3f}   "
                f"family {gg.r2_family.median():.3f}   size larger on {int((gg.r2_size > gg.r2_family).sum())}/{len(gg)}")
        log(f"    size explains more than family on {int((g.r2_size > g.r2_family).sum())} of {len(g)} indicators")

    log(f"\nwrote j46_indicator_sweep.csv, j46_indicator_r2.csv; finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    (LOGS / "j46_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
