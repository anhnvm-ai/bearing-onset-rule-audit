"""J7c -- the one circularity test the other three datasets cannot run.

Why FEMTO and nothing else
--------------------------
J7 v2 killed C1 as originally framed. Comparing datasets with different stopping criteria
cannot separate circularity from plain physics, because every run-to-failure trajectory
degrades monotonically: "late in life" correlates with "high relative amplitude" whatever
stopped the test. With baseline-relative features, NSK, XJTU and IMS all looked alike.

FEMTO breaks that confound because the comparison moves INSIDE a single trajectory, between
two channels:

    stopping criterion = vibration amplitude > 20 g   -> vibration IS the criterion channel
    a PT-100 thermal channel is recorded alongside    -> temperature is NOT

Monotone degradation acts on both channels. Only circularity acts on vibration alone. So
the vibration-minus-temperature margin, measured within the same bearings under the same
folds, isolates the part that plain physics does not explain.

PRE-REGISTERED PREDICTION (written before running this script, 2026-09-11)
    NSK's criterion involves temperature, and J7 v2 already measured that at the 10%-of-life
    horizon temperature beats vibration there: 0.9380 vs 0.7611.
    FEMTO's criterion is vibration. If C1 has any content, the ordering must REVERSE here:
    vibration should beat temperature by a clear margin.
      reversal observed      -> circularity is real and measurable, C1 survives re-scoped
      both channels similar  -> both merely track monotone degradation, C1 is dropped

THE CHECK THAT COMES FIRST
    If temperature simply does not rise on this rig, vibration wins for a trivial physical
    reason and the test says nothing about circularity. Step 3 therefore reports the
    degradation ratio of BOTH channels per bearing before any classification is done. Read
    it before reading the verdict.

Data handling
    13 bearings carry both channels over a complete life (Training_set plus Validation_Set;
    Test_set is truncated for the RUL challenge and is not used).
    acc  files: 2560 rows @ 25.6 kHz = 0.1 s, one every 10 s, columns h,m,s,us,horiz,vert
    temp files:  600 rows @ 10 Hz    = 60 s,  one every 60 s, columns h,m,s,idx,celsius
    The two grids differ by a factor of ~6, so the analysis runs on the TEMPERATURE grid and
    each temperature record is paired with the acc record nearest in time. Step 2 reports the
    worst pairing error in seconds; if that is large the pairing, not the physics, is at fault.

Usage
    python scripts/J7c_femto_channel_test.py
    python scripts/J7c_femto_channel_test.py --check
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import metrics_from_counts  # noqa: E402
from src.threshold import fit_threshold, predict_threshold  # noqa: E402

RAW = Path(os.environ.get("BEARING_DATASETS", "datasets")) / "FEMTO/raw"
TABLES = ROOT / "results" / "tables"
LIFE_PCTS = (0.02, 0.05, 0.10)
BAR = "=" * 78

COND = {"1": "1800 rpm / 4000 N", "2": "1650 rpm / 4200 N", "3": "1500 rpm / 5000 N"}
VIB = ["v_rms", "v_kurtosis", "v_crest", "v_p2p", "v_absmax"]
TMP = ["t_median", "t_mean", "t_slope"]

# --- data-quality decisions forced by FEMTO's thermal channel, stated up front ---------
# 1. Two bearings carry a physically impossible thermal signal and are EXCLUDED:
#    Bearing2_6 (median-of-medians 194.9 C, non-monotone, spikes to 818 C) and
#    Bearing2_7 (127.8 C, starts at 177 C and ENDS at 83 C, only 38 records).
#    Rule, applied before any modelling: exclude a bearing if the median of its per-record
#    median temperatures exceeds T_CORRUPT. The remaining 11 bearings peak at 102-164 C,
#    which is plausible for an accelerated rig.
# 2. Per-record temperature is summarised by the MEDIAN, not the mean, so that out-of-range
#    PT-100 spikes cannot move the feature.
# 3. The baseline for relative features is taken from 10-20% of life, NOT 0-10%. The rig
#    warms up over the first minutes (Bearing1_5 runs 69 -> 101 C before its signal flattens),
#    so a 0-10% baseline would charge warm-up to degradation. The same window is used for
#    BOTH channels, so neither is favoured.
T_CORRUPT = 150.0
BASE_LO, BASE_HI = 0.10, 0.20


def head(step: str, title: str) -> None:
    print(f"\n{BAR}\nSTEP {step}  {title}\n{BAR}", flush=True)


def read_femto(path: Path, n_cols: int) -> np.ndarray:
    """Read one FEMTO csv, sniffing the delimiter per file.

    FEMTO ships MIXED delimiters, and mixed within a single bearing folder: on 12 of the 13
    bearings used here the acc_*.csv are comma-separated while the temp_*.csv are
    semicolon-separated. Bearing1_1 is comma throughout, Bearing1_4 is semicolon throughout.
    Hardcoding either one parses part of the dataset into a single string column and yields
    silent garbage rather than an error, so every file is sniffed and the column count
    checked.
    """
    with path.open() as fh:
        first = fh.readline()
    sep = "," if first.count(",") > first.count(";") else ";"
    a = pd.read_csv(path, header=None, sep=sep, engine="c").to_numpy()
    if a.shape[1] != n_cols:
        raise ValueError(f"{path}: parsed {a.shape[1]} columns with '{sep}', expected {n_cols}")
    return a


def secs(row) -> float:
    return float(row[0]) * 3600 + float(row[1]) * 60 + float(row[2])


def bearings_with_both() -> dict[str, Path]:
    """Longest available copy of each bearing that has both channels."""
    best: dict[str, tuple[int, Path]] = {}
    for d in RAW.rglob("Bearing*"):
        if not d.is_dir() or not list(d.glob("temp_*.csv")):
            continue
        n = len(list(d.glob("acc_*.csv")))
        if d.name not in best or n > best[d.name][0]:
            best[d.name] = (n, d)
    return {k: v[1] for k, v in sorted(best.items())}


def build_features(paths: dict[str, Path]) -> tuple[pd.DataFrame, float]:
    rows, worst_dt = [], 0.0
    for b, d in paths.items():
        accs = sorted(d.glob("acc_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
        tmps = sorted(d.glob("temp_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
        n_t, n_a = len(tmps), len(accs)
        for k, tp in enumerate(tmps):
            T = read_femto(tp, 5)
            j = min(n_a - 1, int(round((k + 0.5) * n_a / n_t)))
            A = read_femto(accs[j], 6)
            worst_dt = max(worst_dt, abs(secs(A[0]) - secs(T[0])))
            mag = np.sqrt(A[:, 4].astype(float) ** 2 + A[:, 5].astype(float) ** 2)
            r = float(np.sqrt((mag ** 2).mean()))
            z = (mag - mag.mean()) / max(mag.std(), 1e-12)
            temp = T[:, 4].astype(float)
            rows.append({
                "bearing": b, "condition": b[7], "record_index": k,
                "life_fraction": (k + 1) / n_t,
                "v_rms": r, "v_kurtosis": float((z ** 4).mean()),
                "v_crest": float(mag.max() / max(r, 1e-12)),
                "v_p2p": float(mag.max() - mag.min()), "v_absmax": float(mag.max()),
                "t_median": float(np.median(temp)), "t_mean": float(temp.mean()),
                "t_slope": float(np.polyfit(np.arange(len(temp)), temp, 1)[0]),
            })
        print(f"    {b:12s} cond {b[7]}  {n_t:4d} paired records  (acc {n_a})", flush=True)
    return pd.DataFrame(rows), worst_dt


def relativise(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Divide by each bearing's own post-warm-up baseline (10-20% of life), both channels."""
    out = df.copy()
    for b in df.bearing.unique():
        m = (df.bearing == b).to_numpy()
        lf = df.loc[m, "life_fraction"].to_numpy()
        win = (lf >= BASE_LO) & (lf < BASE_HI)
        if win.sum() < 3:
            win = np.zeros_like(lf, bool); win[:max(3, len(lf) // 10)] = True
        for c in cols:
            v = df.loc[m, c].to_numpy(float)
            base = np.median(v[win])
            out.loc[m, c] = v / (base if abs(base) > 1e-9 else 1.0)
    return out


def drop_corrupt_thermal(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    bad = [b for b in df.bearing.unique()
           if np.median(df.loc[df.bearing == b, "t_median"].to_numpy(float)) > T_CORRUPT]
    return df[~df.bearing.isin(bad)].reset_index(drop=True), bad


def oof(x, y, folds) -> dict:
    p = np.zeros_like(y)
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        if y[tr].sum() in (0, len(tr)):
            continue
        p[te] = predict_threshold(x[te], fit_threshold(x[tr], y[tr], "balanced_accuracy"))
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    m = metrics_from_counts(tp, fp, fn, tn)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "macro_f1": m["macro_f1"], "mcc": m["mcc"]}


def main(check_only: bool) -> int:
    print(BAR)
    print("J7c  FEMTO -- CRITERION CHANNEL vs NON-CRITERION CHANNEL".center(78))
    print("the comparison lives INSIDE one trajectory, so monotone decay cancels".center(78))
    print(BAR)

    head("1", "Bearings carrying both channels over a complete life")
    paths = bearings_with_both()
    print(f"  stopping criterion: vibration amplitude > 20 g  ->  criterion channel = VIBRATION")
    print(f"  non-criterion channel recorded alongside        ->  PT-100 TEMPERATURE\n")
    for b, d in paths.items():
        print(f"    {b:12s} cond {b[7]} ({COND.get(b[7], '?')})   {d.relative_to(RAW)}")
    print(f"\n  {len(paths)} bearings, conditions {sorted({b[7] for b in paths})}")
    if check_only:
        print("\n  --check: inputs OK.")
        return 0

    head("2", "Pairing the 10 s vibration grid onto the 60 s temperature grid")
    cache = TABLES / "j7c_femto_features.csv"
    if cache.exists():
        print(f"    reusing {cache.name}")
        df = pd.read_csv(cache)
        worst = float("nan")
    else:
        df, worst = build_features(paths)
        df.to_csv(cache, index=False)
        print(f"\n    worst acc-vs-temp timestamp mismatch: {worst:.1f} s "
              f"({'fine, well inside the 60 s temperature window' if worst < 60 else 'TOO LARGE -- pairing is at fault'})")
    print(f"    {len(df)} paired records over {df.bearing.nunique()} bearings")
    df, bad = drop_corrupt_thermal(df)
    if bad:
        print(f"\n    EXCLUDED for a physically impossible thermal channel "
              f"(median-of-medians > {T_CORRUPT:.0f} C): {bad}")
        print(f"    -> {len(df)} records over {df.bearing.nunique()} bearings carry the analysis")

    head("3", "DOES TEMPERATURE EVEN RISE?  read this before the verdict")
    print("    If the thermal channel is flat, vibration wins for a trivial physical reason")
    print("    and this test says nothing about circularity.\n")
    print("    baseline = 10-20% of life (post warm-up), the same window for both channels\n")
    print(f"    {'bearing':12s} {'n':>5s} {'D_vib':>7s} {'D_temp':>7s}   {'t_base':>7s} {'t_end':>7s}")
    Ds = []
    for b in df.bearing.unique():
        g = df[df.bearing == b]
        lf = g.life_fraction.to_numpy()
        win = (lf >= BASE_LO) & (lf < BASE_HI)
        if win.sum() < 3:
            win = np.zeros_like(lf, bool); win[:max(3, len(lf) // 10)] = True
        dv = np.median(g.v_rms.to_numpy()[-3:]) / np.median(g.v_rms.to_numpy()[win])
        tb, tn_ = np.median(g.t_median.to_numpy()[win]), np.median(g.t_median.to_numpy()[-3:])
        dt = tn_ / tb
        Ds.append({"bearing": b, "D_vib": dv, "D_temp": dt, "t_base": tb, "t_end": tn_})
        print(f"    {b:12s} {len(g):>5d} {dv:>6.1f}x {dt:>6.2f}x   {tb:>7.1f} {tn_:>7.1f}")
    Ds = pd.DataFrame(Ds)
    print(f"\n    median D_vib = {Ds.D_vib.median():.1f}x     median D_temp = {Ds.D_temp.median():.2f}x")
    print(f"    temperature range across bearings: {Ds.t_base.min():.0f}-{Ds.t_end.max():.0f} C")
    if Ds.D_temp.median() < 1.05:
        print("\n    WARNING: temperature is essentially FLAT on this rig. Any vibration advantage")
        print("    below is explained by physics, not by circularity. Read the verdict accordingly.")

    head("4", "Criterion channel vs non-criterion channel, leave-one-bearing-out")
    folds = [np.where(df.bearing.to_numpy() == b)[0] for b in df.bearing.unique()]
    rel = relativise(df, VIB + TMP)
    rows = []
    print(f"    {'life%':>6s} {'features':>9s} {'best VIBRATION':>26s} {'best TEMPERATURE':>28s} {'margin':>8s}")
    for pct in LIFE_PCTS:
        y = (df.life_fraction.to_numpy() > 1 - pct).astype(int)
        for tag, F in (("absolute", df), ("relative", rel)):
            res = {c: oof(F[c].to_numpy(float), y, folds) for c in VIB + TMP}
            for c, r in res.items():
                rows.append({"life_pct": pct, "featureset": tag, "channel":
                             "vibration" if c in VIB else "temperature", "feature": c, **r})
            bv = max(VIB, key=lambda c: res[c]["macro_f1"])
            bt = max(TMP, key=lambda c: res[c]["macro_f1"])
            mv, mt = res[bv]["macro_f1"], res[bt]["macro_f1"]
            print(f"    {pct:>6.0%} {tag:>9s} {bv + ' ' + format(mv, '.4f'):>26s} "
                  f"{bt + ' ' + format(mt, '.4f'):>28s} {mv - mt:>+8.4f}")
    res = pd.DataFrame(rows)

    head("5", "Verdict on the pre-registered prediction")
    r = res[res.featureset == "relative"]
    marg = [r[(r.life_pct == p) & (r.channel == "vibration")].macro_f1.max()
            - r[(r.life_pct == p) & (r.channel == "temperature")].macro_f1.max()
            for p in LIFE_PCTS]
    m = float(np.median(marg))

    # NSK's margin is recomputed the SAME way -- median over the same three horizons --
    # from J7's own output. An earlier version of this script hardcoded NSK's 10% value
    # alone, which was the single horizon out of three where temperature led; that is
    # cherry-picking and it manufactured a reversal that does not exist.
    nsk = np.nan
    j7 = TABLES / "j7_circularity.csv"
    if j7.exists():
        d = pd.read_csv(j7)
        d = d[(d.dataset == "nsk") & (d.method.str.startswith("thr:")) & (d.featureset == "relative")]
        if not d.empty:
            d = d.assign(ch=np.where(d.method.str.contains("temp"), "temperature", "vibration"))
            nsk = float(np.median([
                d[(d.life_pct == p) & (d.ch == "vibration")].macro_f1.max()
                - d[(d.life_pct == p) & (d.ch == "temperature")].macro_f1.max()
                for p in sorted(d.life_pct.unique())]))

    print(f"  vibration minus temperature, median over horizons {list(LIFE_PCTS)}")
    print(f"    FEMTO  criterion = VIBRATION     margin {m:+.4f}   per-horizon "
          f"{[round(x, 4) for x in marg]}")
    print(f"    NSK    criterion = TEMPERATURE   margin {nsk:+.4f}   (recomputed from j7_circularity.csv)")
    print()
    if not np.isfinite(nsk):
        print("  >>> CANNOT JUDGE: run scripts/J7_circularity.py first to get the NSK margin.")
    elif m > 0.05 and nsk < -0.05:
        print("  >>> ORDERING REVERSES with the criterion channel. C1 survives, re-scoped.")
    elif np.sign(m) == np.sign(nsk):
        print("  >>> NO REVERSAL. Vibration leads on BOTH datasets, although the criterion")
        print("      channel differs between them. The channel ordering is therefore governed")
        print("      by sensor physics -- vibration is simply the better bearing degradation")
        print("      indicator -- not by which channel defined the label. C1 has NO measurable")
        print("      content: report it as a negative result and drop it from the contributions.")
    else:
        print("  >>> AMBIGUOUS: margins differ in size but not consistently with the criterion.")
    if Ds.D_temp.median() < 1.25:
        print(f"\n  CONTEXT: median D_temp is only {Ds.D_temp.median():.2f}x against D_vib "
              f"{Ds.D_vib.median():.1f}x.")
        print("  The thermal channel barely moves, which on its own explains why vibration leads.")

    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "j7c_channel_test.csv", index=False)
    Ds.to_csv(TABLES / "j7c_degradation_ratios.csv", index=False)
    print(f"\n{BAR}\n  WROTE  results/tables/j7c_channel_test.csv  ({len(res)} rows)"
          f"\n         results/tables/j7c_degradation_ratios.csv ({len(Ds)} rows)\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main("--check" in sys.argv))
