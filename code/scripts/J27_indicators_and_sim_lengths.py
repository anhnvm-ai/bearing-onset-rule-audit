"""E3 + E4 -- how far does the result generalise across health indicators, and what does it look like with no
confounds at all?

Run from bearing_multimodal_atc/:
    python scripts/J27_indicators_and_sim_lengths.py

E3  seven indicators instead of two
    Section 4.5 replicates the analysis on peak-to-peak. But RMS and peak-to-peak are both amplitude statistics of
    the same window, so a reader can answer that two indicators are one indicator. The feature tables already hold
    seven: rms, std, p2p, absmax are amplitude statistics; crest, kurtosis, skew are SHAPE statistics, which change
    when the waveform changes even if its level does not. Running the Section 4.2 measurement on all seven asks
    whether the family separation is a property of amplitude tracking or of onset rules in general.

    FEMTO deposits only five of the seven in j7c_femto_features.csv (v_rms, v_kurtosis, v_crest, v_p2p, v_absmax):
    std and skew are absent. That is stated in the output rather than quietly dropped.

E4  the duration scaling in a simulation, where there is nothing to confound it
    J24 already generates AR(1) series at the fifteen XJTU-SY lengths with a change of known size and position, but
    it pools over lengths and reports only the share firing before the change, the share never firing and the delay.
    Here the same generator is run with the length retained, so that rho(alarm index, n) -- the Section 4.2
    statistic -- can be computed on data where the only thing that differs between series is n. J24 is not modified
    and its outputs are not overwritten.

Outputs
    results/tables/j27_indicators.csv        one row per dataset x indicator x rule
    results/tables/j27_sim_by_length.csv     one row per shape x delta x u x rule
    results/logs/j27_run.log
"""
from __future__ import annotations

import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "colab_bundle"
T = ROOT / "results" / "tables"
LOGS = ROOT / "results" / "logs"
BAR = "=" * 78
BASE_FRAC, CONFIRM = 0.20, 3
R_SIM = int(__import__("os").environ.get("R_SIM", "400"))  # lower it for a quick pass
SEED = 2027
IMS_FAILED = {"IMS_1st_test_B3", "IMS_1st_test_B4", "IMS_2nd_test_B1", "IMS_3rd_test_B3"}
RULES = ["3sig_rms", "ma3sig", "ma5sig", "ma2sig", "ewma", "cusum"]
FAMILY = {**{r: "detector" for r in ("3sig_rms", "ma3sig", "ma5sig", "ma2sig")},
          **{r: "control chart" for r in ("ewma", "cusum")}}
AMPLITUDE = {"rms", "std", "p2p", "absmax"}
SHAPE = {"crest", "kurtosis", "skew"}
_fh = None


def log(m: str = "") -> None:
    print(m, flush=True)
    if _fh:
        _fh.write(m + "\n"); _fh.flush()


def load_mod(name: str, fn: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fn)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def first_confirmed(flag: np.ndarray) -> int | None:
    run = 0
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        if run >= CONFIRM:
            return i - CONFIRM + 1
    return None


def fpt(h: np.ndarray, rule: str) -> int | None:
    n = len(h)
    nb = max(5, int(n * BASE_FRAC))
    mu, sd = float(np.mean(h[:nb])), max(float(np.std(h[:nb])), 1e-12)
    if rule == "3sig_rms":
        return first_confirmed(h > mu + 3 * sd)
    if rule == "ewma":
        lam, L = 0.2, 3.0
        z, out = mu, np.empty(n)
        for i, v in enumerate(h):
            z = lam * v + (1 - lam) * z
            out[i] = z
        return first_confirmed(out > mu + L * sd * np.sqrt(lam / (2 - lam)))
    if rule == "cusum":
        k, H = 0.5 * sd, 5.0 * sd
        s, out = 0.0, np.empty(n)
        for i, v in enumerate(h):
            s = max(0.0, s + (v - mu) - k)
            out[i] = s
        return first_confirmed(out > H)
    if rule.startswith("ma"):
        ma = pd.Series(h).rolling(5, min_periods=1).mean().to_numpy()
        return first_confirmed(ma > mu + float(rule[2]) * sd)
    raise ValueError(rule)


# ------------------------------------------------------------------ E3
def indicator_frames() -> dict[str, pd.DataFrame]:
    """dataset -> DataFrame(trajectory, <indicator columns present in that dataset>)."""
    out = {}
    for ds in ("xjtu", "ims"):
        F = pd.read_csv(T / f"j7_features_{ds}.csv")
        rec = pd.read_csv(BUNDLE / ds / "records.csv")
        keep = rec.trajectory_id.isin(IMS_FAILED).to_numpy() if ds == "ims" else np.ones(len(rec), bool)
        d = F.loc[keep].copy()
        d.insert(0, "trajectory", rec.loc[keep, "trajectory_id"].to_numpy())
        out[ds] = d
    f = pd.read_csv(T / "j7c_femto_features.csv")
    out["femto"] = pd.DataFrame({"trajectory": f.bearing, "rms": f.v_rms, "p2p": f.v_p2p,
                                 "absmax": f.v_absmax, "crest": f.v_crest, "kurtosis": f.v_kurtosis})
    return out


def part_e3() -> pd.DataFrame:
    log(f"\n{BAR}\nE3  the Section 4.2 measurement on every indicator the feature tables hold\n{BAR}")
    frames = indicator_frames()
    rows = []
    for ds, d in frames.items():
        cols = [c for c in d.columns if c != "trajectory"]
        missing = sorted((AMPLITUDE | SHAPE) - set(cols))
        log(f"\n{ds}: {len(cols)} indicators {sorted(cols)}"
            + (f"  | NOT DEPOSITED: {missing}" if missing else ""))
        series = {t: g for t, g in d.groupby("trajectory", sort=False)}
        if len(series) < 4:
            log("  fewer than four trajectories -- no correlation reported")
            continue
        log(f"  {'indicator':10s} {'kind':9s} {'rule':9s} {'fired':>6s} {'rho(i,n)':>9s} {'shift sd':>9s} "
            f"{'inside base':>12s} {'onset frac':>11s}")
        for col in cols:
            kind = "amplitude" if col in AMPLITUDE else ("shape" if col in SHAPE else "other")
            for rule in RULES:
                idx, lens, shifts, inside = [], [], [], 0
                for t, g in series.items():
                    h = g[col].to_numpy(float)
                    h = h[np.isfinite(h)]
                    if len(h) < 25:
                        continue
                    n = len(h)
                    nb = max(5, int(n * BASE_FRAC))
                    i = fpt(h, rule)
                    lens.append(n)
                    if i is None:
                        idx.append(np.nan)
                        continue
                    idx.append(i)
                    inside += int(i < nb)
                    mu, sd = float(h[:nb].mean()), max(float(h[:nb].std()), 1e-12)
                    w = h[i:i + CONFIRM]
                    shifts.append((float(w.mean()) - mu) / sd)
                a, b = np.asarray(idx, float), np.asarray(lens, float)
                ok = np.isfinite(a)
                if ok.sum() < 4:
                    continue
                rho = float(spearmanr(b[ok], a[ok])[0])
                rows.append({"dataset": ds, "indicator": col, "kind": kind, "rule": rule, "family": FAMILY[rule],
                             "n_traj": len(b), "n_fired": int(ok.sum()), "rho_index_length": rho,
                             "median_shift_sd": float(np.median(shifts)) if shifts else np.nan,
                             "inside_baseline": inside,
                             "median_onset_fraction": float(np.nanmedian(a[ok] / b[ok]))})
                r = rows[-1]
                log(f"  {col:10s} {kind:9s} {rule:9s} {r['n_fired']:6d} {rho:+9.3f} "
                    f"{r['median_shift_sd']:9.2f} {inside:12d} {r['median_onset_fraction']:11.3f}")
    ind = pd.DataFrame(rows)
    ind.to_csv(T / "j27_indicators.csv", index=False)
    log(f"\nwrote j27_indicators.csv ({len(ind)} rows)")

    log(f"\n{BAR}\nE3 summary: family gap in rho, per indicator\n{BAR}")
    log(f"  {'dataset':7s} {'indicator':10s} {'kind':9s} {'detectors':>10s} {'charts':>8s} {'gap':>8s}")
    for (ds, col), g in ind.groupby(["dataset", "indicator"], sort=False):
        det = g[g.family == "detector"].rho_index_length.median()
        ch = g[g.family == "control chart"].rho_index_length.median()
        kind = g.kind.iloc[0]
        log(f"  {ds:7s} {col:10s} {kind:9s} {det:+10.3f} {ch:+8.3f} {det - ch:+8.3f}")
    log("\n  by kind of indicator (median over indicators of that kind):")
    for (ds, kind), g in ind.groupby(["dataset", "kind"], sort=False):
        det = g[g.family == "detector"].rho_index_length.median()
        ch = g[g.family == "control chart"].rho_index_length.median()
        log(f"    {ds:7s} {kind:9s} detectors {det:+.3f}  charts {ch:+.3f}  gap {det - ch:+.3f}")
    return ind


# ------------------------------------------------------------------ E4
def part_e4() -> pd.DataFrame:
    log(f"\n{BAR}\nE4  the same statistic in a simulation, where nothing but n differs between series\n{BAR}")
    j20 = load_mod("j20", "J20_arl_mechanism.py")
    base = pd.read_csv(T / "j22_baseline_autocorrelation.csv")
    phi = float(np.clip(base[base.dataset == "xjtu"].phi_detrended.median(), -0.99, 0.99))
    lens = sorted(pd.read_csv(T / "fig1_fpt_matrix.csv").query("dataset == 'xjtu'")
                  .drop_duplicates("trajectory").n_records)
    rng = np.random.default_rng(SEED)
    log(f"  AR(1) phi = {phi:.3f}, the {len(lens)} XJTU-SY lengths ({min(lens)}-{max(lens)}), R={R_SIM} per length")
    log(f"  identical generator to J24; J24's own outputs are not touched\n")
    log(f"  {'shape':5s} {'delta':>5s} {'u':>4s} {'rule':9s} {'family':14s} {'fired':>7s} {'rho(i,n)':>9s} "
        f"{'rho(frac,n)':>12s}")
    rows = []
    for shape in ("step", "ramp"):
        for delta in (1.0, 2.0, 3.0, 6.0):
            for u in (0.3, 0.5, 0.7):
                per = {r: ([], []) for r in RULES}
                for n in lens:
                    tau = int(u * n)
                    mean = np.zeros(n)
                    if shape == "step":
                        mean[tau:] = delta
                    else:
                        mean[tau:] = delta * np.linspace(0, 1, n - tau)
                    H = j20.ar1(rng, R_SIM, n, phi, 0.0, 1.0) + mean   # delta is in baseline sd, so generate standardised
                    for r in RULES:
                        o = j20.onsets(H, r)
                        per[r][0].extend(list(np.asarray(o, float)))
                        per[r][1].extend([float(n)] * len(np.asarray(o, float)))
                for r in RULES:
                    a, b = np.asarray(per[r][0], float), np.asarray(per[r][1], float)
                    ok = np.isfinite(a) & (a >= 0)   # J20.onsets returns -1 when the rule never fires
                    if ok.sum() < 30:
                        continue
                    rho = float(spearmanr(b[ok], a[ok])[0])
                    rhof = float(spearmanr(b[ok], a[ok] / b[ok])[0])
                    rows.append({"shape": shape, "delta_sd": delta, "u": u, "rule": r, "family": FAMILY[r],
                                 "n_series": int(len(a)), "n_fired": int(ok.sum()),
                                 "rho_index_length": rho, "rho_fraction_length": rhof,
                                 "median_index": float(np.median(a[ok]))})
                    s = rows[-1]
                    log(f"  {shape:5s} {delta:5.1f} {u:4.1f} {r:9s} {FAMILY[r]:14s} "
                        f"{s['n_fired']:7d} {rho:+9.3f} {rhof:+12.3f}")
    sim = pd.DataFrame(rows)
    sim.to_csv(T / "j27_sim_by_length.csv", index=False)
    log(f"\nwrote j27_sim_by_length.csv ({len(sim)} rows)")

    log(f"\n{BAR}\nE4 summary: median over u, by size of change\n{BAR}")
    log(f"  {'shape':5s} {'delta':>5s} {'detectors rho':>14s} {'charts rho':>11s} {'gap':>8s}")
    for (shape, delta), g in sim.groupby(["shape", "delta_sd"], sort=False):
        det = g[(g.family == "detector") & g.rule.isin(["3sig_rms", "ma3sig", "ma5sig"])].rho_index_length.median()
        ch = g[g.family == "control chart"].rho_index_length.median()
        log(f"  {shape:5s} {delta:5.1f} {det:+14.3f} {ch:+11.3f} {det - ch:+8.3f}")
    return sim


def main() -> int:
    global _fh
    LOGS.mkdir(parents=True, exist_ok=True)
    _fh = open(LOGS / "j27_run.log", "w", encoding="utf-8")
    log(BAR)
    log("J27  E3 seven indicators  +  E4 duration scaling in simulation".center(78))
    log(BAR)
    log(f"started {datetime.now():%Y-%m-%d %H:%M:%S} | baseline {BASE_FRAC:.0%} | confirm {CONFIRM}")
    part_e3()
    part_e4()
    log(f"\n{BAR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
