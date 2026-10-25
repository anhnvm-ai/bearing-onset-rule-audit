"""J47 -- EXP-4: two sequence models (GRU, TCN) under J35's protocol, with two leakage controls.

The objection J35 and J39 invite: the six regressors read one record at a time, so a reader can say the
labeller dominates only because no model in the pool uses the sequence. This adds the two architectures
that do. It does not touch J35, J37 or J38; it is spliced in the way J39 is, by scoring new models under
the same protocol and recomputing the labeller-versus-model comparison with them in the pool.

EVERYTHING BELOW WAS FIXED BEFORE ANY RESULT OF THIS SCRIPT WAS SEEN (2026-09-24). It follows the
shortened EXP-4 of PLAN_UPGRADE_2026-09-23.md 2.4: GRU and TCN only. No Transformer -- its positional
encoding is t, and the target of Eq. (2) is a function of t, so it would be scored on the leak. No
separate LSTM -- GRU already covers the recurrent axis.

  window      L = 16 records, causal, ending at the record being predicted. The first records of a
              bearing are left-padded by repeating that bearing's first record.
  features    exactly J35's per-record statistics (7 on XJTU-SY, 5 on FEMTO), standardised with the
              mean and standard deviation of the TRAINING bearings only. No time index, as in J35.
  GRU         one layer, hidden 32, linear head on the last hidden state.
  TCN         three causal dilated convolutions (kernel 3, dilations 1, 2, 4; receptive field 15 <= L),
              32 channels, ReLU between them, linear head on the last step.
  training    Adam, lr 1e-3, batch 256, EPOCHS = 30, MSE loss. No early stopping and no validation
              split: this repository has none, and carving one out of 14 bearings would change the
              protocol J35, J37, J38 and J39 all share.
  seeds       SEEDS = (0, 1, 2). The cell value is the MEDIAN over seeds; the seed spread (max - min)
              is reported as its own row, not hidden inside the cell.
  control A   "shuffled order": the L time steps of every window are permuted, a fresh permutation per
              window, at training and at test. If the error does not get worse, what the model learned
              is the amplitude in the window, not the order of it.
  control B   "constant window": every window is replaced by one constant vector, the training-bearing
              feature mean, so the input carries nothing at all about t. Any skill above J35's constant
              baseline under this control is leakage through the protocol rather than learning.
              Both controls are run at seed 0 only, which is declared here rather than chosen later.
  calibration before the run, on one XJTU-SY fold only: 30 epochs at batch 256 take about 11 s for the
              GRU and 5 s for the TCN, and the training loss settles; batch 1024 is faster and, on that
              fold, more accurate. The declared 256 was kept anyway, because the labeller/model comparison
              had not been computed and changing a declared setting after reading a held-out error is how
              tuning creeps in. Note the direction: training these models BETTER can only widen the model
              spread and lower the ratio this paper reports, so the conservative choice here is the one
              that trains them well, not the one that leaves them weak.
  scoring     J35's, unchanged: leave-one-bearing-out over the same usable bearings, the same four
              labellers, the same target, pooled RMSE and MAE over every record, the skill score
              against J35's constant baseline, and the 95% bearing-bootstrap interval of J35/J38
              (seed 0, 20 000 resamples, drawn in the same order).

COMMITMENT 2 of the plan, written before this script existed: a max - min spread can only grow when
models are added, so this experiment can only lower the labeller/model ratio. If the ratio falls below
2, the abstract, the introduction, Section V-F, Section VI and the conclusion change their number and
change the claim from a ratio to a comparison of rank. That is checked at the end of this script and
printed as a verdict, whichever way it comes out.

Inputs : as J35 (features, records, fig1_fpt_matrix.csv) and j35_downstream_rul.csv, j39_two_stage.csv
Outputs: results/tables/j47_sequence_models.csv      dataset x labeller x model x arm: RMSE, MAE, skill
         results/tables/j47_per_bearing.csv          the same cells, per bearing: records, SSE, RMSE
         results/tables/j47_seed_spread.csv          per cell, the spread over the three seeds
         results/tables/j47_compare.csv              the labeller/model comparison with the pool extended
         results/logs/j47_run.log
Run    : python scripts/J47_sequence_models.py
         EPOCHS=2 BEARINGS=2 python scripts/J47_sequence_models.py     a quick pass for timing only
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
TABLES, LOGS = ROOT / "results" / "tables", ROOT / "results" / "logs"

L = 16                       # window length, records
HIDDEN = 32
EPOCHS = int(os.environ.get("EPOCHS", "30"))
BATCH = 256
LR = 1e-3
SEEDS = (0, 1, 2)
CONTROL_SEED = 0
BOOT_SEED, N_BOOT = 0, 20000
ARMS = ("main", "shuffled", "constant")
NEW_MODELS = ("gru", "tcn")
DS_NAME = {"xjtu": "XJTU-SY", "femto": "FEMTO"}
LOG: list[str] = []
torch.set_num_threads(4)      # measured: 1, 2, 4 and 8 threads run this size of model equally fast


def log(s: str = "") -> None:
    print(s)
    LOG.append(s)


def load_j35():
    spec = importlib.util.spec_from_file_location("j35", ROOT / "scripts" / "J35_downstream_rul.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ the two architectures
class GRUHead(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.gru = nn.GRU(n_feat, HIDDEN, batch_first=True)
        self.out = nn.Linear(HIDDEN, 1)

    def forward(self, x):                      # x: (B, L, F)
        h, _ = self.gru(x)
        return self.out(h[:, -1]).squeeze(-1)


class TCNHead(nn.Module):
    """Three causal dilated convolutions; the receptive field 1 + 2*(1+2+4) = 15 fits inside L = 16."""

    def __init__(self, n_feat: int):
        super().__init__()
        ch, k = HIDDEN, 3
        self.convs = nn.ModuleList([nn.Conv1d(n_feat if i == 0 else ch, ch, k, dilation=d)
                                    for i, d in enumerate((1, 2, 4))])
        self.pads = [(k - 1) * d for d in (1, 2, 4)]
        self.out = nn.Linear(ch, 1)

    def forward(self, x):                      # x: (B, L, F)
        h = x.transpose(1, 2)                  # (B, F, L)
        for conv, pad in zip(self.convs, self.pads):
            h = torch.relu(conv(nn.functional.pad(h, (pad, 0))))   # left pad only -> causal
        return self.out(h[:, :, -1]).squeeze(-1)


def build(name: str, n_feat: int) -> nn.Module:
    return GRUHead(n_feat) if name == "gru" else TCNHead(n_feat)


# ------------------------------------------------------------------ windows
def windows(X: np.ndarray, traj: np.ndarray) -> np.ndarray:
    """(n_rec, F) -> (n_rec, L, F): the L records ending at each one, left-padded inside its bearing."""
    out = np.empty((len(X), L, X.shape[1]), dtype=np.float32)
    for b in np.unique(traj):
        m = np.flatnonzero(traj == b)
        Xb = X[m]
        idx = np.clip(np.arange(len(m))[:, None] - np.arange(L - 1, -1, -1)[None, :], 0, None)
        out[m] = Xb[idx]
    return out


def fit_predict(name: str, W_tr: np.ndarray, y_tr: np.ndarray, W_te: np.ndarray, seed: int) -> np.ndarray:
    torch.manual_seed(seed)
    model = build(name, W_tr.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    Xt = torch.from_numpy(W_tr)
    yt = torch.from_numpy(y_tr.astype(np.float32))
    n = len(Xt)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            j = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xt[j]), yt[j])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        p = model(torch.from_numpy(W_te)).numpy()
    return np.clip(p, 0.0, 1.0)


def boot_ci(per_bearing_sq: dict, bearings: list, rng: np.random.Generator) -> tuple[float, float]:
    """J35's interval, recomputed here with the same seed and draw order."""
    ssq = np.array([per_bearing_sq[b][0] for b in bearings], dtype=float)
    cnt = np.array([per_bearing_sq[b][1] for b in bearings], dtype=float)
    idx = rng.integers(0, len(bearings), size=(N_BOOT, len(bearings)))
    vals = np.sqrt(ssq[idx].sum(1) / cnt[idx].sum(1))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def spreads(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v: (labellers, models, ...) -> median label spread, median model spread, as J35, J38 and J39."""
    return (np.median(v.max(axis=0) - v.min(axis=0), axis=0),
            np.median(v.max(axis=1) - v.min(axis=1), axis=0))


def compare(res: pd.DataFrame, J35, log=print) -> pd.DataFrame:
    """The labeller/model comparison over three pools, on both scales J38 reports.

    Pools: the six regressors of J35; those plus the two-stage fit of J39 (seven); those plus the GRU and
    the TCN of this script (nine). The six-model row recomputes J38's own statistic, so it must land on
    J38's factor to the digits it printed -- that is the self-check that the splice is like for like.
    """
    LAB = J35.LABELLERS
    pooled35 = pd.read_csv(TABLES / "j35_downstream_rul.csv")
    two_stage = pd.read_csv(TABLES / "j39_two_stage.csv")
    j38 = pd.read_csv(TABLES / "j38_ratio_bootstrap.csv")
    rows = []
    for ds in J35.DATASETS:
        for scale in ("rmse", "skill"):
            p35 = pooled35[pooled35.dataset == ds].pivot(index="labeller", columns="model",
                                                         values=scale).loc[LAB, J35.MODEL_ORDER]
            ts = two_stage[(two_stage.dataset == ds) & (two_stage.variant == "exp")].set_index("labeller")
            new = res[(res.dataset == ds) & (res.arm == "main")].pivot(index="labeller", columns="model",
                                                                      values=scale).loc[LAB, list(NEW_MODELS)]
            pools = {"six": [p35.to_numpy()],
                     "seven": [p35.to_numpy(), ts.loc[LAB, scale].to_numpy()[:, None]],
                     "nine": [p35.to_numpy(), ts.loc[LAB, scale].to_numpy()[:, None], new.to_numpy()]}
            for tag, parts in pools.items():
                v = np.concatenate(parts, axis=1)
                ls, ms = spreads(v[:, :, None])
                row = {"dataset": ds, "scale": scale, "pool": tag, "n_models": v.shape[1],
                       "label_spread": float(ls[0]), "model_spread_median": float(ms[0]),
                       "model_spread_max": float(np.max(v.max(axis=1) - v.min(axis=1))),
                       "factor": float(ls[0] / ms[0])}
                if tag == "six":
                    ref = j38[(j38.dataset == ds) & (j38.scale == scale)].iloc[0]
                    row["j38_factor"] = float(ref.factor)
                    if abs(row["factor"] - ref.factor) > 5e-4:
                        raise SystemExit(f"{ds}/{scale}: the six-model pool gives {row['factor']:.4f} "
                                         f"but J38 reports {ref.factor:.4f}; the splice is not like for like")
                rows.append(row)
                log(f"  {DS_NAME[ds]:8s} {scale:5s} {tag:5s} ({v.shape[1]} models): labeller spread "
                    f"{ls[0]:.4f}, model spread {ms[0]:.4f}  ->  factor {row['factor']:.2f}"
                    + ("   == J38" if tag == "six" else ""))
    return pd.DataFrame(rows)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    J35 = load_j35()
    LAB = J35.LABELLERS
    ft = pd.read_csv(TABLES / "fig1_fpt_matrix.csv")
    limit_bearings = int(os.environ.get("BEARINGS", "0"))

    log("J47 -- GRU and TCN under J35's protocol, with a shuffled-order and a constant-window control")
    log(f"L={L} hidden={HIDDEN} epochs={EPOCHS} batch={BATCH} lr={LR} seeds={SEEDS}; "
        f"controls at seed {CONTROL_SEED} only; every one of these was fixed before any result was read")
    log("=" * 104)

    rows, seed_rows, per_rows = [], [], []
    for ds in J35.DATASETS:
        traj_all, F_all = J35.load_dataset(ds)
        g = ft[ft.dataset == ds]
        fired = {r: set(g[(g.rule == r) & g.fired].trajectory) for r in LAB}
        usable = sorted(set.intersection(*fired.values()))
        if limit_bearings:
            usable = usable[:limit_bearings]
        keep = np.isin(traj_all, usable)
        traj, F = traj_all[keep], F_all.loc[keep].reset_index(drop=True)
        n_rec = int(keep.sum())
        X = F.to_numpy(dtype=np.float32)
        W = windows(X, traj)

        onset = {(r.rule, r.trajectory): (r.fpt_index if r.fired else None) for r in g.itertuples()}
        n_of = {r.trajectory: int(r.n_records) for r in g.itertuples()}
        Y = {lab: np.concatenate([J35.target(n_of[b], onset[(lab, b)]) for b in usable]) for lab in LAB}
        for lab in LAB:
            if len(Y[lab]) != n_rec:
                raise SystemExit(f"{ds}/{lab}: target and features disagree on the record count")

        log(f"{DS_NAME[ds]}: {len(usable)} bearings, {n_rec} records, {X.shape[1]} features, "
            f"windows {W.shape}")

        for lab in LAB:
            y = Y[lab]
            # J35's constant baseline: the mean target of the TRAINING bearings, held out the same way
            base = np.full(n_rec, np.nan)
            for b in usable:
                te = traj == b
                base[te] = y[~te].mean()
            mse_base = float(((base - y) ** 2).mean())

            for name in NEW_MODELS:
                for arm in ARMS:
                    seeds = SEEDS if arm == "main" else (CONTROL_SEED,)
                    per_seed = {}
                    for seed in seeds:
                        t0 = time.time()
                        p = np.full(n_rec, np.nan)
                        rs = np.random.default_rng([seed, 47])
                        for fold, b in enumerate(usable):
                            te = traj == b
                            tr = ~te
                            mu = W[tr].reshape(-1, X.shape[1]).mean(0)
                            sd = W[tr].reshape(-1, X.shape[1]).std(0)
                            sd[sd == 0] = 1.0
                            Wtr, Wte = (W[tr] - mu) / sd, (W[te] - mu) / sd
                            if arm == "shuffled":
                                for A in (Wtr, Wte):
                                    for i in range(len(A)):
                                        A[i] = A[i][rs.permutation(L)]
                            elif arm == "constant":
                                c = Wtr.reshape(-1, X.shape[1]).mean(0)
                                Wtr = np.broadcast_to(c, Wtr.shape).astype(np.float32).copy()
                                Wte = np.broadcast_to(c, Wte.shape).astype(np.float32).copy()
                            p[te] = fit_predict(name, Wtr, y[tr], Wte, seed * 1000 + fold)
                        if np.isnan(p).any():
                            raise SystemExit(f"{ds}/{lab}/{name}/{arm}: a record was never held out")
                        err = p - y
                        rmse = float(np.sqrt((err ** 2).mean()))
                        per_seed[seed] = (rmse, float(np.abs(err).mean()),
                                          float(1.0 - rmse ** 2 / mse_base), err)
                        log(f"  {DS_NAME[ds]:8s} {lab:9s} {name:4s} {arm:9s} seed {seed}  "
                            f"rmse {rmse:.4f}  skill {1.0 - rmse ** 2 / mse_base:+.3f}  "
                            f"[{time.time() - t0:.0f}s]")

                    rmses = [per_seed[s][0] for s in seeds]
                    mid = seeds[int(np.argsort(rmses)[len(rmses) // 2])]     # the median seed
                    rmse, mae, skill, err = per_seed[mid]
                    pb = {b: (float((err[traj == b] ** 2).sum()), int((traj == b).sum())) for b in usable}
                    lo, hi = boot_ci(pb, usable, np.random.default_rng(BOOT_SEED))
                    for b, (sse, cnt) in pb.items():
                        per_rows.append({"dataset": ds, "labeller": lab, "model": name, "arm": arm,
                                         "bearing": b, "n_records": cnt, "sse": sse,
                                         "rmse": float(np.sqrt(sse / cnt))})
                    rows.append({"dataset": ds, "labeller": lab, "model": name, "arm": arm,
                                 "n_bearings": len(usable), "n_records": n_rec, "seeds": len(seeds),
                                 "median_seed": mid, "rmse": rmse, "rmse_lo": lo, "rmse_hi": hi,
                                 "mae": mae, "skill": skill,
                                 "rmse_baseline": float(np.sqrt(mse_base))})
                    if arm == "main":
                        seed_rows.append({"dataset": ds, "labeller": lab, "model": name,
                                          "rmse_min": float(min(rmses)), "rmse_max": float(max(rmses)),
                                          "seed_spread": float(max(rmses) - min(rmses)),
                                          "rmse_median": rmse})

    res = pd.DataFrame(rows)
    seed_df = pd.DataFrame(seed_rows)

    # ---------------------------------------------------------- the comparison, spliced as J39 splices it
    log("")
    log("=" * 104)
    log("the labeller/model comparison over three pools (six regressors; + two-stage; + GRU and TCN)")
    cmp_df = compare(res, J35, log)

    # ---------------------------------------------------------- what the two controls say
    log("")
    log("the two controls (seed 0), read against the same cell in the main arm")
    for ds in J35.DATASETS:
        for name in NEW_MODELS:
            m = res[(res.dataset == ds) & (res.model == name)].set_index(["labeller", "arm"]).rmse
            for lab in LAB:
                log(f"  {DS_NAME[ds]:8s} {lab:9s} {name:4s}  main {m[(lab, 'main')]:.4f}   "
                    f"shuffled {m[(lab, 'shuffled')]:.4f} ({m[(lab, 'shuffled')] - m[(lab, 'main')]:+.4f})   "
                    f"constant {m[(lab, 'constant')]:.4f} ({m[(lab, 'constant')] - m[(lab, 'main')]:+.4f})")

    log("")
    log("seed spread of the main arm, reported as its own quantity")
    for r in seed_df.itertuples():
        log(f"  {DS_NAME[r.dataset]:8s} {r.labeller:9s} {r.model:4s}  median {r.rmse_median:.4f}  "
            f"spread over {len(SEEDS)} seeds {r.seed_spread:.4f}")

    # ---------------------------------------------------------- commitment 2, checked either way
    log("")
    log("=" * 104)
    for ds in J35.DATASETS:
        q = cmp_df[(cmp_df.dataset == ds) & (cmp_df.scale == "rmse")].set_index("pool").factor
        verdict = ("BELOW 2: commitment 2 applies -- the abstract, the introduction, Section V-F, "
                   "Section VI and the conclusion must state a rank comparison, not a ratio"
                   if q["nine"] < 2.0 else "at or above 2: the ratio stands, with the wider pool named")
        log(f"{DS_NAME[ds]}: factor {q['six']:.2f} (six models, = J38) -> {q['seven']:.2f} (seven) -> "
            f"{q['nine']:.2f} (nine). {verdict}")

    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "j47_sequence_models.csv", index=False)
    pd.DataFrame(per_rows).to_csv(TABLES / "j47_per_bearing.csv", index=False)
    seed_df.to_csv(TABLES / "j47_seed_spread.csv", index=False)
    cmp_df.to_csv(TABLES / "j47_compare.csv", index=False)
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "j47_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    log("")
    log("wrote j47_sequence_models.csv, j47_per_bearing.csv, j47_seed_spread.csv, j47_compare.csv; "
        f"finished {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")
    (LOGS / "j47_run.log").write_text("\n".join(LOG) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
