"""Canonical evaluation protocol — the single source of truth for folds and labels.

Every experiment (E1-E7) MUST use the frozen fold assignment written to
    data/metadata/fold_assignments.csv
so that models, thresholds, normalisation schemes and ablations are compared on
exactly the same held-out hours. This removes the biggest confound in the previous
window-level pipeline, where folds were re-derived on the fly by sklearn GroupKFold
(which does NOT stratify by label and dumped most terminal hours into one fold ->
the "fold-4 collapse").

Key design decisions (see PROGRESS.md plan sections II.2, II.3, R3):
  * UNIT OF EVALUATION = the hour. Models train on windows but every hour receives a
    single out-of-fold prediction (probability-averaged over its windows, see src.oof).
  * GROUPING = the hour. All 50 windows of an hour go entirely to train OR test, so a
    model can never memorise per-hour operating conditions across the split.
  * STRATIFICATION = the terminal regime. The five folds are stratified so the terminal
    hours are spread across folds; for the primary H5 horizon each fold holds exactly
    one terminal hour, so pooled raw counts (TP/FP/FN/TN) are well defined.
  * LABELS are retrospectively-defined terminal-state pseudo-labels at three horizons
    (H3/H5/H8) on the SAME fixed folds, so horizon sensitivity (E4) never confounds the
    label boundary with fold composition.

Fold assignment is fully deterministic (no RNG): late hours (>=121, the pool that
carries label signal at any horizon) and early hours (<121) are each round-robin
assigned to folds in ascending hour order.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from .data_loader import PROJECT_ROOT

METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")
FOLD_FILE = os.path.join(METADATA_DIR, "fold_assignments.csv")

N_SPLITS = 5
N_HOURS = 129                      # hour_id 0..128
LATE_POOL_START = 121              # hours >=121 carry label signal at the widest horizon (H8)

# Retrospectively-defined terminal-state horizons: positive (terminal) iff hour_id >= onset.
HORIZONS = {"h3": 126, "h5": 124, "h8": 121}
PRIMARY_HORIZON = "h5"


def _round_robin(hours: list[int], n_splits: int) -> dict[int, int]:
    """Assign each hour (ascending) to folds 0..n_splits-1 cyclically."""
    return {int(h): i % n_splits for i, h in enumerate(sorted(hours))}


def build_fold_table(n_hours: int = N_HOURS, n_splits: int = N_SPLITS,
                     hours_from_start: pd.Series | None = None) -> pd.DataFrame:
    """Build the canonical hour-level fold/label table (deterministic)."""
    hours = list(range(n_hours))
    late = [h for h in hours if h >= LATE_POOL_START]
    early = [h for h in hours if h < LATE_POOL_START]
    fold_of = {}
    fold_of.update(_round_robin(late, n_splits))     # spread terminal hours across folds
    fold_of.update(_round_robin(early, n_splits))    # spread healthy hours across folds

    # Secondary robustness protocol: contiguous temporal blocks (blocked-temporal CV).
    # NOTE: on single-trajectory data all terminal hours fall in the last block, so this
    # protocol cannot estimate per-fold terminal recall; it is reported only as a check.
    edges = np.linspace(0, n_hours, n_splits + 1).astype(int)
    blocked_of = {h: int(np.searchsorted(edges[1:-1], h, side="right")) for h in hours}

    rows = []
    for h in hours:
        row = {
            "hour_id": h,
            "fold_id": fold_of[h],
            "blocked_fold_id": blocked_of[h],
        }
        for name, onset in HORIZONS.items():
            row[f"label_{name}"] = int(h >= onset)
        rows.append(row)
    df = pd.DataFrame(rows)
    if hours_from_start is not None:
        df["hours_from_start"] = df["hour_id"].map(hours_from_start.to_dict())
    return df


def write_fold_table(df: pd.DataFrame | None = None) -> str:
    os.makedirs(METADATA_DIR, exist_ok=True)
    if df is None:
        df = build_fold_table()
    df.to_csv(FOLD_FILE, index=False)
    return FOLD_FILE


def load_folds() -> pd.DataFrame:
    if not os.path.exists(FOLD_FILE):
        raise FileNotFoundError(
            f"{FOLD_FILE} not found. Run scripts/20_build_protocol.py first.")
    return pd.read_csv(FOLD_FILE)


# ---------------------------------------------------------------------------
# Split helpers operating on the per-window hour_id vector.
# ---------------------------------------------------------------------------
def hour_grouped_splits(window_hours: np.ndarray, folds: pd.DataFrame | None = None,
                        n_splits: int = N_SPLITS):
    """Frozen hour-grouped CV: yield (train_idx, test_idx) window-index arrays per fold.

    test fold f = every window whose hour_id is assigned fold f in fold_assignments.csv.
    """
    if folds is None:
        folds = load_folds()
    fmap = dict(zip(folds["hour_id"], folds["fold_id"]))
    wf = np.array([fmap[int(h)] for h in window_hours])
    out = []
    for f in range(n_splits):
        te = np.where(wf == f)[0]
        tr = np.where(wf != f)[0]
        out.append((tr, te))
    return out


def blocked_splits(window_hours: np.ndarray, folds: pd.DataFrame | None = None,
                   n_splits: int = N_SPLITS):
    """Secondary blocked-temporal CV (contiguous hour blocks)."""
    if folds is None:
        folds = load_folds()
    fmap = dict(zip(folds["hour_id"], folds["blocked_fold_id"]))
    wf = np.array([fmap[int(h)] for h in window_hours])
    return [(np.where(wf != f)[0], np.where(wf == f)[0]) for f in range(n_splits)]


def random_window_splits(n: int, y: np.ndarray, n_splits: int = N_SPLITS, seed: int = 0):
    """LEAKY protocol for E1: stratified split over individual windows, ignoring hour.

    Windows of one hour land in both train and test, so a model can memorise per-hour
    operating conditions. Any above-chance score under a RANDOM per-hour label is
    therefore pure leakage.
    """
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(n), y))


def label_for(window_hours: np.ndarray, horizon: str = PRIMARY_HORIZON) -> np.ndarray:
    """Per-window binary terminal label for a given horizon."""
    onset = HORIZONS[horizon]
    return (np.asarray(window_hours) >= onset).astype(int)


def inner_group_val_split(train_idx: np.ndarray, window_hours: np.ndarray,
                          y_window: np.ndarray, seed: int = 0,
                          val_frac: float = 0.2, min_terminal_val: int = 1):
    """Carve an INNER validation set out of an outer-train fold, grouped by hour.

    Used by the convergence audit (scripts/33_convergence_audit.py). Early stopping must
    never see the outer TEST fold, so the validation set is drawn ONLY from the outer-train
    hours. The split is grouped by hour (all windows of an hour go entirely to inner-train
    OR inner-val) and stratified on the terminal regime so the very scarce terminal hours
    are represented in BOTH parts (inner-train keeps >=1 terminal, inner-val gets >=1).

    Parameters
    ----------
    train_idx : absolute window indices of the outer-train fold (positions into the
                6450-row window arrays), i.e. the ``tr`` array from hour_grouped_splits.
    window_hours : per-window hour_id vector (length = n_windows).
    y_window : per-window binary terminal label (constant within an hour).
    seed : RNG seed (varies the chosen val hours per model seed).
    val_frac : fraction of hours (per regime) held out for validation.
    min_terminal_val : minimum terminal hours forced into inner-val so that val_loss /
                       val_macro_f1 are defined (>=1). inner-train always keeps >=1 too.

    Returns
    -------
    (inner_train_idx, inner_val_idx) : absolute window-index arrays partitioning train_idx.
    """
    train_idx = np.asarray(train_idx)
    ylab = np.asarray(y_window)
    hrs = np.asarray(window_hours)[train_idx]
    uniq = np.unique(hrs)
    # hour -> label (constant within an hour)
    hour_label = {int(h): int(ylab[train_idx[hrs == h][0]]) for h in uniq}
    terminal = np.array([h for h in uniq if hour_label[h] == 1], dtype=int)
    reference = np.array([h for h in uniq if hour_label[h] == 0], dtype=int)
    rng = np.random.RandomState(seed)

    def pick(pool, frac, force_min):
        if len(pool) == 0:
            return np.array([], dtype=int)
        k = int(round(frac * len(pool)))
        k = max(k, force_min)
        k = min(k, len(pool) - 1)          # always leave >=1 hour of this regime in inner-train
        if k <= 0:
            return np.array([], dtype=int)
        return rng.choice(pool, size=k, replace=False)

    val_hours = set(pick(terminal, val_frac, min_terminal_val).tolist()
                    + pick(reference, val_frac, 0).tolist())
    is_val = np.array([int(h) in val_hours for h in hrs])
    inner_val_idx = train_idx[is_val]
    inner_train_idx = train_idx[~is_val]
    return inner_train_idx, inner_val_idx


if __name__ == "__main__":
    from .data_loader import list_files
    meta = list_files()
    hfs = meta.set_index("hour_id")["hours_from_start"]
    tbl = build_fold_table(hours_from_start=hfs)
    path = write_fold_table(tbl)
    print(f"[protocols] wrote {path}")
    print(tbl.to_string(index=False))
    # sanity: positives per fold at each horizon
    for name in HORIZONS:
        piv = tbl.groupby("fold_id")[f"label_{name}"].sum()
        print(f"  {name}: positives/fold = {piv.tolist()}  (total {int(piv.sum())})")
