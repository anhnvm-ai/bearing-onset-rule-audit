"""Single-feature threshold baseline, fit HONESTLY inside each training fold.

The recoverability argument hinges on a FAIR simple baseline: a one-number threshold
that never sees the test hours. Two protocols (plan section P0.3 / E3):

  * train-fold-only : in every fold the threshold is swept on the TRAINING hours only,
    locked, then applied to the held-out hours. This is the honest baseline used in the
    main audit (E2).
  * oracle          : the threshold is swept on ALL hours. Reported ONLY to show the
    (small) optimism from boundary leakage; never used as the headline number.

Operates at the HOUR level on a single scalar feature (default: mean window RMS per hour,
or the full-signal bearing temperature), positive = terminal when feature >= threshold.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .metrics import binary_metrics


def _candidate_thresholds(x: np.ndarray) -> np.ndarray:
    u = np.unique(np.asarray(x, dtype=float))
    if len(u) < 2:
        return np.array([u[0] if len(u) else 0.0])
    mids = (u[:-1] + u[1:]) / 2.0
    # also allow the two extremes so an all-one / all-zero rule is reachable
    return np.concatenate([[u[0] - 1e-9], mids, [u[-1] + 1e-9]])


def _score(y_true, y_pred, metric: str) -> float:
    m = binary_metrics(y_true, y_pred)
    return m[metric]


def fit_threshold(feat: np.ndarray, y: np.ndarray, metric: str = "balanced_accuracy",
                  higher_is_positive: bool = True) -> float:
    """Sweep thresholds on (feat, y) and return the one maximising `metric`.

    Ties are broken by the median threshold among the best (a stable central choice).
    """
    feat = np.asarray(feat, dtype=float)
    y = np.asarray(y).astype(int)
    cands = _candidate_thresholds(feat)
    scores = []
    for t in cands:
        pred = (feat >= t).astype(int) if higher_is_positive else (feat <= t).astype(int)
        scores.append(_score(y, pred, metric))
    scores = np.asarray(scores)
    best = scores.max()
    best_ts = cands[scores >= best - 1e-12]
    return float(np.median(best_ts))


def predict_threshold(feat: np.ndarray, t: float, higher_is_positive: bool = True) -> np.ndarray:
    feat = np.asarray(feat, dtype=float)
    return (feat >= t).astype(int) if higher_is_positive else (feat <= t).astype(int)


def threshold_oof(hour_feat: pd.Series, y_hour: pd.Series, fold_hour: pd.Series,
                  metric: str = "balanced_accuracy", higher_is_positive: bool = True,
                  n_splits: int = 5) -> pd.DataFrame:
    """Per-fold train-only threshold; return HOUR-level OOF predictions + the fold threshold.

    hour_feat/y_hour/fold_hour are aligned pandas Series indexed by hour_id.
    """
    hours = hour_feat.index.to_numpy()
    feat = hour_feat.to_numpy(dtype=float)
    y = y_hour.to_numpy().astype(int)
    fold = fold_hour.to_numpy().astype(int)

    rows = []
    for f in range(n_splits):
        te = fold == f
        tr = ~te
        t = fit_threshold(feat[tr], y[tr], metric, higher_is_positive)
        pred = predict_threshold(feat[te], t, higher_is_positive)
        prob = 1.0 / (1.0 + np.exp(-(feat[te] - t)))  # smooth surrogate for ranking/plots
        for h, yt, pr, pb in zip(hours[te], y[te], pred, prob):
            rows.append({"hour_id": int(h), "fold_id": f, "true_label": int(yt),
                         "prob_terminal": float(pb), "pred_label": int(pr),
                         "fold_threshold": float(t)})
    return pd.DataFrame(rows).sort_values("hour_id").reset_index(drop=True)


def oracle_threshold_oof(hour_feat: pd.Series, y_hour: pd.Series, fold_hour: pd.Series,
                         metric: str = "balanced_accuracy",
                         higher_is_positive: bool = True) -> pd.DataFrame:
    """Oracle: one threshold swept on ALL hours, applied everywhere (upper bound only)."""
    hours = hour_feat.index.to_numpy()
    feat = hour_feat.to_numpy(dtype=float)
    y = y_hour.to_numpy().astype(int)
    fold = fold_hour.to_numpy().astype(int)
    t = fit_threshold(feat, y, metric, higher_is_positive)
    pred = predict_threshold(feat, t, higher_is_positive)
    prob = 1.0 / (1.0 + np.exp(-(feat - t)))
    return pd.DataFrame({"hour_id": hours, "fold_id": fold, "true_label": y,
                         "prob_terminal": prob, "pred_label": pred,
                         "fold_threshold": t}).sort_values("hour_id").reset_index(drop=True)
