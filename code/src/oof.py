"""Out-of-fold prediction at the HOUR level (plan section II.2).

Models are trained on windows, but the unit of evaluation is the hour: every hour gets
ONE out-of-fold prediction obtained by averaging the terminal-class probability over its
windows and thresholding. This is the single most important methodological change from
the old window-level pipeline, where 5 terminal hours became 250 windows and inflated the
effective sample size.

    p_bar_h = mean_j P(terminal | window h,j)        y_hat_h = 1[p_bar_h >= tau]

`aggregate_to_hour` is model-agnostic: give it per-window terminal probabilities (from a
sklearn predict_proba OR a cached softmax of a deep model) and it returns one row per hour.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def aggregate_to_hour(window_hours: np.ndarray, prob_terminal: np.ndarray,
                      true_window: np.ndarray, fold_window: np.ndarray | None = None,
                      threshold: float = 0.5) -> pd.DataFrame:
    """Collapse per-window terminal probabilities to one prediction per hour.

    Returns a DataFrame: hour_id, fold_id, true_label, prob_terminal, pred_label, n_windows.
    true_label is taken as the (constant) window label within the hour.
    """
    df = pd.DataFrame({
        "hour_id": np.asarray(window_hours).astype(int),
        "prob": np.asarray(prob_terminal, dtype=float),
        "true": np.asarray(true_window).astype(int),
    })
    if fold_window is not None:
        df["fold_id"] = np.asarray(fold_window).astype(int)
    agg = {"prob": "mean", "true": "first", "hour_id": "size"}
    g = df.groupby("hour_id")
    out = pd.DataFrame({
        "prob_terminal": g["prob"].mean(),
        "true_label": g["true"].first(),
        "n_windows": g.size(),
    })
    if fold_window is not None:
        out["fold_id"] = g["fold_id"].first()
    out = out.reset_index()
    out["pred_label"] = (out["prob_terminal"] >= threshold).astype(int)
    cols = ["hour_id", "fold_id", "true_label", "prob_terminal", "pred_label", "n_windows"]
    return out[[c for c in cols if c in out.columns]].sort_values("hour_id").reset_index(drop=True)


def classical_oof(model_factory, X: np.ndarray, y_window: np.ndarray,
                  window_hours: np.ndarray, splits, threshold: float = 0.5) -> pd.DataFrame:
    """Run an sklearn model through frozen folds and return HOUR-level OOF predictions.

    model_factory() must return a fresh classifier exposing predict_proba (or decision).
    X is per-window features; y_window per-window training labels; splits from protocols.
    """
    X = np.asarray(X)
    y_window = np.asarray(y_window).astype(int)
    window_hours = np.asarray(window_hours).astype(int)

    prob = np.full(len(X), np.nan)
    fold = np.full(len(X), -1)
    for fi, (tr, te) in enumerate(splits):
        clf = model_factory()
        clf.fit(X[tr], y_window[tr])
        prob[te] = _terminal_prob(clf, X[te])
        fold[te] = fi
    assert not np.isnan(prob).any(), "some windows never appeared in a test fold"
    return aggregate_to_hour(window_hours, prob, y_window, fold, threshold)


def _terminal_prob(clf, X: np.ndarray) -> np.ndarray:
    """P(class==1) for a fitted classifier, robust to single-class train folds."""
    if hasattr(clf, "predict_proba"):
        classes = list(clf.classes_)
        proba = clf.predict_proba(X)
        if 1 in classes:
            return proba[:, classes.index(1)]
        return np.zeros(len(X))          # train fold saw only class 0
    if hasattr(clf, "decision_function"):
        d = clf.decision_function(X)
        return 1.0 / (1.0 + np.exp(-d))
    return clf.predict(X).astype(float)
