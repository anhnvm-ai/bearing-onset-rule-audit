"""Hour-level classification metrics with raw confusion counts.

The whole point of the reporting-honesty argument (E4 in the old numbering / P0.2 in
the plan) is that with only a handful of terminal hours, third-decimal macro-F1
rankings are meaningless. So the primary reported quantity is RAW COUNTS
(TP/FP/FN/TN) at the HOUR level, from which every scalar metric is derived, plus MCC
(robust to class imbalance). Everything here operates on one prediction per hour.
"""
from __future__ import annotations
import numpy as np


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Binary confusion counts with positive = terminal (class 1)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Full metric bundle from raw counts (all derived, no sklearn dependency)."""
    c = confusion_counts(y_true, y_pred)
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    n = tp + fp + fn + tn

    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)                       # terminal recall (sensitivity)
    spec = _safe_div(tn, tn + fp)                      # specificity
    f1_pos = _safe_div(2 * prec * rec, prec + rec)
    # negative-class F1
    prec0 = _safe_div(tn, tn + fn)
    rec0 = _safe_div(tn, tn + fp)
    f1_neg = _safe_div(2 * prec0 * rec0, prec0 + rec0)
    macro_f1 = 0.5 * (f1_pos + f1_neg)
    bal_acc = 0.5 * (rec + spec)
    acc = _safe_div(tp + tn, n)

    # Matthews correlation coefficient (0 when any margin is degenerate)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = float(((tp * tn) - (fp * fn)) / denom) if denom > 0 else 0.0

    out = dict(c)
    out.update({
        "n": n,
        "terminal_precision": prec,
        "terminal_recall": rec,
        "specificity": spec,
        "f1_terminal": f1_pos,
        "f1_reference": f1_neg,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "mcc": mcc,
    })
    return out


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Recompute the full metric bundle from raw confusion counts.

    Reconstructs a consistent (y_true, y_pred) label pair and delegates to binary_metrics,
    so macro_f1 / mcc / balanced_accuracy / precision / recall use the EXACT pipeline
    formulas (single source of truth). Used by the final-table / validator / paper-number
    scripts (37/39/40) so no metric is ever typed by hand.
    """
    tp, fp, fn, tn = int(tp), int(fp), int(fn), int(tn)
    y_true = np.r_[np.ones(tp + fn), np.zeros(tn + fp)].astype(int)
    y_pred = np.r_[np.ones(tp), np.zeros(fn), np.ones(fp), np.zeros(tn)].astype(int)
    return binary_metrics(y_true, y_pred)


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return binary_metrics(y_true, y_pred)["macro_f1"]


def multiclass_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 for >2 classes (used by the E1 3-stage/random-label controls)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1s.append(_safe_div(2 * prec * rec, prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def balanced_accuracy_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    recalls = []
    for c in np.unique(y_true):
        m = y_true == c
        recalls.append(_safe_div(np.sum(y_pred[m] == c), np.sum(m)))
    return float(np.mean(recalls)) if recalls else 0.0
