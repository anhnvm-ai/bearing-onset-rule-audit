"""Assemble hour-level OOF predictions into the paper's tables.

Everything here consumes the standard hour-level OOF DataFrame
(columns: hour_id, fold_id, true_label, prob_terminal, pred_label) so that classical,
threshold and deep models are reported identically.
"""
from __future__ import annotations
import pandas as pd

from .metrics import binary_metrics

COUNT_COLS = ["TP", "FP", "FN", "TN"]
METRIC_COLS = ["terminal_precision", "terminal_recall", "macro_f1",
               "balanced_accuracy", "mcc"]


def metrics_row(name: str, oof: pd.DataFrame) -> dict:
    m = binary_metrics(oof["true_label"].to_numpy(), oof["pred_label"].to_numpy())
    row = {"model": name}
    row.update({k: m[k] for k in COUNT_COLS})
    row.update({k: round(m[k], 4) for k in METRIC_COLS})
    return row


def counts_table(named_oofs: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """One row per model: raw counts + derived metrics (plan Table II)."""
    return pd.DataFrame([metrics_row(n, o) for n, o in named_oofs])


def per_hour_prediction_table(named_oofs: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Wide table hour_id x model of predicted labels, plus true label (plan Table II-b)."""
    base = None
    for name, oof in named_oofs:
        col = oof.set_index("hour_id")["pred_label"].rename(name)
        base = col.to_frame() if base is None else base.join(col)
    truth = named_oofs[0][1].set_index("hour_id")["true_label"]
    base.insert(0, "true_label", truth)
    return base.reset_index()
