"""Cross-validation protocols — the heart of the E1 leakage demonstration.

- random_window_folds: StratifiedKFold over individual windows, IGNORING which hour
  each window came from. This is the common-but-WRONG protocol: windows from the same
  hour land in both train and test, so a model can memorise per-hour operating
  conditions => leakage => inflated scores.
- hour_grouped_folds: GroupKFold by hour_id. Every hour is entirely in train OR test,
  so the model must generalise to unseen hours. This is the correct protocol.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold


def random_window_folds(y, n_splits: int = 5, seed: int = 0):
    """Leaky protocol: random stratified split over windows."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def hour_grouped_folds(y, groups, n_splits: int = 5):
    """Correct protocol: whole hours held out (GroupKFold by hour_id)."""
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(y)), y, groups))
