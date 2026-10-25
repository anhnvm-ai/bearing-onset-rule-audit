"""Small deep models for the E2 label-recoverability audit.

Deliberately standard/compact — the point of E2 is to test whether ANY deep model
beats a 1-feature RMS threshold on this single-trajectory data, not to tune SOTA.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """Raw 2-channel (vx, vy) vibration-window classifier."""

    def __init__(self, n_classes: int, in_ch: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_ch, 16, kernel_size=64, stride=8, padding=32),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=16, stride=2, padding=8),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class ThermalMLP(nn.Module):
    """Hour-level thermal-feature classifier."""

    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)
