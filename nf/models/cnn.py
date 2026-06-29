"""1-D CNN architecture — a faithful re-expression of ``CNNRegressionNet``.

Same conv stack and head the monolith used. Two correctness/clarity wins come
for free from the registry:

* The "input must have >= N features" rule lives in one place
  (``min_input_dim``) and is enforced by the engine, instead of being split
  between a ``freq_bins < 10`` guard in ``__init__`` and a separate
  ``X.shape[1] < 16`` check buried in module-level setup code.
* The (N, features) -> (N, 1, features) reshape is the spec's
  ``expects_channel_dim`` flag, not a free-standing ``preprocess_for_cnn`` that
  every call site had to remember to invoke.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .base import ArchitectureSpec, HParam, register


class CNNRegressionNet(nn.Module):
    """Identical to the original NFTool CNN (kept for checkpoint compatibility)."""

    def __init__(self, freq_bins, hidden_dim=128, layer_size=64, output_dim=1, dropout=0.2):
        super().__init__()
        if freq_bins < 10:
            raise ValueError(f"Input length ({freq_bins}) too short for CNN. Must be >= 10.")
        self.cnn = nn.Sequential(
            nn.Conv1d(1, layer_size, kernel_size=5, padding=2),          # keeps length
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(layer_size, layer_size * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(layer_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.head(self.cnn(x))


class CNNSpec(ArchitectureSpec):
    name = "CNN"
    expects_channel_dim = True
    # The old code required >= 16 features for the CNN data path; enforce it
    # centrally rather than in two different places.
    min_input_dim = 16

    def build(self, input_dim: int, output_dim: int, hp: dict[str, Any]) -> nn.Module:
        return CNNRegressionNet(
            freq_bins=input_dim,
            hidden_dim=int(hp["hidden_dim"]),
            layer_size=int(hp["layer_size"]),
            dropout=float(hp.get("dropout", 0.2)),
            output_dim=output_dim,
        )

    def hyperparameter_space(self) -> list[HParam]:
        # Defaults mirror the original GUI ranges for the "CNN" path.
        return [
            HParam("layer_size", "int", low=1, high=1024, default=64,
                   label="Conv Width"),
            HParam("hidden_dim", "int", low=1, high=100, default=64,
                   label="Hidden Dim"),
            HParam("dropout", "float", low=0.0, high=0.0, default=0.0,
                   label="Dropout"),
        ]


register(CNNSpec())
