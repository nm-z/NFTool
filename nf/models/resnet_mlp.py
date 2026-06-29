"""ResidualMLP — a brand-new architecture added purely via the registry.

This file exists to prove the extension path end to end: it was NOT part of the
original NFTool. Adding it required touching nothing else — no edits to the
engine, the Optuna objective, the data pipeline, or the (hypothetical) UI. The
engine discovers it, the /architectures schema lists it, and it trains.

It is a small fully-connected net with skip connections: each block computes
``x + dropout(relu(linear(x)))`` so the input dimension is preserved and
gradients have a short path. Different forward() from the plain MLP, so it also
demonstrates that the interface is not MLP-shaped.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .base import ArchitectureSpec, HParam, register


class _ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_blocks, dropout=0.0, output_dim=1):
        super().__init__()
        self.stem = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[_ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.head(self.blocks(nn.functional.relu(self.stem(x))))


class ResidualMLPSpec(ArchitectureSpec):
    name = "ResNetMLP"
    expects_channel_dim = False

    def build(self, input_dim: int, output_dim: int, hp: dict[str, Any]) -> nn.Module:
        return ResidualMLP(
            input_dim=input_dim,
            hidden_dim=int(hp["hidden_dim"]),
            n_blocks=int(hp["n_blocks"]),
            dropout=float(hp.get("dropout", 0.0)),
            output_dim=output_dim,
        )

    def hyperparameter_space(self) -> list[HParam]:
        return [
            HParam("hidden_dim", "int", low=8, high=512, default=64,
                   label="Hidden Width"),
            HParam("n_blocks", "int", low=1, high=8, default=2,
                   label="Residual Blocks"),
            HParam("dropout", "float", low=0.0, high=0.5, default=0.0,
                   label="Dropout"),
        ]


register(ResidualMLPSpec())
