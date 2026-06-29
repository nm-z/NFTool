"""MLP architecture — a faithful re-expression of the original ``RegressionNet``.

Behaviour is byte-for-byte the same network the monolith built: a stack of
``Linear -> ReLU -> (Dropout if dropout>0)`` blocks followed by a final linear
projection. The ``nn.Module`` keeps the class name *and* the ``net.*`` parameter
keys so checkpoints saved by the old script still load unchanged.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .base import ArchitectureSpec, HParam, register


class RegressionNet(nn.Module):
    """Identical to the original NFTool MLP (kept for checkpoint compatibility)."""

    def __init__(self, input_size, layers, dropout=0.0, output_dim=1):
        super().__init__()
        net = []
        last = input_size
        for size in layers:
            net.append(nn.Linear(last, size))
            net.append(nn.ReLU())
            if dropout > 0:
                net.append(nn.Dropout(dropout))
            last = size
        net.append(nn.Linear(last, output_dim))   # output layer
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class MLPSpec(ArchitectureSpec):
    name = "MLP"                     # was selected as "NN" in the old GUI
    expects_channel_dim = False

    def build(self, input_dim: int, output_dim: int, hp: dict[str, Any]) -> nn.Module:
        layers = [int(hp["layer_size"])] * int(hp["num_layers"])
        return RegressionNet(
            input_size=input_dim,
            layers=layers,
            dropout=float(hp.get("dropout", 0.0)),
            output_dim=output_dim,
        )

    def hyperparameter_space(self) -> list[HParam]:
        # Defaults mirror the original GUI ranges for the "NN" path.
        return [
            HParam("num_layers", "int", low=1, high=100, default=2,
                   label="Number of Layers"),
            HParam("layer_size", "int", low=1, high=1024, default=64,
                   label="Layer Size"),
            HParam("dropout", "float", low=0.0, high=0.0, default=0.0,
                   label="Dropout"),
        ]


register(MLPSpec())
