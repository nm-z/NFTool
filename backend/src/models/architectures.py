"""Neural network architectures used by the training service.

This module provides small building blocks and full models used for regression
on 1D signals: a residual convolutional block, a feed-forward regression net,
and a CNN-based regression network.
"""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """1D residual block with two convolutional layers and an optional shortcut.

    The block preserves temporal dimensionality when `stride == 1` and will
    adjust the number of channels via the shortcut when `in_channels` differs
    from `out_channels` or when `stride != 1`.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=1, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        """Compute forward pass for the residual block."""
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class RegressionNet(nn.Module):
    """Simple fully-connected regression network.

    Builds a sequence of Linear -> ReLU (-> Dropout) blocks followed by a final
    Linear layer that outputs a single scalar.
    """

    def __init__(self, input_size, layers, dropout=0.0):
        super().__init__()
        net = []
        last = input_size
        for size in layers:
            net.append(nn.Linear(last, size))
            net.append(nn.ReLU())
            if dropout > 0:
                net.append(nn.Dropout(dropout))
            last = size
        net.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        """Run the input through the MLP and return a single-value prediction."""
        return self.net(x)


class CNNRegressionNet(nn.Module):
    """CNN-based regression model for 1D inputs.

    The model stacks residual 1D blocks, applies global pooling (avg & max),
    concatenates pooled features, and passes them through a small MLP head.
    """

    def __init__(self, freq_bins, conv_layers=None, hidden_dim=128, dropout=0.2):
        """
        Args:
            freq_bins: Number of input features (length of 1D signal).
            conv_layers: List of dicts describing conv blocks. Example:
                [
                    {'out_channels': 32, 'kernel': 5, 'pool': 2},
                    ...
                ]
            hidden_dim: Size of the first fully connected layer after CNN features.
            dropout: Dropout rate for the head.
        """
        super().__init__()
        if freq_bins < 8:
            raise ValueError(f"Input length ({freq_bins}) too short for CNN.")

        if conv_layers is None:
            conv_layers = [
                {"out_channels": 32, "kernel": 5, "pool": 2},
                {"out_channels": 64, "kernel": 3, "pool": 2},
                {"out_channels": 128, "kernel": 3, "pool": 2},
            ]

        layers = []
        in_channels = 1
        for layer_cfg in conv_layers:
            out_channels = layer_cfg["out_channels"]
            kernel = layer_cfg["kernel"]
            stride = layer_cfg.get("stride", 1)

            layers.append(
                ResidualBlock1D(
                    in_channels, out_channels, kernel_size=kernel, stride=stride
                )
            )
            if layer_cfg.get("pool"):
                layers.append(nn.MaxPool1d(layer_cfg["pool"]))
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.flatten = nn.Flatten()

        # Head input size is 2 * out_channels because we concatenate avg and max pooling
        self.head = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        """Compute forward pass through CNN backbone and head, returning scalar."""
        features = self.cnn(x)
        avg_f = self.avg_pool(features)
        max_f = self.max_pool(features)
        combined = torch.cat([avg_f, max_f], dim=1)
        flat = self.flatten(combined)
        return self.head(flat)


def model_factory(model_choice, input_size, config, device):
    """Create and return a model instance moved to `device`.

    Args:
        model_choice: 'NN' for `RegressionNet` or 'CNN' for `CNNRegressionNet`.
        input_size: Integer or tuple describing input shape. For CNNs the
            frequency bin count is expected at index 2 if a tuple is provided.
        config: Dictionary with model hyperparameters.
        device: Torch device to `.to(device)`.
    """
    if model_choice == "NN":
        return RegressionNet(
            input_size=input_size, layers=config["layers"], dropout=config["dropout"]
        ).to(device)
    elif model_choice == "CNN":
        # Frequency bins is the feature count
        freq_bins = input_size[2] if isinstance(input_size, tuple) else input_size
        return CNNRegressionNet(
            freq_bins=freq_bins,
            conv_layers=config.get("conv_layers"),
            hidden_dim=config.get("hidden_dim", 128),
            dropout=config.get("dropout", 0.2),
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_choice: {model_choice}")
