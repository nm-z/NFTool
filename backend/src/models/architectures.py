import torch
import torch.nn as nn

class RegressionNet(nn.Module):
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
        return self.net(x)

class CNNRegressionNet(nn.Module):
    def __init__(self, freq_bins, conv_layers=None, hidden_dim=128, dropout=0.2):
        """
        Args:
            freq_bins: Number of input features (length of 1D signal).
            conv_layers: List of dicts, e.g., [{'out_channels': 32, 'kernel': 5, 'pool': 2}, ...]
            hidden_dim: Size of the first fully connected layer after CNN features.
            dropout: Dropout rate for the head.
        """
        super().__init__()
        if freq_bins < 8:
            raise ValueError(f"Input length ({freq_bins}) too short for CNN.")

        if conv_layers is None:
            conv_layers = [
                {'out_channels': 32, 'kernel': 5, 'pool': 2},
                {'out_channels': 64, 'kernel': 3, 'pool': 2},
                {'out_channels': 128, 'kernel': 3, 'pool': 2}
            ]

        layers = []
        in_channels = 1
        for layer_cfg in conv_layers:
            out_channels = layer_cfg['out_channels']
            kernel = layer_cfg['kernel']
            padding = kernel // 2
            
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=padding))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            if layer_cfg.get('pool'):
                layers.append(nn.MaxPool1d(layer_cfg['pool']))
            in_channels = out_channels

        # Combine Average and Max pooling for richer features
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
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        features = self.cnn(x)
        avg_f = self.avg_pool(features)
        max_f = self.max_pool(features)
        combined = torch.cat([avg_f, max_f], dim=1)
        flat = self.flatten(combined)
        return self.head(flat)

def model_factory(model_choice, input_size, config, device):
    if model_choice == "NN":
        return RegressionNet(
            input_size=input_size,
            layers=config['layers'],
            dropout=config['dropout']
        ).to(device)
    elif model_choice == "CNN":
        # Frequency bins is the feature count
        freq_bins = input_size[2] if isinstance(input_size, tuple) else input_size
        return CNNRegressionNet(
            freq_bins=freq_bins,
            conv_layers=config.get('conv_layers'),
            hidden_dim=config.get('hidden_dim', 128),
            dropout=config.get('dropout', 0.2)
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_choice: {model_choice}")
