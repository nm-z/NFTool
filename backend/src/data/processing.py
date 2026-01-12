import numpy as np
import pandas as pd
import torch


def load_dataset(path: str):
    return pd.read_csv(path, header=None)


def preprocess_for_cnn(X_np):
    # Expecting (samples, features)
    if len(X_np.shape) == 2:
        # Add channel dimension: (samples, 1, features)
        X_np = np.expand_dims(X_np, axis=1)
    return torch.tensor(X_np, dtype=torch.float32)


def compute_dataset_snr_from_files(predictor_file: str, target_file: str):
    """
    Computes Signal-to-Noise Ratio (SNR) for the entire dataset.
    Signals are predictors, Noise is the residual from a basic linear fit or standard deviation.
    """
    df_X = load_dataset(predictor_file)
    df_y = load_dataset(target_file).dropna()

    min_len = min(len(df_X), len(df_y))
    X = df_X.iloc[:min_len].values
    y = df_y.iloc[:min_len].values.flatten()

    # Basic SNR estimate: Mean of signal power / Variance of signal
    # Here we treat the mean absolute value of predictors as signal strength
    signal_power = np.mean(np.square(X))
    noise_power = np.var(X)

    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0.0
    correlation = np.corrcoef(X.mean(axis=1), y)[0, 1]

    return float(snr), float(correlation)
