"""Data processing helpers used by the backend training and inference code.

This module provides small utilities to load CSV datasets, prepare numpy
arrays for convolutional models, and compute a dataset-level SNR estimate.
"""

import numpy as np
import pandas as pd
import torch


def load_dataset(path: str):
    """Read a CSV file from `path` and return a pandas DataFrame.

    The CSV is expected to have no header; `header=None` is used for
    compatibility with the project's dataset files.
    """
    return pd.read_csv(path, header=None)


def preprocess_for_cnn(x_array: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert an array or tensor shaped (samples, features) into a
    torch.FloatTensor shaped (samples, 1, features) suitable for 1D CNNs.
    Handles both numpy arrays and torch tensors.
    """
    # If already a torch tensor, adjust channel dim and ensure float32
    if isinstance(x_array, torch.Tensor):
        if x_array.dim() == 2:
            # Add channel dimension: (samples, 1, features)
            x_array = x_array.unsqueeze(1)
        return x_array.to(dtype=torch.float32)

    # Otherwise assume numpy array
    # Expecting (samples, features)
    if len(x_array.shape) == 2:
        # Add channel dimension: (samples, 1, features)
        x_array = np.expand_dims(x_array, axis=1)
    return torch.tensor(x_array, dtype=torch.float32)


def compute_dataset_snr_from_files(predictor_file: str, target_file: str):
    """
    Compute Signal-to-Noise Ratio (SNR) for the dataset defined by the two
    CSV files. Signals are predictors; noise is treated as predictor variance
    (or residuals from a simple model).
    """
    df_predictors = load_dataset(predictor_file)
    df_targets = load_dataset(target_file).dropna()

    min_len = min(len(df_predictors), len(df_targets))
    predictors = df_predictors.iloc[:min_len].values
    targets = df_targets.iloc[:min_len].values.flatten()

    # Basic SNR estimate: mean signal power / variance of signal.
    signal_power = np.mean(np.square(predictors))
    noise_power = np.var(predictors)

    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0.0
    correlation = np.corrcoef(predictors.mean(axis=1), targets)[0, 1]

    return float(snr), float(correlation)


__all__ = ["compute_dataset_snr_from_files", "load_dataset", "preprocess_for_cnn"]
