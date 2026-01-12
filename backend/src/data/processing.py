import torch
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import RidgeCV

def load_dataset(file_path):
    """
    Loads dataset from CSV, Parquet, or JSON.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.csv':
        # Many datasets in this project don't have headers
        return pd.read_csv(file_path, header=None)
    elif ext == '.parquet':
        return pd.read_parquet(file_path)
    elif ext == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def preprocess_for_cnn(X_np):
    X = torch.tensor(X_np, dtype=torch.float32)
    if X.ndim == 2:
        return X.unsqueeze(1)
    elif X.ndim == 3 and X.shape[1] == 1:
        return X
    else:
        raise ValueError(f"Unexpected shape in preprocess_for_cnn: {X.shape}")

def to_2d_numpy(tensor):
    np_array = tensor.detach().cpu().numpy()
    if np_array.ndim == 3:
        np_array = np_array.reshape(np_array.shape[0], -1)
    return np_array

def compute_dataset_snr_from_files(predictor_file, target_file, ridge_eps: float = 1e-1):
    """
    Computes a linear SNR estimate using RidgeCV for robust regularization.
    NOTE: In high-dimensional settings (N < P), even RidgeCV may report 
    optimistic SNR.
    """
    try:
        df_X = load_dataset(predictor_file)
        df_y = load_dataset(target_file).dropna()
        
        min_len = min(len(df_X), len(df_y))
        X = df_X.iloc[:min_len].values
        y = df_y.iloc[:min_len].values.flatten()
        
        if len(X) < 5: # Need more samples for CV
            return 0.0, -99.0

        # Automated Ridge with Leave-One-Out Cross-Validation
        alphas = np.logspace(-3, 3, 10)
        model = RidgeCV(alphas=alphas)
        model.fit(X, y)
        preds = model.predict(X)

        signal_power = np.var(y)
        # Using residual sum of squares for noise estimate
        noise_power  = np.mean((y - preds) ** 2)
        
        # Avoid division by zero
        snr = signal_power / (noise_power + 1e-12)
        snr_db = 10 * np.log10(snr)

        return snr, snr_db

    except Exception as e:
        print(f"Error computing SNR: {e}")
        return 0.0, -99.0
