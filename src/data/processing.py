import torch
import pandas as pd
import numpy as np

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
    Computes a linear SNR estimate. 
    NOTE: In high-dimensional settings (N < P), this will overfit 
    and report a very high SNR unless strong regularization is used.
    """
    try:
        # Fixed: Use header=None to match training data loading
        df_X = pd.read_csv(predictor_file, header=None)
        df_y = pd.read_csv(target_file, header=None).dropna()
        
        min_len = min(len(df_X), len(df_y))
        X = df_X.iloc[:min_len].values
        y = df_y.iloc[:min_len].values.flatten()
        
        if len(X) < 2:
            return 0.0, -99.0

        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean()

        # Use Ridge regression instead of pinv to handle N < P more gracefully
        # XtX is P x P, which is 3204x3204.
        d = X_centered.shape[1]
        n = X_centered.shape[0]
        
        if n < d:
            # Dual form for N < P: w = X.T @ (X @ X.T + eps*I)^-1 @ y
            # (X @ X.T) is N x N (108x108), much faster.
            K = X_centered @ X_centered.T
            w_dual = np.linalg.solve(K + ridge_eps * np.eye(n), y_centered)
            preds = K @ w_dual + y.mean()
        else:
            XtX = X_centered.T @ X_centered
            Xty = X_centered.T @ y_centered
            XtX_reg = XtX + ridge_eps * np.eye(d)
            w = np.linalg.solve(XtX_reg, Xty)
            preds = X_centered @ w + y.mean()

        signal_power = np.var(y)
        noise_power  = np.mean((y - preds) ** 2)
        
        # Avoid division by zero
        snr = signal_power / (noise_power + 1e-12)
        snr_db = 10 * np.log10(snr)

        return snr, snr_db

    except Exception as e:
        print(f"❌ Error computing SNR: {e}")
        return 0.0, -99.0
