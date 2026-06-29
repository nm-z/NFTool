"""Data loading + splitting, decoupled from any architecture.

Mirrors the original NFTool setup (read two headerless CSVs, align rows,
StandardScaler, train/val/test split). One honest knob is exposed that the
monolith did not have: ``fit_scaler_on_train_only``. The original fit the
scaler on the *entire* dataset before splitting (test-set leakage); we keep that
as the default so numbers stay comparable, but allow the correct behaviour too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .engine import SplitData


@dataclass
class LoadedDataset:
    data: SplitData
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    input_dim: int


def load_csv_dataset(
    predictor_file: str,
    target_file: str,
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    fit_scaler_on_train_only: bool = False,
) -> LoadedDataset:
    df_X = pd.read_csv(predictor_file, header=None)
    df_y = pd.read_csv(target_file, header=None)
    n = min(len(df_X), len(df_y))
    df_X, df_y = df_X.iloc[:n], df_y.iloc[:n]
    combined = pd.concat([df_X, df_y], axis=1).dropna()
    X_raw = combined.iloc[:, :-1].values
    y = combined.iloc[:, -1].values
    input_dim = X_raw.shape[1]

    scaler = StandardScaler()
    if not fit_scaler_on_train_only:
        X = scaler.fit_transform(X_raw)        # original behaviour (leaky but comparable)

    # carve off test, then split temp into train/val (same recipe as the monolith)
    if fit_scaler_on_train_only:
        X = X_raw
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=True, random_state=random_state
    )
    val_rel = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_rel, shuffle=True, random_state=random_state
    )

    if fit_scaler_on_train_only:
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    return LoadedDataset(
        data=SplitData(X_train, y_train, X_val, y_val),
        X_test=X_test, y_test=y_test, scaler=scaler, input_dim=input_dim,
    )
