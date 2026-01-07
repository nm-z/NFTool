import os
import sys
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.gui.windows import select_file, prompt_initial_settings
from src.data.processing import compute_dataset_snr_from_files
from src.training.engine import train_model, train_cnn_model, run_optimization, Objective
from src.utils.reporting import analyze_optuna_study

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🟢 GPU Detected: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "🔴 Using CPU")

    # 1. Get Settings from GUI
    settings = prompt_initial_settings()
    if settings is None: return
    
    (load_existing_model, inference_mode, seed, model_choice, patience,
     train_ratio, val_ratio, test_ratio, optuna_trials, target_r2,
     time_limit, optimizers, n_layers_min, n_layers_max, l_size_min, l_size_max,
     lr_min, lr_max, drop_min, drop_max, h_dim_min, h_dim_max, a_min, a_max, k_folds) = settings

    # 2. Load Data
    predictor_file = select_file("Select Predictor CSV")
    target_file = select_file("Select Target CSV")
    if not predictor_file or not target_file: return

    snr, snr_db = compute_dataset_snr_from_files(predictor_file, target_file)
    print(f"📊 Dataset SNR: {snr:.2f} ({snr_db:.2f} dB)")

    df_X = pd.read_csv(predictor_file, header=None)
    df_y = pd.read_csv(target_file, header=None).dropna()
    min_len = min(len(df_X), len(df_y))
    X_raw, y_raw = df_X.iloc[:min_len].values, df_y.iloc[:min_len].values.flatten()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_raw, test_size=test_ratio, random_state=seed)
    val_rel = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_rel, random_state=seed)

    # 4. Optuna Objective
    params = {
        'optimizers': optimizers, 'lr_min': lr_min, 'lr_max': lr_max,
        'drop_min': drop_min, 'drop_max': drop_max, 'n_layers_min': n_layers_min,
        'n_layers_max': n_layers_max, 'l_size_min': l_size_min, 'l_size_max': l_size_max,
        'h_dim_min': h_dim_min, 'h_dim_max': h_dim_max
    }
    obj = Objective(model_choice, X_train, y_train, X_val, y_val, device, patience, params)

    # 5. Run Optimization
    print(f"🚀 Starting {model_choice} Optimization Trials...")
    study = run_optimization(f"NFTool_{model_choice}", optuna_trials, time_limit, obj)
    
    # 6. Report & Save
    report_dir = os.path.join("Training Reports", timestamp)
    os.makedirs(report_dir, exist_ok=True)
    df_trials = analyze_optuna_study(study, report_dir, timestamp)
    
    print(f"🏆 Best Trial R²: {study.best_trial.user_attrs.get('r2', 'N/A')}")
    print(f"📁 Results saved to: {report_dir}")

if __name__ == "__main__":
    main()
