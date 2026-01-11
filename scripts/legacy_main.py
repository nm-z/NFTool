import os
import sys
import logging
import argparse
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

from src.data.processing import compute_dataset_snr_from_files
from src.training.engine import train_model, train_cnn_model, run_optimization, Objective
from src.utils.reporting import analyze_optuna_study

# Configuration
BASE_WORKSPACE = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_WORKSPACE, "Logs")
REPORTS_DIR = os.path.join(BASE_WORKSPACE, "Training Reports")
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "main.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("nftool-cli")

def log_print(*args, **kwargs):
    msg = " ".join(map(str, args))
    logger.info(msg)

def run_cli():
    parser = argparse.ArgumentParser(description="NFTool CLI - Training Engine")
    parser.add_argument("--predictors", type=str, required=True, help="Path to predictors CSV")
    parser.add_argument("--targets", type=str, required=True, help="Path to targets CSV")
    parser.add_argument("--model", type=str, choices=["NN", "CNN"], default="NN", help="Model type")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    device = torch.device(args.device)
    log_print(f"🚀 Starting {args.model} Optimization on {device}...")

    # 1. Load and Scale Data
    snr, snr_db = compute_dataset_snr_from_files(args.predictors, args.targets)
    log_print(f"📊 Dataset SNR: {snr:.2f} ({snr_db:.2f} dB)")

    df_X = pd.read_csv(args.predictors, header=None)
    df_y = pd.read_csv(args.targets, header=None).dropna()
    min_len = min(len(df_X), len(df_y))
    X_raw, y_raw = df_X.iloc[:min_len].values, df_y.iloc[:min_len].values.flatten()
    
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.15, random_state=args.seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=args.seed)

    # 3. Setup Optimization
    params = {
        'optimizers': ["AdamW", "Adam"], 
        'lr_min': 1e-4, 'lr_max': 1e-2,
        'drop_min': 0.0, 'drop_max': 0.5, 
        'n_layers_min': 1, 'n_layers_max': 5, 
        'l_size_min': 64, 'l_size_max': 512,
        'h_dim_min': 32, 'h_dim_max': 256
    }
    
    obj = Objective(args.model, X_train, y_train, X_val, y_val, device, args.patience, params)

    # 4. Run Optimization
    study = run_optimization(f"NFTool_CLI_{args.model}", args.trials, None, obj)
    
    # 5. Report & Save
    report_dir = os.path.join(REPORTS_DIR, f"CLI_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    # Save scalers
    joblib.dump(scaler_X, os.path.join(report_dir, "scaler_x.pkl"))
    joblib.dump(scaler_y, os.path.join(report_dir, "scaler_y.pkl"))
    
    analyze_optuna_study(study, report_dir, timestamp)
    
    log_print(f"🏆 Best Trial R²: {study.best_trial.user_attrs.get('r2', 'N/A'):.4f}")
    log_print(f"📁 Results saved to: {report_dir}")

if __name__ == "__main__":
    run_cli()
