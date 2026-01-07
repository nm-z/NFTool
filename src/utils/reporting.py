import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import scipy.stats as stats

def scatter_fit(ax, x, y, title, color):
    ax.scatter(x, y, alpha=0.6, color=color, label="Data")
    mn, mx = x.min(), x.max()
    ax.plot([mn, mx], [mn, mx], linestyle="--", color="k", label="y = x")
    r2 = r2_score(x, y)
    ax.set_title(f"{title} R² = {r2:.3f}")
    ax.set_xlabel("Target")
    ax.set_ylabel("Output")
    ax.legend()
    ax.grid(True)

def generate_html_report(report_dir, timestamp, config_data, df_trials, metrics):
    # Logic to build the large HTML string from NFTool_V3_071125A.py (lines 1901-1981)
    # This will be refactored to use a template or cleaner string formatting.
    pass

def analyze_optuna_study(study, report_dir, timestamp):
    df_trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration"))
    df_trials.rename(columns={col: col.replace("user_attrs_", "") for col in df_trials.columns if col.startswith("user_attrs_")}, inplace=True)
    df_trials.rename(columns={col: col.replace("params_", "") for col in df_trials.columns if col.startswith("params_")}, inplace=True)
    df_trials.rename(columns={"value": "val_loss"}, inplace=True)
    
    csv_path = os.path.join(report_dir, f"optuna_trials_{timestamp}.csv")
    df_trials.to_csv(csv_path, index=False)
    return df_trials

