import os
import torch
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

def analyze_optuna_study(study, output_dir, run_id):
    """
    Generates diagnostic plots for an Optuna study.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import optuna.visualization as vis
        # 1. Optimization History
        fig = vis.plot_optimization_history(study)
        fig.write_image(os.path.join(output_dir, "optuna_optimization_history.png"))
        
        # 2. Parameter Importance
        if len(study.trials) > 1:
            fig = vis.plot_param_importances(study)
            fig.write_image(os.path.join(output_dir, "optuna_param_importances.png"))
            
        # 3. Parallel Coordinate
        fig = vis.plot_parallel_coordinate(study)
        fig.write_image(os.path.join(output_dir, "optuna_parallel_coordinate.png"))
        
    except ImportError:
        print("Warning: optuna.visualization or kaleido not installed. Skipping plots.")
    except Exception as e:
        print(f"Error generating Optuna plots: {e}")

def generate_regression_plots(y_true, y_pred, output_dir):
    """
    Generates regression-specific plots (Pred vs Actual, Error Histogram).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import matplotlib.pyplot as plt
        # 1. Predicted vs Actual
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, color='#3b82f6')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Predicted vs Actual')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "r2_pred_vs_actual.png"))
        plt.close()
        
        # 2. Error Histogram
        errors = y_true - y_pred
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, color='#3b82f6', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "r2_error_histogram.png"))
        plt.close()
    except ImportError:
        print("Warning: matplotlib not installed. Skipping regression plots.")
