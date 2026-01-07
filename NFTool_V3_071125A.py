# --- NFTool Clean Code v1.0 ---

import os
import sys
import time
import random
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchviz
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

import tkinter as tk
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import scipy.stats as stats
import optuna
import traceback  # Add this at the top if not present
from tkinter import Tk, Label, Entry, Checkbutton, BooleanVar, Button, StringVar, ttk
from tkinter import filedialog
import warnings
from optuna.exceptions import ExperimentalWarning
import glob  
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


# Suppress Optuna experimental‐feature warnings
warnings.filterwarnings("ignore", category=ExperimentalWarning)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only=False.*")

# Try importing optuna
try:
    import optuna.visualization.matplotlib
except ImportError:
    optuna = None
    
# -------------------Imports above this line -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set Torch warning suppress
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cfg = {}


# GPU Check
if torch.cuda.is_available():
    print(f"🟢 GPU Available: {torch.cuda.get_device_name(0)}")
else:
    print("🔴 GPU NOT Available: Using CPU")

# Pause training toggle
pause_training = False
def handle_pause_signal(signum, frame):
    global pause_training
    pause_training = not pause_training
    print("⏸️ Training paused. Press again to resume.")

# Flags for execution flow
load_existing_model = False
inference_mode     = False

num_epochs = 200
# --------------------Global report paths (set early for consistency) ------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
report_dir = os.path.join("Training Reports", timestamp)
best_r2_dir = os.path.join(report_dir, "Best R2 Model")
hist_path = os.path.join(report_dir, "optuna_optimization_history.png")
imp_path = os.path.join(report_dir, "optuna_param_importances.png")
loss_plot_path = os.path.join(report_dir, "loss_plot.png")
mae_r2_plot_path = os.path.join(report_dir, "mae_r2_plot.png")
pred_vs_actual_path = os.path.join(report_dir, "pred_vs_actual.png")
error_hist_path = os.path.join(report_dir, "error_histogram.png")
r2_pred_actual_path = os.path.join(report_dir, "r2_pred_vs_actual.png")
r2_error_hist_path = os.path.join(report_dir, "r2_error_histogram.png")
r2_resid_pred_path = os.path.join(report_dir, "r2_resid_vs_pred.png")
r2_qq_plot_path = os.path.join(report_dir, "r2_qq_plot.png")
r2_corr_heatmap_path = os.path.join(report_dir, "r2_corr_heatmap.png")
r2_four_panel_path = os.path.join(report_dir, "r2_scatter_four_panel.png")
heatmap_path = os.path.join(report_dir, "optuna_param_correlation.png")
graph_path = os.path.join(report_dir, "model_architecture_{timestamp}.png")
total_runtime = 0.0  # Initialize default value, will be overridden post-training

# ──────────────────────────────────────────────
# 2) PROMPT FUNCTIONS
# ──────────────────────────────────────────────
def select_file(title):
    root = Tk()
    root.geometry(f"{400}x{400}")
    root.resizable(False, False)
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, filetypes=[("CSV files", "*.csv")])
    root.destroy()
    return path
#---------------------------------------
def prompt_and_load(predictor_title: str, target_title: str):
    predictor_file = filedialog.askopenfilename(
    	title="Select predictor CSV",
    	filetypes=[("CSV files", "*.csv"), ("All files", "*.")]
    )
    target_file = filedialog.askopenfilename(
    	title="Select target CSV",
    	filetypes=[("CSV files", "*.csv"), ("All files", "*.")]
    )
    if not predictor_file or not target_file:
    	print("❌ Missing files; exiting inference.")
    	sys.exit(0)
		
    df_X = pd.read_csv(predictor_file, header=None)
    df_y = pd.read_csv(target_file, header=None)
    if len(df_X) != len(df_y):
        raise ValueError(
            f"Predictor rows ({len(df_X)}) and target rows ({len(df_y)}) must match"
        )
        
    combined = pd.concat([df_X, df_y], axis=1).dropna()


    X_scaled = scaler.fit_transform(combined.iloc[:, :-1].values)
    y_array = combined.iloc[:, -1].values
    X_val = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_array, dtype=torch.float32).unsqueeze(1).to(device)  
 
    input_size = df_X.shape[1]


    return df_X, df_y, input_size, X, y
#------------------------------------------------------------------
def to_2d_numpy(tensor):
    np_array = tensor.detach().cpu().numpy()
    if np_array.ndim == 3:
        np_array = np_array.reshape(np_array.shape[0], -1)
    return np_array
#----------------------------------------------

def compute_dataset_snr_from_files(predictor_file, target_file, ridge_eps: float = 1e-8):

    try:
        X = pd.read_csv(predictor_file).values
        y = pd.read_csv(target_file).values.flatten()
        if len(X) != len(y):
            raise ValueError(
                f"Predictor rows ({len(df_X)}) and target rows ({len(df_y)}) must match"
            )

        # center
        X_centered = X - X.mean(axis=0)
        y_centered = y - y.mean()

        # try pseudoinverse first
        try:
            w = np.linalg.pinv(X_centered) @ y_centered
        except np.linalg.LinAlgError:
            # fallback: ridge‐regularized normal equations
            XtX = X_centered.T @ X_centered
            Xty = X_centered.T @ y_centered
            d = XtX.shape[0]
            # add tiny ridge to diagonal
            XtX_reg = XtX + ridge_eps * np.eye(d)
            w = np.linalg.solve(XtX_reg, Xty)

        # predictions & SNR
        preds = X_centered @ w + y.mean()
        signal_power = np.mean(y ** 2)
        noise_power  = np.mean((y - preds) ** 2)
        snr = signal_power / (noise_power + 1e-12)
        snr_db = 10 * np.log10(snr)

        print("==================================================================")
        print("📈 Dataset SNR Estimate (Linear Baseline)")
        print(f"  SNR (linear): {snr:.4f}")
        print(f"  SNR (dB):     {snr_db:.2f} dB")
        print("==================================================================")

        return snr, snr_db

    except Exception as e:
        print(f"❌ Error computing SNR: {e}")
        return None, None
#-----------------------------------

def prompt_initial_settings():
    result = [None] * 23   

    def on_submit():
   
        global load_existing_model, inference_mode
        val = load_model_var.get()
        load_existing_model = (val != 0)
        inference_mode     = (val == 2)
        
       # 0) SEED
        s = seed_var.get().strip()
        if s == "":
            result[0] = random.randint(1, 99999)
        else:
            try:
                result[0] = int(s)
            except:
                result[0] = random.randint(1, 99999)

        # 1) model_choice
        result[1] = model_choice_var.get()

        # 3) patience
        try:
            p = int(patience_var.get())
            result[2] = max(1, p)
        except:
            result[2] = 100

        # 3, 4, 5) splits
        try:
            tr = float(train_var.get())
            vr = float(val_var.get())
            te = float(test_var.get())
            total = tr + vr + te
            if abs(total - 100.0) > 1e-3:
                tr = (tr / total) * 100.0
                vr = (vr / total) * 100.0
                te = (te / total) * 100.0
            result[3] = tr / 100.0
            result[4] = vr / 100.0
            result[5] = te / 100.0
        except:
            result[3], result[4], result[5] = 0.70, 0.15, 0.15

        # 6) optuna_trials
        try:
            nt = int(n_trials_var.get())
            result[6] = max(1, nt)
        except:
            result[6] = 150

        # 7) target_r2_value
        if target_r2_enabled.get():
            try:
                v = float(target_r2_var.get())
                result[7] = min(max(v, 0.0), 0.999)
            except:
                result[7] = None
        else:
            result[7] = None

        # 8) optuna_time_limit
        if time_limit_enabled.get():
            try:
                tl = int(time_limit_var.get())
                result[8] = max(1, tl) * 60
            except:
                result[8] = 60
        else:
            result[8] = None
            
        # 22) K-Folds
        try:
            kf = int(float(k_folds_var.get()))
            result[22] = max(0, kf)
        except Exception as e:
            print(f"[⚠️ Error parsing K-Folds: {e}] Defaulting to 1")
            result[22] = 3

        # 9) optimizer_choices_bitmask → list
        bm_str = optimizer_bitmask_var.get().strip()
        OPTIMIZER_MAP = {
            '1': "Adam", '2': "AdamW", '3': "Adamax", '4': "SGD",
            '5': "RMSprop", '6': "Rprop", '7': "LBFGS", '8': "Adadelta",
             
        }
        seen = set()
        opts = []
        for ch in bm_str:
            if ch in OPTIMIZER_MAP and ch not in seen:
                opts.append(OPTIMIZER_MAP[ch])
                seen.add(ch)
        if not opts:
            opts = list(OPTIMIZER_MAP.values())
        result[9] = opts

        # ─── New Range Fields (indices 10–17) ───────────────────────────


        result[10] = num_layers_min.get()
        result[11] = num_layers_max.get()
        result[12] = layer_size_min.get()
        result[13] = layer_size_max.get()
        result[14] = learning_rate_min.get()
        result[15] = learning_rate_max.get()
        result[16] = dropout_min.get()
        result[17] = dropout_max.get()
        result[18] = hidden_dim_min.get()
        result[19] = hidden_dim_max.get()
        result[20] = alpha_min.get()
        result[21] = alpha_max.get()
        
       
        root.destroy()
  
        # ────────────────────────────────────────────────────────────────

    # ─── Build the dark window ───────────────────────────────────────
    root = Tk()
    root.title("NFTool: Initial Settings")
    root.geometry("600x1180")   # increased height for the new fields
    root.configure(bg="#404040")
    root.attributes("-topmost", True)
    pad = 5

    entry_bg = "white"
    entry_disabled_bg = "#a0a0a0"
    label_fg = "white"
    text_fg = "black"

    # 0) Load Existing Model? (radio buttons)
    load_model_var = tk.IntVar(value=0)
    tk.Label(root, text="Load Existing Model?", bg="#404040", fg=label_fg).pack(pady=pad)
    rb_frame = tk.Frame(root, bg="#404040"); rb_frame.pack(pady=(0, pad))
    for txt, val in [("No",0), ("Yes",1), ("Inference",2)]:
        tk.Radiobutton(
            rb_frame,
            text=txt,
            variable=load_model_var,
            value=val,
            bg="#404040",
            fg=label_fg,
            selectcolor="green",           # <-- here!
            activebackground="#404040",
            activeforeground=label_fg,
            highlightthickness=0,           # remove focus ring if you like
        ).pack(side=tk.LEFT, padx=(0,10))

    # 1) Random Seed
    Label(root, text="Random Seed (leave blank to randomize):", bg="#404040", fg=label_fg).pack(pady=pad)
    seed_var = StringVar(value="42")
    Entry(root, textvariable=seed_var, justify='center',
          bg=entry_bg, fg=text_fg, disabledbackground=entry_disabled_bg).pack()

    # 2) Model Type
    Label(root, text="Model Type:", bg="#404040", fg=label_fg).pack(pady=pad)
    model_choice_var = StringVar(value="NN")
    ttk.Combobox(root, values=["NN","CNN"],
                 textvariable=model_choice_var,
                 state="readonly", justify='center').pack()

    # 3) Early Stopping Patience
    Label(root, text="Early Stopping Patience:", bg="#404040", fg=label_fg).pack(pady=pad)
    patience_var = StringVar(value="100")
    Entry(root, textvariable=patience_var, justify='center',
          bg=entry_bg, fg=text_fg, disabledbackground=entry_disabled_bg).pack()

    # 4) Train/Val/Test Split (%)
    Label(root, text="Train/Val/Test Split (%)", bg="#404040", fg=label_fg).pack(pady=(20,2))
    Label(root, text="  Train %:", bg="#404040", fg=label_fg).pack(pady=(5,0))
    train_var = StringVar(value="70")
    Entry(root, textvariable=train_var, width=10, justify='center',
          bg=entry_bg, fg=text_fg, disabledbackground=entry_disabled_bg).pack()
    Label(root, text="  Val %:", bg="#404040", fg=label_fg).pack(pady=(5,0))
    val_var = StringVar(value="15")
    Entry(root, textvariable=val_var, width=10, justify='center',
          bg=entry_bg, fg=text_fg, disabledbackground=entry_disabled_bg).pack()
    Label(root, text="  Test %:", bg="#404040", fg=label_fg).pack(pady=(5,0))
    test_var = StringVar(value="15")
    Entry(root, textvariable=test_var, width=10, justify='center',
          bg=entry_bg, fg=text_fg, disabledbackground=entry_disabled_bg).pack()

    # 5) Optuna Trials
    Label(root, text="Optuna Trials:", bg="#404040", fg=label_fg).pack(pady=(20, pad))
    n_trials_var = StringVar(value="150")
    Entry(root, textvariable=n_trials_var, justify='center',
          bg=entry_bg, fg=text_fg, disabledbackground=entry_disabled_bg).pack()

    # 6) Target R²
    target_r2_enabled = BooleanVar(value=False)
    Checkbutton(root, text="Specify Target R²?", variable=target_r2_enabled,
                bg="#404040", fg=label_fg, selectcolor="#404040",
                activebackground="#404040", activeforeground=label_fg).pack(pady=pad)
    Label(root, text="Target R² value (0.0–0.999):", bg="#404040", fg=label_fg).pack(pady=(5,0))
    target_r2_var = StringVar(value="0.95")
    target_r2_entry = Entry(root, textvariable=target_r2_var, justify='center',
                             bg=entry_bg, fg=text_fg,
                             disabledbackground=entry_disabled_bg,
                             state='disabled')
    target_r2_entry.pack()
    target_r2_enabled.trace_add("write",
        lambda *args: target_r2_entry.config(state='normal' if target_r2_enabled.get() else 'disabled')
    )

    # 7) Time Limit
    time_limit_enabled = BooleanVar(value=False)
    Checkbutton(root, text="Enable Time Limit?", variable=time_limit_enabled,
                bg="#404040", fg=label_fg, selectcolor="#404040",
                activebackground="#404040", activeforeground=label_fg).pack(pady=(20,pad))
    Label(root, text="Time Limit (minutes):", bg="#404040", fg=label_fg).pack(pady=(5,0))
    time_limit_var = StringVar(value="60")
    time_limit_entry = Entry(root, textvariable=time_limit_var, justify='center',
                              bg=entry_bg, fg=text_fg,
                              disabledbackground=entry_disabled_bg,
                              state='disabled')
    time_limit_entry.pack()
    time_limit_enabled.trace_add("write",
        lambda *args: time_limit_entry.config(state='normal' if time_limit_enabled.get() else 'disabled')
    )
    
    
# ───#22) K-Folds Entry
    Label(root, text="K-Folds (To enable, CV folds ≥2 ):", bg="#404040", fg=label_fg).pack(pady=(20, pad))
    k_folds_var = StringVar(value="3")
    Entry(root, textvariable=k_folds_var, justify='center',
          bg=entry_bg, fg=text_fg, width=5,  
          disabledbackground=entry_disabled_bg).pack()


    # ─── New Range Inputs (10 fields) ─────────────────────────────


    Label(root, text="Hyperparameter Ranges:", bg="#404040", fg=label_fg).pack(pady=(20,pad))
    range_frame = tk.Frame(root, bg="#404040")
    range_frame.pack(padx=20, pady=(0,pad), anchor='center')
    for col in range(4):
        range_frame.grid_columnconfigure(col, weight=1)

    num_layers_min    = tk.IntVar(value=1)
    num_layers_max    = tk.IntVar(value=100)
    layer_size_min    = tk.IntVar(value=1)
    layer_size_max    = tk.IntVar(value=1024)
    learning_rate_min = tk.DoubleVar(value=0.000001)
    learning_rate_max = tk.DoubleVar(value=0.001)
    dropout_min       = tk.DoubleVar(value=0.0)
    dropout_max       = tk.DoubleVar(value=0.0)
    hidden_dim_min    = tk.DoubleVar(value=1)
    hidden_dim_max    = tk.DoubleVar(value=100)
    alpha_min         = tk.DoubleVar(value=0.0)
    alpha_max         = tk.DoubleVar(value=0.0)
    fields = [
        ("Min # Layers",       num_layers_min),
        ("Max # Layers",       num_layers_max),
        ("Min Layer Size",     layer_size_min),
        ("Max Layer Size",     layer_size_max),
        ("Min Learning Rate",  learning_rate_min),
        ("Max Learning Rate",  learning_rate_max),
        ("Min Dropout",        dropout_min),
        ("Max Dropout",        dropout_max),
        ("Min # Hidden Dim",   hidden_dim_min),
        ("Max # Hidden Dim",   hidden_dim_max),
        ("Min # Alpha",        alpha_min),
        ("Max # Alpha",        alpha_max),
            ]
    
    for idx, (txt, var) in enumerate(fields):
        r = idx // 4 
        c = idx % 4
        Label(range_frame, text=txt + ":", bg="#404040", fg=label_fg).grid(
            row=r * 2, column=c, sticky='ew', padx=pad, pady=(0,pad)
        )
        Entry(range_frame, textvariable=var, justify='center',
              bg=entry_bg, fg=text_fg
        ).grid(row=r * 2 + 1, column=c, padx=pad, pady=(0,pad))

    # 8) Optimizer Bitmask legend & entry
    Label(root,
          text="Optimizer Bitmask Legend:\n"
               " 1=Adam 2=AdamW 3=Adamax\n"
               " 4=SGD  5=RMSprop 6=Rprop\n"
               " 7=LBFGS 8=Adadelta",
          bg="#404040", fg=label_fg, justify='left'
    ).pack(pady=(20,2))
    optimizer_bitmask_var = StringVar(value="1")
    Entry(root, textvariable=optimizer_bitmask_var, justify='center',
          bg=entry_bg, fg=text_fg, disabledbackground=entry_disabled_bg
    ).pack()

    # Submit button
    Button(root, text="OK", command=on_submit, width=10, height= 2,
           bg="green", fg="black", activebackground="#d0d0d0"
    ).pack(pady=20)

    # Center window
    root.eval('tk::PlaceWindow . center')
    root.mainloop()



    # Extended to include k_folds
    return (load_existing_model, inference_mode) + tuple(result)



# ──────────────────────────────────────────────────────────────────────
def main():
    import torch
    from tkinter import filedialog
    import pandas as pd
    import torch.nn.functional as F
    from sklearn.metrics import r2_score
    global load_existing_model, inference_mode, SEED, model_choice, patience
    global train_ratio, val_ratio, test_ratio, optuna_trials, target_r2_value
    global optuna_time_limit, optimizer_choices, num_layers_min, num_layers_max
    global layer_size_min, layer_size_max, learning_rate_min, learning_rate_max
    global hidden_dim_min, hidden_dim_max, dropout_min, dropout_max
    global X_train, hidden_dim, sampler,  alpha_min,  alpha_max, k_folds
    


    # 1) Prompt GUI and unpack EVERY value, including the two flags up front
    (
        load_existing_model,  # True if “Yes” or “Inference”
        inference_mode,       # True only if “Inference”
        SEED,
        model_choice,
        patience,
        train_ratio, val_ratio, test_ratio,
        optuna_trials,
        target_r2_value,
        optuna_time_limit,
        optimizer_choices,
        num_layers_min, num_layers_max,
        layer_size_min, layer_size_max,
        learning_rate_min, learning_rate_max,
        dropout_min, dropout_max,
        hidden_dim_min, hidden_dim_max, alpha_min,  alpha_max,
        k_folds
    ) = prompt_initial_settings()

    # 2)_____________________________________________________________
    
    sampler = optuna.samplers.TPESampler(seed=SEED) if optuna else None
    if not inference_mode:
        print("🚧 Training setup — implement training logic here.")
        os.makedirs(report_dir, exist_ok=True)
        os.makedirs(best_r2_dir, exist_ok=True)
        # ✅ Initialize K-Fold if requested
        if k_folds >= 2:
            print(f"📊 K-Fold Cross-Validation Enabled (k = {k_folds})")
            
        else:
            print("🚀 Standard train/val/test split (k-fold disabled)")
            

    if load_existing_model or inference_mode:
        model, scaler, device, tk_root = build_model()
        
   

    if inference_mode:
        predictor_file = filedialog.askopenfilename(
            title="Select predictor CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.")]
        )
        target_file = filedialog.askopenfilename(
            title="Select target CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.")]
        )
        if not predictor_file or not target_file:
            print("❌ Missing files; exiting inference.")
            sys.exit(0)
            
        df_X = pd.read_csv(predictor_file, header=None)
        df_y = pd.read_csv(target_file, header=None)
        if len(df_X) != len(df_y):
            raise ValueError(
                f"Predictor rows ({len(df_X)}) and target rows ({len(df_y)}) must match"
            )
        combined = pd.concat([df_X, df_y], axis=1).dropna()
        X_scaled = scaler.fit_transform(combined.iloc[:, :-1].values)
        y_array = combined.iloc[:, -1].values
        X_val = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_array, dtype=torch.float32).unsqueeze(1).to(device)

        model.eval()
        with torch.no_grad():
            preds = model(X_val)
            scaled_y = y_val.cpu().numpy().ravel()
            scaled_p = preds
            
            r2_scaled = r2_score(scaled_y, scaled_p)

        snr, snr_db = compute_dataset_snr_from_files(predictor_file, target_file)
        mse = F.mse_loss(preds, y_val).item()
        mae = F.l1_loss(preds, y_val).item()
        r2 = r2_score(y_val.cpu().numpy().ravel(), preds.cpu().numpy().ravel())
        print(" ==================================================================")
        print("🔍 Inference Results")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R² : {r2:.4f}")
        print(f"  SNR (linear): {snr:.4f}")
        print(f"  SNR (dB): {snr_db:.2f} dB")
        print(" ==================================================================")

        sys.exit(0)
  
   
#---------------------------------------------------------------------------

def build_model():
    
    tk_root = Tk()
    tk_root.withdraw()
    tk_root.attributes('-topmost', True)
    
    # 1) Select checkpoint
    ckpt_path = filedialog.askopenfilename(
        title="Select model checkpoint (.pt/.pth)",
        filetypes=[("PyTorch Model", "*.pt *.pth"), ("All files", "*")]
    )
    if not ckpt_path:
        print("❌ No checkpoint selected; exiting.")
        sys.exit(1)
    print(f"📂 Loaded checkpoint path: {ckpt_path}")
    
    # 2) Auto-detect or select scaler
    pkl_list = glob.glob(os.path.join(os.path.dirname(ckpt_path), "*.pkl"))
    if pkl_list:
        scaler_path = max(pkl_list, key=os.path.getmtime)
        print(f"⚠️ Auto-detected scaler: {scaler_path}")
    else:
        scaler_path = filedialog.askopenfilename(
            title="Select scaler pickle (.pkl)",
            initialdir=os.path.dirname(ckpt_path),
            filetypes=[("Pickle file", "*.pkl"), ("All files", "*")]
        )
        if not scaler_path:
            print("❌ No scaler selected; exiting.")
            sys.exit(1)
        print(f"📂 Loaded scaler path: {scaler_path}")
    
    # 3) Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print("\n✅ Checkpoint contents:")
    for k, v in checkpoint.items():
        if isinstance(v, dict):
            print(f"  - {k}: dict keys={list(v.keys())}")
        elif hasattr(v, 'shape'):
            print(f"  - {k}: tensor shape={tuple(v.shape)}")
        else:
            print(f"  - {k}: {v}")
    
    # 4) Load scaler
    scaler = joblib.load(scaler_path)
    print(f"\n✅ Scaler loaded; params: {scaler.__dict__}")
    
    print("    🧪 Model reconstruction check:")
    print(f" - input_size: {checkpoint.get('input_size')}")
    print(f" - layers: {checkpoint.get('layers')}")
    print(f" - dropout: {checkpoint.get('dropout')}")
    
    # 5) Rebuild model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint.get("model_choice") == "NN":
            # Dynamically infer layer count and size from state_dict
        state_dict = checkpoint["model_state_dict"]
        first_weight = state_dict.get("net.0.weight")
        hidden_size = first_weight.shape[0] if first_weight is not None else 16
        num_layers = sum(1 for k in state_dict if k.startswith("net.") and ".weight" in k) - 1
        model = RegressionNet(
            input_size=checkpoint["input_size"],
            layers=[hidden_size] * num_layers,
            dropout=checkpoint["dropout"]
        ).to(device)
    else:
        model = CNNRegressionNet(
            freq_bins=checkpoint["input_size"],
            hidden_dim=checkpoint.get("hidden_dim"),
            layer_size=checkpoint.get("layer_size"),
            dropout=checkpoint.get("dropout")
        ).to(device)
    model.load_state_dict(checkpoint.get("model_state_dict"))
    model.eval()
    print("\n✅ Model loaded and ready for inference.")
    return model, scaler, device, tk_root


#------------------------------------------------------------
def build_model_training(input_size, config, device, model_choice):
    """
    Build the appropriate model for training based on model_choice.
    - model_choice == "NN" → Standard RegressionNet
    - model_choice == "CNN" → CNNRegressionNet
    """
    if model_choice == "NN":
        model = RegressionNet(
            input_size=input_size,
            layers=config['layers'],
            dropout=config['dropout']
        ).to(device)
    elif model_choice == "CNN":
        freq_bins = input_size[2] if isinstance(input_size, tuple) else input_size
        model = CNNRegressionNet(
            freq_bins=freq_bins,
            hidden_dim=config['hidden_dim'],
            layer_size=config['layer_size'],
            dropout=config['dropout']
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_choice: {model_choice}")
    return model

#-----------------------------------------------------------


def preprocess_for_cnn(X_np):
    X = torch.tensor(X_np, dtype=torch.float32)
    if X.ndim == 2:
        X = X.unsqueeze(1)
        
        return X
    elif X.ndim == 3 and X.shape[1] == 1:
        
        return X
    else:
        raise ValueError(f"Unexpected shape in preprocess_for_cnn: {X.shape}")
# ---- MODEL I/O --------------
 

# --- CNN + MLP Hybrid Support for NFTool ---


class CNNRegressionNet(nn.Module):
    def __init__(self, freq_bins, hidden_dim=128, layer_size=64, output_dim=1, dropout=0.2):
        super().__init__()
        if freq_bins < 10:
            raise ValueError(f"Input length ({freq_bins}) too short for CNN. Must be ≥ 8.")
        self.cnn = nn.Sequential(
            nn.Conv1d(1, layer_size, kernel_size=5, padding=2),  # maintains length
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(layer_size, layer_size * 2, kernel_size=3, padding=1),  # maintains shape better
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.head = nn.Sequential(
            nn.Linear(layer_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        features = self.cnn(x)
        return self.head(features)


# ---- CNN DATA PROCESSING WRAPPER ----
def train_cnn_model(X_train_np, y_train_np,
                    X_val_np, y_val_np,
                    config, device, patience,
                    model_choice, alpha=0.0):
    print(f"🚀 train_cnn_model: Starting — Optimizer={config['optimizer']}, LR={config['lr']:.2e}")

    # Diagnostics before preprocessing
  
    X_train = preprocess_for_cnn(X_train_np).to(device)
    X_val = preprocess_for_cnn(X_val_np).to(device)

    y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(device)
    y_val = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1).to(device)

  
    model = build_model_training(X_train.shape, config, device, model_choice)
    

    try:
        optimizer_class = torch.optim.LBFGS if config['optimizer'] == 'LBFGS' else getattr(torch.optim, config['optimizer'])
        optimizer = optimizer_class(model.parameters(), lr=config['lr'])

        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        counter = 0
        history = {"train": [], "val": [], "r2": [], "mae": []}

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output, y_val)
                preds = val_output.cpu().numpy().flatten()
                targets = y_val.cpu().numpy().flatten()

            if (
                np.isnan(preds).any() or np.isnan(targets).any() or
                np.isinf(preds).any() or np.isinf(targets).any() or
                np.abs(preds).max() > 1e6 or np.abs(targets).max() > 1e6
            ):
                print("⚠️ CNN trial produced NaNs/Infs/extreme values. Skipping trial.")
                return model, float("inf"), history

            r2_val = r2_score(targets, preds)
            mae_val = mean_absolute_error(targets, preds)

            history['train'].append(loss.item())
            history['val'].append(val_loss.item())
            history['r2'].append(r2_val)
            history['mae'].append(mae_val)

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"⛔️ Early stopping at epoch {epoch + 1}")
                    break

        model.load_state_dict(best_model_state)
        return model, best_val_loss, history

    except Exception as e:
        print(f"❌ train_model crashed: {e}")
        return model, float("inf"), {'train': [], 'val': [], 'r2': [], 'mae': []}




# ---- OPTUNA OBJECTIVE FOR CNN ----
# Put this once globally (not inside the function)
best_r2_so_far = [-float("inf")]

def cnn_objective(trial,
                 X_train_np, y_train_np,
                 X_val_np, y_val_np,
                 device, patience,
                 layer_size_min, layer_size_max,
                 hidden_dim_min, hidden_dim_max,
                 dropout_min, dropout_max,
                 learning_rate_min, learning_rate_max,
                 alpha_min, alpha_max,
                 model_choice
):
    trial_start = time.time()  # ⏱️ Start time for this trial

    # ✅ Define config dict from trial suggestions
    config = {
        'optimizer': trial.suggest_categorical("optimizer", optimizer_choices),
        'layer_size': trial.suggest_int("layer_size", layer_size_min, layer_size_max),
        'hidden_dim': trial.suggest_int("hidden_dim", hidden_dim_min, hidden_dim_max),
        'dropout': trial.suggest_float("dropout", dropout_min, dropout_max),
        'lr': trial.suggest_float("lr", learning_rate_min, learning_rate_max, log=True),
        'alpha': trial.suggest_float("alpha", alpha_min, alpha_max)
    }


    model, val_loss, _ = train_cnn_model(X_train_np, y_train_np,
    X_val_np, y_val_np,
    config, device, patience,
    model_choice, alpha=0.0
    )
    

    model.eval()
    with torch.no_grad():
        val_output = model(preprocess_for_cnn(X_val_np).to(device))
        val_preds = val_output.cpu().numpy().flatten()
        val_targets = y_val_np.flatten()

    if (
        np.isnan(val_preds).any() or np.isinf(val_preds).any() or
        np.isnan(val_targets).any() or np.isinf(val_targets).any() or 
        np.abs(val_preds).max() > 1e6 or np.abs(val_targets).max() > 1e6
    ):
        print("⚠️ NaNs/Infs or extreme values detected. Skipping this trial.")
        return float("inf")

    r2_val = r2_score(val_targets, val_preds)
    mae_val = mean_absolute_error(val_targets, val_preds)

    if r2_val > best_r2_so_far[0]:
        best_r2_so_far[0] = r2_val

    duration = time.time() - trial_start
    trial.set_user_attr("r2",           r2_val)
    trial.set_user_attr("mae",          mae_val)
    trial.set_user_attr("val_loss",     val_loss)
    trial.set_user_attr("duration_sec", duration)
    print(f"⏱️ Trial Duration: {duration:.2f} sec")
    print(f"Trial #{trial.number} — R²: {r2_val:.4f}, MAE: {mae_val:.4f}, Loss: {val_loss:.4f},🌟 New Best R²: {best_r2_so_far[0]:.4f}")
    print(f"🌟 New Best R² So Far: {best_r2_so_far[0]:.4f}") 
    
    return val_loss


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
        net.append(nn.Linear(last, 1))  # Output layer
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

# ---- TRAINING FUNCTION ----

def train_model(X_train, y_train, X_val, y_val, input_size, config, device, patience, trial=None, alpha=0.0):
    if config is None:
        print("❌ Config is undefined. Skipping training.")
        return None, float("inf"), {'train': [], 'val': [], 'r2': [], 'mae': []}
         

    # fallback to PyTorch training logic
    model = RegressionNet(input_size, config['layers'], config['dropout']).to(device)

    try:
        optimizer_class = optim.LBFGS if config['optimizer'] == 'LBFGS' else getattr(optim, config['optimizer'])
        optimizer = optimizer_class(model.parameters(), lr=config['lr'])

        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        counter = 0
        history = {'train': [], 'val': [], 'r2': [], 'mae': []}

        pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")

        for epoch in pbar:
            model.train()

            if config['optimizer'] == 'LBFGS':
                def closure():
                    optimizer.zero_grad()
                    output = model(X_train)
                    loss = criterion(output, y_train)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
                loss_val = loss.item()
            else:
                optimizer.zero_grad()
                output = model(X_train)
                loss = criterion(output, y_train)
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
            history['train'].append(loss_val)
            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = criterion(val_output, y_val)
                val_preds = val_output.cpu().numpy().flatten()
                val_targets = y_val.cpu().numpy().flatten()

            if (
                np.isnan(val_preds).any() or np.isnan(val_targets).any() or
                np.isinf(val_preds).any() or np.isinf(val_targets).any() or
                np.abs(val_preds).max() > 1e6 or np.abs(val_targets).max() > 1e6
            ):
                print("⚠️ NaNs/Infs or extreme values detected. Skipping this trial.")
                return model, float("inf"), history

            r2_val = r2_score(val_targets, val_preds)
            mae_val = mean_absolute_error(val_targets, val_preds)

            history['train'].append(loss_val)
            history['val'].append(val_loss.item())
            history['r2'].append(r2_val)
            history['mae'].append(mae_val)

            pbar.set_postfix({
                "Train Loss": f"{loss_val:.4f}",
                "Val Loss": f"{val_loss.item():.4f}",
                "R²": f"{r2_val:.4f}",
                "MAE": f"{mae_val:.4f}"
            })

            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_model_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"⛔ Early stopping triggered at epoch {epoch + 1}")
                    break

        model.load_state_dict(best_model_state)
        
        
        return model, val_loss, history

    except Exception as e:
        print(f"❌ train_model crashed: {e}")
        return model, float("inf"), {'train': [], 'val': [], 'r2': [], 'mae': []}



# ---- OPTUNA TRIAL FUNCTION ----

def run_optuna(
    sampler,
    optuna_time_limit,
    n_trials
):
    try:
        study = optuna.create_study(direction="minimize", sampler=sampler)

        if optuna_time_limit:
            study.optimize(objective, timeout=optuna_time_limit, show_progress_bar=False)
        else:
            study.optimize(objective, n_trials=n_trials)

        return study

    except Exception as e:
        print(f"❌ Optuna failed with error: {e}")
        traceback.print_exc()
        return None


#------------------------------------------------------------------------------
def objective(trial):
    trial_start = time.time()
    
    # 1) Determine seed: use the GUI’s SEED if set, otherwise sample one
    seed = SEED if SEED is not None else np.random.randint(0, 2**31)
    print(f"[Objective] Using seed = {seed}")
    
    # 2) Apply it everywhere
    import random, torch, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark    = False
    
    # 3) Log it for later replay
    trial.set_user_attr("seed", seed)

    
    global best_r2_so_far

    optimizer = trial.suggest_categorical("optimizer", optimizer_choices)
    num_layers = trial.suggest_int("num_layers", num_layers_min, num_layers_max)
    layer_size = trial.suggest_int("layer_size", layer_size_min, layer_size_max)
    dropout    = trial.suggest_float("dropout", dropout_min, dropout_max)
    lr         = trial.suggest_float("lr", learning_rate_min, learning_rate_max, log=True)
    alpha = trial.suggest_float("alpha", alpha_min, alpha_max)
    if model_choice == "NN":
        num_layers = trial.suggest_int("num_layers", num_layers_min, num_layers_max)
    else:  # CNN
        layer_size = trial.suggest_int("layer_size", layer_size_min, layer_size_max)
        hidden_dim = trial.suggest_int("hidden_dim", hidden_dim_min, hidden_dim_max)
                
  
    config = {
        'layers': [layer_size] * num_layers,
        'dropout': dropout,
        'lr': lr,
        'optimizer': optimizer,
        'alpha': alpha
    }

    if model_choice == "NN":
        model, val_loss, history = train_model(
            X_train, y_train, X_val, y_val,
            input_size, config, device, patience, trial=trial
        )
    else:
        model, val_loss, history = train_cnn_model(
            X_train, y_train, X_val, y_val,
            config=config,
            device=device,
            patience=patience,
            model_choice=model_choice,
            alpha=config['alpha']
        )
    
    # 1) Extract the final epoch’s training loss from history
    train_loss = history.get("train", [None])[-1]
    
    # 2) Evaluate on validation set and log metrics
    try:
        model.eval()
        with torch.no_grad():
            val_output  = model(X_val)
            val_preds   = val_output.cpu().numpy().flatten()
            val_targets = y_val.cpu().numpy().flatten()
    
        r2_val  = r2_score(val_targets, val_preds)
        mae_val = mean_absolute_error(val_targets, val_preds)
    
        trial.set_user_attr("r2",         r2_val)
        trial.set_user_attr("mae",        mae_val)
        trial.set_user_attr("val_loss",   val_loss)
        trial.set_user_attr("train_loss", train_loss)
        trial.set_user_attr("duration_sec", time.time() - trial_start)
    
        if r2_val > best_r2_so_far[0]:
            best_r2_so_far[0] = r2_val
    
        print(
            f"Trial #{trial.number} — "
            f"R²: {r2_val:.4f}, MAE: {mae_val:.4f}, "
            f"Val Loss: {val_loss:.4f}, Train Loss: {train_loss:.4f}, "
            f"🌟 New Best R²: {best_r2_so_far[0]:.4f}🌟"
        )
        return val_loss
    
    except Exception as e:
        print(f"⚠️ Trial #{trial.number} failed during evaluation: {e}")
        raise optuna.exceptions.TrialPruned()



#------------------------------------------------------------------------


def analyze_optuna_study(study, report_dir):
    
    print("=============================================================")
    # 1) Build a DataFrame of all trials
    df_trials = study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "duration")
    )
    
    # 2) Strip off the user_attrs_/params_ prefixes
    df_trials.rename(columns={
        col: col.replace("user_attrs_", "")
        for col in df_trials.columns if col.startswith("user_attrs_")
    }, inplace=True)
    df_trials.rename(columns={
        col: col.replace("params_", "")
        for col in df_trials.columns if col.startswith("params_")
    }, inplace=True)
    
    # 3) Rename the main “value” column → “val_loss”
    df_trials.rename(columns={"value": "val_loss"}, inplace=True)
    
    # 4) Drop any accidental duplicate columns (e.g. extra val_loss)
    df_trials = df_trials.loc[:, ~df_trials.columns.duplicated()]
    
    # 5) Move train_loss so it sits immediately after val_loss
    if "train_loss" in df_trials.columns:
        cols = df_trials.columns.tolist()
        i = cols.index("val_loss")
        cols.insert(i+1, cols.pop(cols.index("train_loss")))
        df_trials = df_trials[cols]
    
    # 4) Pick best trial
    best_row = df_trials.sort_values("r2", ascending=False).iloc[0]
    best_r2_trial_n = int(best_row["number"])
    best_r2_trial = next(t for t in study.trials if t.number == best_r2_trial_n)

    print(f"🏆 True Best-R² Trial #{best_r2_trial_n}")
    print(f"  R²          = {best_row['r2']:.4f}")
    for k, v in best_r2_trial.params.items():
        print(f"  {k:12} = {v}")
    print("=============================================================")

    # 5) Save cleaned CSV
    df_trials.sort_values("r2", ascending=False, inplace=True)
    csv_path = os.path.join(report_dir, f"optuna_trials_{timestamp}.csv")
    df_trials.to_csv(csv_path, index=False)
    print(f"📁 Trials CSV saved: {csv_path}")

    return df_trials
    

#--------------------------------------------------------
#----- Ensure everything is indented the if ------------------


def print_script_name():
    # Try __file__; if not present (e.g. interactive), fall back to sys.argv
    try:
        name = os.path.basename(__file__)
    except NameError:
        name = os.path.basename(sys.argv[0]) or "<unknown script>"
    print(f"🖥️ Running script: {name}")



if __name__ == "__main__":
    print_script_name()

    main()


    
# --- SETUP PHASE ----------------------------------------------------- 

# ✅ Load data first
predictor_file = select_file("Select the Predictor CSV file")
target_file = select_file("Select the Target CSV file")
snr, snr_db = compute_dataset_snr_from_files(predictor_file, target_file)

if not predictor_file or not target_file:
    print("❌ File selection cancelled. Exiting.")
    exit()


# ✅ Load and clean data
df_X = pd.read_csv(predictor_file, header=None)
df_y = pd.read_csv(target_file, header=None).dropna()
if len(df_X) != len(df_y):
    raise ValueError(
        f"Predictor rows ({len(df_X)}) and target rows ({len(df_y)}) must match"
    )

# ✅ Ensure alignment
min_len = min(len(df_X), len(df_y))
df_X = df_X.iloc[:min_len]
df_y = df_y.iloc[:min_len]

# This is needed because Python needs parameters names in Python function
X_np = df_X.values


input_size = df_X.shape[1]
combined = pd.concat([df_X, df_y], axis=1).dropna()
scaler = StandardScaler()
X = scaler.fit_transform(combined.iloc[:, :-1].values)
y = combined.iloc[:, -1].values



target_variance = np.var(y)

print(f"📈 Target Variance: {target_variance:.6f}")


# ✅ Data checks
if model_choice == "CNN":
    if X.shape[0] < 10 or X.shape[1] < 16:
        raise ValueError(f"❌ CNN input must be shaped (samples ≥10, features ≥16). Got: {X.shape}")


N = X.shape[0]  # total number of rows


# 1) carve off the TEST set
X_temp_np, X_test_np, y_temp_np, y_test_np = train_test_split(
    X, y,
    test_size=test_ratio,
    shuffle=True,
    random_state=42
)

# 2) split that temp-pool into TRAIN vs VAL
#    compute val’s share of the remaining data
val_rel = val_ratio / (train_ratio + val_ratio)

X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X_temp_np, y_temp_np,
    test_size=val_rel,
    shuffle=True,
    random_state=42
)

for name, arr in [("TRAIN", y_train_np), ("VAL", y_val_np), ("TEST", y_test_np)]:
    print(
        f"{name:>4}: n={len(arr)}  "
        f"mean={arr.mean():.4f}  "
        f"min={arr.min():.4f}  "
        f"max={arr.max():.4f}"
    )



# ✅ Optional: sanity check

to_tensor = lambda x: torch.tensor(x, dtype=torch.float32)
X_train, y_train = to_tensor(X_train_np), to_tensor(y_train_np).unsqueeze(1)
X_val, y_val = to_tensor(X_val_np), to_tensor(y_val_np).unsqueeze(1)
X_test, y_test = to_tensor(X_test_np), to_tensor(y_test_np).unsqueeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, y_train = X_train.to(device), y_train.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)


#------------------------ ✅ Get config  ----------------------

n_trials = optuna_trials 
target_loss = (1 - target_r2_value) * np.var(y) if target_r2_value else None

# --- Optuna Block for CNN and NN (preserving original logic) ---
if model_choice == "CNN":
    result = optuna.create_study(direction="minimize", sampler=sampler)
    result.optimize(lambda trial: cnn_objective(trial,
        X_train_np, y_train_np,
        X_val_np, y_val_np,
        device, patience,
        layer_size_min, layer_size_max,
        hidden_dim_min, hidden_dim_max,
        dropout_min, dropout_max,
        learning_rate_min, learning_rate_max,
        alpha_min, alpha_max, model_choice
    ),
    timeout=optuna_time_limit if optuna_time_limit else None,
    n_trials=None if optuna_time_limit else optuna_trials
)
    study = result
    best_params = study.best_params
    config = {
        'hidden_dim': best_params['hidden_dim'],
        'dropout': best_params['dropout'],
        'lr': best_params['lr'],
        'optimizer': best_params['optimizer']
    }

else:
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda trial: objective(trial),
    timeout=optuna_time_limit if optuna_time_limit else None,
    n_trials=None if optuna_time_limit else optuna_trials
    )
    best_params = study.best_params
    config = {
        'layers': [best_params['layer_size']] * best_params['num_layers'],
        'dropout': best_params['dropout'],
        'lr': best_params['lr'],
        'optimizer': best_params['optimizer'],
        'alpha': best_params.get('alpha', 0.0)
    }

# from optuna.visualization import plot_param_importances, plot_optimization_history

try:
    ax1 = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig1 = ax1.figure
    fig1.set_size_inches(10, 6)
    fig1.tight_layout()
    fig1.savefig(hist_path)
    plt.close(fig1)
except Exception as e:
    print(f"⚠️ Could not save optimization history plot: {e}")

try:
    ax2 = optuna.visualization.matplotlib.plot_param_importances(study)
    fig2 = ax2.figure
    fig2.set_size_inches(10, 6)
    fig2.tight_layout()
    fig2.savefig(imp_path)
    plt.close(fig2)
except Exception as e:
    print(f"⚠️ Could not save parameter importance plot: {e}")


# --- TRAIN MODEL -----------------------------------------------------
start_time = time.time()

if model_choice == "CNN":
    config.setdefault('layer_size', 32)
    config.setdefault('hidden_dim', 64)
    config.setdefault('dropout', 0.2)

    model, best_val_loss, history = train_cnn_model(
        X_train_np, y_train_np, X_val_np, y_val_np,
        config=config,
        device=device,
        patience=patience,
        model_choice="CNN",
        alpha=0.0
    )

else:  # NN path
    model, best_val_loss, history = train_model(
        X_train, y_train, X_val, y_val,
        X_train.shape[1],
        config,
        device,
        patience,
        alpha=0.0
    )

end_time = time.time()
total_runtime = end_time - start_time
print(f"⏱️ Training completed in {total_runtime:.2f} seconds.")

# --- EVALUATE ON TEST SET ---------------------------------------------
model.eval()
with torch.no_grad():
    X_test_cnn = preprocess_for_cnn(X_test_np).to(device)
    preds = model(X_test_cnn).cpu().numpy().flatten()
    targets = y_test.cpu().numpy().flatten()

    
    # 🔁 Flatten X_test if it's CNN
X_test_np = X_test.detach().cpu().numpy()
if len(X_test_np.shape) == 3:  # CNN shape: (N, C, T)
    X_test_np = X_test_np.reshape(X_test_np.shape[0], -1)

mae = mean_absolute_error(targets, preds)
r2 = r2_score(targets, preds)
print(f"📊 Test Results — MAE: {mae:.6f}, R²: {r2:.6f}")
    
    
#--------------------------------------------------------------------------------

if study is None:
    print("❌ Optuna study is not defined. Skipping best trial re-training.")
else:
    try:
           
        best_r2_trial = max(
            (t for t in study.trials if "r2" in t.user_attrs),
            key=lambda t: t.user_attrs["r2"]
        )
    except Exception as e:
        print(f"⚠️ Could not retrieve best trial: {e}")
        best_r2_trial = None
        
    #Ensures the same seed is used during Best R2 re-train
        
    seed = best_r2_trial.user_attrs.get("seed", None)
    if seed is not None:
        print(f"[Retrain] Re‐using seed = {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark    = False
    else:
        print("[Retrain] No seed found on trial—results may differ")

    if best_r2_trial is not None:
        start_time = time.time()

        alpha = best_r2_trial.params.get("alpha", 0.0)

        if model_choice == "CNN":
            layer_size = best_r2_trial.params.get("hidden_dim", 128)
            best_r2_config = {
                'layer_size': best_r2_trial.params["layer_size"],
                'hidden_dim': best_r2_trial.params["hidden_dim"],
                'dropout': best_r2_trial.params["dropout"],
                'lr': best_r2_trial.params["lr"],
                'optimizer': best_r2_trial.params.get("optimizer", "Adam")
            }

            print(f"\nRe-training CNN model for best R2 trial #{best_r2_trial.number}...")
            model_r2, _, history = train_cnn_model(
                X_train_np, y_train_np, X_val_np, y_val_np,
                best_r2_config, device, patience, model_choice, alpha=0.0
            )
            
            with torch.no_grad():
                model_r2.eval()
                preds_train = model_r2(preprocess_for_cnn(X_train_np).to(device)).cpu().numpy().flatten()
                preds_val = model_r2(preprocess_for_cnn(X_val_np).to(device)).cpu().numpy().flatten()
                preds_test = model_r2(preprocess_for_cnn(X_test_np).to(device)).cpu().numpy().flatten()
                targets_train = y_train_np
                targets_val = y_val_np
                targets_test = y_test_np

        else:
            opt_used = best_r2_trial.params["optimizer"]
            layer_size = best_r2_trial.params["layer_size"]
            num_layers = best_r2_trial.params["num_layers"]
            best_r2_config = {
                'layers': [layer_size] * num_layers,
                'dropout': best_r2_trial.params["dropout"],
                'lr': best_r2_trial.params["lr"],
                'optimizer': opt_used
            }

            print(f"\nRe-training NN model for best R2 trial #{best_r2_trial.number}...")
            model_r2, _, history = train_model(
                X_train, y_train, X_val, y_val,
                input_size=X_train.shape[1],
                config=best_r2_config,
                device=device,
                patience=patience,
                alpha=0.0
            )
         
            with torch.no_grad():
                model_r2.eval()
                preds_train = model_r2(X_train).cpu().numpy().flatten()
                preds_val = model_r2(X_val).cpu().numpy().flatten()
                preds_test = model_r2(X_test).cpu().numpy().flatten()
                targets_train = y_train.cpu().numpy().flatten()
                targets_val = y_val.cpu().numpy().flatten()
                targets_test = y_test.cpu().numpy().flatten()

        total_runtime = time.time() - start_time
       

        # ✅ Analyze Optuna study results
        if study.best_trial is not None:
            df_trials = analyze_optuna_study(study, report_dir)
        else:
            print("⚠️ Optuna analysis skipped: study missing or incomplete.")

        total_runtime = total_runtime if 'total_runtime' in locals() else 0.0




#---------Unified Diagnotists------------------------------------------------------------------

preds_all = np.concatenate([preds_train, preds_val, preds_test])
targets_all = np.concatenate([targets_train, targets_val, targets_test])

# --- Determine correct test set and predictions based on model type ---
if model_choice == "CNN":
    X_eval = preprocess_for_cnn(X_test_np).to(device)
    y_eval = y_test_np
    X_test_cpu = X_test_np
else:
    X_eval = X_test
    y_eval = y_test.cpu().numpy()
    X_test_cpu = X_test.cpu().numpy()

with torch.no_grad():
    preds_test = model_r2(X_eval).cpu().numpy().flatten()

y_eval = y_eval.flatten()  # Ensure shapes match
errors_r2 = preds_test - y_eval

# --- Predicted vs Actual ---
plt.figure()
plt.scatter(y_eval, preds_test, alpha=0.6)
plt.plot([y_eval.min(), y_eval.max()], [y_eval.min(), y_eval.max()], 'r--')
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual (best R2)")
plt.grid(True)
plt.tight_layout()
plt.savefig(r2_pred_actual_path)
plt.close()

# --- Error Histogram ---
plt.figure()
plt.hist(errors_r2, bins=50, color='gray', edgecolor='black')
plt.title("Prediction Error Histogram (best R2)")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig(r2_error_hist_path)
plt.close()

# --- Residuals vs Predicted ---
plt.figure()
plt.scatter(preds_test, errors_r2, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted (best R2)")
plt.grid(True)
plt.tight_layout()
plt.savefig(r2_resid_pred_path)
plt.close()

# --- QQ Plot ---
plt.figure()
stats.probplot(errors_r2, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals (best R2)")
plt.tight_layout()
plt.savefig(r2_qq_plot_path)
plt.close()


def scatter_fit(ax, x, y, title, color):
    # scatter the model’s predictions vs. truth
    ax.scatter(x, y, alpha=0.6, color=color, label="Data")
    
    # 1:1 reference line
    mn, mx = x.min(), x.max()
    ax.plot([mn, mx], [mn, mx], linestyle="--", color="k", label="y = x")
    
    # compute R² on the exact arrays you passed in
    r2 = r2_score(x, y)
    ax.set_title(f"{title} R² = {r2:.3f}")
    
    # labels & legend
    ax.set_xlabel("Target")
    ax.set_ylabel("Output")
    ax.legend()
    ax.grid(True)


# --- 4-panel R² scatter plot using best trial predictions ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
scatter_fit(axs[0, 0], targets_train, preds_train, "Train", "blue")
scatter_fit(axs[0, 1], targets_val, preds_val, "Validation", "green")
scatter_fit(axs[1, 0], targets_test, preds_test, "Test", "red")
scatter_fit(axs[1, 1], targets_all, preds_all, "All", "gray")


fig.tight_layout()
r2_four_panel_path = os.path.join(report_dir, "r2_scatter_four_panel.png")
plt.savefig(r2_four_panel_path)
plt.close()

#--- Correlation Heatmap (Features vs Target, Best R2) ---
X_test_np = X_test.cpu().numpy()
df_corr = pd.DataFrame(X_test_np, columns=[f"x{i}" for i in range(X_test_np.shape[1])])
df_corr["target"] = targets_test
corr_matrix = df_corr.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix[["target"]].drop("target"), annot=True, cmap="coolwarm", cbar=True)
plt.title("Feature Correlation with Target (best R2)")
plt.tight_layout()
r2_corr_heatmap_path = os.path.join(report_dir, "r2_corr_heatmap.png")
plt.savefig(r2_corr_heatmap_path)
plt.close()


# #==========================================================

# Save R² and MAE per epoch
fig, ax1 = plt.subplots()
ax1.set_xlabel('Epoch')
ax1.set_ylabel('R²', color='tab:blue')
ax1.plot(history['r2'], color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel('MAE', color='tab:red')
ax2.plot(history['mae'], color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

fig.tight_layout()
mae_r2_plot_path = os.path.join(report_dir, "mae_r2_plot.png")
plt.savefig(mae_r2_plot_path)
plt.close()
   
    

# --- Best Metrics Across All Trials (Optuna) ---

r2_trials = [(t.number, t.user_attrs.get("r2")) for t in study.trials if "r2" in t.user_attrs]
best_r2_trial_num, best_r2_val = max(r2_trials, key=lambda x: x[1]) if r2_trials else (None, None)

mae_trials = [(t.number, t.user_attrs.get("mae")) for t in study.trials if "mae" in t.user_attrs]
best_mae_trial_num, best_mae_val = min(mae_trials, key=lambda x: x[1]) if mae_trials else (None, None)

best_r2_trial = study.trials[best_r2_trial_num] if best_r2_trial_num is not None else None


if best_r2_trial is not None:
    params = best_r2_trial.params
    optimizer = params.get("optimizer", "N/A")
    lr = params.get("lr", "N/A")
    dropout = params.get("dropout", "N/A")
    layer_size = params.get("layer_size", params.get("layer_size", "N/A"))
    num_layers = params.get("num_layers", params.get("num_layers", "N/A"))

    # Loss Plot for Best R²
    loss_plot_r2_path = os.path.join(report_dir, "loss_plot_r2.png")
    plt.figure()
    plt.plot(history['train'], label="Train Loss")
    plt.plot(history['val'], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch (Best R² Trial)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_plot_r2_path)
    plt.close()


    # All R² plots
    r2_plot_paths = [
        ("Predicted vs Actual (Best R²)", r2_pred_actual_path),
        ("Prediction Error Histogram (Best R²)", r2_error_hist_path),
        ("Residuals vs Predicted (Best R²)", r2_resid_pred_path),
        ("QQ Plot of Residuals (Best R²)", r2_qq_plot_path),
        ("Feature Correlation with Target (Best R²)", r2_corr_heatmap_path),
        ("R² Scatter Plots (Train/Val/Test/All)", r2_four_panel_path)
    ]



# --- Save config.txt for predict_api ---
config_txt_path = os.path.join(report_dir, f"config_{timestamp}.txt")
num_layers = int(best_r2_config.get("num_layers", best_r2_config.get("num_layers", 1)))
layer_size = int(best_r2_config.get("layer_size", best_r2_config.get("layer_size", 64)))
dropout = float(best_r2_config.get("dropout", 0.0))
with open(config_txt_path, "w") as f:
    f.write(f"model_choice: {model_choice}\n")
    f.write(f"input_size: {input_size}\n")
    
    if model_choice == "NN":
        f.write(f"layer_size: {best_r2_trial.params['layer_size']}\n")
    elif model_choice == "CNN":
        f.write(f"hidden_dim: {best_r2_trial.params['hidden_dim']}\n")
        f.write(f"layer_size: {best_r2_trial.params['layer_size']}\n")  # CNN uses this too if defined
    
    f.write(f"dropout: {best_r2_trial.params['dropout']}\n")
    f.write(f"lr: {best_r2_trial.params['lr']}\n")
    f.write(f"alpha: {best_r2_trial.params.get('alpha', 0.0)}\n")  # optional alpha
    f.write(f"SEED: {SEED}\n")
    f.write(f"optimizer: {best_r2_trial.params['optimizer']}\n")


# --- SAVE MODEL ---

# (A’) Save the full checkpoint dict under bestmodel_{timestamp}.pt
best_model_path = os.path.join(best_r2_dir, f"bestmodel_{timestamp}.pt")

config_path = os.path.join(best_r2_dir, f"bestconfig_{timestamp}.txt")

best_model_path = os.path.join(best_r2_dir, f"bestmodel_{timestamp}.pt")

if model_choice == "NN":
    num_layers = best_r2_trial.params["num_layers"]
    layer_size = best_r2_trial.params["layer_size"]
    layers_list = [layer_size] * num_layers

elif model_choice == "CNN":
    # CNN doesn't use num_layers the same way
    hidden_dim = best_r2_trial.params["hidden_dim"]
    layer_size = best_r2_trial.params["layer_size"]
    layers_list = [hidden_dim, layer_size]  # For reference or logging only

checkpoint = {
    "model_state_dict": model.state_dict(),
    "model_choice":     model_choice,
    "layers": [best_r2_trial.params["layer_size"]] * best_r2_trial.params["num_layers"] if model_choice == "NN" else None,
    "dropout":          best_r2_trial.params["dropout"],
    "hidden_dim":       best_r2_trial.params.get("hidden_dim"),
    "lr":               best_r2_trial.params["lr"],
    "optimizer":        best_r2_trial.params["optimizer"],
    "r2":               best_r2_trial.value,                     # ← the best R² metric
    "SEED":             SEED,
    "input_size":       input_size
}
torch.save(checkpoint, best_model_path)

print("🔍 Best trial params:", best_r2_trial.params)

best_scaler_path = os.path.join(best_r2_dir, f"bestscaler_{timestamp}.pkl")
joblib.dump(scaler, best_scaler_path)
with open(config_path, "w") as f:
    f.write(f"model_choice: {model_choice}\n")
    f.write(f"input_size: {input_size}\n")
    if model_choice == "NN":
        f.write(f"hidden_dim: {num_layers}\n")
    if model_choice == "CNN":
        f.write(f"hidden_dim: {hidden_dim}\n")
    f.write(f"layer_size: {layer_size}\n")
    f.write(f"dropout: {dropout}\n")
    f.write(f"lr: {lr}\n") 
    f.write(f"SEED: {SEED}\n") 
    f.write(f"optimizer: {optimizer}\n")
 
print("✅ Model and scaler saved successfully.")

# print(f"📄 Training Report saved to:\n{report_path}")

             
# ─── Save‐time Model Summary ────────────────────────────────────
print("\n🌟 Saved Model Summary 🌟")

if model_choice == "NN":
    layers_list = [best_r2_trial.params["layer_size"]] * best_r2_trial.params["num_layers"]
    num_layers  = len(layers_list)
    print(f"  Model Choice     : {model_choice}")
    print(f"  Number of Layers : {num_layers}")
    print(f"  Layer Sizes      : {layers_list}")
    print(f"  Dropout          : {best_r2_trial.params['dropout']:.6f}")
    print("  Hidden Dims      : (CNN Only)")
    print(f"  Learning Rate    : {best_r2_trial.params['lr']:.6e}")
    print(f"  Optimizer        : {best_r2_trial.params['optimizer']}")
    print(f"  SEED             : {SEED}")
    print(f"  Input Size       : {input_size}")

else:  # CNN
    print(f"  Model Choice     : {model_choice}")
    print(f"  Num Conv Layers  : {sum(isinstance(m, torch.nn.Conv1d) for m in model.cnn)}")
    print(f"  Conv Width       : {best_r2_trial.params['layer_size']}")
    print(f"  Hidden Dim       : {best_r2_trial.params['hidden_dim']}")
    print(f"  Dropout          : {best_r2_trial.params['dropout']:.6f}")
    print(f"  Learning Rate    : {best_r2_trial.params['lr']:.6e}")
    print(f"  Optimizer        : {best_r2_trial.params['optimizer']}")
    print(f"  SEED             : {SEED}")
    print(f"  Input Size       : {input_size}")
    print(f"\n  Best R²           : {best_r2_trial.value:.4f}")
print("──────────────────────────────────────────────────────────────\n")




  # Generate model architecture graph (optional)
try:
    sample_input = sample_input = X_train[:1]
    model_graph = torchviz.make_dot(model(sample_input), params=dict(model.named_parameters()))
    graph_path = os.path.join(report_dir, "model_architecture.png")
    model_graph.render(graph_path[:-4], format="png")
except Exception as e:
    print(f"⚠️ Could not generate model architecture graph: {e}")
    graph_path = None

#/////////////////////////////////////////////////////////////////////////////////////////////////
top_trials = df_trials.sort_values(by="r2", ascending=False).head(25)

all_plots = sorted(f for f in os.listdir(report_dir) if f.endswith(".png"))

# 2) Categorize plots into sections
optuna_imgs = []
intermediate_imgs = []
split_imgs = []
other_imgs = []
arch_html = ""
for plot in all_plots:
    title = plot.replace(".png", "").replace("_", " ").title()
    img_tag = f"<h3>{title}</h3><img src=\"{plot}\">"
    if plot.startswith("optuna_"):
        optuna_imgs.append(img_tag)
    elif plot in (
        "loss_plot.png", "mae_r2_plot.png", "pred_vs_actual.png", "error_histogram.png",
        "r2_pred_vs_actual.png", "r2_error_histogram.png", "r2_resid_vs_pred.png",
        "r2_qq_plot.png", "r2_corr_heatmap.png", "r2_scatter_four_panel.png"
    ):
        intermediate_imgs.append(img_tag)
    elif plot.startswith(("train_", "val_", "test_")):
        split_imgs.append(img_tag)
    elif plot.startswith("model_architecture"):
        arch_html = img_tag
    else:
        other_imgs.append(img_tag)

# 3) Join HTML sections
optuna_html = "".join(optuna_imgs)
intermediate_html = "".join(intermediate_imgs)
split_html = "".join(split_imgs)
other_html = "".join(other_imgs)




html_rows = "\n".join(
    "<tr>"
    + "".join(
        f"<td>{v:.9f}</td>" if isinstance(v, float) else f"<td>{v}</td>"
        for v in row
    )
    + "</tr>"
    for row in df_trials.itertuples(index=False)
)

# Dynamically scan report_dir for plot images
all_plots = sorted([f for f in os.listdir(report_dir) if f.endswith(".png")])

# Categorize plots
optuna_html = []
intermediate_html = []
split_html = []
other_html = []
arch_html = ""

for plot in all_plots:
    title = plot.replace(".png", "").replace("_", " ").title()
    img_tag = f"<h3>{title}</h3><img src=\"{plot}\">"

    if plot.startswith("optuna_"):
        optuna_html.append(img_tag)
    elif plot in [
        "loss_plot.png", "mae_r2_plot.png", "pred_vs_actual.png", "error_histogram.png",
        "r2_pred_vs_actual.png", "r2_error_histogram.png", "r2_resid_vs_pred.png",
        "r2_qq_plot.png", "r2_corr_heatmap.png", "r2_scatter_four_panel.png"
    ]:
        intermediate_html.append(img_tag)
    elif plot.startswith("train_") or plot.startswith("val_") or plot.startswith("test_"):
        split_html.append(img_tag)
    elif plot.startswith("model_architecture"):
        arch_html = f"<h3>Model Architecture</h3><img src=\"{plot}\">"
    else:
        other_html.append(img_tag)

# Assemble sections
optuna_html = ''.join(optuna_html)
intermediate_html = ''.join(intermediate_html)
split_html = ''.join(split_html)
other_html = ''.join(other_html)

html_content = f"""
<html>
<head>
    <title>NFTool Training Report - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        img {{ max-width: 100%; height: auto; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        a {{ color: #0077cc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>NFTool Training Report</h1>
    <h2>Generated: {timestamp}</h2>

    <h3>Table of Contents</h3>
    <ul>
        <li><a href="#config">Training Configuration</a></li>
        <li><a href="#best">Best R² Data</a></li>
        <li><a href="#trials">Top 25 Trials by R²</a></li>
        <li><a href="#plots">Training Plots</a></li>
    </ul>

    <h2 id="config">Training Configuration</h2>
    <ul>
      <li>Predictor File: {predictor_file}</li>
      <li>Target File: {target_file}</li>
      <li>Model Type: {model_choice}</li>
      <li>Optimizer: {study.best_trial.params.get('optimizer')}</li>
      <li>Number of Layers: {study.best_trial.params.get('num_layers', 'N/A')}</li>
      <li>Hidden Dim: {study.best_trial.params.get('hidden_dim', 'N/A')}</li>
      <li>Layer Size: {study.best_trial.params.get('layer_size')}</li>
      <li>Patience: {patience}</li>
      <li>Target Variance: {target_variance:.6f}</li>
      <li>SNR: {snr:.2f}</li>
      <li>Random Seed: {SEED}</li>
      <li>Learning Rate: {study.best_trial.params.get('lr')}</li>
      <li>Train/Val/Test Split: 70.0% / 15.0% / 15.0%</li>
      <li>K-Folds Used: {k_folds} {'(K-Fold CV enabled)' if k_folds >= 2 else '(Standard split)'}</li>
      <li>Total Runtime: {total_runtime:.2f} s</li>
    </ul>

    <h2 id="best">Best R² Data</h2>
    <ul>
        <li>Best R²: {top_trials.iloc[0]['r2']}</li>
        <li>Best MAE: {study.best_trial.user_attrs.get('mae')}</li>
        <li>Optimizer: {study.best_trial.params.get('optimizer')}</li>
        <li>Number of Layers: {study.best_trial.params.get('num_layers')}</li>
        <li>Layer Size: {study.best_trial.params.get('layer_size')}</li>
        <li>Learning Rate: {study.best_trial.params.get('lr')}</li>
        <li>Dropout Rate: {study.best_trial.params.get('dropout')}</li>
        <li>Alpha: {study.best_trial.params.get('alpha', 0.0):.3f}</li>
    </ul>

    <h2 id="trials">Top 25 Trials by R²</h2>
    <table>
        <tr>
            {''.join(f'<th>{col}</th>' for col in df_trials.columns)}
        </tr>
        {html_rows}
    </table>

    <h2 id="plots">Training Plots</h2>
    <h3>Optuna Trial Diagnostics</h3>
    {optuna_html}

    <h3>Highest R² Run - Global Metrics</h3>
    {intermediate_html}

    <h3>Highest R² Run - Train/Val/Test Metrics</h3>
    {split_html}

    <h3>Other Visualizations</h3>
    {other_html}

    {arch_html}
</body>
</html>
"""



# Save HTML
report_path_html = os.path.join(report_dir, f"NFTool_Report_{timestamp}.html")
with open(report_path_html, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"✅ HTML report saved: {report_path_html}")

print("✅ NFtool completed successfully. Exiting.")
sys.exit(0)


if __name__ == "__main__":
    print("🚀 About to call main()")
    main()
    