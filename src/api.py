from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
import json

# Import our modular logic
from src.training.engine import train_model, train_cnn_model, run_optimization, Objective
from src.data.processing import compute_dataset_snr_from_files
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = FastAPI(title="NFTool API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for training status and logs
class TrainingState:
    def __init__(self):
        self.is_running = False
        self.progress = 0
        self.logs = []
        self.result = None
        self.clients: List[WebSocket] = []

    async def broadcast(self, message: Dict[str, Any]):
        if not self.clients:
            return
        
        # Remove disconnected clients while broadcasting
        disconnected = []
        for client in self.clients:
            try:
                await client.send_text(json.dumps(message))
            except:
                disconnected.append(client)
        
        for client in disconnected:
            if client in self.clients:
                self.clients.remove(client)

    def add_log(self, msg: str, type: str = "default"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {"time": timestamp, "msg": msg, "type": type}
        self.logs.append(log_entry)
        # We'll broadcast this separately in the async loop
        return log_entry

state = TrainingState()

class TrainingConfig(BaseModel):
    model_choice: str
    seed: int
    patience: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    optuna_trials: int
    optimizers: List[str]
    n_layers_min: int
    n_layers_max: int
    l_size_min: int
    l_size_max: int
    lr_min: float
    lr_max: float
    drop_min: float
    drop_max: float
    h_dim_min: float
    h_dim_max: float
    # CNN specific
    conv_blocks_min: int = 1
    conv_blocks_max: int = 5
    kernel_size: int = 3
    predictor_path: str
    target_path: str

async def log_and_broadcast(msg: str, type: str = "default"):
    entry = state.add_log(msg, type)
    await state.broadcast({"type": "log", "data": entry})
    print(f"[{entry['time']}] {msg}")

async def run_training_task(config: TrainingConfig):
    state.is_running = True
    state.progress = 0
    state.logs = []
    state.result = None
    
    await state.broadcast({"type": "status", "data": {"is_running": True, "progress": 0}})

    try:
        await log_and_broadcast("Starting training engine...", "info")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        await log_and_broadcast(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", "success")

        # Load Data
        await log_and_broadcast(f"Loading datasets from {config.predictor_path}...", "default")
        df_X = pd.read_csv(config.predictor_path, header=None)
        df_y = pd.read_csv(config.target_path, header=None).dropna()
        min_len = min(len(df_X), len(df_y))
        X_raw, y_raw = df_X.iloc[:min_len].values, df_y.iloc[:min_len].values.flatten()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        await log_and_broadcast(f"Data loaded: {X_raw.shape[0]} samples, {X_raw.shape[1]} features", "success")

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_raw, test_size=config.test_ratio, random_state=config.seed)
        val_rel = config.val_ratio / (config.train_ratio + config.val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_rel, random_state=config.seed)

        # Optuna Objective
        def objective(trial):
            optimizer = trial.suggest_categorical("optimizer", config.optimizers)
            lr = trial.suggest_float("lr", config.lr_min, config.lr_max, log=True)
            dropout = trial.suggest_float("dropout", config.drop_min, config.drop_max)
            
            if config.model_choice == "NN":
                n_layers = trial.suggest_int("num_layers", config.n_layers_min, config.n_layers_max)
                l_size = trial.suggest_int("layer_size", config.l_size_min, config.l_size_max)
                cfg = {'layers': [l_size] * n_layers, 'dropout': dropout, 'lr': lr, 'optimizer': optimizer}
                _, loss, history = train_model(
                    torch.tensor(X_train, dtype=torch.float32).to(device),
                    torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device),
                    torch.tensor(X_val, dtype=torch.float32).to(device),
                    torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device),
                    X_train.shape[1], cfg, device, config.patience
                )
            else: # CNN
                n_conv = trial.suggest_int("num_conv_blocks", config.conv_blocks_min, config.conv_blocks_max)
                base_filters = trial.suggest_int("base_filters", config.l_size_min, config.l_size_max)
                h_dim = trial.suggest_int("hidden_dim", int(config.h_dim_min), int(config.h_dim_max))
                conv_layers = [{'out_channels': base_filters * (2**i), 'kernel': config.kernel_size, 'pool': 2} for i in range(n_conv)]
                cfg = {'conv_layers': conv_layers, 'hidden_dim': h_dim, 'dropout': dropout, 'lr': lr, 'optimizer': optimizer}
                _, loss, history = train_cnn_model(X_train, y_train, X_val, y_val, cfg, device, config.patience)
            
            if history['r2']:
                r2 = history['r2'][-1]
                trial.set_user_attr("r2", r2)
                trial.set_user_attr("mae", history['mae'][-1])
                # Note: We can't await inside the objective function since optuna isn't async aware
                # but we can print and the state will be updated
                print(f"Trial #{trial.number} complete. R²: {r2:.4f}")
            
            state.progress = int((trial.number + 1) / config.optuna_trials * 100)
            return loss

        await log_and_broadcast(f"Running {config.optuna_trials} Optuna trials...", "info")
        # Optimization is blocking, so progress updates happen via the objective function
        # For a truly async experience, we'd need to run this in a thread and use a queue for logs
        study = run_optimization(f"NFTool_{config.model_choice}", config.optuna_trials, None, objective)
        
        state.result = {
            "best_r2": study.best_trial.user_attrs.get("r2"),
            "best_params": study.best_params
        }
        await log_and_broadcast(f"Optimization complete! Best R²: {study.best_trial.user_attrs.get('r2'):.4f}", "success")

    except Exception as e:
        await log_and_broadcast(f"Error: {str(e)}", "warn")
    finally:
        state.is_running = False
        state.progress = 100
        await state.broadcast({"type": "status", "data": {"is_running": False, "progress": 100, "result": state.result}})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.clients.append(websocket)
    try:
        # Send initial state
        await websocket.send_text(json.dumps({
            "type": "init",
            "data": {
                "is_running": state.is_running,
                "progress": state.progress,
                "logs": state.logs,
                "result": state.result
            }
        }))
        while True:
            await websocket.receive_text() # Keep connection alive
    except WebSocketDisconnect:
        if websocket in state.clients:
            state.clients.remove(websocket)

@app.get("/health")
def health_check():
    return {"status": "ok", "gpu": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}

@app.post("/train")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    if state.is_running:
        raise HTTPException(status_code=400, detail="Training already in progress")
    background_tasks.add_task(run_training_task, config)
    return {"message": "Training started"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
