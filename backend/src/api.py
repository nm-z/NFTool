import os
import asyncio
import json
import logging
import sys
import multiprocessing
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Local Imports
from src.config import REPO_ROOT, WORKSPACE_DIR, DATABASE_URL, API_KEY, LOGS_DIR, RESULTS_DIR, REPORTS_DIR
from src.database.models import Base, Run, ModelCheckpoint
from src.schemas.training import TrainingConfig, safe_path
from src.utils.hardware import HardwareMonitor
from src.training.engine import train_model, train_cnn_model, run_optimization

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "api.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("nftool")

def log_print(*args, **kwargs):
    msg = " ".join(map(str, args))
    logger.info(msg)

# Database Setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Auth
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

class TrainingState:
    def __init__(self):
        self.active_run_id = None
        self.active_process = None
        self.clients: List[WebSocket] = []
        self.client_lock = asyncio.Lock()
        self.loop = None
        self.hardware_stats: Dict[str, Any] = {}
        self.monitor = HardwareMonitor()

    def set_loop(self, loop):
        self.loop = loop

    async def broadcast(self, message: Dict[str, Any]):
        async with self.client_lock:
            if not self.clients:
                return
            disconnected = []
            msg_json = json.dumps(message)
            for client in self.clients:
                try:
                    await client.send_text(msg_json)
                except:
                    disconnected.append(client)
            for client in disconnected:
                if client in self.clients:
                    self.clients.remove(client)

    def broadcast_sync(self, message: Dict[str, Any]):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)

state = TrainingState()

app = FastAPI(title="NFTool API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

async def log_and_broadcast(msg: str, type: str = "default"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {"time": timestamp, "msg": msg, "type": type}
    await state.broadcast({"type": "log", "data": log_entry})
    log_print(f"[{timestamp}] {msg}")

def db_log_and_broadcast(db: Session, run_id: str, msg: str, type: str = "default"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {"time": timestamp, "msg": msg, "type": type}
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if run:
        current_logs = list(run.logs) if run.logs else []
        current_logs.append(log_entry)
        run.logs = current_logs
        db.commit()
    state.broadcast_sync({"type": "log", "data": log_entry})
    log_print(f"[{timestamp}] {msg}")

async def hardware_monitor_task():
    while True:
        try:
            if not state.clients:
                await asyncio.sleep(5)
                continue
            is_running = state.active_process is not None and state.active_process.is_alive()
            interval = 2 if is_running else 10
            gpu_id = getattr(state, 'current_gpu_id', 0)
            gpu_stats = state.monitor.get_gpu_stats(gpu_id)
            system_stats = state.monitor.get_system_stats()
            state.hardware_stats = {**gpu_stats, **system_stats}
            await state.broadcast({"type": "hardware", "data": state.hardware_stats})
            await asyncio.sleep(interval)
        except Exception as e:
            log_print(f"Hardware monitor error: {e}")
            await asyncio.sleep(5)

def run_training_task(config_dict: Dict[str, Any], run_id: str):
    db = SessionLocal()
    try:
        run = db.query(Run).filter(Run.run_id == run_id).first()
        if not run: return
        config = TrainingConfig(**config_dict)
        db_log_and_broadcast(db, run_id, f"Starting training engine for {run_id}...", "info")
        
        if config.device == "cuda" and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_id = min(config.gpu_id, gpu_count - 1)
            device = torch.device(f"cuda:{gpu_id}")
            device_name = torch.cuda.get_device_name(gpu_id)
        else:
            device = torch.device("cpu")
            device_name = "CPU"
        db_log_and_broadcast(db, run_id, f"Using device: {device_name}", "success")

        predictor_path = safe_path(config.predictor_path)
        target_path = safe_path(config.target_path)
        df_X = pd.read_csv(predictor_path, header=None)
        df_y = pd.read_csv(target_path, header=None).dropna()
        min_len = min(len(df_X), len(df_y))
        X_raw, y_raw = df_X.iloc[:min_len].values, df_y.iloc[:min_len].values.flatten()
        
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_scaled = scaler_X.fit_transform(X_raw)
        y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
        
        run_dir = os.path.join(REPORTS_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)
        scaler_x_path, scaler_y_path = os.path.join(run_dir, "scaler_x.pkl"), os.path.join(run_dir, "scaler_y.pkl")
        joblib.dump(scaler_X, scaler_x_path)
        joblib.dump(scaler_y, scaler_y_path)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=config.test_ratio, random_state=config.seed)
        val_rel = config.val_ratio / (config.train_ratio + config.val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_rel, random_state=config.seed)

        metrics_history = []

        def objective(trial):
            run.current_trial = trial.number + 1
            run.progress = int((trial.number) / config.optuna_trials * 100)
            db.commit()
            state.broadcast_sync({"type": "status", "data": {"is_running": True, "progress": run.progress, "current_trial": run.current_trial, "total_trials": config.optuna_trials}})

            def on_epoch_end(epoch, num_epochs, loss, val_loss, r2):
                if epoch % 20 == 0: db_log_and_broadcast(db, run_id, f"Trial #{trial.number} - Epoch {epoch}/{num_epochs}: loss={loss:.4f}, r2={r2:.4f}", "info")

            optimizer = trial.suggest_categorical("optimizer", config.optimizers)
            lr = trial.suggest_float("lr", config.lr_min, config.lr_max, log=True)
            dropout = trial.suggest_float("dropout", config.drop_min, config.drop_max)
            
            if config.model_choice == "NN":
                n_layers = trial.suggest_int("num_layers", config.n_layers_min, config.n_layers_max)
                l_size = trial.suggest_int("layer_size", config.l_size_min, config.l_size_max)
                cfg = {'layers': [l_size] * n_layers, 'dropout': dropout, 'lr': lr, 'optimizer': optimizer}
                model, loss, history = train_model(
                    torch.tensor(X_train, dtype=torch.float32).to(device),
                    torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device),
                    torch.tensor(X_val, dtype=torch.float32).to(device),
                    torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device),
                    X_train.shape[1], cfg, device, config.patience, num_epochs=config.max_epochs, gpu_throttle_sleep=config.gpu_throttle_sleep, on_epoch_end=on_epoch_end
                )
            else:
                n_conv = trial.suggest_int("num_conv_blocks", config.conv_blocks_min, config.conv_blocks_max)
                base_filters = trial.suggest_int("base_filters", config.l_size_min, config.l_size_max)
                h_dim = trial.suggest_int("hidden_dim", int(config.h_dim_min), int(config.h_dim_max))
                current_cap = trial.suggest_int("cnn_filter_cap", config.cnn_filter_cap_min, config.cnn_filter_cap_max)
                conv_layers = [{'out_channels': min(base_filters * (2**i), current_cap), 'kernel': config.kernel_size, 'pool': 2} for i in range(n_conv)]
                cfg = {'conv_layers': conv_layers, 'hidden_dim': h_dim, 'dropout': dropout, 'lr': lr, 'optimizer': optimizer}
                model, loss, history = train_cnn_model(X_train, y_train, X_val, y_val, cfg, device, config.patience, num_epochs=config.max_epochs, gpu_throttle_sleep=config.gpu_throttle_sleep, on_epoch_end=on_epoch_end)
            
            if history['r2']:
                r2, mae = history['r2'][-1], history['mae'][-1]
                trial.set_user_attr("r2", r2); trial.set_user_attr("mae", mae)
                metric_entry = {"trial": trial.number + 1, "loss": float(loss), "r2": float(r2), "mae": float(mae), "val_loss": float(history['val'][-1]) if history['val'] else 0.0}
                metrics_history.append(metric_entry)
                state.broadcast_sync({"type": "metrics", "data": metric_entry})
                if r2 == max([m['r2'] for m in metrics_history]):
                    model_path = os.path.join(run_dir, "best_model.pt")
                    torch.save({"model_state_dict": model.state_dict(), "model_choice": config.model_choice, "input_size": X_train.shape[1], "r2": r2, "mae": mae, "config": cfg, "scaler_x_path": scaler_x_path, "scaler_y_path": scaler_y_path}, model_path)
                    db.add(ModelCheckpoint(run_id=run.id, model_path=model_path, scaler_path=scaler_x_path, r2_score=r2, params=cfg))
                    db.commit()
            
            run.progress = int((trial.number + 1) / config.optuna_trials * 100)
            db.commit()
            state.broadcast_sync({"type": "status", "data": {"is_running": True, "progress": run.progress, "current_trial": run.current_trial, "total_trials": config.optuna_trials}})
            if device.type == "cuda": torch.cuda.empty_cache()
            return loss

        study = run_optimization(f"NFTool_{config.model_choice}", config.optuna_trials, None, objective)
        run.best_r2, run.metrics_history, run.status, run.report_path = study.best_trial.user_attrs.get("r2"), metrics_history, "completed", run_dir
        db.commit()
        db_log_and_broadcast(db, run_id, f"Optimization finished. Best R²: {run.best_r2:.4f}", "success")
    except Exception as e:
        db_log_and_broadcast(db, run_id, f"Critical training error: {str(e)}", "warn")
        if run: run.status = "failed"; db.commit()
    finally:
        if 'device' in locals() and device.type == "cuda": torch.cuda.empty_cache()
        if run:
            final_progress = 100
            final_trial = run.current_trial
            final_total = run.optuna_trials
        else:
            final_progress = 0
            final_trial = 0
            final_total = 0
            
        db.close()
        state.broadcast_sync({"type": "status", "data": {
            "is_running": False, 
            "progress": final_progress, 
            "run_id": run_id,
            "current_trial": final_trial,
            "total_trials": final_total
        }})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    protocols = websocket.headers.get("Sec-WebSocket-Protocol", "").split(",")
    client_api_key = next((p.strip().replace("api-key-", "") for p in protocols if p.strip().startswith("api-key-")), None)
    if client_api_key != API_KEY:
        await websocket.accept()
        await websocket.send_text(json.dumps({"type": "error", "data": "Unauthorized"}))
        await websocket.close(code=4003); return
    await websocket.accept(subprotocol=f"api-key-{client_api_key}" if client_api_key else None)
    state.set_loop(asyncio.get_event_loop())
    async with state.client_lock: state.clients.append(websocket)
    try:
        db = SessionLocal(); latest_run = db.query(Run).order_by(Run.timestamp.desc()).first(); db.close()
        is_running = state.active_process is not None and state.active_process.is_alive()
        await websocket.send_text(json.dumps({"type": "init", "data": {"is_running": is_running, "progress": latest_run.progress if latest_run else 0, "current_trial": latest_run.current_trial if latest_run else 0, "total_trials": latest_run.optuna_trials if latest_run else 0, "logs": latest_run.logs if latest_run else [], "metrics_history": latest_run.metrics_history if latest_run else [], "hardware_stats": state.hardware_stats}}))
        while True:
            if await websocket.receive_text() == "ping": await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        async with state.client_lock:
            if websocket in state.clients: state.clients.remove(websocket)

@app.get("/gpus", dependencies=[Depends(verify_api_key)])
def list_gpus():
    return [{"id": i, "name": torch.cuda.get_device_name(i), "is_available": True} for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []

@app.get("/health")
def health_check():
    return {"status": "ok", "gpu": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}

@app.post("/train", dependencies=[Depends(verify_api_key)])
async def start_training(config: TrainingConfig):
    # Check if process exists and is actually alive
    if state.active_process is not None:
        if state.active_process.is_alive():
            raise HTTPException(status_code=400, detail="Training already in progress")
        else:
            # Clean up finished process
            state.active_process.join()
            state.active_process = None

    db = SessionLocal(); run_id = f"PASS_{datetime.now().strftime('%H%M%S%f')[:9]}"
    db.add(Run(run_id=run_id, model_choice=config.model_choice, status="running", optuna_trials=config.optuna_trials, config=config.model_dump(), logs=[]))
    db.commit(); db.close()
    state.active_run_id, state.current_gpu_id = run_id, config.gpu_id
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=run_training_task, args=(config.model_dump(), run_id))
    process.start(); state.active_process = process
    return {"message": "Training started", "run_id": run_id}

@app.post("/abort", dependencies=[Depends(verify_api_key)])
async def abort_training():
    if state.active_process and state.active_process.is_alive():
        state.active_process.terminate(); state.active_process.join()
        db = SessionLocal(); run = db.query(Run).filter(Run.run_id == state.active_run_id).first()
        if run: run.status = "aborted"; db.commit()
        db.close()
        await log_and_broadcast("ABORT SIGNAL SENT: Process terminated.", "warn")
        await state.broadcast({"type": "status", "data": {"is_running": False, "is_aborting": False, "progress": 0}})
        state.active_process = state.active_run_id = None
    return {"status": "aborted"}

@app.get("/dataset/preview", dependencies=[Depends(verify_api_key)])
async def preview_dataset(path: str, rows: int = 10):
    target = safe_path(path)
    if not os.path.exists(target): raise HTTPException(status_code=404, detail="File not found")
    df = pd.read_csv(target, header=None, nrows=rows)
    return {"headers": [f"Feature_{i}" for i in range(df.shape[1])], "rows": df.values.tolist(), "shape": list(df.shape), "total_rows": len(pd.read_csv(target, header=None))}

@app.post("/inference", dependencies=[Depends(verify_api_key)])
async def run_inference(model_path: str, features: List[float]):
    target = safe_path(model_path)
    if not os.path.exists(target): raise HTTPException(status_code=404, detail="Model file not found")
    checkpoint = torch.load(target, map_location="cpu")
    input_size = checkpoint.get("input_size")
    if len(features) != input_size: raise HTTPException(status_code=400, detail=f"Expected {input_size} features, got {len(features)}")
    scaler_X, scaler_y = joblib.load(checkpoint.get("scaler_x_path")), joblib.load(checkpoint.get("scaler_y_path"))
    from src.models.architectures import model_factory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory(checkpoint.get("model_choice", "NN"), input_size, checkpoint.get("config", {}), device)
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    X_scaled = scaler_X.transform(np.array([features]))
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    if checkpoint.get("model_choice") == "CNN":
        from src.data.processing import preprocess_for_cnn
        X_tensor = preprocess_for_cnn(X_scaled).to(device)
    with torch.no_grad():
        prediction_raw = scaler_y.inverse_transform(model(X_tensor).cpu().numpy().reshape(-1, 1)).flatten()[0]
    return {"prediction": float(prediction_raw), "model_r2": checkpoint.get("r2", 0.0)}

@app.get("/runs", dependencies=[Depends(verify_api_key)])
def list_runs(db: Session = Depends(lambda: SessionLocal())):
    return db.query(Run).order_by(Run.timestamp.desc()).all()

@app.get("/datasets", dependencies=[Depends(verify_api_key)])
def list_datasets():
    dataset_dir = os.path.join(REPO_ROOT, "data")
    if not os.path.exists(dataset_dir): return []
    return [{"name": f, "path": os.path.join("data", f)} for f in sorted([f for f in os.listdir(dataset_dir) if f.endswith(".csv")])]

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(hardware_monitor_task())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)