import os
import asyncio
import json
import logging
import sys
import multiprocessing
import shutil
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
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
from src.training.engine import train_model, train_cnn_model, run_optimization, Objective
from src.data.processing import load_dataset, preprocess_for_cnn
from src.utils.reporting import analyze_optuna_study, generate_regression_plots

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
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},
    pool_size=20,
    max_overflow=10,
    pool_timeout=60
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

async def log_and_broadcast(msg: str, type: str = "default", epoch: int = None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {"time": timestamp, "msg": msg, "type": type, "epoch": epoch}
    await state.broadcast({"type": "log", "data": log_entry})
    log_print(f"[{timestamp}] {msg}")

def db_log_and_broadcast(db: Session, run_id: str, msg: str, type: str = "default", epoch: int = None):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {"time": timestamp, "msg": msg, "type": type}
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if run:
        current_logs = list(run.logs) if run.logs else []
        log_entry["epoch"] = epoch
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
            db_log_and_broadcast(db, run_id, f"GPU ACCELERATION ENABLED: {device_name}", "success")
        else:
            device = torch.device("cpu")
            device_name = "CPU"
            db_log_and_broadcast(db, run_id, f"Using device: {device_name}", "info")

        predictor_path = safe_path(config.predictor_path)
        target_path = safe_path(config.target_path)
        df_X = load_dataset(predictor_path)
        df_y = load_dataset(target_path).dropna()
        min_len = min(len(df_X), len(df_y))
        X_raw, y_raw = df_X.iloc[:min_len].values, df_y.iloc[:min_len].values.flatten()
        
        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_scaled = scaler_X.fit_transform(X_raw)
        y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
        
        run_dir = os.path.join(REPORTS_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        db_log_and_broadcast(db, run_id, f"Dataset loaded: {len(X_raw)} samples, {X_raw.shape[1]} features", "info")
        db_log_and_broadcast(db, run_id, f"Split Ratios: {config.train_ratio*100:.0f}% Train, {config.val_ratio*100:.0f}% Val, {config.test_ratio*100:.0f}% Test", "info")
        
        scaler_x_path, scaler_y_path = os.path.join(run_dir, "scaler_x.pkl"), os.path.join(run_dir, "scaler_y.pkl")
        joblib.dump(scaler_X, scaler_x_path)
        joblib.dump(scaler_y, scaler_y_path)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=config.test_ratio, random_state=config.seed)
        val_rel = config.val_ratio / (config.train_ratio + config.val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_rel, random_state=config.seed)

        def on_epoch_end(epoch, num_epochs, loss, val_loss, r2):
            # Update progress
            current_progress = int((run.current_trial / config.optuna_trials) * 100)
            run.progress = current_progress
            db.commit()
            
            # Broadcast metrics for live chart
            metric_data = {
                "trial": run.current_trial,
                "loss": loss,
                "r2": r2,
                "mae": 0, # MAE usually calculated at checkpoint
                "val_loss": val_loss
            }
            state.broadcast_sync({"type": "metrics", "data": metric_data})
            
            if epoch % 10 == 0:
                db_log_and_broadcast(db, run_id, f"Epoch {epoch}/{num_epochs}: val_loss={val_loss:.6f}, r2={r2:.4f}", "info", epoch=epoch)

        def on_checkpoint(trial_num, model, loss, r2, mae):
            checkpoint_path = os.path.join(run_dir, f"best_model_trial_{trial_num}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_choice": config.model_choice,
                "input_size": X_train.shape[1],
                "r2": r2,
                "mae": mae,
                "scaler_x_path": scaler_x_path,
                "scaler_y_path": scaler_y_path,
                "config": model.config if hasattr(model, 'config') else {}
            }, checkpoint_path)
            
            best_model_path = os.path.join(run_dir, "best_model.pt")
            is_new_best = False
            if not os.path.exists(best_model_path) or r2 > (run.best_r2 if run.best_r2 is not None else -float('inf')):
                shutil.copy2(checkpoint_path, best_model_path)
                run.best_r2 = r2
                db.commit()
                is_new_best = True

            db.add(ModelCheckpoint(
                run_id=run.id, 
                model_path=checkpoint_path, 
                scaler_path=scaler_x_path, 
                r2_score=r2, 
                params={}
            ))
            
            # Append to metrics history in DB
            history = list(run.metrics_history) if run.metrics_history else []
            metric_point = {"trial": trial_num, "loss": loss, "r2": r2, "mae": mae, "val_loss": loss}
            history.append(metric_point)
            run.metrics_history = history
            db.commit()
            
            # Broadcast structured metric and status update
            state.broadcast_sync({"type": "metrics", "data": metric_point})
            if is_new_best:
                state.broadcast_sync({"type": "status", "data": {
                    "is_running": True,
                    "progress": run.progress,
                    "run_id": run_id,
                    "current_trial": trial_num,
                    "total_trials": config.optuna_trials,
                    "result": {"best_r2": run.best_r2}
                }})

            db_log_and_broadcast(db, run_id, f"Checkpoint saved for Trial #{trial_num} (R²: {r2:.4f})", "success")

        params = config.model_dump()
        params['on_epoch_end'] = on_epoch_end
        
        # Wrapped objective to track trial number
        class TrackedObjective(Objective):
            def __call__(self, trial):
                run.current_trial = trial.number
                db.commit()
                # Fetch latest run to get updated metrics_history and best_r2
                updated_run = db.query(Run).filter(Run.run_id == run_id).first()
                state.broadcast_sync({"type": "status", "data": {
                    "is_running": True,
                    "progress": updated_run.progress,
                    "run_id": run_id,
                    "current_trial": trial.number,
                    "total_trials": config.optuna_trials,
                    "metrics_history": updated_run.metrics_history if updated_run else [],
                    "result": {"best_r2": updated_run.best_r2} if updated_run and updated_run.best_r2 else None
                }})
                # Broadcast current trial's initial metrics to update frontend charts
                trial_metrics = {
                    "trial": trial.number,
                    "loss": None, # Will be updated by on_epoch_end
                    "r2": None,   # Will be updated by on_epoch_end
                    "mae": None,
                    "val_loss": None
                }
                state.broadcast_sync({"type": "metrics", "data": trial_metrics})
                return super().__call__(trial)

        obj = TrackedObjective(
            config.model_choice, X_train, y_train, X_val, y_val, 
            device, config.patience, params, on_checkpoint=on_checkpoint
        )

        study = run_optimization(f"NFTool_{config.model_choice}", config.optuna_trials, None, obj)
        
        # Final Reporting
        db_log_and_broadcast(db, run_id, "Generating diagnostic reports...", "info")
        analyze_optuna_study(study, run_dir, run_id)
        
        # Generate Regression Plots with Best Model
        best_model_path = os.path.join(run_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            from src.models.architectures import model_factory
            best_model = model_factory(config.model_choice, X_train.shape[1], checkpoint.get("config", {}), device)
            best_model.load_state_dict(checkpoint["model_state_dict"])
            best_model.eval()
            
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            if config.model_choice == "CNN":
                X_test_tensor = preprocess_for_cnn(X_test).to(device)
                
            with torch.no_grad():
                y_pred = best_model(X_test_tensor).cpu().numpy().flatten()
                
            y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            generate_regression_plots(y_test_orig, y_pred_orig, RESULTS_DIR) # For current display
            generate_regression_plots(y_test_orig, y_pred_orig, run_dir)    # For history
            
        run.status = "completed"
        db.commit()
        db_log_and_broadcast(db, run_id, f"Optimization finished. Best R²: {run.best_r2:.4f}", "success")
        
    except Exception as e:
        db_log_and_broadcast(db, run_id, f"Critical training error: {str(e)}", "warn")
        if run: run.status = "failed"; db.commit()
    finally:
        if 'device' in locals() and device.type == "cuda": torch.cuda.empty_cache()
        if run:
            run.progress = 100
            db.commit()
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
    
    db = SessionLocal()
    try:
        latest_run = db.query(Run).order_by(Run.timestamp.desc()).first()
        is_running = state.active_process is not None and state.active_process.is_alive()
        
        init_data = {
            "is_running": is_running, 
            "progress": latest_run.progress if latest_run else 0, 
            "current_trial": latest_run.current_trial if latest_run else 0, 
            "total_trials": latest_run.optuna_trials if latest_run else 0, 
            "logs": latest_run.logs if latest_run else [], 
            "metrics_history": latest_run.metrics_history if latest_run else [], 
            "result": {"best_r2": latest_run.best_r2} if latest_run and latest_run.best_r2 else None,
            "hardware_stats": state.hardware_stats
        }
        await websocket.send_text(json.dumps({"type": "init", "data": init_data}))
        
        while True:
            data = await websocket.receive_text()
            if data == "ping": 
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log_print(f"WebSocket error: {e}")
    finally:
        db.close()
        async with state.client_lock:
            if websocket in state.clients: state.clients.remove(websocket)

from fastapi.responses import FileResponse

@app.get("/download-weights/{run_id}", dependencies=[Depends(verify_api_key)])
async def download_weights(run_id: str, db: Session = Depends(get_db)):
    checkpoint = db.query(ModelCheckpoint).join(Run).filter(Run.run_id == run_id).order_by(ModelCheckpoint.id.desc()).first()
    if not checkpoint or not os.path.exists(checkpoint.model_path):
        raise HTTPException(status_code=404, detail="Model weights not found for this run")
    return FileResponse(path=checkpoint.model_path, filename=f"{run_id}_weights.pt")

@app.get("/gpus", dependencies=[Depends(verify_api_key)])
def list_gpus():
    return [{"id": i, "name": torch.cuda.get_device_name(i), "is_available": True} for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []

@app.get("/health")
@app.head("/health")
def health_check():
    return {"status": "ok", "gpu": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}

@app.post("/train", dependencies=[Depends(verify_api_key)])
async def start_training(config: TrainingConfig, db: Session = Depends(get_db)):
    if state.active_process is not None:
        if state.active_process.is_alive():
            raise HTTPException(status_code=400, detail="Training already in progress")
        else:
            state.active_process.join()
            state.active_process = None

    run_id = f"PASS_{datetime.now().strftime('%H%M%S%f')[:9]}"
    db.add(Run(run_id=run_id, model_choice=config.model_choice, status="running", optuna_trials=config.optuna_trials, config=config.model_dump(), logs=[]))
    db.commit()
    
    state.active_run_id, state.current_gpu_id = run_id, config.gpu_id
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=run_training_task, args=(config.model_dump(), run_id))
    process.start(); state.active_process = process
    return {"message": "Training started", "run_id": run_id}

@app.post("/abort", dependencies=[Depends(verify_api_key)])
async def abort_training(db: Session = Depends(get_db)):
    if state.active_process and state.active_process.is_alive():
        state.active_process.terminate(); state.active_process.join()
        run = db.query(Run).filter(Run.run_id == state.active_run_id).first()
        if run: run.status = "aborted"; db.commit()
        await log_and_broadcast("ABORT SIGNAL SENT: Process terminated.", "warn")
        await state.broadcast({"type": "status", "data": {"is_running": False, "is_aborting": False, "progress": 0}})
        state.active_process = state.active_run_id = None
    return {"status": "aborted"}

@app.post("/load-weights", dependencies=[Depends(verify_api_key)])
async def load_weights(file: UploadFile = File(...)):
    if not file.filename.endswith((".pt", ".pth")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .pt or .pth allowed.")
    
    upload_dir = os.path.join(WORKSPACE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    target_path = os.path.join(upload_dir, file.filename)
    
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        checkpoint = torch.load(target_path, map_location="cpu")
        if "model_state_dict" not in checkpoint:
            os.remove(target_path)
            raise HTTPException(status_code=400, detail="Invalid model file: missing state dict")
        
        return {
            "message": "Weights loaded successfully",
            "path": target_path,
            "info": {
                "r2": checkpoint.get("r2", "N/A"),
                "mae": checkpoint.get("mae", "N/A"),
                "model": checkpoint.get("model_choice", "Unknown")
            }
        }
    except Exception as e:
        if os.path.exists(target_path): os.remove(target_path)
        raise HTTPException(status_code=400, detail=f"Failed to load weights: {str(e)}")

@app.get("/dataset/preview", dependencies=[Depends(verify_api_key)])
async def preview_dataset(path: str, rows: int = 10):
    target = safe_path(path)
    if not os.path.exists(target): raise HTTPException(status_code=404, detail="File not found")
    
    df = load_dataset(target)
    preview = df.head(rows)
    
    stats = {
        "count": len(df),
        "columns": df.shape[1],
        "mean": df.mean().tolist(),
        "std": df.std().tolist(),
        "min": df.min().tolist(),
        "max": df.max().tolist(),
        "missing": df.isnull().sum().sum().item()
    }
    
    return {
        "headers": [f"Feature_{i}" for i in range(df.shape[1])],
        "rows": preview.values.tolist(),
        "shape": list(df.shape),
        "total_rows": len(df),
        "stats": stats
    }

@app.post("/inference", dependencies=[Depends(verify_api_key)])
async def run_inference(model_path: str, features: List[float], db: Session = Depends(get_db)):
    checkpoint = db.query(ModelCheckpoint).join(Run).filter(Run.run_id == model_path).order_by(ModelCheckpoint.id.desc()).first()
    
    if checkpoint:
        target = checkpoint.model_path
    else:
        target = safe_path(model_path)
        
    if not os.path.exists(target): raise HTTPException(status_code=404, detail="Model file not found")
    checkpoint_data = torch.load(target, map_location="cpu")
    input_size = checkpoint_data.get("input_size")
    if len(features) != input_size: raise HTTPException(status_code=400, detail=f"Expected {input_size} features, got {len(features)}")
    
    scaler_x_path = checkpoint_data.get("scaler_x_path")
    scaler_y_path = checkpoint_data.get("scaler_y_path")
    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        raise HTTPException(status_code=400, detail="Model scalers missing")
        
    scaler_X, scaler_y = joblib.load(scaler_x_path), joblib.load(scaler_y_path)
    from src.models.architectures import model_factory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_factory(checkpoint_data.get("model_choice", "NN"), input_size, checkpoint_data.get("config", {}), device)
    model.load_state_dict(checkpoint_data["model_state_dict"]); model.eval()
    X_scaled = scaler_X.transform(np.array([features]))
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    if checkpoint_data.get("model_choice") == "CNN":
        from src.data.processing import preprocess_for_cnn
        X_tensor = preprocess_for_cnn(X_scaled).to(device)
    with torch.no_grad():
        prediction_raw = scaler_y.inverse_transform(model(X_tensor).cpu().numpy().reshape(-1, 1)).flatten()[0]
    return {"prediction": float(prediction_raw), "model_r2": checkpoint_data.get("r2", 0.0)}

@app.get("/runs", dependencies=[Depends(verify_api_key)])
def list_runs(db: Session = Depends(get_db)):
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
