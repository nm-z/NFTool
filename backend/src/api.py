import os
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import optuna
from datetime import datetime
import asyncio
import json
import psutil
import subprocess
import re
from sqlalchemy import create_engine, Column, Integer, String, Float, JSON, ForeignKey, DateTime
from sqlalchemy.orm import sessionmaker, relationship, Session, declarative_base

import multiprocessing
import pickle
import joblib

# Configuration
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
WORKSPACE_DIR = os.path.join(REPO_ROOT, "workspace")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(WORKSPACE_DIR, 'nftool.db')}")
API_KEY = os.getenv("API_KEY", "nftool-dev-key")
LOGS_DIR = os.path.join(WORKSPACE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

import logging
import sys

# Configure logging to write to workspace/logs/api.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "api.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("nftool")

# Redirect print to logger for centralized logging
def log_print(*args, **kwargs):
    msg = " ".join(map(str, args))
    logger.info(msg)

# ... existing code ...

# Database Setup
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

from fastapi.security import APIKeyHeader
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    run_id = Column(String, unique=True, index=True) # e.g. PASS_7721
    model_choice = Column(String)
    status = Column(String) # running, completed, aborted, failed
    progress = Column(Integer, default=0)
    current_trial = Column(Integer, default=0)
    best_r2 = Column(Float, nullable=True)
    optuna_trials = Column(Integer)
    config = Column(JSON)
    report_path = Column(String, nullable=True)
    metrics_history = Column(JSON, nullable=True)
    logs = Column(JSON, default=[])

class ModelCheckpoint(Base):
    __tablename__ = "checkpoints"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"))
    timestamp = Column(DateTime, default=datetime.now)
    model_path = Column(String)
    scaler_path = Column(String)
    r2_score = Column(Float)
    params = Column(JSON)
    
    run = relationship("Run", back_populates="checkpoints")

Run.checkpoints = relationship("ModelCheckpoint", back_populates="run")

Base.metadata.create_all(bind=engine)

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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "runs/results")
REPORTS_DIR = os.path.join(WORKSPACE_DIR, "runs/reports")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

# Global state for training status and logs
def safe_path(relative_path: str):
    """Sanitize and validate path to prevent directory traversal"""
    try:
        # Resolve to absolute path
        base_path = Path(REPO_ROOT).resolve()
        target_path = (base_path / relative_path).resolve()
        
        # Define allowed roots
        workspace_root = Path(WORKSPACE_DIR).resolve()
        data_root = (base_path / "data").resolve()
        
        # Check if target is within workspace or data directories
        if target_path.is_relative_to(workspace_root) or target_path.is_relative_to(data_root):
            return str(target_path)
            
        raise HTTPException(status_code=403, detail="Access denied: Path outside allowed directories")
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid path structure")

class TrainingState:
    def __init__(self):
        self.active_run_id = None
        self.active_process = None
        self.clients: List[WebSocket] = []
        self.client_lock = asyncio.Lock()
        self.loop = None
        self.hardware_stats: Dict[str, Any] = {}

    def set_loop(self, loop):
        self.loop = loop

    async def broadcast(self, message: Dict[str, Any]):
        async with self.client_lock:
            if not self.clients:
                return
            
            # Remove disconnected clients while broadcasting
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

from pydantic import BaseModel, Field, field_validator, model_validator

class TrainingConfig(BaseModel):
    model_choice: str = Field(..., pattern="^(NN|CNN)$")
    seed: int = Field(default=42, ge=0)
    patience: int = Field(default=100, ge=1)
    train_ratio: float = Field(default=0.7, ge=0.1, le=0.9)
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.45)
    test_ratio: float = Field(default=0.15, ge=0.05, le=0.45)
    optuna_trials: int = Field(default=10, ge=1, le=1000)
    optimizers: List[str] = Field(default=["AdamW"])
    n_layers_min: int = Field(default=1, ge=1, le=20)
    n_layers_max: int = Field(default=8, ge=1, le=20)
    l_size_min: int = Field(default=128, ge=8, le=2048)
    l_size_max: int = Field(default=1024, ge=8, le=2048)
    lr_min: float = Field(default=1e-4, ge=1e-7, le=1.0)
    lr_max: float = Field(default=1e-3, ge=1e-7, le=1.0)
    drop_min: float = Field(default=0.0, ge=0.0, le=0.9)
    drop_max: float = Field(default=0.0, ge=0.0, le=0.9)
    h_dim_min: float = Field(default=32, ge=8, le=1024)
    h_dim_max: float = Field(default=256, ge=8, le=1024)
    # CNN specific
    conv_blocks_min: int = Field(default=1, ge=1, le=10)
    conv_blocks_max: int = Field(default=5, ge=1, le=10)
    kernel_size: int = Field(default=3, ge=1, le=15)
    # Performance/System
    gpu_throttle_sleep: float = Field(default=0.1, ge=0.0, le=5.0)
    cnn_filter_cap_min: int = Field(default=512, ge=16, le=4096)
    cnn_filter_cap_max: int = Field(default=512, ge=16, le=4096)
    max_epochs: int = Field(default=200, ge=1, le=10000)
    device: str = Field(default="cuda", pattern="^(cuda|cpu)$")
    gpu_id: int = Field(default=0, ge=0)
    predictor_path: str
    target_path: str

    @model_validator(mode='after')
    def validate_ratios(self) -> 'TrainingConfig':
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Ratios must sum to 1.0 (got {total:.2f})")
        return self

    @field_validator('predictor_path', 'target_path')
    @classmethod
    def validate_paths(cls, v: str) -> str:
        # Check path existence via safe_path
        try:
            target = safe_path(v)
            if not os.path.exists(target):
                raise ValueError(f"File not found: {v}")
        except HTTPException as e:
            raise ValueError(f"Invalid path: {e.detail}")
        return v

async def log_and_broadcast(msg: str, type: str = "default"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {"time": timestamp, "msg": msg, "type": type}
    await state.broadcast({"type": "log", "data": log_entry})
    log_print(f"[{timestamp}] {msg}")

def log_and_broadcast_sync(msg: str, type: str = "default"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {"time": timestamp, "msg": msg, "type": type}
    state.broadcast_sync({"type": "log", "data": log_entry})
    log_print(f"[{timestamp}] {msg}")

def db_log_and_broadcast(db: Session, run_id: str, msg: str, type: str = "default"):
    """Update DB with log and broadcast to WS clients"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {"time": timestamp, "msg": msg, "type": type}
    
    # Update DB
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if run:
        current_logs = list(run.logs) if run.logs else []
        current_logs.append(log_entry)
        run.logs = current_logs
        db.commit()
    
    # Broadcast
    state.broadcast_sync({"type": "log", "data": log_entry})
    log_print(f"[{timestamp}] {msg}")

def get_gpu_stats(gpu_id: int = 0):
    """Get GPU stats from rocm-smi with accurate VRAM and Use parsing"""
    try:
        def parse_rocm_json(output):
            """Helper to extract and parse JSON from rocm-smi output that might contain warnings"""
            try:
                # Find the first '{' and last '}'
                start = output.find('{')
                end = output.rfind('}')
                if start != -1 and end != -1:
                    return json.loads(output[start:end+1])
            except:
                pass
            return None

        # Get usage and temp
        res_use = subprocess.run(
            ["rocm-smi", "--showuse", "--showtemp", "--json"],
            capture_output=True, text=True, timeout=2
        )
        # Get memory info in bytes
        res_mem = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True, text=True, timeout=2
        )
        
        gpu_stats = {
            "vram_total_gb": 0.0,
            "vram_used_gb": 0.0,
            "vram_percent": 0,
            "gpu_use_percent": 0,
            "gpu_temp_c": 0
        }

        target_card = f"card{gpu_id}"
        target_device = f"device{gpu_id}"
        target_idx = str(gpu_id)

        if res_use.returncode == 0:
            data = parse_rocm_json(res_use.stdout)
            if data:
                try:
                    # Find the specific card entry
                    card = next((k for k in [target_card, target_device, target_idx] if k in data), None)
                    if not card and len(data) > 0:
                        card = list(data.keys())[0]
                        
                    if card:
                        # Try various common keys for GPU usage
                        use_val = data[card].get("GPU use (%)") or \
                                  data[card].get("GPU use") or \
                                  data[card].get("GPU usage (%)") or 0
                        gpu_stats["gpu_use_percent"] = int(float(str(use_val).strip('%')))
                        
                        temp = data[card].get("Temperature (Sensor edge) (C)") or \
                               data[card].get("Temperature (Sensor junction) (C)") or \
                               data[card].get("Temperature (C)") or 0
                        gpu_stats["gpu_temp_c"] = int(float(temp))
                except Exception as e:
                    log_print(f"Error parsing ROCm use JSON: {e}")
            else:
                # Fallback to regex if JSON fails
                output = res_use.stdout
                # This fallback might be harder for specific IDs without more regex work
                use_match = re.search(r'(?:GPU use \(%\)|"GPU use \(%\)"|GPU use):?\s+["\']?([\d\.]+)["\']?', output, re.I)
                if use_match:
                    gpu_stats["gpu_use_percent"] = int(float(use_match.group(1)))
                temp_match = re.search(r'(?:Temperature|temp):?\s+([\d\.]+)', output, re.I)
                if temp_match:
                    gpu_stats["gpu_temp_c"] = int(float(temp_match.group(1)))

        if res_mem.returncode == 0:
            data = parse_rocm_json(res_mem.stdout)
            if data:
                try:
                    card = next((k for k in [target_card, target_device, target_idx] if k in data), None)
                    if not card and len(data) > 0:
                        card = list(data.keys())[0]
                        
                    if card:
                        # Try bytes first
                        total_b = int(data[card].get("VRAM Total Memory (B)", 0))
                        used_b = int(data[card].get("VRAM Total Used Memory (B)", 0))
                        
                        # Fallback to MB if bytes are 0
                        if total_b == 0:
                            total_b = int(data[card].get("VRAM Total Memory (MiB)", 0)) * 1024 * 1024
                            used_b = int(data[card].get("VRAM Total Used Memory (MiB)", 0)) * 1024 * 1024
                            
                        gpu_stats["vram_total_gb"] = round(total_b / (1024**3), 2)
                        gpu_stats["vram_used_gb"] = round(used_b / (1024**3), 2)
                        gpu_stats["vram_percent"] = int((used_b / total_b * 100)) if total_b > 0 else 0
                except Exception as e:
                    log_print(f"Error parsing ROCm mem JSON: {e}")
            else:
                output = res_mem.stdout
                total_match = re.search(r'VRAM Total Memory \(B\):\s+(\d+)', output)
                used_match = re.search(r'VRAM Total Used Memory \(B\):\s+(\d+)', output)
                if total_match and used_match:
                    total_b = int(total_match.group(1))
                    used_b = int(used_match.group(1))
                    gpu_stats["vram_total_gb"] = round(total_b / (1024**3), 2)
                    gpu_stats["vram_used_gb"] = round(used_b / (1024**3), 2)
                    gpu_stats["vram_percent"] = int((used_b / total_b * 100)) if total_b > 0 else 0

        return gpu_stats
    except Exception as e:
        log_print(f"ROCm SMI Error: {e}")
        return {
            "vram_total_gb": 0.0, "vram_used_gb": 0.0, "vram_percent": 0,
            "gpu_use_percent": 0, "gpu_temp_c": 0
        }

def get_system_stats():
    """Get CPU and RAM stats using psutil"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": cpu_percent,
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_used_gb": round(mem.used / (1024**3), 2),
        "ram_percent": mem.percent
    }

async def hardware_monitor_task():
    """Background task to poll hardware stats and broadcast via WebSocket"""
    while True:
        try:
            # Only poll if we have connected clients
            if not state.clients:
                await asyncio.sleep(5)
                continue

            # Poll every 2s if running, every 10s if idle
            is_running = state.active_process is not None and state.active_process.is_alive()
            interval = 2 if is_running else 10
            
            # Use specific GPU ID if training state has one, else default to 0
            gpu_id = 0
            if is_running and hasattr(state, 'current_gpu_id'):
                gpu_id = state.current_gpu_id
                
            gpu_stats = get_gpu_stats(gpu_id)
            system_stats = get_system_stats()
            state.hardware_stats = {**gpu_stats, **system_stats}
            await state.broadcast({
                "type": "hardware",
                "data": state.hardware_stats
            })
            await asyncio.sleep(interval)
        except Exception as e:
            log_print(f"Hardware monitor error: {e}")
            await asyncio.sleep(5)

def run_training_task(config_dict: Dict[str, Any], run_id: str):
    """Standalone worker process function"""
    db = SessionLocal()
    
    try:
        run = db.query(Run).filter(Run.run_id == run_id).first()
        if not run:
            log_print(f"Error: Run {run_id} not found in DB")
            return

        # Convert dict back to TrainingConfig for validation
        config = TrainingConfig(**config_dict)
        
        db_log_and_broadcast(db, run_id, f"Starting training engine for {run_id}...", "info")
        
        # Select device
        if config.device == "cuda" and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_id = min(config.gpu_id, gpu_count - 1)
            device = torch.device(f"cuda:{gpu_id}")
            device_name = torch.cuda.get_device_name(gpu_id)
        else:
            device = torch.device("cpu")
            device_name = "CPU"
            
        db_log_and_broadcast(db, run_id, f"Using device: {device_name}", "success")

        # Load Data
        predictor_path = safe_path(config.predictor_path)
        target_path = safe_path(config.target_path)
        
        db_log_and_broadcast(db, run_id, f"Loading datasets...", "default")
        
        if not os.path.exists(predictor_path) or not os.path.exists(target_path):
            raise FileNotFoundError("One or both dataset files not found")

        df_X = pd.read_csv(predictor_path, header=None)
        df_y = pd.read_csv(target_path, header=None).dropna()
        
        if len(df_X) == 0 or len(df_y) == 0:
            raise ValueError("Datasets cannot be empty")

        min_len = min(len(df_X), len(df_y))
        if len(df_X) != len(df_y):
            db_log_and_broadcast(db, run_id, f"Warning: Dataset length mismatch ({len(df_X)} vs {len(df_y)}). Truncating to {min_len}.", "warn")
            
        X_raw, y_raw = df_X.iloc[:min_len].values, df_y.iloc[:min_len].values.flatten()
        
        # Scale Predictors
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_raw)
        
        # Scale Targets
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()
        
        # Create run directory and save scalers
        run_dir = os.path.join(REPORTS_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)
        scaler_x_path = os.path.join(run_dir, "scaler_x.pkl")
        scaler_y_path = os.path.join(run_dir, "scaler_y.pkl")
        joblib.dump(scaler_X, scaler_x_path)
        joblib.dump(scaler_y, scaler_y_path)
        
        db_log_and_broadcast(db, run_id, f"Data loaded: {X_raw.shape[0]} samples, {X_raw.shape[1]} features", "success")

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=config.test_ratio, random_state=config.seed)
        val_rel = config.val_ratio / (config.train_ratio + config.val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_rel, random_state=config.seed)

        metrics_history = []

        def objective(trial):
            # Update DB state
            run.current_trial = trial.number + 1
            run.progress = int((trial.number) / config.optuna_trials * 100)
            db.commit()

            state.broadcast_sync({
                "type": "status", 
                "data": {
                    "is_running": True, 
                    "progress": run.progress,
                    "current_trial": run.current_trial,
                    "total_trials": config.optuna_trials
                }
            })

            def on_epoch_end(epoch, num_epochs, loss, val_loss, r2):
                if epoch % 20 == 0:
                    db_log_and_broadcast(db, run_id, f"Trial #{trial.number} - Epoch {epoch}/{num_epochs}: loss={loss:.4f}, r2={r2:.4f}", "info")

            optimizer = trial.suggest_categorical("optimizer", config.optimizers)
            lr = trial.suggest_float("lr", config.lr_min, config.lr_max, log=True)
            dropout = trial.suggest_float("dropout", config.drop_min, config.drop_max)
            
            try:
                if config.model_choice == "NN":
                    n_layers = trial.suggest_int("num_layers", config.n_layers_min, config.n_layers_max)
                    l_size = trial.suggest_int("layer_size", config.l_size_min, config.l_size_max)
                    cfg = {'layers': [l_size] * n_layers, 'dropout': dropout, 'lr': lr, 'optimizer': optimizer}
                    model, loss, history = train_model(
                        torch.tensor(X_train, dtype=torch.float32).to(device),
                        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device),
                        torch.tensor(X_val, dtype=torch.float32).to(device),
                        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device),
                        X_train.shape[1], cfg, device, config.patience,
                        num_epochs=config.max_epochs,
                        gpu_throttle_sleep=config.gpu_throttle_sleep,
                        on_epoch_end=on_epoch_end
                    )
                else: # CNN
                    n_conv = trial.suggest_int("num_conv_blocks", config.conv_blocks_min, config.conv_blocks_max)
                    base_filters = trial.suggest_int("base_filters", config.l_size_min, config.l_size_max)
                    h_dim = trial.suggest_int("hidden_dim", int(config.h_dim_min), int(config.h_dim_max))
                    current_cap = trial.suggest_int("cnn_filter_cap", config.cnn_filter_cap_min, config.cnn_filter_cap_max)
                    conv_layers = [{'out_channels': min(base_filters * (2**i), current_cap), 'kernel': config.kernel_size, 'pool': 2} for i in range(n_conv)]
                    cfg = {'conv_layers': conv_layers, 'hidden_dim': h_dim, 'dropout': dropout, 'lr': lr, 'optimizer': optimizer}
                    model, loss, history = train_cnn_model(X_train, y_train, X_val, y_val, cfg, device, config.patience,
                        num_epochs=config.max_epochs,
                        gpu_throttle_sleep=config.gpu_throttle_sleep,
                        on_epoch_end=on_epoch_end)
            except Exception as e:
                db_log_and_broadcast(db, run_id, f"Error in trial #{trial.number}: {str(e)}", "warn")
                raise e
            
            if history['r2']:
                r2 = history['r2'][-1]
                mae = history['mae'][-1]
                trial.set_user_attr("r2", r2)
                trial.set_user_attr("mae", mae)
                db_log_and_broadcast(db, run_id, f"Trial #{trial.number} complete. R²: {r2:.4f}", "optuna")
                
                metric_entry = {
                    "trial": trial.number + 1,
                    "loss": float(loss),
                    "r2": float(r2),
                    "mae": float(mae),
                    "val_loss": float(history['val'][-1]) if history['val'] else 0.0
                }
                metrics_history.append(metric_entry)
                state.broadcast_sync({"type": "metrics", "data": metric_entry})

                # Save best model checkpoint
                if r2 == max([m['r2'] for m in metrics_history]):
                    model_path = os.path.join(run_dir, "best_model.pt")
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "model_choice": config.model_choice,
                        "input_size": X_train.shape[1],
                        "r2": r2,
                        "mae": mae,
                        "config": cfg,
                        "scaler_x_path": scaler_x_path,
                        "scaler_y_path": scaler_y_path
                    }
                    torch.save(checkpoint, model_path)
                    
                    # Update DB checkpoint
                    cp = ModelCheckpoint(
                        run_id=run.id,
                        model_path=model_path,
                        scaler_path=scaler_x_path,
                        r2_score=r2,
                        params=cfg
                    )
                    db.add(cp)
                    db.commit()
            
            run.progress = int((trial.number + 1) / config.optuna_trials * 100)
            db.commit()
            
            state.broadcast_sync({
                "type": "status", 
                "data": {
                    "is_running": True, 
                    "progress": run.progress,
                    "current_trial": run.current_trial,
                    "total_trials": config.optuna_trials
                }
            })
            
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
            return loss

        db_log_and_broadcast(db, run_id, f"Running {config.optuna_trials} Optuna trials...", "info")
        study = run_optimization(f"NFTool_{config.model_choice}", config.optuna_trials, None, objective)
        
        best_trial = study.best_trial
        run.best_r2 = best_trial.user_attrs.get("r2")
        run.metrics_history = metrics_history
        run.status = "completed"
        run.report_path = run_dir
        db.commit()
        db_log_and_broadcast(db, run_id, f"Optimization finished. Best R²: {run.best_r2:.4f}", "success")

    except Exception as e:
        db_log_and_broadcast(db, run_id, f"Critical training error: {str(e)}", "warn")
        if run:
            run.status = "failed"
            db.commit()
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        db.close()
        state.broadcast_sync({
            "type": "status", 
            "data": {
                "is_running": False, 
                "progress": 100, 
                "run_id": run_id
            }
        })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # Retrieve API Key from Sec-WebSocket-Protocol header
    protocols = websocket.headers.get("Sec-WebSocket-Protocol", "").split(",")
    client_api_key = None
    
        for p in protocols:
            p = p.strip()
            if p.startswith("api-key-"):
                client_api_key = p.replace("api-key-", "")
                break
            
    if client_api_key != API_KEY:
        print(f"WS Unauthorized: client_api_key={client_api_key}")
        await websocket.accept()
        await websocket.send_text(json.dumps({"type": "error", "data": "Unauthorized"}))
        await websocket.close(code=4003)
        return

    # Accept the specific protocol to avoid browser errors
    subprotocol = f"api-key-{client_api_key}" if client_api_key else None
    print(f"WS Accepted. Subprotocol: {subprotocol}")
    await websocket.accept(subprotocol=subprotocol)
    state.set_loop(asyncio.get_event_loop())
    async with state.client_lock:
        state.clients.append(websocket)
    try:
        # Get latest run state from DB
        db = SessionLocal()
        latest_run = db.query(Run).order_by(Run.timestamp.desc()).first()
        db.close()

        # Send initial state
        is_running = state.active_process is not None and state.active_process.is_alive()
        init_data = {
            "type": "init",
            "data": {
                "is_running": is_running,
                "progress": latest_run.progress if latest_run else 0,
                "current_trial": latest_run.current_trial if latest_run else 0,
                "total_trials": latest_run.optuna_trials if latest_run else 0,
                "logs": latest_run.logs if latest_run else [],
                "metrics_history": latest_run.metrics_history if latest_run else [],
                "hardware_stats": state.hardware_stats
            }
        }
        print(f"WS Sending init data for run: {latest_run.run_id if latest_run else 'None'}")
        await websocket.send_text(json.dumps(init_data))
        
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        print("WS Client disconnected")
        async with state.client_lock:
            if websocket in state.clients:
                state.clients.remove(websocket)
    except Exception as e:
        print(f"WS Error: {str(e)}")
        async with state.client_lock:
            if websocket in state.clients:
                state.clients.remove(websocket)

@app.get("/gpus", dependencies=[Depends(verify_api_key)])
def list_gpus():
    """List available GPU devices"""
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpus.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "is_available": True
            })
    return gpus

@app.get("/health")
def health_check():
    return {"status": "ok", "gpu": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}

@app.post("/train", dependencies=[Depends(verify_api_key)])
async def start_training(config: TrainingConfig):
    if state.active_process is not None and state.active_process.is_alive():
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    db = SessionLocal()
    run_id = f"PASS_{datetime.now().strftime('%H%M%S')}"
    new_run = Run(
        run_id=run_id,
        model_choice=config.model_choice,
        status="running",
        optuna_trials=config.optuna_trials,
        config=config.model_dump(),
        logs=[]
    )
    db.add(new_run)
    db.commit()
    db.close()

    state.active_run_id = run_id
    state.current_gpu_id = config.gpu_id
    
    # Start worker process
    process = multiprocessing.Process(
        target=run_training_task,
        args=(config.model_dump(), run_id)
    )
    process.start()
    state.active_process = process
    
    return {"message": "Training started", "run_id": run_id}

@app.post("/abort", dependencies=[Depends(verify_api_key)])
async def abort_training():
    """Abort the current training run"""
    if state.active_process is None or not state.active_process.is_alive():
        return {"message": "No training in progress"}
    
    state.active_process.terminate()
    state.active_process.join()
    
    db = SessionLocal()
    run = db.query(Run).filter(Run.run_id == state.active_run_id).first()
    if run:
        run.status = "aborted"
        db.commit()
    db.close()
    
    await log_and_broadcast("ABORT SIGNAL SENT: Process terminated.", "warn")
    await state.broadcast({
        "type": "status", 
        "data": {
            "is_running": False, 
            "is_aborting": False,
            "progress": 0
        }
    })
    
    state.active_process = None
    state.active_run_id = None
    
    return {"status": "aborted"}

@app.get("/dataset/preview", dependencies=[Depends(verify_api_key)])
async def preview_dataset(path: str, rows: int = 10):
    """Preview first N rows of a CSV file"""
    target = safe_path(path)
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        df = pd.read_csv(target, header=None, nrows=rows)
        return {
            "headers": [f"Feature_{i}" for i in range(df.shape[1])],
            "rows": df.values.tolist(),
            "shape": list(df.shape),
            "total_rows": len(pd.read_csv(target, header=None))
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

@app.post("/inference", dependencies=[Depends(verify_api_key)])
async def run_inference(model_path: str, features: List[float]):
    """Run inference with proper scaling"""
    target = safe_path(model_path)
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    try:
        checkpoint = torch.load(target, map_location="cpu")
        model_choice = checkpoint.get("model_choice", "NN")
        input_size = checkpoint.get("input_size")
        
        if len(features) != input_size:
            raise HTTPException(status_code=400, detail=f"Expected {input_size} features, got {len(features)}")
        
        # Load scalers
        scaler_x_path = checkpoint.get("scaler_x_path")
        scaler_y_path = checkpoint.get("scaler_y_path")
        
        if not scaler_x_path or not os.path.exists(scaler_x_path):
            raise HTTPException(status_code=500, detail="Predictor scaler not found")
        if not scaler_y_path or not os.path.exists(scaler_y_path):
            raise HTTPException(status_code=500, detail="Target scaler not found")
            
        scaler_X = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        
        from src.models.architectures import model_factory
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Reconstruct model config from checkpoint
        model_cfg = checkpoint.get("config", {})
        model = model_factory(model_choice, input_size, model_cfg, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # Scale features
        X_raw = np.array([features])
        X_scaled = scaler_X.transform(X_raw)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        if model_choice == "CNN":
            from src.data.processing import preprocess_for_cnn
            X_tensor = preprocess_for_cnn(X_scaled).to(device)
        else:
            X_tensor = X_tensor.to(device)
        
        with torch.no_grad():
            prediction_scaled = model(X_tensor).cpu().numpy()
            # Inverse scale prediction
            prediction_raw = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]
        
        return {
            "prediction": float(prediction_raw),
            "model_r2": checkpoint.get("r2", 0.0)
        }
    except Exception as e:
        import traceback
        log_print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/runs", dependencies=[Depends(verify_api_key)])
def list_runs(db: Session = Depends(get_db)):
    runs = db.query(Run).order_by(Run.timestamp.desc()).all()
    return runs

@app.get("/datasets", dependencies=[Depends(verify_api_key)])
def list_datasets():
    dataset_dir = os.path.join(REPO_ROOT, "data")
    if not os.path.exists(dataset_dir):
        return []
    # Sort files alphabetically to ensure consistent indexing
    files = sorted([f for f in os.listdir(dataset_dir) if f.endswith(".csv")])
    return [{"name": f, "path": os.path.join("data", f)} for f in files]

@app.get("/results", dependencies=[Depends(verify_api_key)])
def list_results():
    if not os.path.exists(RESULTS_DIR):
        return []
    dirs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    return [{"name": d, "path": os.path.join("workspace/runs/results", d)} for d in sorted(dirs, reverse=True)]

@app.on_event("startup")
async def startup_event():
    """Start hardware monitoring task on app startup"""
    asyncio.create_task(hardware_monitor_task())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
