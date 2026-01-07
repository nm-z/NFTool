from fastapi import FastAPI, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
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

# Database Setup
DATABASE_URL = "sqlite:///./nftool.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    run_id = Column(String, unique=True, index=True) # e.g. PASS_7721
    model_choice = Column(String)
    status = Column(String) # running, completed, aborted, failed
    best_r2 = Column(Float, nullable=True)
    optuna_trials = Column(Integer)
    config = Column(JSON)
    report_path = Column(String, nullable=True)
    metrics_history = Column(JSON, nullable=True)

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
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files
os.makedirs("/home/nate/Desktop/NFTool/results", exist_ok=True)
os.makedirs("/home/nate/Desktop/NFTool/Training Reports", exist_ok=True)
app.mount("/results", StaticFiles(directory="/home/nate/Desktop/NFTool/results"), name="results")
app.mount("/reports", StaticFiles(directory="/home/nate/Desktop/NFTool/Training Reports"), name="reports")

# Global state for training status and logs
BASE_WORKSPACE = "/home/nate/Desktop/NFTool"

def safe_path(relative_path: str):
    """Sanitize and validate path to prevent directory traversal"""
    # If path is relative, make it absolute based on workspace
    if not os.path.isabs(relative_path):
        full_path = os.path.abspath(os.path.join(BASE_WORKSPACE, relative_path))
    else:
        full_path = os.path.abspath(relative_path)
        
    if not full_path.startswith(os.path.abspath(BASE_WORKSPACE)):
        raise HTTPException(status_code=403, detail="Access denied: Path outside workspace")
    return full_path

class TrainingState:
    def __init__(self):
        self.is_running = False
        self.should_stop = False
        self.progress = 0
        self.current_trial = 0
        self.total_trials = 0
        self.logs = []
        self.result = None
        self.clients: List[WebSocket] = []
        self.loop = None
        self.metrics_history: List[Dict[str, Any]] = []  # Store trial metrics for charts
        self.hardware_stats: Dict[str, Any] = {}

    def set_loop(self, loop):
        self.loop = loop

    async def broadcast(self, message: Dict[str, Any]):
        if not self.clients:
            return
        
        # print(f"BROADCAST: {message['type']}") # Debug
        
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

    def broadcast_sync(self, message: Dict[str, Any]):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)

    def add_log(self, msg: str, type: str = "default"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {"time": timestamp, "msg": msg, "type": type}
        self.logs.append(log_entry)
        return log_entry

state = TrainingState()

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
    cnn_filter_cap: int = Field(default=512, ge=16, le=4096)
    max_epochs: int = Field(default=200, ge=1, le=10000)
    predictor_path: str
    target_path: str

async def log_and_broadcast(msg: str, type: str = "default"):
    entry = state.add_log(msg, type)
    await state.broadcast({"type": "log", "data": entry})
    print(f"[{entry['time']}] {msg}")

def log_and_broadcast_sync(msg: str, type: str = "default"):
    entry = state.add_log(msg, type)
    state.broadcast_sync({"type": "log", "data": entry})
    print(f"[{entry['time']}] {msg}")

def get_gpu_stats():
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

        if res_use.returncode == 0:
            data = parse_rocm_json(res_use.stdout)
            if data:
                try:
                    card = list(data.keys())[0] if data else None
                    if card:
                        # Some versions use "GPU use (%)", some use different keys
                        use_val = data[card].get("GPU use (%)") or data[card].get("GPU use") or 0
                        gpu_stats["gpu_use_percent"] = int(use_val)
                        temp = data[card].get("Temperature (Sensor edge) (C)") or \
                               data[card].get("Temperature (Sensor junction) (C)") or 0
                        gpu_stats["gpu_temp_c"] = int(float(temp))
                except Exception as e:
                    print(f"Error parsing ROCm use JSON: {e}")
            else:
                # Fallback to regex if JSON fails
                output = res_use.stdout
                use_match = re.search(r'(?:GPU use \(%\)|"GPU use \(%\)"):?\s+["\']?(\d+)["\']?', output)
                if use_match:
                    gpu_stats["gpu_use_percent"] = int(use_match.group(1))
                temp_match = re.search(r'Temperature \(Sensor edge\) \(C\):\s+([\d\.]+)', output)
                if temp_match:
                    gpu_stats["gpu_temp_c"] = int(float(temp_match.group(1)))

        if res_mem.returncode == 0:
            data = parse_rocm_json(res_mem.stdout)
            if data:
                try:
                    card = list(data.keys())[0] if data else None
                    if card:
                        total_b = int(data[card].get("VRAM Total Memory (B)", 0))
                        used_b = int(data[card].get("VRAM Total Used Memory (B)", 0))
                        gpu_stats["vram_total_gb"] = round(total_b / (1024**3), 2)
                        gpu_stats["vram_used_gb"] = round(used_b / (1024**3), 2)
                        gpu_stats["vram_percent"] = int((used_b / total_b * 100)) if total_b > 0 else 0
                except Exception as e:
                    print(f"Error parsing ROCm mem JSON: {e}")
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
        print(f"ROCm SMI Error: {e}")
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
            gpu_stats = get_gpu_stats()
            system_stats = get_system_stats()
            state.hardware_stats = {**gpu_stats, **system_stats}
            await state.broadcast({
                "type": "hardware",
                "data": state.hardware_stats
            })
        except Exception as e:
            print(f"Hardware monitor error: {e}")
        await asyncio.sleep(2)  # Poll every 2 seconds

def run_training_task(config: TrainingConfig):
    db = SessionLocal()
    run_id = f"PASS_{datetime.now().strftime('%H%M%S')}"
    new_run = Run(
        run_id=run_id,
        model_choice=config.model_choice,
        status="running",
        optuna_trials=config.optuna_trials,
        config=config.model_dump()
    )
    db.add(new_run)
    db.commit()
    db.refresh(new_run)

    state.is_running = True
    state.should_stop = False
    state.progress = 0
    state.current_trial = 0
    state.total_trials = config.optuna_trials
    state.logs = []
    state.result = None
    state.metrics_history = []  # Clear metrics history for new run
    
    state.broadcast_sync({
        "type": "status", 
        "data": {
            "is_running": True, 
            "progress": 0,
            "current_trial": 0,
            "total_trials": config.optuna_trials
        }
    })

    try:
        log_and_broadcast_sync(f"Starting training engine for {run_id}...", "info")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_and_broadcast_sync(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", "success")

        # Load Data
        log_and_broadcast_sync(f"Loading datasets from {config.predictor_path}...", "default")
        df_X = pd.read_csv(config.predictor_path, header=None)
        df_y = pd.read_csv(config.target_path, header=None).dropna()
        min_len = min(len(df_X), len(df_y))
        X_raw, y_raw = df_X.iloc[:min_len].values, df_y.iloc[:min_len].values.flatten()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        log_and_broadcast_sync(f"Data loaded: {X_raw.shape[0]} samples, {X_raw.shape[1]} features", "success")

        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_raw, test_size=config.test_ratio, random_state=config.seed)
        val_rel = config.val_ratio / (config.train_ratio + config.val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_rel, random_state=config.seed)

        # Optuna Objective
        def objective(trial):
            state.current_trial = trial.number + 1
            state.progress = int((trial.number) / config.optuna_trials * 100)
            state.broadcast_sync({
                "type": "status", 
                "data": {
                    "is_running": True, 
                    "progress": state.progress,
                    "current_trial": state.current_trial,
                    "total_trials": state.total_trials
                }
            })

            def check_stop():
                return state.should_stop

            def on_epoch_end(epoch, num_epochs, loss, val_loss, r2):
                if epoch % 20 == 0:
                    log_and_broadcast_sync(f"Trial #{trial.number} - Epoch {epoch}/{num_epochs}: loss={loss:.4f}, r2={r2:.4f}", "info")

            if state.should_stop:
                trial.study.stop()
                raise optuna.exceptions.TrialPruned("Training aborted by user.")

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
                    X_train.shape[1], cfg, device, config.patience,
                    num_epochs=config.max_epochs,
                    gpu_throttle_sleep=config.gpu_throttle_sleep,
                    check_stop=check_stop, on_epoch_end=on_epoch_end
                )
            else: # CNN
                n_conv = trial.suggest_int("num_conv_blocks", config.conv_blocks_min, config.conv_blocks_max)
                base_filters = trial.suggest_int("base_filters", config.l_size_min, config.l_size_max)
                h_dim = trial.suggest_int("hidden_dim", int(config.h_dim_min), int(config.h_dim_max))
                # Use dynamic filter cap from config
                conv_layers = [{'out_channels': min(base_filters * (2**i), config.cnn_filter_cap), 'kernel': config.kernel_size, 'pool': 2} for i in range(n_conv)]
                cfg = {'conv_layers': conv_layers, 'hidden_dim': h_dim, 'dropout': dropout, 'lr': lr, 'optimizer': optimizer}
                _, loss, history = train_cnn_model(X_train, y_train, X_val, y_val, cfg, device, config.patience,
                    num_epochs=config.max_epochs,
                    gpu_throttle_sleep=config.gpu_throttle_sleep,
                    check_stop=check_stop, on_epoch_end=on_epoch_end)
            
            if state.should_stop:
                trial.study.stop()
                raise optuna.exceptions.TrialPruned("Training aborted by user.")

            if history['r2']:
                r2 = history['r2'][-1]
                mae = history['mae'][-1]
                trial.set_user_attr("r2", r2)
                trial.set_user_attr("mae", mae)
                log_and_broadcast_sync(f"Trial #{trial.number} complete. R²: {r2:.4f}", "optuna")
                
                # Track metrics for charts
                metric_entry = {
                    "trial": trial.number + 1,
                    "loss": float(loss),
                    "r2": float(r2),
                    "mae": float(mae),
                    "val_loss": float(history['val'][-1]) if history['val'] else 0.0
                }
                state.metrics_history.append(metric_entry)
                state.broadcast_sync({
                    "type": "metrics",
                    "data": metric_entry
                })
            
            state.progress = int((trial.number + 1) / config.optuna_trials * 100)
            state.broadcast_sync({
                "type": "status", 
                "data": {
                    "is_running": True, 
                    "progress": state.progress,
                    "current_trial": state.current_trial,
                    "total_trials": state.total_trials
                }
            })
            return loss

        log_and_broadcast_sync(f"Running {config.optuna_trials} Optuna trials...", "info")
        study = run_optimization(f"NFTool_{config.model_choice}", config.optuna_trials, None, objective)
        
        # Filter out pruned trials when finding best trial if aborted
        try:
            best_trial = study.best_trial
            state.result = {
                "best_r2": best_trial.user_attrs.get("r2"),
                "best_params": study.best_params
            }
            log_and_broadcast_sync(f"Optimization finished. Best R²: {best_trial.user_attrs.get('r2', 0):.4f}", "success")
            
            # Update DB with result
            new_run.best_r2 = state.result["best_r2"]
            new_run.metrics_history = state.metrics_history
            db.commit()

        except ValueError:
            log_and_broadcast_sync("Optimization ended with no completed trials.", "warn")
            state.result = None

    except Exception as e:
        log_and_broadcast_sync(f"Error: {str(e)}", "warn")
        new_run.status = "failed"
        db.commit()
    finally:
        state.is_running = False
        state.progress = 100
        
        # Final DB update
        if new_run.status == "running":
            new_run.status = "completed" if not state.should_stop else "aborted"
        db.commit()
        db.close()

        state.broadcast_sync({
            "type": "status", 
            "data": {
                "is_running": False, 
                "progress": 100, 
                "current_trial": state.current_trial,
                "total_trials": state.total_trials,
                "result": state.result
            }
        })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.set_loop(asyncio.get_event_loop())
    state.clients.append(websocket)
    try:
        # Send initial state
        await websocket.send_text(json.dumps({
            "type": "init",
            "data": {
                "is_running": state.is_running,
                "progress": state.progress,
                "current_trial": state.current_trial,
                "total_trials": state.total_trials,
                "logs": state.logs,
                "result": state.result,
                "metrics_history": state.metrics_history,
                "hardware_stats": state.hardware_stats
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

@app.post("/abort")
async def abort_training():
    """Abort the current training run"""
    if not state.is_running:
        return {"message": "No training in progress"}
    state.should_stop = True
    log_and_broadcast_sync("ABORT SIGNAL SENT: Terminating training cores...", "warn")
    return {"status": "aborting"}

@app.get("/dataset/preview")
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

@app.post("/inference")
async def run_inference(model_path: str, features: List[float]):
    """Run inference on a single row of features using a saved model"""
    target = safe_path(model_path)
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    try:
        checkpoint = torch.load(target, map_location="cpu")
        model_choice = checkpoint.get("model_choice", "NN")
        input_size = checkpoint.get("input_size")
        
        if len(features) != input_size:
            raise HTTPException(status_code=400, detail=f"Expected {input_size} features, got {len(features)}")
        
        from src.models.architectures import model_factory
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Reconstruct model config from checkpoint
        if model_choice == "NN":
            layers = checkpoint.get("layers", [512])
            config = {
                "layers": layers,
                "dropout": checkpoint.get("dropout", 0.0),
                "lr": checkpoint.get("lr", 1e-3),
                "optimizer": checkpoint.get("optimizer", "AdamW")
            }
        else:  # CNN
            config = {
                "conv_layers": checkpoint.get("conv_layers", []),
                "hidden_dim": checkpoint.get("hidden_dim", 128),
                "dropout": checkpoint.get("dropout", 0.0),
                "lr": checkpoint.get("lr", 1e-3),
                "optimizer": checkpoint.get("optimizer", "AdamW")
            }
        
        model = model_factory(model_choice, input_size, config, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # Prepare input
        X = torch.tensor([features], dtype=torch.float32)
        if model_choice == "CNN":
            from src.data.processing import preprocess_for_cnn
            X = preprocess_for_cnn(X.numpy()).to(device)
        else:
            X = X.to(device)
        
        with torch.no_grad():
            prediction = model(X).cpu().item()
        
        return {"prediction": prediction, "model_r2": checkpoint.get("r2", 0.0)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/runs")
def list_runs(db: Session = Depends(get_db)):
    runs = db.query(Run).order_by(Run.timestamp.desc()).all()
    return runs

@app.get("/datasets")
def list_datasets():
    dataset_dir = "/home/nate/Desktop/NFTool/dataset"
    if not os.path.exists(dataset_dir):
        return []
    files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
    return [{"name": f, "path": os.path.join(dataset_dir, f)} for f in files]

@app.get("/results")
def list_results():
    results_dir = "/home/nate/Desktop/NFTool/results"
    if not os.path.exists(results_dir):
        return []
    dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    return [{"name": d, "path": os.path.join(results_dir, d)} for d in sorted(dirs, reverse=True)]

@app.on_event("startup")
async def startup_event():
    """Start hardware monitoring task on app startup"""
    asyncio.create_task(hardware_monitor_task())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
