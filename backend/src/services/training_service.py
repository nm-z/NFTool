"""Training service task runner.

This module contains the long-running training worker invoked by the job
queue. It persists epoch metrics and logs to the database so the main
process can forward them to connected WebSocket clients.
"""

import logging
import os
import shutil
from datetime import datetime
from typing import Any, Protocol

import torch
from src.config import REPORTS_DIR, RESULTS_DIR
from src.data.processing import load_dataset, preprocess_for_cnn
from src.database.database import SESSION_LOCAL
from src.database.models import ModelCheckpoint, Run
from src.manager import manager as connection_manager_instance
from src.models import architectures as architectures_module
from src.schemas.training import TrainingConfig
from src.schemas.websocket import MetricData, TelemetryMessage
from src.services import queue_instance as queue_instance_module
from src.training.engine import Objective, run_optimization
from src.utils.broadcast_utils import db_log_and_broadcast
from src.utils.reporting import analyze_optuna_study, generate_regression_plots

# TYPE AIRLOCK: Use dynamic imports to bypass reportMissingTypeStubs
# This hides the libraries from Pyright's stub crawler but keeps them at runtime.
jb: Any = __import__("joblib")
optuna: Any = __import__("optuna")
# Enable experimental sklearn features before importing model_selection
__import__("sklearn.experimental.enable_halving_search_cv")
sk_model: Any = __import__("sklearn.model_selection", fromlist=["*"])
sk_pre: Any = __import__("sklearn.preprocessing", fromlist=["*"])
np: Any = __import__("numpy")
torch_any: Any = torch


class ConnectionManager(Protocol):
    """Protocol for the connection manager to satisfy strict typing."""

    def broadcast_sync(self, message: TelemetryMessage) -> None:
        """Broadcast a message to all connected clients."""


def _send_metrics_sync(conn_mgr: Any, metric: dict[str, Any]):
    """Send a structured metrics TelemetryMessage synchronously."""
    tm = TelemetryMessage(type="metrics", data=MetricData(**metric))
    conn_mgr.broadcast_sync(tm)


def _send_status_sync(conn_mgr: Any, status_data: dict[str, Any]):
    """Send a structured status TelemetryMessage synchronously."""
    try:
        q_status = queue_instance_module.job_queue.get_status()
        queue_size = q_status["queue_size"]
        active_job_id = q_status["active_job_id"]
        if "queue_size" not in status_data:
            status_data["queue_size"] = queue_size
        if "active_job_id" not in status_data:
            status_data["active_job_id"] = active_job_id
    except (ImportError, RuntimeError, AttributeError):
        status_data.setdefault("queue_size", 0)
        status_data.setdefault("active_job_id", None)

    tm = TelemetryMessage(type="status", data=status_data)
    conn_mgr.broadcast_sync(tm)


def _prepare_data(
    db: Any,
    predictor_path: str,
    target_path: str,
    config: TrainingConfig,
    run_dir: str,
    run_id: str,
    conn_mgr: Any,
) -> tuple[
    Any, Any, Any,
    Any, Any, Any,
    str, str, Any, Any
]:
    """Load datasets, fit scalers, and split into train/val/test."""
    try:
        df_x: Any = load_dataset(predictor_path)
        df_y: Any = load_dataset(target_path)
    except Exception as exc:
        msg = f"Dataset load failed: {exc}"
        db_log_and_broadcast(db, run_id, msg, conn_mgr, "error")
        raise
    df_y = df_y.dropna()
    common_index = df_x.index.intersection(df_y.index)
    df_x = df_x.loc[common_index]
    df_y = df_y.loc[common_index]

    min_len: int = int(min(len(df_x), len(df_y)))
    x_raw: Any = df_x.iloc[:min_len].values
    y_raw: Any = df_y.iloc[:min_len].values.flatten()

    scaler_x = sk_pre.StandardScaler()
    scaler_y = sk_pre.StandardScaler()

    x_scaled: Any = scaler_x.fit_transform(x_raw)
    y_scaled: Any = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

    os.makedirs(run_dir, exist_ok=True)
    scaler_x_path = os.path.join(run_dir, "scaler_x.pkl")
    scaler_y_path = os.path.join(run_dir, "scaler_y.pkl")

    jb.dump(scaler_x, scaler_x_path)
    jb.dump(scaler_y, scaler_y_path)

    split_result = sk_model.train_test_split(
        x_scaled,
        y_scaled,
        test_size=config.test_ratio,
        random_state=config.seed
    )
    x_train: Any = split_result[0]
    x_test: Any = split_result[1]
    y_train: Any = split_result[2]
    y_test: Any = split_result[3]

    np.save(os.path.join(run_dir, "x_test.npy"), x_test)
    np.save(os.path.join(run_dir, "y_test.npy"), y_test)

    val_rel = config.val_ratio / (config.train_ratio + config.val_ratio)
    split_result2 = sk_model.train_test_split(
        x_train, y_train, test_size=val_rel, random_state=config.seed
    )

    x_train_final = split_result2[0]
    x_val: Any = split_result2[1]
    y_train_final = split_result2[2]
    y_val: Any = split_result2[3]

    msg = f"Dataset loaded: {len(x_raw)} samples, {x_raw.shape[1]} features"
    db_log_and_broadcast(db, run_id, msg, conn_mgr, "info")

    return (
        x_train_final, x_val, x_test,
        y_train_final, y_val, y_test,
        scaler_x_path, scaler_y_path,
        scaler_x, scaler_y,
    )


logger = logging.getLogger("nftool")


def run_training_task(
    config_dict: dict[str, Any],
    run_id: str,
    conn_mgr: Any = None,
) -> None:
    """Execute a training run for the provided `config_dict` and `run_id`."""
    with SESSION_LOCAL() as db:
        run = db.query(Run).filter(Run.run_id == run_id).one()
        config = TrainingConfig(**config_dict)

        if conn_mgr is None:
            conn_mgr = connection_manager_instance

        db_log_and_broadcast(
            db, run_id, f"Starting training engine for {run_id}...",
            conn_mgr, "info",
        )

        if config.device == "cuda" and torch.cuda.is_available():
            device_limit = torch.cuda.device_count() - 1
            gpu_id = min(config.gpu_id, device_limit)
            device = torch.device(f"cuda:{gpu_id}")
            device_name = torch.cuda.get_device_name(gpu_id)
            db_log_and_broadcast(
                db, run_id, f"GPU ACCELERATION: {device_name}",
                conn_mgr, "success",
            )
        else:
            device = torch.device("cpu")
            db_log_and_broadcast(
                db, run_id, "Using device: CPU",
                conn_mgr, "info",
            )

        run_dir = os.path.join(REPORTS_DIR, run_id)
        (
            x_train, x_val, x_test,
            y_train, y_val, y_test,
            sx_p, sy_p, _sx, sy,
        ) = _prepare_data(
            db, str(config.predictor_path), str(config.target_path),
            config, run_dir, run_id, conn_mgr,
        )

        def on_epoch_end(epoch: int, n_eps: int, loss: float, v_loss: float, r2: float):
            cur_trial = int(getattr(run, "current_trial", 0))
            prog = int((cur_trial / max(1, config.optuna_trials)) * 100)
            run_any: Any = run
            run_any.progress = prog
            metric_point = {
                "trial": cur_trial, "epoch": epoch, "loss": loss, "r2": r2,
                "mae": 0, "val_loss": v_loss,
            }
            hist = list(getattr(run, "metrics_history", []) or [])
            hist.append(metric_point)
            run_any.metrics_history = hist
            logs = list(getattr(run, "logs", []) or [])
            logs.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "msg": f"Epoch {epoch}/{n_eps}: v_loss={v_loss:.6f}, r2={r2:.4f}",
                "type": "info", "epoch": epoch,
            })
            run_any.logs = logs
            db.commit()

        def on_checkpoint(
            trial_num: int, model: Any,
            loss: float, r2: float, mae: float, epoch: int,
        ):
            _save_checkpoint(
                db, run, run_dir, trial_num, model, loss, r2, mae, epoch,
                sx_p, sy_p, conn_mgr, run_id, config,
            )

        params = config.model_dump()
        params["on_epoch_end"] = on_epoch_end

        obj = _make_tracked_objective(
            db, run, run_id, conn_mgr, config,
            x_train, y_train, x_val, y_val, device, config.patience,
            params, on_checkpoint,
        )

        opt_name = f"NFTool_{config.model_choice}"
        study = run_optimization(opt_name, config.optuna_trials, None, obj)

        _finalize_run(
            db, run, run_dir, study, device, x_test, y_test,
            sy, conn_mgr, run_id, config,
        )


def _save_checkpoint(
    db: Any, run: Any, run_dir: str, trial_num: int, model: Any,
    loss: float, r2: float, mae: float, epoch: int, sx_p: str, sy_p: str,
    conn_mgr: Any, run_id: str, config: Any,
) -> None:
    """Persist a trial checkpoint and update DB history."""
    checkpoint_path = os.path.join(run_dir, f"best_model_trial_{trial_num}.pt")
    torch_any.save({
        "model_state_dict": model.state_dict(),
        "model_choice": config.model_choice if config else "NN",
        "input_size": getattr(model, "input_size", None),
        "r2": r2, "mae": mae,
        "scaler_x_path": sx_p, "scaler_y_path": sy_p,
        "config": getattr(model, "config", {}),
    }, checkpoint_path)

    best_path = os.path.join(run_dir, "best_model.pt")
    prev_r2 = float(getattr(run, "best_r2", -float("inf")) or -float("inf"))
    is_new_best = False
    if not os.path.exists(best_path) or r2 > prev_r2:
        shutil.copy2(checkpoint_path, best_path)
        run.best_r2 = r2
        db.commit()
        is_new_best = True

    db.add(ModelCheckpoint(
        run_id=run.id, model_path=checkpoint_path,
        scaler_path=sx_p, r2_score=r2, params={},
    ))

    metric_point = {
        "trial": trial_num, "epoch": epoch, "loss": loss, "r2": r2,
        "mae": mae, "val_loss": loss
    }
    hist = list(getattr(run, "metrics_history", []) or [])
    hist.append(metric_point)
    run.metrics_history = hist
    db.commit()

    _send_metrics_sync(conn_mgr, metric_point)
    if is_new_best:
        _send_status_sync(conn_mgr, {
            "is_running": True,
            "progress": int(getattr(run, "progress", 0)),
            "run_id": run_id,
            "current_trial": trial_num,
            "total_trials": config.optuna_trials if config else 0,
            "result": {"best_r2": float(getattr(run, "best_r2", 0.0))},
        })

    display_trial = trial_num + 1
    db_log_and_broadcast(
        db, run_id, f"Saved Trial #{display_trial}", conn_mgr, "success"
    )


def _finalize_run(
    db: Any, run: Any, run_dir: str, study: Any, device: Any,
    x_test: Any, y_test: Any, scaler_y: Any,
    conn_mgr: Any, run_id: str, config: Any,
) -> None:
    """Generate final reports, plots, and set run status."""
    db_log_and_broadcast(db, run_id, "Generating final reports...", conn_mgr, "info")
    analyze_optuna_study(study, run_dir, run_id)

    best_path = os.path.join(run_dir, "best_model.pt")
    if os.path.exists(best_path):
        checkpoint: Any = torch_any.load(best_path, map_location=device)
        arch: Any = architectures_module
        factory: Any = arch.model_factory

        c_config = checkpoint.get("config", {})
        best_params = getattr(study, "best_params", {}) or {}
        merged = {**config.model_dump(), **best_params, **c_config}
        m_choice = merged.get("model_choice", config.model_choice)
        if m_choice == "NN":
            if "layers" not in merged:
                n_layers = int(merged.get("num_layers") or config.n_layers_min)
                layer_size = int(merged.get("layer_size") or config.l_size_min)
                merged["layers"] = [layer_size] * n_layers
            merged.setdefault("dropout", float(merged.get("dropout", config.drop_min)))
        elif m_choice == "CNN":
            if "conv_layers" not in merged:
                n_conv = int(merged.get("num_conv_blocks") or config.conv_blocks_min)
                base_filters = int(merged.get("base_filters") or config.l_size_min)
                cap_min = int(merged.get("cnn_filter_cap") or config.cnn_filter_cap_min)
                cap_max = int(merged.get("cnn_filter_cap") or config.cnn_filter_cap_max)
                current_cap = max(
                    cap_min, min(base_filters * (2 ** (n_conv - 1)), cap_max)
                )
                merged["conv_layers"] = [
                    {
                        "out_channels": min(base_filters * (2**i), current_cap),
                        "kernel": int(merged.get("kernel_size") or config.kernel_size),
                        "pool": 2,
                    }
                    for i in range(n_conv)
                ]

        best_model = factory(m_choice, x_test.shape[1], merged, device)
        best_model.load_state_dict(checkpoint["model_state_dict"])
        best_model.eval()

        with torch.no_grad():
            x_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
            if config.model_choice == "CNN":
                x_tensor = preprocess_for_cnn(x_test).to(device)
            y_pred = best_model(x_tensor).cpu().numpy().flatten()

        y_t_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_p_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        generate_regression_plots(y_t_orig, y_p_orig, RESULTS_DIR)
        generate_regression_plots(y_t_orig, y_p_orig, run_dir)

    run.status = "completed"
    db.commit()
    if getattr(device, "type", "") == "cuda":
        torch.cuda.empty_cache()

    _send_status_sync(conn_mgr, {
        "is_running": False,
        "progress": 100,
        "run_id": run_id,
        "current_trial": int(getattr(run, "current_trial", 0)),
        "total_trials": int(getattr(run, "optuna_trials", 0)),
    })


def _make_tracked_objective(
    db: Any, run: Any, run_id: str, conn_mgr: Any, config: TrainingConfig,
    x_train: Any, y_train: Any, x_val: Any, y_val: Any,
    device: Any, patience: int, params: dict[str, Any], on_checkpoint: Any,
) -> Objective:
    """Create an Objective subclass that updates DB/run state and broadcasts status."""

    class TrackedObjective(Objective):
        """Internal objective wrapper that manages DB progress state."""

        def __call__(self, trial: Any):
            run.current_trial = trial.number
            db.commit()
            best_r2_raw = getattr(run, "best_r2", 0.0)
            best_r2 = 0.0 if best_r2_raw is None else float(best_r2_raw)
            _send_status_sync(conn_mgr, {
                "is_running": True,
                "progress": int(getattr(run, "progress", 0)),
                "run_id": run_id,
                "current_trial": trial.number,
                "total_trials": config.optuna_trials,
                "metrics_history": getattr(run, "metrics_history", []),
                "result": {"best_r2": best_r2},
            })
            return super().__call__(trial)

    return TrackedObjective(
        config.model_choice, x_train, y_train, x_val, y_val,
        device, patience, params, on_checkpoint=on_checkpoint,
    )
