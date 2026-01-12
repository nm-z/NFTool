"""Training service task runner.

This module contains the long-running training worker invoked by the job
queue. It persists epoch metrics and logs to the database so the main
process can forward them to connected WebSocket clients.
"""

import logging
from datetime import datetime
import os
import shutil
from typing import Dict, Any as TypingAny, cast as typing_cast

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import (
    REPORTS_DIR,
    RESULTS_DIR,
)
from src.data.processing import load_dataset, preprocess_for_cnn
from src.database.database import SessionLocal
from src.database.models import ModelCheckpoint, Run
from src.schemas.training import TrainingConfig
from src.schemas.websocket import MetricData, TelemetryMessage
from src.training.engine import Objective, run_optimization
from src.utils.broadcast_utils import db_log_and_broadcast
from src.utils.reporting import analyze_optuna_study, generate_regression_plots


def _send_metrics_sync(conn_mgr: TypingAny, metric: Dict[str, TypingAny]):
    """Send a structured metrics TelemetryMessage synchronously."""
    tm = TelemetryMessage(type="metrics", data=MetricData(**metric))
    conn_mgr.broadcast_sync(tm)


def _send_status_sync(conn_mgr: TypingAny, status_data: Dict[str, TypingAny]):
    """Send a structured status TelemetryMessage synchronously."""
    tm = TelemetryMessage(type="status", data=status_data)
    conn_mgr.broadcast_sync(tm)


def _prepare_data(
    db: TypingAny,
    predictor_path: str,
    target_path: str,
    config: TrainingConfig,
    run_dir: str,
    run_id: str,
    connection_manager: TypingAny,
) -> tuple:
    """Load datasets, fit scalers, split into train/val/test and persist scalers.

    Returns: (X_train, X_val, X_test, y_train, y_val, y_test, scaler_x_path, scaler_y_path, scaler_X, scaler_y)
    """
    df_X = load_dataset(predictor_path)
    df_y = load_dataset(target_path).dropna()
    min_len = min(len(df_X), len(df_y))
    X_raw = df_X.iloc[:min_len].values
    y_raw = df_y.iloc[:min_len].values.flatten()

    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

    os.makedirs(run_dir, exist_ok=True)
    scaler_x_path = os.path.join(run_dir, "scaler_x.pkl")
    scaler_y_path = os.path.join(run_dir, "scaler_y.pkl")
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=config.test_ratio, random_state=config.seed
    )
    val_rel = config.val_ratio / (config.train_ratio + config.val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_rel, random_state=config.seed
    )

    # Ensure numpy arrays for shape/reshape access and downstream processing
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)

    # Small status/log messages
    msg = f"Dataset loaded: {len(X_raw)} samples, {X_raw.shape[1]} features"
    db_log_and_broadcast(db, run_id, msg, connection_manager, "info")
    split_msg = (
        f"Split Ratios: {int(config.train_ratio * 100)}% Train, "
        f"{int(config.val_ratio * 100)}% Val, {int(config.test_ratio * 100)}% Test"
    )
    db_log_and_broadcast(db, run_id, split_msg, connection_manager, "info")

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler_x_path,
        scaler_y_path,
        scaler_X,
        scaler_y,
    )

logger = logging.getLogger("nftool")


def run_training_task(
    config_dict: Dict[str, TypingAny], run_id: str, connection_manager: TypingAny = None
) -> None:
    """Execute a training run for the provided `config_dict` and `run_id`.

    This function runs in a separate process (spawned by the JobQueue). It
    persists epoch-level metrics and logs to the database so the manager
    process can forward them to connected WebSocket clients.
    """
    # Use context manager for DB session and allow exceptions to propagate to the caller.
    with SessionLocal() as db:
        # Let it crash if run doesn't exist; caller will observe the exception.
        run = db.query(Run).filter(Run.run_id == run_id).one()
        config = TrainingConfig(**config_dict)

        if connection_manager is None:
            from src.manager import manager as connection_manager

        db_log_and_broadcast(
            db,
            run_id,
            f"Starting training engine for {run_id}...",
            connection_manager,
            "info",
        )

        if config.device == "cuda" and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_id = min(config.gpu_id, gpu_count - 1)
            device = torch.device(f"cuda:{gpu_id}")
            device_name = torch.cuda.get_device_name(gpu_id)
            db_log_and_broadcast(
                db,
                run_id,
                f"GPU ACCELERATION ENABLED: {device_name}",
                connection_manager,
                "success",
            )
        else:
            device = torch.device("cpu")
            device_name = "CPU"
            db_log_and_broadcast(
                db,
                run_id,
                f"Using device: {device_name}",
                connection_manager,
                "info",
            )

        predictor_path = str(config.predictor_path)
        target_path = str(config.target_path)

        # Load, scale and split the data; persist scalers and emit initial logs.
        run_dir = os.path.join(REPORTS_DIR, run_id)
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            scaler_x_path,
            scaler_y_path,
            _scaler_X,
            scaler_y,
        ) = _prepare_data(
            db,
            predictor_path,
            target_path,
            config,
            run_dir,
            run_id,
            connection_manager,
        )

        # Data prepared and scalers persisted by _prepare_data.

        def on_epoch_end(epoch, num_epochs, loss, val_loss, r2):
            # Update progress
            current_progress = int(
                (
                    typing_cast(int, getattr(run, "current_trial", 0))
                    / max(1, config.optuna_trials)
                )
                * 100
            )
            # Persist progress and metric point to DB so the manager (main process)
            # can detect new metrics/logs and forward them to connected clients.
            typing_cast(TypingAny, run).progress = current_progress
            # Append metric point to run.metrics_history (ensure list)
            metric_point = {
                "trial": typing_cast(int, getattr(run, "current_trial", 0)),
                "loss": loss,
                "r2": r2,
                "mae": 0,
                "val_loss": val_loss,
            }
            current_metrics = list(getattr(run, "metrics_history", []) or [])
            current_metrics.append(metric_point)
            typing_cast(TypingAny, run).metrics_history = current_metrics

            # Append a short log entry for the epoch so it appears in the process stream.
            current_logs = list(getattr(run, "logs", []) or [])
            epoch_msg = "Epoch {}/{}: val_loss={:.6f}, r2={:.4f}".format(
                epoch, num_epochs, val_loss, r2
            )
            current_logs.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "msg": epoch_msg,
                    "type": "info",
                    "epoch": epoch,
                }
            )
            typing_cast(TypingAny, run).logs = current_logs
            db.commit()

        def on_checkpoint(trial_num, model, loss, r2, mae):
            _save_checkpoint(
                db,
                run,
                run_dir,
                trial_num,
                model,
                loss,
                r2,
                mae,
                scaler_x_path,
                scaler_y_path,
                connection_manager,
                run_id,
                config,
            )

        params = config.model_dump()
        params["on_epoch_end"] = on_epoch_end

        obj = _make_tracked_objective(
            db=db,
            run=run,
            run_id=run_id,
            connection_manager=connection_manager,
            config=config,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            device=device,
            patience=config.patience,
            params=params,
            on_checkpoint=on_checkpoint,
        )

        def _run_optuna(
            study_name: str,
            trials: int,
            objective: Objective,
        ):
            """Run Optuna optimization with the provided Objective and return study."""
            return run_optimization(study_name, trials, None, objective)

        study = _run_optuna(f"NFTool_{config.model_choice}", config.optuna_trials, obj)

        _finalize_run(
            db=db,
            run=run,
            run_dir=run_dir,
            study=study,
            device=device,
            X_test=X_test,
            y_test=y_test,
            scaler_y=scaler_y,
            connection_manager=connection_manager,
            run_id=run_id,
            config=config,
        )


def _save_checkpoint(
    db: TypingAny,
    run: TypingAny,
    run_dir: str,
    trial_num: int,
    model: TypingAny,
    loss: float,
    r2: float,
    mae: float,
    scaler_x_path: str = "",
    scaler_y_path: str = "",
    connection_manager: TypingAny = None,
    run_id: str = "",
    config: TypingAny = None,
) -> None:
    """Persist a trial checkpoint, update DB history and broadcast updates."""
    checkpoint_path = os.path.join(run_dir, f"best_model_trial_{trial_num}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_choice": config.model_choice if config is not None else "NN",
            "input_size": model.input_size if hasattr(model, "input_size") else None,
            "r2": r2,
            "mae": mae,
            "scaler_x_path": scaler_x_path,
            "scaler_y_path": scaler_y_path,
            "config": getattr(model, "config", {}) if hasattr(model, "config") else {},
        },
        checkpoint_path,
    )

    best_model_path = os.path.join(run_dir, "best_model.pt")
    is_new_best = False
    previous_best = getattr(run, "best_r2", None)
    prev_val = previous_best if previous_best is not None else -float("inf")
    if not os.path.exists(best_model_path) or r2 > prev_val:
        shutil.copy2(checkpoint_path, best_model_path)
        typing_cast(TypingAny, run).best_r2 = r2
        db.commit()
        is_new_best = True

    db.add(
        ModelCheckpoint(
            run_id=run.id,
            model_path=checkpoint_path,
            scaler_path=scaler_x_path,
            r2_score=r2,
            params={},
        )
    )

    # Append to metrics history in DB
    history = list(getattr(run, "metrics_history", []) or [])
    metric_point = {
        "trial": trial_num,
        "loss": loss,
        "r2": r2,
        "mae": mae,
        "val_loss": loss,
    }
    history.append(metric_point)
    typing_cast(TypingAny, run).metrics_history = history
    db.commit()

    # Broadcast structured metric and status update
    _send_metrics_sync(connection_manager, metric_point)
    if is_new_best:
        status_payload = {
            "is_running": True,
            "progress": typing_cast(int, getattr(run, "progress", 0)),
            "run_id": run_id,
            "current_trial": trial_num,
            "total_trials": config.optuna_trials if config is not None else 0,
            "result": {"best_r2": typing_cast(float, getattr(run, "best_r2", 0.0))},
        }
        _send_status_sync(connection_manager, status_payload)

    db_log_and_broadcast(
        db,
        run_id,
        f"Checkpoint saved for Trial #{trial_num} (R²: {r2:.4f})",
        connection_manager,
        "success",
    )


def _finalize_run(
    db: TypingAny,
    run: TypingAny,
    run_dir: str,
    study: TypingAny,
    device: TypingAny,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler_y: TypingAny,
    connection_manager: TypingAny,
    run_id: str,
    config: TypingAny,
) -> None:
    """Generate final reports, plots, set run status and broadcast final state."""
    db_log_and_broadcast(db, run_id, "Generating diagnostic reports...", connection_manager, "info")
    analyze_optuna_study(study, run_dir, run_id)

    # Generate Regression Plots with Best Model if available
    best_model_path = os.path.join(run_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        from src.models.architectures import model_factory

        checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        merged_config = {**config.model_dump(), **checkpoint_config}
        best_model = model_factory(config.model_choice, X_test.shape[1], merged_config, device)
        best_model.load_state_dict(checkpoint["model_state_dict"])
        best_model.eval()

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        if config.model_choice == "CNN":
            X_test_tensor = preprocess_for_cnn(X_test).to(device)

        with torch.no_grad():
            y_pred = best_model(X_test_tensor).cpu().numpy().flatten()

        y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        generate_regression_plots(y_test_orig, y_pred_orig, RESULTS_DIR)
        generate_regression_plots(y_test_orig, y_pred_orig, run_dir)

    typing_cast(TypingAny, run).status = "completed"
    db.commit()
    msg = f"Optimization finished. Best R²: {typing_cast(float, getattr(run, 'best_r2', 0.0)):.4f}"
    db_log_and_broadcast(db, run_id, msg, connection_manager, "success")

    # Cleanup and final broadcast. Compute final progress from run if available.
    if "device" in locals() and getattr(device, "type", "") == "cuda":
        torch.cuda.empty_cache()

    if run:
        final_progress = 100
        final_trial = typing_cast(int, getattr(run, "current_trial", 0))
        final_total = typing_cast(int, getattr(run, "optuna_trials", 0))
    else:
        final_progress = 0
        final_trial = 0
        final_total = 0

    connection_manager.broadcast_sync(
        TelemetryMessage(
            type="status",
            data={
                "is_running": False,
                "progress": final_progress,
                "run_id": run_id,
                "current_trial": final_trial,
                "total_trials": final_total,
            },
        )
    )


def _make_tracked_objective(
    db: TypingAny,
    run: TypingAny,
    run_id: str,
    connection_manager: TypingAny,
    config: TrainingConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: TypingAny,
    patience: int,
    params: dict,
    on_checkpoint: TypingAny,
) -> Objective:
    """Create an Objective subclass that updates DB/run state and broadcasts status."""

    class TrackedObjective(Objective):
        def __call__(self, trial):
            typing_cast(TypingAny, run).current_trial = trial.number
            db.commit()
            status_payload = {
                "is_running": True,
                "progress": typing_cast(int, getattr(run, "progress", 0)),
                "run_id": run_id,
                "current_trial": trial.number,
                "total_trials": config.optuna_trials,
                "metrics_history": getattr(run, "metrics_history", []),
                "result": {"best_r2": typing_cast(float, getattr(run, "best_r2", 0.0))},
            }
            _send_status_sync(connection_manager, status_payload)
            return super().__call__(trial)

    return TrackedObjective(
        config.model_choice,
        X_train,
        y_train,
        X_val,
        y_val,
        device,
        patience,
        params,
        on_checkpoint=on_checkpoint,
    )
