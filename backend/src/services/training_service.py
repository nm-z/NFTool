import logging
import os
import shutil
from typing import Any, Dict
from typing import Any as TypingAny
from typing import cast as typing_cast

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

logger = logging.getLogger("nftool")


def run_training_task(
    config_dict: Dict[str, Any], run_id: str, connection_manager: TypingAny = None
) -> None:
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
                db, run_id, f"Using device: {device_name}", connection_manager, "info"
            )

        predictor_path = str(config.predictor_path)
        target_path = str(config.target_path)

        df_X = load_dataset(predictor_path)
        df_y = load_dataset(target_path).dropna()
        min_len = min(len(df_X), len(df_y))
        X_raw, y_raw = df_X.iloc[:min_len].values, df_y.iloc[:min_len].values.flatten()

        scaler_X, scaler_y = StandardScaler(), StandardScaler()
        X_scaled = scaler_X.fit_transform(X_raw)
        y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).flatten()

        run_dir = os.path.join(REPORTS_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)

        db_log_and_broadcast(
            db,
            run_id,
            f"Dataset loaded: {len(X_raw)} samples, {X_raw.shape[1]} features",
            connection_manager,
            "info",
        )
        db_log_and_broadcast(
            db,
            run_id,
            f"Split Ratios: {config.train_ratio * 100:.0f}% Train, {config.val_ratio * 100:.0f}% Val, {config.test_ratio * 100:.0f}% Test",
            connection_manager,
            "info",
        )

        scaler_x_path, scaler_y_path = (
            os.path.join(run_dir, "scaler_x.pkl"),
            os.path.join(run_dir, "scaler_y.pkl"),
        )
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

        def on_epoch_end(epoch, num_epochs, loss, val_loss, r2):
            # Update progress
            current_progress = int(
                (
                    typing_cast(int, getattr(run, "current_trial", 0))
                    / max(1, config.optuna_trials)
                )
                * 100
            )
            typing_cast(TypingAny, run).progress = current_progress
            db.commit()

            # Broadcast metrics for live chart
            metric_data = {
                "trial": typing_cast(int, getattr(run, "current_trial", 0)),
                "loss": loss,
                "r2": r2,
                "mae": 0,  # MAE usually calculated at checkpoint
                "val_loss": val_loss,
            }
            connection_manager.broadcast_sync(
                TelemetryMessage(type="metrics", data=MetricData(**metric_data))
            )

            if epoch % 10 == 0:
                db_log_and_broadcast(
                    db,
                    run_id,
                    f"Epoch {epoch}/{num_epochs}: val_loss={val_loss:.6f}, r2={r2:.4f}",
                    connection_manager,
                    "info",
                    epoch=epoch,
                )

        def on_checkpoint(trial_num, model, loss, r2, mae):
            checkpoint_path = os.path.join(run_dir, f"best_model_trial_{trial_num}.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_choice": config.model_choice,
                    "input_size": X_train.shape[1],
                    "r2": r2,
                    "mae": mae,
                    "scaler_x_path": scaler_x_path,
                    "scaler_y_path": scaler_y_path,
                    "config": model.config if hasattr(model, "config") else {},
                },
                checkpoint_path,
            )

            best_model_path = os.path.join(run_dir, "best_model.pt")
            is_new_best = False
            if not os.path.exists(best_model_path) or r2 > (
                getattr(run, "best_r2", None)
                if getattr(run, "best_r2", None) is not None
                else -float("inf")
            ):
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
            connection_manager.broadcast_sync(
                TelemetryMessage(type="metrics", data=MetricData(**metric_point))
            )
            if is_new_best:
                connection_manager.broadcast_sync(
                    TelemetryMessage(
                        type="status",
                        data={
                            "is_running": True,
                            "progress": typing_cast(int, getattr(run, "progress", 0)),
                            "run_id": run_id,
                            "current_trial": trial_num,
                            "total_trials": config.optuna_trials,
                            "result": {
                                "best_r2": typing_cast(
                                    float, getattr(run, "best_r2", 0.0)
                                )
                            },
                        },
                    )
                )

            db_log_and_broadcast(
                db,
                run_id,
                f"Checkpoint saved for Trial #{trial_num} (R²: {r2:.4f})",
                connection_manager,
                "success",
            )

        params = config.model_dump()
        params["on_epoch_end"] = on_epoch_end

        # Wrapped objective to track trial number
        class TrackedObjective(Objective):
            def __call__(self, trial):
                typing_cast(TypingAny, run).current_trial = trial.number
                db.commit()
                connection_manager.broadcast_sync(
                    TelemetryMessage(
                        type="status",
                        data={
                            "is_running": True,
                            "progress": typing_cast(int, getattr(run, "progress", 0)),
                            "run_id": run_id,
                            "current_trial": trial.number,
                            "total_trials": config.optuna_trials,
                            "metrics_history": getattr(run, "metrics_history", []),
                            "result": {
                                "best_r2": typing_cast(
                                    float, getattr(run, "best_r2", 0.0)
                                )
                            },
                        },
                    )
                )
                return super().__call__(trial)

        obj = TrackedObjective(
            config.model_choice,
            X_train,
            y_train,
            X_val,
            y_val,
            device,
            config.patience,
            params,
            on_checkpoint=on_checkpoint,
        )

        study = run_optimization(
            f"NFTool_{config.model_choice}", config.optuna_trials, None, obj
        )

        # Final Reporting
        db_log_and_broadcast(
            db, run_id, "Generating diagnostic reports...", connection_manager, "info"
        )
        analyze_optuna_study(study, run_dir, run_id)

        # Generate Regression Plots with Best Model
        best_model_path = os.path.join(run_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            from src.models.architectures import model_factory

            # Merge checkpoint config with original config to ensure required keys exist
            checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
            merged_config = {**config.model_dump(), **checkpoint_config}
            best_model = model_factory(
                config.model_choice,
                X_train.shape[1],
                merged_config,
                device,
            )
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
        db_log_and_broadcast(
            db, run_id, f"Optimization finished. Best R²: {typing_cast(float, getattr(run, 'best_r2', 0.0)):.4f}", connection_manager, "success"
        )

        # Cleanup and final broadcast. Compute final progress from run if available.
        if "device" in locals() and device.type == "cuda":
            torch.cuda.empty_cache()
        if "run" in locals() and run:
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
