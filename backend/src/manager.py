"""Connection manager for WebSocket clients.

This module manages connected WebSocket clients, broadcasts structured
telemetry messages, and polls the database for persisted metrics/logs to
forward to clients. The implementation keeps forwarding lightweight to
avoid blocking the main event loop.
"""

import asyncio
import json
import logging
import os
import subprocess
from importlib import import_module
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy.exc import SQLAlchemyError
from src.database.database import SESSION_LOCAL
from src.database.models import ModelCheckpoint, Run
from src.schemas.websocket import (
    HardwareStats,
    LogMessage,
    MetricData,
    StatusData,
    TelemetryMessage,
)
from src.utils.hardware import HardwareMonitor

logger = logging.getLogger("nftool.manager")


class ConnectionManager:
    """Manage WebSocket clients and forward telemetry.

    Responsibilities:
    - Maintain connected clients list and a lock protecting it.
    - Periodically poll hardware stats and forward them.
    - Poll the DB for new metrics/logs/status/checkpoints for the active run
      and forward any newly committed items to connected clients.
    """

    def __init__(self):
        # Don't inject bogus/placeholder stats.
        # Only store obtained or actually measured stats.
        self.clients: list[WebSocket] = []
        self.client_lock = asyncio.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None
        # Consolidate tracking maps to reduce instance attribute count.
        self._tracking: dict[str, Any] = {
            "metrics_index": {},
            "logs_index": {},
            "status": {},
            "checkpoint_id": {},
        }

        # Initialize the monitor and probe hardware immediately.
        # If probing fails, abort startup so the application fails loudly rather
        # than reporting fake stats.
        self.monitor = HardwareMonitor()
        try:
            gpu_stats = self.monitor.get_gpu_stats(0)
            system_stats = self.monitor.get_system_stats()
        except (
            RuntimeError,
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            logger.error(
                "Hardware probe failed during startup; aborting.", exc_info=True
            )
            raise RuntimeError("Hardware probe failed during startup") from exc

        # Require actual measured values; do not fabricate defaults.
        self._tracking["hardware_stats"] = {
            **(gpu_stats or {}),
            **(system_stats or {}),
        }
        self.active_run_id: str | None = None
        # Track how many metrics/logs have been forwarded for a given run so
        # we only broadcast new items observed in the DB (useful when the
        # training worker runs in a separate process).

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Register the asyncio event loop used for threadsafe broadcasts."""
        self.loop = loop

    @property
    def current_gpu_id(self) -> int:
        """Access the configured GPU id for monitoring (stored in tracking)."""
        return int(self._tracking.get("current_gpu_id", 0) or 0)

    @current_gpu_id.setter
    def current_gpu_id(self, value: int) -> None:
        """Set the configured GPU id for monitoring."""
        try:
            self._tracking["current_gpu_id"] = int(value or 0)
        except (TypeError, ValueError):
            self._tracking["current_gpu_id"] = 0

    def _normalize_stats(
        self,
        gpu_stats: dict[str, Any],
        system_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize GPU and system stat keys to the canonical schema.

        Returns a merged dict suitable for constructing `HardwareStats`.
        """
        normalized: dict[str, Any] = {}

        # VRAM / memory
        if "vram_total_gb" in gpu_stats:
            normalized["vram_total_gb"] = gpu_stats.get("vram_total_gb")
        elif "gpu_mem_total" in gpu_stats:
            normalized["vram_total_gb"] = gpu_stats.get("gpu_mem_total")

        if "vram_used_gb" in gpu_stats:
            normalized["vram_used_gb"] = gpu_stats.get("vram_used_gb")
        elif "gpu_mem_used" in gpu_stats:
            normalized["vram_used_gb"] = gpu_stats.get("gpu_mem_used")

        if "vram_percent" in gpu_stats:
            normalized["vram_percent"] = gpu_stats.get("vram_percent")
        elif "gpu_mem_percent" in gpu_stats:
            normalized["vram_percent"] = gpu_stats.get("gpu_mem_percent")

        # GPU utilization / temp
        if "gpu_use_percent" in gpu_stats:
            normalized["gpu_use_percent"] = gpu_stats.get("gpu_use_percent")
        elif "gpu_util" in gpu_stats:
            normalized["gpu_use_percent"] = gpu_stats.get("gpu_util")

        if "gpu_temp_c" in gpu_stats:
            normalized["gpu_temp_c"] = gpu_stats.get("gpu_temp_c")
        elif "gpu_temp" in gpu_stats:
            normalized["gpu_temp_c"] = gpu_stats.get("gpu_temp")

        # System stats
        if "cpu_percent" in system_stats:
            normalized["cpu_percent"] = system_stats.get("cpu_percent")
        elif "cpu_util" in system_stats:
            normalized["cpu_percent"] = system_stats.get("cpu_util")

        if "ram_total_gb" in system_stats:
            normalized["ram_total_gb"] = system_stats.get("ram_total_gb")
        elif "ram_total" in system_stats:
            normalized["ram_total_gb"] = system_stats.get("ram_total")

        if "ram_used_gb" in system_stats:
            normalized["ram_used_gb"] = system_stats.get("ram_used_gb")
        elif "ram_used" in system_stats:
            normalized["ram_used_gb"] = system_stats.get("ram_used")

        if "ram_percent" in system_stats:
            normalized["ram_percent"] = system_stats.get("ram_percent")

        # Merge normalized keys with originals (originals supply any other fields)
        merged = {**gpu_stats, **system_stats, **normalized}
        return merged

    def _build_status_snapshot(self, run: Run) -> dict[str, Any]:
        """Construct a small status snapshot dict for a Run ORM instance."""
        total_trials = int(getattr(run, "optuna_trials", 0) or 0)
        cfg = getattr(run, "config", {}) or {}
        try:
            cfg_trials = int(cfg.get("optuna_trials") or cfg.get("trials") or 0)
        except (TypeError, ValueError):
            cfg_trials = 0
        if total_trials <= 0 and cfg_trials > 0:
            total_trials = cfg_trials
        current_trial = int(getattr(run, "current_trial", 0) or 0)
        if total_trials > 0 and current_trial > total_trials:
            current_trial = total_trials
        return {
            "is_running": getattr(run, "status", None) == "running",
            "progress": int(getattr(run, "progress", 0) or 0),
            "run_id": getattr(run, "run_id", None),
            "current_trial": current_trial,
            "total_trials": total_trials,
            "result": {"best_r2": float(getattr(run, "best_r2", 0.0) or 0.0)},
        }

    async def _forward_checkpoints(self, db: Any, run: Run) -> None:
        """Query ModelCheckpoint rows for a run and broadcast them as metric points.
        
        Optimized to only fetch checkpoints newer than the last forwarded one.
        """
        last_ck = self._tracking["checkpoint_id"].get(self.active_run_id, 0)
        try:
            # Only query checkpoints we haven't forwarded yet
            ckpts = (
                db.query(ModelCheckpoint)
                .filter(
                    ModelCheckpoint.run_id == run.id,
                    ModelCheckpoint.id > last_ck
                )
                .order_by(ModelCheckpoint.id)
                .all()
            )
        except SQLAlchemyError:
            logger.exception("Failed to query ModelCheckpoint rows")
            return

        for ck in ckpts:
            fname = os.path.basename(str(getattr(ck, "model_path", "") or ""))
            trial_num = int(getattr(ck, "trial", 0) or 0)
            epoch_num = int(getattr(ck, "epoch", 0) or 0)
            r2_val = getattr(ck, "r2_score", None)
            val_loss = float(getattr(ck, "val_loss", 0.0) or 0.0)
            mae = float(getattr(ck, "mae", 0.0) or 0.0)
            try:
                if trial_num == 0 and "trial_" in fname:
                    part = fname.split("trial_")[-1]
                    trial_num = int(part.split(".")[0])
                r2_float = float(r2_val) if r2_val is not None else None
            except (ValueError, IndexError, TypeError):
                r2_float = None
            ck_tm = TelemetryMessage(
                type="metrics",
                data=MetricData(
                    trial=trial_num,
                    epoch=epoch_num,
                    loss=val_loss,
                    r2=r2_float,
                    mae=mae,
                    val_loss=val_loss,
                ),
            )
            await self.broadcast(ck_tm)
            last_ck = max(last_ck, int(getattr(ck, "id", last_ck)))
        self._tracking["checkpoint_id"][self.active_run_id] = last_ck

    async def _broadcast_metrics(self, metrics: list[dict[str, Any]]) -> None:
        """Broadcast a list of persisted metric dicts, skipping already-sent items."""
        if not metrics:
            return
        last_m = self._tracking["metrics_index"].get(self.active_run_id, 0)
        for m in metrics[last_m:]:
            try:
                m_tm = TelemetryMessage(type="metrics", data=MetricData(**m))
                await self.broadcast(m_tm)
            except (
                ValueError,
                TypeError,
                WebSocketDisconnect,
                OSError,
                RuntimeError,
            ) as exc:
                logger.exception("Failed to forward metric point: %s", exc)
        if self.active_run_id:
            self._tracking["metrics_index"][self.active_run_id] = len(metrics)

    async def _broadcast_logs(self, logs: list[dict[str, Any]]) -> None:
        """Broadcast a list of persisted log dicts, skipping already-sent items."""
        if not logs:
            return
        last_l = self._tracking["logs_index"].get(self.active_run_id, 0)
        for lg in logs[last_l:]:
            try:
                l_tm = TelemetryMessage(type="log", data=LogMessage(**lg))
                await self.broadcast(l_tm)
            except (
                ValueError,
                TypeError,
                WebSocketDisconnect,
                OSError,
                RuntimeError,
            ) as exc:
                logger.exception("Failed to forward log entry: %s", exc)
        if self.active_run_id:
            self._tracking["logs_index"][self.active_run_id] = len(logs)

    async def hardware_monitor_task(self):
        """Background task: poll hardware and DB for updates and forward them.

        This task runs forever in the application's event loop and performs:
        - hardware stats polling at a configurable interval,
        - DB polling for the active run to forward new metrics/logs/status/checkpoints.
        """
        # Resolve job_queue at runtime to avoid a circular import.
        job_queue = import_module("src.services.queue_instance").job_queue

        while True:
            if not self.clients:
                # No clients connected — sleep longer to reduce DB/hardware pressure.
                await asyncio.sleep(5)
                continue

            is_running = job_queue.active_job is not None
            interval = 2 if is_running else 10
            gpu_id = self.current_gpu_id
            gpu_stats = self.monitor.get_gpu_stats(gpu_id)
            system_stats = self.monitor.get_system_stats()
            # Normalize keys and build a single merged dict
            merged = self._normalize_stats(gpu_stats, system_stats)
            hw = HardwareStats(**merged)
            self._tracking["hardware_stats"] = hw.model_dump()
            # Broadcast hardware snapshot
            hw_tm = TelemetryMessage(type="hardware", data=hw)
            await self.broadcast(hw_tm)
            # If an active run exists, poll the DB for new metrics/logs and
            # forward any newly committed points so connected clients see
            # live updates even when the training worker runs in a separate
            # process.
            if self.active_run_id:
                db = None
                try:
                    with SESSION_LOCAL() as db:
                        run = (
                            db.query(Run)
                            .filter(Run.run_id == self.active_run_id)
                            .first()
                        )
                    if not run:
                        # Active run disappeared; clear tracking and continue.
                        self.active_run_id = None
                        await asyncio.sleep(interval)
                        continue

                    # Forward new metric points
                    metrics = list(getattr(run, "metrics_history", []) or [])
                    await self._broadcast_metrics(metrics)

                    # Forward new log entries
                    logs = list(getattr(run, "logs", []) or [])
                    await self._broadcast_logs(logs)

                    # Forward status updates (progress, current_trial, status, best_r2)
                    # Build status snapshot with defensively-extracted totals.
                    total_trials = int(getattr(run, "optuna_trials", 0) or 0)
                    cfg = getattr(run, "config", {}) or {}
                    cfg_trials = 0
                    try:
                        cfg_trials = int(
                            cfg.get("optuna_trials") or cfg.get("trials") or 0
                        )
                    except (TypeError, ValueError):
                        cfg_trials = 0
                    if total_trials <= 0 and cfg_trials > 0:
                        total_trials = cfg_trials
                    current_trial = int(getattr(run, "current_trial", 0) or 0)
                    if total_trials > 0 and current_trial > total_trials:
                        current_trial = total_trials
                    # Include queue snapshot so the UI process monitor has a
                    # consistent payload shape across status broadcasts.
                    q_status = (
                        job_queue.get_status()
                        if "job_queue" in locals()
                        else {"queue_size": 0, "active_job_id": None}
                    )
                    status_snapshot = {
                        "is_running": getattr(run, "status", None) == "running",
                        "progress": int(getattr(run, "progress", 0) or 0),
                        "run_id": getattr(run, "run_id", None),
                        "current_trial": current_trial,
                        "total_trials": total_trials,
                        "result": {
                            "best_r2": float(getattr(run, "best_r2", 0.0) or 0.0)
                        },
                        "queue_size": q_status.get("queue_size", 0),
                        "active_job_id": q_status.get("active_job_id"),
                    }
                    last_status = self._tracking["status"].get(self.active_run_id)
                    if last_status != status_snapshot:
                        s_tm = TelemetryMessage(type="status", data=status_snapshot)
                        await self.broadcast(s_tm)
                        self._tracking["status"][self.active_run_id] = status_snapshot

                    # Also forward any newly created ModelCheckpoint rows as
                    # metric points. Only query checkpoints newer than last forwarded.
                    try:
                        last_ck = self._tracking["checkpoint_id"].get(
                            self.active_run_id, 0
                        )
                        ckpts = (
                            db.query(ModelCheckpoint)
                            .filter(
                                ModelCheckpoint.run_id == run.id,
                                ModelCheckpoint.id > last_ck
                            )
                            .order_by(ModelCheckpoint.id)
                            .all()
                        )
                        for ck in ckpts:
                            fname = os.path.basename(
                                str(getattr(ck, "model_path", "")) or ""
                            )
                            trial_num = int(getattr(ck, "trial", 0) or 0)
                            epoch_num = int(getattr(ck, "epoch", 0) or 0)
                            r2_val = getattr(ck, "r2_score", None)
                            val_loss = float(getattr(ck, "val_loss", 0.0) or 0.0)
                            mae = float(getattr(ck, "mae", 0.0) or 0.0)
                            try:
                                if trial_num == 0 and "trial_" in fname:
                                    part = fname.split("trial_")[-1]
                                    trial_num = int(part.split(".")[0])
                                r2_float = float(r2_val) if r2_val is not None else None
                            except (ValueError, IndexError, TypeError):
                                r2_float = None
                            ck_tm = TelemetryMessage(
                                type="metrics",
                                data=MetricData(
                                    trial=trial_num,
                                    epoch=epoch_num,
                                    loss=val_loss,
                                    r2=r2_float,
                                    mae=mae,
                                    val_loss=val_loss,
                                ),
                            )
                            await self.broadcast(ck_tm)
                            last_ck = max(last_ck, int(getattr(ck, "id", last_ck)))
                        self._tracking["checkpoint_id"][self.active_run_id] = last_ck
                    except SQLAlchemyError:
                        logger.exception(
                            "Failed to query/forward ModelCheckpoint metrics"
                        )
                finally:
                    if db is not None:
                        db.close()
            await asyncio.sleep(interval)

    async def connect(self, websocket: WebSocket, client_api_key: str | None = None):
        """Accept and register an incoming websocket connection."""
        # Accept without subprotocol when no API key is provided to avoid typing issues.
        if client_api_key:
            await websocket.accept(subprotocol=f"api-key-{client_api_key}")
        else:
            await websocket.accept()
        async with self.client_lock:
            self.clients.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        """Remove a websocket from the active clients list."""
        async with self.client_lock:
            if websocket in self.clients:
                self.clients.remove(websocket)

    async def broadcast(self, message: TelemetryMessage):
        """Broadcast a TelemetryMessage to all connected clients (async)."""
        async with self.client_lock:
            if not self.clients:
                return
            disconnected: list[WebSocket] = []
            msg_json = message.model_dump_json()
            for client in list(self.clients):
                try:
                    await client.send_text(msg_json)
                except (WebSocketDisconnect, OSError, RuntimeError) as exc:
                    # Mark client as disconnected; cleanup after attempting all sends
                    disconnected.append(client)
                    # Handle benign WebSocket disconnects quietly
                    if isinstance(exc, WebSocketDisconnect):
                        logger.info(
                            "Client disconnected during broadcast (code: %s)",
                            exc.code,
                        )
                    else:
                        logger.exception("WS send err; sched disconnect")
            for client in disconnected:
                if client in self.clients:
                    self.clients.remove(client)

    def broadcast_sync(self, message: TelemetryMessage):
        """Synchronous wrapper to broadcast from other threads."""
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)

    def set_metrics_index(self, run_id: str, index: int) -> None:
        """Set the persisted metrics index for a run (thread-safe-ish helper)."""
        if not run_id:
            return
        self._tracking.setdefault("metrics_index", {})[run_id] = int(index or 0)

    def set_logs_index(self, run_id: str, index: int) -> None:
        """Set the persisted logs index for a run (thread-safe-ish helper)."""
        if not run_id:
            return
        self._tracking.setdefault("logs_index", {})[run_id] = int(index or 0)


manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, api_key: str | None):
    """WebSocket entrypoint for UI clients.

    Accepts incoming websocket connections. When api_key is None (Tauri mode),
    skips authentication. Otherwise validates API key via subprotocol.
    Primes the client with persisted run state (metrics/logs/status),
    and then proxies incoming messages (simple ping/pong).
    """
    client_api_key = None

    # Only validate API key if one is configured (Docker/dev mode)
    if api_key is not None:
        protocols = websocket.headers.get("Sec-WebSocket-Protocol", "").split(",")
        client_api_key = next(
            (
                p.strip().replace("api-key-", "")
                for p in protocols
                if p.strip().startswith("api-key-")
            ),
            None,
        )
        if client_api_key != api_key:
            await websocket.accept()
            await websocket.send_text(json.dumps({"type": "error", "data": "Unauthorized"}))
            await websocket.close(code=4003)
            return

    await manager.connect(websocket, client_api_key)
    manager.set_loop(asyncio.get_event_loop())

    # Use context manager for DB session.
    # Allow exceptions to propagate but ensure disconnect in finally.
    db = SESSION_LOCAL()
    try:
        # For an empty initial UI, send basic init first.
        # (This primes the client with a minimal status payload.)
        init_data = StatusData(
            is_running=False,
            progress=0,
            current_trial=0,
            total_trials=0,
            logs=[],
            metrics_history=[],
            result=None,
            hardware_stats=None,
            queue_size=0,
            active_job_id=None,
        )
        init_msg = TelemetryMessage(
            type="init",
            data=init_data,
        ).model_dump_json()
        await websocket.send_text(init_msg)

        # Immediately catch the client up with any persisted run state.
        # (If a run exists, send its latest persisted state.)
        try:
            latest_run = db.query(Run).order_by(Run.id.desc()).first()
            if latest_run:
                run_id = getattr(latest_run, "run_id", None)
                status_val = getattr(latest_run, "status", None)
                metrics = list(getattr(latest_run, "metrics_history", []) or [])
                logs = list(getattr(latest_run, "logs", []) or [])
                should_sync_history = status_val in ("running", "queued")

                # If there's an active/running job, ensure the manager polls it
                # for new updates.
                if status_val == "running":
                    manager.active_run_id = run_id

                # Send a status snapshot so UI shows run state immediately.
                try:
                    # Same robust total_trials extraction as above for initial catch-up.
                    total_trials = int(getattr(latest_run, "optuna_trials", 0) or 0)
                    cfg = getattr(latest_run, "config", {}) or {}
                    try:
                        cfg_trials = int(
                            cfg.get("optuna_trials") or cfg.get("trials") or 0
                        )
                    except (TypeError, ValueError):
                        cfg_trials = 0
                    if total_trials <= 0 and cfg_trials > 0:
                        total_trials = cfg_trials
                    current_trial = int(getattr(latest_run, "current_trial", 0) or 0)
                    if total_trials > 0 and current_trial > total_trials:
                        current_trial = total_trials
                    status_snapshot = {
                        "is_running": status_val == "running",
                        "progress": int(getattr(latest_run, "progress", 0) or 0),
                        "run_id": run_id,
                        "current_trial": current_trial,
                        "total_trials": total_trials,
                        "result": {
                            "best_r2": float(getattr(latest_run, "best_r2", 0.0) or 0.0)
                        },
                    }
                    status_msg = TelemetryMessage(
                        type="status",
                        data=status_snapshot,
                    ).model_dump_json()
                    await websocket.send_text(status_msg)
                except (WebSocketDisconnect, OSError, RuntimeError):
                    logger.exception("Failed to send initial status snapshot")

                if should_sync_history:
                    # Send persisted metrics and logs one message at a time so the
                    # client can render the full history immediately.
                    # Mark indexes so the manager doesn't forward duplicates later.
                    try:
                        for m in metrics:
                            try:
                                metric_msg = TelemetryMessage(
                                    type="metrics", data=MetricData(**m)
                                ).model_dump_json()
                                await websocket.send_text(metric_msg)
                            except (
                                ValueError,
                                TypeError,
                                WebSocketDisconnect,
                                OSError,
                                RuntimeError,
                            ) as exc:
                                logger.exception(
                                    "Failed to send persisted metric to client: %s",
                                    exc,
                                )
                        if run_id:
                            manager.set_metrics_index(run_id, len(metrics))
                    except (WebSocketDisconnect, OSError, RuntimeError):
                        logger.exception(
                            "Failed to send persisted metrics to websocket client"
                        )

                    try:
                        for lg in logs:
                            try:
                                log_msg = TelemetryMessage(
                                    type="log", data=LogMessage(**lg)
                                ).model_dump_json()
                                await websocket.send_text(log_msg)
                            except (
                                ValueError,
                                TypeError,
                                WebSocketDisconnect,
                                OSError,
                                RuntimeError,
                            ) as exc:
                                logger.exception(
                                    "Failed to send persisted log to client: %s",
                                    exc,
                                )
                        if run_id:
                            manager.set_logs_index(run_id, len(logs))
                    except (WebSocketDisconnect, OSError, RuntimeError) as exc:
                        logger.exception(
                            "Failed to send persisted logs to websocket client: %s",
                            exc,
                        )
        except (SQLAlchemyError, WebSocketDisconnect, OSError, RuntimeError) as exc:
            if isinstance(exc, WebSocketDisconnect):
                logger.info(
                    "WebSocket client disconnected during catch-up (code: %s)",
                    exc.code,
                )
            else:
                logger.exception(
                    "Failed to catch up client with persisted run data: %s", exc
                )

        try:
            while True:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
        except WebSocketDisconnect as exc:
            logger.info(
                "WebSocket client disconnected (code: %s)", exc.code
            )
    finally:
        db.close()
        await manager.disconnect(websocket)
