import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import WebSocket

from src.config import LOGS_DIR
from src.database.database import SessionLocal
from src.database.models import Run
from src.database.models import ModelCheckpoint
from src.schemas.websocket import (
    HardwareStats,
    LogMessage,
    MetricData,
    StatusData,
    TelemetryMessage,
)
from src.utils.hardware import HardwareMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "api.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("nftool")


def log_print(*args, **kwargs):
    msg = " ".join(map(str, args))
    logger.info(msg)


"""Connection manager for WebSocket clients.

This module manages connected WebSocket clients, broadcasts structured
telemetry messages, and polls the database for persisted metrics/logs to
forward to clients. The implementation keeps forwarding lightweight to
avoid blocking the main event loop.
"""


class ConnectionManager:
    """Manage WebSocket clients and forward telemetry.

    Responsibilities:
    - Maintain connected clients list and a lock protecting it.
    - Periodically poll hardware stats and forward them.
    - Poll the DB for new metrics/logs/status/checkpoints for the active run
      and forward any newly committed items to connected clients.
    """
    def __init__(self):
        # Don't inject bogus/placeholder stats. Only store obtained or actually measured stats.
        self.clients: List[WebSocket] = []
        self.client_lock = asyncio.Lock()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        # Initialize the monitor and probe hardware immediately. If probing fails,
        # abort startup so the application fails loudly rather than reporting fake stats.
        self.monitor = HardwareMonitor()
        try:
            gpu_stats = self.monitor.get_gpu_stats(0)
            system_stats = self.monitor.get_system_stats()
        except Exception as exc:
            logger.error("Hardware probe failed during startup; aborting.", exc_info=True)
            raise RuntimeError("Hardware probe failed during startup") from exc

        # Require actual measured values; do not fabricate defaults.
        self.hardware_stats: Dict[str, Any] = {**(gpu_stats or {}), **(system_stats or {})}
        self.active_run_id: Optional[str] = None
        self.current_gpu_id: int = 0
        # Track how many metrics/logs have been forwarded for a given run so
        # we only broadcast new items observed in the DB (useful when the
        # training worker runs in a separate process).
        self._last_metrics_index: Dict[str, int] = {}
        self._last_logs_index: Dict[str, int] = {}
        self._last_status: Dict[str, Dict[str, Any]] = {}
        self._last_checkpoint_id: Dict[str, int] = {}

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Register the asyncio event loop used for threadsafe broadcasts."""
        self.loop = loop

    async def hardware_monitor_task(self):
        """Background task: poll hardware and DB for updates and forward them.

        This task runs forever in the application's event loop and performs:
        - hardware stats polling at a configurable interval,
        - DB polling for the active run to forward new metrics/logs/status/checkpoints.
        """
        from src.services.queue_instance import job_queue  # local import to avoid cycle
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
            # Normalize legacy keys to the current schema so frontend receives consistent fields.
            normalized = {}
            # GPU / VRAM fields (prefer new names if present)
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

            # Merge normalized with originals (originals provide any other fields)
            merged = {**gpu_stats, **system_stats, **normalized}
            # Store a serializable snapshot for other callers
            hw = HardwareStats(**merged)
            self.hardware_stats = hw.model_dump()
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
                    db = SessionLocal()
                    run = db.query(Run).filter(Run.run_id == self.active_run_id).first()
                    if not run:
                        # Active run disappeared; clear tracking and continue.
                        self.active_run_id = None
                        await asyncio.sleep(interval)
                        continue

                    # Forward new metric points
                    metrics = list(getattr(run, "metrics_history", []) or [])
                    last_m = self._last_metrics_index.get(self.active_run_id, 0)
                    for m in metrics[last_m:]:
                            try:
                                m_tm = TelemetryMessage(type="metrics", data=MetricData(**m))
                                await self.broadcast(m_tm)
                            except Exception:
                                logger.exception("Failed to forward metric point")
                            # handled above
                    if self.active_run_id:
                        self._last_metrics_index[self.active_run_id] = len(metrics)

                    # Forward new log entries
                    logs = list(getattr(run, "logs", []) or [])
                    last_l = self._last_logs_index.get(self.active_run_id, 0)
                    for lg in logs[last_l:]:
                            try:
                                l_tm = TelemetryMessage(type="log", data=LogMessage(**lg))
                                await self.broadcast(l_tm)
                            except Exception:
                                logger.exception("Failed to forward log entry")
                    if self.active_run_id:
                        self._last_logs_index[self.active_run_id] = len(logs)

                    # Forward status updates (progress, current_trial, status, best_r2)
                    # Build status snapshot with defensively-extracted totals.
                    total_trials = int(getattr(run, "optuna_trials", 0) or 0)
                    cfg = getattr(run, "config", {}) or {}
                    cfg_trials = 0
                    try:
                        cfg_trials = int(cfg.get("optuna_trials") or cfg.get("trials") or 0)
                    except Exception:
                        cfg_trials = 0
                    if total_trials <= 0 and cfg_trials > 0:
                        total_trials = cfg_trials
                    current_trial = int(getattr(run, "current_trial", 0) or 0)
                    if total_trials > 0 and current_trial > total_trials:
                        current_trial = total_trials
                    status_snapshot = {
                        "is_running": getattr(run, "status", None) == "running",
                        "progress": int(getattr(run, "progress", 0) or 0),
                        "run_id": getattr(run, "run_id", None),
                        "current_trial": current_trial,
                        "total_trials": total_trials,
                        "result": {"best_r2": float(getattr(run, "best_r2", 0.0) or 0.0)},
                    }
                    last_status = self._last_status.get(self.active_run_id)
                    if last_status != status_snapshot:
                        s_tm = TelemetryMessage(type="status", data=status_snapshot)
                        await self.broadcast(s_tm)
                        self._last_status[self.active_run_id] = status_snapshot

                    # Also forward any newly created ModelCheckpoint rows as metric points
                    try:
                        ckpts = (
                            db.query(ModelCheckpoint)
                            .join(Run)
                            .filter(ModelCheckpoint.run_id == run.id)
                            .order_by(ModelCheckpoint.id)
                            .all()
                        )
                        last_ck = self._last_checkpoint_id.get(self.active_run_id, 0)
                        for ck in ckpts:
                            if int(getattr(ck, "id", 0)) <= last_ck:
                                continue
                            trial_num = None
                            try:
                                fname = os.path.basename(str(getattr(ck, "model_path", "") or ""))
                                if "trial_" in fname:
                                    part = fname.split("trial_")[-1]
                                    trial_num = int(part.split(".")[0])
                            except Exception:
                                trial_num = None
                            r2_val = getattr(ck, "r2_score", None)
                            try:
                                r2_float = float(r2_val) if r2_val is not None else None
                            except Exception:
                                r2_float = None
                            metric_point = {
                                "trial": int(trial_num or 0),
                                "loss": None,
                                "r2": r2_float,
                                "mae": None,
                                "val_loss": None,
                            }
                                try:
                                    ck_tm = TelemetryMessage(
                                        type="metrics", data=MetricData(**metric_point)
                                    )
                                    await self.broadcast(ck_tm)
                                except Exception:
                                    logger.exception("Failed to forward checkpoint-based metric")
                            last_ck = max(last_ck, int(getattr(ck, "id", last_ck)))
                        self._last_checkpoint_id[self.active_run_id] = last_ck
                    except Exception:
                        logger.exception("Failed to query/forward ModelCheckpoint metrics")
                finally:
                    if db is not None:
                        try:
                            db.close()
                        except Exception:
                            pass
            await asyncio.sleep(interval)

    async def connect(self, websocket: WebSocket, client_api_key: Optional[str] = None):
        # Accept without subprotocol when no API key is provided to avoid typing issues.
        if client_api_key:
            await websocket.accept(subprotocol=f"api-key-{client_api_key}")
        else:
            await websocket.accept()
        async with self.client_lock:
            self.clients.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self.client_lock:
            if websocket in self.clients:
                self.clients.remove(websocket)

    async def broadcast(self, message: TelemetryMessage):
        async with self.client_lock:
            if not self.clients:
                return
            disconnected = []
            msg_json = message.model_dump_json()
            for client in list(self.clients):
                try:
                    await client.send_text(msg_json)
                except Exception:
                    # Mark client as disconnected; cleanup after attempting all sends
                    logger.exception("Error sending websocket message to client; scheduling removal")
                    disconnected.append(client)
            for client in disconnected:
                if client in self.clients:
                    self.clients.remove(client)

    def broadcast_sync(self, message: TelemetryMessage):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self.loop)


manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket, api_key: str):
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

    # Use context manager for DB session; allow exceptions to propagate, but ensure disconnect in finally.
    db = SessionLocal()
    try:
        # For an empty initial UI, send basic init first.
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
        init_msg = TelemetryMessage(type="init", data=init_data).model_dump_json()
        await websocket.send_text(init_msg)

        # Immediately catch the client up with any persisted run state (if a run exists).
        try:
            latest_run = db.query(Run).order_by(Run.id.desc()).first()
            if latest_run:
                run_id = getattr(latest_run, "run_id", None)
                status_val = getattr(latest_run, "status", None)
                metrics = list(getattr(latest_run, "metrics_history", []) or [])
                logs = list(getattr(latest_run, "logs", []) or [])

                # If there's an active/running job, ensure the manager polls it for new updates.
                if status_val == "running":
                    manager.active_run_id = run_id

                # Send a status snapshot so UI shows run state immediately.
                try:
                    # Same robust total_trials extraction as above for initial catch-up.
                    total_trials = int(getattr(latest_run, "optuna_trials", 0) or 0)
                    cfg = getattr(latest_run, "config", {}) or {}
                    try:
                        cfg_trials = int(cfg.get("optuna_trials") or cfg.get("trials") or 0)
                    except Exception:
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
                        "result": {"best_r2": float(getattr(latest_run, "best_r2", 0.0) or 0.0)},
                    }
                    status_msg = TelemetryMessage(type="status", data=status_snapshot).model_dump_json()
                    await websocket.send_text(status_msg)
                except Exception:
                    logger.exception("Failed to send initial status snapshot to websocket client")

                # Send persisted metrics and logs (one message per item) so client can render
                # the full history immediately. Mark indexes so the manager doesn't forward
                # duplicates later.
                try:
                    for m in metrics:
                        try:
                            metric_msg = TelemetryMessage(type="metrics", data=MetricData(**m)).model_dump_json()
                            await websocket.send_text(metric_msg)
                        except Exception:
                            logger.exception("Failed to send persisted metric to client")
                    if run_id:
                        manager._last_metrics_index[run_id] = len(metrics)
                except Exception:
                    logger.exception("Failed to send persisted metrics to websocket client")

                try:
                    for lg in logs:
                        try:
                            log_msg = TelemetryMessage(type="log", data=LogMessage(**lg)).model_dump_json()
                            await websocket.send_text(log_msg)
                        except Exception:
                            logger.exception("Failed to send persisted log to client")
                    if run_id:
                        manager._last_logs_index[run_id] = len(logs)
                except Exception:
                    logger.exception("Failed to send persisted logs to websocket client")
        except Exception:
            logger.exception("Failed to catch up client with persisted run data")

        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    finally:
        db.close()
        await manager.disconnect(websocket)
