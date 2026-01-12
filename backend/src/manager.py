import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from fastapi import WebSocket, WebSocketDisconnect

from src.config import LOGS_DIR
from src.database.database import SessionLocal
from src.database.models import Run
from src.schemas.websocket import (
    HardwareStats,
    LogMessage,
    MetricData,
    ResultData,
    StatusData,
    TelemetryMessage,
)
from src.services.queue_instance import job_queue
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


class ConnectionManager:
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

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    async def hardware_monitor_task(self):
        from src.services.queue_instance import job_queue
        while True:
            if not self.clients:
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
            # Broadcast a structured HardwareStats object inside a TelemetryMessage
            await self.broadcast(TelemetryMessage(type="hardware", data=hw))
            # If an active run exists, poll the DB for new metrics/logs and
            # forward any newly committed points so connected clients see
            # live updates even when the training worker runs in a separate
            # process.
            if self.active_run_id:
                try:
                    db = SessionLocal()
                    run = db.query(Run).filter(Run.run_id == self.active_run_id).first()
                    if run:
                        # Forward new metric points
                        metrics = list(getattr(run, "metrics_history", []) or [])
                        last_m = self._last_metrics_index.get(self.active_run_id, 0)
                        for m in metrics[last_m:]:
                            # ensure dict shape matches MetricData
                            try:
                                await self.broadcast(
                                    TelemetryMessage(type="metrics", data=MetricData(**m))
                                )
                            except Exception:
                                # skip malformed points
                                logger.exception("Failed to forward metric point")
                        self._last_metrics_index[self.active_run_id] = len(metrics)

                        # Forward new log entries
                        logs = list(getattr(run, "logs", []) or [])
                        last_l = self._last_logs_index.get(self.active_run_id, 0)
                        for lg in logs[last_l:]:
                            try:
                                await self.broadcast(
                                    TelemetryMessage(type="log", data=LogMessage(**lg))
                                )
                            except Exception:
                                logger.exception("Failed to forward log entry")
                        self._last_logs_index[self.active_run_id] = len(logs)
                finally:
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
            for client in self.clients:
                await client.send_text(msg_json)
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
        # For an empty initial UI, do not send past run logs, metrics or hardware stats.
        # The frontend will render an empty slate until a training run starts.
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
        await websocket.send_text(
            TelemetryMessage(type="init", data=init_data).model_dump_json()
        )

        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    finally:
        db.close()
        await manager.disconnect(websocket)
