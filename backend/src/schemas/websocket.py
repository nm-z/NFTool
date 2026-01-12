from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LogMessage(BaseModel):
    time: str = Field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    msg: str
    type: str = "default"
    epoch: int | None = None


class MetricData(BaseModel):
    trial: int
    loss: float | None = None
    r2: float | None = None
    mae: float | None = None
    val_loss: float | None = None


class HardwareStats(BaseModel):
    vram_total_gb: float | None = None
    vram_used_gb: float | None = None
    vram_percent: int | None = None
    gpu_use_percent: int | None = None
    gpu_temp_c: int | None = None
    cpu_percent: float | None = None
    ram_used_gb: float | None = None
    ram_total_gb: float | None = None
    ram_percent: float | None = None


class ResultData(BaseModel):
    best_r2: float | None = None


class StatusData(BaseModel):
    is_running: bool
    is_aborting: bool = False
    progress: int
    run_id: str | None = None
    current_trial: int
    total_trials: int
    result: ResultData | None = None
    metrics_history: list[MetricData] = []
    logs: list[LogMessage] = []
    hardware_stats: HardwareStats | None = None
    queue_size: int
    active_job_id: str | None = None


class TelemetryMessage(BaseModel):
    type: str = Field(..., pattern="^(init|status|log|metrics|hardware|error)$")
    data: Any  # This will be validated dynamically based on `type`
