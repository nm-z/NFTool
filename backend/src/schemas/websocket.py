from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class LogMessage(BaseModel):
    time: str = Field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))
    msg: str
    type: str = "default"
    epoch: Optional[int] = None


class MetricData(BaseModel):
    trial: int
    loss: Optional[float] = None
    r2: Optional[float] = None
    mae: Optional[float] = None
    val_loss: Optional[float] = None


class HardwareStats(BaseModel):
    vram_total_gb: Optional[float] = None
    vram_used_gb: Optional[float] = None
    vram_percent: Optional[int] = None
    gpu_use_percent: Optional[int] = None
    gpu_temp_c: Optional[int] = None
    cpu_percent: Optional[float] = None
    ram_used_gb: Optional[float] = None
    ram_total_gb: Optional[float] = None
    ram_percent: Optional[float] = None


class ResultData(BaseModel):
    best_r2: Optional[float] = None


class StatusData(BaseModel):
    is_running: bool
    is_aborting: bool = False
    progress: int
    run_id: Optional[str] = None
    current_trial: int
    total_trials: int
    result: Optional[ResultData] = None
    metrics_history: List[MetricData] = []
    logs: List[LogMessage] = []
    hardware_stats: Optional[HardwareStats] = None
    queue_size: int
    active_job_id: Optional[str] = None


class TelemetryMessage(BaseModel):
    type: str = Field(..., pattern="^(init|status|log|metrics|hardware|error)$")
    data: Any  # This will be validated dynamically based on `type`
