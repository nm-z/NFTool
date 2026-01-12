from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from fastapi import HTTPException
from src.config import REPO_ROOT, WORKSPACE_DIR

def safe_path(relative_path: str):
    """Sanitize and validate path to prevent directory traversal"""
    try:
        base_path = Path(REPO_ROOT).resolve()
        target_path = (base_path / relative_path).resolve()
        
        workspace_root = Path(WORKSPACE_DIR).resolve()
        data_root = (base_path / "data").resolve()
        
        if target_path.is_relative_to(workspace_root) or target_path.is_relative_to(data_root):
            return str(target_path)
            
        raise HTTPException(status_code=403, detail="Access denied: Path outside allowed directories")
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid path structure")

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
    conv_blocks_min: int = Field(default=1, ge=1, le=10)
    conv_blocks_max: int = Field(default=5, ge=1, le=10)
    kernel_size: int = Field(default=3, ge=1, le=15)
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
        try:
            target = safe_path(v)
            if not os.path.exists(target):
                raise ValueError(f"File not found: {v}")
        except HTTPException as e:
            raise ValueError(f"Invalid path: {e.detail}")
        return v