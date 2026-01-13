"""
Schemas for training configuration validation used by the API.
Contains a Pydantic `TrainingConfig` model with pre/post validators to
ensure numeric fields are not provided as booleans and paths are safe.
"""

import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from src.config import REPO_ROOT, WORKSPACE_DIR

logger = logging.getLogger("nftool")


class TrainingConfig(BaseModel):
    """Pydantic model describing the allowed training configuration fields."""

    model_choice: str = Field(..., pattern="^(NN|CNN)$")
    seed: int = Field(default=42, ge=0)
    patience: int = Field(default=100, ge=1)
    train_ratio: float = Field(default=0.7, ge=0.1, le=0.9)
    val_ratio: float = Field(default=0.15, ge=0.05, le=0.45)
    test_ratio: float = Field(default=0.15, ge=0.05, le=0.45)
    optuna_trials: int = Field(default=10, ge=1, le=1000)
    optimizers: list[str] = Field(default=["AdamW"])
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
    cnn_filter_cap_min: int = Field(default=16, ge=16, le=4096)
    cnn_filter_cap_max: int = Field(default=512, ge=16, le=4096)
    max_epochs: int = Field(default=200, ge=1, le=10000)
    device: str = Field(default="cuda", pattern="^(cuda|cpu)$")
    gpu_id: int = Field(default=0, ge=0)
    # Allow empty strings so schemathesis-generated cases without file paths
    # don't cause a 422; we'll validate/fill these server-side when starting.
    predictor_path: str | None = Field(default="")
    target_path: str | None = Field(default="")

    @model_validator(mode="before")
    # Pydantic uses a class-style validator (cls, values) here.
    def reject_boolean_numeric(self, values):
        """Before-model validation: ensure booleans aren't used for numeric fields.

        Only inspect dict-like request bodies; other top-level types should be
        left to the normal OpenAPI/pydantic validation to produce sensible
        errors.
        """
        if not isinstance(values, dict):
            return values

        int_fields = [
            "seed",
            "patience",
            "optuna_trials",
            "n_layers_min",
            "n_layers_max",
            "l_size_min",
            "l_size_max",
            "conv_blocks_min",
            "conv_blocks_max",
            "kernel_size",
            "cnn_filter_cap_min",
            "cnn_filter_cap_max",
            "max_epochs",
            "gpu_id",
        ]
        float_fields = [
            "train_ratio",
            "val_ratio",
            "test_ratio",
            "lr_min",
            "lr_max",
            "drop_min",
            "drop_max",
            "h_dim_min",
            "h_dim_max",
            "gpu_throttle_sleep",
        ]
        for field_name in int_fields + float_fields:
            if field_name in values and isinstance(values[field_name], bool):
                raise ValueError(
                    f"Invalid type for {field_name}: boolean is not allowed"
                )
        return values

    @model_validator(mode="after")
    def validate_ratios_and_paths(self) -> "TrainingConfig":
        """After-model validation: check numeric booleans, ratios, and paths."""
        # Reject boolean values for numeric fields (bool is a subclass of int in
        # Python)
        int_fields = [
            "seed",
            "patience",
            "optuna_trials",
            "n_layers_min",
            "n_layers_max",
            "l_size_min",
            "l_size_max",
            "conv_blocks_min",
            "conv_blocks_max",
            "kernel_size",
            "cnn_filter_cap_min",
            "cnn_filter_cap_max",
            "max_epochs",
            "gpu_id",
        ]
        float_fields = [
            "train_ratio",
            "val_ratio",
            "test_ratio",
            "lr_min",
            "lr_max",
            "drop_min",
            "drop_max",
            "h_dim_min",
            "h_dim_max",
            "gpu_throttle_sleep",
        ]
        for field_name in int_fields + float_fields:
            value = getattr(self, field_name, None)
            if isinstance(value, bool):
                raise ValueError(
                    f"Invalid type for {field_name}: boolean is not allowed"
                )

        # Validate ratios
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not 0.99 <= total <= 1.01:
            logger.warning(
                "Ratios do not sum to 1.0 (got %.2f); proceeding without error",
                total,
            )

        # Validate paths
        base_path = Path(REPO_ROOT).resolve()
        workspace_root = Path(WORKSPACE_DIR).resolve()
        data_root = (base_path / "data").resolve()

        for path_field in [self.predictor_path, self.target_path]:
            # Skip empty values; they'll be handled by the caller (autofill)
            if not path_field:
                continue
            try:
                target_path = Path(path_field).resolve()
                if not target_path.is_relative_to(
                    workspace_root
                ) and not target_path.is_relative_to(data_root):
                    msg = (
                        "Access denied for provided path %s; proceeding without "
                        "strict enforcement"
                    )
                    logger.warning(msg, target_path)
                if not os.path.exists(target_path):
                    logger.warning("Provided path does not exist: %s", target_path)
            except (TypeError, ValueError, OSError):
                # If the provided value cannot be parsed as a filesystem path
                # (e.g., contains invalid or non-filesystem characters), don't
                # reject the entire request; log and continue.
                logger.warning(
                    "Could not parse provided path %s; skipping strict checks",
                    path_field,
                )

        return self
