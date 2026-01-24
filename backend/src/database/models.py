"""SQLAlchemy ORM models for storing run metadata and model checkpoints."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from src.database.database import Base


class Run(Base):
    """Persistent metadata for a training run.

    Attributes:
        id: Primary key.
        timestamp: Creation time of the run record.
        run_id: External run identifier (e.g. PASS_7721).
        model_choice: Name of the chosen model architecture.
        status: Current run status (running, completed, aborted, failed).
        progress: Integer percent progress.
        current_trial: Current Optuna trial number.
        best_r2: Best observed R^2 score.
        optuna_trials: Number of trials run.
        config: JSON configuration used for the run.
        report_path: Path to the run report directory.
        metrics_history: JSON history of metrics.
        logs: JSON array of log entries.
    """

    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # Index on timestamp for efficient ordering by creation time
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False, index=True)
    # e.g. PASS_7721
    run_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    model_choice: Mapped[str] = mapped_column(String, nullable=False)
    # Index on status for efficient filtering of running/completed runs
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)  # running, completed, aborted, failed
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    current_trial: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    best_r2: Mapped[float] = mapped_column(Float, default=-1.0e9, nullable=False)
    optuna_trials: Mapped[int] = mapped_column(Integer, nullable=False)
    config: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    report_path: Mapped[str] = mapped_column(String, default="", nullable=False)
    metrics_history: Mapped[list] = mapped_column(JSON, default=list, nullable=False)
    logs: Mapped[list] = mapped_column(JSON, default=list, nullable=False)

    checkpoints: Mapped[List["ModelCheckpoint"]] = relationship("ModelCheckpoint", back_populates="run")
    log_entries: Mapped[List["LogEntry"]] = relationship("LogEntry", back_populates="run")
    epoch_metrics: Mapped[List["EpochMetric"]] = relationship("EpochMetric", back_populates="run")
    artifacts: Mapped[List["ModelArtifact"]] = relationship("ModelArtifact", back_populates="run")


class LogEntry(Base):
    """Structured log entry persisted for a run."""

    __tablename__ = "log_entries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("runs.id"), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    time: Mapped[str] = mapped_column(String, default="", nullable=False)
    msg: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, default="default", nullable=False)
    epoch: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    run: Mapped["Run"] = relationship("Run", back_populates="log_entries")


class ModelCheckpoint(Base):
    """A saved model checkpoint and associated artifacts.

    Attributes:
        id: Primary key.
        run_id: Foreign key to the owning `Run`.
        timestamp: When the checkpoint was created.
        model_path: Filesystem path to the saved model.
        scaler_path: Path to the saved input scaler.
        r2_score: R^2 score of this checkpoint.
        epoch: Epoch index when the checkpoint was saved.
        trial: Trial index when the checkpoint was saved.
        val_loss: Validation loss for the checkpoint epoch.
        mae: Mean absolute error for the checkpoint epoch.
        params: JSON of training/hyperparameters used for this checkpoint.
    """

    __tablename__ = "checkpoints"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # Index on run_id for efficient checkpoint queries per run
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("runs.id"), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    model_path: Mapped[str] = mapped_column(String, nullable=False)
    scaler_path: Mapped[str] = mapped_column(String, nullable=False)
    r2_score: Mapped[float] = mapped_column(Float, nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    trial: Mapped[int] = mapped_column(Integer, nullable=False)
    val_loss: Mapped[float] = mapped_column(Float, nullable=False)
    mae: Mapped[float] = mapped_column(Float, nullable=False)
    params: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    run: Mapped["Run"] = relationship("Run", back_populates="checkpoints")


class EpochMetric(Base):
    """Per-epoch metric record for a run/trial."""

    __tablename__ = "epoch_metrics"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("runs.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    trial: Mapped[int] = mapped_column(Integer, nullable=False)
    epoch: Mapped[int] = mapped_column(Integer, nullable=False)
    loss: Mapped[float] = mapped_column(Float, nullable=False)
    r2: Mapped[float] = mapped_column(Float, nullable=False)
    mae: Mapped[float] = mapped_column(Float, nullable=False)
    val_loss: Mapped[float] = mapped_column(Float, nullable=False)

    run: Mapped["Run"] = relationship("Run", back_populates="epoch_metrics")


class ModelArtifact(Base):
    """Filesystem artifacts produced during a run."""

    __tablename__ = "model_artifacts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("runs.id"), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, nullable=False)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)

    run: Mapped["Run"] = relationship("Run", back_populates="artifacts")