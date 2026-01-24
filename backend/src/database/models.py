"""SQLAlchemy ORM models for storing run metadata and model checkpoints."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import mapped_column, relationship
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
    id = mapped_column(Integer, primary_key=True, index=True)
    # Index on timestamp for efficient ordering by creation time
    timestamp = mapped_column(DateTime, default=datetime.now, nullable=False, index=True)
    # e.g. PASS_7721
    run_id = mapped_column(String, unique=True, index=True, nullable=False)
    model_choice = mapped_column(String, nullable=False)
    # Index on status for efficient filtering of running/completed runs
    status = mapped_column(String, nullable=False, index=True)  # running, completed, aborted, failed
    progress = mapped_column(Integer, default=0, nullable=False)
    current_trial = mapped_column(Integer, default=0, nullable=False)
    best_r2 = mapped_column(Float, default=-1.0e9, nullable=False)
    optuna_trials = mapped_column(Integer, nullable=False)
    config = mapped_column(JSON, default=dict, nullable=False)
    report_path = mapped_column(String, default="", nullable=False)
    metrics_history = mapped_column(JSON, default=list, nullable=False)
    logs = mapped_column(JSON, default=list, nullable=False)


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
    id = mapped_column(Integer, primary_key=True, index=True)
    # Index on run_id for efficient checkpoint queries per run
    run_id = mapped_column(Integer, ForeignKey("runs.id"), nullable=False, index=True)
    timestamp = mapped_column(DateTime, default=datetime.now, nullable=False)
    model_path = mapped_column(String, nullable=False)
    scaler_path = mapped_column(String, nullable=False)
    r2_score = mapped_column(Float, nullable=False)
    epoch = mapped_column(Integer, nullable=False)
    trial = mapped_column(Integer, nullable=False)
    val_loss = mapped_column(Float, nullable=False)
    mae = mapped_column(Float, nullable=False)
    params = mapped_column(JSON, default=dict, nullable=False)

    run = relationship("Run", back_populates="checkpoints")


Run.checkpoints = relationship("ModelCheckpoint", back_populates="run")


class EpochMetric(Base):
    """Per-epoch metric record for a run/trial."""

    __tablename__ = "epoch_metrics"
    id = mapped_column(Integer, primary_key=True, index=True)
    run_id = mapped_column(Integer, ForeignKey("runs.id"), nullable=False)
    timestamp = mapped_column(DateTime, default=datetime.now, nullable=False)
    trial = mapped_column(Integer, nullable=False)
    epoch = mapped_column(Integer, nullable=False)
    loss = mapped_column(Float, nullable=False)
    r2 = mapped_column(Float, nullable=False)
    mae = mapped_column(Float, nullable=False)
    val_loss = mapped_column(Float, nullable=False)


class ModelArtifact(Base):
    """Filesystem artifacts produced during a run."""

    __tablename__ = "model_artifacts"
    id = mapped_column(Integer, primary_key=True, index=True)
    run_id = mapped_column(Integer, ForeignKey("runs.id"), nullable=False)
    timestamp = mapped_column(DateTime, default=datetime.now, nullable=False)
    kind = mapped_column(String, nullable=False)
    path = mapped_column(String, nullable=False)


Run.epoch_metrics = relationship("EpochMetric", back_populates="run")
Run.artifacts = relationship("ModelArtifact", back_populates="run")
EpochMetric.run = relationship("Run", back_populates="epoch_metrics")
ModelArtifact.run = relationship("Run", back_populates="artifacts")
