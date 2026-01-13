"""SQLAlchemy ORM models for storing run metadata and model checkpoints."""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

# Use the shared Base from src.database.database so create_all() creates these tables.
from src.database.database import Base  # type: ignore


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
        best_r2: Best observed R^2 score (nullable).
        optuna_trials: Number of trials run.
        config: JSON configuration used for the run.
        report_path: Optional path to the run report.
        metrics_history: Optional JSON history of metrics.
        logs: JSON array of log entries.
    """

    __tablename__ = "runs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now)
    run_id = Column(String, unique=True, index=True)  # e.g. PASS_7721
    model_choice = Column(String)
    status = Column(String)  # running, completed, aborted, failed
    progress = Column(Integer, default=0)
    current_trial = Column(Integer, default=0)
    best_r2 = Column(Float, nullable=True)
    optuna_trials = Column(Integer)
    config = Column(JSON)
    report_path = Column(String, nullable=True)
    metrics_history = Column(JSON, nullable=True)
    logs = Column(JSON, default=[])


class ModelCheckpoint(Base):
    """A saved model checkpoint and associated artifacts.

    Attributes:
        id: Primary key.
        run_id: Foreign key to the owning `Run`.
        timestamp: When the checkpoint was created.
        model_path: Filesystem path to the saved model.
        scaler_path: Path to the saved input scaler.
        r2_score: R^2 score of this checkpoint.
        params: JSON of training/hyperparameters used for this checkpoint.
    """

    __tablename__ = "checkpoints"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"))
    timestamp = Column(DateTime, default=datetime.now)
    model_path = Column(String)
    scaler_path = Column(String)
    r2_score = Column(Float)
    params = Column(JSON)

    run = relationship("Run", back_populates="checkpoints")


Run.checkpoints = relationship("ModelCheckpoint", back_populates="run")
