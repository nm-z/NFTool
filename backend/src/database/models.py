from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

# Use the shared Base from src.database.database so create_all() creates these tables.
from src.database.database import Base  # type: ignore


class Run(Base):
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
