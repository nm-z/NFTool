"""Utilities to persist and broadcast log/telemetry messages to WebSocket and DB."""

import logging
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy.orm import Session
from src.database.models import LogEntry, Run
from src.schemas.websocket import LogMessage, TelemetryMessage


class ConnectionBroadcaster(Protocol):
    """Protocol describing a connection manager capable of broadcasting messages."""

    def broadcast_sync(self, message: TelemetryMessage) -> None:
        """Synchronously broadcast a TelemetryMessage to all connected clients."""
        raise NotImplementedError

    async def broadcast(self, message: TelemetryMessage) -> None:
        """Asynchronously broadcast a TelemetryMessage to all connected clients."""
        raise NotImplementedError


logger = logging.getLogger("nftool")


def db_log_and_broadcast(
    db: Session,
    run_id: str,
    msg: str,
    connection_manager: ConnectionBroadcaster,
    level: str = "default",
    epoch: int | None = None,
) -> None:
    """Persist a log message to the DB and broadcast it over WebSocket."""
    timestamp = datetime.now(tz=UTC).strftime("%H:%M:%S")
    log_entry = LogMessage(msg=msg, type=level, epoch=epoch)
    run = db.query(Run).filter(Run.run_id == run_id).first()
    if run:
        db.add(
            LogEntry(
                run_id=run.id,
                timestamp=datetime.now(),
                time=log_entry.time,
                msg=msg,
                type=level,
                epoch=epoch,
            )
        )
        db.commit()
    # Broadcast a structured Pydantic model
    connection_manager.broadcast_sync(TelemetryMessage(type="log", data=log_entry))
    logger.info("[%s] %s", timestamp, msg)


async def log_and_broadcast(
    msg: str,
    connection_manager: ConnectionBroadcaster,
    level: str = "default",
    epoch: int | None = None,
) -> None:
    """Broadcast a log message via WebSocket without persisting to DB."""
    timestamp = datetime.now(tz=UTC).strftime("%H:%M:%S")
    log_entry = LogMessage(msg=msg, type=level, epoch=epoch)
    await connection_manager.broadcast(TelemetryMessage(type="log", data=log_entry))
    logger.info("[%s] %s", timestamp, msg)
