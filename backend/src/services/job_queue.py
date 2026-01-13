"""Job queue for running training tasks in isolated processes."""

import asyncio
import importlib
import logging
import multiprocessing
from collections import deque
from typing import Any, cast

from sqlalchemy.orm import Session
from src.database.database import SessionLocal
from src.database.models import Run
from src.services.training_service import run_training_task
from src.utils.broadcast_utils import db_log_and_broadcast

logger = logging.getLogger("nftool")


class JobQueue:
    """Manage a FIFO queue of training jobs and run them in separate processes."""

    def __init__(self):
        self.queue = deque()
        self.active_job: dict[str, Any] | None = None
        # Use a loose typing for the active process object coming from
        # multiprocessing contexts (ctx.Process). Static checkers can be
        # conservative here; accept any to avoid false-positive "cannot assign"
        # errors.
        self.active_process: Any | None = None
        self.queue_lock = asyncio.Lock()
        # Keep references to background tasks created via asyncio.create_task to
        # avoid dangling-task warnings and premature GC.
        self._tasks: list[asyncio.Task] = []

    async def add_job(self, config_dict: dict[str, Any], run_id: str):
        """Add a training job to the FIFO queue.

        Updates the Run status to 'queued' in the database, broadcasts a
        telemetry message about the queued status, and starts queue processing
        if no job is currently active.
        """
        async with self.queue_lock:
            self.queue.append({"config": config_dict, "run_id": run_id})
            db = SessionLocal()
            try:
                run = db.query(Run).filter(Run.run_id == run_id).first()
                if run:
                    # Cast to Any to satisfy static type checkers for mapped ORM
                    # attributes on ORM result objects.
                    cast(Any, run).status = "queued"
                    db.commit()
                    # db.close()
            finally:
                db.close()
            # Import manager and message class dynamically to avoid circular
            # import at module import time while not using import statements
            # inside functions (avoids pylint import-outside-toplevel).
            connection_manager = importlib.import_module("src.manager").manager
            telemetry_message_cls = importlib.import_module(
                "src.schemas.websocket"
            ).TelemetryMessage

            await connection_manager.broadcast(
                telemetry_message_cls(
                    type="status",
                    data={
                        "status": "queued",
                        "run_id": run_id,
                        "queue_size": len(self.queue),
                        "active_job_id": (
                            self.active_job["run_id"] if self.active_job else None
                        ),
                    },
                )
            )
            logger.info(
                "Job %s added to queue. Queue size: %d",
                run_id,
                len(self.queue),
            )
            if not self.active_job:
                t = asyncio.create_task(self.process_queue())
                self._tasks.append(t)

    async def process_queue(self):
        """Pop the next job and start it in a separate process.

        This acquires the queue lock, updates the Run status to 'running',
        configures the manager state, spawns a process for `run_training_task`,
        and starts monitoring the process.
        """
        async with self.queue_lock:
            if self.active_job:
                return
            if not self.queue:
                logger.info("Job queue is empty.")
                return

            job = self.queue.popleft()
            self.active_job = job
            run_id = job["run_id"]
            config_dict = job["config"]

            db = SessionLocal()
            try:
                run = db.query(Run).filter(Run.run_id == run_id).first()
                if run:
                    cast(Any, run).status = "running"
                    db.commit()
            finally:
                db.close()

            # Lazily obtain the manager singleton dynamically to avoid import cycles.
            connection_manager = importlib.import_module("src.manager").manager

            connection_manager.active_run_id = run_id
            connection_manager.current_gpu_id = config_dict.get("gpu_id", 0)

            ctx = multiprocessing.get_context("spawn")
            # Do NOT pass the connection_manager object into the child process.
            # It may not be picklable; the training worker should import manager
            # itself if it needs to broadcast, or use DB updates instead.
            self.active_process = ctx.Process(
                target=run_training_task, args=(config_dict, run_id)
            )
            self.active_process.start()
            logger.info(
                "Job %s started. Active process PID: %s",
                run_id,
                getattr(self.active_process, "pid", None),
            )

            t = asyncio.create_task(self._monitor_active_job())
            self._tasks.append(t)

    async def _monitor_active_job(self):
        """Monitor the currently active process until it finishes.

        Safely handles races where `self.active_process` or `self.active_job`
        may be cleared by other coroutines and performs DB updates and
        broadcasts when the job completes.
        """
        # Robustly monitor the active_process. Read the process into a local
        # variable so we don't race with other coroutines that may clear
        # `self.active_process`. If the process is cleared to None while we're
        # running, stop monitoring gracefully.
        while True:
            proc = self.active_process
            if proc is None:
                # Nothing to monitor.
                return
            # Wait while the captured process is alive.
            try:
                while proc.is_alive():
                    await asyncio.sleep(1)
                break
            except (AttributeError, RuntimeError, TypeError):
                # In case proc becomes invalid or raises, bail out of monitoring
                # loop and continue with cleanup below.
                break

        # Process finished, update status and clear active job.
        # Capture the run id from the active job safely before any other mutations.
        if not self.active_job:
            logger.warning("Monitor ended but active_job is missing.")
            self.active_process = None
            t = asyncio.create_task(self.process_queue())
            self._tasks.append(t)
            return

        finished_run_id = self.active_job.get("run_id")
        db = SessionLocal()
        try:
            run = db.query(Run).filter(Run.run_id == finished_run_id).first()
            if run and cast(Any, run).status == "running":
                finished_proc = self.active_process
                exitcode = getattr(finished_proc, "exitcode", None)
                cast(Any, run).status = "completed" if exitcode == 0 else "failed"
                db.commit()
                if finished_run_id is not None:
                    # Dynamically obtain the manager singleton for broadcasting.
                    connection_manager = importlib.import_module("src.manager").manager
                    message = (
                        f"Job finished with exit code {exitcode}. "
                        f"Status: {cast(Any, run).status}"
                    )

                    db_log_and_broadcast(
                        db,
                        finished_run_id,
                        message,
                        connection_manager,
                        "info",
                    )
        finally:
            db.close()

        finished_proc = self.active_process
        exitcode = getattr(finished_proc, "exitcode", None)
        logger.info("Job %s finished with exit code %s", finished_run_id, exitcode)
        self.active_job = None
        self.active_process = None
        t = asyncio.create_task(self.process_queue())
        self._tasks.append(t)

    async def abort_active_job(self, db: Session):
        """Abort the currently active job if one exists.

        Terminates the child process, updates the Run status to 'aborted',
        and broadcasts the abort status.
        """
        if self.active_process and self.active_process.is_alive():
            # Ensure that active_job is present before attempting to index it.
            if not self.active_job:
                logger.warning("Abort requested but no active_job is present.")
                self.active_process.terminate()
                self.active_process.join()
                self.active_process = None
                t = asyncio.create_task(self.process_queue())
                self._tasks.append(t)
                return True

            run_id = self.active_job.get("run_id")
            self.active_process.terminate()
            self.active_process.join()
            run = db.query(Run).filter(Run.run_id == run_id).first()
            if run:
                cast(Any, run).status = "aborted"
                db.commit()
            # Dynamically obtain manager and TelemetryMessage to avoid import cycles.
            connection_manager = importlib.import_module("src.manager").manager
            telemetry_message_cls = importlib.import_module(
                "src.schemas.websocket"
            ).TelemetryMessage

            await connection_manager.broadcast(
                telemetry_message_cls(
                    type="status",
                    data={
                        "is_running": False,
                        "is_aborting": False,
                        "progress": 0,
                        "run_id": run_id,
                        "queue_size": len(self.queue),
                        "active_job_id": None,
                    },
                )
            )
            logger.info("Job %s aborted.", run_id)
            self.active_job = None
            self.active_process = None
            t = asyncio.create_task(self.process_queue())
            self._tasks.append(t)
            return True
        return False

    def get_status(self):
        """Return a small dictionary describing queue and active-job status."""
        return {
            "active_job_id": self.active_job["run_id"] if self.active_job else None,
            "queue_size": len(self.queue),
            "is_processing": self.active_job is not None,
        }
