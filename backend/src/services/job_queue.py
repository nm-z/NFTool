import asyncio
import logging
import multiprocessing
from collections import deque
from typing import Any, cast
from typing import Any as TypingAny

from sqlalchemy.orm import Session
from src.database.database import SessionLocal
from src.database.models import Run
from src.services.training_service import run_training_task
from src.utils.broadcast_utils import db_log_and_broadcast

logger = logging.getLogger("nftool")


class JobQueue:
    def __init__(self):
        self.queue = deque()
        self.active_job: dict[str, Any] | None = None
        # Use a loose typing for the active process object coming from
        # multiprocessing contexts (ctx.Process). Static checkers can be
        # conservative here; accept any to avoid false-positive "cannot assign"
        # errors.
        self.active_process: TypingAny | None = None
        self.queue_lock = asyncio.Lock()

    async def add_job(self, config_dict: dict[str, Any], run_id: str):
        async with self.queue_lock:
            self.queue.append({"config": config_dict, "run_id": run_id})
            db = SessionLocal()
            try:
                run = db.query(Run).filter(Run.run_id == run_id).first()
                if run:
                    # Cast to Any to satisfy static type checkers for mapped ORM attributes
                    cast(TypingAny, run).status = "queued"
                    db.commit()
                    # db.close()
            finally:
                db.close()
            # Import manager lazily to avoid circular import at module import time.
            from src.manager import manager as connection_manager
            from src.schemas.websocket import TelemetryMessage

            await connection_manager.broadcast(
                TelemetryMessage(
                    type="status", data={"status": "queued", "run_id": run_id}
                )
            )
            logger.info(f"Job {run_id} added to queue. Queue size: {len(self.queue)}")
            if not self.active_job:
                asyncio.create_task(self.process_queue())

    async def process_queue(self):
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
                    cast(TypingAny, run).status = "running"
                    db.commit()
            finally:
                db.close()

            # Lazily import manager to avoid circular import during module load.
            from src.manager import manager as connection_manager

            connection_manager.active_run_id = run_id
            connection_manager.current_gpu_id = config_dict.get("gpu_id", 0)

            ctx = multiprocessing.get_context("spawn")
            # Do NOT pass the connection_manager object into the child process (may not be picklable).
            # The training worker should import manager itself if it needs to broadcast, or use db updates.
            self.active_process = ctx.Process(
                target=run_training_task, args=(config_dict, run_id)
            )
            self.active_process.start()
            logger.info(
                f"Job {run_id} started. Active process PID: {self.active_process.pid}"
            )

            asyncio.create_task(self._monitor_active_job())

    async def _monitor_active_job(self):
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
            except Exception:
                # In case proc becomes invalid or raises, bail out of monitoring
                # loop and continue with cleanup below.
                break

        # Process finished, update status and clear active job.
        # Capture the run id from the active job safely before any other mutations.
        if not self.active_job:
            logger.warning("Monitor ended but active_job is missing.")
            self.active_process = None
            asyncio.create_task(self.process_queue())
            return

        finished_run_id = self.active_job.get("run_id")
        db = SessionLocal()
        try:
            run = db.query(Run).filter(Run.run_id == finished_run_id).first()
            if run and cast(TypingAny, run).status == "running":
                finished_proc = self.active_process
                exitcode = getattr(finished_proc, "exitcode", None)
                cast(TypingAny, run).status = "completed" if exitcode == 0 else "failed"
                db.commit()
                if finished_run_id is not None:
                    # Lazy import manager for broadcasting
                    from src.manager import manager as connection_manager

                    db_log_and_broadcast(
                        db,
                        finished_run_id,
                        f"Job finished with exit code {exitcode}. Status: {cast(TypingAny, run).status}",
                        connection_manager,
                        "info",
                    )
        finally:
            db.close()

        finished_proc = self.active_process
        exitcode = getattr(finished_proc, "exitcode", None)
        logger.info(f"Job {finished_run_id} finished with exit code {exitcode}")
        self.active_job = None
        self.active_process = None
        asyncio.create_task(self.process_queue())

    async def abort_active_job(self, db: Session):
        if self.active_process and self.active_process.is_alive():
            # Ensure that active_job is present before attempting to index it.
            if not self.active_job:
                logger.warning("Abort requested but no active_job is present.")
                self.active_process.terminate()
                self.active_process.join()
                self.active_process = None
                asyncio.create_task(self.process_queue())
                return True

            run_id = self.active_job.get("run_id")
            self.active_process.terminate()
            self.active_process.join()
            run = db.query(Run).filter(Run.run_id == run_id).first()
            if run:
                cast(TypingAny, run).status = "aborted"
                db.commit()
            # Lazy import manager for broadcasting
            from src.manager import manager as connection_manager
            from src.schemas.websocket import TelemetryMessage

            await connection_manager.broadcast(
                TelemetryMessage(
                    type="status",
                    data={
                        "is_running": False,
                        "is_aborting": False,
                        "progress": 0,
                        "run_id": run_id,
                    },
                )
            )
            logger.info(f"Job {run_id} aborted.")
            self.active_job = None
            self.active_process = None
            asyncio.create_task(self.process_queue())
            return True
        return False

    def get_status(self):
        return {
            "active_job_id": self.active_job["run_id"] if self.active_job else None,
            "queue_size": len(self.queue),
            "is_processing": self.active_job is not None,
        }
