"""Expose a singleton JobQueue instance used across the application.

This module provides `job_queue`, a single `JobQueue` instance that
other modules import to enqueue background jobs.
"""

from src.services.job_queue import JobQueue

job_queue = JobQueue()
