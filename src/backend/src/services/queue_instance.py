"""Expose a singleton JobQueue instance used across the application.

This module provides `job_queue`, a single `JobQueue` instance that
other modules import to enqueue background jobs.
"""

# Import the module, not the class, to avoid circular imports
import src.services.job_queue as job_queue_module

# Create the singleton instance using the module
job_queue = job_queue_module.JobQueue()
