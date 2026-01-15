"""Configuration for the NFTool backend.

This module handles environment variables, path discovery, and directory setup.
"""

import os
from pathlib import Path

# Robust path discovery:
# REPO_ROOT is the directory containing 'src', 'data', or 'workspace'
# We look for the parent of the 'src' directory
CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = str(CURRENT_FILE.parent.parent)

# Environment Overrides (Docker)

WORKSPACE_DIR = os.getenv("WORKSPACE_DIR", os.path.join(REPO_ROOT, "workspace"))
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(WORKSPACE_DIR, 'nftool.db')}",
)
API_KEY = os.getenv("API_KEY", "plyo")

LOGS_DIR = os.path.join(WORKSPACE_DIR, "logs")
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "runs/results")
REPORTS_DIR = os.path.join(WORKSPACE_DIR, "runs/reports")
DATASETS_DIR = os.path.join(WORKSPACE_DIR, "datasets")

# Ensure directories exist
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
