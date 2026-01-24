"""Configuration for the NFTool backend.

This module handles environment variables, path discovery, and directory setup.
Designed to work both in development (source) and production (Tauri bundled).
"""

import os
import sys
from pathlib import Path

# 1. Determine the Root Workspace
# NFTOOL_WORKSPACE is set by Tauri's run_tauri.py sidecar launcher
# pointing to AppData on Windows for writable storage
WORKSPACE_OVERRIDE = os.environ.get("NFTOOL_WORKSPACE")

if WORKSPACE_OVERRIDE:
    # PRODUCTION: Write to %APPDATA%/com.nftool.app/ (passed from Rust)
    BASE_DIR = Path(WORKSPACE_OVERRIDE)
    REPO_ROOT = BASE_DIR  # In production, everything is in the workspace
else:
    # DEVELOPMENT: Write to local repo folders
    if getattr(sys, 'frozen', False):
        # If frozen but no override, use exe directory
        BASE_DIR = Path(os.path.dirname(sys.executable))
        REPO_ROOT = BASE_DIR
    else:
        # Running from source
        # This file is at backend/src/config.py
        # BASE_DIR is workspace/ for runtime artifacts
        # REPO_ROOT is the actual repository root for data/
        source_file = Path(__file__).resolve()  # backend/src/config.py
        REPO_ROOT = source_file.parent.parent.parent  # Go up to repo root
        BASE_DIR = REPO_ROOT / "workspace"  # Runtime artifacts in workspace/

# 2. Define Subdirectories
WORKSPACE_DIR = str(BASE_DIR)
LOGS_DIR = str(BASE_DIR / "logs")
RESULTS_DIR = str(BASE_DIR / "runs" / "results")
REPORTS_DIR = str(BASE_DIR / "runs" / "reports")
DATASETS_DIR = str(BASE_DIR / "datasets")

# 3. Create directories immediately so other modules don't fail
for path_str in [LOGS_DIR, RESULTS_DIR, REPORTS_DIR, DATASETS_DIR]:
    os.makedirs(path_str, exist_ok=True)

# 4. Database Path
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR / 'nftool.db'}",
)

# 5. No API Key - Tauri apps run locally without authentication
API_KEY = None
