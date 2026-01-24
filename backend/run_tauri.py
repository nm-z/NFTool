"""Tauri Sidecar entry point for NFTool.

This is a headless entry point for the Python backend when running
as a Tauri sidecar. Unlike run_windows.py, it does not open a browser
since Tauri handles the window management.
"""

import multiprocessing
import os
import sys

import uvicorn

# Ensure imports work in frozen mode
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api import app  # noqa: E402

if __name__ == "__main__":
    # Required for multiprocessing to work correctly on Windows with PyInstaller
    multiprocessing.freeze_support()

    # Run on the fixed port expected by the frontend
    # Tauri's webview will connect to this port
    uvicorn.run(app, host="127.0.0.1", port=8001, workers=1)
