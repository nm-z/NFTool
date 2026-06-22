"""Windows executable entry point for NFTool.

This script is the main entry point when running NFTool as a PyInstaller-built
Windows executable. It handles multiprocessing support and optionally opens
the browser to the frontend.
"""

import multiprocessing
import os
import sys
import webbrowser
from threading import Timer

import uvicorn

# Ensure imports work in frozen mode by adding the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api import app  # noqa: E402


def open_browser():
    """Open the default web browser to the NFTool frontend."""
    webbrowser.open("http://127.0.0.1:8001")


if __name__ == "__main__":
    # Required for multiprocessing to work correctly on Windows with PyInstaller
    multiprocessing.freeze_support()

    # Open browser after 1.5 seconds to allow the server to start
    Timer(1.5, open_browser).start()

    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)
