"""Tauri Sidecar entry point for NFTool.

This script orchestrates the startup handshake between Python and Rust:
1. Receives workspace path from Tauri
2. Sets NFTOOL_WORKSPACE env var BEFORE importing config
3. Selects an available port
4. Prints the port to STDOUT (Rust is listening)
5. Starts the FastAPI server

Usage:
    nftool-backend --workspace /path/to/app/data
"""

import argparse
import multiprocessing
import os
import socket
import sys

# Ensure imports work in frozen mode
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def get_free_port() -> int:
    """Find an available port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    parser = argparse.ArgumentParser(description="NFTool Backend Sidecar")
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Path to writable workspace directory (passed from Tauri)",
    )
    args = parser.parse_args()

    # 1. SET ENV VAR BEFORE IMPORTING APP
    # This ensures src.config sees the correct path
    os.environ["NFTOOL_WORKSPACE"] = args.workspace

    # 2. Pick a Port
    port = get_free_port()

    # 3. Print Handshake (Rust is listening for this specific string)
    # We flush immediately to ensure Rust sees it instantly
    print(f"NFTOOL_READY:{port}", flush=True)

    # 4. Import & Run App
    # Delay import until env is set
    import uvicorn
    from src.api import app  # noqa: E402

    # Run strictly on localhost for security
    uvicorn.run(app, host="127.0.0.1", port=port, workers=1, log_level="error")


if __name__ == "__main__":
    # Required for multiprocessing to work correctly on Windows with PyInstaller
    multiprocessing.freeze_support()
    main()
