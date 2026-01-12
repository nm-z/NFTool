import os
from pathlib import Path

REPO_ROOT = os.getenv("REPO_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
WORKSPACE_DIR = os.path.join(REPO_ROOT, "workspace")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(WORKSPACE_DIR, 'nftool.db')}")
API_KEY = os.getenv("API_KEY", "nftool-dev-key")
LOGS_DIR = os.path.join(WORKSPACE_DIR, "logs")

RESULTS_DIR = os.path.join(WORKSPACE_DIR, "runs/results")
REPORTS_DIR = os.path.join(WORKSPACE_DIR, "runs/reports")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
