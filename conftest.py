"""Pytest bootstrap: make the repo importable and keep runs headless."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Never try to open a GUI backend during tests (matters on Windows CI too).
os.environ.setdefault("MPLBACKEND", "Agg")
