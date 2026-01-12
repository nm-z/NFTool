import sys
import unittest.mock as mock
from unittest.mock import MagicMock
import os
import logging

# 1. Mock tkinter entirely before importing anything that uses it
mock_tk = MagicMock()
sys.modules["tkinter"] = mock_tk
sys.modules["tkinter.filedialog"] = mock_tk
sys.modules["tkinter.ttk"] = mock_tk

# Configuration
BASE_WORKSPACE = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_WORKSPACE, "Logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "headless.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("nftool-headless")

def log_print(*args, **kwargs):
    msg = " ".join(map(str, args))
    logger.info(msg)

# 2. Define the parameters to bypass the GUI
# ... existing code ...

# Headless defaults
HEADLESS_SETTINGS = {}

def mock_select_file(title):
    if "Predictor" in title:
        return "/home/nate/Desktop/NFTool/dataset/Predictors_2025-04-15_10-43_Hold-2.csv"
    if "Target" in title:
        return "/home/nate/Desktop/NFTool/dataset/9_10_24_Hold_02_targets.csv"
    return ""

def mock_prompt_initial_settings():
    return HEADLESS_SETTINGS

# 3. Patch the functions in the module
with mock.patch('NFTool_V3_071125A.select_file', side_effect=mock_select_file):
    with mock.patch('NFTool_V3_071125A.prompt_initial_settings', side_effect=mock_prompt_initial_settings):
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
        log_print("🚀 Starting Headless Training Run (Tkinter Mocked)...")
        import NFTool_V3_071125A  # noqa: F401
