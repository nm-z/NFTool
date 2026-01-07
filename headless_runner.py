import sys
from unittest.mock import MagicMock

# 1. Mock tkinter entirely before importing anything that uses it
mock_tk = MagicMock()
sys.modules["tkinter"] = mock_tk
sys.modules["tkinter.filedialog"] = mock_tk
sys.modules["tkinter.ttk"] = mock_tk

import unittest.mock as mock
import pandas as pd
import numpy as np
import os

# 2. Define the parameters to bypass the GUI
# Replicating the client's "Gold Standard" environment
HEADLESS_SETTINGS = (
    False,          # load_existing_model
    False,          # inference_mode
    15557,          # seed
    "NN",           # model_choice
    100,            # patience
    0.70,           # train_ratio
    0.15,           # val_ratio
    0.15,           # test_ratio
    10,             # trials (Reduced for verification)
    None,           # target_r2_value
    None,           # optuna_time_limit
    ["AdamW"],      # optimizer_choices
    1,              # num_layers_min
    8,              # num_layers_max
    128,            # layer_size_min
    1024,           # layer_size_max
    1e-6,           # learning_rate_min
    1e-2,           # learning_rate_max
    0.0,            # dropout_min
    0.5,            # dropout_max
    1,              # hidden_dim_min
    100,            # hidden_dim_max
    0.0,            # alpha_min
    0.0,            # alpha_max
    0               # k_folds
)

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
        print("🚀 Starting Headless Training Run (Tkinter Mocked)...")
        import NFTool_V3_071125A
