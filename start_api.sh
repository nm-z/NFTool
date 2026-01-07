#!/bin/bash
cd /home/nate/Desktop/NFTool
. .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
uvicorn src.api:app --host 0.0.0.0 --port 8001

