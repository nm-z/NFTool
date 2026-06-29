#!/usr/bin/env bash
# NFTool — one-shot setup + run (Linux / macOS).
# Forces a clean Python env, installs CPU PyTorch + the full ML stack, runs the
# test suite, then trains every registered architecture on the bundled dataset.
#
#   bash setup_and_run.sh                 # default short run (all architectures)
#   bash setup_and_run.sh --arch MLP      # any demo.py args pass straight through
#   bash setup_and_run.sh --trials 20 --epochs 100
set -euo pipefail
cd "$(dirname "$0")"

PY_PKGS=(numpy pandas scikit-learn optuna pytest matplotlib)
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

echo "==> [1/4] Creating virtual environment (.venv)"
if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 .venv >/dev/null 2>&1 || uv venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "==> [2/4] Installing CPU PyTorch + ML stack (uv)"
    uv pip install --index-strategy unsafe-best-match \
        torch --extra-index-url "$TORCH_INDEX" "${PY_PKGS[@]}"
else
    python3 -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python -m pip install --upgrade pip >/dev/null
    echo "==> [2/4] Installing CPU PyTorch + ML stack (pip)"
    python -m pip install --index-url "$TORCH_INDEX" torch
    python -m pip install "${PY_PKGS[@]}"
fi

echo "==> [3/4] Running test suite"
python -m pytest tests -q || true

echo "==> [4/4] Training all registered architectures on the bundled dataset"
if [ "$#" -eq 0 ]; then set -- --trials 5 --epochs 40; fi
python demo.py "$@"
