# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NFTool is a deep learning tool for modular regression analysis and training. It uses a FastAPI backend (Python/PyTorch) with a Next.js frontend, packaged as a Tauri desktop app.
Development is native-first; run tests, linting, and build steps on the host to match the desktop runtime.

Companion docs in the repo root: `AGENTS.md` (contributor/style conventions), `TECHNICAL.md`, and `TAURI_IMPLEMENTATION.md` (desktop packaging details).

## Development Commands

### Running the Application
```bash
# Start the Tauri desktop app (spins up the backend sidecar)
npm run tauri:dev

# Start the backend API directly (API-only mode)
python backend/src/api.py
```

The backend runs on `http://localhost:8001` and frontend on `http://localhost:3000`.

### Testing
```bash
# Backend (pytest + Schemathesis). The npm wrapper uses backend/.venv:
npm run test:backend                              # == cd backend && .venv/bin/python -m pytest tests
python -m pytest backend/tests                    # direct, if deps are on the active interpreter
python -m pytest backend/tests/test_api_schemathesis.py   # single file
python -m pytest -v backend/tests                 # verbose

# Frontend e2e (Playwright). `npm run test:e2e` boots the backend on :8001 and
# runs the suite; it kills anything already on :8001/:3000 first.
npm run test:e2e
npm --prefix frontend run test:ui        # ui-only project (no backend needed)
npm --prefix frontend run test:workflow  # workflow project (needs backend reachable)
```
Playwright projects are defined in `frontend/playwright.config.ts`: `ui-only`, `workflow`, `chromium`. The `workflow`/`chromium` suites wait for a CONNECTED state, so the backend must be running and reachable.

### Linting & Build
```bash
python -m pylint backend/src/            # backend lint (config in .pylintrc)
npm --prefix frontend run lint           # frontend ESLint (frontend/eslint.config.mjs)

npm run tauri:build                      # full desktop build
npm run frontend:build                   # Next.js production build only
```

## Architecture

### Backend (FastAPI/PyTorch)
Entry point: `backend/src/api.py`

**Core Modules:**
- `src.models`: Neural network architectures (`RegressionNet`, `CNNRegressionNet` with residual blocks)
- `src.training`: Training loops, early stopping, Optuna hyperparameter optimization
- `src.data`: Multi-format data loading (CSV/Parquet/JSON), preprocessing, SNR calculation via RidgeCV
- `src.services`: Job queue system using multiprocessing for isolated training runs
- `src.routers`: REST API endpoints (datasets, hardware, training)
- `src.database`: SQLite ORM models for run metadata and logs
- `src.manager`: WebSocket connection manager for real-time updates
- `src.schemas`: Pydantic request/response models (`training.py`, `websocket.py`) — API contract lives here
- `src.utils`: `hardware.py` (GPU/CPU telemetry), `broadcast_utils.py` (WS fan-out), `reporting.py` (run reports)

**Key Design Patterns:**
- **Job Isolation**: Training runs execute in separate `multiprocessing.Process` instances to prevent blocking the API server
- **State Polling**: Training state is persisted to `workspace/nftool.db` and polled by the API
- **Scaler Persistence**: `StandardScaler` objects are saved as `.pkl` files alongside model checkpoints for consistent inference transformations
- **Authentication (disabled by default)**: `config.py` hardcodes `API_KEY = None` ("Tauri apps run locally without authentication"). With `API_KEY = None`, the `verify_api_key` dependency accepts any request and the `X-API-Key` header is optional — it is NOT read from the environment. The header / WebSocket `Sec-WebSocket-Protocol: api-key-<KEY>` plumbing exists in `auth.py` and `manager.py` and only enforces a key if `API_KEY` is set to a non-None value in `config.py`.

### Frontend (Next.js/React)
Entry point: `frontend/src/app/`

**Architecture:**
- **State Management**: Zustand stores with localStorage persistence
- **Real-time Updates**: WebSocket subscriptions for logs, metrics, and hardware telemetry
- **Layout**: Resizable panels via `react-resizable-panels` for workspace flexibility
- **Data Visualization**: Recharts for training metrics and performance graphs

## File Structure

### Runtime Artifacts (`workspace/` - gitignored)
- `nftool.db`: SQLite database storing run metadata and training logs
- `logs/`: Application logs (e.g., `api.log`)
- `runs/`: Per-training-run outputs
  - `results/`: Optuna trial summaries and optimization results
  - `reports/`: Model checkpoints (`.pt`), scalers (`.pkl`), and performance reports

### Data (`data/`)
Training datasets in CSV, Parquet, or JSON format.

### Scripts (`scripts/`)
Developer utilities for headless execution and legacy compatibility.

## Configuration

### Immutable Configuration Files
The following files are read-only and must NOT be modified:
- `.vscode/settings.json`, `.vscode/tasks.json`
- `pyrightconfig.json`, `.pylintrc`, `.trivyignore`
- `backend/pyproject.toml`
- `.cursorrules`

These files define strict typing rules, linting configurations, and project constraints.

### Environment Variables
No API key is required by default (`config.py` sets `API_KEY = None`). `.env.example` ships `API_KEY=nftool-dev-key` and `NEXT_PUBLIC_API_KEY=nftool-dev-key`, but the backend does not currently read `API_KEY` from the environment — auth only activates if `config.py` is edited to a non-None value.

ROCm-specific overrides:
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` (for RDNA3 GPUs like RX 7700 XT)
- Device mappings: `/dev/kfd`, `/dev/dri`

## API Reference

### REST Endpoints (`X-API-Key` header optional unless `API_KEY` is set in `config.py`)
- `POST /train`: Start Optuna optimization run (body: `TrainingConfig` JSON)
- `POST /abort`: Terminate active training process
- `GET /runs`: List all historical runs from database
- `GET /datasets`: List available datasets in `data/` directory
- `GET /dataset/preview?path=<path>`: Preview first 20 rows with summary statistics
- `POST /load-weights`: Upload `.pt`/`.pth` model file
- `GET /download-weights/{run_id}`: Export best model weights
- `POST /inference`: Execute single prediction (body: `{"model_path": "...", "features": [...]}`)

### WebSocket (`/ws`)
Authentication: Subprotocol `api-key-{YOUR_KEY}` (only enforced when `API_KEY` is non-None in `config.py`; otherwise accepted without a key)

**Message Types (server → client):**
- `init`: Full state on connection (logs, metrics, hardware)
- `status`: Training progress, current trial, engine state
- `log`: Single log line from training process
- `metrics`: Trial results (R², MAE, loss)
- `hardware`: Real-time GPU/CPU utilization and temperature

## Model Architectures

### RegressionNet (Dense MLP)
Configurable fully-connected layers with dropout and batch normalization.

### CNNRegressionNet (1D-CNN)
Uses `ResidualBlock1D` modules with skip connections for stable gradient flow in deep networks (10+ blocks). Optimized for high-dimensional time-series regression.

## Performance Notes

### GPU Acceleration
Configured for AMD ROCm on RDNA3 hardware when available on the host system.

### SNR Calculation
Uses `RidgeCV` with leave-one-out cross-validation to provide regularized SNR estimates, preventing optimistic bias in high-dimensional feature spaces.

## Important Constraints and Common Issues

### BatchNorm Batch Size Requirement
**Critical**: CNN models use `BatchNorm1d` layers which require **batch_size >= 2**. Single-sample batches will fail with:
```
ValueError: Expected more than 1 value per channel when training
```

**Solutions implemented:**
- Frontend: Batch size slider minimum is set to 2 in `Inspector.tsx`
- Backend: DataLoader uses `drop_last=True` to prevent incomplete batches (training/engine.py:52,149)

### Training Update Frequency
- Optuna trials update **once per epoch**, not per batch
- The `on_epoch_end` callback fires after each full epoch completes
- Live metrics (loss, R², MAE) are calculated and broadcast after each epoch
- WebSocket clients receive real-time updates via `TelemetryMessage` broadcasts

### Database Schema
The SQLite database (`workspace/nftool.db`) stores runs with JSON columns:
- `logs`: Array of `{time, msg, type, epoch}` objects
- `metrics_history`: Array of `{trial, epoch, loss, r2, mae, val_loss}` objects
- `config`: Full training configuration as JSON

Logs and metrics are stored in-process during training and persisted to disk for WebSocket polling.

### Frontend Development
After modifying frontend code, users must **refresh the browser** to see changes. The Next.js dev server hot-reloads JavaScript but state persists in Zustand stores.
