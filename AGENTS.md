# Repository Guidelines

## Project Structure & Module Organization
- `backend/`: FastAPI + PyTorch service. Entry point is `backend/src/api.py`; core packages live under `backend/src/{models,training,data,routers,services}`. Tests are in `backend/tests/`.
- `frontend/`: Next.js app. UI code is in `frontend/src/`, with e2e tests in `frontend/tests/e2e/`.
- `data/`: Training datasets (CSV/Parquet/JSON). Keep sample files small and sanitized.
- `workspace/`: Runtime artifacts (SQLite DB, logs, model outputs). Treat as generated; don’t commit.
- `docker-compose.yml`: Primary orchestration for local dev and ROCm GPU passthrough.

## Build, Test, and Development Commands
- `docker compose up`: Run full stack (backend on `http://localhost:8001`, frontend on `http://localhost:3000`).
- `docker compose restart backend`: Quick backend restart after Python-only changes.
- `docker compose up -d --no-deps --build backend`: Rebuild backend image after dependency/Dockerfile edits.
- `docker compose exec frontend npm run lint`: ESLint check (inside container).
- `docker compose exec frontend npm run build` / `npm run start`: Production build and serve (inside container).
- `docker compose exec backend python -m pytest`: Run backend tests (inside container).
- `docker compose exec backend python -m pylint src/`: Lint backend code (inside container).
- Use `docker compose exec <service> ...` for dev tooling so it matches the container runtime.

## Coding Style & Naming Conventions
- Python: 4-space indentation, `snake_case` modules/functions. Keep API types in `backend/src/schemas/`.
- TypeScript/React: follow ESLint rules from `frontend/eslint.config.mjs`. Prefer `PascalCase` for components and `camelCase` for hooks/vars.
- Match existing structure before adding new folders or patterns.

## Testing Guidelines
- Backend: pytest with Schemathesis in `backend/tests/`; name files `test_*.py`.
- Frontend: Playwright e2e tests in `frontend/tests/e2e/*.spec.ts`.
- Playwright expects the frontend running and the backend reachable (the tests wait for a CONNECTED state). Prefer running tests inside the containers so dependencies match runtime.

## Commit & Pull Request Guidelines
- Recent history uses short, sentence-case, imperative messages (e.g., “Fix training completion…”). Avoid ticket prefixes unless required.
- PRs should include: a brief summary, tests run, and UI screenshots when changing the frontend.

## Configuration & Runtime Notes
- API calls require `X-API-Key` (default `plyo` in `docker-compose.yml`).
- ROCm settings (e.g., `HSA_OVERRIDE_GFX_VERSION=11.0.0`) are wired in Docker for AMD GPUs.
- For deeper architecture details, see `CLAUDE.md`.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NFTool is a deep learning tool for modular regression analysis and training. It uses a FastAPI backend (Python/PyTorch) with a Next.js frontend, orchestrated via Docker Compose with ROCm GPU acceleration. Development is container-first; run tests, linting, and build steps inside the Docker services to match runtime dependencies.

## Development Commands

### Running the Application
```bash
# Start full stack
docker compose up

# Restart backend only (fastest, for Python code changes)
docker compose restart backend

# Rebuild backend (for dependency or Dockerfile changes)
docker compose up -d --no-deps --build backend

# Force recreate backend container
docker compose up -d --no-deps --build --force-recreate backend
```

The backend runs on `http://localhost:8001` and frontend on `http://localhost:3000`.

### Testing
```bash
# Run all backend tests
docker compose exec backend python -m pytest

# Run specific test file
docker compose exec backend python -m pytest tests/test_api_schemathesis.py

# Run with verbose output
docker compose exec backend python -m pytest -v
```

### Linting
```bash
# Run pylint on backend code
docker compose exec backend python -m pylint src/

# Check specific module
docker compose exec backend python -m pylint src/training/
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

**Key Design Patterns:**
- **Job Isolation**: Training runs execute in separate `multiprocessing.Process` instances to prevent blocking the API server
- **State Polling**: Training state is persisted to `workspace/nftool.db` and polled by the API
- **Scaler Persistence**: `StandardScaler` objects are saved as `.pkl` files alongside model checkpoints for consistent inference transformations
- **Authentication**: REST endpoints require `X-API-Key` header; WebSockets use `Sec-WebSocket-Protocol: api-key-<KEY>` subprotocol

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
Default API key: `plyo` (set in `.env` or `docker-compose.yml`)

ROCm-specific overrides:
- `HSA_OVERRIDE_GFX_VERSION=11.0.0` (for RDNA3 GPUs like RX 7700 XT)
- Device mappings: `/dev/kfd`, `/dev/dri`

## API Reference

### REST Endpoints (require `X-API-Key` header)
- `POST /train`: Start Optuna optimization run (body: `TrainingConfig` JSON)
- `POST /abort`: Terminate active training process
- `GET /runs`: List all historical runs from database
- `GET /datasets`: List available datasets in `data/` directory
- `GET /dataset/preview?path=<path>`: Preview first 20 rows with summary statistics
- `POST /load-weights`: Upload `.pt`/`.pth` model file
- `GET /download-weights/{run_id}`: Export best model weights
- `POST /inference`: Execute single prediction (body: `{"model_path": "...", "features": [...]}`)

### WebSocket (`/ws`)
Authentication: Subprotocol `api-key-{YOUR_KEY}`

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
Configured for AMD ROCm on RDNA3 hardware. Docker-level device passthrough enables GPU utilization within containers.

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
- Backend: DataLoader uses `drop_last=True` to prevent incomplete batches (training/engine.py:52,147)

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
