# Technical Documentation: NFTool V3

## Architecture Overview

NFTool V3 follows a decoupled Client-Server architecture.

### Runtime & Local Development
NFTool runs natively as a Tauri desktop app. For local development, run the backend and frontend directly on the host so they match the desktop runtime.

### 1. Backend (Python/FastAPI)
The backend is modularized into several internal packages:
- `src.models`: PyTorch implementations of `RegressionNet` (MLP) and `CNNRegressionNet` (1D-CNN) with Residual connections.
- `src.training`: Core engine handling training loops, early stopping, and Optuna optimization.
- `src.data`: Preprocessing pipelines, multi-format loading (CSV, Parquet, JSON), and SNR calculation (RidgeCV).
- `src.api`: FastAPI application exposing REST and WebSocket endpoints.

### 2. Frontend (Next.js/React)
- **State Management**: Managed via `Zustand` with `persist` middleware for local storage synchronization.
- **Real-time Communication**: Uses WebSockets to receive live log streams, metrics, and hardware stats.
- **Error Handling**: Global React Error Boundaries and optimistic UI updates with automatic recovery.
- **Layout**: `react-resizable-panels` for a flexible, VS-Code like developer workspace.

## Key Innovations

### Regularized SNR Calculation
To provide robust metrics, the tool uses `RidgeCV` for SNR estimation. It automatically optimizes the regularization strength using leave-one-out cross-validation to prevent optimistic bias in high-dimensional settings.

### Advanced Residual CNN
The `CNNRegressionNet` uses `ResidualBlock1D` modules. These blocks implement skip connections ($x + f(x)$) allowing for deeper architectures (up to 10+ blocks) while maintaining stable gradient flow during hyperparameter optimization.

## Performance Optimization (AMD ROCm)
The tool is configured for RDNA3 hardware (e.g., RX 7700 XT) using:
- `HSA_OVERRIDE_GFX_VERSION=11.0.0`: Enables compatibility for newer GPU architectures.

## API Reference

### REST Endpoints (All require `X-API-Key` header)

#### `POST /train`
Starts a new Optuna optimization run.
- **Body**: `TrainingConfig` (JSON)
- **Returns**: `{"run_id": "PASS_..."}`

#### `POST /abort`
Terminates the active training process.

#### `GET /runs`
Returns all historical runs from the SQLite database.

#### `GET /datasets`
Lists available `.csv`, `.parquet`, and `.json` files in the `data/` directory.

#### `GET /dataset/preview?path={path}`
Returns the first 20 rows and detailed summary statistics (mean, std, missing values) for a dataset.

#### `POST /load-weights`
Uploads a `.pt` or `.pth` model file to the workspace.

#### `GET /download-weights/{run_id}`
Exports the best model weights from a specific run.

#### `POST /inference`
Executes a single prediction.
- **Body**: `{"model_path": "run_id_or_path", "features": [0.1, ...]}`

### WebSocket Protocol (`/ws`)
Subprotocol authentication: `api-key-{YOUR_KEY}`

**Message Types (Outbound):**
- `init`: Full state on connection (logs, metrics, hardware).
- `status`: Training progress, current trial, and engine state.
- `log`: Single log line from the process stream.
- `metrics`: New trial result (R², MAE, Loss).
- `hardware`: Real-time GPU/CPU utilization and temperature.
