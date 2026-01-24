# Tauri Desktop App Implementation

This document summarizes the implementation of the Tauri-based Windows desktop application for NFTool.

## Architecture Overview

The application follows a **sidecar architecture** where:

1. **Rust (Main Process)** - Tauri manages the application lifecycle
2. **Python (Sidecar)** - FastAPI backend runs as a subprocess
3. **Frontend (UI)** - Next.js static export bundled with the app

### Key Features

- ✅ **Dynamic Port Selection**: Python backend auto-selects an available port
- ✅ **Writable AppData**: All runtime data stored in `%APPDATA%\com.nftool.app\`
- ✅ **No Authentication**: API key system removed for local-only access
- ✅ **Startup Handshake**: Rust waits for Python to be ready before UI initialization

## File Structure

```
NFTool/
├── backend/
│   ├── src/
│   │   ├── config.py          # Updated to use NFTOOL_WORKSPACE env var
│   │   ├── api.py             # Removed API key authentication
│   │   └── manager.py         # Updated WebSocket to skip auth
│   └── run_tauri.py           # Sidecar entrypoint with handshake
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx     # Updated CSP for Tauri
│   │   │   └── page.tsx       # Removed API key headers
│   │   ├── components/
│   │   │   └── ApiProvider.tsx # NEW: Dynamic API URL provider
│   │   └── lib/
│   │       └── api.ts         # NEW: Tauri integration utilities
│   └── out/                   # Static export (created by build)
├── src-tauri/
│   ├── src/
│   │   └── main.rs            # Simplified: No API key, auto port
│   ├── binaries/              # Sidecar binaries (created by build)
│   │   └── nftool-backend-*.exe
│   ├── Cargo.toml             # Removed unused `rand` dependency
│   └── tauri.conf.json        # Sidecar configuration
└── .github/
    └── workflows/
        └── deploy_windows.yml # CI/CD pipeline
```

## Implementation Details

### Phase 1: Backend Refactor

#### 1.1 Dynamic Configuration (`backend/src/config.py`)

```python
# Before importing, check for Tauri-provided workspace override
WORKSPACE_OVERRIDE = os.environ.get("NFTOOL_WORKSPACE")

if WORKSPACE_OVERRIDE:
    # PRODUCTION: Write to %APPDATA%/com.nftool.app/
    BASE_DIR = Path(WORKSPACE_OVERRIDE)
else:
    # DEVELOPMENT: Write to local repo folders
    BASE_DIR = Path(__file__).resolve().parent.parent
```

**Why**: Tauri apps installed in `Program Files` are read-only. All runtime data (logs, database, models) must be stored in a writable location like AppData.

#### 1.2 Sidecar Entrypoint (`backend/run_tauri.py`)

```python
def main():
    # 1. Set environment variable BEFORE importing app
    os.environ["NFTOOL_WORKSPACE"] = args.workspace

    # 2. Auto-select available port
    port = get_free_port()

    # 3. Print handshake (Rust listens for this)
    print(f"NFTOOL_READY:{port}", flush=True)

    # 4. Import and start server
    import uvicorn
    from src.api import app
    uvicorn.run(app, host="127.0.0.1", port=port)
```

**Gotcha**: The handshake must be printed **before** uvicorn starts, and the env var must be set **before** importing `src.config`.

#### 1.3 Remove Authentication (`backend/src/api.py`, `backend/src/manager.py`)

- Removed `X-API-Key` header validation from all routes
- Updated CORS to allow Tauri origins (`tauri://localhost`)
- WebSocket accepts connections without subprotocol authentication

### Phase 2: Rust Glue Layer

#### 2.1 Startup Sequence (`src-tauri/src/main.rs`)

```rust
// 1. Determine writable AppData directory
let app_data_dir = app.path_resolver().app_data_dir()?;

// 2. Spawn Python sidecar with workspace argument
Command::new_sidecar("nftool-backend")
    .args(["--workspace", &workspace_path])
    .spawn()?;

// 3. Listen for "NFTOOL_READY:PORT" on STDOUT
if line.starts_with("NFTOOL_READY:") {
    let port = parse_port(line);
    state.api_port.set(port);
    app_handle.emit_all("backend-ready", port);
}
```

**Gotcha**: Parse the port from STDOUT **before** the frontend tries to connect.

#### 2.2 Tauri Command for Frontend (`src-tauri/src/main.rs`)

```rust
#[tauri::command]
fn get_api_port(state: State<AppState>) -> u16 {
    *state.api_port.lock().unwrap()
}
```

Frontend calls this via `invoke("get_api_port")` to construct the base URL.

### Phase 3: Frontend Integration

#### 3.1 API Utility (`frontend/src/lib/api.ts`)

```typescript
export async function getBaseUrl(): Promise<string> {
    if (isTauri) {
        const { invoke } = window.__TAURI__.tauri;
        const port = await invoke<number>("get_api_port");
        return `http://127.0.0.1:${port}`;
    }
    return "http://localhost:8001"; // Dev fallback
}
```

**Gotcha**: Include retry logic in case the port isn't ready yet.

#### 3.2 API Provider (`frontend/src/components/ApiProvider.tsx`)

Wraps the entire app and resolves API URLs **before** rendering children:

```tsx
<ApiProvider>
    <Dashboard />
</ApiProvider>
```

Shows a loading screen until `getBaseUrl()` resolves.

#### 3.3 Updated CSP (`frontend/src/app/layout.tsx`)

```html
<meta httpEquiv="Content-Security-Policy"
      content="connect-src 'self' http://127.0.0.1:* ws://127.0.0.1:*" />
```

Allows dynamic ports (`:*` wildcard).

### Phase 4: CI/CD Pipeline

#### 4.1 Build Order (`.github/workflows/deploy_windows.yml`)

1. **Build Frontend** → `frontend/out/`
2. **Build Python Sidecar** → `src-tauri/binaries/nftool-backend.exe`
3. **Rename for Tauri** → `nftool-backend-x86_64-pc-windows-msvc.exe`
4. **Build Tauri Bundle** → `.msi` and `.exe` installers

**Gotcha**: Use `--onefile` with PyInstaller to avoid directory complexity.

## Testing Locally

### Development Mode (API-only)

```bash
# Start backend (API-only mode)
python backend/src/api.py

# Start frontend
cd frontend && npm run dev

# Navigate to http://localhost:3000
```

### Tauri Development Mode

```bash
# Terminal 1: Start Next.js dev server
cd frontend && npm run dev

# Terminal 2: Start Tauri (launches Python sidecar)
npm run tauri dev
```

The Tauri window will auto-reload when you edit frontend code.

### Production Build

```bash
# Build everything and create installer
npm run tauri build

# Output:
# - src-tauri/target/release/bundle/msi/NFTool_0.1.0_x64_en-US.msi
# - src-tauri/target/release/bundle/nsis/NFTool_0.1.0_x64-setup.exe
```

## Troubleshooting

### "Port 0 returned from backend"

- Python backend hasn't finished starting yet
- Check: `src-tauri/binaries/` contains the sidecar exe
- Check: Sidecar prints `NFTOOL_READY:PORT` to stdout

### "Module not found" when running sidecar

- PyInstaller didn't bundle all dependencies
- Solution: Add `--hidden-import=<module>` to pyinstaller command
- Check: Run `dist/nftool-backend/nftool-backend.exe` directly to test

### "Failed to create workspace directory"

- AppData path is invalid or inaccessible
- Check: `app.path_resolver().app_data_dir()` returns valid path
- Typical path: `C:\Users\<Name>\AppData\Roaming\com.nftool.app\`

### WebSocket connection fails

- CSP blocks dynamic ports
- Solution: Ensure CSP includes `ws://127.0.0.1:*`
- Check: Browser console for CSP violations

## Key Differences: API-only vs Tauri

| Feature | API-only (Dev) | Tauri (Production) |
|---------|---------------|-------------------|
| Backend URL | `http://localhost:8001` | `http://127.0.0.1:<dynamic>` |
| Authentication | API key required | None (local-only) |
| Data Storage | `workspace/` (repo) | `%APPDATA%\com.nftool.app\` |
| Port Selection | Hardcoded 8001 | Auto-selected |
| Process Model | Local process | Sidecar subprocess |

## Next Steps

- [ ] Add auto-updater (Tauri supports signed updates)
- [ ] Implement crash reporting (Sentry integration)
- [ ] Add Linux/macOS builds (update CI matrix)
- [ ] Sign Windows installer (requires code signing certificate)
- [ ] Add telemetry for usage analytics

## Resources

- [Tauri Sidecar Documentation](https://tauri.app/v1/guides/building/sidecar)
- [PyInstaller Options](https://pyinstaller.org/en/stable/usage.html)
- [Next.js Static Export](https://nextjs.org/docs/app/building-your-application/deploying/static-exports)
- [FastAPI CORS](https://fastapi.tiangolo.com/tutorial/cors/)
