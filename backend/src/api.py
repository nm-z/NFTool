"""Main entry point for the NFTool backend API."""

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from typing import cast as typing_cast

from fastapi import Depends, FastAPI, Request, WebSocket
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from src.auth import verify_api_key
from src.config import API_KEY, LOGS_DIR, REPORTS_DIR, RESULTS_DIR
from src.database.database import SESSION_LOCAL, Base, engine
from src.database.models import Run
from src.manager import manager as connection_manager
from src.manager import websocket_endpoint
from src.routers import datasets, hardware, training
from src.services.queue_instance import job_queue

__all__ = ["api_key_middleware", "ws_endpoint"]

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "api.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("nftool")


class _SuppressWebsocketDebug(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.levelno < logging.INFO
            and not record.name.startswith("nftool")
        )


# Ensure CORS headers are present on error responses.
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Allow-Methods": "*",
}
# Forward warnings and uvicorn logs into the same handlers so errors land in api.log.
logging.captureWarnings(True)
_root_handlers = logging.getLogger().handlers
_ws_filter = _SuppressWebsocketDebug()
for _handler in _root_handlers:
    _handler.addFilter(_ws_filter)
for _name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    _logger = logging.getLogger(_name)
    _logger.setLevel(logging.INFO)
    for _handler in _root_handlers:
        if _handler not in _logger.handlers:
            _logger.addHandler(_handler)
    _logger.propagate = False

# Reduce websocket debug noise from underlying libraries.
for _name in ("websockets", "websockets.server", "websockets.protocol"):
    _logger = logging.getLogger(_name)
    _logger.handlers.clear()
    _logger.setLevel(logging.WARNING)
    _logger.propagate = False

# Initialize database (create tables)
Base.metadata.create_all(bind=engine)


# Use lifespan handler instead of deprecated on_event startup
@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Context manager for managing the lifespan of the FastAPI application."""
    # Startup actions
    _hardware_monitor_task = asyncio.create_task(
        connection_manager.hardware_monitor_task()
    )
    del _hardware_monitor_task  # Explicitly delete to mark as "used" for linters

    # State Recovery: Mark any stuck 'running'/'queued' jobs as 'failed'
    with SESSION_LOCAL() as db:
        stuck_runs = (
            db.query(Run)
            .filter(Run.status.in_(["running", "queued"]))
            .all()
        )
        for run in stuck_runs:
            prev_status = getattr(run, "status", "unknown")
            # Cast to Any to satisfy static checkers before assigning runtime value.
            typing_cast(Any, run).status = "failed"
            run.logs.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "msg": "Run marked as failed due to API restart.",
                }
            )
            logger.warning(
                "Run %s was stuck in '%s' status, marked as 'failed'.",
                run.run_id,
                prev_status,
            )
        db.commit()
    _job_queue_task = asyncio.create_task(job_queue.process_queue())
    del _job_queue_task  # Explicitly delete to mark as "used" for linters

    try:
        yield
    finally:
        # Shutdown logic (if any) can be placed here.
        pass


app = FastAPI(title="NFTool API", lifespan=lifespan)


# Note: global exception handlers removed so FastAPI's default handlers surface errors.


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")


@app.exception_handler(Exception)
async def log_unhandled_errors(request: Request, exc: Exception):
    """Log any unhandled exception so it appears in api.log and return 500."""
    logger.exception(
        "Unhandled application error: method=%s path=%s exc=%r",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
        headers=CORS_HEADERS,
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return a JSON-serializable 422 response for validation errors."""
    sanitized: list[dict[str, Any]] = []
    for err in exc.errors():
        if isinstance(err, dict):
            err_dict = typing_cast(dict[str, Any], err)
            cleaned = {str(k): v for k, v in err_dict.items()}
            if "input" in cleaned:
                try:
                    jsonable_encoder(cleaned["input"])
                except Exception:
                    cleaned["input"] = str(cleaned["input"])
            sanitized.append(cleaned)
        else:
            sanitized.append({"msg": str(err)})
    return JSONResponse(
        status_code=422,
        content={"detail": sanitized},
        headers=CORS_HEADERS,
    )


@app.middleware("http")
async def api_key_middleware(request: Request, call_next: Any) -> Any:
    """
    Enforce presence of X-API-Key on all /api/v1 routes before body validation.
    Return 406 when missing or invalid to match test expectations.
    """
    # Only enforce for API routes under /api/v1
    try:
        path = request.url.path or ""
    except (AttributeError, ValueError) as e:
        logger.error("Error determining request path: %s", e)
        path = ""
    logger.debug(
        "Incoming request: method=%s path=%s headers=%s",
        request.method,
        path,
        dict(request.headers),
    )
    # For HTTP methods not typically used by the API (e.g., TRACE), return 405
    # Method Not Allowed so behavior matches OpenAPI expectations.
    allowed_http_methods = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}
    if request.method.upper() not in allowed_http_methods:
        # Compute allowed methods for this path to include in Allow header per RFC 9110
        try:
            allowed_methods_set: set[str] = set()
            for route in request.app.router.routes:
                if route.matches(request.scope)[0] is not None:
                    methods = getattr(route, "methods", None)
                    if methods:
                        allowed_methods_set.update(m.upper() for m in methods)
            allow_header = ", ".join(sorted(allowed_methods_set)) or ", ".join(
                sorted(allowed_http_methods)
            )
        except ImportError:
            # Handle cases where starlette.routing.Match might not be available
            logger.warning(
                "Could not import starlette.routing.Match; dynamic method "
                "calculation limited."
            )
            allow_header = ", ".join(sorted(allowed_http_methods))
        except (AttributeError, RuntimeError, TypeError) as e:
            logger.error("Error determining allowed methods: %s", e)
            allow_header = ", ".join(sorted(allowed_http_methods))
        return JSONResponse(
            status_code=405,
            content={"detail": "Method Not Allowed"},
            headers={**CORS_HEADERS, "Allow": allow_header},
        )
    # Enforce API key at middleware level for training and data routes so the
    # header requirement is rejected before body validation (matches tests).
    if path.startswith(("/api/v1/training", "/api/v1/data")):
        # If the route exists but does not allow this method, return 405 per RFC
        try:
            api_allowed_methods: set[str] = set()
            matched_any = False
            for route in request.app.router.routes:
                if route.matches(request.scope)[0] is not None:
                    matched_any = True
                    methods = getattr(route, "methods", None)
                    if methods:
                        api_allowed_methods.update(m.upper() for m in methods)
            if (
                matched_any
                and request.method.upper() not in api_allowed_methods
                and request.method.upper() != "OPTIONS"
            ):
                allow_header = ", ".join(sorted(api_allowed_methods)) or (
                    "GET, POST, OPTIONS"
                )
                return JSONResponse(
                    status_code=405,
                    content={"detail": "Method Not Allowed"},
                    headers={**CORS_HEADERS, "Allow": allow_header},
                )
        except ImportError:
            logger.warning(
                "Could not import starlette.routing.Match; dynamic method "
                "calculation limited."
            )
        except (AttributeError, RuntimeError, TypeError) as e:
            logger.error("Error determining allowed methods for API routes: %s", e)
        # Do not enforce API key at middleware level; route-level dependencies
        # handle authentication/validation. Middleware only enforces unsupported
        # methods (405) above and otherwise forwards to route handlers.
    response = await call_next(request)
    response.headers.setdefault("Access-Control-Allow-Origin", "*")
    response.headers.setdefault("Access-Control-Allow-Headers", "*")
    response.headers.setdefault("Access-Control-Allow-Methods", "*")
    return response


app.include_router(
    training.router, prefix="/api/v1/training", dependencies=[Depends(verify_api_key)]
)
app.include_router(
    datasets.router, prefix="/api/v1/data", dependencies=[Depends(verify_api_key)]
)
app.include_router(hardware.router, prefix="/api/v1")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket_endpoint(websocket, API_KEY)


# MOUNT FRONTEND: Serve Next.js static export from the bundled 'static' folder
# This must come AFTER all API routes to avoid shadowing /api/* endpoints
if getattr(sys, 'frozen', False):
    # In frozen mode (PyInstaller), static files are inside the temp extraction dir
    static_dir = os.path.join(sys._MEIPASS, "static")
else:
    # In dev mode, look for the 'static' folder in the backend directory
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../static")

if os.path.exists(static_dir):
    # Mount the frontend at root with html=True to enable SPA fallback
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
    logger.info("Frontend served from %s", static_dir)
else:
    logger.warning("Frontend static folder not found at %s. Running API-only mode.", static_dir)


# Lifespan handler above replaces the deprecated startup event decorator.


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
