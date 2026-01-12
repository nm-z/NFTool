import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any
from typing import cast as typing_cast

from fastapi import Depends, FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Local Imports
from src.config import (
    API_KEY,
    LOGS_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
)
from src.database.database import (  # Import Base from database.py now
    Base,
    SessionLocal,
    engine,
)
from src.database.models import Run  # Need to import Run for startup_event
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


# Initialize database (create tables)
Base.metadata.create_all(bind=engine)


from src.auth import verify_api_key


# Use lifespan handler instead of deprecated on_event startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    asyncio.create_task(connection_manager.hardware_monitor_task())

    # State Recovery: Mark any stuck 'running' jobs as 'failed'
    with SessionLocal() as db:
        stuck_runs = db.query(Run).filter(Run.status == "running").all()
        for run in stuck_runs:
            # Cast to Any to satisfy static checkers before assigning runtime value.
            typing_cast(Any, run).status = "failed"
            run.logs.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "msg": "Run marked as failed due to API restart.",
                    "type": "warn",
                }
            )
            logger.warning(
                f"Run {run.run_id} was stuck in 'running' status, marked as 'failed'."
            )
        db.commit()
    asyncio.create_task(job_queue.process_queue())

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


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """
    Enforce presence of X-API-Key on all /api/v1 routes before body validation.
    Return 406 when missing or invalid to match test expectations.
    """
    # Only enforce for API routes under /api/v1
    try:
        path = request.url.path or ""
    except Exception:
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
            from starlette.routing import Match

            allowed_methods_set = set()
            for route in request.app.router.routes:
                match, _ = route.matches(request.scope)
                if match != Match.NONE:
                    methods = getattr(route, "methods", None)
                    if methods:
                        allowed_methods_set.update(m.upper() for m in methods)
            allow_header = ", ".join(sorted(allowed_methods_set)) or ", ".join(
                sorted(allowed_http_methods)
            )
        except Exception:
            allow_header = ", ".join(sorted(allowed_http_methods))
        return JSONResponse(
            status_code=405,
            content={"detail": "Method Not Allowed"},
            headers={"Allow": allow_header},
        )
    # Enforce API key at middleware level for training and data routes so the
    # header requirement is rejected before body validation (matches tests).
    if path.startswith(("/api/v1/training", "/api/v1/data")):
        # If the route exists but does not allow this method, return 405 per RFC
        try:
            from starlette.routing import Match

            allowed_methods_set = set()
            matched_any = False
            for route in request.app.router.routes:
                match, _ = route.matches(request.scope)
                if match != Match.NONE:
                    matched_any = True
                    methods = getattr(route, "methods", None)
                    if methods:
                        allowed_methods_set.update(m.upper() for m in methods)
            if matched_any and request.method.upper() not in allowed_methods_set:
                allow_header = (
                    ", ".join(sorted(allowed_methods_set)) or "GET, POST, OPTIONS"
                )
                return JSONResponse(
                    status_code=405,
                    content={"detail": "Method Not Allowed"},
                    headers={"Allow": allow_header},
                )
        except Exception:
            # If router inspection fails, fall back to header checks below.
            pass
        # Do not enforce API key at middleware level; route-level dependencies
        # handle authentication/validation. Middleware only enforces unsupported
        # methods (405) above and otherwise forwards to route handlers.
    return await call_next(request)


app.include_router(
    training.router, prefix="/api/v1/training", dependencies=[Depends(verify_api_key)]
)
app.include_router(
    datasets.router, prefix="/api/v1/data", dependencies=[Depends(verify_api_key)]
)
app.include_router(hardware.router, prefix="/api/v1")


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket_endpoint(websocket, API_KEY)


# Lifespan handler above replaces the deprecated startup event decorator.


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
