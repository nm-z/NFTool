"""Dataset router endpoints for listing, uploading, and previewing datasets."""

import logging
import os
import tempfile
import time
import zipfile
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from src.auth import verify_api_key
from src.config import DATASETS_DIR, REPORTS_DIR, REPO_ROOT
from src.data.processing import load_dataset

router = APIRouter()

__all__ = ["list_datasets", "preview_dataset", "router"]

# Avoid calling Depends(...) in function defaults to satisfy the linter.
# Create a module-level dependency object and use an "unused" parameter name.
VERIFY_API_KEY_DEP = Depends(verify_api_key)
logger = logging.getLogger("nftool.datasets")
ALLOWED_ROOTS = [
    os.path.join(REPO_ROOT, "data"),
    DATASETS_DIR,
    REPORTS_DIR,
]
DATASET_FOLDER_DEFAULT = Form(...)
PREDICTOR_FILES_DEFAULT = File(None)
TARGET_FILES_DEFAULT = File(None)

# Cache for asset tree to avoid repeated filesystem traversal
_asset_tree_cache: dict[str, Any] = {}
_asset_tree_cache_time: float = 0.0
_ASSET_TREE_CACHE_TTL: float = 5.0  # Cache TTL in seconds


def _ensure_list_like(value):
    """
    Normalize a pandas Series/ndarray or scalar into a plain Python list.
    Avoid try/except per workspace rules; use hasattr checks.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    # Fallback for scalar values (float/int)
    return [value]


def _is_safe_subpath(root: str, target: str) -> bool:
    root_norm = os.path.normpath(root)
    try:
        return os.path.commonpath([root_norm, target]) == root_norm
    except ValueError:
        return False


def _resolve_allowed_path(path: str) -> str:
    if os.path.isabs(path):
        target = os.path.normpath(path)
    else:
        target = os.path.normpath(os.path.join(REPO_ROOT, path))
    for root in ALLOWED_ROOTS:
        if _is_safe_subpath(root, target):
            return target
    raise HTTPException(status_code=400, detail="Invalid path")


def _build_tree(path: str, file_filter: Any | None = None) -> dict[str, Any]:
    name = os.path.basename(path.rstrip(os.path.sep)) or path
    if os.path.isdir(path):
        children: list[dict[str, Any]] = []
        for entry in sorted(os.listdir(path)):
            full = os.path.join(path, entry)
            if os.path.isdir(full) or not file_filter or file_filter(full):
                children.append(_build_tree(full, file_filter))
        return {"name": name, "type": "folder", "path": path, "children": children}
    size = os.path.getsize(path) if os.path.exists(path) else 0
    return {"name": name, "type": "file", "path": path, "size": size}


def _sanitize_folder_name(folder_name: str) -> str:
    cleaned = folder_name.strip()
    if not cleaned or cleaned in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid folder name")
    if os.path.sep in cleaned or (os.path.altsep and os.path.altsep in cleaned):
        raise HTTPException(status_code=400, detail="Invalid folder name")
    return cleaned


@router.get(
    "/datasets",
    responses={
        200: {"description": "List of datasets"},
        406: {"description": "Missing or invalid API key"},
    },
)
def list_datasets(_api_key: str = VERIFY_API_KEY_DEP):
    """Return a list of CSV datasets under the repository `data/` directory."""
    dataset_dir = os.path.join(REPO_ROOT, "data")
    if not os.path.exists(dataset_dir):
        return []
    return [
        {"name": f, "path": os.path.join("data", f)}
        for f in sorted([f for f in os.listdir(dataset_dir) if f.endswith(".csv")])
    ]


def _build_asset_tree_uncached() -> dict[str, Any]:
    """Build the asset tree without caching (internal helper)."""
    builtin_data_dir = os.path.join(REPO_ROOT, "data")
    hold2_files: list[dict[str, Any]] = []
    if os.path.exists(builtin_data_dir):
        for entry in sorted(os.listdir(builtin_data_dir)):
            full = os.path.join(builtin_data_dir, entry)
            if os.path.isfile(full) and entry.endswith(".csv"):
                hold2_files.append(_build_tree(full))

    datasets_root: list[dict[str, Any]] = []
    if hold2_files:
        datasets_root.append(
            {
                "name": "Hold-2",
                "type": "folder",
                "path": builtin_data_dir,
                "children": hold2_files,
            }
        )

    if os.path.exists(DATASETS_DIR):
        for entry in sorted(os.listdir(DATASETS_DIR)):
            full = os.path.join(DATASETS_DIR, entry)
            if os.path.isdir(full):
                datasets_root.append(_build_tree(full))

    models_root: list[dict[str, Any]] = []
    if os.path.exists(REPORTS_DIR):
        for entry in sorted(os.listdir(REPORTS_DIR)):
            full = os.path.join(REPORTS_DIR, entry)
            if os.path.isdir(full):
                models_root.append(
                    _build_tree(full, file_filter=lambda p: p.endswith(".pt"))
                )

    return {
        "roots": [
            {
                "name": "Datasets",
                "type": "folder",
                "path": DATASETS_DIR,
                "children": datasets_root,
            },
            {
                "name": "Models",
                "type": "folder",
                "path": REPORTS_DIR,
                "children": models_root,
            },
        ]
    }


@router.get(
    "/assets/tree",
    responses={
        200: {"description": "Dataset + model asset tree"},
        406: {"description": "Missing or invalid API key"},
    },
)
def list_asset_tree(_api_key: str = VERIFY_API_KEY_DEP):
    """Return a tree of datasets and model checkpoints for the UI explorer.
    
    Uses a short TTL cache to avoid repeated filesystem traversals on frequent requests.
    """
    global _asset_tree_cache, _asset_tree_cache_time  # noqa: PLW0603
    
    current_time = time.time()
    if _asset_tree_cache and (current_time - _asset_tree_cache_time) < _ASSET_TREE_CACHE_TTL:
        return _asset_tree_cache
    
    result = _build_asset_tree_uncached()
    _asset_tree_cache = result
    _asset_tree_cache_time = current_time
    return result


@router.post(
    "/datasets/upload",
    responses={
        200: {"description": "Datasets uploaded"},
        400: {"description": "Invalid upload"},
        406: {"description": "Missing or invalid API key"},
    },
)
async def upload_datasets(
    folder_name: str = DATASET_FOLDER_DEFAULT,
    predictor_files: list[UploadFile] | None = PREDICTOR_FILES_DEFAULT,
    target_files: list[UploadFile] | None = TARGET_FILES_DEFAULT,
    _api_key: str = VERIFY_API_KEY_DEP,
):
    """Upload predictor/target dataset CSVs into a named dataset folder."""
    safe_folder = _sanitize_folder_name(folder_name)
    predictor_files = predictor_files or []
    target_files = target_files or []
    if not predictor_files and not target_files:
        raise HTTPException(status_code=400, detail="No files provided")

    target_root = os.path.join(DATASETS_DIR, safe_folder)
    os.makedirs(target_root, exist_ok=True)

    async def _save_files(files: list[UploadFile], dest_dir: str) -> int:
        os.makedirs(dest_dir, exist_ok=True)
        saved = 0
        for upload in files:
            filename = os.path.basename(upload.filename or "")
            if not filename:
                raise HTTPException(status_code=400, detail="Invalid filename")
            dest_path = os.path.join(dest_dir, filename)
            with open(dest_path, "wb") as buffer:
                buffer.write(await upload.read())
            saved += 1
        return saved

    saved_total = 0
    if len(predictor_files) == 1 and len(target_files) == 1:
        saved_total += await _save_files(predictor_files, target_root)
        saved_total += await _save_files(target_files, target_root)
    else:
        pred_dir = (
            os.path.join(target_root, "Predictors")
            if len(predictor_files) > 2
            else target_root
        )
        targ_dir = (
            os.path.join(target_root, "Targets")
            if len(target_files) > 2
            else target_root
        )
        saved_total += await _save_files(predictor_files, pred_dir)
        saved_total += await _save_files(target_files, targ_dir)

    return {
        "message": "Upload complete",
        "folder": safe_folder,
        "files_saved": saved_total,
        "path": target_root,
    }


@router.get(
    "/dataset/preview",
    responses={
        200: {"description": "Dataset preview"},
        404: {"description": "File not found"},
        406: {"description": "Missing or invalid API key"},
        422: {"description": "Validation error"},
    },
)
async def preview_dataset(
    path: str, rows: int = 10, _api_key: str = VERIFY_API_KEY_DEP
):
    """Return a small preview and basic statistics for the dataset at `path`."""
    # Resolve relative paths against the repository root so clients can pass
    # paths like "data/myfile.csv". If an absolute path is provided, use it.
    if os.path.isabs(path):
        target = os.path.normpath(path)
    else:
        target = os.path.normpath(os.path.join(REPO_ROOT, path))

    # Prevent directory traversal by ensuring the resolved path is under REPO_ROOT
    common = os.path.commonpath([REPO_ROOT, target])
    if common != os.path.normpath(REPO_ROOT):
        logger.warning(
            "Invalid dataset preview path attempt: %s (resolved: %s)",
            path,
            target,
        )
        raise HTTPException(status_code=400, detail="Invalid path")

    if not os.path.exists(target) or not os.path.isfile(target):
        logger.warning("Dataset preview file not found: %s", target)
        raise HTTPException(status_code=404, detail="File not found")

    df = load_dataset(target)
    preview = df.head(rows)

    stats = {
        "count": len(df),
        "columns": df.shape[1],
        "mean": _ensure_list_like(df.mean()),
        "std": _ensure_list_like(df.std()),
        "min": _ensure_list_like(df.min()),
        "max": _ensure_list_like(df.max()),
        # isnull().sum().sum() can be numpy/int-like; ensure int for JSON serialization
        "missing": int(df.isnull().sum().sum()),
    }

    return {
        "headers": [f"Feature_{i}" for i in range(df.shape[1])],
        "rows": preview.values.tolist(),
        "shape": list(df.shape),
        "total_rows": len(df),
        "stats": stats,
    }


@router.get(
    "/assets/download",
    responses={
        200: {"description": "Download dataset/model asset"},
        400: {"description": "Invalid path"},
        404: {"description": "Asset not found"},
        406: {"description": "Missing or invalid API key"},
    },
)
def download_asset(path: str, _api_key: str = VERIFY_API_KEY_DEP):
    """Download a file or zipped folder from datasets/models."""
    target = _resolve_allowed_path(path)
    if not os.path.exists(target):
        raise HTTPException(status_code=404, detail="Asset not found")

    if os.path.isfile(target):
        return FileResponse(path=target, filename=os.path.basename(target))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
    tmp_path = tmp.name
    tmp.close()
    base_name = os.path.basename(target.rstrip(os.path.sep)) or "download"
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _dirs, files in os.walk(target):
            rel_root = os.path.relpath(root, target)
            for fname in files:
                full = os.path.join(root, fname)
                arc = os.path.join(base_name, rel_root, fname)
                zipf.write(full, arc)

    return FileResponse(
        path=tmp_path,
        filename=f"{base_name}.zip",
        media_type="application/zip",
        background=BackgroundTask(os.unlink, tmp_path),
    )
