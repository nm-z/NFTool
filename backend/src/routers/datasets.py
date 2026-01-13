"""Dataset router endpoints for listing and previewing CSV datasets."""

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from src.auth import verify_api_key
from src.config import REPO_ROOT
from src.data.processing import load_dataset

router = APIRouter()

__all__ = ["list_datasets", "preview_dataset", "router"]

# Avoid calling Depends(...) in function defaults to satisfy the linter.
# Create a module-level dependency object and use an "unused" parameter name.
VERIFY_API_KEY_DEP = Depends(verify_api_key)
logger = logging.getLogger("nftool.datasets")


def _ensure_list_like(value):
    """
    Normalize a pandas Series/ndarray or scalar into a plain Python list.
    Avoid try/except per workspace rules; use hasattr checks.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    # Fallback for scalar values (float/int)
    return [value]


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
