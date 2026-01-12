import os

from fastapi import APIRouter, HTTPException
from src.config import REPO_ROOT
from src.data.processing import load_dataset

router = APIRouter()


def _ensure_list_like(value):
    """
    Normalize a pandas Series/ndarray or scalar into a plain Python list.
    Avoid try/except per workspace rules; use hasattr checks.
    """
    if hasattr(value, "tolist"):
        return value.tolist()
    # Fallback for scalar values (float/int)
    return [value]


@router.get("/datasets")
def list_datasets():
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
async def preview_dataset(path: str, rows: int = 10):
    target = path
    if not os.path.exists(target):
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
