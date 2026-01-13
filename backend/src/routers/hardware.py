"""Hardware router exposing endpoints for system/GPU capabilities.

This module provides lightweight endpoints used by the API to inspect
CUDA availability and device properties.
"""

from typing import Any, Protocol, cast

import torch
from fastapi import APIRouter


class CudaProps(Protocol):
    """Protocol defining the expected fields from CUDA device properties."""
    name: str
    major: int
    minor: int
    total_memory: int
    multi_processor_count: int


def get_cuda_info(device_id: int) -> dict[str, Any]:
    """Retrieve and format properties for a specific CUDA device."""
    if not torch.cuda.is_available():
        return {}

    # TYPE AIRLOCK:
    # 1. Cast module to Any to silence Pyright's 'Unknown Member' error.
    # 2. Call directly to silence Flake8's B009 'getattr' error.
    cuda_mod: Any = torch.cuda
    props = cast(CudaProps, cuda_mod.get_device_properties(device_id))

    return {
        "name": props.name,
        "vram": props.total_memory,
        "compute": f"{props.major}.{props.minor}",
        "cores": props.multi_processor_count,
    }


router = APIRouter()

__all__ = ["get_hardware_capabilities", "health_check", "list_gpus", "router"]


@router.get("/gpus")
def list_gpus():
    """Return a list of available CUDA GPUs.

    Each item contains `id`, `name`, and `is_available`. Returns an empty
    list when CUDA is not available.
    """
    return (
        [
            {"id": i, "name": torch.cuda.get_device_name(i), "is_available": True}
            for i in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else []
    )


@router.get("/health")
def health_check():
    """Return a simple health status for the service.

    Includes overall status, whether a GPU is available, and the active
    device name.
    """
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


@router.get("/capabilities")
def get_hardware_capabilities() -> dict[str, Any]:
    """Return detailed hardware capabilities when CUDA is available.

    For CUDA devices this includes device name, total memory (GB),
    compute capability, FP16/BF16 support and driver version. For CPU-only
    systems returns defaults indicating CUDA is unavailable.
    """
    if torch.cuda.is_available():
        device_id = 0
        info = get_cuda_info(device_id)

        # Safe access to version string to satisfy Pylint
        # torch.version is a module, so .get() fails. We use getattr instead.
        v_mod = getattr(torch, "version", None)
        cuda_version = getattr(v_mod, "cuda", "N/A") if v_mod is not None else "N/A"

        return {
            "cuda_available": True,
            "device_name": info["name"],
            "total_memory_gb": round(info["vram"] / (1024**3), 2),
            "compute_capability": info["compute"],
            "cores": info["cores"],
            "supports_fp16": int(info["compute"].split(".")[0]) >= 7,
            "supports_bf16": int(info["compute"].split(".")[0]) >= 8,
            "driver_version": cuda_version,
        }
    else:
        return {
            "cuda_available": False,
            "device_name": "CPU",
            "total_memory_gb": 0,
            "compute_capability": "N/A",
            "cores": 0,
            "supports_fp16": False,
            "supports_bf16": False,
            "driver_version": "N/A",
        }
