import torch
from fastapi import APIRouter

router = APIRouter()

__all__ = ["router", "list_gpus", "health_check", "get_hardware_capabilities"]


@router.get("/gpus")
def list_gpus():
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
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


@router.get("/capabilities")
def get_hardware_capabilities():
    if torch.cuda.is_available():
        device_id = 0  # Assuming single GPU for simplicity, or can iterate
        properties = torch.cuda.get_device_properties(device_id)
        return {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(device_id),
            "total_memory_gb": round(properties.total_memory / (1024**3), 2),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "supports_fp16": properties.major
            >= 7,  # Generally true for Pascal (sm_60) and later
            "supports_bf16": properties.major
            >= 8,  # Generally true for Ampere (sm_80) and later
            "driver_version": getattr(torch, "__version__", None),
        }
    else:
        return {
            "cuda_available": False,
            "device_name": "CPU",
            "total_memory_gb": 0,
            "compute_capability": "N/A",
            "supports_fp16": False,
            "supports_bf16": False,
            "driver_version": "N/A",
        }
