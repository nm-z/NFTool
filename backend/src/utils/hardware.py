import json
import logging
import shutil
import subprocess
from typing import Any

import psutil

logger = logging.getLogger("nftool")


class HardwareMonitor:
    def __init__(self):
        self.has_rocm = shutil.which("rocm-smi") is not None
        self.has_nvidia = shutil.which("nvidia-smi") is not None

    def get_gpu_stats(self, gpu_id: int = 0) -> dict[str, Any]:
        # Attempt to query vendor tooling. If neither ROCm nor NVIDIA tooling is
        # present, fail fast — do not return fabricated defaults.
        if self.has_rocm:
            return self._get_rocm_stats(gpu_id)
        if self.has_nvidia:
            return self._get_nvidia_stats(gpu_id)
        raise RuntimeError(
            "No GPU management tool found (rocm-smi or nvidia-smi). Hardware probing is required."
        )

    def _get_rocm_stats(self, gpu_id: int) -> dict[str, Any]:
        def parse_rocm_json(output):
            start = output.find("{")
            end = output.rfind("}")
            if start != -1 and end != -1:
                return json.loads(output[start : end + 1])
            return None

        res_use = subprocess.run(
            ["rocm-smi", "--showuse", "--showtemp", "--json"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        res_mem = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )

        gpu_stats = self._get_empty_stats()
        target_keys = [f"card{gpu_id}", f"device{gpu_id}", str(gpu_id)]

        data_use = parse_rocm_json(res_use.stdout)
        if data_use:
            card = next((k for k in target_keys if k in data_use), None)
            if not card and len(data_use) > 0:
                card = list(data_use.keys())[0]
            if card:
                use_val = (
                    data_use[card].get("GPU use (%)")
                    or data_use[card].get("GPU use")
                    or data_use[card].get("GPU utilization (%)")
                    or data_use[card].get("GPU Utilization")
                    or 0
                )
                gpu_stats["gpu_use_percent"] = int(float(str(use_val).strip("%")))
                temp = (
                    data_use[card].get("Temperature (Sensor edge) (C)")
                    or data_use[card].get("Temperature (C)")
                    or data_use[card].get("Temperature (Sensor junction) (C)")
                    or 0
                )
                gpu_stats["gpu_temp_c"] = int(float(temp))

        data_mem = parse_rocm_json(res_mem.stdout)
        if data_mem:
            card = next((k for k in target_keys if k in data_mem), None)
            if not card and len(data_mem) > 0:
                card = list(data_mem.keys())[0]
            if card:
                total_b = int(data_mem[card].get("VRAM Total Memory (B)", 0))
                used_b = int(data_mem[card].get("VRAM Total Used Memory (B)", 0))
                if total_b == 0:
                    total_b = (
                        int(data_mem[card].get("VRAM Total Memory (MiB)", 0))
                        * 1024
                        * 1024
                    )
                    used_b = (
                        int(data_mem[card].get("VRAM Total Used Memory (MiB)", 0))
                        * 1024
                        * 1024
                    )
                gpu_stats["vram_total_gb"] = round(total_b / (1024**3), 2)
                gpu_stats["vram_used_gb"] = round(used_b / (1024**3), 2)
                gpu_stats["vram_percent"] = (
                    int((used_b / total_b * 100)) if total_b > 0 else 0
                )
        return gpu_stats

    def _get_nvidia_stats(self, gpu_id: int) -> dict[str, Any]:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,memory.total,memory.used",
                "--format=csv,noheader,nounits",
                f"--id={str(gpu_id)}",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        use, temp, total_mb, used_mb = res.stdout.strip().split(", ")
        total_gb = round(float(total_mb) / 1024, 2)
        used_gb = round(float(used_mb) / 1024, 2)
        return {
            "vram_total_gb": total_gb,
            "vram_used_gb": used_gb,
            "vram_percent": int((float(used_mb) / float(total_mb)) * 100),
            "gpu_use_percent": int(use),
            "gpu_temp_c": int(temp),
        }

    def _get_empty_stats(self) -> dict[str, Any]:
        return {
            "vram_total_gb": 0.0,
            "vram_used_gb": 0.0,
            "vram_percent": 0,
            "gpu_use_percent": 0,
            "gpu_temp_c": 0,
        }

    def get_system_stats(self) -> dict[str, Any]:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        return {
            "cpu_percent": cpu_percent,
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "ram_percent": mem.percent,
        }
