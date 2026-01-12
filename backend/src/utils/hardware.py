import subprocess
import json
import re
import psutil
import shutil
import logging
from typing import Dict, Any

logger = logging.getLogger("nftool")

class HardwareMonitor:
    def __init__(self):
        self.has_rocm = shutil.which("rocm-smi") is not None
        self.has_nvidia = shutil.which("nvidia-smi") is not None

    def get_gpu_stats(self, gpu_id: int = 0) -> Dict[str, Any]:
        if self.has_rocm:
            return self._get_rocm_stats(gpu_id)
        elif self.has_nvidia:
            return self._get_nvidia_stats(gpu_id)
        return self._get_empty_stats()

    def _get_rocm_stats(self, gpu_id: int) -> Dict[str, Any]:
        try:
            def parse_rocm_json(output):
                try:
                    start = output.find('{')
                    end = output.rfind('}')
                    if start != -1 and end != -1:
                        return json.loads(output[start:end+1])
                except:
                    pass
                return None

            res_use = subprocess.run(
                ["rocm-smi", "--showuse", "--showtemp", "--json"],
                capture_output=True, text=True, timeout=2
            )
            res_mem = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                capture_output=True, text=True, timeout=2
            )
            
            gpu_stats = self._get_empty_stats()
            target_keys = [f"card{gpu_id}", f"device{gpu_id}", str(gpu_id)]

            if res_use.returncode == 0:
                data = parse_rocm_json(res_use.stdout)
                if data:
                    card = next((k for k in target_keys if k in data), None)
                    if not card and len(data) > 0: card = list(data.keys())[0]
                    if card:
                        use_val = data[card].get("GPU use (%)") or data[card].get("GPU use") or data[card].get("GPU utilization (%)") or data[card].get("GPU Utilization") or 0
                        gpu_stats["gpu_use_percent"] = int(float(str(use_val).strip('%')))
                        temp = data[card].get("Temperature (Sensor edge) (C)") or data[card].get("Temperature (C)") or data[card].get("Temperature (Sensor junction) (C)") or 0
                        gpu_stats["gpu_temp_c"] = int(float(temp))

            if res_mem.returncode == 0:
                data = parse_rocm_json(res_mem.stdout)
                if data:
                    card = next((k for k in target_keys if k in data), None)
                    if not card and len(data) > 0: card = list(data.keys())[0]
                    if card:
                        total_b = int(data[card].get("VRAM Total Memory (B)", 0))
                        used_b = int(data[card].get("VRAM Total Used Memory (B)", 0))
                        if total_b == 0:
                            total_b = int(data[card].get("VRAM Total Memory (MiB)", 0)) * 1024 * 1024
                            used_b = int(data[card].get("VRAM Total Used Memory (MiB)", 0)) * 1024 * 1024
                        gpu_stats["vram_total_gb"] = round(total_b / (1024**3), 2)
                        gpu_stats["vram_used_gb"] = round(used_b / (1024**3), 2)
                        gpu_stats["vram_percent"] = int((used_b / total_b * 100)) if total_b > 0 else 0
            return gpu_stats
        except Exception as e:
            logger.error(f"ROCm SMI Error: {e}")
            return self._get_empty_stats()

    def _get_nvidia_stats(self, gpu_id: int) -> Dict[str, Any]:
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.total,memory.used", "--format=csv,noheader,nounits", f"--id={gpu_id}"],
                capture_output=True, text=True, timeout=2
            )
            if res.returncode == 0:
                use, temp, total_mb, used_mb = res.stdout.strip().split(", ")
                total_gb = round(float(total_mb) / 1024, 2)
                used_gb = round(float(used_mb) / 1024, 2)
                return {
                    "vram_total_gb": total_gb,
                    "vram_used_gb": used_gb,
                    "vram_percent": int((float(used_mb) / float(total_mb)) * 100),
                    "gpu_use_percent": int(use),
                    "gpu_temp_c": int(temp)
                }
        except Exception as e:
            logger.error(f"NVIDIA SMI Error: {e}")
        return self._get_empty_stats()

    def _get_empty_stats(self) -> Dict[str, Any]:
        return {
            "vram_total_gb": 0.0, "vram_used_gb": 0.0, "vram_percent": 0,
            "gpu_use_percent": 0, "gpu_temp_c": 0
        }

    def get_system_stats(self) -> Dict[str, Any]:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        return {
            "cpu_percent": cpu_percent,
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_used_gb": round(mem.used / (1024**3), 2),
            "ram_percent": mem.percent
        }
