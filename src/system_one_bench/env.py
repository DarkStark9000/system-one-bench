"""The machine fingerprint committed with every result.

It records what changes a timing: the processor, the accelerator and its
driver, the memory, the power source, the operating system, and the version
of every package on the measured path. It records no user name, host name,
serial number or network address.
"""

import importlib.metadata
import os
import platform
import subprocess
import sys

import psutil

PACKAGES = (
    "laya",
    "torch",
    "transformers",
    "tokenizers",
    "safetensors",
    "huggingface_hub",
    "numpy",
    "onnx",
    "onnxruntime",
    "onnxruntime-gpu",
    "onnxruntime-directml",
    "onnxruntime-vitisai",
    "tensorrt",
    "tensorrt-cu12",
    "coremltools",
    "system-one-bench",
)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=20, check=True,
            )
            return out.stdout.strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            return platform.processor()
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        return out.stdout.strip()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _nvidia() -> dict[str, str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    name, driver, memory, power = (part.strip() for part in out.stdout.splitlines()[0].split(","))
    return {"gpu": name, "gpu_driver": driver, "gpu_memory": memory, "gpu_power_limit": power}


def _power_source() -> str:
    battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    if battery is None:
        return "mains (no battery)"
    return "mains" if battery.power_plugged else "battery"


def fingerprint() -> dict[str, str]:
    import torch

    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "cpu_physical_cores": str(psutil.cpu_count(logical=False)),
        "cpu_logical_cores": str(psutil.cpu_count(logical=True)),
        "memory_gib": f"{psutil.virtual_memory().total / 2**30:.1f}",
        "power_source": _power_source(),
        "torch_threads": str(torch.get_num_threads()),
        "torch_interop_threads": str(torch.get_num_interop_threads()),
        "cuda_runtime": torch.version.cuda or "",
        "mps_available": str(bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
    }
    info.update(_nvidia())
    for package in PACKAGES:
        version = _version(package)
        if version:
            info[f"pkg_{package}"] = version
    return info
