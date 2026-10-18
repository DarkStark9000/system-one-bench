"""The GPU's temperature and clock, and a rest before each backend.

A laptop slows its clocks as it heats. Run after an hour of CPU load, the
same GPU call took nearly twice as long as on a cool machine. So each backend
starts from rest, and each row records the GPU's state at its start and end:
a reader can see when a number came from a hot machine.
"""

import subprocess
import time

COOL_C = 65
MIN_REST_S = 30.0
MAX_REST_S = 180.0


def _query(fields: str) -> list[float]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return [float(part) for part in out.stdout.splitlines()[0].split(",")]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return [float("nan")] * len(fields.split(","))


def gpu_state() -> tuple[float, float]:
    """(temperature in C, SM clock in MHz) of the first NVIDIA GPU, or NaNs when there is none."""
    temperature, clock = _query("temperature.gpu,clocks.sm")
    return temperature, clock


def gpu_memory_used_mib() -> float:
    """GPU memory in use, all processes. Near the card's total, Windows may spill to system RAM and slow every call."""
    return _query("memory.used")[0]


def rest() -> float:
    """Idle for at least MIN_REST_S, then until the GPU is at or below COOL_C, at most MAX_REST_S. Returns seconds."""
    start = time.monotonic()
    time.sleep(MIN_REST_S)
    while time.monotonic() - start < MAX_REST_S:
        temperature, _ = gpu_state()
        if temperature != temperature or temperature <= COOL_C:  # NaN: no NVIDIA GPU to wait for
            break
        time.sleep(5)
    return time.monotonic() - start
