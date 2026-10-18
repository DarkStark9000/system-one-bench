"""How much of a call the device spends computing, from the PyTorch profiler.

Wall time says how long the caller waits. The profiler adds two facts the clock
cannot give: how many kernels one call launches, and how long those kernels run
in total. When the kernels run for much less time than the call takes, the
device waits for the host, and the call is bound by launches, not by compute.
"""

import torch
from torch.profiler import ProfilerActivity, profile

from system_one_bench.staged import device_sync


def kernel_profile(agent, states: list, questions: dict, calls: int = 10) -> dict[str, float]:
    """Kernels per call, device time per call and host wall time per call, averaged over `calls`."""
    if agent.device.type != "cuda":
        return {}
    sync = device_sync(agent.device)
    agent.predict_batch(states, questions)
    sync()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(calls):
            agent.predict_batch(states, questions)
        sync()
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    kernel_us = sum(e.device_time for e in kernels)
    copies = [e for e in kernels if "memcpy" in e.name.lower() or "memset" in e.name.lower()]
    return {
        "kernels_per_call": len(kernels) / calls,
        "copies_per_call": len(copies) / calls,
        "device_ms_per_call": kernel_us / 1000.0 / calls,
    }
