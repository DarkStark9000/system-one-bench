"""What this GPU can do at best, measured on it: the yardstick for Laya's rates.

A vendor's peak is not always published (NVIDIA gives no tensor TFLOPS for its
laptop GPUs), and a laptop's clocks depend on its power mode. So each run
measures two ceilings on the machine itself, under the same settings:

- matmul: a large square matrix multiply in BF16 and FP16, the best case for
  tensor cores. Rate = 2 n^3 / time.
- copy: a large device-to-device copy, the best case for memory bandwidth.
  Rate = 2 x bytes / time (each byte is read once and written once).

Each is the median of 50 timed runs after 10 warm-up runs, timed with CUDA events.
"""

import statistics


def _timed_ms(fn, warmup: int = 10, runs: int = 50) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def measure() -> dict[str, float]:
    """Measured ceilings of the first CUDA GPU, or an empty dict without one."""
    import torch

    if not torch.cuda.is_available():
        return {}
    out: dict[str, float] = {}
    n = 8192
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        a = torch.randn(n, n, device="cuda", dtype=dtype)
        b = torch.randn(n, n, device="cuda", dtype=dtype)
        ms = _timed_ms(lambda: a @ b)
        out[f"matmul_{name}_tflops"] = 2 * n**3 / (ms / 1000) / 1e12
        del a, b
    size = 1 << 30  # 1 GiB
    src = torch.empty(size, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    ms = _timed_ms(lambda: dst.copy_(src))
    out["copy_gbps"] = 2 * size / (ms / 1000) / 1e9
    del src, dst
    torch.cuda.empty_cache()
    return {key: round(value, 2) for key, value in out.items()}
