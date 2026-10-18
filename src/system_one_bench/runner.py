"""Run one sweep row: load a backend, time it, split it, profile it and check its answers."""

import json
import math
import statistics
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from system_one_bench import flops, inputs, stats
from system_one_bench.models import Checkpoint

TORCH_BACKENDS = (
    "torch-cpu",
    "torch-cuda-bf16",
    "torch-cuda-fp16",
    "torch-cuda-fp32",
    "torch-cuda-graphs",
    "torch-cuda-tilelang",
    "torch-mps",
)
ORT_BACKENDS = ("ort-cpu", "ort-cuda", "ort-trt-fp16", "ort-dml", "ort-coreml", "ort-vitisai")
BACKENDS = TORCH_BACKENDS + ORT_BACKENDS


@dataclass
class Settings:
    warmup: int = 5
    iterations: int = 30
    repeats: int = 5
    min_iterations: int = 10
    round_budget_s: float = 60.0
    cache: Path = Path(".cache")


@dataclass
class Row:
    checkpoint: str
    revision: str
    backend: str
    workload: str
    size: int
    status: str = "ok"
    error: str = ""
    precision: str = ""
    rows: int = 0
    tokens: int = 0
    calls: int = 0
    p50_ms: float = math.nan
    p99_ms: float = math.nan
    p50_low_ms: float = math.nan
    p50_high_ms: float = math.nan
    decisions_per_s: float = math.nan
    phases_ms: dict[str, float] = field(default_factory=dict)
    kernels_per_call: float = math.nan
    device_ms_per_call: float = math.nan
    gflop: float = math.nan
    tflops_wall: float = math.nan
    tflops_device: float = math.nan
    parity_max_prob_diff: float = math.nan
    parity_argmax: str = ""


def model_dir(ckpt: Checkpoint) -> str:
    """Download the pinned checkpoint, then refuse it unless its weights match the digest."""
    from huggingface_hub import snapshot_download
    from laya.revisions import verify_digests

    path = snapshot_download(
        ckpt.repo,
        revision=ckpt.revision,
        allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"],
    )
    verify_digests(path, {"model.safetensors": ckpt.weights_sha256})
    return path


def _onnx_path(ckpt: Checkpoint, directory: str, settings: Settings) -> Path:
    from system_one_bench import export

    path = settings.cache / "onnx" / f"{ckpt.name}-{ckpt.revision[:7]}.onnx"
    if path.exists():
        return path
    import laya

    agent = laya.load(directory, device="cpu")
    feeds = [_feed(agent, *inputs.workload(kind, size)) for kind, size in (("questions", 1), ("questions", 5), ("batch", 3))]
    export.export(agent, path)
    export.verify(agent, path, feeds)
    return path


def _feed(agent, states, questions):
    from laya.common import collate_items

    ids = list(questions)
    internal = {q: agent._to_internal(questions[q]) for q in ids}
    return collate_items([agent._encode_state(s, ids, internal) for s in states], agent.tok.pad_token_id)


def load(backend: str, ckpt: Checkpoint, directory: str, settings: Settings):
    """The agent for `backend`, and the precision it really runs in."""
    import laya
    import torch

    if backend in ORT_BACKENDS:
        from system_one_bench.ort_backend import load_onnx_agent

        onnx = _onnx_path(ckpt, directory, settings)
        agent = load_onnx_agent(directory, onnx, backend, settings.cache / "trt" / ckpt.name)
        return agent, ("fp16" if backend == "ort-trt-fp16" else "fp32")
    if backend == "torch-cpu":
        agent = laya.load(directory, device="cpu")
    elif backend == "torch-mps":
        agent = laya.load(directory, device="mps")
    elif backend == "torch-cuda-graphs":
        agent = laya.load(directory, device="cuda", compile=True, compile_mode="reduce-overhead")
    elif backend == "torch-cuda-tilelang":
        # Strict, so a failure raises TileLang's own reason instead of falling back to eager.
        agent = laya.load(directory, device="cuda")
        agent.set_backend("tilelang", strict=True)
    else:
        agent = laya.load(directory, device="cuda")
        if backend == "torch-cuda-fp16":
            agent.dtype = torch.float16
        elif backend == "torch-cuda-fp32":
            agent.amp_enabled = False
            agent.dtype = torch.float32
    precision = str(agent.dtype).removeprefix("torch.") if agent.amp_enabled else "float32"
    return agent, precision


def _answers(results: list[dict]) -> tuple[np.ndarray, list[str]]:
    probs, picks = [], []
    for result in results:
        for qid in sorted(result["answers"]):
            answer = result["answers"][qid]
            if answer["type"] == "noul":
                probs.append(answer["noul"])
                picks.append("true" if answer["noul"] >= 0.5 else "false")
            else:
                table = answer["probabilities"]
                probs.extend(table[key] for key in sorted(table))
                picks.append(max(table, key=table.get))
    return np.asarray(probs, dtype=np.float64), picks


def reference(ckpt: Checkpoint, directory: str, kind: str, size: int, settings: Settings) -> dict:
    """FP32 answers from PyTorch on the CPU, cached: the yardstick every backend is checked against."""
    path = settings.cache / "reference" / f"{ckpt.name}-{ckpt.revision[:7]}-{kind}-{size}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    import laya

    agent = laya.load(directory, device="cpu")
    states, questions = inputs.workload(kind, size)
    probs, picks = _answers(agent.predict_batch(states, questions))
    data = {"probabilities": probs.tolist(), "picks": picks}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _time_calls(call, settings: Settings) -> list[list[float]]:
    rounds = []
    for _ in range(settings.repeats):
        for _ in range(settings.warmup):
            call()
        start = time.perf_counter()
        call()
        first = time.perf_counter() - start
        count = max(settings.min_iterations, min(settings.iterations, int(settings.round_budget_s / max(first, 1e-6))))
        samples = [first * 1000.0]
        for _ in range(count - 1):
            start = time.perf_counter()
            call()
            samples.append((time.perf_counter() - start) * 1000.0)
        rounds.append(samples)
    return rounds


def measure(row: Row, agent, ckpt: Checkpoint, directory: str, settings: Settings) -> None:
    from system_one_bench.ort_backend import staged_ort_predict
    from system_one_bench.profile import kernel_profile
    from system_one_bench.staged import staged_predict

    states, questions = inputs.workload(row.workload, row.size)
    is_ort = row.backend in ORT_BACKENDS
    split = staged_ort_predict if is_ort else staged_predict

    rounds = _time_calls(lambda: agent.predict_batch(states, questions), settings)
    summary = stats.summarize(rounds)
    row.calls, row.p50_ms, row.p99_ms = summary.calls, summary.p50_ms, summary.p99_ms
    row.p50_low_ms, row.p50_high_ms = summary.p50_low_ms, summary.p50_high_ms

    samples: dict[str, list[float]] = {}
    staged_results = None
    for _ in range(max(settings.min_iterations, min(settings.iterations, len(rounds[0])))):
        staged_results, phases = split(agent, states, questions)
        for phase, ms in phases.items():
            samples.setdefault(phase, []).append(ms)
    row.phases_ms = {phase: statistics.median(values) for phase, values in samples.items()}

    ids = list(questions)
    internal = {q: agent._to_internal(questions[q]) for q in ids}
    from laya.common import _reuse_question_tokens

    budget = (agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192))
    encode = _reuse_question_tokens(lambda: [agent._encode_state(s, ids, internal, *budget) for s in states])
    lengths = [len(item["ids"]) for items in encode() for item in items]
    row.rows, row.tokens = len(lengths), sum(lengths)
    row.decisions_per_s = row.rows / (row.p50_ms / 1000.0)
    encoder_cfg, head_layers = flops.configs(directory)
    work = flops.batch_flops(encoder_cfg, head_layers, lengths)
    row.gflop = work / 1e9
    row.tflops_wall = work / (row.p50_ms / 1000.0) / 1e12

    if not is_ort and row.backend.startswith("torch-cuda"):
        profile = kernel_profile(agent, states, questions)
        row.kernels_per_call = profile["kernels_per_call"]
        row.device_ms_per_call = profile["device_ms_per_call"]
        row.tflops_device = work / (row.device_ms_per_call / 1000.0) / 1e12

    ref = reference(ckpt, directory, row.workload, row.size, settings)
    probs, picks = _answers(staged_results)
    row.parity_max_prob_diff = float(np.max(np.abs(probs - np.asarray(ref["probabilities"]))))
    agree = sum(a == b for a, b in zip(picks, ref["picks"]))
    row.parity_argmax = f"{agree}/{len(picks)}"


def run_backend(backend: str, ckpt: Checkpoint, sweeps: list[tuple[str, int]], settings: Settings) -> list[Row]:
    """Every sweep row for one backend and checkpoint. A failure becomes a row, never a gap."""
    directory = model_dir(ckpt)
    rows = [Row(ckpt.name, ckpt.revision, backend, kind, size) for kind, size in sweeps]
    try:
        agent, precision = load(backend, ckpt, directory, settings)
    except BaseException as exc:  # noqa: BLE001 - a backend that cannot load is a result to report
        if isinstance(exc, KeyboardInterrupt):
            raise
        for row in rows:
            row.status, row.error = "load failed", _reason(exc)
            print(_line(row), flush=True)
        return rows
    for row in rows:
        row.precision = precision
        try:
            measure(row, agent, ckpt, directory, settings)
        except Exception as exc:  # noqa: BLE001 - one failing shape must not hide the others
            row.status, row.error = "run failed", _reason(exc)
            traceback.print_exc()
        print(_line(row), flush=True)
    return rows


def _reason(exc: BaseException) -> str:
    text = str(exc)
    # A kernel compiler prints the generated source before its error; keep the error.
    marker = text.rfind("Compilation error")
    if marker != -1:
        text = text[marker:]
    text = " ".join(f"{type(exc).__name__}: {text}".split())
    return text[:300]


def _line(row: Row) -> str:
    if row.status != "ok":
        return f"{row.checkpoint:16} {row.backend:20} {row.workload:9} {row.size:3}  {row.status}: {row.error}"
    return (
        f"{row.checkpoint:16} {row.backend:20} {row.workload:9} {row.size:3}  "
        f"p50 {row.p50_ms:8.2f} ms  p99 {row.p99_ms:8.2f} ms  "
        f"{row.decisions_per_s:8.1f} dec/s  parity {row.parity_argmax} ({row.parity_max_prob_diff:.1e})"
    )


def as_record(row: Row) -> dict:
    record = asdict(row)
    phases = record.pop("phases_ms")
    for phase in ("prepare", "tokenize", "collate", "transfer_in", "dispatch", "encoder", "head", "transfer_out", "model", "decode"):
        record[f"ms_{phase}"] = phases.get(phase, math.nan)
    return record
