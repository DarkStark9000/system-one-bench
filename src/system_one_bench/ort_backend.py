"""ONNX Runtime backends: one exported graph, run by a chosen execution provider.

laya's `ONNXAgent` picks CUDA or the CPU by itself. The bench needs every
provider (TensorRT, DirectML, Core ML, Vitis AI for the Ryzen AI NPU), so it
builds the session here and hands it to `ONNXAgent`, which keeps doing the
tokenizing and decoding. A session whose first provider is not the one asked
for is refused: a silent fallback to the CPU would time the wrong device.

ONNX Runtime runs the encoder and the head as one graph, and copies inputs and
outputs inside `session.run`, so its phase split has one `model` phase where
the PyTorch split has five.
"""

import time
from pathlib import Path

import numpy as np
from laya.common import _disable_question_token_reuse, _reuse_question_tokens

ORT_PHASES = ("prepare", "tokenize", "collate", "model", "decode")

# One TensorRT optimization profile covers every workload in inputs.py: up to
# 64 rows of up to 512 tokens and 16 options. Without it TensorRT rebuilds its
# engine for each new shape.
_TRT_PROFILE = {
    "min": "input_ids:1x8,attention_mask:1x8,marker_pos:1x2,marker_mask:1x2,qtype:1",
    "opt": "input_ids:8x192,attention_mask:8x192,marker_pos:8x3,marker_mask:8x3,qtype:8",
    "max": "input_ids:64x512,attention_mask:64x512,marker_pos:64x16,marker_mask:64x16,qtype:64",
}


def providers_for(name: str, cache_dir: Path) -> list:
    if name == "ort-cpu":
        return ["CPUExecutionProvider"]
    if name == "ort-cuda":
        return [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
    if name == "ort-trt-fp16":
        cache_dir.mkdir(parents=True, exist_ok=True)
        options = {
            "trt_fp16_enable": True,
            # LayerNorm in FP16 overflows on long rows; the independent capacity study
            # cited in laya's BENCHMARKS.md kept it in FP32 for the same reason.
            "trt_layer_norm_fp32_fallback": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_dir),
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": str(cache_dir),
            "trt_profile_min_shapes": _TRT_PROFILE["min"],
            "trt_profile_opt_shapes": _TRT_PROFILE["opt"],
            "trt_profile_max_shapes": _TRT_PROFILE["max"],
        }
        return [("TensorrtExecutionProvider", options), ("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
    if name == "ort-dml":
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if name == "ort-coreml":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if name == "ort-vitisai":
        return ["VitisAIExecutionProvider", "CPUExecutionProvider"]
    raise SystemExit(f"unknown ONNX Runtime backend {name!r}")


def _expose_tensorrt() -> None:
    """Put the pip TensorRT libraries where ONNX Runtime's TensorRT provider looks for them."""
    import os

    try:
        import tensorrt_libs
    except ImportError:
        raise SystemExit("ort-trt-fp16 needs TensorRT: install the tensorrt extra") from None
    folder = os.path.dirname(tensorrt_libs.__file__)
    os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(folder)


def load_onnx_agent(model_dir: str, onnx_path: Path, backend: str, cache_dir: Path):
    import onnxruntime as ort
    from laya.onnx_agent import ONNXAgent

    if hasattr(ort, "preload_dlls"):
        # Finds the CUDA and cuDNN libraries that the PyTorch wheel installed.
        ort.preload_dlls()
    if backend == "ort-trt-fp16":
        _expose_tensorrt()
    agent = ONNXAgent(model_dir, onnx_path=str(onnx_path))
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = providers_for(backend, cache_dir)
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)
    wanted = providers[0] if isinstance(providers[0], str) else providers[0][0]
    if session.get_providers()[0] != wanted:
        raise SystemExit(f"{backend}: ONNX Runtime fell back to {session.get_providers()[0]}; refusing to time it")
    agent.session = session
    return agent


@_reuse_question_tokens
def staged_ort_predict(agent, states: list, questions: dict) -> tuple[list[dict], dict[str, float]]:
    from laya.agent import Agent
    from laya.onnx_agent import _collate_items_numpy

    ms: dict[str, float] = {}
    last = time.perf_counter()

    def mark(phase: str) -> None:
        nonlocal last
        now = time.perf_counter()
        ms[phase] = (now - last) * 1000.0
        last = now

    if len(states) == 1:
        _disable_question_token_reuse()
    ids = list(questions)
    for qid in ids:
        Agent._check_question(qid, questions[qid])
    internal = {qid: agent._to_internal(questions[qid]) for qid in ids}
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    mark("prepare")

    encoded = [agent._encode_state(state, ids, internal, max_len, head_max_len) for state in states]
    mark("tokenize")

    feed = _collate_items_numpy(encoded, agent.tok.pad_token_id)
    mark("collate")

    logits, act_logits = agent.session.run(["logits", "act_logits"], feed)
    mark("model")

    act = np.exp(act_logits - act_logits.max(axis=-1, keepdims=True))
    act = act / act.sum(axis=-1, keepdims=True)
    results, row = [], 0
    for items in encoded:
        results.append({"answers": agent._decode_answers(logits, act, items, ids, internal, row)})
        row += len(items)
    mark("decode")
    return results, ms
