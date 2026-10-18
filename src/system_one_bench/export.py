"""Export a checkpoint to ONNX the way laya's own exporter does, then prove the graph.

This follows `scripts/export_onnx.py` in laya at tag v0.4.11 (Apache 2.0,
Convai Innovations): opset 18, the five sequential-layout inputs, one shared
symbolic batch axis, and a two-row trace example, because `torch.export`
bakes in any example dimension of size 1. The check afterwards runs the graph
in ONNX Runtime at batch 1, 2 and 3 with real token sequences, and compares
probabilities with the PyTorch model.
"""

from pathlib import Path

import numpy as np
import torch

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act_logits"]
DYNAMIC_AXES = {
    "input_ids": {0: "batch_size", 1: "seq_len"},
    "attention_mask": {0: "batch_size", 1: "seq_len"},
    "marker_pos": {0: "batch_size", 1: "num_markers"},
    "marker_mask": {0: "batch_size", 1: "num_markers"},
    "qtype": {0: "batch_size"},
    "logits": {0: "batch_size", 1: "num_markers"},
    "act_logits": {0: "batch_size"},
}
PARITY_ATOL = 1e-3


def _trace_inputs(batch: int = 2, seq_len: int = 24, markers: int = 3):
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 100, (batch, seq_len), generator=generator, dtype=torch.long)
    att = torch.ones((batch, seq_len), dtype=torch.long)
    pos = torch.arange(1, markers + 1, dtype=torch.long).repeat(batch, 1)
    mask = torch.ones((batch, markers), dtype=torch.bool)
    for row in range(1, batch):
        att[row, max(1, seq_len - row) :] = 0
        mask[row, -1] = False
    qtype = torch.arange(batch, dtype=torch.long) % 3
    return ids, att, pos, mask, qtype


def export(agent, path: Path) -> Path:
    """Write the agent's model, on the CPU in FP32, to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model = agent.model.to("cpu").float().eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            _trace_inputs(),
            str(path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes=DYNAMIC_AXES,
        )
    return path


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def verify(agent, path: Path, feeds: list[dict]) -> float:
    """Largest probability difference between the graph and PyTorch over real batches."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    model = agent.model.to("cpu").float().eval()
    worst = 0.0
    for feed in feeds:
        inputs = [feed[name] for name in INPUT_NAMES]
        with torch.no_grad():
            want = [t.numpy() for t in model(*inputs)]
        got = session.run(OUTPUT_NAMES, {name: t.numpy() for name, t in zip(INPUT_NAMES, inputs)})
        for g, w in zip(got, want):
            worst = max(worst, float(np.max(np.abs(_softmax(g) - _softmax(w)))))
    if worst > PARITY_ATOL:
        raise SystemExit(f"export check failed: probabilities differ from PyTorch by {worst:.2e}")
    return worst
