"""Useful floating-point operations in one forward pass, from the checkpoint's own config.

The count covers matrix multiplications only, at two operations per
multiply-add, and only the real tokens of each row (padding does no useful
work). Embeddings, normalizations, activations and softmax are left out: they
are memory-bound and small next to the projections. The result is a floor on
the work, so the achieved rate it implies is a floor too.

ModernBERT layers (encoder):
    attention projections  Wqkv d x 3d, Wo d x d       -> 2 * 4 * d^2 per token
    gated MLP              Wi d x 2f, Wo f x d          -> 2 * 3 * d * f per token
    attention              scores and values over k keys -> 2 * 2 * k * d per token
    where k is every token in a full-attention layer and at most
    `local_attention` + 1 tokens in a sliding-window layer.

Decision head (laya `DecisionModel`): `head_layers` standard encoder layers of
width d with a 4d feed-forward, attending over the whole row.
"""

import json
from pathlib import Path


def _attention_keys(layer_type: str, length: int, window: int) -> int:
    if layer_type == "sliding_attention":
        return min(length, window + 1)
    return length


def row_flops(encoder_cfg: dict, head_layers: int, length: int) -> float:
    d = encoder_cfg["hidden_size"]
    f = encoder_cfg["intermediate_size"]
    window = encoder_cfg.get("local_attention", length)
    layer_types = encoder_cfg.get("layer_types") or ["full_attention"] * encoder_cfg["num_hidden_layers"]
    per_token = 0.0
    for layer_type in layer_types:
        keys = _attention_keys(layer_type, length, window)
        per_token += 2 * 4 * d * d + 2 * 3 * d * f + 2 * 2 * keys * d
    head_per_token = head_layers * (2 * 4 * d * d + 2 * 2 * d * (4 * d) + 2 * 2 * length * d)
    return (per_token + head_per_token) * length


def batch_flops(encoder_cfg: dict, head_layers: int, lengths: list[int]) -> float:
    return sum(row_flops(encoder_cfg, head_layers, n) for n in lengths)


def configs(model_dir: str) -> tuple[dict, int]:
    """The encoder config and the head depth of a downloaded checkpoint."""
    root = Path(model_dir)
    encoder_cfg = json.loads((root / "encoder" / "config.json").read_text(encoding="utf-8"))
    agent_cfg = json.loads((root / "rl_agent_config.json").read_text(encoding="utf-8"))
    return encoder_cfg, int(agent_cfg.get("head_layers", 2))
