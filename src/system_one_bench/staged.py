"""One Laya call, split into timed phases, on the PyTorch runtime.

`staged_predict` runs the same steps as `laya.Agent.predict_batch` (one chunk, no
hooks), in the same order, through the agent's own methods. Between steps it
waits for the device to finish, so each phase is wall time from the host's view:

    prepare       validate and normalize the questions
    tokenize      serialize each state and build one token sequence per question
    collate       pad the sequences into batch tensors on the host
    transfer_in   copy the batch to the device
    dispatch      from the copy to the first encoder op: autocast, locks, Python
    encoder       the encoder's forward pass
    head          the decision head, scorer and act head
    forward       dispatch, encoder and head as one phase, when a compiled or
                  fast-path forward hides the encoder's boundary
    transfer_out  copy the logits back to the host
    decode        softmax, temperature and typed answers in NumPy and Python

The waits change the timing a little, so every result also carries the
unsplit time of the real `predict_batch` call. `parity` proves that the staged
call returns the same answers as the real one.
"""

import time
from collections.abc import Callable
from contextlib import contextmanager

import numpy as np
import torch
from laya.common import _disable_question_token_reuse, _reuse_question_tokens, collate_items

PHASES = (
    "prepare",
    "tokenize",
    "collate",
    "transfer_in",
    "dispatch",
    "encoder",
    "head",
    "forward",
    "transfer_out",
    "decode",
)


def device_sync(device: torch.device) -> Callable[[], None]:
    if device.type == "cuda":
        return torch.cuda.synchronize
    if device.type == "mps":
        return torch.mps.synchronize
    return lambda: None


class PhaseClock:
    def __init__(self, sync: Callable[[], None]):
        self.sync = sync
        self.ms: dict[str, float] = {}
        self.last = 0.0

    def start(self) -> None:
        self.sync()
        self.last = time.perf_counter()

    def mark(self, phase: str) -> None:
        self.sync()
        now = time.perf_counter()
        self.ms[phase] = self.ms.get(phase, 0.0) + (now - self.last) * 1000.0
        self.last = now


@contextmanager
def encoder_marks(agent, clock: PhaseClock):
    """Mark the start and end of the encoder's forward pass without changing the model."""
    encoder = agent.model.encoder
    before = encoder.register_forward_pre_hook(lambda *_: clock.mark("dispatch"))
    after = encoder.register_forward_hook(lambda *_: clock.mark("encoder"))
    try:
        yield
    finally:
        before.remove()
        after.remove()


@torch.no_grad()
@_reuse_question_tokens
def staged_predict(agent, states: list, questions: dict) -> tuple[list[dict], dict[str, float]]:
    """Answer `questions` for every state in one forward pass, timing each phase."""
    clock = PhaseClock(device_sync(agent.device))
    clock.start()
    if len(states) == 1:
        _disable_question_token_reuse()
    ids = list(questions)
    for qid in ids:
        agent._check_question(qid, questions[qid])
    internal = {qid: agent._to_internal(questions[qid]) for qid in ids}
    clock.mark("prepare")

    encoded = [agent._encode_state(state, ids, internal) for state in states]
    clock.mark("tokenize")

    batch = collate_items(encoded, agent.tok.pad_token_id)
    clock.mark("collate")

    on_device = {k: (v.to(agent.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    clock.mark("transfer_in")

    with encoder_marks(agent, clock):
        logits, act = agent._infer(on_device)
    # A compiled or fast-path forward replaces the encoder module's own forward, so its
    # hooks never fire; the whole pass is then one "forward" phase, not a false "head".
    clock.mark("head" if "encoder" in clock.ms else "forward")

    logits_np = logits.float().cpu().numpy()
    act_np = torch.softmax(act.float(), -1).cpu().numpy()
    clock.mark("transfer_out")

    results, row = [], 0
    for items in encoded:
        results.append({"answers": agent._decode_answers(logits_np, act_np, items, ids, internal, row)})
        row += len(items)
    clock.mark("decode")
    return results, clock.ms


def probabilities(results: list[dict]) -> np.ndarray:
    """Every option probability of every answer, in a fixed order, for parity checks."""
    values = []
    for result in results:
        for qid in sorted(result["answers"]):
            answer = result["answers"][qid]
            if answer["type"] == "noul":
                values.append(answer["noul"])
            else:
                probs = answer["probabilities"]
                values.extend(probs[key] for key in sorted(probs))
    return np.asarray(values, dtype=np.float64)


def parity(agent, states: list, questions: dict) -> float:
    """Largest probability difference between the staged call and `predict_batch`."""
    staged, _ = staged_predict(agent, states, questions)
    real = agent.predict_batch(states, questions)
    return float(np.max(np.abs(probabilities(staged) - probabilities(real))))
