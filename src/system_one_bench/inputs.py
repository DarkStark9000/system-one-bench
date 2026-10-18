"""The fixed inputs every run uses, so two machines time the same work.

Two workloads:

- ``questions``: one state with 1, 5, 10 or 50 questions per call. The state and
  the two alternating questions are copied from laya's own latency script
  (`research/scripts/bench_latency.py` at tag v0.4.11, Apache 2.0), so these
  rows compare directly with the tables in laya's BENCHMARKS.md.
- ``batch``: 1, 8, 32 or 64 states with one question each, passed to
  `predict_batch` in one call, so the rows share one forward pass.

Each state is synthetic support-ticket text written for this repository. No
state comes from a customer, an employer or a private dataset.
"""

import hashlib
import json

# From laya v0.4.11, research/scripts/bench_latency.py (Apache 2.0, Convai Innovations).
STATE_EN = {
    "ticket": {
        "subject": "Payout failing",
        "messages": [
            {
                "from": "customer",
                "text": "Hi, my Stripe payouts have failed for 3 days and I am losing sales. Please help ASAP. " * 6,
            }
        ],
    }
}
Q_NOUL = {"type": "noul", "instructions": "Does `ticket.messages[0].text` express urgency?"}
Q_CHOICE = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "payments", "technical": "bugs and integrations", "sales": "pricing"},
}


def questions(n: int) -> dict:
    """laya's `qs(n)`: n questions that alternate a 3-option choice and a yes-or-no."""
    return {f"q{i}": (Q_NOUL if i % 2 else Q_CHOICE) for i in range(n)}


_OPENERS = [
    "Hi team,",
    "Hello,",
    "Good morning,",
    "To whoever reads this,",
    "Hey support,",
    "Dear billing team,",
    "Quick question:",
    "Following up on my last email:",
]

_BODIES = [
    "we were charged twice for the March invoice and need the duplicate refunded.",
    "the dashboard shows a blank page after login since this morning's update.",
    "can you send a quote for 40 more seats on the annual plan?",
    "our webhook deliveries fail with a 502 error on every retry.",
    "I want to change the card on file before the next renewal date.",
    "the export to CSV drops every row after the first thousand.",
    "is there a discount for nonprofits, and how do we apply for it?",
    "two users in our workspace cannot reset their passwords; the email never arrives.",
]


def batch_states() -> list[str]:
    """64 distinct short tickets: every opener with every body, in a fixed order."""
    return [f"{opener} {body}" for opener in _OPENERS for body in _BODIES]


BATCH_QUESTION = {"route": Q_CHOICE}

QUESTION_COUNTS = (1, 5, 10, 50)
BATCH_SIZES = (1, 8, 32, 64)


def workload(kind: str, size: int) -> tuple[list, dict]:
    """The states and questions for one row of a sweep."""
    if kind == "questions":
        return [STATE_EN], questions(size)
    if kind == "batch":
        states = batch_states()
        if size > len(states):
            raise SystemExit(f"batch size {size} exceeds the {len(states)} fixed states")
        return states[:size], BATCH_QUESTION
    raise SystemExit(f"unknown workload {kind!r}; choose questions or batch")


def input_digest() -> str:
    """SHA-256 of every fixed input, recorded with each result."""
    payload = {
        "state_en": STATE_EN,
        "questions_50": questions(50),
        "batch_states": batch_states(),
        "batch_question": BATCH_QUESTION,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
