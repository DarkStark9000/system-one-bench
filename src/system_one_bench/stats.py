"""Percentiles, repeats and the noise rule.

A row runs `repeats` independent rounds of `iterations` timed calls each. Each
round gives its own p50 and p99. The reported p50 and p99 are the medians of
the round values, and the spread is the range of the round p50s. Percentiles
use the nearest-rank method, so a p99 is always a call that really happened.

Noise rule: two results differ only when their p50 ranges do not overlap. A
difference inside the overlap is reported as no difference.
"""

import math
import statistics
from dataclasses import dataclass


def nearest_rank(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class Summary:
    p50_ms: float
    p99_ms: float
    p50_low_ms: float
    p50_high_ms: float
    calls: int


def summarize(rounds: list[list[float]]) -> Summary:
    p50s = [nearest_rank(r, 50) for r in rounds]
    p99s = [nearest_rank(r, 99) for r in rounds]
    return Summary(
        p50_ms=statistics.median(p50s),
        p99_ms=statistics.median(p99s),
        p50_low_ms=min(p50s),
        p50_high_ms=max(p50s),
        calls=sum(len(r) for r in rounds),
    )


def differs(a: Summary, b: Summary) -> bool:
    """True when the two p50 ranges do not overlap."""
    return a.p50_high_ms < b.p50_low_ms or b.p50_high_ms < a.p50_low_ms
