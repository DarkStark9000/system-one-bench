"""Turn a result folder into Markdown tables and a JSON summary.

    sob report results/<machine>/<run>

Writes report.md and summary.json beside results.csv. Every number in them is
read from results.csv; nothing is estimated. Differences inside the noise band
(the p50 ranges overlap) are marked "same", following the noise rule in stats.py.
"""

import csv
import json
import math
from pathlib import Path

from system_one_bench.runner import PHASE_COLUMNS as PHASES


def _num(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _fmt(value: float, digits: int = 1) -> str:
    return "" if math.isnan(value) else f"{value:.{digits}f}"


def load(folder: Path) -> list[dict]:
    with (folder / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["size"] = int(row["size"])
        for key, value in list(row.items()):
            if key not in ("checkpoint", "revision", "backend", "workload", "status", "error", "precision", "parity_argmax", "size"):
                row[key] = _num(value)
    return rows


def _find(rows, checkpoint, backend, workload, size):
    for row in rows:
        if (row["checkpoint"], row["backend"], row["workload"], row["size"]) == (checkpoint, backend, workload, size):
            return row
    return None


def latency_table(rows, checkpoint: str, workload: str) -> list[str]:
    sizes = sorted({r["size"] for r in rows if r["workload"] == workload and r["checkpoint"] == checkpoint})
    if not sizes:
        return ["No rows."]
    backends = list(dict.fromkeys(r["backend"] for r in rows if r["checkpoint"] == checkpoint))
    unit = "questions" if workload == "questions" else "states"
    lines = [f"| Backend | " + " | ".join(f"{s} {unit}" for s in sizes) + " |", "| --- |" + " --- |" * len(sizes)]
    for backend in backends:
        cells = []
        for size in sizes:
            row = _find(rows, checkpoint, backend, workload, size)
            if row is None:
                cells.append("")
            elif row["status"] != "ok":
                cells.append(row["status"])
            else:
                cells.append(f"{_fmt(row['p50_ms'])} ms")
        lines.append(f"| {backend} | " + " | ".join(cells) + " |")
    return lines


def phase_table(rows, checkpoint: str, backend: str) -> list[str]:
    picked = [r for r in rows if r["checkpoint"] == checkpoint and r["backend"] == backend and r["status"] == "ok"]
    present = [p for p in PHASES if any(not math.isnan(r.get(f"ms_{p}", math.nan)) for r in picked)]
    lines = ["| Workload | " + " | ".join(present) + " | split total | p50 |", "| --- |" + " --- |" * (len(present) + 2)]
    for row in picked:
        values = [row.get(f"ms_{p}", math.nan) for p in present]
        total = sum(v for v in values if not math.isnan(v))
        lines.append(
            f"| {row['workload']} {row['size']} | " + " | ".join(_fmt(v, 2) for v in values)
            + f" | {_fmt(total, 2)} | {_fmt(row['p50_ms'], 2)} |"
        )
    return lines


def gpu_table(rows, checkpoint: str) -> list[str]:
    lines = [
        "| Backend | Workload | Kernels per call | GPU busy, ms | p50, ms | GPU busy share | Useful GFLOP | TFLOP/s while busy |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row["checkpoint"] != checkpoint or math.isnan(row.get("kernels_per_call", math.nan)) or row["status"] != "ok":
            continue
        share = row["device_ms_per_call"] / row["p50_ms"] if row["p50_ms"] else math.nan
        lines.append(
            f"| {row['backend']} | {row['workload']} {row['size']} | {_fmt(row['kernels_per_call'], 0)} | "
            f"{_fmt(row['device_ms_per_call'], 2)} | {_fmt(row['p50_ms'], 2)} | {_fmt(100 * share, 0)}% | "
            f"{_fmt(row['gflop'], 1)} | {_fmt(row['tflops_device'], 2)} |"
        )
    return lines


def parity_lines(rows) -> list[str]:
    ok = [r for r in rows if r["status"] == "ok"]
    worst = max(ok, key=lambda r: r["parity_max_prob_diff"]) if ok else None
    disagree = [r for r in ok if r["parity_argmax"].split("/")[0] != r["parity_argmax"].split("/")[-1]]
    lines = [f"Rows measured: {len(ok)} of {len(rows)}."]
    if worst is not None:
        lines.append(
            f"Largest probability difference from FP32 PyTorch on the CPU: {worst['parity_max_prob_diff']:.1e} "
            f"({worst['checkpoint']}, {worst['backend']}, {worst['workload']} {worst['size']})."
        )
    lines.append(f"Rows where any answer picked a different option: {len(disagree)}.")
    for row in disagree:
        lines.append(f"- {row['checkpoint']} {row['backend']} {row['workload']} {row['size']}: {row['parity_argmax']}")
    failed = [r for r in rows if r["status"] != "ok"]
    for row in failed:
        lines.append(f"- {row['status']}: {row['checkpoint']} {row['backend']} {row['workload']} {row['size']}: {row['error'][:160]}")
    return lines


def checkpoints_compared(rows) -> list[str]:
    """English against typed-decisions: same encoder and size, so latency should match within noise."""
    lines = ["| Backend | Workload | english p50 | typed-decisions p50 | Differs? |", "| --- | --- | --- | --- | --- |"]
    for row in rows:
        if row["checkpoint"] != "english" or row["status"] != "ok":
            continue
        other = _find(rows, "typed-decisions", row["backend"], row["workload"], row["size"])
        if other is None or other["status"] != "ok":
            continue
        overlap = not (row["p50_high_ms"] < other["p50_low_ms"] or other["p50_high_ms"] < row["p50_low_ms"])
        lines.append(
            f"| {row['backend']} | {row['workload']} {row['size']} | {_fmt(row['p50_ms'])} | {_fmt(other['p50_ms'])} | "
            f"{'same (ranges overlap)' if overlap else 'yes'} |"
        )
    return lines


def summary(rows) -> dict:
    """The values the essay and its figure read, keyed checkpoint/backend/workload/size."""
    out = {}
    for row in rows:
        key = f"{row['checkpoint']}/{row['backend']}/{row['workload']}/{row['size']}"
        out[key] = {
            k: (None if isinstance(v, float) and math.isnan(v) else v)
            for k, v in row.items()
            if k not in ("checkpoint", "backend", "workload", "size", "revision")
        }
    return out


def write(folder: Path) -> Path:
    rows = load(folder)
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    env = json.loads((folder / "env.json").read_text(encoding="utf-8"))
    parts = [
        f"# Results: {run['machine']}",
        "",
        f"{env.get('cpu', '')}; {env.get('gpu', '')} (driver {env.get('gpu_driver', '')}); {env.get('memory_gib', '')} GiB; "
        f"{env.get('os', '')}. Note: {run.get('note', '')}. Bench commit {run.get('bench_commit', '')}.",
        "",
    ]
    # A resumed run is a mixed run: name every session's commit, so a reader sees which code measured what.
    sessions = run.get("sessions", [])
    if len(sessions) > 1:
        listed = "; ".join(f"{i}: {s['started']}, commit {s['bench_commit']}" for i, s in enumerate(sessions, 1))
        parts += [f"Sessions ({listed}). Each later session measured only the rows missing from the folder.", ""]
    superseded = folder / "results.superseded.csv"
    if superseded.exists():
        with superseded.open(newline="", encoding="utf-8") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        parts += [f"`results.superseded.csv` keeps {count} earlier rows that a later session measured again.", ""]
    session = 0
    for ceiling in run.get("ceilings", []):
        session += ceiling["when"] == "start"
        prefix = f"Session {session}: measured" if len(sessions) > 1 else "Measured"
        parts.append(
            f"{prefix} ceiling at the {ceiling['when']}: BF16 matmul {ceiling.get('matmul_bf16_tflops')} TFLOP/s, "
            f"FP16 matmul {ceiling.get('matmul_fp16_tflops')} TFLOP/s, device copy {ceiling.get('copy_gbps')} GB/s."
        )
    parts.append("")
    for checkpoint in dict.fromkeys(r["checkpoint"] for r in rows):
        parts += [f"## {checkpoint}: p50 latency, questions per call", "", *latency_table(rows, checkpoint, "questions"), ""]
        parts += [f"## {checkpoint}: p50 latency, states per call (one question each)", "", *latency_table(rows, checkpoint, "batch"), ""]
        parts += [f"## {checkpoint}: GPU work per call", "", *gpu_table(rows, checkpoint), ""]
        for backend in dict.fromkeys(r["backend"] for r in rows if r["checkpoint"] == checkpoint):
            parts += [f"### {checkpoint}, {backend}: phase split (median ms)", "", *phase_table(rows, checkpoint, backend), ""]
    parts += ["## english against typed-decisions", "", *checkpoints_compared(rows), ""]
    parts += ["## Answers and failures", "", *parity_lines(rows), ""]
    (folder / "report.md").write_text("\n".join(parts), encoding="utf-8")
    (folder / "summary.json").write_text(json.dumps(summary(rows), indent=2), encoding="utf-8")
    return folder / "report.md"
