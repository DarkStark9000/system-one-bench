"""`sob`: run the benchmark, or reproduce a tagged release's matrix on this machine.

    sob env                               print this machine's fingerprint
    sob run --backend torch-cuda-bf16     run chosen rows
    sob reproduce --tag v0.1.0 --machine laptop-b
                                          run every row the release ran, for what this machine has

Results go to results/<machine>/<run id>/: results.csv, env.json and run.json.
"""

import argparse
import csv
import datetime
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PRESETS = {
    "v0.1.0": {
        "checkpoints": ["english", "typed-decisions"],
        "sweeps": [("questions", n) for n in (1, 5, 10, 50)] + [("batch", n) for n in (1, 8, 32, 64)],
        "backends": {
            # GPU backends first, CPU last: an hour of CPU load heats a laptop and slows its GPU.
            "cuda": [
                "torch-cuda-bf16",
                "torch-cuda-fp16",
                "torch-cuda-fp32",
                "torch-cuda-graphs",
                "torch-cuda-tilelang",
                "ort-cuda",
                "ort-trt-fp16",
                "torch-cpu",
                "ort-cpu",
            ],
            "mps": ["torch-cpu", "torch-mps", "ort-cpu", "ort-coreml"],
            "dml": ["torch-cpu", "ort-cpu", "ort-dml", "ort-vitisai"],
            "cpu": ["torch-cpu", "ort-cpu"],
        },
    },
}


def _platform() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if sys.platform == "win32":
        return "dml"
    return "cpu"


def _bench_commit() -> str:
    """This repository's commit, with a mark when the working tree has uncommitted changes."""
    import subprocess

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "pyproject.toml", "uv.lock"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}+uncommitted" if dirty else commit


def _write(rows, machine: str, settings, label: str) -> Path:
    from system_one_bench import inputs, models
    from system_one_bench.env import fingerprint
    from system_one_bench.runner import as_record

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = Path("results") / machine / f"{stamp}-{label}"
    folder.mkdir(parents=True, exist_ok=True)
    records = [as_record(row) for row in rows]
    with (folder / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (folder / "env.json").write_text(json.dumps(fingerprint(), indent=2), encoding="utf-8")
    run = {
        "label": label,
        "utc": stamp,
        "machine": machine,
        "bench_commit": _bench_commit(),
        "input_sha256": inputs.input_digest(),
        "pins_match_laya": not models.check_pins(),
        "settings": {k: str(v) for k, v in vars(settings).items()},
        "command": " ".join(sys.argv),
    }
    (folder / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return folder


def _settings(args):
    from system_one_bench.runner import Settings

    return Settings(
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
        rest=not args.no_rest,
        cache=Path(args.cache),
    )


def _run(checkpoints, backends, sweeps, settings):
    from system_one_bench.models import checkpoint
    from system_one_bench.runner import run_backend

    rows = []
    for name in checkpoints:
        for backend in backends:
            rows.extend(run_backend(backend, checkpoint(name), sweeps, settings))
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="sob", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("env", help="print this machine's fingerprint")
    for name in ("run", "reproduce"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--machine", required=True, help="a short label for this machine, such as laptop-b")
        cmd.add_argument("--warmup", type=int, default=5)
        cmd.add_argument("--iterations", type=int, default=30)
        cmd.add_argument("--repeats", type=int, default=5)
        cmd.add_argument("--no-rest", action="store_true", help="skip the rest before each backend")
        cmd.add_argument("--cache", default=".cache")
    run = sub.choices["run"]
    run.add_argument("--checkpoint", default="english")
    run.add_argument("--backend", required=True, help="comma-separated backends")
    run.add_argument("--workload", default="questions,batch")
    run.add_argument("--sizes", default="", help="comma-separated sizes; default is the release sizes")
    sub.choices["reproduce"].add_argument("--tag", required=True, choices=sorted(PRESETS))
    args = parser.parse_args(argv)

    if args.command == "env":
        from system_one_bench.env import fingerprint

        print(json.dumps(fingerprint(), indent=2))
        return 0

    settings = _settings(args)
    if args.command == "reproduce":
        preset = PRESETS[args.tag]
        backends = preset["backends"][_platform()]
        rows = _run(preset["checkpoints"], backends, preset["sweeps"], settings)
        label = args.tag
    else:
        workloads = args.workload.split(",")
        defaults = PRESETS["v0.1.0"]["sweeps"]
        sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else None
        sweeps = [(k, n) for k, n in defaults if k in workloads and (sizes is None or n in sizes)]
        rows = _run(args.checkpoint.split(","), args.backend.split(","), sweeps, settings)
        label = "run"
    folder = _write(rows, args.machine, settings, label)
    print(f"wrote {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
