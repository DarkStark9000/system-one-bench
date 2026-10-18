# system-one-bench

Reproducible latency measurements for [Laya](https://huggingface.co/convaiinnovations/laya), the System 1 decision model by Convai Innovations. One command times Laya on your machine, splits each call into phases, counts the GPU kernels it launches, and checks that every backend gives the same answers.

Laya, its checkpoints and its runtime are the work of Convai Innovations, released under Apache 2.0. This repository only measures them. It is independent of Convai and redistributes no weights; see `NOTICE`.

## What it measures

For each checkpoint, backend and workload row:

- **Latency.** p50 and p99 of the real `predict_batch` call, over five independent rounds. The spread is the range of the round p50s.
- **Phases.** The same call, run step by step through laya's own methods, with the device synchronized between steps: prepare, tokenize, collate, copy in, dispatch, encoder, head, copy out, decode. ONNX Runtime runs the encoder and the head as one graph, so its split has one `model` phase.
- **Kernels.** On CUDA, the PyTorch profiler counts the kernels one call launches and sums their device time. When the kernels run for much less time than the call takes, the GPU waits for the host.
- **Work.** Useful floating-point operations from the checkpoint's own config, real tokens only, and the rate they imply.
- **Answers.** Every backend's probabilities against FP32 PyTorch on the CPU: the largest difference and how many answers pick the same option.

Two workloads, fixed in `src/system_one_bench/inputs.py`:

| Workload | Sizes | What changes |
|---|---|---|
| `questions` | 1, 5, 10, 50 | Questions per call on one state. The state and questions are laya's own latency-script inputs, so rows compare with laya's published tables. |
| `batch` | 1, 8, 32, 64 | States per call, one question each, in one forward pass. |

## Backends

| Backend | Runs on | What it is |
|---|---|---|
| `torch-cpu` | CPU | laya's PyTorch runtime in FP32, the reference for answers |
| `torch-cuda-bf16` | NVIDIA | laya's default on CUDA: eager PyTorch under BF16 autocast |
| `torch-cuda-fp16` | NVIDIA | the same under FP16 autocast |
| `torch-cuda-fp32` | NVIDIA | the same with autocast off |
| `torch-cuda-graphs` | NVIDIA | laya with `torch.compile` in `reduce-overhead` mode (CUDA graphs) |
| `torch-cuda-tilelang` | NVIDIA | laya's TileLang fast path (`laya[fast]`) |
| `torch-mps` | Apple GPU | laya on Metal |
| `ort-cpu`, `ort-cuda` | CPU, NVIDIA | an ONNX export in ONNX Runtime, FP32 |
| `ort-trt-fp16` | NVIDIA | the export in ONNX Runtime's TensorRT provider, FP16 with LayerNorm in FP32 |
| `ort-dml` | DirectX 12 GPU | ONNX Runtime with DirectML |
| `ort-coreml` | Apple | ONNX Runtime with Core ML |
| `ort-vitisai` | AMD Ryzen AI NPU | ONNX Runtime with Vitis AI |

A backend that cannot load, or that falls back to another device, becomes a row with its status and the error. It never becomes a gap or a silent CPU result.

## Run it

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/DarkStark9000/system-one-bench
cd system-one-bench
uv sync --extra onnx                    # add --extra tensorrt --extra graphs --extra fast on NVIDIA
uv run sob env                          # what will be recorded about this machine
uv run sob reproduce --tag v0.1.0 --machine my-laptop
```

`reproduce` runs every row the release ran, for the backends this machine has. One row at a time:

```bash
uv run sob run --machine my-laptop --checkpoint english --backend torch-cuda-bf16 --workload batch --sizes 1,64
```

The first run downloads each checkpoint at its pinned commit and refuses weights whose SHA-256 does not match. Results go to `results/<machine>/<run>/`.

On Windows, `torch-cuda-graphs` compiles with MSVC: run the command from an "x64 Native Tools" prompt of Visual Studio, or after `vcvars64.bat`. Without it the row fails with "Compiler: cl is not found". `torch-cuda-tilelang` compiles kernels with the CUDA toolkit. Install its compiler with NVVM (on Windows, the network installer's `nvcc_13.0`, `crt_13.0`, `nvvm_13.0`, `cudart_13.0` and `thrust_13.0` components; no driver is needed) and set `CUDA_HOME`. On Windows also set `TILELANG_DISABLE_CLANG_CL=1`: TileLang 0.1.15 otherwise hands `nvcc` the `clang-cl` host compiler, which `nvcc` rejects ([tile-ai/tilelang#3509](https://github.com/tile-ai/tilelang/issues/3509), fixed after 0.1.15).

## Pinned inputs

| Checkpoint | Repository | Commit | Weights SHA-256 |
|---|---|---|---|
| english | `convaiinnovations/laya` | `55cf4c4` | `891102d3...` |
| typed-decisions | `convaiinnovations/laya-typed-decisions` | `1a793eb` | `4fa56de7...` |
| multilingual | `convaiinnovations/laya-multilingual` | `e4e9ddf` | `9d628fd9...` |

The commits are the reviewed pins in laya's own `laya.revisions.PINNED_REVISIONS`. The weights at these commits are byte-identical to each repository's main branch on 2026-10-18. The full digests are in `src/system_one_bench/models.py`, and the package versions are locked in `uv.lock`.

## Read a result

- Laptop GPUs change clocks with power and temperature. `env.json` records the power source; run on mains power.
- Two rows differ only when their p50 ranges do not overlap. A difference inside the overlap is noise.
- The phase split adds device waits, so its total can differ from the unsplit p50 by a few percent. The unsplit p50 is the latency a caller sees.
- A FLOPs rate here is a floor: it counts matrix multiplications on real tokens only.

## Add your machine

Read `CONTRIBUTING.md`. A result from a machine we do not have is the most useful pull request this repository can get.

## License

MIT for this repository's code and results. Laya is Apache 2.0, by Convai Innovations.
