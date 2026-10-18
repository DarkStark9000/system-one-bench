# Results

One folder per machine, and one folder per run inside it: `results/<machine>/<UTC time>-<tag>/`.

## results.csv columns

| Column | Meaning |
|---|---|
| `checkpoint`, `revision` | The Laya checkpoint and its pinned commit |
| `backend` | The runtime and device (see the README) |
| `workload`, `size` | `questions` with 1, 5, 10 or 50 questions on one state, or `batch` with 1, 8, 32 or 64 states |
| `status`, `error` | `ok`, `load failed` or `run failed`, and the error text |
| `precision` | The precision the forward pass really ran in |
| `rows`, `tokens` | Question rows in the forward pass, and their real tokens |
| `calls` | Timed calls across all rounds |
| `p50_ms`, `p99_ms` | Medians of the round p50s and p99s, nearest-rank percentiles |
| `p50_low_ms`, `p50_high_ms` | The lowest and highest round p50: the noise band |
| `decisions_per_s` | Rows answered per second at the p50 |
| `ms_prepare` to `ms_decode` | Median phase times; `ms_model` replaces `ms_transfer_in` to `ms_transfer_out` for ONNX Runtime |
| `kernels_per_call`, `device_ms_per_call` | CUDA only: kernels launched per call, and their summed device time |
| `gflop` | Useful work per call: matrix multiplications on real tokens |
| `tflops_wall`, `tflops_device` | That work over the p50, and over the device time |
| `gpu_temp_start_c`, `gpu_clock_start_mhz`, `gpu_temp_end_c`, `gpu_clock_end_mhz` | NVIDIA only: the GPU's temperature and SM clock when the row's timing began and ended |
| `rss_mb`, `peak_mb` | The benchmark process's memory after the row, and its peak so far |
| `gpu_mem_start_mib` | NVIDIA only: GPU memory in use when the row's timing began. Near the card's total, Windows can spill GPU memory to system RAM and slow every call |
| `parity_max_prob_diff`, `parity_argmax` | Against FP32 PyTorch on the CPU: the largest probability difference, and answers that pick the same option |

An empty or `nan` cell means the column does not apply to that backend.
