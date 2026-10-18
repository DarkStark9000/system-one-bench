# Results: rtx4060-laptop

Intel(R) Core(TM) i9-14900HX; NVIDIA GeForce RTX 4060 Laptop GPU (driver 591.86); 31.7 GiB; Windows 10 (10.0.22631). Note: Lenovo Legion, Legion Edge thermal mode Performance, GPU overclock off, GPU working mode Hybrid, on mains power. Bench commit 67bb83bfd92ea03842538f4bf2a9d15475ccfe24.

Sessions (1: 20261018T184214Z, commit 67bb83bfd92ea03842538f4bf2a9d15475ccfe24; 2: 20261018T225215Z, commit 9ab879506a355ba08295e0fe70f784e845a480ab). Each later session measured only the rows missing from the folder.

`results.superseded.csv` keeps 16 earlier rows that a later session measured again.

Session 1: measured ceiling at the start: BF16 matmul 27.78 TFLOP/s, FP16 matmul 26.1 TFLOP/s, device copy 218.9 GB/s.
Session 1: measured ceiling at the end: BF16 matmul 28.67 TFLOP/s, FP16 matmul 27.47 TFLOP/s, device copy 220.97 GB/s.
Session 2: measured ceiling at the start: BF16 matmul 29.5 TFLOP/s, FP16 matmul 27.31 TFLOP/s, device copy 220.68 GB/s.
Session 2: measured ceiling at the end: BF16 matmul 25.48 TFLOP/s, FP16 matmul 25.02 TFLOP/s, device copy 220.72 GB/s.

## english: p50 latency, questions per call

| Backend | 1 questions | 5 questions | 10 questions | 50 questions |
| --- | --- | --- | --- | --- |
| torch-cuda-bf16 | 24.4 ms | 62.3 ms | 110.8 ms | 588.2 ms |
| torch-cuda-fp16 | 27.8 ms | 63.3 ms | 110.2 ms | 599.2 ms |
| torch-cuda-fp32 | 39.6 ms | 169.3 ms | 309.4 ms | 1651.6 ms |
| torch-cuda-graphs | 18.2 ms | 51.6 ms | 91.1 ms | 430.0 ms |
| torch-cuda-tilelang | 12.9 ms | 59.3 ms | 121.5 ms | 507.7 ms |
| ort-trt-fp16 | 10.8 ms | 36.2 ms | 68.6 ms | 397.0 ms |
| torch-cpu | 400.0 ms | 1673.2 ms | 3344.8 ms | 14995.6 ms |
| ort-cpu | 275.7 ms | 1422.8 ms | 2938.7 ms | 14885.8 ms |
| ort-cuda | 22.7 ms | 95.2 ms | 222.7 ms | 1281.1 ms |

## english: p50 latency, states per call (one question each)

| Backend | 1 states | 8 states | 32 states | 64 states |
| --- | --- | --- | --- | --- |
| torch-cuda-bf16 | 29.1 ms | 30.9 ms | 84.6 ms | 173.9 ms |
| torch-cuda-fp16 | 20.9 ms | 31.4 ms | 86.7 ms | 176.2 ms |
| torch-cuda-fp32 | 20.4 ms | 58.3 ms | 232.2 ms | 483.8 ms |
| torch-cuda-graphs | 11.2 ms | 22.3 ms | 74.8 ms | 145.2 ms |
| torch-cuda-tilelang | 6.4 ms | 16.3 ms | 76.4 ms | 152.9 ms |
| ort-trt-fp16 | 8.5 ms | 14.1 ms | 53.5 ms | 105.9 ms |
| torch-cpu | 97.2 ms | 651.2 ms | 2320.1 ms | 4685.0 ms |
| ort-cpu | 81.8 ms | 493.5 ms | 2044.9 ms | 4312.5 ms |
| ort-cuda | 13.7 ms | 35.0 ms | 160.7 ms | 350.5 ms |

## english: GPU work per call

| Backend | Workload | Kernels per call | GPU busy, ms | p50, ms | GPU busy share | Useful GFLOP | TFLOP/s while busy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| torch-cuda-bf16 | questions 1 | 1329 | 19.58 | 24.45 | 80% | 146.6 | 7.49 |
| torch-cuda-bf16 | questions 5 | 1343 | 57.15 | 62.27 | 92% | 749.8 | 13.12 |
| torch-cuda-bf16 | questions 10 | 1317 | 105.37 | 110.77 | 95% | 1508.1 | 14.31 |
| torch-cuda-bf16 | questions 50 | 1313 | 575.39 | 588.25 | 98% | 7540.6 | 13.11 |
| torch-cuda-bf16 | batch 1 | 1165 | 12.90 | 29.10 | 44% | 34.1 | 2.65 |
| torch-cuda-bf16 | batch 8 | 1283 | 25.53 | 30.94 | 83% | 268.7 | 10.52 |
| torch-cuda-bf16 | batch 32 | 1373 | 76.54 | 84.58 | 90% | 1080.7 | 14.12 |
| torch-cuda-bf16 | batch 64 | 1315 | 174.15 | 173.88 | 100% | 2185.3 | 12.55 |
| torch-cuda-fp16 | questions 1 | 1329 | 19.41 | 27.81 | 70% | 146.6 | 7.55 |
| torch-cuda-fp16 | questions 5 | 1343 | 57.07 | 63.28 | 90% | 749.8 | 13.14 |
| torch-cuda-fp16 | questions 10 | 1317 | 111.10 | 110.17 | 101% | 1508.1 | 13.57 |
| torch-cuda-fp16 | questions 50 | 1313 | 590.05 | 599.16 | 98% | 7540.6 | 12.78 |
| torch-cuda-fp16 | batch 1 | 1165 | 12.75 | 20.94 | 61% | 34.1 | 2.68 |
| torch-cuda-fp16 | batch 8 | 1285 | 24.96 | 31.36 | 80% | 268.7 | 10.76 |
| torch-cuda-fp16 | batch 32 | 1373 | 82.17 | 86.75 | 95% | 1080.7 | 13.15 |
| torch-cuda-fp16 | batch 64 | 1315 | 166.92 | 176.21 | 95% | 2185.3 | 13.09 |
| torch-cuda-fp32 | questions 1 | 975 | 36.37 | 39.60 | 92% | 146.6 | 4.03 |
| torch-cuda-fp32 | questions 5 | 980 | 157.48 | 169.32 | 93% | 749.8 | 4.76 |
| torch-cuda-fp32 | questions 10 | 980 | 299.24 | 309.38 | 97% | 1508.1 | 5.04 |
| torch-cuda-fp32 | questions 50 | 921 | 1670.73 | 1651.57 | 101% | 7540.6 | 4.51 |
| torch-cuda-fp32 | batch 1 | 867 | 10.02 | 20.41 | 49% | 34.1 | 3.41 |
| torch-cuda-fp32 | batch 8 | 982 | 52.97 | 58.34 | 91% | 268.7 | 5.07 |
| torch-cuda-fp32 | batch 32 | 981 | 227.51 | 232.21 | 98% | 1080.7 | 4.75 |
| torch-cuda-fp32 | batch 64 | 923 | 491.24 | 483.82 | 102% | 2185.3 | 4.45 |
| torch-cuda-graphs | questions 1 | 610 | 16.52 | 18.24 | 91% | 146.6 | 8.87 |
| torch-cuda-graphs | questions 5 | 552 | 49.71 | 51.59 | 96% | 749.8 | 15.08 |
| torch-cuda-graphs | questions 10 | 528 | 91.35 | 91.06 | 100% | 1508.1 | 16.51 |
| torch-cuda-graphs | questions 50 | 526 | 423.17 | 430.03 | 98% | 7540.6 | 17.82 |
| torch-cuda-graphs | batch 1 | 554 | 9.87 | 11.19 | 88% | 34.1 | 3.46 |
| torch-cuda-graphs | batch 8 | 554 | 20.59 | 22.28 | 92% | 268.7 | 13.05 |
| torch-cuda-graphs | batch 32 | 528 | 73.06 | 74.78 | 98% | 1080.7 | 14.79 |
| torch-cuda-graphs | batch 64 | 554 | 135.09 | 145.21 | 93% | 2185.3 | 16.18 |
| torch-cuda-tilelang | questions 1 | 313 | 11.10 | 12.93 | 86% | 146.6 | 13.20 |
| torch-cuda-tilelang | questions 5 | 313 | 58.21 | 59.32 | 98% | 749.8 | 12.88 |
| torch-cuda-tilelang | questions 10 | 313 | 120.16 | 121.45 | 99% | 1508.1 | 12.55 |
| torch-cuda-tilelang | questions 50 | 313 | 502.24 | 507.66 | 99% | 7540.6 | 15.01 |
| torch-cuda-tilelang | batch 1 | 313 | 5.44 | 6.44 | 84% | 34.1 | 6.28 |
| torch-cuda-tilelang | batch 8 | 313 | 14.42 | 16.28 | 89% | 268.7 | 18.63 |
| torch-cuda-tilelang | batch 32 | 313 | 71.42 | 76.44 | 93% | 1080.7 | 15.13 |
| torch-cuda-tilelang | batch 64 | 313 | 142.95 | 152.92 | 93% | 2185.3 | 15.29 |

### english, torch-cuda-bf16: phase split (median ms)

| Workload | prepare | tokenize | collate | transfer_in | dispatch | encoder | head | transfer_out | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.01 | 0.55 | 0.09 | 0.13 | 0.06 | 21.75 | 2.08 | 0.11 | 0.08 | 24.86 | 24.45 |
| questions 5 | 0.02 | 0.98 | 0.25 | 0.16 | 0.08 | 51.35 | 8.84 | 0.17 | 0.20 | 62.04 | 62.27 |
| questions 10 | 0.02 | 1.57 | 0.42 | 0.18 | 0.08 | 99.03 | 11.80 | 0.16 | 0.27 | 113.52 | 110.77 |
| questions 50 | 0.06 | 7.60 | 2.33 | 0.30 | 0.10 | 548.60 | 31.76 | 0.19 | 0.90 | 591.85 | 588.25 |
| batch 1 | 0.01 | 0.27 | 0.08 | 0.14 | 0.06 | 17.75 | 2.43 | 0.12 | 0.08 | 20.93 | 29.10 |
| batch 8 | 0.01 | 0.69 | 0.23 | 0.15 | 0.06 | 26.31 | 3.16 | 0.15 | 0.24 | 31.00 | 30.94 |
| batch 32 | 0.01 | 2.18 | 0.75 | 0.19 | 0.08 | 75.46 | 6.11 | 0.16 | 0.74 | 85.68 | 84.58 |
| batch 64 | 0.01 | 3.97 | 1.33 | 0.18 | 0.08 | 156.95 | 9.63 | 0.16 | 1.31 | 173.63 | 173.88 |

### english, torch-cuda-fp16: phase split (median ms)

| Workload | prepare | tokenize | collate | transfer_in | dispatch | encoder | head | transfer_out | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.01 | 0.41 | 0.09 | 0.14 | 0.05 | 21.97 | 2.45 | 0.12 | 0.08 | 25.32 | 27.81 |
| questions 5 | 0.02 | 0.95 | 0.24 | 0.15 | 0.06 | 54.52 | 6.40 | 0.16 | 0.18 | 62.67 | 63.28 |
| questions 10 | 0.02 | 1.54 | 0.41 | 0.16 | 0.06 | 102.55 | 9.35 | 0.16 | 0.27 | 114.53 | 110.17 |
| questions 50 | 0.06 | 7.07 | 2.07 | 0.32 | 0.09 | 559.59 | 31.86 | 0.18 | 0.90 | 602.14 | 599.16 |
| batch 1 | 0.01 | 0.33 | 0.10 | 0.17 | 0.07 | 25.62 | 3.21 | 0.14 | 0.10 | 29.76 | 20.94 |
| batch 8 | 0.01 | 0.68 | 0.23 | 0.15 | 0.06 | 26.30 | 2.62 | 0.14 | 0.24 | 30.43 | 31.36 |
| batch 32 | 0.01 | 2.15 | 0.74 | 0.17 | 0.07 | 75.27 | 6.13 | 0.16 | 0.73 | 85.42 | 86.75 |
| batch 64 | 0.01 | 3.96 | 1.32 | 0.21 | 0.07 | 161.16 | 10.45 | 0.16 | 1.30 | 178.65 | 176.21 |

### english, torch-cuda-fp32: phase split (median ms)

| Workload | prepare | tokenize | collate | transfer_in | dispatch | encoder | head | transfer_out | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.01 | 0.44 | 0.10 | 0.14 | 0.04 | 40.06 | 3.35 | 0.13 | 0.09 | 44.37 | 39.60 |
| questions 5 | 0.02 | 0.98 | 0.25 | 0.16 | 0.05 | 159.48 | 11.34 | 0.15 | 0.19 | 172.61 | 169.32 |
| questions 10 | 0.02 | 1.71 | 0.42 | 0.17 | 0.05 | 282.35 | 20.41 | 0.18 | 0.29 | 305.60 | 309.38 |
| questions 50 | 0.06 | 6.11 | 1.71 | 0.39 | 0.07 | 1540.62 | 100.19 | 0.29 | 0.87 | 1650.30 | 1651.57 |
| batch 1 | 0.01 | 0.26 | 0.08 | 0.14 | 0.04 | 13.24 | 1.46 | 0.11 | 0.08 | 15.42 | 20.41 |
| batch 8 | 0.01 | 0.76 | 0.26 | 0.16 | 0.04 | 52.70 | 4.45 | 0.13 | 0.25 | 58.77 | 58.34 |
| batch 32 | 0.01 | 2.20 | 0.72 | 0.17 | 0.05 | 208.23 | 15.41 | 0.16 | 0.73 | 227.68 | 232.21 |
| batch 64 | 0.01 | 4.08 | 1.39 | 0.30 | 0.06 | 455.34 | 31.28 | 0.23 | 1.39 | 494.08 | 483.82 |

### english, torch-cuda-graphs: phase split (median ms)

| Workload | prepare | tokenize | collate | transfer_in | forward | transfer_out | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.01 | 0.40 | 0.09 | 0.14 | 17.33 | 0.15 | 0.09 | 18.21 | 18.24 |
| questions 5 | 0.02 | 0.98 | 0.26 | 0.15 | 49.97 | 0.22 | 0.19 | 51.79 | 51.59 |
| questions 10 | 0.02 | 1.47 | 0.40 | 0.15 | 90.53 | 0.27 | 0.25 | 93.08 | 91.06 |
| questions 50 | 0.06 | 6.03 | 1.69 | 0.39 | 398.81 | 0.41 | 0.87 | 408.25 | 430.03 |
| batch 1 | 0.01 | 0.27 | 0.08 | 0.14 | 10.46 | 0.14 | 0.09 | 11.19 | 11.19 |
| batch 8 | 0.01 | 0.68 | 0.22 | 0.14 | 20.88 | 0.18 | 0.24 | 22.36 | 22.28 |
| batch 32 | 0.01 | 2.06 | 0.71 | 0.15 | 72.25 | 0.26 | 0.67 | 76.12 | 74.78 |
| batch 64 | 0.01 | 4.00 | 1.38 | 0.27 | 131.85 | 0.30 | 1.37 | 139.19 | 145.21 |

### english, torch-cuda-tilelang: phase split (median ms)

| Workload | prepare | tokenize | collate | transfer_in | forward | transfer_out | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.02 | 0.63 | 0.15 | 0.21 | 11.65 | 0.23 | 0.13 | 13.01 | 12.93 |
| questions 5 | 0.02 | 1.46 | 0.39 | 0.21 | 57.73 | 0.29 | 0.28 | 60.36 | 59.32 |
| questions 10 | 0.03 | 2.38 | 0.63 | 0.22 | 116.60 | 0.34 | 0.38 | 120.59 | 121.45 |
| questions 50 | 0.07 | 7.48 | 2.02 | 0.40 | 493.17 | 0.36 | 0.96 | 504.44 | 507.66 |
| batch 1 | 0.01 | 0.25 | 0.08 | 0.14 | 5.79 | 0.13 | 0.08 | 6.49 | 6.44 |
| batch 8 | 0.01 | 0.66 | 0.25 | 0.15 | 14.52 | 0.15 | 0.25 | 15.99 | 16.28 |
| batch 32 | 0.01 | 2.10 | 0.73 | 0.15 | 74.08 | 0.24 | 0.70 | 78.00 | 76.44 |
| batch 64 | 0.01 | 5.01 | 1.94 | 0.30 | 143.30 | 0.27 | 1.42 | 152.25 | 152.92 |

### english, ort-trt-fp16: phase split (median ms)

| Workload | prepare | tokenize | collate | model | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.01 | 0.40 | 0.02 | 10.28 | 0.08 | 10.78 | 10.83 |
| questions 5 | 0.01 | 0.92 | 0.04 | 38.71 | 0.20 | 39.89 | 36.20 |
| questions 10 | 0.02 | 1.43 | 0.06 | 69.01 | 0.30 | 70.82 | 68.61 |
| questions 50 | 0.05 | 5.97 | 0.26 | 387.33 | 1.02 | 394.63 | 397.00 |
| batch 1 | 0.01 | 0.30 | 0.02 | 8.23 | 0.10 | 8.65 | 8.51 |
| batch 8 | 0.01 | 0.65 | 0.03 | 12.38 | 0.26 | 13.32 | 14.12 |
| batch 32 | 0.01 | 2.06 | 0.07 | 47.34 | 0.67 | 50.16 | 53.52 |
| batch 64 | 0.01 | 3.83 | 0.14 | 93.15 | 1.26 | 98.38 | 105.91 |

### english, torch-cpu: phase split (median ms)

| Workload | prepare | tokenize | collate | transfer_in | dispatch | encoder | head | transfer_out | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.02 | 1.26 | 0.26 | 0.02 | 0.09 | 345.20 | 51.64 | 0.04 | 0.21 | 398.74 | 399.99 |
| questions 5 | 0.03 | 3.18 | 0.88 | 0.02 | 0.10 | 1459.70 | 221.25 | 0.05 | 0.55 | 1685.75 | 1673.24 |
| questions 10 | 0.04 | 4.64 | 1.38 | 0.02 | 0.09 | 2973.75 | 362.61 | 0.06 | 0.76 | 3343.34 | 3344.79 |
| questions 50 | 0.11 | 13.37 | 4.58 | 0.02 | 0.09 | 13439.08 | 1521.34 | 0.03 | 1.38 | 14980.00 | 14995.65 |
| batch 1 | 0.02 | 0.80 | 0.24 | 0.02 | 0.09 | 90.74 | 9.22 | 0.04 | 0.20 | 101.37 | 97.24 |
| batch 8 | 0.02 | 2.41 | 1.02 | 0.02 | 0.10 | 595.06 | 52.18 | 0.05 | 0.87 | 651.72 | 651.22 |
| batch 32 | 0.02 | 6.29 | 3.04 | 0.02 | 0.09 | 2143.20 | 181.27 | 0.05 | 2.78 | 2336.75 | 2320.09 |
| batch 64 | 0.01 | 9.90 | 4.13 | 0.02 | 0.09 | 4322.22 | 351.83 | 0.05 | 5.16 | 4693.40 | 4684.98 |

### english, ort-cpu: phase split (median ms)

| Workload | prepare | tokenize | collate | model | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.02 | 1.24 | 0.05 | 277.35 | 0.25 | 278.92 | 275.69 |
| questions 5 | 0.03 | 3.15 | 0.11 | 1442.74 | 0.65 | 1446.68 | 1422.80 |
| questions 10 | 0.04 | 3.71 | 0.16 | 2909.05 | 0.66 | 2913.62 | 2938.73 |
| questions 50 | 0.13 | 17.89 | 0.52 | 15028.64 | 3.90 | 15051.08 | 14885.77 |
| batch 1 | 0.02 | 0.70 | 0.04 | 79.76 | 0.22 | 80.74 | 81.84 |
| batch 8 | 0.02 | 1.70 | 0.08 | 490.29 | 0.58 | 492.66 | 493.54 |
| batch 32 | 0.02 | 7.27 | 0.22 | 2024.25 | 3.01 | 2034.76 | 2044.86 |
| batch 64 | 0.02 | 8.95 | 0.34 | 4271.02 | 3.22 | 4283.54 | 4312.54 |

### english, ort-cuda: phase split (median ms)

| Workload | prepare | tokenize | collate | model | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.01 | 0.43 | 0.02 | 21.94 | 0.10 | 22.50 | 22.66 |
| questions 5 | 0.02 | 1.10 | 0.05 | 95.98 | 0.26 | 97.39 | 95.21 |
| questions 10 | 0.02 | 1.68 | 0.07 | 221.47 | 0.34 | 223.58 | 222.72 |
| questions 50 | 0.06 | 6.21 | 0.28 | 1277.92 | 1.02 | 1285.48 | 1281.09 |
| batch 1 | 0.01 | 0.27 | 0.01 | 13.28 | 0.09 | 13.66 | 13.70 |
| batch 8 | 0.01 | 0.82 | 0.03 | 33.89 | 0.28 | 35.03 | 35.02 |
| batch 32 | 0.01 | 2.43 | 0.09 | 154.31 | 0.77 | 157.61 | 160.68 |
| batch 64 | 0.01 | 5.16 | 0.20 | 337.42 | 1.37 | 344.16 | 350.51 |

## typed-decisions: p50 latency, questions per call

| Backend | 1 questions | 5 questions | 10 questions | 50 questions |
| --- | --- | --- | --- | --- |
| torch-cuda-bf16 | 24.6 ms | 61.2 ms | 110.0 ms | 589.6 ms |
| torch-cuda-fp16 | 24.7 ms | 60.5 ms | 109.3 ms | 598.1 ms |
| torch-cuda-fp32 | 37.9 ms | 167.2 ms | 310.0 ms | 1649.4 ms |
| torch-cuda-graphs | 17.4 ms | 52.2 ms | 91.5 ms | 428.2 ms |
| torch-cuda-tilelang | 12.3 ms | 57.8 ms | 118.3 ms | 509.2 ms |
| ort-trt-fp16 | 11.4 ms | 34.5 ms | 68.3 ms | 392.0 ms |
| torch-cpu | 411.2 ms | 1613.5 ms | 3189.9 ms | 15210.5 ms |
| ort-cpu | 273.5 ms | 1409.2 ms | 2936.8 ms | 15008.2 ms |
| ort-cuda | 22.5 ms | 98.9 ms | 226.6 ms | 1288.4 ms |

## typed-decisions: p50 latency, states per call (one question each)

| Backend | 1 states | 8 states | 32 states | 64 states |
| --- | --- | --- | --- | --- |
| torch-cuda-bf16 | 22.8 ms | 31.8 ms | 82.5 ms | 171.3 ms |
| torch-cuda-fp16 | 20.7 ms | 31.3 ms | 86.4 ms | 176.0 ms |
| torch-cuda-fp32 | 20.5 ms | 59.2 ms | 226.3 ms | 483.7 ms |
| torch-cuda-graphs | 11.1 ms | 22.4 ms | 75.0 ms | 144.1 ms |
| torch-cuda-tilelang | 6.5 ms | 16.0 ms | 77.6 ms | 153.2 ms |
| ort-trt-fp16 | 8.7 ms | 13.8 ms | 52.5 ms | 104.6 ms |
| torch-cpu | 98.8 ms | 660.0 ms | 2300.0 ms | 4629.1 ms |
| ort-cpu | 81.6 ms | 489.0 ms | 2028.5 ms | 4327.9 ms |
| ort-cuda | 14.6 ms | 35.4 ms | 161.1 ms | 351.0 ms |

## typed-decisions: GPU work per call

| Backend | Workload | Kernels per call | GPU busy, ms | p50, ms | GPU busy share | Useful GFLOP | TFLOP/s while busy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| torch-cuda-bf16 | questions 1 | 1329 | 19.57 | 24.60 | 80% | 146.6 | 7.49 |
| torch-cuda-bf16 | questions 5 | 1343 | 57.20 | 61.20 | 93% | 749.8 | 13.11 |
| torch-cuda-bf16 | questions 10 | 1317 | 105.60 | 109.98 | 96% | 1508.1 | 14.28 |
| torch-cuda-bf16 | questions 50 | 1313 | 582.31 | 589.63 | 99% | 7540.6 | 12.95 |
| torch-cuda-bf16 | batch 1 | 1165 | 12.97 | 22.83 | 57% | 34.1 | 2.63 |
| torch-cuda-bf16 | batch 8 | 1283 | 25.56 | 31.82 | 80% | 268.7 | 10.51 |
| torch-cuda-bf16 | batch 32 | 1373 | 78.92 | 82.51 | 96% | 1080.7 | 13.69 |
| torch-cuda-bf16 | batch 64 | 1315 | 183.05 | 171.27 | 107% | 2185.3 | 11.94 |
| torch-cuda-fp16 | questions 1 | 1329 | 19.51 | 24.71 | 79% | 146.6 | 7.51 |
| torch-cuda-fp16 | questions 5 | 1343 | 55.82 | 60.55 | 92% | 749.8 | 13.43 |
| torch-cuda-fp16 | questions 10 | 1317 | 108.85 | 109.26 | 100% | 1508.1 | 13.85 |
| torch-cuda-fp16 | questions 50 | 1313 | 586.74 | 598.13 | 98% | 7540.6 | 12.85 |
| torch-cuda-fp16 | batch 1 | 1165 | 12.74 | 20.67 | 62% | 34.1 | 2.68 |
| torch-cuda-fp16 | batch 8 | 1285 | 25.02 | 31.35 | 80% | 268.7 | 10.74 |
| torch-cuda-fp16 | batch 32 | 1373 | 77.99 | 86.38 | 90% | 1080.7 | 13.86 |
| torch-cuda-fp16 | batch 64 | 1315 | 177.03 | 175.98 | 101% | 2185.3 | 12.34 |
| torch-cuda-fp32 | questions 1 | 975 | 39.13 | 37.91 | 103% | 146.6 | 3.75 |
| torch-cuda-fp32 | questions 5 | 980 | 154.93 | 167.22 | 93% | 749.8 | 4.84 |
| torch-cuda-fp32 | questions 10 | 980 | 325.36 | 309.95 | 105% | 1508.1 | 4.64 |
| torch-cuda-fp32 | questions 50 | 921 | 1647.84 | 1649.37 | 100% | 7540.6 | 4.58 |
| torch-cuda-fp32 | batch 1 | 867 | 10.07 | 20.53 | 49% | 34.1 | 3.39 |
| torch-cuda-fp32 | batch 8 | 982 | 63.48 | 59.23 | 107% | 268.7 | 4.23 |
| torch-cuda-fp32 | batch 32 | 981 | 230.52 | 226.34 | 102% | 1080.7 | 4.69 |
| torch-cuda-fp32 | batch 64 | 923 | 469.60 | 483.70 | 97% | 2185.3 | 4.65 |
| torch-cuda-graphs | questions 1 | 610 | 15.69 | 17.43 | 90% | 146.6 | 9.34 |
| torch-cuda-graphs | questions 5 | 552 | 50.94 | 52.22 | 98% | 749.8 | 14.72 |
| torch-cuda-graphs | questions 10 | 528 | 90.10 | 91.49 | 98% | 1508.1 | 16.74 |
| torch-cuda-graphs | questions 50 | 526 | 411.04 | 428.25 | 96% | 7540.6 | 18.35 |
| torch-cuda-graphs | batch 1 | 554 | 9.87 | 11.07 | 89% | 34.1 | 3.46 |
| torch-cuda-graphs | batch 8 | 554 | 20.05 | 22.40 | 90% | 268.7 | 13.40 |
| torch-cuda-graphs | batch 32 | 528 | 72.32 | 75.03 | 96% | 1080.7 | 14.94 |
| torch-cuda-graphs | batch 64 | 554 | 135.71 | 144.09 | 94% | 2185.3 | 16.10 |
| torch-cuda-tilelang | questions 1 | 313 | 11.08 | 12.34 | 90% | 146.6 | 13.23 |
| torch-cuda-tilelang | questions 5 | 313 | 57.95 | 57.82 | 100% | 749.8 | 12.94 |
| torch-cuda-tilelang | questions 10 | 313 | 114.95 | 118.34 | 97% | 1508.1 | 13.12 |
| torch-cuda-tilelang | questions 50 | 313 | 496.84 | 509.20 | 98% | 7540.6 | 15.18 |
| torch-cuda-tilelang | batch 1 | 313 | 5.47 | 6.50 | 84% | 34.1 | 6.24 |
| torch-cuda-tilelang | batch 8 | 313 | 14.18 | 16.04 | 88% | 268.7 | 18.95 |
| torch-cuda-tilelang | batch 32 | 313 | 71.29 | 77.65 | 92% | 1080.7 | 15.16 |
| torch-cuda-tilelang | batch 64 | 313 | 144.74 | 153.16 | 95% | 2185.3 | 15.10 |

### typed-decisions, torch-cuda-bf16: phase split (median ms)

| Workload | prepare | tokenize | collate | transfer_in | dispatch | encoder | head | transfer_out | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.01 | 0.46 | 0.10 | 0.15 | 0.07 | 22.19 | 2.63 | 0.13 | 0.08 | 25.82 | 24.60 |
| questions 5 | 0.02 | 0.96 | 0.24 | 0.15 | 0.07 | 51.24 | 8.68 | 0.17 | 0.19 | 61.72 | 61.20 |
| questions 10 | 0.02 | 1.63 | 0.42 | 0.17 | 0.08 | 94.85 | 11.45 | 0.17 | 0.27 | 109.06 | 109.98 |
| questions 50 | 0.06 | 6.41 | 1.74 | 0.38 | 0.10 | 549.06 | 32.23 | 0.19 | 0.92 | 591.09 | 589.63 |
| batch 1 | 0.01 | 0.27 | 0.08 | 0.14 | 0.06 | 19.35 | 2.61 | 0.13 | 0.08 | 22.73 | 22.83 |
| batch 8 | 0.01 | 0.70 | 0.24 | 0.15 | 0.06 | 26.43 | 3.18 | 0.15 | 0.25 | 31.17 | 31.82 |
| batch 32 | 0.01 | 2.15 | 0.73 | 0.16 | 0.07 | 73.53 | 6.00 | 0.16 | 0.72 | 83.53 | 82.51 |
| batch 64 | 0.01 | 4.09 | 1.39 | 0.21 | 0.08 | 154.19 | 9.41 | 0.16 | 1.37 | 170.92 | 171.27 |

### typed-decisions, torch-cuda-fp16: phase split (median ms)

| Workload | prepare | tokenize | collate | transfer_in | dispatch | encoder | head | transfer_out | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.01 | 0.45 | 0.09 | 0.14 | 0.06 | 22.11 | 2.56 | 0.13 | 0.08 | 25.64 | 24.71 |
| questions 5 | 0.02 | 0.95 | 0.25 | 0.15 | 0.06 | 54.47 | 6.38 | 0.16 | 0.18 | 62.62 | 60.55 |
| questions 10 | 0.02 | 1.72 | 0.44 | 0.18 | 0.07 | 103.61 | 9.61 | 0.17 | 0.29 | 116.12 | 109.26 |
| questions 50 | 0.06 | 6.38 | 1.72 | 0.35 | 0.09 | 554.85 | 32.17 | 0.20 | 0.92 | 596.75 | 598.13 |
| batch 1 | 0.01 | 0.27 | 0.08 | 0.14 | 0.06 | 17.93 | 2.44 | 0.12 | 0.08 | 21.13 | 20.67 |
| batch 8 | 0.01 | 0.86 | 0.31 | 0.18 | 0.07 | 27.03 | 3.26 | 0.16 | 0.29 | 32.17 | 31.35 |
| batch 32 | 0.01 | 2.19 | 0.75 | 0.17 | 0.07 | 77.74 | 6.28 | 0.17 | 0.73 | 88.09 | 86.38 |
| batch 64 | 0.01 | 4.05 | 1.37 | 0.26 | 0.08 | 157.54 | 10.04 | 0.16 | 1.39 | 174.89 | 175.98 |

### typed-decisions, torch-cuda-fp32: phase split (median ms)

| Workload | prepare | tokenize | collate | transfer_in | dispatch | encoder | head | transfer_out | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.01 | 0.46 | 0.10 | 0.15 | 0.04 | 34.67 | 2.76 | 0.13 | 0.09 | 38.41 | 37.91 |
| questions 5 | 0.02 | 1.09 | 0.27 | 0.17 | 0.05 | 151.64 | 11.01 | 0.14 | 0.21 | 164.59 | 167.22 |
| questions 10 | 0.02 | 1.73 | 0.41 | 0.17 | 0.06 | 284.15 | 20.82 | 0.19 | 0.30 | 307.84 | 309.95 |
| questions 50 | 0.06 | 6.23 | 1.71 | 0.39 | 0.07 | 1557.22 | 102.23 | 0.29 | 0.90 | 1669.09 | 1649.37 |
| batch 1 | 0.01 | 0.26 | 0.08 | 0.14 | 0.04 | 13.13 | 1.47 | 0.10 | 0.08 | 15.33 | 20.53 |
| batch 8 | 0.01 | 0.69 | 0.23 | 0.16 | 0.04 | 51.92 | 4.15 | 0.14 | 0.26 | 57.59 | 59.23 |
| batch 32 | 0.01 | 2.23 | 0.75 | 0.18 | 0.05 | 203.49 | 15.36 | 0.15 | 0.76 | 222.99 | 226.34 |
| batch 64 | 0.01 | 4.07 | 1.34 | 0.28 | 0.06 | 437.69 | 31.14 | 0.26 | 1.40 | 476.25 | 483.70 |

### typed-decisions, torch-cuda-graphs: phase split (median ms)

| Workload | prepare | tokenize | collate | transfer_in | forward | transfer_out | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.01 | 0.45 | 0.10 | 0.15 | 16.47 | 0.14 | 0.09 | 17.42 | 17.43 |
| questions 5 | 0.01 | 0.91 | 0.23 | 0.14 | 51.33 | 0.23 | 0.18 | 53.04 | 52.22 |
| questions 10 | 0.02 | 1.57 | 0.42 | 0.15 | 88.73 | 0.27 | 0.25 | 91.41 | 91.49 |
| questions 50 | 0.06 | 7.66 | 2.41 | 0.40 | 407.67 | 0.36 | 0.92 | 419.47 | 428.25 |
| batch 1 | 0.01 | 0.27 | 0.08 | 0.15 | 10.48 | 0.14 | 0.08 | 11.22 | 11.07 |
| batch 8 | 0.01 | 0.69 | 0.22 | 0.14 | 20.79 | 0.17 | 0.25 | 22.26 | 22.40 |
| batch 32 | 0.01 | 2.10 | 0.71 | 0.15 | 70.52 | 0.26 | 0.69 | 74.44 | 75.03 |
| batch 64 | 0.01 | 5.15 | 1.96 | 0.30 | 135.04 | 0.30 | 1.48 | 144.24 | 144.09 |

### typed-decisions, torch-cuda-tilelang: phase split (median ms)

| Workload | prepare | tokenize | collate | transfer_in | forward | transfer_out | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.01 | 0.41 | 0.09 | 0.14 | 11.45 | 0.18 | 0.09 | 12.37 | 12.34 |
| questions 5 | 0.02 | 1.14 | 0.28 | 0.16 | 58.69 | 0.32 | 0.22 | 60.82 | 57.82 |
| questions 10 | 0.02 | 1.61 | 0.41 | 0.16 | 119.36 | 0.29 | 0.27 | 122.13 | 118.34 |
| questions 50 | 0.06 | 6.28 | 1.72 | 0.38 | 499.65 | 0.40 | 0.93 | 509.42 | 509.20 |
| batch 1 | 0.01 | 0.25 | 0.08 | 0.15 | 5.80 | 0.15 | 0.08 | 6.53 | 6.50 |
| batch 8 | 0.01 | 0.66 | 0.25 | 0.15 | 14.53 | 0.16 | 0.25 | 16.02 | 16.04 |
| batch 32 | 0.01 | 2.18 | 0.72 | 0.18 | 71.62 | 0.27 | 0.70 | 75.68 | 77.65 |
| batch 64 | 0.01 | 3.99 | 1.33 | 0.28 | 146.16 | 0.29 | 1.34 | 153.40 | 153.16 |

### typed-decisions, ort-trt-fp16: phase split (median ms)

| Workload | prepare | tokenize | collate | model | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.01 | 0.41 | 0.02 | 10.25 | 0.09 | 10.77 | 11.43 |
| questions 5 | 0.01 | 0.86 | 0.04 | 33.52 | 0.21 | 34.64 | 34.54 |
| questions 10 | 0.01 | 1.49 | 0.06 | 65.23 | 0.32 | 67.11 | 68.29 |
| questions 50 | 0.05 | 5.91 | 0.26 | 387.59 | 1.01 | 394.83 | 392.00 |
| batch 1 | 0.01 | 0.26 | 0.01 | 8.19 | 0.09 | 8.55 | 8.65 |
| batch 8 | 0.01 | 0.81 | 0.04 | 12.76 | 0.28 | 13.89 | 13.76 |
| batch 32 | 0.01 | 2.04 | 0.08 | 46.60 | 0.67 | 49.40 | 52.50 |
| batch 64 | 0.01 | 4.00 | 0.16 | 93.07 | 1.31 | 98.54 | 104.63 |

### typed-decisions, torch-cpu: phase split (median ms)

| Workload | prepare | tokenize | collate | transfer_in | dispatch | encoder | head | transfer_out | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.02 | 1.37 | 0.26 | 0.02 | 0.09 | 342.49 | 70.42 | 0.04 | 0.21 | 414.93 | 411.25 |
| questions 5 | 0.03 | 3.31 | 0.87 | 0.02 | 0.10 | 1379.48 | 226.29 | 0.06 | 0.56 | 1610.72 | 1613.47 |
| questions 10 | 0.04 | 3.76 | 0.91 | 0.01 | 0.08 | 2824.32 | 378.34 | 0.04 | 0.49 | 3207.98 | 3189.86 |
| questions 50 | 0.09 | 10.06 | 2.81 | 0.01 | 0.08 | 13611.97 | 1592.70 | 0.03 | 1.28 | 15219.02 | 15210.47 |
| batch 1 | 0.01 | 0.70 | 0.19 | 0.02 | 0.08 | 88.06 | 8.93 | 0.03 | 0.17 | 98.21 | 98.76 |
| batch 8 | 0.02 | 2.42 | 0.98 | 0.02 | 0.09 | 606.53 | 53.99 | 0.05 | 0.84 | 664.94 | 660.05 |
| batch 32 | 0.02 | 6.81 | 3.15 | 0.02 | 0.09 | 2122.99 | 181.87 | 0.05 | 2.76 | 2317.76 | 2299.97 |
| batch 64 | 0.01 | 11.70 | 5.71 | 0.02 | 0.09 | 4243.10 | 361.26 | 0.05 | 5.36 | 4627.30 | 4629.07 |

### typed-decisions, ort-cpu: phase split (median ms)

| Workload | prepare | tokenize | collate | model | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.02 | 1.13 | 0.05 | 270.74 | 0.23 | 272.16 | 273.55 |
| questions 5 | 0.03 | 3.03 | 0.11 | 1425.46 | 0.64 | 1429.27 | 1409.23 |
| questions 10 | 0.05 | 4.97 | 0.17 | 2927.66 | 1.06 | 2933.90 | 2936.77 |
| questions 50 | 0.13 | 17.45 | 0.60 | 15082.11 | 3.99 | 15104.28 | 15008.15 |
| batch 1 | 0.02 | 0.73 | 0.04 | 80.48 | 0.23 | 81.51 | 81.64 |
| batch 8 | 0.02 | 2.25 | 0.09 | 484.26 | 0.86 | 487.48 | 489.01 |
| batch 32 | 0.02 | 7.07 | 0.22 | 2019.49 | 3.14 | 2029.93 | 2028.50 |
| batch 64 | 0.02 | 12.36 | 0.36 | 4313.68 | 5.37 | 4331.79 | 4327.93 |

### typed-decisions, ort-cuda: phase split (median ms)

| Workload | prepare | tokenize | collate | model | decode | split total | p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| questions 1 | 0.01 | 0.43 | 0.02 | 22.56 | 0.09 | 23.11 | 22.49 |
| questions 5 | 0.01 | 1.04 | 0.04 | 102.67 | 0.23 | 104.00 | 98.85 |
| questions 10 | 0.02 | 1.70 | 0.07 | 225.62 | 0.34 | 227.75 | 226.58 |
| questions 50 | 0.06 | 6.74 | 0.30 | 1281.94 | 1.07 | 1290.11 | 1288.45 |
| batch 1 | 0.01 | 0.28 | 0.01 | 13.30 | 0.09 | 13.70 | 14.64 |
| batch 8 | 0.01 | 0.80 | 0.04 | 34.12 | 0.31 | 35.27 | 35.44 |
| batch 32 | 0.01 | 2.19 | 0.08 | 154.53 | 0.71 | 157.53 | 161.05 |
| batch 64 | 0.01 | 3.99 | 0.15 | 336.81 | 1.29 | 342.25 | 351.05 |

## english against typed-decisions

| Backend | Workload | english p50 | typed-decisions p50 | Differs? |
| --- | --- | --- | --- | --- |
| torch-cuda-bf16 | questions 1 | 24.4 | 24.6 | same (ranges overlap) |
| torch-cuda-bf16 | questions 5 | 62.3 | 61.2 | same (ranges overlap) |
| torch-cuda-bf16 | questions 10 | 110.8 | 110.0 | same (ranges overlap) |
| torch-cuda-bf16 | questions 50 | 588.2 | 589.6 | same (ranges overlap) |
| torch-cuda-bf16 | batch 1 | 29.1 | 22.8 | same (ranges overlap) |
| torch-cuda-bf16 | batch 8 | 30.9 | 31.8 | same (ranges overlap) |
| torch-cuda-bf16 | batch 32 | 84.6 | 82.5 | same (ranges overlap) |
| torch-cuda-bf16 | batch 64 | 173.9 | 171.3 | same (ranges overlap) |
| torch-cuda-fp16 | questions 1 | 27.8 | 24.7 | same (ranges overlap) |
| torch-cuda-fp16 | questions 5 | 63.3 | 60.5 | same (ranges overlap) |
| torch-cuda-fp16 | questions 10 | 110.2 | 109.3 | same (ranges overlap) |
| torch-cuda-fp16 | questions 50 | 599.2 | 598.1 | same (ranges overlap) |
| torch-cuda-fp16 | batch 1 | 20.9 | 20.7 | same (ranges overlap) |
| torch-cuda-fp16 | batch 8 | 31.4 | 31.3 | same (ranges overlap) |
| torch-cuda-fp16 | batch 32 | 86.7 | 86.4 | same (ranges overlap) |
| torch-cuda-fp16 | batch 64 | 176.2 | 176.0 | same (ranges overlap) |
| torch-cuda-fp32 | questions 1 | 39.6 | 37.9 | same (ranges overlap) |
| torch-cuda-fp32 | questions 5 | 169.3 | 167.2 | same (ranges overlap) |
| torch-cuda-fp32 | questions 10 | 309.4 | 310.0 | same (ranges overlap) |
| torch-cuda-fp32 | questions 50 | 1651.6 | 1649.4 | same (ranges overlap) |
| torch-cuda-fp32 | batch 1 | 20.4 | 20.5 | same (ranges overlap) |
| torch-cuda-fp32 | batch 8 | 58.3 | 59.2 | same (ranges overlap) |
| torch-cuda-fp32 | batch 32 | 232.2 | 226.3 | same (ranges overlap) |
| torch-cuda-fp32 | batch 64 | 483.8 | 483.7 | same (ranges overlap) |
| torch-cuda-graphs | questions 1 | 18.2 | 17.4 | yes |
| torch-cuda-graphs | questions 5 | 51.6 | 52.2 | same (ranges overlap) |
| torch-cuda-graphs | questions 10 | 91.1 | 91.5 | same (ranges overlap) |
| torch-cuda-graphs | questions 50 | 430.0 | 428.2 | same (ranges overlap) |
| torch-cuda-graphs | batch 1 | 11.2 | 11.1 | yes |
| torch-cuda-graphs | batch 8 | 22.3 | 22.4 | same (ranges overlap) |
| torch-cuda-graphs | batch 32 | 74.8 | 75.0 | same (ranges overlap) |
| torch-cuda-graphs | batch 64 | 145.2 | 144.1 | same (ranges overlap) |
| torch-cuda-tilelang | questions 1 | 12.9 | 12.3 | yes |
| torch-cuda-tilelang | questions 5 | 59.3 | 57.8 | same (ranges overlap) |
| torch-cuda-tilelang | questions 10 | 121.5 | 118.3 | same (ranges overlap) |
| torch-cuda-tilelang | questions 50 | 507.7 | 509.2 | same (ranges overlap) |
| torch-cuda-tilelang | batch 1 | 6.4 | 6.5 | same (ranges overlap) |
| torch-cuda-tilelang | batch 8 | 16.3 | 16.0 | same (ranges overlap) |
| torch-cuda-tilelang | batch 32 | 76.4 | 77.6 | same (ranges overlap) |
| torch-cuda-tilelang | batch 64 | 152.9 | 153.2 | same (ranges overlap) |
| ort-trt-fp16 | questions 1 | 10.8 | 11.4 | yes |
| ort-trt-fp16 | questions 5 | 36.2 | 34.5 | yes |
| ort-trt-fp16 | questions 10 | 68.6 | 68.3 | same (ranges overlap) |
| ort-trt-fp16 | questions 50 | 397.0 | 392.0 | same (ranges overlap) |
| ort-trt-fp16 | batch 1 | 8.5 | 8.7 | same (ranges overlap) |
| ort-trt-fp16 | batch 8 | 14.1 | 13.8 | same (ranges overlap) |
| ort-trt-fp16 | batch 32 | 53.5 | 52.5 | same (ranges overlap) |
| ort-trt-fp16 | batch 64 | 105.9 | 104.6 | same (ranges overlap) |
| torch-cpu | questions 1 | 400.0 | 411.2 | yes |
| torch-cpu | questions 5 | 1673.2 | 1613.5 | yes |
| torch-cpu | questions 10 | 3344.8 | 3189.9 | yes |
| torch-cpu | questions 50 | 14995.6 | 15210.5 | yes |
| torch-cpu | batch 1 | 97.2 | 98.8 | same (ranges overlap) |
| torch-cpu | batch 8 | 651.2 | 660.0 | same (ranges overlap) |
| torch-cpu | batch 32 | 2320.1 | 2300.0 | same (ranges overlap) |
| torch-cpu | batch 64 | 4685.0 | 4629.1 | yes |
| ort-cpu | questions 1 | 275.7 | 273.5 | same (ranges overlap) |
| ort-cpu | questions 5 | 1422.8 | 1409.2 | same (ranges overlap) |
| ort-cpu | questions 10 | 2938.7 | 2936.8 | same (ranges overlap) |
| ort-cpu | questions 50 | 14885.8 | 15008.2 | same (ranges overlap) |
| ort-cpu | batch 1 | 81.8 | 81.6 | same (ranges overlap) |
| ort-cpu | batch 8 | 493.5 | 489.0 | yes |
| ort-cpu | batch 32 | 2044.9 | 2028.5 | same (ranges overlap) |
| ort-cpu | batch 64 | 4312.5 | 4327.9 | same (ranges overlap) |
| ort-cuda | questions 1 | 22.7 | 22.5 | same (ranges overlap) |
| ort-cuda | questions 5 | 95.2 | 98.9 | yes |
| ort-cuda | questions 10 | 222.7 | 226.6 | yes |
| ort-cuda | questions 50 | 1281.1 | 1288.4 | same (ranges overlap) |
| ort-cuda | batch 1 | 13.7 | 14.6 | yes |
| ort-cuda | batch 8 | 35.0 | 35.4 | same (ranges overlap) |
| ort-cuda | batch 32 | 160.7 | 161.1 | same (ranges overlap) |
| ort-cuda | batch 64 | 350.5 | 351.0 | same (ranges overlap) |

## Answers and failures

Rows measured: 144 of 144.
Largest probability difference from FP32 PyTorch on the CPU: 1.2e-02 (english, torch-cuda-bf16, batch 64).
Rows where any answer picked a different option: 0.
