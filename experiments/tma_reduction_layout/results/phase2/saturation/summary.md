# Phase 2 B-Saturation Pilot Report

## Hardware & Provenance
- **GPU**: `NVIDIA H100 80GB HBM3` (CC: `[9, 0]`, Driver: `580.95.05`)
- **PyTorch / CUDA**: `2.14.0+cu130` / CUDA `13.0`
- **Triton Version**: `3.9.0` (`/opt/triton-src/python/triton/__init__.py`)
- **SM Count**: `132`
- **Test Shape**: `M=32, N=128, num_warps=4`
- **Evaluated B**: `[64, 256, 1024, 4096, 8192]`

> [!NOTE]
> `amortized_grid_time_per_cta_ns` represents total grid execution time divided by CTA count.
> It is a throughput-normalization metric, not the execution latency of one CTA.

## Convergence and Grid Amortization Analysis

| B (Grid Size) | Candidate | Total Median (us) | Amortized Time / CTA (ns) | Effective (GB/s) | Delta vs Prev B (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 64 | default | 23.65 | 369.5 | 22.2 | baseline |
| 64 | 8 | 23.87 | 373.0 | 22.0 | baseline |
| 64 | 4 | 23.78 | 371.5 | 22.1 | baseline |
| 64 | 2 | 23.68 | 370.0 | 22.1 | baseline |
| 64 | 1 | 23.70 | 370.2 | 22.1 | baseline |
| 256 | default | 23.74 | 92.8 | 88.3 | -74.90% |
| 256 | 8 | 23.60 | 92.2 | 88.9 | -75.28% |
| 256 | 4 | 23.52 | 91.9 | 89.2 | -75.27% |
| 256 | 2 | 23.55 | 92.0 | 89.0 | -75.14% |
| 256 | 1 | 23.55 | 92.0 | 89.0 | -75.15% |
| 1024 | default | 23.92 | 23.4 | 350.7 | -74.81% |
| 1024 | 8 | 23.89 | 23.3 | 351.2 | -74.69% |
| 1024 | 4 | 23.87 | 23.3 | 351.4 | -74.63% |
| 1024 | 2 | 23.97 | 23.4 | 350.0 | -74.55% |
| 1024 | 1 | 23.84 | 23.3 | 351.9 | -74.70% |
| 4096 | default | 27.57 | 6.7 | 1217.2 | -71.19% |
| 4096 | 8 | 27.47 | 6.7 | 1221.4 | -71.24% |
| 4096 | 4 | 26.82 | 6.5 | 1251.3 | -71.90% |
| 4096 | 2 | 26.78 | 6.5 | 1252.8 | -72.06% |
| 4096 | 1 | 26.50 | 6.5 | 1266.4 | -72.21% |
| 8192 | default | 44.10 | 5.4 | 1521.9 | -20.06% |
| 8192 | 8 | 44.19 | 5.4 | 1518.6 | -19.67% |
| 8192 | 4 | 43.71 | 5.3 | 1535.3 | -18.47% |
| 8192 | 2 | 43.71 | 5.3 | 1535.3 | -18.35% |
| 8192 | 1 | 43.63 | 5.3 | 1538.1 | -17.62% |

## Plateau Evaluation
At small B (B <= 1024), the measured GPU kernel duration remains nearly flat (~23.6-23.9 us),
which is consistent with insufficient grid-level work to expose steady-state throughput
and/or fixed device-side kernel costs. The current experiment does not isolate the source of that fixed cost.

From B=4096 (6.7 ns/CTA) to B=8192 (5.4 ns/CTA), normalized time/CTA decreased by ~20%.
Therefore, B=4096 did not establish a throughput plateau.
An extended saturation pilot across larger B is required to identify a true plateau.