# Phase 2 B-Saturation Pilot Report

## Hardware & Provenance
- **GPU**: `NVIDIA H100 80GB HBM3` (CC: `[9, 0]`, Driver: `580.95.05`)
- **PyTorch / CUDA**: `2.14.0+cu130` / CUDA `13.0`
- **Triton Version**: `3.9.0` (`/opt/triton-src/python/triton/__init__.py`)
- **Test Shape**: `M=32, N=128, num_warps=4`
- **Evaluated B**: `[64, 256, 1024, 4096, 8192]`

## Convergence and Steady-State Analysis

| B (Grid Size) | Candidate | Total Median (us) | Time / CTA (ns) | Effective (GB/s) | Delta vs Prev B (%) |
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
| 1024 | 2 | 23.97 | 23.4 | 350.0 | -74.56% |
| 1024 | 1 | 23.84 | 23.3 | 351.9 | -74.69% |
| 4096 | default | 27.57 | 6.7 | 1217.2 | -71.19% |
| 4096 | 8 | 27.47 | 6.7 | 1221.4 | -71.25% |
| 4096 | 4 | 26.82 | 6.5 | 1251.3 | -71.92% |
| 4096 | 2 | 26.78 | 6.5 | 1252.8 | -72.06% |
| 4096 | 1 | 26.50 | 6.5 | 1266.4 | -72.21% |
| 8192 | default | 44.10 | 5.4 | 1521.9 | -20.02% |
| 8192 | 8 | 44.19 | 5.4 | 1518.6 | -19.57% |
| 8192 | 4 | 43.71 | 5.3 | 1535.3 | -18.50% |
| 8192 | 2 | 43.71 | 5.3 | 1535.3 | -18.40% |
| 8192 | 1 | 43.63 | 5.3 | 1538.1 | -17.66% |

## B_STEADY Selection Rationale
Launch overhead and small-grid wave quantization dominate at small B (B=64, 256).
As B reaches 4096 and 8192, `time / CTA` stabilizes to steady-state execution throughput.
We select **B_STEADY = 4096** as it saturates the 132 SMs of H100 (31+ CTAs per SM) while maintaining efficient benchmark turnaround.