# Phase 2 Extended B-Saturation Pilot Report

## Hardware & Execution Environment
- **GPU**: `NVIDIA H100 80GB HBM3` (CC: `[9, 0]`, Driver: `580.95.05`)
- **PyTorch / CUDA**: `2.14.0+cu130` / CUDA `13.0`
- **Triton Version**: `3.9.0` (`/opt/triton-src/python/triton/__init__.py`)
- **SM Count**: `132`
- **Evaluated B (Grid Sizes)**: `[4096, 8192, 16384, 32768, 65536]`
- **Operational Plateau Criterion**: Two consecutive B doublings where change in `amortized_grid_time_per_cta_ns` is `< 5.0%` (`| (t_{2B} - t_B) / t_B | < 0.05`).

> [!NOTE]
> `amortized_grid_time_per_cta_ns` represents total grid execution time divided by CTA count.
> It is a throughput-normalization metric, not the execution latency of one CTA.

## 1. Per-Configuration Saturation Analysis

### Configuration: `M32_N16_w8` (`M=32, N=16, num_warps=8`)

| B (Grid) | Candidate | Total Median (us) | Amortized Time/CTA (ns) | Effective GB/s | vs Default (%) | Step Delta vs Prev B (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 4096 | default | 33.22 | 8.11 | 126.3 | 0.00% (base) | baseline |
| 4096 | 8 | INVALID | - | - | - | forcedVec 8 > numElemsPerThread (2) |
| 4096 | 4 | INVALID | - | - | - | forcedVec 4 > numElemsPerThread (2) |
| 4096 | 2 | 31.02 | 7.57 | 135.2 | -6.60% | baseline |
| 4096 | 1 | 31.10 | 7.59 | 134.8 | -6.36% | baseline |
| 8192 | default | 34.26 | 4.18 | 244.9 | 0.00% (base) | -48.46% |
| 8192 | 8 | INVALID | - | - | - | forcedVec 8 > numElemsPerThread (2) |
| 8192 | 4 | INVALID | - | - | - | forcedVec 4 > numElemsPerThread (2) |
| 8192 | 2 | 37.65 | 4.60 | 222.8 | +9.90% | -39.23% |
| 8192 | 1 | 32.13 | 3.92 | 261.1 | -6.21% | -48.35% |
| 16384 | default | 46.10 | 2.81 | 364.0 | 0.00% (base) | -32.78% |
| 16384 | 8 | INVALID | - | - | - | forcedVec 8 > numElemsPerThread (2) |
| 16384 | 4 | INVALID | - | - | - | forcedVec 4 > numElemsPerThread (2) |
| 16384 | 2 | 46.06 | 2.81 | 364.2 | -0.07% | -38.91% |
| 16384 | 1 | 45.04 | 2.75 | 372.5 | -2.29% | -29.85% |
| 32768 | default | 84.42 | 2.58 | 397.5 | 0.00% (base) | -8.19% |
| 32768 | 8 | INVALID | - | - | - | forcedVec 8 > numElemsPerThread (2) |
| 32768 | 4 | INVALID | - | - | - | forcedVec 4 > numElemsPerThread (2) |
| 32768 | 2 | 84.13 | 2.57 | 398.9 | -0.34% | -8.54% |
| 32768 | 1 | 81.68 | 2.49 | 410.8 | -3.24% | -9.45% |
| 65536 | default | 158.80 | 2.42 | 422.6 | 0.00% (base) | -6.20% |
| 65536 | 8 | INVALID | - | - | - | forcedVec 8 > numElemsPerThread (2) |
| 65536 | 4 | INVALID | - | - | - | forcedVec 4 > numElemsPerThread (2) |
| 65536 | 2 | 158.83 | 2.42 | 422.5 | +0.02% | -5.84% |
| 65536 | 1 | 155.15 | 2.37 | 432.5 | -2.30% | -4.82% |

#### Doubling Steps & Operational Criterion Evaluation (`default` candidate):
- Doubling Step 1 (`B=4096 -> 8192`): `-48.46%` (exceeds 5%)
- Doubling Step 2 (`B=8192 -> 16384`): `-32.78%` (exceeds 5%)
- Doubling Step 3 (`B=16384 -> 32768`): `-8.19%` (exceeds 5%)
- Doubling Step 4 (`B=32768 -> 65536`): `-6.20%` (exceeds 5%)

> **Operational Criterion NOT Met**: No two consecutive doubling steps exhibited `< 5.0%` change within tested B range.

### Configuration: `M32_N64_w8` (`M=32, N=64, num_warps=8`)

| B (Grid) | Candidate | Total Median (us) | Amortized Time/CTA (ns) | Effective GB/s | vs Default (%) | Step Delta vs Prev B (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 4096 | default | 32.02 | 7.82 | 524.0 | 0.00% (base) | baseline |
| 4096 | 8 | 32.19 | 7.86 | 521.2 | +0.55% | baseline |
| 4096 | 4 | 25.84 | 6.31 | 649.3 | -19.29% | baseline |
| 4096 | 2 | 25.04 | 6.11 | 670.0 | -21.79% | baseline |
| 4096 | 1 | 24.93 | 6.09 | 673.0 | -22.14% | baseline |
| 8192 | default | 46.91 | 5.73 | 715.3 | 0.00% (base) | -26.73% |
| 8192 | 8 | 47.26 | 5.77 | 709.9 | +0.75% | -26.59% |
| 8192 | 4 | 35.79 | 4.37 | 937.5 | -23.70% | -30.74% |
| 8192 | 2 | 33.31 | 4.07 | 1007.3 | -28.99% | -33.39% |
| 8192 | 1 | 32.94 | 4.02 | 1018.5 | -29.77% | -33.99% |
| 16384 | default | 81.90 | 5.00 | 819.4 | 0.00% (base) | -12.74% |
| 16384 | 8 | 82.43 | 5.03 | 814.1 | +0.64% | -12.82% |
| 16384 | 4 | 58.59 | 3.58 | 1145.4 | -28.46% | -18.08% |
| 16384 | 2 | 55.10 | 3.36 | 1217.9 | -32.72% | -17.44% |
| 16384 | 1 | 54.77 | 3.34 | 1225.3 | -33.13% | -16.92% |
| 32768 | default | 147.42 | 4.50 | 910.4 | 0.00% (base) | -10.00% |
| 32768 | 8 | 147.31 | 4.50 | 911.1 | -0.08% | -10.54% |
| 32768 | 4 | 99.49 | 3.04 | 1349.1 | -32.52% | -15.08% |
| 32768 | 2 | 92.72 | 2.83 | 1447.6 | -37.11% | -15.77% |
| 32768 | 1 | 91.44 | 2.79 | 1467.8 | -37.97% | -16.47% |
| 65536 | default | 273.92 | 4.18 | 980.0 | 0.00% (base) | -7.11% |
| 65536 | 8 | 275.18 | 4.20 | 975.5 | +0.46% | -6.67% |
| 65536 | 4 | 179.82 | 2.74 | 1492.8 | -34.35% | -9.87% |
| 65536 | 2 | 166.02 | 2.53 | 1616.9 | -39.39% | -10.60% |
| 65536 | 1 | 164.42 | 2.51 | 1632.7 | -39.98% | -10.04% |

#### Doubling Steps & Operational Criterion Evaluation (`default` candidate):
- Doubling Step 1 (`B=4096 -> 8192`): `-26.73%` (exceeds 5%)
- Doubling Step 2 (`B=8192 -> 16384`): `-12.74%` (exceeds 5%)
- Doubling Step 3 (`B=16384 -> 32768`): `-10.00%` (exceeds 5%)
- Doubling Step 4 (`B=32768 -> 65536`): `-7.11%` (exceeds 5%)

> **Operational Criterion NOT Met**: No two consecutive doubling steps exhibited `< 5.0%` change within tested B range.

### Configuration: `M32_N128_w4` (`M=32, N=128, num_warps=4`)

| B (Grid) | Candidate | Total Median (us) | Amortized Time/CTA (ns) | Effective GB/s | vs Default (%) | Step Delta vs Prev B (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 4096 | default | 27.12 | 6.62 | 1237.3 | 0.00% (base) | baseline |
| 4096 | 8 | 27.07 | 6.61 | 1239.5 | -0.18% | baseline |
| 4096 | 4 | 26.18 | 6.39 | 1281.9 | -3.48% | baseline |
| 4096 | 2 | 26.05 | 6.36 | 1288.2 | -3.95% | baseline |
| 4096 | 1 | 25.89 | 6.32 | 1296.1 | -4.54% | baseline |
| 8192 | default | 42.66 | 5.21 | 1573.3 | 0.00% (base) | -21.30% |
| 8192 | 8 | 42.61 | 5.20 | 1575.0 | -0.11% | -21.33% |
| 8192 | 4 | 41.98 | 5.12 | 1598.4 | -1.58% | -19.87% |
| 8192 | 2 | 42.13 | 5.14 | 1593.0 | -1.24% | -19.18% |
| 8192 | 1 | 41.97 | 5.12 | 1599.0 | -1.61% | -18.99% |
| 16384 | default | 66.62 | 4.07 | 2014.6 | 0.00% (base) | -21.88% |
| 16384 | 8 | 66.96 | 4.09 | 2004.5 | +0.50% | -21.35% |
| 16384 | 4 | 66.10 | 4.03 | 2030.7 | -0.79% | -21.29% |
| 16384 | 2 | 66.03 | 4.03 | 2032.6 | -0.89% | -21.60% |
| 16384 | 1 | 66.22 | 4.04 | 2026.7 | -0.60% | -21.09% |
| 32768 | default | 114.67 | 3.50 | 2340.9 | 0.00% (base) | -14.00% |
| 32768 | 8 | 114.38 | 3.49 | 2346.8 | -0.25% | -14.67% |
| 32768 | 4 | 114.02 | 3.48 | 2354.4 | -0.57% | -13.65% |
| 32768 | 2 | 113.74 | 3.47 | 2360.0 | -0.81% | -13.90% |
| 32768 | 1 | 113.68 | 3.47 | 2361.3 | -0.87% | -14.11% |
| 65536 | default | 212.67 | 3.25 | 2524.4 | 0.00% (base) | -7.14% |
| 65536 | 8 | 212.05 | 3.24 | 2531.8 | -0.29% | -7.16% |
| 65536 | 4 | 211.58 | 3.23 | 2537.4 | -0.51% | -7.18% |
| 65536 | 2 | 210.75 | 3.22 | 2547.4 | -0.90% | -7.20% |
| 65536 | 1 | 211.63 | 3.23 | 2536.8 | -0.49% | -6.92% |

#### Doubling Steps & Operational Criterion Evaluation (`default` candidate):
- Doubling Step 1 (`B=4096 -> 8192`): `-21.30%` (exceeds 5%)
- Doubling Step 2 (`B=8192 -> 16384`): `-21.88%` (exceeds 5%)
- Doubling Step 3 (`B=16384 -> 32768`): `-14.00%` (exceeds 5%)
- Doubling Step 4 (`B=32768 -> 65536`): `-7.14%` (exceeds 5%)

> **Operational Criterion NOT Met**: No two consecutive doubling steps exhibited `< 5.0%` change within tested B range.

## 2. Cross-Configuration Plateau Synthesis

| Configuration | Operational Criterion Met? | Plateau B | Status |
| :--- | :---: | :---: | :--- |
| `M32_N16_w8` | NO | None (<=65536) | Throughput not fully saturated |
| `M32_N64_w8` | NO | None (<=65536) | Throughput not fully saturated |
| `M32_N128_w4` | NO | None (<=65536) | Throughput not fully saturated |