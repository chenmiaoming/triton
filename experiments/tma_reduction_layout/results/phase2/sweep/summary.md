# Phase 2 Multi-CTA Steady-State TMA Reduction Layout Sweep Report

## Execution Environment
- **GPU**: `NVIDIA H100 80GB HBM3` (CC: `[9, 0]`, Driver: `580.95.05`)
- **Grid Configuration**: `grid = (B,) = (4096,)`, each CTA processes `[1, M, N]` tile
- **Reduction**: BF16 -> FP32 `tl.max(axis=1)` -> write `[B, N]`
- **Total Configurations**: 30 shape/warp tuples x 5 candidates = 150 combinations

## 1. Full Structural & Performance Table

| M | N | Warps | Cand | Legal | LocalLoad | lanePart[M] | warpPart[M] | Regs | Median (us) | Time/CTA (ns) | vs Default | Transitions | Repeat Status |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| 32 | 16 | 4 | default | YES | `1x ld.shared.v4.b16` | 8 | 4 | 21 | 43.22 | 10.6 | 0.00% (base) | BASELINE | - |
| 32 | 16 | 4 | 8 | NO | - | - | - | - | - | - | - | INVALID: forcedVec 8 > numElemsPerThread (4) | - |
| 32 | 16 | 4 | 4 | YES | `1x ld.shared.v4.b16` | 8 | 4 | 21 | 42.83 | 10.5 | -0.89% |  | - |
| 32 | 16 | 4 | 2 | YES | `1x ldmatrix.sync.aligned.m8n8.x2.shared.b16` | 4 | 4 | 22 | 42.61 | 10.4 | -1.41% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(3->1) | - |
| 32 | 16 | 4 | 1 | YES | `4x ld.shared.b16` | 2 | 4 | 21 | 42.86 | 10.5 | -0.81% | T1:lanePart_change | - |
| 32 | 16 | 8 | default | YES | `1x ldmatrix.sync.aligned.m8n8.x1.shared.b16` | 4 | 8 | 22 | 45.09 | 11.0 | 0.00% (base) | BASELINE | - |
| 32 | 16 | 8 | 8 | NO | - | - | - | - | - | - | - | INVALID: forcedVec 8 > numElemsPerThread (2) | - |
| 32 | 16 | 8 | 4 | NO | - | - | - | - | - | - | - | INVALID: forcedVec 4 > numElemsPerThread (2) | - |
| 32 | 16 | 8 | 2 | YES | `1x ldmatrix.sync.aligned.m8n8.x1.shared.b16` | 4 | 8 | 22 | 45.14 | 11.0 | +0.11% |  | - |
| 32 | 16 | 8 | 1 | YES | `2x ld.shared.b16` | 2 | 8 | 21 | 44.70 | 10.9 | -0.85% | T1:lanePart_change<br>T4:ldmatrix_transition | - |
| 32 | 32 | 4 | default | YES | `1x ld.shared.v4.b32` | 8 | 4 | 29 | 45.06 | 11.0 | 0.00% (base) | BASELINE | - |
| 32 | 32 | 4 | 8 | YES | `1x ld.shared.v4.b32` | 8 | 4 | 29 | 45.14 | 11.0 | +0.18% |  | - |
| 32 | 32 | 4 | 4 | YES | `2x ld.shared.v4.b16` | 4 | 4 | 20 | 44.90 | 11.0 | -0.36% | T1:lanePart_change<br>T6:st_shared_traffic(5->3) | - |
| 32 | 32 | 4 | 2 | YES | `1x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 2 | 4 | 22 | 45.22 | 11.0 | +0.36% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(5->3) | - |
| 32 | 32 | 4 | 1 | YES | `8x ld.shared.b16` | 1 | 4 | 22 | 44.99 | 11.0 | -0.14% | T1:lanePart_change<br>T6:st_shared_traffic(5->3) | - |
| 32 | 32 | 8 | default | YES | `1x ld.shared.v4.b16` | 4 | 8 | 20 | 46.74 | 11.4 | 0.00% (base) | BASELINE | - |
| 32 | 32 | 8 | 8 | NO | - | - | - | - | - | - | - | INVALID: forcedVec 8 > numElemsPerThread (4) | - |
| 32 | 32 | 8 | 4 | YES | `1x ld.shared.v4.b16` | 4 | 8 | 20 | 46.91 | 11.5 | +0.38% |  | - |
| 32 | 32 | 8 | 2 | YES | `1x ldmatrix.sync.aligned.m8n8.x2.shared.b16` | 2 | 8 | 20 | 46.58 | 11.4 | -0.34% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(4->3) | - |
| 32 | 32 | 8 | 1 | YES | `4x ld.shared.b16` | 1 | 8 | 22 | 46.61 | 11.4 | -0.27% | T1:lanePart_change<br>T6:st_shared_traffic(4->3) | - |
| 32 | 64 | 4 | default | YES | `2x ld.shared.v4.b32` | 4 | 4 | 28 | 43.73 | 10.7 | 0.00% (base) | BASELINE | - |
| 32 | 64 | 4 | 8 | YES | `2x ld.shared.v4.b32` | 4 | 4 | 28 | 43.70 | 10.7 | -0.07% |  | - |
| 32 | 64 | 4 | 4 | YES | `4x ld.shared.v2.b32` | 2 | 4 | 22 | 43.66 | 10.7 | -0.15% | T1:lanePart_change<br>T6:st_shared_traffic(5->4) | - |
| 32 | 64 | 4 | 2 | YES | `2x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 4 | 22 | 43.58 | 10.6 | -0.33% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(5->3) | - |
| 32 | 64 | 4 | 1 | YES | `16x ld.shared.b16` | 1 | 2 | 26 | 43.62 | 10.6 | -0.26% | T2:warpPart_decrease<br>T6:st_shared_traffic(5->3) | - |
| 32 | 64 | 8 | default | YES | `1x ld.shared.v4.b32` | 4 | 8 | 29 | 45.47 | 11.1 | 0.00% (base) | BASELINE | - |
| 32 | 64 | 8 | 8 | YES | `1x ld.shared.v4.b32` | 4 | 8 | 29 | 45.46 | 11.1 | -0.04% |  | - |
| 32 | 64 | 8 | 4 | YES | `2x ld.shared.v2.b32` | 2 | 8 | 22 | 42.80 | 10.4 | -5.88% | T1:lanePart_change<br>T6:st_shared_traffic(7->4) | reproduced |
| 32 | 64 | 8 | 2 | YES | `1x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 8 | 21 | 42.18 | 10.3 | -7.25% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(7->3) | reproduced |
| 32 | 64 | 8 | 1 | YES | `8x ld.shared.b16` | 1 | 4 | 21 | 42.05 | 10.3 | -7.53% | T2:warpPart_decrease<br>T6:st_shared_traffic(7->3) | reproduced |
| 32 | 128 | 4 | default | YES | `4x ld.shared.v4.b32` | 2 | 4 | 32 | 44.66 | 10.9 | 0.00% (base) | BASELINE | - |
| 32 | 128 | 4 | 8 | YES | `4x ld.shared.v4.b32` | 2 | 4 | 32 | 44.58 | 10.9 | -0.18% |  | - |
| 32 | 128 | 4 | 4 | YES | `8x ld.shared.v2.b32` | 1 | 4 | 25 | 43.79 | 10.7 | -1.93% | T1:lanePart_change<br>T6:st_shared_traffic(7->4) | - |
| 32 | 128 | 4 | 2 | YES | `4x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 2 | 23 | 43.89 | 10.7 | -1.72% | T2:warpPart_decrease<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(7->5) | - |
| 32 | 128 | 4 | 1 | YES | `32x ld.shared.b16` | 1 | 1 | 32 | 43.54 | 10.6 | -2.51% | T3:warpPart_eq_1<br>T6:st_shared_traffic(7->1) | - |
| 32 | 128 | 8 | default | YES | `2x ld.shared.v4.b32` | 2 | 8 | 32 | 47.10 | 11.5 | 0.00% (base) | BASELINE | - |
| 32 | 128 | 8 | 8 | YES | `2x ld.shared.v4.b32` | 2 | 8 | 32 | 47.30 | 11.5 | +0.41% |  | - |
| 32 | 128 | 8 | 4 | YES | `4x ld.shared.v2.b32` | 1 | 8 | 21 | 44.96 | 11.0 | -4.55% | T1:lanePart_change<br>T6:st_shared_traffic(7->4) | reproduced |
| 32 | 128 | 8 | 2 | YES | `2x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 4 | 22 | 45.17 | 11.0 | -4.11% | T2:warpPart_decrease<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(7->5) | reproduced |
| 32 | 128 | 8 | 1 | YES | `16x ld.shared.b16` | 1 | 2 | 25 | 45.01 | 11.0 | -4.45% | T2:warpPart_decrease<br>T6:st_shared_traffic(7->3) | reproduced |
| 32 | 256 | 4 | default | YES | `8x ld.shared.v4.b32` | 1 | 4 | 32 | 58.14 | 14.2 | 0.00% (base) | BASELINE | - |
| 32 | 256 | 4 | 8 | YES | `8x ld.shared.v4.b32` | 1 | 4 | 32 | 57.98 | 14.2 | -0.28% |  | - |
| 32 | 256 | 4 | 4 | YES | `16x ld.shared.v2.b32` | 1 | 2 | 39 | 57.46 | 14.0 | -1.18% | T2:warpPart_decrease<br>T6:st_shared_traffic(7->5) | - |
| 32 | 256 | 4 | 2 | YES | `4x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 1 | 32 | 57.52 | 14.0 | -1.07% | T3:warpPart_eq_1<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(7->1) | - |
| 32 | 256 | 4 | 1 | YES | `64x ld.shared.b16` | 1 | 1 | 37 | 57.65 | 14.1 | -0.85% | T3:warpPart_eq_1<br>T6:st_shared_traffic(7->3) | - |
| 32 | 256 | 8 | default | YES | `4x ld.shared.v4.b32` | 1 | 8 | 32 | 56.88 | 13.9 | 0.00% (base) | BASELINE | - |
| 32 | 256 | 8 | 8 | YES | `4x ld.shared.v4.b32` | 1 | 8 | 32 | 56.85 | 13.9 | -0.06% |  | - |
| 32 | 256 | 8 | 4 | YES | `8x ld.shared.v2.b32` | 1 | 4 | 24 | 56.43 | 13.8 | -0.79% | T2:warpPart_decrease<br>T6:st_shared_traffic(7->4) | - |
| 32 | 256 | 8 | 2 | YES | `4x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 2 | 23 | 55.65 | 13.6 | -2.17% | T2:warpPart_decrease<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(7->5) | - |
| 32 | 256 | 8 | 1 | YES | `32x ld.shared.b16` | 1 | 1 | 32 | 55.79 | 13.6 | -1.91% | T3:warpPart_eq_1<br>T6:st_shared_traffic(7->1) | - |
| 64 | 16 | 4 | default | YES | `1x ld.shared.v4.b32` | 16 | 4 | 31 | 45.12 | 11.0 | 0.00% (base) | BASELINE | - |
| 64 | 16 | 4 | 8 | YES | `1x ld.shared.v4.b32` | 16 | 4 | 31 | 44.78 | 10.9 | -0.74% |  | - |
| 64 | 16 | 4 | 4 | YES | `2x ld.shared.v4.b16` | 8 | 4 | 21 | 45.12 | 11.0 | +0.00% | T1:lanePart_change<br>T6:st_shared_traffic(5->3) | - |
| 64 | 16 | 4 | 2 | YES | `1x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 4 | 4 | 22 | 44.83 | 10.9 | -0.64% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(5->1) | - |
| 64 | 16 | 4 | 1 | YES | `8x ld.shared.b16` | 2 | 4 | 21 | 44.58 | 10.9 | -1.21% | T1:lanePart_change<br>T6:st_shared_traffic(5->3) | - |
| 64 | 16 | 8 | default | YES | `1x ld.shared.v4.b16` | 8 | 8 | 21 | 43.23 | 10.6 | 0.00% (base) | BASELINE | - |
| 64 | 16 | 8 | 8 | NO | - | - | - | - | - | - | - | INVALID: forcedVec 8 > numElemsPerThread (4) | - |
| 64 | 16 | 8 | 4 | YES | `1x ld.shared.v4.b16` | 8 | 8 | 21 | 43.18 | 10.5 | -0.11% |  | - |
| 64 | 16 | 8 | 2 | YES | `1x ldmatrix.sync.aligned.m8n8.x2.shared.b16` | 4 | 8 | 22 | 43.02 | 10.5 | -0.48% | T1:lanePart_change<br>T4:ldmatrix_transition | - |
| 64 | 16 | 8 | 1 | YES | `4x ld.shared.b16` | 2 | 8 | 21 | 42.80 | 10.4 | -1.00% | T1:lanePart_change | - |
| 64 | 32 | 4 | default | YES | `2x ld.shared.v4.b32` | 8 | 4 | 29 | 42.02 | 10.3 | 0.00% (base) | BASELINE | - |
| 64 | 32 | 4 | 8 | YES | `2x ld.shared.v4.b32` | 8 | 4 | 29 | 42.22 | 10.3 | +0.50% |  | - |
| 64 | 32 | 4 | 4 | YES | `4x ld.shared.v4.b16` | 4 | 4 | 22 | 42.11 | 10.3 | +0.23% | T1:lanePart_change<br>T6:st_shared_traffic(5->3) | - |
| 64 | 32 | 4 | 2 | YES | `2x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 2 | 4 | 21 | 42.19 | 10.3 | +0.42% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(5->3) | - |
| 64 | 32 | 4 | 1 | YES | `16x ld.shared.b16` | 1 | 4 | 23 | 42.30 | 10.3 | +0.69% | T1:lanePart_change<br>T6:st_shared_traffic(5->3) | - |
| 64 | 32 | 8 | default | YES | `1x ld.shared.v4.b32` | 8 | 8 | 29 | 46.66 | 11.4 | 0.00% (base) | BASELINE | - |
| 64 | 32 | 8 | 8 | YES | `1x ld.shared.v4.b32` | 8 | 8 | 29 | 46.66 | 11.4 | +0.00% |  | - |
| 64 | 32 | 8 | 4 | YES | `2x ld.shared.v4.b16` | 4 | 8 | 22 | 45.63 | 11.1 | -2.19% | T1:lanePart_change<br>T6:st_shared_traffic(5->4) | - |
| 64 | 32 | 8 | 2 | YES | `1x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 2 | 8 | 21 | 45.76 | 11.2 | -1.92% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(5->3) | - |
| 64 | 32 | 8 | 1 | YES | `8x ld.shared.b16` | 1 | 8 | 21 | 45.47 | 11.1 | -2.54% | T1:lanePart_change<br>T6:st_shared_traffic(5->3) | - |
| 64 | 64 | 4 | default | YES | `4x ld.shared.v4.b32` | 4 | 4 | 28 | 45.04 | 11.0 | 0.00% (base) | BASELINE | - |
| 64 | 64 | 4 | 8 | YES | `4x ld.shared.v4.b32` | 4 | 4 | 28 | 45.12 | 11.0 | +0.18% |  | - |
| 64 | 64 | 4 | 4 | YES | `8x ld.shared.v2.b32` | 2 | 4 | 22 | 44.74 | 10.9 | -0.67% | T1:lanePart_change<br>T6:st_shared_traffic(5->4) | - |
| 64 | 64 | 4 | 2 | YES | `4x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 4 | 22 | 44.74 | 10.9 | -0.67% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(5->3) | - |
| 64 | 64 | 4 | 1 | YES | `32x ld.shared.b16` | 1 | 2 | 32 | 44.74 | 10.9 | -0.67% | T2:warpPart_decrease<br>T6:st_shared_traffic(5->3) | - |
| 64 | 64 | 8 | default | YES | `2x ld.shared.v4.b32` | 4 | 8 | 29 | 49.33 | 12.0 | 0.00% (base) | BASELINE | - |
| 64 | 64 | 8 | 8 | YES | `2x ld.shared.v4.b32` | 4 | 8 | 29 | 49.36 | 12.1 | +0.06% |  | - |
| 64 | 64 | 8 | 4 | YES | `4x ld.shared.v2.b32` | 2 | 8 | 22 | 46.27 | 11.3 | -6.20% | T1:lanePart_change<br>T6:st_shared_traffic(7->4) | reproduced |
| 64 | 64 | 8 | 2 | YES | `2x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 8 | 22 | 46.13 | 11.3 | -6.49% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(7->3) | reproduced |
| 64 | 64 | 8 | 1 | YES | `16x ld.shared.b16` | 1 | 4 | 25 | 46.58 | 11.4 | -5.58% | T2:warpPart_decrease<br>T6:st_shared_traffic(7->3) | reproduced |
| 64 | 128 | 4 | default | YES | `8x ld.shared.v4.b32` | 2 | 4 | 38 | 58.13 | 14.2 | 0.00% (base) | BASELINE | - |
| 64 | 128 | 4 | 8 | YES | `8x ld.shared.v4.b32` | 2 | 4 | 38 | 57.90 | 14.1 | -0.39% |  | - |
| 64 | 128 | 4 | 4 | YES | `16x ld.shared.v2.b32` | 1 | 4 | 32 | 58.05 | 14.2 | -0.14% | T1:lanePart_change<br>T6:st_shared_traffic(7->4) | - |
| 64 | 128 | 4 | 2 | YES | `8x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 2 | 34 | 57.74 | 14.1 | -0.66% | T2:warpPart_decrease<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(7->5) | - |
| 64 | 128 | 4 | 1 | YES | `64x ld.shared.b16` | 1 | 1 | 32 | 57.89 | 14.1 | -0.41% | T3:warpPart_eq_1<br>T6:st_shared_traffic(7->1) | - |
| 64 | 128 | 8 | default | YES | `4x ld.shared.v4.b32` | 2 | 8 | 32 | 56.16 | 13.7 | 0.00% (base) | BASELINE | - |
| 64 | 128 | 8 | 8 | YES | `4x ld.shared.v4.b32` | 2 | 8 | 32 | 56.03 | 13.7 | -0.23% |  | - |
| 64 | 128 | 8 | 4 | YES | `8x ld.shared.v2.b32` | 1 | 8 | 24 | 55.15 | 13.5 | -1.79% | T1:lanePart_change<br>T6:st_shared_traffic(7->4) | - |
| 64 | 128 | 8 | 2 | YES | `4x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 4 | 22 | 54.85 | 13.4 | -2.34% | T2:warpPart_decrease<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(7->5) | - |
| 64 | 128 | 8 | 1 | YES | `32x ld.shared.b16` | 1 | 2 | 32 | 55.18 | 13.5 | -1.74% | T2:warpPart_decrease<br>T6:st_shared_traffic(7->3) | - |
| 64 | 256 | 4 | default | YES | `16x ld.shared.v4.b32` | 1 | 4 | 45 | 78.94 | 19.3 | 0.00% (base) | BASELINE | - |
| 64 | 256 | 4 | 8 | YES | `16x ld.shared.v4.b32` | 1 | 4 | 45 | 78.91 | 19.3 | -0.04% |  | - |
| 64 | 256 | 4 | 4 | YES | `32x ld.shared.v2.b32` | 1 | 2 | 34 | 78.56 | 19.2 | -0.49% | T2:warpPart_decrease<br>T6:st_shared_traffic(7->5) | - |
| 64 | 256 | 4 | 2 | YES | `8x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 1 | 34 | 78.35 | 19.1 | -0.75% | T3:warpPart_eq_1<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(7->1) | - |
| 64 | 256 | 4 | 1 | YES | `128x ld.shared.b16` | 1 | 1 | 40 | 78.78 | 19.2 | -0.20% | T3:warpPart_eq_1<br>T6:st_shared_traffic(7->3) | - |
| 64 | 256 | 8 | default | YES | `8x ld.shared.v4.b32` | 1 | 8 | 38 | 80.70 | 19.7 | 0.00% (base) | BASELINE | - |
| 64 | 256 | 8 | 8 | YES | `8x ld.shared.v4.b32` | 1 | 8 | 38 | 80.54 | 19.7 | -0.20% |  | - |
| 64 | 256 | 8 | 4 | YES | `16x ld.shared.v2.b32` | 1 | 4 | 37 | 80.32 | 19.6 | -0.48% | T2:warpPart_decrease<br>T6:st_shared_traffic(7->4) | - |
| 64 | 256 | 8 | 2 | YES | `8x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 2 | 34 | 79.90 | 19.5 | -0.99% | T2:warpPart_decrease<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(7->5) | - |
| 64 | 256 | 8 | 1 | YES | `64x ld.shared.b16` | 1 | 1 | 32 | 79.98 | 19.5 | -0.89% | T3:warpPart_eq_1<br>T6:st_shared_traffic(7->1) | - |
| 128 | 16 | 4 | default | YES | `2x ld.shared.v4.b32` | 16 | 4 | 31 | 43.66 | 10.7 | 0.00% (base) | BASELINE | - |
| 128 | 16 | 4 | 8 | YES | `2x ld.shared.v4.b32` | 16 | 4 | 31 | 43.76 | 10.7 | +0.22% |  | - |
| 128 | 16 | 4 | 4 | YES | `4x ld.shared.v4.b16` | 8 | 4 | 21 | 43.49 | 10.6 | -0.40% | T1:lanePart_change<br>T6:st_shared_traffic(5->3) | - |
| 128 | 16 | 4 | 2 | YES | `2x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 4 | 4 | 22 | 43.57 | 10.6 | -0.22% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(5->1) | - |
| 128 | 16 | 4 | 1 | YES | `16x ld.shared.b16` | 2 | 4 | 22 | 43.33 | 10.6 | -0.77% | T1:lanePart_change<br>T6:st_shared_traffic(5->3) | - |
| 128 | 16 | 8 | default | YES | `1x ld.shared.v4.b32` | 16 | 8 | 31 | 45.82 | 11.2 | 0.00% (base) | BASELINE | - |
| 128 | 16 | 8 | 8 | YES | `1x ld.shared.v4.b32` | 16 | 8 | 31 | 45.57 | 11.1 | -0.56% |  | - |
| 128 | 16 | 8 | 4 | YES | `2x ld.shared.v4.b16` | 8 | 8 | 21 | 43.09 | 10.5 | -5.97% | T1:lanePart_change<br>T6:st_shared_traffic(5->3) | reproduced |
| 128 | 16 | 8 | 2 | YES | `1x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 4 | 8 | 22 | 43.92 | 10.7 | -4.16% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(5->3) | reproduced |
| 128 | 16 | 8 | 1 | YES | `8x ld.shared.b16` | 2 | 8 | 21 | 42.98 | 10.5 | -6.22% | T1:lanePart_change<br>T6:st_shared_traffic(5->3) | reproduced |
| 128 | 32 | 4 | default | YES | `4x ld.shared.v4.b32` | 8 | 4 | 30 | 45.86 | 11.2 | 0.00% (base) | BASELINE | - |
| 128 | 32 | 4 | 8 | YES | `4x ld.shared.v4.b32` | 8 | 4 | 30 | 45.62 | 11.1 | -0.52% |  | - |
| 128 | 32 | 4 | 4 | YES | `8x ld.shared.v4.b16` | 4 | 4 | 22 | 45.65 | 11.1 | -0.45% | T1:lanePart_change<br>T6:st_shared_traffic(5->3) | - |
| 128 | 32 | 4 | 2 | YES | `4x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 2 | 4 | 23 | 45.23 | 11.0 | -1.36% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(5->3) | - |
| 128 | 32 | 4 | 1 | YES | `32x ld.shared.b16` | 1 | 4 | 32 | 45.36 | 11.1 | -1.08% | T1:lanePart_change<br>T6:st_shared_traffic(5->3) | - |
| 128 | 32 | 8 | default | YES | `2x ld.shared.v4.b32` | 8 | 8 | 29 | 46.54 | 11.4 | 0.00% (base) | BASELINE | - |
| 128 | 32 | 8 | 8 | YES | `2x ld.shared.v4.b32` | 8 | 8 | 29 | 46.72 | 11.4 | +0.38% |  | - |
| 128 | 32 | 8 | 4 | YES | `4x ld.shared.v4.b16` | 4 | 8 | 22 | 44.80 | 10.9 | -3.75% | T1:lanePart_change<br>T6:st_shared_traffic(5->4) | reproduced |
| 128 | 32 | 8 | 2 | YES | `2x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 2 | 8 | 21 | 44.64 | 10.9 | -4.09% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(5->3) | reproduced |
| 128 | 32 | 8 | 1 | YES | `16x ld.shared.b16` | 1 | 8 | 22 | 44.53 | 10.9 | -4.33% | T1:lanePart_change<br>T6:st_shared_traffic(5->3) | reproduced |
| 128 | 64 | 4 | default | YES | `8x ld.shared.v4.b32` | 4 | 4 | 32 | 55.22 | 13.5 | 0.00% (base) | BASELINE | - |
| 128 | 64 | 4 | 8 | YES | `8x ld.shared.v4.b32` | 4 | 4 | 32 | 55.78 | 13.6 | +1.01% |  | - |
| 128 | 64 | 4 | 4 | YES | `16x ld.shared.v2.b32` | 2 | 4 | 36 | 54.62 | 13.3 | -1.07% | T1:lanePart_change<br>T6:st_shared_traffic(5->4) | - |
| 128 | 64 | 4 | 2 | YES | `8x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 4 | 34 | 54.59 | 13.3 | -1.13% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(5->3) | - |
| 128 | 64 | 4 | 1 | YES | `64x ld.shared.b16` | 1 | 2 | 32 | 54.66 | 13.3 | -1.01% | T2:warpPart_decrease<br>T6:st_shared_traffic(5->3) | - |
| 128 | 64 | 8 | default | YES | `4x ld.shared.v4.b32` | 4 | 8 | 29 | 57.15 | 14.0 | 0.00% (base) | BASELINE | - |
| 128 | 64 | 8 | 8 | YES | `4x ld.shared.v4.b32` | 4 | 8 | 29 | 57.02 | 13.9 | -0.22% |  | - |
| 128 | 64 | 8 | 4 | YES | `8x ld.shared.v2.b32` | 2 | 8 | 22 | 56.38 | 13.8 | -1.34% | T1:lanePart_change<br>T6:st_shared_traffic(7->4) | - |
| 128 | 64 | 8 | 2 | YES | `4x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 8 | 24 | 55.76 | 13.6 | -2.44% | T1:lanePart_change<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(7->3) | - |
| 128 | 64 | 8 | 1 | YES | `32x ld.shared.b16` | 1 | 4 | 32 | 56.27 | 13.7 | -1.54% | T2:warpPart_decrease<br>T6:st_shared_traffic(7->3) | - |
| 128 | 128 | 4 | default | YES | `16x ld.shared.v4.b32` | 2 | 4 | 36 | 77.92 | 19.0 | 0.00% (base) | BASELINE | - |
| 128 | 128 | 4 | 8 | YES | `16x ld.shared.v4.b32` | 2 | 4 | 36 | 78.26 | 19.1 | +0.43% |  | - |
| 128 | 128 | 4 | 4 | YES | `32x ld.shared.v2.b32` | 1 | 4 | 31 | 78.02 | 19.0 | +0.12% | T1:lanePart_change<br>T6:st_shared_traffic(7->4) | - |
| 128 | 128 | 4 | 2 | YES | `16x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 2 | 32 | 77.66 | 19.0 | -0.33% | T2:warpPart_decrease<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(7->5) | - |
| 128 | 128 | 4 | 1 | YES | `128x ld.shared.b16` | 1 | 1 | 40 | 77.71 | 19.0 | -0.27% | T3:warpPart_eq_1<br>T6:st_shared_traffic(7->1) | - |
| 128 | 128 | 8 | default | YES | `8x ld.shared.v4.b32` | 2 | 8 | 38 | 78.62 | 19.2 | 0.00% (base) | BASELINE | - |
| 128 | 128 | 8 | 8 | YES | `8x ld.shared.v4.b32` | 2 | 8 | 38 | 78.66 | 19.2 | +0.04% |  | - |
| 128 | 128 | 8 | 4 | YES | `16x ld.shared.v2.b32` | 1 | 8 | 36 | 78.10 | 19.1 | -0.67% | T1:lanePart_change<br>T6:st_shared_traffic(7->4) | - |
| 128 | 128 | 8 | 2 | YES | `8x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 4 | 34 | 77.66 | 19.0 | -1.22% | T2:warpPart_decrease<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(7->5) | - |
| 128 | 128 | 8 | 1 | YES | `64x ld.shared.b16` | 1 | 2 | 32 | 77.76 | 19.0 | -1.10% | T2:warpPart_decrease<br>T6:st_shared_traffic(7->3) | - |
| 128 | 256 | 4 | default | YES | `27x ld.shared.v4.b32` | 1 | 4 | 45 | 123.89 | 30.2 | 0.00% (base) | BASELINE | - |
| 128 | 256 | 4 | 8 | YES | `27x ld.shared.v4.b32` | 1 | 4 | 45 | 123.87 | 30.2 | -0.01% |  | - |
| 128 | 256 | 4 | 4 | YES | `64x ld.shared.v2.b32` | 1 | 2 | 36 | 123.66 | 30.2 | -0.18% | T2:warpPart_decrease<br>T6:st_shared_traffic(7->5) | - |
| 128 | 256 | 4 | 2 | YES | `16x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 1 | 40 | 123.26 | 30.1 | -0.50% | T3:warpPart_eq_1<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(7->1) | - |
| 128 | 256 | 4 | 1 | YES | `256x ld.shared.b16` | 1 | 1 | 40 | 123.90 | 30.2 | +0.01% | T3:warpPart_eq_1<br>T6:st_shared_traffic(7->3) | - |
| 128 | 256 | 8 | default | YES | `16x ld.shared.v4.b32` | 1 | 8 | 33 | 123.70 | 30.2 | 0.00% (base) | BASELINE | - |
| 128 | 256 | 8 | 8 | YES | `16x ld.shared.v4.b32` | 1 | 8 | 33 | 123.81 | 30.2 | +0.09% |  | - |
| 128 | 256 | 8 | 4 | YES | `32x ld.shared.v2.b32` | 1 | 4 | 36 | 123.17 | 30.1 | -0.43% | T2:warpPart_decrease<br>T6:st_shared_traffic(7->4) | - |
| 128 | 256 | 8 | 2 | YES | `16x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 2 | 32 | 123.30 | 30.1 | -0.32% | T2:warpPart_decrease<br>T4:ldmatrix_transition<br>T6:st_shared_traffic(7->5) | - |
| 128 | 256 | 8 | 1 | YES | `128x ld.shared.b16` | 1 | 1 | 37 | 123.70 | 30.2 | +0.00% | T3:warpPart_eq_1<br>T6:st_shared_traffic(7->1) | - |

## 2. Grouped Structural Analysis

### Group A: Lane Partition Transitions (`lanePart > 1 -> 1`)
Examines configs where threadsPerWarp[M] changes from >1 to 1 (thread-local along reduction axis M).

### Group B: Warp Partition Transitions (`warpPart > 1 -> 1`)
Examines configs where warpsPerCTA[M] reduces to 1, eliminating cross-warp shared memory reduction exchange.

### Group C: LocalLoad Instruction Lowering (ld.shared vs ldmatrix)
Examines the impact of vectorization width forcing the backend to select `ldmatrix.sync.aligned.m8n8.x4` vs `ld.shared.v4.b32` / `ld.shared.v2.b32`.

### Group D: Post-Reduction Shared Memory / Layout Conversion Traffic
Examines whether narrower contiguous layouts introduce st.shared / ld.shared traffic after `tt.reduce`.