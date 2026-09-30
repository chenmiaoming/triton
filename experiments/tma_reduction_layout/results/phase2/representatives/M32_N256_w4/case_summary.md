# Representative Case Analysis: `M32_N256_w4`

- **Tile Shape**: `M=32, N=256, num_warps=4`

| Candidate | LocalLoad Lowering | lanePart[M] | warpPart[M] | Regs | Median (us) | vs Default | Transitions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| default | `8x ld.shared.v4.b32` | 1 | 4 | 32 | 58.14 | 0.00% (base) | BASELINE |
| 8 | `8x ld.shared.v4.b32` | 1 | 4 | 32 | 57.98 | -0.28% |  |
| 4 | `16x ld.shared.v2.b32` | 1 | 2 | 39 | 57.46 | -1.18% | T2:warpPart_decrease, T6:st_shared_traffic(7->5) |
| 2 | `4x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 1 | 32 | 57.52 | -1.07% | T3:warpPart_eq_1, T4:ldmatrix_transition, T6:st_shared_traffic(7->1) |
| 1 | `64x ld.shared.b16` | 1 | 1 | 37 | 57.65 | -0.85% | T3:warpPart_eq_1, T6:st_shared_traffic(7->3) |