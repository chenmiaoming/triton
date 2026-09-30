# Representative Case Analysis: `M32_N128_w4`

- **Tile Shape**: `M=32, N=128, num_warps=4`

| Candidate | LocalLoad Lowering | lanePart[M] | warpPart[M] | Regs | Median (us) | vs Default | Transitions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| default | `4x ld.shared.v4.b32` | 2 | 4 | 32 | 44.66 | 0.00% (base) | BASELINE |
| 8 | `4x ld.shared.v4.b32` | 2 | 4 | 32 | 44.58 | -0.18% |  |
| 4 | `8x ld.shared.v2.b32` | 1 | 4 | 25 | 43.79 | -1.93% | T1:lanePart_change, T6:st_shared_traffic(7->4) |
| 2 | `4x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 2 | 23 | 43.89 | -1.72% | T2:warpPart_decrease, T4:ldmatrix_transition, T6:st_shared_traffic(7->5) |
| 1 | `32x ld.shared.b16` | 1 | 1 | 32 | 43.54 | -2.51% | T3:warpPart_eq_1, T6:st_shared_traffic(7->1) |