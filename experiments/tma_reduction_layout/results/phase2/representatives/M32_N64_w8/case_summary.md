# Representative Case Analysis: `M32_N64_w8`

- **Tile Shape**: `M=32, N=64, num_warps=8`

| Candidate | LocalLoad Lowering | lanePart[M] | warpPart[M] | Regs | Median (us) | vs Default | Transitions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| default | `1x ld.shared.v4.b32` | 4 | 8 | 29 | 45.47 | 0.00% (base) | BASELINE |
| 8 | `1x ld.shared.v4.b32` | 4 | 8 | 29 | 45.46 | -0.04% |  |
| 4 | `2x ld.shared.v2.b32` | 2 | 8 | 22 | 42.80 | -5.88% | T1:lanePart_change, T6:st_shared_traffic(7->4) |
| 2 | `1x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 8 | 21 | 42.18 | -7.25% | T1:lanePart_change, T4:ldmatrix_transition, T6:st_shared_traffic(7->3) |
| 1 | `8x ld.shared.b16` | 1 | 4 | 21 | 42.05 | -7.53% | T2:warpPart_decrease, T6:st_shared_traffic(7->3) |