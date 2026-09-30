# Representative Case Analysis: `M32_N128_w4`

- **Tile Shape**: `M=32, N=128, num_warps=4`

| Candidate | LocalLoad Lowering | lanePart[M] | warpPart[M] | Regs | Median (us) | Amortized Time/CTA (ns) | vs Default | Transitions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| default | `4x ld.shared.v4.b32` | 2 | 4 | 32 | 44.66 | 10.9 | 0.00% (base) | BASELINE |
| 8 | `4x ld.shared.v4.b32` | 2 | 4 | 32 | 44.58 | 10.9 | -0.18% |  |
| 4 | `8x ld.shared.v2.b32` | 1 | 4 | 25 | 43.79 | 10.7 | -1.93% | T1_lane_change_warp_same, T5_register_count_change(32->25) |
| 2 | `4x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 2 | 23 | 43.89 | 10.7 | -1.72% | T2_warp_decrease_gt1, T4_ldmatrix_family_change, T5_register_count_change(32->23) |
| 1 | `32x ld.shared.b16` | 1 | 1 | 32 | 43.54 | 10.6 | -2.51% | T3_warp_becomes1 |