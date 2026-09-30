# Representative Case Analysis: `M32_N64_w8`

- **Tile Shape**: `M=32, N=64, num_warps=8`

| Candidate | LocalLoad Lowering | lanePart[M] | warpPart[M] | Regs | Median (us) | Amortized Time/CTA (ns) | vs Default | Transitions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| default | `1x ld.shared.v4.b32` | 4 | 8 | 29 | 45.47 | 11.1 | 0.00% (base) | BASELINE |
| 8 | `1x ld.shared.v4.b32` | 4 | 8 | 29 | 45.46 | 11.1 | -0.04% |  |
| 4 | `2x ld.shared.v2.b32` | 2 | 8 | 22 | 42.80 | 10.4 | -5.88% | T1_lane_change_warp_same, T5_register_count_change(29->22) |
| 2 | `1x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 8 | 21 | 42.18 | 10.3 | -7.25% | T1_lane_change_warp_same, T4_ldmatrix_family_change, T5_register_count_change(29->21) |
| 1 | `8x ld.shared.b16` | 1 | 4 | 21 | 42.05 | 10.3 | -7.53% | T2_warp_decrease_gt1, T5_register_count_change(29->21) |