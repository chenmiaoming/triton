# Representative Case Analysis: `M32_N256_w4`

- **Tile Shape**: `M=32, N=256, num_warps=4`

| Candidate | LocalLoad Lowering | lanePart[M] | warpPart[M] | Regs | Median (us) | Amortized Time/CTA (ns) | vs Default | Transitions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| default | `8x ld.shared.v4.b32` | 1 | 4 | 32 | 58.14 | 14.2 | 0.00% (base) | BASELINE |
| 8 | `8x ld.shared.v4.b32` | 1 | 4 | 32 | 57.98 | 14.2 | -0.28% |  |
| 4 | `16x ld.shared.v2.b32` | 1 | 2 | 39 | 57.46 | 14.0 | -1.18% | T2_warp_decrease_gt1, T5_register_count_change(32->39) |
| 2 | `4x ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 1 | 1 | 32 | 57.52 | 14.0 | -1.07% | T3_warp_becomes1, T4_ldmatrix_family_change |
| 1 | `64x ld.shared.b16` | 1 | 1 | 37 | 57.65 | 14.1 | -0.85% | T3_warp_becomes1, T5_register_count_change(32->37) |