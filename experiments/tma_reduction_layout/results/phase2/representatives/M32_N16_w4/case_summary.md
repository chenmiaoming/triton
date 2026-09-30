# Representative Case Analysis: `M32_N16_w4`

- **Tile Shape**: `M=32, N=16, num_warps=4`

| Candidate | LocalLoad Lowering | lanePart[M] | warpPart[M] | Regs | Median (us) | vs Default | Transitions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| default | `1x ld.shared.v4.b16` | 8 | 4 | 21 | 43.22 | 0.00% (base) | BASELINE |
| 8 | INVALID | - | - | - | - | - | forcedVec 8 > numElemsPerThread (4) |
| 4 | `1x ld.shared.v4.b16` | 8 | 4 | 21 | 42.83 | -0.89% |  |
| 2 | `1x ldmatrix.sync.aligned.m8n8.x2.shared.b16` | 4 | 4 | 22 | 42.61 | -1.41% | T1:lanePart_change, T4:ldmatrix_transition, T6:st_shared_traffic(3->1) |
| 1 | `4x ld.shared.b16` | 2 | 4 | 21 | 42.86 | -0.81% | T1:lanePart_change |