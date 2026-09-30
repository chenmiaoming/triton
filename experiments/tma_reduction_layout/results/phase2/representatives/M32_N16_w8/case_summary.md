# Representative Case Analysis: `M32_N16_w8`

- **Tile Shape**: `M=32, N=16, num_warps=8`

| Candidate | LocalLoad Lowering | lanePart[M] | warpPart[M] | Regs | Median (us) | vs Default | Transitions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| default | `1x ldmatrix.sync.aligned.m8n8.x1.shared.b16` | 4 | 8 | 22 | 45.09 | 0.00% (base) | BASELINE |
| 8 | INVALID | - | - | - | - | - | forcedVec 8 > numElemsPerThread (2) |
| 4 | INVALID | - | - | - | - | - | forcedVec 4 > numElemsPerThread (2) |
| 2 | `1x ldmatrix.sync.aligned.m8n8.x1.shared.b16` | 4 | 8 | 22 | 45.14 | +0.11% |  |
| 1 | `2x ld.shared.b16` | 2 | 8 | 21 | 44.70 | -0.85% | T1:lanePart_change, T4:ldmatrix_transition |