# TMA Reduction Layout Experiment Results: [1, 32, 128] BF16 -> FP32 Max Axis=1

- **Hardware**: NVIDIA H100 80GB HBM3 (580.95.05, CC [9, 0])
- **Git HEAD**: `7587f00627` (branch: `explore/tma-reduction-layout`, dirty: `True`)
- **Triton**: `3.9.0` (`/opt/triton-src/python/triton/__init__.py`)

### Characterization Table

| Candidate | sizePerThread [A] | threadsPerWarp [B] | warpsPerCTA [B] | Lane Parts (M) [A] | Warp Parts (M) [A] | M Ownership [A] | ld.shared [B] | shfl.sync [B] | st.shared [B] | bar.sync [B] | Regs [B] | Correct [C] | Median (us) [C] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `default` | `[1, 1, 8]` | `[1, 2, 16]` | `[1, 4, 1]` | 2 | 4 | 4 | 8 | 25 | 7 | 14 | None | PASS | **24.10** |
| `8` | `[1, 1, 8]` | `[1, 2, 16]` | `[1, 4, 1]` | 2 | 4 | 4 | 8 | 25 | 7 | 14 | None | PASS | **24.14** |
| `4` | `[1, 1, 4]` | `[1, 1, 32]` | `[1, 4, 1]` | 1 | 4 | 8 | 10 | 9 | 4 | 10 | None | PASS | **24.06** |
| `2` | `[1, 1, 2]` | `[1, 1, 32]` | `[1, 2, 2]` | 1 | 2 | 16 | 3 | 3 | 5 | 10 | None | PASS | **24.03** |
| `1` | `[1, 1, 1]` | `[1, 1, 32]` | `[1, 1, 4]` | 1 | 1 | 32 | 32 | 1 | 1 | 4 | None | PASS | **23.97** |

**Legend**:
- **[A. static layout-derived]**: Theoretically derived from tensor dimensions, vector size, and BlockedEncoding rules.
- **[B. observed from TTGIR/PTX/SASS]**: Extracted directly from compiled IR / PTX assembly.
- **[C. measured on H100]**: Empirically measured via CUDA events on strict NVIDIA H100.