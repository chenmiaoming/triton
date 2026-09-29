# TMA Reduction Layout Baseline Characterization: [1, 32, 128] BF16 -> FP32 Max Axis=1

- **Hardware**: NVIDIA H100 80GB HBM3 (580.95.05, CC [9, 0])
- **Git HEAD**: `3c6d6d1b64db6f077b4a19a6aba487b3ae4bfead` (branch: `explore/tma-reduction-layout`, dirty: `False`)
- **Source Manifest**: `2bd2d1c377f832daa43612dbdbeffb2a71ab50ea6f3526ee3a4c332edbd092aa`
- **Triton**: `3.9.0` (`/opt/triton-src/python/triton/__init__.py`)

---

## Table 1: Layout Specifications & Derived Partition Structure

> [!NOTE]
> Attributes `sizePerThread`, `threadsPerWarp`, `warpsPerCTA` are directly **OBSERVED** from the `ttg.local_load` destination `#blocked` layout. > Lane partitions, warp partitions, and elements/partition are **DERIVED** via exact formulas from the layout specification. Cross-CTA reduction is absent as `ttg.num-ctas = 1`.

| Candidate | sizePerThread [OBS] | threadsPerWarp [OBS] | warpsPerCTA [OBS] | numCTAs [OBS] | Lane Parts (M) [DER] | Warp Parts (M) [DER] | CTA Parts (M) [DER] | M Elems/Partition [DER] | Total Elems/Thread [DER] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `default` | `[1, 1, 8]` | `[1, 2, 16]` | `[1, 4, 1]` | `1` | 2 | 4 | 1 | 4 | 32 |
| `8` | `[1, 1, 8]` | `[1, 2, 16]` | `[1, 4, 1]` | `1` | 2 | 4 | 1 | 4 | 32 |
| `4` | `[1, 1, 4]` | `[1, 1, 32]` | `[1, 4, 1]` | `1` | 1 | 4 | 1 | 8 | 32 |
| `2` | `[1, 1, 2]` | `[1, 1, 32]` | `[1, 2, 2]` | `1` | 1 | 2 | 1 | 16 | 32 |
| `1` | `[1, 1, 1]` | `[1, 1, 32]` | `[1, 1, 4]` | `1` | 1 | 1 | 1 | 32 | 32 |

---

## Table 2: Observed LocalLoad Lowering & Shared Layout Facts

> [!NOTE]
> Shared memory descriptor layout is `#ttg.nvmma_shared` with `swizzlingByteWidth=128, elementBitWidth=16`. > Opcode counts represent whole-kernel occurrences across all phases. Physical registers are extracted via `cuobjdump -res-usage`.

| Candidate | Initial LocalLoad Lowering [OBS] | ld.shared (Total) [OBS] | ldmatrix (Total) [OBS] | st.shared (Total) [OBS] | shfl.sync (Setup + Reduct) [OBS] | bar.sync (Total) [OBS] | Physical Regs/Thread [OBS] | Shared Mem (B) [OBS] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `default` | 4 × `ld.shared.v4.b32` | 8 | 1 | 7 | 25 (1 + 24) | 14 | `32` | `1024` |
| `8` | 4 × `ld.shared.v4.b32` | 8 | 1 | 7 | 25 (1 + 24) | 14 | `32` | `1024` |
| `4` | 8 × `ld.shared.v2.b32` | 10 | 1 | 4 | 9 (1 + 8) | 10 | `25` | `1024` |
| `2` | 4 × `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 3 | 4 | 5 | 3 (1 + 2) | 10 | `23` | `1024` |
| `1` | 32 × `ld.shared.b16` | 32 | 0 | 1 | 1 (1 + 0) | 4 | `32` | `1024` |

---

## Table 3: Empirical Execution Timing on NVIDIA H100 (Single-CTA Tile)

> [!IMPORTANT]
> **Timing Distinguishability**: For this single-CTA 8-KiB tile benchmark, the current measurement does not reliably distinguish the candidates' runtime. > Median differences across candidates (~0.1 µs, ~0.5%) fall well within measurement noise and run-to-run variation. **No statistically reliable performance ordering is claimed.**

| Candidate | Correctness [OBS] | Median (µs) [MEA] | P10 (µs) [MEA] | P90 (µs) [MEA] | IQR (µs) [MEA] | MAD (µs) [MEA] | Block Medians Range (µs) [MEA] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `default` | PASS | **22.27** | 21.76 | 23.23 | 0.65 | 0.32 | [21.97, 23.10] |
| `8` | PASS | **22.24** | 21.86 | 22.79 | 0.45 | 0.22 | [22.11, 22.43] |
| `4` | PASS | **22.27** | 21.79 | 22.92 | 0.58 | 0.29 | [21.95, 22.64] |
| `2` | PASS | **22.21** | 21.69 | 22.79 | 0.54 | 0.26 | [21.81, 22.34] |
| `1` | PASS | **22.24** | 21.76 | 22.94 | 0.55 | 0.29 | [22.08, 22.48] |