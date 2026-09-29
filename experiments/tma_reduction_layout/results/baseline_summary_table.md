# TMA Reduction Layout Baseline Characterization: [1, 32, 128] BF16 -> FP32 Max Axis=1

- **Hardware**: NVIDIA H100 80GB HBM3 (580.95.05, CC [9, 0])
- **Git HEAD**: `e6df4ea43ea75c632ad54ad0698fd7b300590ac3` (branch: `explore/tma-reduction-layout`, dirty: `False`)
- **Source Manifest**: `cdceb86aa28fe5057ceefee69e3f2760db764810921ccb7d5b2ae6ae2b6607d6`
- **Triton**: `3.9.0` (`/opt/triton-src/python/triton/__init__.py`)

---

## Table 1: Layout Specifications & Derived Partition Structure

> [!NOTE]
> Attributes `sizePerThread`, `threadsPerWarp`, `warpsPerCTA` are directly **OBSERVED** from the `ttg.local_load` destination `#blocked` layout. Lane partitions, warp partitions, and elements/partition are **DERIVED** via exact formulas from the layout specification. Cross-CTA reduction is absent as `ttg.num-ctas = 1`.

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
> Shared memory descriptor layout is `#ttg.nvmma_shared` with `swizzlingByteWidth=128, elementBitWidth=16`. Opcode counts represent whole-kernel occurrences across all phases. Physical registers are extracted via `cuobjdump -res-usage`. For this artifact, 8200 B is consistent with: 8192 B TMA tile storage (1 * 32 * 128 * 2 B) + 8 B mbarrier storage.
| Candidate | Initial LocalLoad Lowering [OBS] | ld.shared (Total) [OBS] | ldmatrix (Total) [OBS] | st.shared (Total) [OBS] | shfl.sync (Setup + Reduct) [OBS] | bar.sync (Total) [OBS] | Physical Regs/Thread [OBS] | cuobjdump SHARED (B) [OBS] | Triton metadata.shared (B) [OBS] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `default` | 4 × `ld.shared.v4.b32` | 8 | 1 | 7 | 25 (1 + 24) | 14 | `32` | `1024` | `8200` |
| `8` | 4 × `ld.shared.v4.b32` | 8 | 1 | 7 | 25 (1 + 24) | 14 | `32` | `1024` | `8200` |
| `4` | 8 × `ld.shared.v2.b32` | 10 | 1 | 4 | 9 (1 + 8) | 10 | `25` | `1024` | `8200` |
| `2` | 4 × `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 3 | 4 | 5 | 3 (1 + 2) | 10 | `23` | `1024` | `8200` |
| `1` | 32 × `ld.shared.b16` | 32 | 0 | 1 | 1 (1 + 0) | 4 | `32` | `1024` | `8200` |

---

## Table 3: Empirical Execution Timing on NVIDIA H100 (Single-CTA Tile)

> [!IMPORTANT]
> **Timing Distinguishability**: For this single-CTA 8-KiB tile benchmark, the current measurement protocol does not establish a reliable performance difference among the candidates. No performance ordering is supported by this run.

| Candidate | Correctness [OBS] | Median (µs) [MEA] | P10 (µs) [MEA] | P90 (µs) [MEA] | IQR (µs) [MEA] | MAD (µs) [MEA] | Block Medians Range (µs) [MEA] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `default` | PASS | **22.82** | 21.43 | 28.56 | 1.70 | 0.74 | [21.58, 23.41] |
| `8` | PASS | **22.56** | 21.31 | 29.64 | 2.06 | 0.90 | [21.46, 23.41] |
| `4` | PASS | **22.56** | 21.31 | 28.45 | 1.92 | 0.93 | [21.76, 23.30] |
| `2` | PASS | **22.82** | 21.37 | 26.67 | 1.62 | 0.83 | [21.60, 23.46] |
| `1` | PASS | **22.75** | 21.50 | 30.06 | 1.92 | 0.93 | [21.49, 23.50] |