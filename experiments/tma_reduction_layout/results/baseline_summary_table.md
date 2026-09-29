# TMA Reduction Layout Baseline Characterization: [1, 32, 128] BF16 -> FP32 Max Axis=1

- **Hardware**: NVIDIA H100 80GB HBM3 (Driver 580.95.05, Compute Capability [9, 0])
- **Git HEAD**: `b0832694a1` (branch: `explore/tma-reduction-layout`)
- **Triton**: `3.9.0` (`/opt/triton-src/python/triton/__init__.py`)

---

## Table 1: Layout Specifications & Derived Partition Structure

> [!NOTE]
> Attributes `sizePerThread`, `threadsPerWarp`, `warpsPerCTA` are directly **OBSERVED** from the `ttg.local_load` destination `#blocked` layout.
> Lane partitions, warp partitions, and elements/partition are **DERIVED** via exact formulas from the layout specification.
> Cross-CTA reduction is absent as module attribute `ttg.num-ctas = 1`.

| Candidate | sizePerThread [OBS] | threadsPerWarp [OBS] | warpsPerCTA [OBS] | numCTAs [OBS] | Lane Parts (M) [DER] | Warp Parts (M) [DER] | CTA Parts (M) [DER] | M Elems/Partition [DER] | Total Elems/Thread [DER] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `default` | `[1, 1, 8]` | `[1, 2, 16]` | `[1, 4, 1]` | 1 | 2 | 4 | 1 | 4 | 32 |
| `8` | `[1, 1, 8]` | `[1, 2, 16]` | `[1, 4, 1]` | 1 | 2 | 4 | 1 | 4 | 32 |
| `4` | `[1, 1, 4]` | `[1, 1, 32]` | `[1, 4, 1]` | 1 | 1 | 4 | 1 | 8 | 32 |
| `2` | `[1, 1, 2]` | `[1, 1, 32]` | `[1, 2, 2]` | 1 | 1 | 2 | 1 | 16 | 32 |
| `1` | `[1, 1, 1]` | `[1, 1, 32]` | `[1, 1, 4]` | 1 | 1 | 1 | 1 | 32 | 32 |

---

## Table 2: Observed LocalLoad Lowering & Shared Layout Facts

> [!NOTE]
> Shared memory descriptor layout is `#ttg.nvmma_shared` with `swizzlingByteWidth=128, elementBitWidth=16`.
> Opcode counts represent whole-kernel occurrences across all phases.
> Physical registers are marked `UNKNOWN` pending cubin resource extraction via `cuobjdump -res-usage`.

| Candidate | Initial LocalLoad Lowering [OBS] | ld.shared (Total) [OBS] | ldmatrix (Total) [OBS] | st.shared (Total) [OBS] | shfl.sync (Setup + Reduct) [OBS] | bar.sync (Total) [OBS] | Physical Regs/Thread [OBS] | Shared Mem (B) [OBS] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `default` | 4 × `ld.shared.v4.b32` | 8 | 1 | 7 | 25 (1 + 24) | 14 | UNKNOWN | UNKNOWN |
| `8` | 4 × `ld.shared.v4.b32` | 8 | 1 | 7 | 25 (1 + 24) | 14 | UNKNOWN | UNKNOWN |
| `4` | 8 × `ld.shared.v2.b32` | 10 | 1 | 4 | 9 (1 + 8) | 10 | UNKNOWN | UNKNOWN |
| `2` | 4 × `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 3 | 4 | 5 | 3 (1 + 2) | 10 | UNKNOWN | UNKNOWN |
| `1` | 32 × `ld.shared.b16` | 32 | 0 | 1 | 1 (1 + 0) | 4 | UNKNOWN | UNKNOWN |

---

## Table 3: Empirical Execution Timing on NVIDIA H100 (Single-CTA Tile)

> [!IMPORTANT]
> **Timing Distinguishability**: For this single-CTA 8-KiB tile benchmark, the current measurement does not reliably distinguish the candidates' runtime.
> Median differences across candidates (~0.1 µs, ~0.5%) fall well within measurement noise and run-to-run variation.
> **No statistically reliable performance ordering is claimed.**

| Candidate | Correctness [OBS] | Median (µs) [MEA] | Mean (µs) [MEA] | Min (µs) [MEA] | Max (µs) [MEA] | Stdev (µs) [MEA] |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `default` | PASS | **24.10** | 24.12 | 23.97 | 25.12 | 0.14 |
| `8` | PASS | **24.14** | 24.15 | 23.97 | 25.06 | 0.13 |
| `4` | PASS | **24.06** | 24.09 | 23.94 | 24.96 | 0.12 |
| `2` | PASS | **24.03** | 24.07 | 23.90 | 25.02 | 0.14 |
| `1` | PASS | **23.97** | 24.02 | 23.87 | 24.93 | 0.12 |