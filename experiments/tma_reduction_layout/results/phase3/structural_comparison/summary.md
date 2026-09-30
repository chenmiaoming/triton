# Phase 3 Structural Decomposition & Mechanism Isolation Report

> [!NOTE]
> **Core Research Question**: Why does `M32_N64_w8` exhibit ~37%–43% marginal throughput separation across layout candidates (`3.88 -> 2.45 -> 2.25 ns/CTA`),
> whereas `M32_N128_w4` exhibits near-zero layout sensitivity (`~2.92 ns/CTA` across all candidates)?
>
> **Evidence Discipline**:
> 1. All structural metrics and opcode counts below are extracted directly from the verified canonical fixed-binary artifacts (`results/phase3/fixed_binary_artifacts/canonical/`), proven identical across three sequential invocations on the same NVIDIA H100 GPU.
> 2. Throughput metrics (`Logical Input GB/s`, `Logical I/O GB/s`) represent logical-byte transfer rates derived strictly as (logical bytes / fitted marginal grid time). They are **NOT** measured DRAM/HBM traffic or hardware bandwidth.
> 3. Opcode counts in the main tables represent **whole-kernel** occurrences across all phases. Phase-specific breakdowns are reported in Section 4 and 5.

## 1. Structural Decomposition Table: Positive Case (`M32_N64_w8`)

- **Tile Shape**: `M=32, N=64, num_warps=8`, Working Set: `4096 bytes input + 256 bytes output` per CTA.

| Candidate | LocalLoad | lanePart[M] | warpPart[M] | M Elems/Th | max.bf16x2 | cvt.f32 | shfl.sync (Whole) | max.f32 (Whole) | st.shared (Whole) | bar.sync (Whole) | Regs | Total SASS | Marginal Slope | vs Default | Logical Input GB/s | Logical I/O GB/s |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | `1x ld.shared.v4` | 4 | 8 | 1 | 0 | 8 | 40 | 40 | 7 | 14 | 29 | 296 | **3.9449 ns** | 0.00% (base) | 1038.3 GB/s | 1103.2 GB/s |
| `8` | `1x ld.shared.v4` | 4 | 8 | 1 | 0 | 8 | 40 | 40 | 7 | 14 | 29 | 296 | **3.9438 ns** | -0.03% | 1038.6 GB/s | 1103.5 GB/s |
| `4` | `2x ld.shared.v2` | 2 | 8 | 2 | 2 | 4 | 16 | 16 | 4 | 10 | 22 | 232 | **2.4854 ns** | -37.00% | 1648.1 GB/s | 1751.1 GB/s |
| `2` | `1x ldmatrix.x4` | 1 | 8 | 4 | 3 | 2 | 8 | 6 | 3 | 8 | 21 | 208 | **2.2766 ns** | -42.29% | 1799.2 GB/s | 1911.6 GB/s |
| `1` | `8x ld.shared.b16` | 1 | 4 | 8 | 0 | 1 | 2 | 2 | 3 | 8 | 21 | 200 | **2.2548 ns** | -42.84% | 1816.6 GB/s | 1930.1 GB/s |

## 2. Structural Decomposition Table: Negative Control Case (`M32_N128_w4`)

- **Tile Shape**: `M=32, N=128, num_warps=4`, Working Set: `8192 bytes input + 512 bytes output` per CTA.

| Candidate | LocalLoad | lanePart[M] | warpPart[M] | M Elems/Th | max.bf16x2 | cvt.f32 | shfl.sync (Whole) | max.f32 (Whole) | st.shared (Whole) | bar.sync (Whole) | Regs | Total SASS | Marginal Slope | vs Default | Logical Input GB/s | Logical I/O GB/s |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | `4x ld.shared.v4` | 2 | 4 | 4 | 12 | 8 | 25 | 24 | 7 | 14 | 32 | 280 | **2.9771 ns** | 0.00% (base) | 2751.7 GB/s | 2923.7 GB/s |
| `8` | `4x ld.shared.v4` | 2 | 4 | 4 | 12 | 8 | 25 | 24 | 7 | 14 | 32 | 280 | **2.9809 ns** | +0.13% | 2748.2 GB/s | 2919.9 GB/s |
| `4` | `8x ld.shared.v2` | 1 | 4 | 8 | 14 | 4 | 9 | 8 | 4 | 10 | 25 | 232 | **2.9804 ns** | +0.11% | 2748.7 GB/s | 2920.4 GB/s |
| `2` | `4x ldmatrix.x4` | 1 | 2 | 16 | 15 | 2 | 3 | 2 | 5 | 10 | 23 | 224 | **2.9681 ns** | -0.30% | 2760.1 GB/s | 2932.6 GB/s |
| `1` | `32x ld.shared.b16` | 1 | 1 | 32 | 0 | 1 | 1 | 0 | 1 | 4 | 32 | 232 | **2.9698 ns** | -0.24% | 2758.4 GB/s | 2930.8 GB/s |

## 3. Structural Decomposition Table: Weak-Effect Control Case (`M32_N16_w8`)

- **Tile Shape**: `M=32, N=16, num_warps=8`, Working Set: `1024 bytes input + 64 bytes output` per CTA.
- **Legality Note**: For shape `[32, 16]` with `num_warps=8`, candidate `8` and candidate `4` are **INVALID** (a 256-thread CTA cannot partition `N=16` with vector width 8 or 4).
- **Equivalence Note**: Candidate `2` produces an identical distributed layout (`sizePerThread=[1, 1, 2]`), resulting in bit-for-bit identical TTGIR, PTX, and SASS binaries to `default`.

| Candidate | LocalLoad | lanePart[M] | warpPart[M] | M Elems/Th | max.bf16x2 | cvt.f32 | shfl.sync (Whole) | max.f32 (Whole) | st.shared (Whole) | bar.sync (Whole) | Regs | Total SASS | Marginal Slope | vs Default | Logical Input GB/s | Logical I/O GB/s |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | `1x ldmatrix.x1` | 4 | 8 | 1 | 0 | 2 | 12 | 10 | 3 | 8 | 22 | 232 | **2.2044 ns** | 0.00% (base) | 464.5 GB/s | 493.6 GB/s |
| `2` | `1x ldmatrix.x1` | 4 | 8 | 1 | 0 | 2 | 12 | 10 | 3 | 8 | 22 | 232 | **2.2030 ns** | -0.06% | 464.8 GB/s | 493.9 GB/s |
| `1` | `2x ld.shared.b16` | 2 | 8 | 2 | 0 | 1 | 4 | 4 | 3 | 8 | 21 | 208 | **2.1435 ns** | -2.76% | 477.7 GB/s | 507.6 GB/s |

## 4. Itemized Delta Table: `default` -> `cand4` in `M32_N64_w8`

Holding `warpPart[M]=8` constant while transitioning `lanePart[M]` from 4 to 2 coincides with a reduction in the empirical marginal grid slope from **3.8822 ns** to **2.4543 ns** (-36.78%).

| Structural Metric | default (lanePart[M]=4) | cand4 (lanePart[M]=2) | Absolute Delta | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **LocalLoad family** | `1x ld.shared.v4.b32` | `2x ld.shared.v2.b32` | +1 issue, 2x narrower | Vector width halved |
| **M elements per thread** | 1 | 2 | +1 element | 2x increase |
| **Thread-local packed max (`max.bf16x2`)** | 0 | 2 | +2 insts | Enabled (was 0) |
| **Precision conversion (`cvt.f32.bf16`)** | 8 | 4 | -4 insts | -50.0% |
| **Intra-warp reduction shuffles (`shfl.sync`)** | 16 (offsets 16, 8) | 4 (offset 16) | -12 insts | -75.0% |
| **Intra-warp visible serial reduction stages** | 2 shuffle+max stages | 1 shuffle+max stage | -1 stage | -50.0% stages |
| **Cross-warp shared memory exchanges** | 2 rounds (4 st.shared, 4 ld.shared) | 1 round (2 st.shared, 2 ld.shared) | -2 st, -2 ld | -50.0% |
| **Cross-warp combine shuffles** | 24 (offsets 4, 2, 1) | 12 (offsets 4, 2, 1) | -12 insts | -50.0% |
| **Whole-kernel float max (`max.f32`)** | 40 | 16 | -24 insts | -60.0% |
| **Whole-kernel reduction shuffles (`shfl.sync`)** | 40 | 16 | -24 insts | -60.0% |
| **Whole-kernel synchronization barriers (`bar.sync`)** | 14 | 10 | -4 barriers | -28.6% |
| **Post-reduction convert shared stores** | 2 (`st.shared.v4.b32`) | 1 (`st.shared.v4.b32`) | -1 store | -50.0% |
| **Post-reduction convert barriers** | 2 (`bar.sync 0`) | 2 (`bar.sync 0`) | 0 | Same |
| **Physical registers / thread** | 29 | 22 | -7 registers | -24.1% |
| **Total SASS instructions** | 296 | 232 | -64 instructions | -21.6% |
| **Marginal grid slope per CTA** | **3.8822 ns** | **2.4543 ns** | **-1.4279 ns** | **-36.78%** |
| **Logical input throughput** | 1055.1 GB/s | 1668.9 GB/s | +613.8 GB/s | +58.2% |

## 5. Itemized Delta Table: `cand4` -> `cand2` -> `cand1` in `M32_N64_w8`

| Structural Metric | cand4 (vec=4) | cand2 (vec=2) | Delta (4 -> 2) | cand1 (vec=1) | Delta (2 -> 1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LocalLoad family** | `2x ld.shared.v2` | `1x ldmatrix.x4` | Lowering family switch | `8x ld.shared.b16` | Scalar degradation |
| **lanePart[M]** | 2 | 1 | -1 (warp spans N only) | 1 | 0 |
| **warpPart[M]** | 8 | 8 | 0 | 4 | -4 (warps span N) |
| **M elements per thread** | 2 | 4 | +2 elems | 8 | +4 elems |
| **Thread-local max (`max.bf16x2`)** | 2 | 3 | +1 inst | 0 (7x scalar max.bf16) | Packed -> Scalar |
| **Precision conversion (`cvt.f32`)** | 4 | 2 | -2 insts | 1 | -1 inst |
| **Intra-warp reduction shuffles** | 4 | 0 | -4 insts (-100%) | 0 | 0 |
| **Cross-warp combine shuffles** | 12 | 6 | -6 insts (-50%) | 2 | -4 insts |
| **Post-convert mechanism** | Shared mem + ldmatrix | **In-register index shuffle** | Bypasses shared mem | Shared mem scalar | Re-enters shared mem |
| **Post-convert shared stores** | 1 | 0 | -1 store | 1 | +1 store |
| **Post-convert barriers** | 2 | 0 | -2 barriers | 1 | +1 barrier |
| **Total barriers (`bar.sync`)** | 10 | 8 | -2 barriers | 8 | 0 |
| **Total SASS instructions** | 232 | 208 | -24 insts (-10.3%) | 200 | -8 insts (-3.8%) |
| **Physical registers / thread** | 22 | 21 | -1 register | 21 | 0 |
| **Marginal slope (ns/CTA)** | **2.4543 ns** | **2.2537 ns** | **-0.2006 ns (-8.17%)** | **2.2249 ns** | **-0.0288 ns (-1.28%)** |
| **Logical input throughput** | 1668.9 GB/s | 1817.5 GB/s | +148.6 GB/s | 1841.0 GB/s | +23.5 GB/s |

## 6. Answers to the 5 Research Questions

### Question 1: What reduction instructions disappear from `default` -> `cand4` in `M32_N64_w8` while `warpPart` remains constant?
1. **Thread-local reduction is enabled**: Because `lanePart[M]` drops from 4 to 2, each thread owns 2 elements along reduction axis M instead of 1. The thread folds these locally via **2x `max.bf16x2`** before precision conversion.
2. **Conversions cut in half**: `cvt.f32.bf16` drops from 8 to 4.
3. **Intra-warp shuffles cut by 75%**: With 2 lanes on M instead of 4, the offset-8 butterfly shuffle stage disappears. Intra-warp shuffles drop from 16 to 4 (-12 shuffles, -12 max.f32).
4. **Cross-warp exchanges halved**: Cross-warp shared memory roundtrips drop from 2 rounds to 1 round (shared stores drop from 4 to 2, shared loads drop from 4 to 2).
5. **Cross-warp combine cut by 50%**: Combine shuffles drop from 24 to 12 (-12 shuffles, -12 max.f32).
6. **Barriers reduced**: Total CTA barriers drop from 14 to 10 (-4 barriers).
7. **In SASS**: Total instructions drop from 296 to 232 (-64 instructions), with SHFL dropping from 40 to 16 (-60%) and FMNMX dropping from 40 to 16 (-60%).

### Question 2: Do these changes also occur in `M32_N128_w4`? Why is there no performance difference?
- **OBSERVED**: `M32_N128_w4` shows substantial reductions in shuffles, barriers, and instruction count across layouts (e.g. shuffles drop from 25 to 9, float maxes drop from 24 to 8, barriers drop from 14 to 10, total SASS drops from 280 to 232; in `cand1`, reduction communication is 100% eliminated), yet empirical marginal slopes remain near parity (`~2.92 ns/CTA` across all candidates, delta < 0.12%).
- **UNKNOWN**: The current evidence does not identify why those structural reductions do not change throughput. Candidate explanations include a memory-system limitation (e.g. high logical traffic rate of ~2.8 TB/s operating near an empirical throughput ceiling), execution overlap hiding SM-side work, issue-resource behavior, or another bottleneck, but none is established without hardware-counter proof.

### Question 3: Does the ~37% slope difference in `M32_N64_w8` correspond to an identifiable dependency-chain reduction?
- **A strong structural correlation exists**: Transitioning `default -> cand4` coincides with:
  - fewer PTX reduction shuffles (40 -> 16)
  - fewer max.f32 operations (40 -> 16)
  - fewer shared memory exchanges (2 rounds -> 1 round)
  - fewer CTA barriers (14 -> 10)
  - fewer physical registers (29 -> 22)
  - a shorter visible reduction-stage sequence (from 2 visible shuffle+max stages to 1 visible stage)
  - and an empirical marginal slope that is ~37% lower (3.8822 -> 2.4543 ns/CTA).
- **No individual mechanism is yet causally isolated**: Whether the runtime reduction is primarily driven by fewer barrier synchronizations, fewer shuffles, packed arithmetic folding, or lower register pressure cannot be determined from this single transition alone.

### Question 4: Which structural changes coincide with the additional ~8% gain from `cand4` -> `cand2`?
- Coinciding structural changes include:
  1. `lanePart[M]` drops from 2 to 1: Intra-warp reduction shuffles along M completely disappear (4 -> 0). All intra-warp M reduction folds into registers via 3x packed `max.bf16x2`.
  2. **LocalLoad lowering switch**: Lowered to hardware `1x ldmatrix.x4` instead of `2x ld.shared.v2`.
  3. **Post-reduction conversion change**: Instead of storing to shared memory and re-loading with ldmatrix, `cand2` performs layout redistribution directly in registers via `2x shfl.sync.idx.b32` and `1x selp.b32`, eliminating 2 CTA barriers in the epilogue.
  4. **Barrier count**: Drops from 10 to 8.
  5. **Register count**: Drops from 22 to 21.
- **Conclusion**: The current evidence cannot determine which of these structural changes accounts for the ~0.20 ns marginal-slope difference.

### Question 5: Why does `cand2` -> `cand1` show near-zero additional gain (~1.3%)?
- Transitioning `cand2 -> cand1` simultaneously:
  1. Replaces `1x ldmatrix.x4` with `8x ld.shared.b16` scalar shared loads.
  2. Replaces packed BF16 reduction (`max.bf16x2`) with scalar BF16 reduction (`7x max.bf16`).
  3. Reduces remaining cross-warp communication (4 fewer combine shuffles).
  4. Reintroduces post-convert shared-memory work (`1x st.shared.b32`, `1x bar.sync`, `1x ld.shared.b32`).
- The net measured slope change is only -1.28% (2.2537 -> 2.2249 ns/CTA).
- **Which positive and negative costs cancel is UNKNOWN**: We cannot determine whether scalar load overhead offsets communication savings without targeted differential microbenchmarks.
