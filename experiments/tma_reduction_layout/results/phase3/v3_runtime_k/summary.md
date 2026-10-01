# Phase 3 Step B v3: Single-Binary Runtime-K Isolation Summary

## 1. Overall Isolation Status
**Status**: `V3_CODEGEN_CONFOUNDED_NOT_ACCEPTED_FOR_TIMING`

## 2. Criteria Evaluation
| Criterion | Status |
| :--- | :--- |
| Criterion A (TMA Descriptor Load Invariant) | **PASS** |
| Criterion B (Initial LocalLoad Invariant) | **FAIL_LOCAL_LOAD_SUNK_INTO_LOOP** |
| Criterion C (Reduction Body Template Invariant) | **FAIL_TEMPLATE_CONFOUNDED_BY_LOCAL_LOAD** |
| Criterion D (Distributed Layout Invariant) | **PASS** |
| Criterion E (Self-Contained Committed Artifacts) | **PASS** |
| Criterion F (Residency & Occupancy Matched) | **FAIL_RESIDENCY_DISPARITY** |
| Criterion G (Spill Local/Stack == 0) | **PASS** |
| Criterion H (Runtime Loop Verified) | **PASS** |
| Criterion I (Runtime-K Specialization Absent) | **PASS** |
| Criterion J (Numerical Correctness across K) | **PASS** |

## 3. Discovered Confounds Preventing Timing Isolation
- Residency disparity in M32_N64_w8: default has 6 blocks/SM (48 warps), cand4 has 8 blocks/SM (64 warps)
- Triton lowered ttg.local_load inside loop (7 instructions) in M32_N64_w8 default
- Triton lowered ttg.local_load inside loop (8 instructions) in M32_N64_w8 4
- Triton lowered ttg.local_load inside loop (14 instructions) in M32_N128_w4 default
- Triton lowered ttg.local_load inside loop (22 instructions) in M32_N128_w4 4

### Detail on Confound 1: Residency & Occupancy Disparity (Criterion F)
- **Positive Case (`M32_N64_w8`)**:
  - `default`: **39 registers**, 5120B dynamic smem. Active Blocks/SM: **6**, Active Warps/SM: **48** (75% occupancy).
  - `cand4`: **29 registers**, 6144B dynamic smem. Active Blocks/SM: **8**, Active Warps/SM: **64** (100% occupancy).
  - **Occupancy Disparity**: **25% reduction in active warps** for `default` vs `cand4` due to 39 vs 29 register allocation on SM90.
- **Control Case (`M32_N128_w4`)**:
  - `default`: **32 registers**, Active Blocks/SM: **16**, Active Warps/SM: **64** (100% occupancy).
  - `cand4`: **32 registers**, Active Blocks/SM: **16**, Active Warps/SM: **64** (100% occupancy).
  - Strictly matched.

### Detail on Confound 2: LocalLoad Sunk Inside Loop (Criterion B & C)
- Although `x = desc.load(...)` is defined before the loop, Triton's TMA lowering / loop pass pipeline places `ttg.local_load` inside `scf.for` right before the first use (`elementwise_inline_asm`).
- Consequently, `ld.shared` is repeated $K$ times rather than remaining an invariant 1x execution outside the loop.
- The inner loop scales **both LocalLoad memory bandwidth and reduction computation**, violating pure reduction isolation.

## 4. Hardware Verification & Implementation Milestones
- **Runtime Loop Emission**: Verified in all 4 conditions (`scf.for` in TTGIR, backward `bra` in PTX, `BRA` in SASS, `loop_body_copy_count = 1`).
- **Single CUBIN Binary Reuse**: Verified via JIT device caches; cache length remained 1 across all $K \in \{1, 2, 4, 8\}$, confirming zero recompilation.
- **Zero Spill**: 0 local bytes, 0 stack bytes in all conditions.
- **Numerical Correctness**: 100% bitwise/tolerance match across all $K$ against $K \times \text{max}(x, \text{dim}=1)$.
