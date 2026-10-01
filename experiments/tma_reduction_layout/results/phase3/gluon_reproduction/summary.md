# Phase 3 Step C: Gluon Canonical Structural Reproduction Report

**Overall Reproduction Status**: `GLUON_CANONICAL_REPRODUCTION_SUCCESS`

## 1. Executive Summary

Phase 3 Step C determines whether the built-in Gluon language can explicitly reproduce the
TMA load, shared memory layout, shared-to-register load lowering, distributed register layout,
and reduction topology of Canonical Step A without opaque hacks or handwritten PTX.

## 2. Hardware Limits & Target GPU

| Attribute | Value | Driver Description |
| :--- | :--- | :--- |
| `max_threads_per_sm` | 2048 | Max threads per Multiprocessor |
| `max_registers_per_sm` | 65536 | Total 32-bit registers per SM |
| `max_shared_memory_per_sm` | 233472 bytes | Total addressable shared memory per SM |
| `warp_size` | 32 | Hardware threads per warp |
| `max_warps_per_sm` | 64 | Derived maximum warp concurrency |

## 3. Level 1 — Layout Equivalence & Mapping Verification

| Config | Candidate | Explicit Gluon BlockedLayout | Canonical Layout String | Mapping Match |
| :--- | :--- | :--- | :--- | :---: |
| `M32_N64_w8` | `default` | `#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>` | `#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>` | **`MATCH`** |
| `M32_N64_w8` | `4` | `#blocked = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>` | `#blocked = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>` | **`MATCH`** |
| `M32_N128_w4` | `default` | `#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>` | `#blocked = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 2, 16], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>` | **`MATCH`** |
| `M32_N128_w4` | `4` | `#blocked = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>` | `#blocked = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>` | **`MATCH`** |

### Hardware View & Lane/Warp Partitioning

Tensor axes are `[B, M, N]` (rank 3 in Gluon: `[1, M, N]`).
- `lanePart[M] = threadsPerWarp[1]`
- `warpPart[M] = warpsPerCTA[1]`
- `N-lane partition = threadsPerWarp[2]`

| Configuration | Candidate | `sizePerThread` | `threadsPerWarp` | `warpsPerCTA` | `order` | `lanePart[M]` | `warpPart[M]` | N-lane partition |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| `M32_N64_w8` | `default` | `[1, 1, 8]` | `[1, 4, 8]` | `[1, 8, 1]` | `[2, 1, 0]` | 4 | 8 | 8 |
| `M32_N64_w8` | `4` | `[1, 1, 4]` | `[1, 2, 16]` | `[1, 8, 1]` | `[2, 1, 0]` | 2 | 8 | 16 |
| `M32_N128_w4` | `default` | `[1, 1, 8]` | `[1, 2, 16]` | `[1, 4, 1]` | `[2, 1, 0]` | 2 | 4 | 16 |
| `M32_N128_w4` | `4` | `[1, 1, 4]` | `[1, 1, 32]` | `[1, 4, 1]` | `[2, 1, 0]` | 1 | 4 | 32 |

## 4. LocalLoad Instruction Structural Verification

| Config | Candidate | Expected LocalLoad | Actual Gluon LocalLoad | Structural Match |
| :--- | :--- | :--- | :--- | :---: |
| `M32_N64_w8` | `default` | 1 × ld.shared.v4.b32 | 1 × ld.shared.v4.b32 | **`MATCH`** |
| `M32_N64_w8` | `4` | 2 × ld.shared.v2.b32 | 2 × ld.shared.v2.b32 | **`MATCH`** |
| `M32_N128_w4` | `default` | 4 × ld.shared.v4.b32 | 4 × ld.shared.v4.b32 | **`MATCH`** |
| `M32_N128_w4` | `4` | 8 × ld.shared.v2.b32 | 8 × ld.shared.v2.b32 | **`MATCH`** |

## 5. Level 2 — Reduction Structural Equivalence (Contiguous Subsequence Match)

| Config | Candidate | Canonical Reduction Ops | Gluon Reduction Ops | Contiguous Range | Match Count | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `M32_N64_w8` | `default` | 104 | 104 | `L106..L338` | 1 | **`REDUCTION_STRUCTURALLY_EQUIVALENT`** |
| `M32_N64_w8` | `4` | 46 | 46 | `L104..L224` | 1 | **`REDUCTION_STRUCTURALLY_EQUIVALENT`** |
| `M32_N128_w4` | `default` | 84 | 84 | `L118..L272` | 1 | **`REDUCTION_STRUCTURALLY_EQUIVALENT`** |
| `M32_N128_w4` | `4` | 42 | 42 | `L124..L216` | 1 | **`REDUCTION_STRUCTURALLY_EQUIVALENT`** |

## 6. Physical Resources & SM Occupancy (H100 SM90)

| Config | Candidate | CUBIN SHA256 (prefix) | Regs | Local / Stack | Smem | Blocks/SM | Active Warps/SM | Residency vs Counterpart |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `M32_N64_w8` | `default` | `e9942d7355af` | 27 | 0 B / 0 B | 4104 B | **8** | 64 | **MATCHED** |
| `M32_N64_w8` | `4` | `fa86ff2e5a5e` | 21 | 0 B / 0 B | 4104 B | **8** | 64 | **MATCHED** |
| `M32_N128_w4` | `default` | `912a815196a6` | 29 | 0 B / 0 B | 8200 B | **16** | 64 | **MATCHED** |
| `M32_N128_w4` | `4` | `5f73e0e617fb` | 24 | 0 B / 0 B | 8200 B | **16** | 64 | **MATCHED** |

## 7. Pre-Registered Acceptance Criteria A Through H

| Criterion | Description | Status |
| :--- | :--- | :--- |
| `Criterion A (Explicit Distributed Layout Attribute Equivalence)` | Structural requirement | **`PASS`** |
| `Criterion B (Shared Layout NVMMA Mapping Equivalence)` | Structural requirement | **`PASS`** |
| `Criterion C (TMA Count == 1)` | Structural requirement | **`PASS`** |
| `Criterion D (Explicit smem.load Generates Expected LocalLoad)` | Structural requirement | **`PASS`** |
| `Criterion E (Reduction Core Normalized Fingerprint Equivalence)` | Structural requirement | **`PASS`** |
| `Criterion F (Zero Local Memory & Stack Spills)` | Structural requirement | **`PASS`** |
| `Criterion G (Same-Config Residency & Occupancy Matched)` | Structural requirement | **`PASS`** |
| `Criterion H (Numerical Correctness)` | Structural requirement | **`PASS`** |

## 8. Conclusions & Findings

- **Overall Status**: `GLUON_CANONICAL_REPRODUCTION_SUCCESS`.
- **Explicit Layout Representation**: Gluon's `gl.BlockedLayout` directly expresses the canonical distributed layouts without compiler inference.
- **Shared-Memory LocalLoad Lowering**: `smem.load(register_layout)` generates the exact target LocalLoad instruction families (vector widths and counts) instruction-for-instruction.
- **Reduction Topology Equivalence**: Gluon's native `gl.max` along axis 1 compiles to the exact canonical reduction sequence across all specializations.
- **Residency & Occupancy**: Full 100% theoretical occupancy (64 warps/SM) is achieved across all 4 specializations with 0 local memory or stack spills.
- **Hypothesis H2 Status**: Strictly remains **`UNVERIFIED`** (no runtime-K amplification or timing was performed).
