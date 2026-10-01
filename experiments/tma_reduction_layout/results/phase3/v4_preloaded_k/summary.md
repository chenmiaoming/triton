# Phase 3 Step B v4: Preloaded-Register Runtime-K Isolation Feasibility Report

**Overall Feasibility Status**: `LOCALLOAD_ISOLATED_BUT_RESIDENCY_CONFOUNDED`

## 1. Executive Summary

Phase 3 Step B v4 evaluates the feasibility of strictly isolating the TMA reduction loop body
by preloading the input tile from shared memory into registers prior to entering the runtime `k_iters` loop.
An un-CSE-able assembly barrier (`tl.inline_asm_elementwise("mov.b32 $0, $1;", ...)` with `pack=2`)
is inserted both before the loop (to force shared-to-register load materialization) and inside the loop
(to prevent the compiler from lifting or eliminating per-iteration reduction compute).

## 2. Hardware Device Limits (H100 SM90 via Driver API)

| Attribute | Value | Driver Code |
| :--- | :--- | :--- |
| `max_threads_per_sm` | 2048 | CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR (39) |
| `max_threads_per_block` | 1024 | CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X (1) |
| `max_registers_per_block` | 65536 | CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK (12) |
| `max_registers_per_sm` | 65536 | CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR (82) |
| `max_shared_memory_per_sm` | 233472 bytes | CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR (81) |
| `warp_size` | 32 | CU_DEVICE_ATTRIBUTE_WARP_SIZE (10) |
| `max_warps_per_sm` | 64 | Derived (`max_threads_per_sm // warp_size`) |

## 3. Part B: Canonical Step A Specialization Occupancy Baseline

| Config | Candidate | CUBIN SHA256 (prefix) | Regs | Dynamic Smem | Blocks/SM (Actual) | Blocks/SM (Zero Smem) | Active Warps/SM | Limiter |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `M32_N64_w8` | `default` | `437e67488632` | 29 | 4104 B | **8** | 8 | 64 | Registers / Warps |
| `M32_N64_w8` | `4` | `3bb96f9be2f8` | 22 | 4104 B | **8** | 8 | 64 | Registers / Warps |
| `M32_N128_w4` | `default` | `de7b2828edc5` | 32 | 8200 B | **16** | 16 | 64 | Registers / Warps |
| `M32_N128_w4` | `4` | `d5327bfa02fe` | 25 | 8200 B | **16** | 16 | 64 | Registers / Warps |

## 4. Part C: Phase 3 Step B v4 Preloaded-Register Loop Decomposition

| Config | Cand | Loop Insts | Pre-Loop Loads | In-Loop Loads | In-Loop Anti-CSE | Canonical Red | Acc Adds | Loop Ctrl |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `M32_N64_w8` | `default` | 138 | **1** | **0** | 4 | 123 | 8 | 3 |
| `M32_N64_w8` | `4` | 75 | **2** | **0** | 4 | 64 | 4 | 3 |
| `M32_N128_w4` | `default` | 131 | **4** | **0** | 16 | 104 | 8 | 3 |
| `M32_N128_w4` | `4` | 80 | **8** | **0** | 16 | 57 | 4 | 3 |

## 5. Part C: v4 Occupancy & Residency Comparison

| Config | Candidate | CUBIN SHA256 (prefix) | Regs | Dynamic Smem | Blocks/SM (Actual) | Blocks/SM (Zero Smem) | Active Warps/SM | Residency Match |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `M32_N64_w8` | `default` | `5c7127efd838` | 40 | 4104 B | **6** | 6 | 48 | DISPARITY |
| `M32_N64_w8` | `4` | `a2d3fc5d517e` | 32 | 4104 B | **8** | 8 | 64 | DISPARITY |
| `M32_N128_w4` | `default` | `b17701f7afc3` | 40 | 8200 B | **12** | 12 | 48 | DISPARITY |
| `M32_N128_w4` | `4` | `594870919a42` | 29 | 8200 B | **16** | 16 | 64 | DISPARITY |

## 6. Pre-Registered Criteria Evaluation

| Criterion | Description | Status |
| :--- | :--- | :--- |
| `Criterion A (TMA Descriptor Load Invariant)` | Structural requirement | **`PASS`** |
| `Criterion B (Initial LocalLoad Invariant - Inside Loop == 0)` | Structural requirement | **`PASS`** |
| `Criterion C (Canonical Reduction Body Template Equivalence)` | Structural requirement | **`PASS`** |
| `Criterion D (Distributed Layout Invariance)` | Structural requirement | **`PASS`** |
| `Criterion E (Self-Contained Complete Executable Artifacts)` | Structural requirement | **`PASS`** |
| `Criterion F (Residency & Occupancy Matched)` | Structural requirement | **`FAIL_RESIDENCY_DISPARITY`** |
| `Criterion G (Loop Body Single Copy)` | Structural requirement | **`PASS`** |
| `Criterion H (Single-Binary Runtime-K Invariance)` | Structural requirement | **`PASS`** |
| `Criterion I (Numerical Correctness Across K)` | Structural requirement | **`PASS`** |
| `Criterion J (Sunk LocalLoad Count Inside Loop == 0)` | Structural requirement | **`PASS`** |
| `Criterion K (Anti-CSE Barrier Symmetry)` | Structural requirement | **`PASS`** |
| `Criterion L (Accumulator Structural Consistency)` | Structural requirement | **`PASS`** |
| `Criterion M (Zero Local Memory & Stack Spills)` | Structural requirement | **`PASS`** |
| `Criterion N (Dynamic Smem Limiter Disambiguation)` | Structural requirement | **`PASS`** |

## 7. Conclusions & Findings

- **LocalLoad Isolation Status**: Criterion B and Criterion J PASSED. The initial tile shared-to-register load is strictly outside the runtime loop, achieving 0 tile loads inside the loop.
- **Residency & Occupancy Status**: Criterion F is `FAIL_RESIDENCY_DISPARITY`. 
- **Hypothesis H2 Status**: Strictly remains **`UNVERIFIED`**.
- **Timing Prohibition**: No timing measurements, CUDA events, elapsed times, or marginal slopes were evaluated.
