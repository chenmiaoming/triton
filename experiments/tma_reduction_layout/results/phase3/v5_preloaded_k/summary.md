# Phase 3 Step B v5: Preloaded-Register Last-Result Carry Feasibility Report

**Overall Feasibility Status**: `REDUCTION_ISOLATED_BUT_TEMPLATE_CONFOUNDED`

## 1. Executive Summary

Phase 3 Step B v5 evaluates the feasibility of eliminating the candidate-dependent accumulator adds
by carrying only the last reduction result across the runtime `k_iters` loop.

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

## 4. Part C: Phase 3 Step B v5 Preloaded-Register Loop Decomposition

| Config | Cand | Loop Insts | Pre-Loop Loads | Sunk In-Loop Loads | In-Loop Anti-CSE | Canonical Red | Acc Adds | Global Stores | Loop Ctrl |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `M32_N64_w8` | `default` | 126 | **1** | **0** | 4 | 119 | **0** | **0** | 3 |
| `M32_N64_w8` | `4` | 69 | **2** | **0** | 4 | 62 | **0** | **0** | 3 |
| `M32_N128_w4` | `default` | 121 | **4** | **0** | 16 | 102 | **0** | **0** | 3 |
| `M32_N128_w4` | `4` | 74 | **8** | **0** | 16 | 55 | **0** | **0** | 3 |

## 5. Part C: v5 Occupancy, Residency & Register Pressure Delta (v4 vs v5)

| Config | Candidate | CUBIN SHA256 (prefix) | v4 Regs | v5 Regs | Delta | Dynamic Smem | Blocks/SM | Active Warps/SM | Residency Match |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `M32_N64_w8` | `default` | `852024b6547b` | 40 | 32 | -8 | 4104 B | **8** | 64 | MATCH |
| `M32_N64_w8` | `4` | `bcc961699fa4` | 32 | 27 | -5 | 4104 B | **8** | 64 | MATCH |
| `M32_N128_w4` | `default` | `2ade9df8c706` | 40 | 32 | -8 | 8200 B | **16** | 64 | MATCH |
| `M32_N128_w4` | `4` | `b99d388cb272` | 29 | 27 | -2 | 8200 B | **16** | 64 | MATCH |

## 6. Pre-Registered Criteria Evaluation

| Criterion | Description | Status |
| :--- | :--- | :--- |
| `Criterion A (TMA Issue Once Outside Loop)` | Structural requirement | **`PASS`** |
| `Criterion B (Initial LocalLoad Once Outside Loop)` | Structural requirement | **`PASS`** |
| `Criterion C (Zero Initial LocalLoad Inside Loop)` | Structural requirement | **`PASS`** |
| `Criterion D (One Runtime Loop / One CUBIN)` | Structural requirement | **`PASS`** |
| `Criterion E (Runtime K Unspecialized)` | Structural requirement | **`PASS`** |
| `Criterion F (Distributed Layout Invariance)` | Structural requirement | **`PASS`** |
| `Criterion G (Canonical Reduction Core Fingerprint Equivalence)` | Structural requirement | **`FAIL`** |
| `Criterion H (Reduction Remains Inside Runtime Loop)` | Structural requirement | **`PASS`** |
| `Criterion I (Zero Accumulator Adds Inside Loop)` | Structural requirement | **`PASS`** |
| `Criterion J (Zero Global Store Inside Loop)` | Structural requirement | **`PASS`** |
| `Criterion K (In-Loop Anti-CSE Region Candidate Symmetry)` | Structural requirement | **`PASS`** |
| `Criterion L (Zero Local Memory & Stack Spills)` | Structural requirement | **`PASS`** |
| `Criterion M (Numerical Correctness across K)` | Structural requirement | **`PASS`** |
| `Criterion N (Residency & Occupancy Matched)` | Structural requirement | **`PASS`** |
| `Criterion O (Dynamic Smem Limiter Disambiguation)` | Structural requirement | **`PASS`** |

## 7. Conclusions & Findings

- **Overall Status**: `REDUCTION_ISOLATED_BUT_TEMPLATE_CONFOUNDED`.
- **Accumulator Elimination**: Criterion I is `PASS`. All per-iteration FP32 accumulator adds were eliminated.
- **Reduction Placement**: Criterion H is `PASS`.
- **Residency & Occupancy Status**: Criterion N is `PASS`. 
- **Hypothesis H2 Status**: Strictly remains **`UNVERIFIED`**.
- **Timing Prohibition**: No timing measurements, CUDA events, elapsed times, or marginal slopes were evaluated.
