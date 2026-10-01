# Phase 3 Step D: Gluon Repeated-Reduction Isolation Feasibility Report

**Overall Feasibility Status**: `GLUON_REDUCTION_AMPLIFICATION_FEASIBLE`

## 1. Executive Summary

Phase 3 Step D investigates whether Gluon can repeat the exact canonical reduction body
R times (R in {1, 2, 4, 8}) in a single unspecialized binary while keeping TMA loads,
initial LocalLoads, SM residency, and auxiliary runtime work strictly invariant.

## 2. Hardware Limits & Target Device (H100 SM90)

| Attribute | Value | Description |
| :--- | :--- | :--- |
| `max_threads_per_sm` | 2048 | Max threads per Multiprocessor |
| `max_registers_per_sm` | 65536 | Total 32-bit registers per SM |
| `max_shared_memory_per_sm` | 233472 bytes | Total addressable shared memory per SM |
| `warp_size` | 32 | Hardware threads per warp |
| `max_warps_per_sm` | 64 | Derived maximum warp concurrency |

## 3. Structural Decomposition across Specializations

| Config | Candidate | CUBIN SHA (prefix) | Pre-Loop TMA | Pre-Loop LocalLoad | Loop Range | Reduction Core inside Loop | In-Loop LocalLoad | In-Loop Barrier Insts | Extra In-Loop Ops |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `M32_N64_w8` | `default` | `8e295a532181` | **1** | **1** | `L125..L364` | **`MATCH (L138..L356)`** | **0** | **0** | **0** |
| `M32_N64_w8` | `4` | `2c938c8251cc` | **1** | **2** | `L124..L242` | **`MATCH (L139..L234)`** | **0** | **0** | **0** |
| `M32_N128_w4` | `default` | `c3a82950d704` | **1** | **4** | `L152..L354` | **`MATCH (L191..L346)`** | **0** | **0** | **0** |
| `M32_N128_w4` | `4` | `88c234adc884` | **1** | **8** | `L142..L272` | **`MATCH (L181..L264)`** | **0** | **0** | **0** |

## 4. Physical Resources & SM Occupancy (H100 SM90)

| Config | Candidate | CUBIN SHA (prefix) | Regs | Local / Stack | Dynamic Smem | Blocks/SM | Active Warps/SM | Residency Match | Single Binary Across R |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `M32_N64_w8` | `default` | `8e295a532181` | 32 | 0 B / 0 B | 4104 B | **8** | 64 | **MATCHED** | **`PASS`** |
| `M32_N64_w8` | `4` | `2c938c8251cc` | 23 | 0 B / 0 B | 4104 B | **8** | 64 | **MATCHED** | **`PASS`** |
| `M32_N128_w4` | `default` | `c3a82950d704` | 32 | 0 B / 0 B | 8200 B | **16** | 64 | **MATCHED** | **`PASS`** |
| `M32_N128_w4` | `4` | `88c234adc884` | 24 | 0 B / 0 B | 8200 B | **16** | 64 | **MATCHED** | **`PASS`** |

## 5. Pre-Registered Criteria Evaluation (Criteria A Through L)

| Criterion | Description | Status |
| :--- | :--- | :--- |
| `Criterion A (TMA Once Outside Loop)` | Structural requirement | **`PASS`** |
| `Criterion B (Initial LocalLoad Once Outside Loop, Zero Inside)` | Structural requirement | **`PASS`** |
| `Criterion C (Runtime R Unspecialized, Single Binary)` | Structural requirement | **`PASS`** |
| `Criterion D (One Runtime Loop, One Static Canonical Body)` | Structural requirement | **`PASS`** |
| `Criterion E (Input Anti-LICM Barrier Emits Zero Instructions)` | Structural requirement | **`PASS`** |
| `Criterion F (Result Sink Emits Zero Instructions)` | Structural requirement | **`PASS`** |
| `Criterion G (Exact Canonical Reduction Fingerprint Inside Loop)` | Structural requirement | **`PASS`** |
| `Criterion H (Terminal Canonical ld.shared Remains Inside Loop)` | Structural requirement | **`PASS`** |
| `Criterion I (Zero Accumulator / Global Store Inside Loop)` | Structural requirement | **`PASS`** |
| `Criterion J (Same-Config Residency & Occupancy Matched)` | Structural requirement | **`PASS`** |
| `Criterion K (Zero Local Memory & Stack Spills)` | Structural requirement | **`PASS`** |
| `Criterion L (Identical CUBIN Across R in {1,2,4,8})` | Structural requirement | **`PASS`** |

## 6. Conclusions & Findings

- **Overall Feasibility Status**: `GLUON_REDUCTION_AMPLIFICATION_FEASIBLE`.
- **Compiler Barrier Overhead**: Prototype X (input tied constraint) and Prototype Y (result sink) emit exactly 0 machine instructions in both PTX and SASS.
- **Reduction Core Isolation**: The complete canonical reduction topology (including cross-warp exchanges and terminal shared reload) executes wholly inside the runtime loop.
- **Single-Binary Invariance**: A single CUBIN serves R in {1, 2, 4, 8} without specialization.
- **Residency & Occupancy**: Full theoretical occupancy (64 warps/SM) with 0 spills and identical blocks/SM for default vs cand4.
- **Hypothesis H2 Status**: Strictly remains **`UNVERIFIED`** (NO timing or benchmarking was conducted).
