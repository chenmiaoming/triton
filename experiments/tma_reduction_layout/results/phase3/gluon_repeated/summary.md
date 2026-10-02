# Phase 3 Step D.1: Gluon Repeated-Reduction Isolation Feasibility & Barrier Audit

**Overall Feasibility Status**: `GLUON_REDUCTION_AMPLIFICATION_TIMING_READY_WITH_COMPILER_BARRIER`
**Timing Gate Status**: `PASS_UNLOCKED_FOR_STEP_E`

## 1. Executive Summary

Phase 3 Step D.1 completes the formal compiler-barrier audit and timing gate verification
for Gluon repeated reduction isolation across `R in {1, 2, 4, 8}`.

> [!NOTE] Compiler Barrier Classification
> The input anti-LICM tied barrier is **not PTX-zero**: it induces candidate-symmetric `mov.b16` register copies
> (8 copies for `M32_N64_w8`, 32 copies for `M32_N128_w4`).
> SASS inspection: **no explicit MOV or IMAD.MOV was observed** in the runtime reduction region.
> Allocation, live-range, and scheduler effects remain unmeasured; zero total overhead is not established.

## 2. Hardware Limits & Target Device (H100 SM90)

| Attribute | Value | Description |
| :--- | :--- | :--- |
| `max_threads_per_sm` | 2048 | Max threads per Multiprocessor |
| `max_registers_per_sm` | 65536 | Total 32-bit registers per SM |
| `max_shared_memory_per_sm` | 233472 bytes | Total addressable shared memory per SM |
| `warp_size` | 32 | Hardware threads per warp |
| `max_warps_per_sm` | 64 | Derived maximum warp concurrency |

## 3. Barrier & Reduction Decomposition across Specializations

| Config | Candidate | Role | Loop Range | Branches | Input Barrier PTX Copies | In-Asm PTX | SASS Loop MOVs | SASS Observation | Reduction Stratification | Extra Ops |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `M32_N64_w8` | `default` | `PRIMARY` | `L125..L364` | **1** | **8 × mov.b16** | **0** | **0** | **NO_EXPLICIT_LOOP_MOV_OBSERVED** | `EXACT_SEQUENCE_EQUIVALENT` | **0** |
| `M32_N64_w8` | `4` | `PRIMARY` | `L124..L242` | **1** | **8 × mov.b16** | **0** | **0** | **NO_EXPLICIT_LOOP_MOV_OBSERVED** | `EXACT_SEQUENCE_EQUIVALENT` | **0** |
| `M32_N128_w4` | `default` | `SECONDARY_CONTROL` | `L152..L354` | **1** | **32 × mov.b16** | **0** | **0** | **NO_EXPLICIT_LOOP_MOV_OBSERVED** | `PIPELINED_OPCODE_EQUIVALENT` | **0** |
| `M32_N128_w4` | `4` | `SECONDARY_CONTROL` | `L142..L272` | **1** | **32 × mov.b16** | **0** | **0** | **NO_EXPLICIT_LOOP_MOV_OBSERVED** | `EXACT_SEQUENCE_EQUIVALENT` | **0** |

## 4. Physical Resources & SM Occupancy (H100 SM90)

| Config | Candidate | CUBIN SHA (prefix) | Regs | Local / Stack | Dynamic Smem | Blocks/SM | Active Warps/SM | Residency Match | Single Binary Across R |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `M32_N64_w8` | `default` | `8e295a532181` | 32 | 0 B / 0 B | 4104 B | **8** | 64 | **MATCHED** | **`PASS`** |
| `M32_N64_w8` | `4` | `2c938c8251cc` | 23 | 0 B / 0 B | 4104 B | **8** | 64 | **MATCHED** | **`PASS`** |
| `M32_N128_w4` | `default` | `c3a82950d704` | 32 | 0 B / 0 B | 8200 B | **16** | 64 | **MATCHED** | **`PASS`** |
| `M32_N128_w4` | `4` | `88c234adc884` | 24 | 0 B / 0 B | 8200 B | **16** | 64 | **MATCHED** | **`PASS`** |

## 5. Pre-Registered Criteria Evaluation (Criteria A Through L)

| Criterion | Description | Status | Rationale |
| :--- | :--- | :---: | :--- |
| `Criterion A (TMA Once Outside Loop)` | Structural requirement | **`PASS`** | Exactly 1 ttng.async_tma_copy_global_to_local before scf.for |
| `Criterion B (Initial LocalLoad Once Outside Loop, Zero Inside)` | Structural requirement | **`PASS`** | Primary TTGIR: 1 ttg.local_load outside scf.for, 0 inside |
| `Criterion C (Runtime R Unspecialized, Single Binary)` | Structural requirement | **`PASS`** | Single unspecialized binary reused across all R in {1,2,4,8} |
| `Criterion D (One Runtime Loop, One Static Canonical Body)` | Structural requirement | **`PASS`** | One compiler runtime reduction backedge; separate TMA polling edge is also present |
| `Criterion E (Input Anti-LICM Barrier Emits Zero Instructions)` | Structural requirement | **`FAIL_AT_PTX_LEVEL`** | FAIL_AT_PTX_LEVEL: 0 explicit asm, 8/32 induced mov.b16 copies (candidate-symmetric; no explicit loop MOV observed) |
| `Criterion F (Result Sink Emits Zero Instructions)` | Structural requirement | **`PASS`** | 0 explicit asm and 0 induced copies |
| `Criterion G (Complete Filtered Reduction Fingerprint Equivalence)` | Structural requirement | **`PASS_STRATIFIED`** | PASS_STRATIFIED: PRIMARY w8 is EXACT_SEQUENCE_EQUIVALENT; SECONDARY w4 def is PIPELINED_OPCODE_EQUIVALENT (opcode multiset only) |
| `Criterion H (Terminal Canonical ld.shared Remains Inside Loop)` | Structural requirement | **`PASS`** | Terminal shared exchange ld.shared verified inside runtime loop |
| `Criterion I (Zero Accumulator / Global Store Inside Loop)` | Structural requirement | **`PASS`** | Zero accumulator adds and zero global stores inside loop |
| `Criterion J (Same-Config Residency & Occupancy Matched)` | Structural requirement | **`PASS`** | Blocks/SM and active warps/SM identical between default and cand4 |
| `Criterion K (Zero Local Memory & Stack Spills)` | Structural requirement | **`PASS`** | 0 local memory bytes, 0 stack bytes |
| `Criterion L (Identical CUBIN Across R in {1,2,4,8})` | Structural requirement | **`PASS`** | Identical CUBIN SHA256 across all R values |

## 6. Archived Timing Gate Verification (11 Conditions)

| # | Condition | Result | Notes |
| :-: | :--- | :---: | :--- |
| 1 | M32_N64_w8 default exact canonical reduction sequence | **`PASS`** | Complete filtered sequence match verified |
| 2 | M32_N64_w8 cand4 exact canonical reduction sequence | **`PASS`** | Complete filtered sequence match verified |
| 3 | Input barrier PTX copies candidate-symmetric | **`PASS`** | w8: 8 == 8; w4: 32 == 32 |
| 4 | No explicit MOV/IMAD.MOV in runtime region | **`PASS`** | NO_EXPLICIT_LOOP_MOV_OBSERVED |
| 5 | Result sink empty PTX and no copies | **`PASS`** | 0 explicit PTX / 0 copies |
| 6 | Default / cand4 blocks/SM matched | **`PASS`** | w8: 8 blk/SM, w4: 16 blk/SM |
| 7 | Active warps/SM matched | **`PASS`** | 64 warps/SM across all candidates |
| 8 | Zero local memory and stack spills | **`PASS`** | LOCAL=0, STACK=0 |
| 9 | One CUBIN per candidate | **`PASS`** | Single binary across all conditions |
| 10 | Runtime R unspecialized | **`PASS`** | do_not_specialize=['num_reductions'] |
| 11 | All structural conditions | **`PASS`** | Complete artifact audit required |

**Gate Verdict**: **`PASS_UNLOCKED_FOR_STEP_E`** -> archived Step E structural protocol passes; no new timing is authorized.

## 7. Conclusions & Findings

- **Overall Feasibility Status**: `GLUON_REDUCTION_AMPLIFICATION_TIMING_READY_WITH_COMPILER_BARRIER`.
- **Compiler Barrier Overhead**: Prototype X induces candidate-symmetric `mov.b16` register copies at PTX level (Criterion E = `FAIL_AT_PTX_LEVEL`), no explicit loop MOV/IMAD.MOV is observed; indirect compiler effects remain possible.
- **Reduction Fingerprint Stratification**: PRIMARY configuration (`M32_N64_w8`) exhibits exact canonical reduction sequence match in both default and cand4. SECONDARY configuration (`M32_N128_w4`) exhibits pipelined opcode equivalence for default with an opcode multiset match; register-dependency topology is not established.
- **Timing Gate Cleared**: All structural and observational checks pass for the archived Step E protocol.
- **Hypothesis H2 Status**: This stage supplies structural evidence; the current scientific decision is reported in Step E.


Exact sequence compares the complete filtered normalized reduction fingerprint, including selected opcodes and shuffle/barrier immediates. Most operands and predicates are ignored; this is not dataflow/full PTX/SASS/CUBIN equality. Secondary opcode multiset equality does not prove topology.
All structural conditions, complete loop memory signature, resources, and recorded binary bindings are required. The archived gate authorizes no new timing. Indirect live-range, allocation, and scheduler effects remain possible.