# PHASE 5 ARTIFACT GATE REPORT

Starting HEAD: `113748ae8f8e4817f8070847156f885c7a92a860`.

Held-out confirmatory extension, artifact/coverage stage only. Stage C remains development data. No timing, latency, throughput, slopes, G/D/g0, locked-model prediction or hypothesis outcome evaluation.

All 19 structural included cases attempted; 114 original binary attempts. Five structural excluded cases never compiled. Counts: `{"EXCLUDE_FROM_TIMING": 6, "PENDING": 0, "PRIMARY": 2, "SECONDARY": 11}`.

GPU UUID: `GPU-05ce4fb6-9787-c93e-86b3-61f430adf376`; NVIDIA H100 80GB HBM3; CC=[9, 0].

Toolchain and actual image/build identities: environment.json, modal_dispatch.json and source_bindings.json. Actual compiler ptxas: `ptxas: NVIDIA (R) Ptx optimizing assembler; Copyright (c) 2005-2025 NVIDIA Corporation; Built on Tue_May_27_02:18:05_PDT_2025; Cuda compilation tools, release 12.9, V12.9.86; Build cuda_12.9.r12.9/compiler.36037853_0`.

| Case | Class | Single d/4 body | Repeated d/4 body | Repeated preload d/4 | Registers canonical/single/repeated d→4 | Blocks/SM canonical/single/repeated d→4 | Reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M16_N128_w4 | SECONDARY | EXACT_SEQUENCE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | PIPELINED_OPCODE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | EXACT/DIFFERENT_ENCODING | canonical:32→20; single:28→21; repeated:32→23 | canonical:16→16; single:16→16; repeated:16→16 | ALL_GATES_PASS |
| M16_N128_w8 | SECONDARY | EXACT_SEQUENCE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | PIPELINED_OPCODE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | EXACT/DIFFERENT_ENCODING | canonical:32→21; single:27→21; repeated:32→25 | canonical:8→8; single:8→8; repeated:8→8 | ALL_GATES_PASS |
| M16_N64_w4 | SECONDARY | PIPELINED_OPCODE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | PIPELINED_OPCODE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | EXACT/DIFFERENT_ENCODING | canonical:28→22; single:27→20; repeated:28→22 | canonical:16→16; single:16→16; repeated:16→16 | ALL_GATES_PASS |
| M256_N128_w4 | EXCLUDE_FROM_TIMING | EXACT_SEQUENCE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | REDUCTION_FINGERPRINT_MISMATCH/EXACT_SEQUENCE_EQUIVALENT | MISMATCH/DIFFERENT_ENCODING | canonical:36→32; single:38→31; repeated:41→36 | canonical:3→3; single:3→3; repeated:3→3 | REPEATED_EXTRA_MEMORY_EFFECT, REPEATED_REDUCTION_MISMATCH, REPEATED_TILE_RELOAD |
| M256_N128_w8 | SECONDARY | EXACT_SEQUENCE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | PIPELINED_OPCODE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | EXACT/DIFFERENT_ENCODING | canonical:33→39; single:33→36; repeated:35→31 | canonical:3→3; single:3→3; repeated:3→3 | ALL_GATES_PASS |
| M256_N16_w4 | SECONDARY | PIPELINED_OPCODE_EQUIVALENT/PIPELINED_OPCODE_EQUIVALENT | PIPELINED_OPCODE_EQUIVALENT/PIPELINED_OPCODE_EQUIVALENT | EXACT/EXACT | canonical:31→24; single:29→24; repeated:30→23 | canonical:16→16; single:16→16; repeated:16→16 | ALL_GATES_PASS |
| M256_N16_w8 | SECONDARY | PIPELINED_OPCODE_EQUIVALENT/PIPELINED_OPCODE_EQUIVALENT | PIPELINED_OPCODE_EQUIVALENT/PIPELINED_OPCODE_EQUIVALENT | EXACT/EXACT | canonical:31→22; single:28→19; repeated:30→21 | canonical:8→8; single:8→8; repeated:8→8 | ALL_GATES_PASS |
| M256_N32_w4 | EXCLUDE_FROM_TIMING | PIPELINED_OPCODE_EQUIVALENT/PIPELINED_OPCODE_EQUIVALENT | PIPELINED_OPCODE_EQUIVALENT/PIPELINED_OPCODE_EQUIVALENT | EXACT/EXACT | canonical:32→32; single:39→32; repeated:33→33 | canonical:13→13; single:12→13; repeated:12→12 | RESIDENCY_MISMATCH |
| M256_N32_w8 | SECONDARY | PIPELINED_OPCODE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | PIPELINED_OPCODE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | EXACT/EXACT | canonical:29→22; single:30→22; repeated:29→23 | canonical:8→8; single:8→8; repeated:8→8 | ALL_GATES_PASS |
| M256_N64_w4 | SECONDARY | PIPELINED_OPCODE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | PIPELINED_OPCODE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | EXACT/DIFFERENT_ENCODING | canonical:42→35; single:41→32; repeated:42→32 | canonical:6→6; single:6→6; repeated:6→6 | ALL_GATES_PASS |
| M256_N64_w8 | PRIMARY | EXACT_SEQUENCE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | EXACT_SEQUENCE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | EXACT/DIFFERENT_ENCODING | canonical:32→36; single:39→37; repeated:39→30 | canonical:6→6; single:6→6; repeated:6→6 | ALL_GATES_PASS |
| M512_N128_w4 | EXCLUDE_FROM_TIMING | EXACT_SEQUENCE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | REDUCTION_FINGERPRINT_MISMATCH/EXACT_SEQUENCE_EQUIVALENT | MISMATCH/DIFFERENT_ENCODING | canonical:54→34; single:57→33; repeated:52→34 | canonical:1→1; single:1→1; repeated:1→1 | REPEATED_EXTRA_MEMORY_EFFECT, REPEATED_REDUCTION_MISMATCH, REPEATED_TILE_RELOAD |
| M512_N128_w8 | EXCLUDE_FROM_TIMING | EXACT_SEQUENCE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | REDUCTION_FINGERPRINT_MISMATCH/EXACT_SEQUENCE_EQUIVALENT | MISMATCH/DIFFERENT_ENCODING | canonical:53→35; single:38→34; repeated:43→33 | canonical:1→1; single:1→1; repeated:1→1 | REPEATED_EXTRA_MEMORY_EFFECT, REPEATED_REDUCTION_MISMATCH, REPEATED_TILE_RELOAD |
| M512_N16_w4 | EXCLUDE_FROM_TIMING | PIPELINED_OPCODE_EQUIVALENT/PIPELINED_OPCODE_EQUIVALENT | PIPELINED_OPCODE_EQUIVALENT/PIPELINED_OPCODE_EQUIVALENT | EXACT/EXACT | canonical:32→32; single:38→32; repeated:33→34 | canonical:13→13; single:12→13; repeated:12→12 | RESIDENCY_MISMATCH |
| M512_N16_w8 | SECONDARY | PIPELINED_OPCODE_EQUIVALENT/PIPELINED_OPCODE_EQUIVALENT | PIPELINED_OPCODE_EQUIVALENT/PIPELINED_OPCODE_EQUIVALENT | EXACT/EXACT | canonical:31→24; single:30→24; repeated:32→23 | canonical:8→8; single:8→8; repeated:8→8 | ALL_GATES_PASS |
| M512_N32_w4 | SECONDARY | PIPELINED_OPCODE_EQUIVALENT/PIPELINED_OPCODE_EQUIVALENT | PIPELINED_OPCODE_EQUIVALENT/PIPELINED_OPCODE_EQUIVALENT | EXACT/EXACT | canonical:42→45; single:41→46; repeated:42→46 | canonical:6→6; single:6→6; repeated:6→6 | ALL_GATES_PASS |
| M512_N32_w8 | SECONDARY | PIPELINED_OPCODE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | PIPELINED_OPCODE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | EXACT/EXACT | canonical:38→32; single:38→32; repeated:33→34 | canonical:6→6; single:6→6; repeated:6→6 | ALL_GATES_PASS |
| M512_N64_w4 | EXCLUDE_FROM_TIMING | PIPELINED_OPCODE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | PIPELINED_OPCODE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | EXACT/DIFFERENT_ENCODING | canonical:72→35; single:73→31; repeated:170→33 | canonical:3→3; single:3→3; repeated:2→3 | RESIDENCY_MISMATCH, SPILL |
| M512_N64_w8 | PRIMARY | EXACT_SEQUENCE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | EXACT_SEQUENCE_EQUIVALENT/EXACT_SEQUENCE_EQUIVALENT | EXACT/DIFFERENT_ENCODING | canonical:42→35; single:42→32; repeated:80→32 | canonical:3→3; single:3→3; repeated:3→3 | ALL_GATES_PASS |

Reuse frozen Stage A/B: matched candidate blocks/SM and active warps/SM for canonical, single and repeated.

Driver static SMEM, dynamic launch SMEM and cuobjdump SHARED stay separate; driver local bytes and raw cuobjdump LOCAL/STACK stay separate (driver local=LOCAL+STACK); nonzero LOCAL or STACK excludes; exact-CUBIN occupancy only.

Tied-copy mapping/count/symmetry and SASS loop MOV/IMAD.MOV remain observed; indirect effects UNISOLATED_NOT_ASSUMED_ZERO.

Unchanged frozen family: exact full sequence or exact full multiset; not operand/dataflow/whole-binary equivalence.

R=0 subtraction controls the fixed one-time differential, not absence of live-range/register/scheduling interactions.

Actual CUBIN bytes/SHAs, source stages, full LocalLoad sequences, body fingerprints, every backedge, tied-copy mappings and SMEM/resources/occupancy are retained for every exported bundle. All failed attempts and partial exports remain immutable.

## M512 resources and correctness

M512/N128 uses full B_DESC=65536, 8GiB BF16 input and 128KiB logical shared tile. No smaller descriptor, altered grid or resource workaround. Every smoke allocates the full descriptor and reads only four initialized prefix tiles. Repeated R0/R1 checks launch safety only.

M512_N128_w4: EXCLUDE_FROM_TIMING; REPEATED_EXTRA_MEMORY_EFFECT, REPEATED_REDUCTION_MISMATCH, REPEATED_TILE_RELOAD; resources `{"canonical": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 131080, "local_bytes": 0, "num_regs": 34, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 131080, "local_bytes": 0, "num_regs": 54, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}, "repeated": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 131080, "local_bytes": 0, "num_regs": 34, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 131080, "local_bytes": 0, "num_regs": 52, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}, "single": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 131080, "local_bytes": 0, "num_regs": 33, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 131080, "local_bytes": 0, "num_regs": 57, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}}`.

M512_N128_w8: EXCLUDE_FROM_TIMING; REPEATED_EXTRA_MEMORY_EFFECT, REPEATED_REDUCTION_MISMATCH, REPEATED_TILE_RELOAD; resources `{"canonical": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 131080, "local_bytes": 0, "num_regs": 35, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 131080, "local_bytes": 0, "num_regs": 53, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}, "repeated": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 131080, "local_bytes": 0, "num_regs": 33, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 131080, "local_bytes": 0, "num_regs": 43, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}, "single": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 131080, "local_bytes": 0, "num_regs": 34, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 131080, "local_bytes": 0, "num_regs": 38, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}}`.

M512_N16_w4: EXCLUDE_FROM_TIMING; RESIDENCY_MISMATCH; resources `{"canonical": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 16392, "local_bytes": 0, "num_regs": 32, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 16392, "local_bytes": 0, "num_regs": 32, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}, "repeated": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 16392, "local_bytes": 0, "num_regs": 34, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 16392, "local_bytes": 0, "num_regs": 33, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}, "single": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 16392, "local_bytes": 0, "num_regs": 32, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 16392, "local_bytes": 0, "num_regs": 38, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}}`.

M512_N16_w8: SECONDARY; all gates pass; resources `{"canonical": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 16392, "local_bytes": 0, "num_regs": 24, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 16392, "local_bytes": 0, "num_regs": 31, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}, "repeated": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 16392, "local_bytes": 0, "num_regs": 23, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 16392, "local_bytes": 0, "num_regs": 32, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}, "single": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 16392, "local_bytes": 0, "num_regs": 24, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 16392, "local_bytes": 0, "num_regs": 30, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}}`.

M512_N32_w4: SECONDARY; all gates pass; resources `{"canonical": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 32776, "local_bytes": 0, "num_regs": 45, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 32776, "local_bytes": 0, "num_regs": 42, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}, "repeated": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 32776, "local_bytes": 0, "num_regs": 46, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 32776, "local_bytes": 0, "num_regs": 42, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}, "single": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 32776, "local_bytes": 0, "num_regs": 46, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 32776, "local_bytes": 0, "num_regs": 41, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}}`.

M512_N32_w8: SECONDARY; all gates pass; resources `{"canonical": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 32776, "local_bytes": 0, "num_regs": 32, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 32776, "local_bytes": 0, "num_regs": 38, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}, "repeated": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 32776, "local_bytes": 0, "num_regs": 34, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 32776, "local_bytes": 0, "num_regs": 33, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}, "single": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 32776, "local_bytes": 0, "num_regs": 32, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 32776, "local_bytes": 0, "num_regs": 38, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}}`.

M512_N64_w4: EXCLUDE_FROM_TIMING; RESIDENCY_MISMATCH, SPILL; resources `{"canonical": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 65544, "local_bytes": 0, "num_regs": 35, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 8, "dynamic_smem_bytes": 65544, "local_bytes": 0, "num_regs": 72, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 8, "static_smem_bytes": 0}}, "repeated": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 65544, "local_bytes": 0, "num_regs": 33, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 65544, "local_bytes": 0, "num_regs": 170, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}, "single": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 65544, "local_bytes": 0, "num_regs": 31, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 65544, "local_bytes": 0, "num_regs": 73, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}}`.

M512_N64_w8: PRIMARY; all gates pass; resources `{"canonical": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 65544, "local_bytes": 0, "num_regs": 35, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 65544, "local_bytes": 0, "num_regs": 42, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}, "repeated": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 65544, "local_bytes": 0, "num_regs": 32, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 65544, "local_bytes": 0, "num_regs": 80, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}, "single": {"4": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 65544, "local_bytes": 0, "num_regs": 32, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}, "default": {"cuobjdump_reported_shared_bytes": 1024, "driver_local_bytes": 0, "dynamic_smem_bytes": 65544, "local_bytes": 0, "num_regs": 42, "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed", "stack_bytes": 0, "static_smem_bytes": 0}}}`.

Complete correctness/launch-safety ledger:

```json
[
  {
    "candidate": "default",
    "case": "M16_N128_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M16_N128_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M16_N128_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M16_N128_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M16_N128_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M16_N128_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M16_N128_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M16_N128_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M16_N128_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M16_N128_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M16_N128_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M16_N128_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M16_N64_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M16_N64_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M16_N64_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M16_N64_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M16_N64_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M16_N64_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N128_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N128_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N128_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N128_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N128_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N128_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N128_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N128_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N128_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N128_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N128_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N128_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N16_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N16_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N16_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N16_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N16_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N16_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N16_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N16_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N16_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N16_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N16_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N16_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N32_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N32_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N32_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N32_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N32_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N32_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N32_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N32_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N32_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N32_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N32_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N32_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N64_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N64_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N64_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N64_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N64_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N64_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N64_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N64_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N64_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N64_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M256_N64_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M256_N64_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N128_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N128_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N128_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N128_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N128_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N128_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N128_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N128_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N128_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N128_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N128_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N128_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N16_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N16_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N16_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N16_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N16_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N16_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N16_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N16_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N16_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N16_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N16_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N16_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N32_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N32_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N32_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N32_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N32_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N32_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N32_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N32_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N32_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N32_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N32_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N32_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N64_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N64_w4",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N64_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N64_w4",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N64_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N64_w4",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N64_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N64_w8",
    "harness": "canonical",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N64_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N64_w8",
    "harness": "single",
    "max_abs_diff": 0.0,
    "passed": true,
    "purpose": "reduction correctness"
  },
  {
    "candidate": "default",
    "case": "M512_N64_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  },
  {
    "candidate": "4",
    "case": "M512_N64_w8",
    "harness": "repeated",
    "max_abs_diff": null,
    "passed": true,
    "purpose": "launch safety / structural execution; not reduction correctness"
  }
]
```

## Future coverage feasibility only

```json
{
  "eligible_by_warps": {
    "4": 5,
    "8": 8
  },
  "future_nonconstant_variables_unknown": true,
  "hypotheses": {
    "H5_01_EXACT_BODY_DIRECTIONAL_TRACKING": {
      "coverage": {
        "PRIMARY_n": 2,
        "required": 5
      },
      "coverage_feasibility": "INCONCLUSIVE_BY_COVERAGE_BEFORE_TIMING",
      "status": "PREREGISTERED_HELD_OUT_NOT_YET_TESTED"
    },
    "H5_02_FIXED_DIFFERENTIAL_ALONE_INSUFFICIENT": {
      "coverage": {
        "eligible_n": 13,
        "required": 5
      },
      "coverage_feasibility": "TESTABLE",
      "status": "PREREGISTERED_HELD_OUT_NOT_YET_TESTED"
    },
    "H5_03_WARP_REGIME_CONTEXT_PREDICTION": {
      "coverage": {
        "eligible_by_warps": {
          "4": 5,
          "8": 8
        },
        "required_each": 3
      },
      "coverage_feasibility": "TESTABLE",
      "status": "PREREGISTERED_HELD_OUT_NOT_YET_TESTED"
    }
  },
  "hypothesis_outcome_evaluation": false,
  "model_predictions_or_errors_computed": false,
  "phase": "PHASE_5_STAGE_B_COVERAGE_ONLY",
  "timing_samples": 0
}
```

G/D variance is unknown until separately authorized timing. TESTABLE means coverage only, never support/falsification. No H5 outcome evaluation or model prediction/error calculation.

Full 19-case master schedule and stable eligibility filter are preview only: three invocations, ten rounds, ten samples/visit, three warmups, unchanged B_DESC/B_RUN/R. Neither schedule was executed.

Protected evidence is checked against the entire starting Git tree: Phase 3, Phase 4 A-D and Phase 5 preregistration unchanged. Scientific status and all hypothesis definitions are unchanged.

The frozen preregistration validator inventories its entire phase5 results parent. New raw gate exports therefore live in the sibling phase4/results/phase5_artifact_gate/, preserving that validator and every frozen byte.

Validator evidence: validator_report.json and validator_suite.json. Runtime Git commit/local/remote/clean checks are reported after normal push.

NO Phase 5 timing. NO hypothesis outcome evaluation. NO production heuristic or PR #11991 change. STOP before timing.
