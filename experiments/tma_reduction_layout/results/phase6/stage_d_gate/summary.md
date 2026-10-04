# Phase 6 Stage D — Fresh artifact admission

32 frozen warp16 transitions; 18 structurally included; 14 excluded without compilation. 108 complete original binary attempts retained. No performance observation.

| Case | Class | Reasons |
| --- | --- | --- |
| M1024_N128_w16 | EXCLUDE_FROM_TIMING | RESOURCE_UNSUPPORTED |
| M1024_N16_w16 | SECONDARY | ALL_GATES_PASS |
| M1024_N32_w16 | PRIMARY | ALL_GATES_PASS |
| M1024_N64_w16 | PRIMARY | ALL_GATES_PASS |
| M128_N128_w16 | SECONDARY | ALL_GATES_PASS |
| M128_N32_w16 | PRIMARY | ALL_GATES_PASS |
| M128_N64_w16 | PRIMARY | ALL_GATES_PASS |
| M256_N128_w16 | SECONDARY | ALL_GATES_PASS |
| M256_N16_w16 | SECONDARY | ALL_GATES_PASS |
| M256_N32_w16 | PRIMARY | ALL_GATES_PASS |
| M256_N64_w16 | PRIMARY | ALL_GATES_PASS |
| M32_N128_w16 | SECONDARY | ALL_GATES_PASS |
| M512_N128_w16 | SECONDARY | ALL_GATES_PASS |
| M512_N16_w16 | SECONDARY | ALL_GATES_PASS |
| M512_N32_w16 | PRIMARY | ALL_GATES_PASS |
| M512_N64_w16 | PRIMARY | ALL_GATES_PASS |
| M64_N128_w16 | SECONDARY | ALL_GATES_PASS |
| M64_N64_w16 | PRIMARY | ALL_GATES_PASS |

{'EXCLUDE_FROM_TIMING': 1, 'PRIMARY': 9, 'SECONDARY': 8}

Cache build delta: hits=136, misses=251, hit rate=0.35142118863049093. Full before/after counters, compiler/native SHA, core/runtime image IDs and build log are retained.

PRIMARY coverage before timing: TESTABLE. All-eligible coverage before timing: TESTABLE.

Admission matches the frozen full family/load/loop/spill/occupancy gate. Equal opcodes or occupancy do not establish operand/dataflow/whole-program equivalence or isolate registers and scheduling.
