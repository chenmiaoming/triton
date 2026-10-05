# Phase8 StageB — Exact-artifact admission

| Case | Role | Class | Reasons |
| --- | --- | --- | --- |
| M1024_N32_w16 | DIAGNOSTIC | PRIMARY | ALL_GATES_PASS |
| M128_N128_w32 | FRESH | PRIMARY | ALL_GATES_PASS |
| M128_N32_w16 | DIAGNOSTIC | PRIMARY | ALL_GATES_PASS |
| M128_N64_w32 | FRESH | PRIMARY | ALL_GATES_PASS |
| M2048_N32_w16 | DIAGNOSTIC | PRIMARY | ALL_GATES_PASS |
| M256_N128_w32 | FRESH | PRIMARY | ALL_GATES_PASS |
| M256_N32_w32 | FRESH | PRIMARY | ALL_GATES_PASS |
| M256_N64_w32 | FRESH | PRIMARY | ALL_GATES_PASS |
| M512_N128_w32 | FRESH | PRIMARY | ALL_GATES_PASS |
| M512_N32_w32 | FRESH | PRIMARY | ALL_GATES_PASS |
| M512_N64_w16 | DIAGNOSTIC | PRIMARY | ALL_GATES_PASS |
| M512_N64_w32 | FRESH | PRIMARY | ALL_GATES_PASS |
| M64_N128_w32 | FRESH | PRIMARY | ALL_GATES_PASS |

All 104 attempts/partial exports/source/ABIs/correctness/resources retained. No timing. Exact Phase6 core image reused: im-joNh6Ry3lq2oFqOues9VFh.

| Future stage | Eligible | PRIMARY | PRIMARY coverage |
| --- | --- | --- | --- |
| stage_c | 4 | 4 | INCONCLUSIVE_BY_COVERAGE |
| stage_d | 9 | 9 | TESTABLE |

The runtime switch is unspecialized at PTX parameter11. Every switch ELF passed both-mode untimed smoke, with two TMA arms and exactly one common IR LocalLoad/reduction. Future mode0/1 launches must share module/function, resources and CUBIN SHA.

PRIMARY refers only to complete source-bound reduction opcode sequence plus the separate same-binary mode gates. All operand-sensitive body and full-PTX/SASS SHA observations retained; complete machine dataflow/scheduling equivalence between different binaries is not inferred. Cross-cell register/residency changes are compiler responses, not hidden or artificially normalized.
