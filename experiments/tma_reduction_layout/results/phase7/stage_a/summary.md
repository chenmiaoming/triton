# Phase 7 Stage A — Existing complete-kernel audit

All 17 eligible Phase 6 cases / 102 actual binary bundles retained. Complete parsed PTX operands/predicates, SASS operands/instruction words/control-word lines, descriptor/synchronization/shared/store operations, IR SSA-token records and resources are in results.json.

| Case | Class | G−S ns/additional CTA | Canonical output conversions [default,4] | Single conversions | Canonical device descriptor |
| --- | --- | --- | --- | --- | --- |
| M1024_N16_w16 | SECONDARY | -0.0933779076495694 | [1, 1] | [0, 0] | [False, False] |
| M1024_N32_w16 | PRIMARY | -0.02355357407525697 | [1, 1] | [0, 0] | [False, False] |
| M1024_N64_w16 | PRIMARY | 2.0415281185359198 | [1, 1] | [0, 0] | [False, False] |
| M128_N128_w16 | SECONDARY | 0.20000951345926823 | [1, 1] | [0, 0] | [False, False] |
| M128_N32_w16 | PRIMARY | 0.38878780677415675 | [1, 1] | [0, 0] | [False, False] |
| M128_N64_w16 | PRIMARY | 1.8678850838547252 | [1, 1] | [0, 0] | [False, False] |
| M256_N128_w16 | SECONDARY | -0.01971663247483472 | [1, 1] | [0, 0] | [False, False] |
| M256_N16_w16 | SECONDARY | 0.17199105665592182 | [1, 1] | [0, 0] | [False, False] |
| M256_N32_w16 | PRIMARY | 1.4831426445501468 | [1, 1] | [0, 0] | [False, False] |
| M256_N64_w16 | PRIMARY | 0.3698377917143745 | [1, 1] | [0, 0] | [False, False] |
| M32_N128_w16 | SECONDARY | 1.1579707223150073 | [1, 1] | [0, 0] | [False, False] |
| M512_N128_w16 | SECONDARY | 2.3803716300920215 | [1, 1] | [0, 0] | [False, False] |
| M512_N16_w16 | SECONDARY | 1.293317371251067 | [1, 1] | [0, 0] | [False, False] |
| M512_N32_w16 | PRIMARY | 0.3860912416019436 | [1, 1] | [0, 0] | [False, False] |
| M512_N64_w16 | PRIMARY | -0.01743860810252329 | [1, 1] | [0, 0] | [False, False] |
| M64_N128_w16 | SECONDARY | 1.9719817113842242 | [1, 1] | [0, 0] | [False, False] |
| M64_N64_w16 | PRIMARY | 0.749953642298351 | [1, 1] | [0, 0] | [False, False] |

These are static instruction inventories; no liveness/CFG solver or whole-program semantic-equivalence proof is claimed. The descriptor and output-layout differences motivate independently controlled Gluon interventions. The all-nine PRIMARY diagnostic cohort is outcome-informed; it is not a new confirmatory cohort.

- Existing PRIMARY is opcode sequence equivalence of the reduction stage only.
- Operands, full-program control/dataflow, schedules and CUBIN equivalence remain separate questions.
- Static differences and G-S are descriptive; no component latency or causal share inferred.
- All 17 cases retained; all nine PRIMARY form the intervention diagnostic cohort.
- No GPU, timing, compiler execution or new model fitting in Stage A.
