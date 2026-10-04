# Phase7 completion and final evidence closure

Stages A–D completed. All 4888 earlier experiment files remain byte-identical. No compiler or production heuristic changes.

StageA audited complete kernel contexts for17 Phase6 cases without GPU. StageB preregistered the four Gluon descriptor/store interventions, canonical reference, nine diagnostic identities and20 entirely unseen identities before compilation/timing. All290 actual binary attempts and deterministic resource failures are retained.

| Stage | Eligible | Samples | Physical GPU UUIDs |
| --- | --- | --- | --- |
| stage_c | 9 | 81000 | 2 |
| stage_d | 6 | 54000 | 3 |

StageC is an outcome-informed diagnostic cohort; StageD uses the fixed independent cohort and all three uncalibrated hypotheses. No refit, case removal, threshold tuning or outcome-based rerun. Three separately dispatched processes per timing stage; physicalUUID counts are measured, not assumed.

| Scope | n | H7_01_OUTPUT_CONTEXT_TRANSFER | H7_02_FULL_CONTEXT_TRANSFER | H7_03_DESCRIPTOR_INCREMENT |
| --- | --- | --- | --- | --- |
| ALL_ELIGIBLE | 6 | SUPPORTED | SUPPORTED | SUPPORTED |
| PRIMARY | 1 | INCONCLUSIVE_BY_COVERAGE | INCONCLUSIVE_BY_COVERAGE | INCONCLUSIVE_BY_COVERAGE |

| Scope | n | Fixed predictor | MAE | RMSE |
| --- | --- | --- | --- | --- |
| ALL_ELIGIBLE | 6 | H00 | 0.4418751929719414 | 1.0317508877963184 |
| ALL_ELIGIBLE | 6 | H01 | 0.4417816305090492 | 1.0315281493396027 |
| ALL_ELIGIBLE | 6 | H10 | 0.016008463775044415 | 0.018982941736232956 |
| ALL_ELIGIBLE | 6 | H11 | 0.014233227563045212 | 0.02171607374047002 |
| ALL_ELIGIBLE | 6 | zero | 0.4345196942715181 | 1.019345498686312 |
| PRIMARY | 1 | H00 | 2.526345616345118 | 2.526345616345118 |
| PRIMARY | 1 | H01 | 2.52590204278628 | 2.52590204278628 |
| PRIMARY | 1 | H10 | 0.019019872914730396 | 0.019019872914730396 |
| PRIMARY | 1 | H11 | 0.04950215086518517 | 0.04950215086518517 |
| PRIMARY | 1 | zero | 2.4960705299495842 | 2.4960705299495842 |

The all-eligible scope has six cases (one PRIMARY, five SECONDARY); the PRIMARY scope is coverage-inconclusive. All three comparisons satisfy the frozen strict MAE/RMSE decision, but H7_01 improves MAE by only about 0.00009356 ns/CTA (0.0212%). Its SUPPORTED label does not establish a practically large output-store effect. Full-context MAE is about 0.01423 versus host/native 0.44188 ns/CTA. These case-level aggregate errors are heavily influenced by M2048_N32_w16; all five other cases, including near-zero/negative gaps, are retained.

Gaps/effects/errors use ns/additional CTA. Store/descriptor interventions include compiler register allocation and scheduling responses. Reduction opcode matching and equal theoretical residency do not establish complete machine dataflow equivalence or pure component cost. Interaction is retained per case and invocation.

The exact completed Phase6 native image was reused with unchanged native/compiler SHA identities and persistent cache. No native rebuild. Formal timing loads archived ELF CUBINs and guards SHA before every launch, with no Triton import/JIT/compiler.

Profiler availability: COUNTERS_COLLECTED. Profiler replay duration is excluded from formal timing.

Finalvalidator PASS: nine Phase7 checks, trusted Phase6 final closure, eleven historical validators replayed, two isolated tamper probes, and first-Git-commit raw/artifact byte checks.

[Artifact admission](../stage_b_artifacts/summary.md) · [Diagnostic results](../stage_c/summary.md) · [Counters](../stage_c/profiling/counter_summary.md) · [Held-out results](../stage_d/summary.md) · [Validator evidence](suite.json)

Work stops after Phase7 StageD. H2b/H2c remain UNVERIFIED. No later phase has been started.
