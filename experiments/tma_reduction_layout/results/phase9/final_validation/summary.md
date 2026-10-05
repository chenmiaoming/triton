# Phase9 completion and final evidence closure

Stages A-D completed with separate stage commits. No compiler/production changes, PR creation or later phase. All9,489 pre-Phase9 experiment files remain byte-identical.

The strict targets H100/SM90, B200/SM100 and RTX PRO6000 Blackwell Server Edition/SM120 were all allocated. Twelve retrospective reduction identities and six non-reduction TMA load/store controls were frozen before dispatch. All108 compiler attempts and original source/IR/PTX/CUBIN/SASS/resource/occupancy/smoke/failure artifacts remain retained. The exact Phase6 native core and unchanged native extension SHA were reused; ccache counters did not advance. Read-only CUDA13 disassembly tools were layered without a native rebuild. Actual architecture-selected compiler ptxas identities are recorded separately.

| Target | Eligible pairs | Samples | Processes | Physical UUIDs | Original case decision | Mean improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| sm90 | 18 | 32400 | 3 | 3 | BENEFIT | 34.725083 |
| sm100 | 18 | 32400 | 3 | 3 | BENEFIT | 16.453098 |
| sm120 | 16 | 28800 | 3 | 2 | BENEFIT | 15.424337 |

Total93600 exact-binary event samples. Actual ABI parameters, module/function identities, per-launch SHA guards, checked CUDA call totals and worker UUIDs pass. Two SM120 large tiles exceed shared-memory limits and remain unavailable, with no replacement. All valid pairs and resource strata are retained.

The fixed3% practical bands classify0 regression pairs and0 unresolved pairs. A global vector4 policy is not justified. Full timings at all three grids,100-sample medians/statistics, OLS intercepts/slopes/R²/residuals and individual invocation effects remain in results.json. Bands are descriptive three-process summaries, not familywise significance or proof of universal safety.

Secondary grids retain9 unresolved practical effects and0 classified regressions. Some small-grid bands include decreases beyond3%; primary-grid non-regression cannot be transferred to those conditions. No GPU timing was repeated after these outcomes.

Final validator PASS: six checks including read-only Phase8 committed-byte closure, independent order-statistic medians and rational OLS, four isolated tamper probes, same-source checks across targets and first-commit B/C original-byte verification. No raw samples or original compiler artifacts were modified.

This phase establishes cross-architecture candidate contrasts for historically selected BF16/max shapes. It does not identify a historical good/bad compiler commit, isolate intrinsic descriptor/bank/lane cost, or validate a production patch on held-out cases. The next fix requires explicit target/IR applicability and a separate frozen-patch validation. User approval is mandatory before creating any PR, including draft. Work stops after Phase9 StageD.

[Scope decision](../stage_d/summary.md) · [Full suite](suite.json) · [Preregistration](../stage_a/summary.md) · [Artifact gate](../stage_b/summary.md)
