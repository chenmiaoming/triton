# Phase 6 completion and final evidence closure

Stages A–D completed with separate commits. All prior protocols, samples, compiler exports, classifications, models and Phase 5 hypothesis statuses remain byte-identical.

Stage A rederived 31 historical cases and 39 frozen prediction errors without GPU or refitting. Stage B froze five retrospective supplement cases, three harnesses, all five R values and the unseen warp16 domain before new timing. Stage C collected 63,000 archived-binary samples; its R curves reveal context dependence and departures from R0/R1 extrapolation. These old cases are diagnostic supplements, not a new confirmatory cohort.

Stage D accounted for all 32 frozen source transitions: 14 structural exclusions uncompiled, 18 cases / 108 binary attempts, one resource exclusion, nine PRIMARY and eight SECONDARY admitted before timing. All 17 eligible cases retained across canonical, single and repeated harnesses: 214,200 samples, three independently dispatched single-use process invocations. Two actual physical GPU UUIDs were observed; invocations 1 and 2 shared one GPU. No dispatch was repeated to obtain a different outcome or hardware UUID.

| Scope | n | H6_01 single S vs D1 | H6_02 beta vs D1 |
| --- | --- | --- | --- |
| ALL_ELIGIBLE | 17 | SUPPORTED | FALSIFIED |
| PRIMARY | 9 | SUPPORTED | FALSIFIED |

| Scope | n | Fixed predictor | MAE | RMSE |
| --- | --- | --- | --- | --- |
| ALL_ELIGIBLE | 17 | D1 | 1.0690718375522252 | 1.3811741544243141 |
| ALL_ELIGIBLE | 17 | S | 0.8598208856934889 | 1.1758345091825657 |
| ALL_ELIGIBLE | 17 | beta | 1.3780854521055186 | 1.802136374016334 |
| ALL_ELIGIBLE | 17 | zero | 1.5581438697345298 | 1.9628296123113596 |
| PRIMARY | 9 | D1 | 0.9072058055704312 | 1.2057186356224094 |
| PRIMARY | 9 | S | 0.8142465012785998 | 1.0983237525410143 |
| PRIMARY | 9 | beta | 1.524458282608701 | 1.9715803905820104 |
| PRIMARY | 9 | zero | 1.5830699775357044 | 2.022151431997739 |

Error units: ns/additional CTA. Predictions use coefficient 1 and intercept 0; no cross-case refit or calibration. S improves over D1 within the frozen tested warp16 domain and retains substantial residual error. The five-point repetition trend beta worsens both frozen metrics. These comparisons do not isolate lane/warp communication, register allocation or scheduling. H2b/H2c remain UNVERIFIED; no production heuristic follows from this result.

Persistent ccache migration retained 1,043 original files. The completed native-build transaction recorded 136 cache hits and 251 misses (35.1421%); full before/after counters and actual native/compiler SHA identities are retained. The ZIP export repair reused the exact completed native core image. Core sources exclude experiments, so experiment Python iterations reuse that image. Timing C and D load archived CUBINs with no Triton import or compilation. Quota failure, stopped builds, cache-path correction and source-ZIP failure logs/partial returns are retained. MODAL_PROFILE was selected per command; the globally active profile was not changed.

Final validator: PASS; seven Phase 6 validators, eleven unchanged historical validators replayed at their trusted stage baselines, two isolated tamper probes, and first-commit byte checks for every new raw/artifact manifest. Current protected historical files: 3563.

[Supplement analysis](../stage_c/summary.md) · [Fresh admission](../stage_d_gate/summary.md) · [Fresh analysis](../stage_d/summary.md) · [Complete validator evidence](suite.json)

Work stops at Phase 6 Stage D. No next phase has been started.
