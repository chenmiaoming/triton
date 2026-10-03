# Phase 5 — Held-Out Validation

This is preregistration only. No Phase 5 artifact gate, compiler invocation,
GPU workload, Modal command or timing is authorized or performed in Stage D.
Stage C is the development/hypothesis-generation set. Future Phase 5 data are
a held-out confirmatory extension for the three OUTCOME_INFORMED_HYPOTHESIS
records, with equations, scope, support/falsification comparisons and prediction
coefficients locked before collecting any held-out timings. Stage C cannot
confirm hypotheses generated from its own results.

## Fixed source domain and structural projection

Generate every transition in M={16,256,512}, N={16,32,64,128}, num_warps={4,8}:
24 distinct default→cand4 transitions. M16 is one power of two below the
development range M32/64/128; M256/512 are above it. This is deliberate
extrapolation, with no pruning for speedup, expected equivalence, N64 or warp8.

The membership function accepts only M, N and num_warps. It analytically replays
the frozen Coalesce.cpp descriptor-vector rule and BlockedEncoding source
construction, then applies the unmodified Stage A selector: legal default and
cand4, distinct supported layouts, unchanged warpPart[M], strictly decreasing
lanePart[M], M reduction axis, unchanged warp count. All 24 records, all
nonexclusive machine-readable exclusion reasons and included coverage are
retained. The `compiled_num_warps` selector field is an expected source
attribute, explicitly not compiled metadata. Layouts are predictions, not
actual TTGIR. The validator independently computes layouts and exclusions,
checks predictions against all frozen Stage A source layouts and rejects any
nonstructural access with guarded and performance-poisoned inputs.

## Future artifact gate, resources and hierarchy

No future class is assigned from source predictions. A separately authorized
artifact gate must check actual archived canonical/single/repeated TTGIR/PTX,
source-bound complete reduction fingerprints, LocalLoad semantics, no reload,
unspecialized R, zero spill, exact CUBIN and matched-candidate residency. Reuse
the frozen Stage A/B isolation and resource gates without relaxation.

PRIMARY requires EXACT_SEQUENCE_EQUIVALENT for both candidates in both
canonical-to-single and repeated-runtime-body reproductions, with all gates
passed. SECONDARY requires all gates and at least one
PIPELINED_OPCODE_EQUIVALENT reproduction. Other cases are excluded from timing
or pending, without changing structural membership. Outcome-dependent class
upgrades are forbidden. PRIMARY n>=5 is necessary to consider generalization
within the observed new exact-equivalence scope, never sufficient on its own.
Report M/N/warp/lane coverage, sign agreement and all counterexamples; report
SECONDARY separately. No development cases rescue a small held-out PRIMARY.

All included source predictions remain STRUCTURAL_OR_RESOURCE_PENDING.
M512/N128 at frozen B_DESC needs 8GiB BF16 input and a 128KiB logical shared
tile; these are logical allocation estimates, not compiled dynamic/static
SMEM, spill or occupancy. Unknown/illegal future resource constraints remain
pending or fail the gate. Do not reduce descriptors/grids or otherwise adapt
the frozen protocol to make a case fit.

## Future timing and analysis, not execution

Retain B_DESC=65536, B_RUN={16384,32768,65536}, R={0,1}, three independent
single-use invocations, ten rounds with ten scalar CUDA-event samples per
condition per round and three untimed warmups per condition. Use one exact
archived kernel per scalar sample with a SHA guard before every launch.
Preserve whole-invocation invalidation/repair and archived partial failure
records; unfavorable outcomes are never grounds for replacement.

Per invocation use the median of 100 scalar samples per condition and the
three-B OLS slope, retaining R² and residuals. Derive paired G, g0, g1, D=g1−g0
and G−D with invocation values, means and sample SDs. G−D is a cross-harness
descriptive residual, not an additive causal remainder. g0 is the fixed-harness
differential, not pure load cost. D+g0 is a descriptive constructed score.
Retain the three-invocation t(df=2) sign band and the complete nine-grid taxonomy;
SIGN_UNRESOLVED does not imply practical equivalence. Spearman uses continuous
values and exact-tie average ranks, PRIMARY n<5 remains insufficient for
established generalization, and correlations/LOO with n<3 or constants are
UNDEFINED. SECONDARY Pearson and counterexamples stay separate.

Only A: G~1+D, B: G~1+D+g0 and C: G~1+D+g0+I(warps==8) are registered here.
For prospective held-out hypothesis prediction, use the locked Stage D
ALL_ELIGIBLE coefficients in protocol.json; do not refit after observing
Phase 5 outcomes. Evaluate every future timing-eligible case and retain
separate strata. H5_01 tests a positive rank direction in new PRIMARY n>=5.
H5_02 compares A/B held-out MAE and RMSE on n>=5 eligible cases.
H5_03 compares B/C errors with at least three eligible cases in each warp
regime. Mixed error orderings, insufficient coverage, invalid protocol or
undefined predictions are INCONCLUSIVE, as specified in each hypothesis.
These are fixed directional predictions, with no tuned cutoff, p-value,
significance claim, automatic generalization or causal inference.

## Immutable evidence and paths

Every prior experiment byte is bound to Git baseline
11310ee3b6c92e188c66999a5dacd1ece87d7079. Stage C remains PRIMARY=3,
SECONDARY=15, EXCLUDE=2 and
INSUFFICIENT_PRIMARY_COHORT_SIZE_FOR_ESTABLISHED_GENERALIZATION.
H2a historical remains SUPPORTED_AT_REDUCTION_BODY_LEVEL; H2b/H2c UNVERIFIED.

The frozen Stage A validator admits new files only under phase4/ or its
existing preregistration directory and hash-binds itself. Therefore the
distinct Phase 5 code lives at phase4/phase5/ and its new outputs at
phase4/results/phase5/preregistration/, preserving all frozen validator/source
bindings. This directory placement does not turn the new hypotheses into
Phase 4 preregistered evidence. After all eight validators pass, commit and
normal-push Stage D, verify remote SHA and STOP. A separate authorization is
required for the future artifact gate.
