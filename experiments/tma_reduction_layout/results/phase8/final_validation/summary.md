# Phase8 completion and final evidence closure

All four stages completed with separate stage commits. No compiler/production changes or later phase. Every one of the8,174 pre-Phase8 experiment files remains byte-identical.

StageA retained the Phase7 influence and leave-one-out audit, and froze four old diagnostic cases plus16 unseen identities. Seven prospective identities failed the unchanged structural rules before compilation; none were replaced. StageB retained all104 actual compiler attempts. Four diagnostic andnine unseen cases passed PRIMARY gates, zero spills and candidate residency matching.

The Gluon switch uses a block-uniform, unspecialized mode scalar. Both modes share one archived CUBIN/module/function and resource allocation, with two TMA arms and one common LocalLoad/reduction/store. Both-mode smoke and all launch SHA guards passed. Full operand/PTX/SASS inventories are retained. This establishes a valid same-binary runtime descriptor-path intervention, not isolated intrinsic tensormap creation latency.

| Stage | Eligible | PRIMARY | Samples | Independent processes | GPU UUIDs |
| --- | --- | --- | --- | --- | --- |
| stage_c | 4 | 4 | 36000 | 3 | 3 |
| stage_d | 9 | 9 | 81000 | 3 | 3 |

| Scope | Hypothesis | n | Reference MAE / RMSE | Alternative MAE / RMSE | Decision | Practical threshold |
| --- | --- | --- | --- | --- | --- | --- |
| ALL_ELIGIBLE | H8_01_SAME_BINARY_DESCRIPTOR_TRANSFER | 9 | {'MAE': 2.207124351924803, 'RMSE': 2.3106573831611734} | {'MAE': 0.38505001934479366, 'RMSE': 0.6285074003776717} | SUPPORTED | True |
| ALL_ELIGIBLE | H8_02_DESCRIPTOR_INCREMENT | 9 | {'MAE': 2.1741953840502046, 'RMSE': 2.2934753880962946} | {'MAE': 0.4755396605500089, 'RMSE': 0.6791363047722652} | SUPPORTED | True |
| PRIMARY | H8_01_SAME_BINARY_DESCRIPTOR_TRANSFER | 9 | {'MAE': 2.207124351924803, 'RMSE': 2.3106573831611734} | {'MAE': 0.38505001934479366, 'RMSE': 0.6285074003776717} | SUPPORTED | True |
| PRIMARY | H8_02_DESCRIPTOR_INCREMENT | 9 | {'MAE': 2.1741953840502046, 'RMSE': 2.2934753880962946} | {'MAE': 0.4755396605500089, 'RMSE': 0.6791363047722652} | SUPPORTED | True |

Fresh decisions use the fixed coefficient1/intercept0 predictors, strict MAE/RMSE rule, and separate preregistered practical threshold (MAE decrease≥0.01 ns/CTA and≥10%). No model fit, case removal, threshold tuning or adverse-result rerun. All individual errors, counterexamples, sign bands, leave-one-out comparisons and fit residuals remain in the detailed results.

H8_01_SAME_BINARY_DESCRIPTOR_TRANSFER: 9/9 individual absolute errors improve; every leave-one-out improves both MAE/RMSE=True; relative MAE decrease=82.554222%.
H8_02_DESCRIPTOR_INCREMENT: 9/9 individual absolute errors improve; every leave-one-out improves both MAE/RMSE=True; relative MAE decrease=78.128016%.

NCU: 8/8 original NVR reports and 8/8 complete requested metric sets. Original and explicit-base-unit offline views agree. Initial NCU2026 import auto-scaled DRAM values; precise base-unit derived views are retained. Counter aggregates are descriptive and profiler duration is excluded from formal timing.

Metadata deviation: the original profiling worker export omitted its UUID (recorded as unavailable, with H100 name/CC and dispatch identity retained). Formal timing worker UUIDs are complete. No GPU profile was repeated to repair that metadata omission.

The exact Phase6 native image and persistent ccache/triton-home were reused with unchanged native/compiler SHA identities. Formal timing workers only load archived ELF bytes and never import Triton/JIT/compiler.

Final validator PASS: seven Phase8 checks, trusted Phase7 closure including its Phase6/historical replay, prior bank-probe replay, first-commit B/C original-byte checks, frozen D original-byte closure and four isolated corruption probes. Prior replays run in a separate checkout at 7907562cae9d6fc76184ef17d07fb9e28c364ecf without altering current prior results.

The prospective scope has nine PRIMARY cases extending warp count to32; it does not establish arbitrary shape/warp generalization. Runtime mode includes its branch and descriptor issue path. H2b/H2c remain UNVERIFIED. No pure component-cost, causal-share, or compiler heuristic claim is made.

[Preregistration](../stage_a/summary.md) · [Artifacts](../stage_b/summary.md) · [Diagnostic timing](../stage_c/summary.md) · [NCU](../stage_c/profiling/summary.md) · [Held-out timing](../stage_d/summary.md) · [Validator](suite.json)

Work stops after Phase8 StageD. No Phase9 is started.
