# G8 reproduction and evidence order

This stage replaces the production Gluon comparisons identified in
[IMPACT_AUDIT.md](IMPACT_AUDIT.md). It leaves all earlier measurements and compiler
artifacts intact. Run from the repository root on `explore/tma-reduction-layout`.
The commands below record the actual workflow; they are write-once and must not
be rerun into an existing evidence destination.

## Frozen inputs

- `preregistration/protocol.json` contains all 517 logical conditions, source
  hashes, historical-artifact protection hashes and the measurement policy.
- `preregistration/source_snapshot.zip` contains the worker source used by the
  Modal images; `reference_payload.zip` contains the hash-verified reusable U,
  SGLang stage2 and LayerNorm dX artifacts.
- The compiler is pinned to `20d9d98fead78c2f65e1b3ba5a7881d86a836dad` and the
  native library to
  `fc5f2ad73f9f8c8ab31c3a01b8124329ccec391cdc5dbf2dafb2d42778cf6adb`.
  The standard compiler pipeline is used. No compiler/native rebuild is needed.
- The Modal compiler image is `im-qzTUZumWu2JXjWic8K82cZ`, the tools image is
  `im-FUYtAVgmAyw1ZiFjfu6cmg`, and the requested GPU is `H100!:1`.
- The pre-existing build cache is reused. Source compilation uses
  `/cache/phase11-upstream20d9-kernel-cache-g6-source-only`; no historical cache
  or artifact is edited.

## Ordered execution

Use the configured `chenmiaoming` profile without printing credential files.
The initial preparation command creates the frozen manifest and source snapshot;
it is not a command for overwriting this committed stage.

```sh
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g8_prepare
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g8_run --action cpu
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g8_build
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g8_run --action check
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g8_run --action timing
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g8_analyze
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g8_profile
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g8_profile_supplement --action prepare
MODAL_PROFILE=chenmiaoming .venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g8_profile_supplement --action collect
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g8_counters
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g8_report
```

CPU compilation must pass the full effective-layout, default-anchor machine-code
and ABI checks before GPU execution. Correctness must pass before formal timing.
The formal timings use driver-loaded frozen CUBINs with no Triton import or
compilation in the GPU process. All three timing invocations must finish before
NCU collection. Profiling uses at most three GPU groups concurrently and is not
included in the timing estimator.

To audit committed evidence, use the offline validator rather than launching any
of the Modal commands:

```sh
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g8_validate
```

The initial final-validation record is created once with `--write`; subsequent
audits omit that flag and compare against the committed record.

The six supplemental reports are post-timing exploratory mechanism diagnostics
for the newly observed `vec4:within` vLLM candidates in two geometries. Their
separate preregistration fixes source, target and replay settings before those
profiles; it does not turn the already observed timing winners into preregistered
performance hypotheses. The original 167-report plan is unchanged. Supplemental
collection starts after all original profiling groups have returned.

Local NCU re-import uses `/usr/local/cuda/bin/ncu`; source correlation uses the
current CUBIN's debug records through `/usr/local/cuda/bin/nvdisasm -gi -c`.

## Supplemental actual-attribute audit

The original G8 matrix deliberately retained its frozen generic attributes.
After reviewing G7's actual JIT metadata, a separate CPU-only audit compiled the
G0 and target for two vLLM batch64 groups with the observed `num_seqs` divisibility
attribute. Its four Gluon comparisons and two U-reference comparisons found equal
execution code, ABI and resources, so no extra GPU timing was required.

The frozen original analyzer wrote `actual_abi_audit/analysis.json` successfully
and then failed in its terminal summary because `json` was not imported. The
trace is preserved in `actual_abi_audit/analysis_cli_failure.txt`; the analyzer
source and written artifacts are unchanged. A separate offline verifier
recomputes the equivalence checks:

```sh
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g8_actualabi_verify
```

Its initial record is written once with `--write`; the command above audits the
committed record. It does not allocate a GPU, call JIT, or repeat compilation.
Do not rerun the original analyzer into its write-once destination.

## Measurements and interpretation

The matrix retains G2 (294), G3b (99), G4a (40) and G4b (84) logical conditions.
G3a contributes NCU questions rather than another timing matrix. Three H100
workers each perform five randomized visits and 31 event samples per physical
condition. Exact CUBIN/ABI aliases share physical observations and remain
explicit in the logical matrix. Counts of physical samples must be read from the
completed timing analysis rather than inferred as 517 times the replay count.

The estimator is the mean of the 15 worker/visit medians. The shared hierarchical
bootstrap resamples workers and visits together for each paired comparison
(4,096 replicates). Intervals are exploratory pointwise 95% intervals, with no
multiplicity correction. The evidence is not a held-out evaluation of a layout
selector, and choosing the fastest observed candidate does not establish a
validated selection policy.

Each candidate is compared with its contemporaneous U and G0 from the same
panel, workload and actual/B4 label. Historical comparisons use each experiment's
own normalized gains; no historical U duration is combined with a new candidate
duration. LayerNorm merge and the full backward pipeline are reported separately:
the latter includes the unchanged dX kernel and three zeroing nodes.

Software instruction counts, warm application-replay hardware counters and cold
isolated kernel-replay hardware counters are retained separately. Counts describe
the profiled graph; neither instruction counts nor wavefront counts alone are
converted into a measured latency contribution.
