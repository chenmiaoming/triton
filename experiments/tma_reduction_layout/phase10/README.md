# Phase 10 — Current upstream and compiler prototype

The scope is Stage A (current upstream residual audit) and Stage B (an actual
compiler prototype with structural and correctness checks). Stage C held-out
prototype timing and any PR creation/update require a subsequent decision.

The motivation is [PR #11991's requested changes](https://github.com/triton-lang/triton/pull/11991#pullrequestreview-5346968250):
a shape heuristic needs broader justification. The original regression is
[issue #9703](https://github.com/triton-lang/triton/issues/9703). These stages
separate the current baseline question from validation of a proposed rule.

## Stage A

Clean upstream `20d9d98fead78c2f65e1b3ba5a7881d86a836dad` is archived and
compiled at the cached `/opt/triton-src` path with `make`, using the persistent
`triton-build-cache` volume. The historical experimental native extension is
checked against Phase 9's archived SHA. Native build logs, ccache counters,
assembler identities and Modal image/call IDs are retained.

The frozen BF16→FP32 max workload has global shape `[1,8192,8192]` and three
tile/warp identities: `(32,128,4)`, `(32,64,8)`, `(32,128,8)`. Both host and
device descriptor paths compare clean current upstream, historical default and
historical vector4. Strict H100/SM90, B200/SM100 and RTX PRO 6000/SM120 workers
retain all 54 complete compiler artifacts and exact-output correctness checks.

Each target has three separate single-use timing calls, each comparing all 18
frozen CUBINs. Ten rotating rounds contain five samples per visit, after three
warmups per binary: 8,100 valid samples total. The timing process imports no
Triton, launches the archived binary with checked driver ABI arguments, and
checks its SHA before every launch. Invocation medians and descriptive
three-process bands are independently recomputed. Modal namespaces all report
`modal:2` as hostname/PID; distinct call IDs identify dispatches, and physical
GPU UUIDs are recorded separately.

The historical vector4 contrast is a diagnostic reference, **not validation of
a patch to current upstream**. H100 also changes ptxas from 12.9.86 to 13.4.59;
B200 and SM120 use the same assembler SHA between compiler variants. The full
analysis retains default/vector4 effects within the historical compiler.
Neither B200 nor SM120 substitutes for the reported GB300/SM103 system.

An initial run incorrectly reused an in-memory JIT object across experimental
environment changes. Its nominal default/vector4 binaries both had vector8.
All 54 artifacts and 8,100 samples remain unchanged under
`results/phase10/invalid_invocations/jit_cache_alias`, with an invalidation
record and a byte manifest. They are excluded from the accepted analysis.
A fresh JIT object, actual post-compile event count and observed TTGIR layout
assertion repaired the collector; the entire cohort was then rerun. This was
a validity correction, not a rerun selected by timing outcomes.

Local logs also retain the earlier source-upload mismatch, interrupted upload,
and dispatch filename collision. Immutable source snapshots and
`include_source=False` resolved the upload problems. No prior experiment file
was changed: `stage_a/prior_manifest.json` protects 10,892 files.

Read-only validation:

```sh
.venv/bin/python -m experiments.tma_reduction_layout.phase10.audit_stage_a --validate
```

See [accepted Stage A results](../results/phase10/stage_a/summary.md),
[protocol](../results/phase10/stage_a/protocol.json), and
[full analysis](../results/phase10/stage_a/analysis.json).

## Stage B

[compiler_prototype.patch](compiler_prototype.patch) contains the actual native
change and tests against the pinned clean upstream. It was built and checked
in a separate checkout of that upstream. The experiment branch stores this
applyable patch and checked source exports because its historical native API
differs from current upstream. A future PR should use a clean upstream branch.

The rule belongs to `OptimizeThreadLocality`, before TMA lowering, and compares
smaller power-of-two contiguous vectors against the existing layout. Its
instruction-count proxy includes descriptor shared-load vectorization,
optional widening, in-thread combines, lane shuffles, inter-warp conversions
and conversion to the existing output ownership. It uses current linear
layouts, `ReduceOpHelper::getInterWarpReductionLayout` and the lowering's
shuffle eligibility. Candidates must not increase modeled rendezvous or
unique scalar input values per lane; strict cost improvement is required.
No coefficient was fitted to the new timing outcomes. This is a prototype
instruction proxy, not a calibrated GPU latency model or actual register count.

The scope is one CTA, CUDA with 32 lanes/warp, a single-use descriptor load,
an optional single-use widening, and one-input non-innermost FP32 reduction
with a supported single scalar combine. It requires a contiguous-last-dimension
blocked layout. Multi-use inputs, complex combiners and unsupported layouts
keep the existing policy. Cross-operation reasoning is in the transformation;
the lowerings are unchanged. The current Gluon pipeline does not run this pass.

Nine added lit cases cover two positive choices and seven fallback boundaries,
including an eligible large-input case where load cost retains the default.
The positive choices are vector1 for `(32,64,8)` and vector2 for `(32,128,4)`.
The complete H100 pipeline also selects vector2 for `(32,128,8)` in these tests.
This differs from the historical fixed-vector4 candidate.

Validation ran `make`, `make triton-opt`, two targeted lit files and the full
lit suite: 303 passed, two unsupported, no failures. The existing CUDA descriptor
test file adds 54 correctness cases: host/device × three tile/warp identities ×
BF16/FP16/FP32 × max/min/sum. Exact finite quarter-multiple inputs give an
independent PyTorch oracle with zero tolerance. All passed on strict H100.
The final formatted Python test reused the same native image without `make`.
A fresh disk cache retains all 54 complete compiler outputs and verifies the
observed layouts in the full pipeline. No prototype GPU timing was performed.

The initial design snapshot is retained before Stage A timing outcomes. Failed
attempts preserve the outdated helper API compile error, missing LLVM tools
without the cache mount, and incorrect predicted lit vector expectations.
The cost equations were retained while fixing API names and test expectations.
A final contiguous-order guard adds a scope boundary, and Python formatting is
checked with the repository's pinned YAPF and Ruff versions. The final Python
check's first import failure is also retained; explicitly mounting its module
fixed it. Every successful and failed original export remains unchanged.

The clean upstream build reused 68 direct ccache hits with 305 misses for the
updated upstream. The final prototype native build compiled one C++ translation
unit (one cache miss) and linked existing objects. Artifact/timing workers did
not rebuild the native compiler. The persistent cache and exact Modal image
IDs are recorded, and the active local profile was not changed.

Read-only final validation, including first-commit raw-byte checks:

```sh
.venv/bin/python -m experiments.tma_reduction_layout.phase10.validate_final --committed
```

See [prototype results](../results/phase10/stage_b/summary.md) and
[final closure](../results/phase10/final_validation/summary.md). The next gate
is separately frozen performance validation of this actual patch against the
same upstream/toolchain, across held-out shapes and architectures. Instruction
counts do not cover bank conflicts, latency, scheduling, cache/bandwidth or
actual register allocation/residency. Prototype correctness has only been
checked on H100. No PR was created/updated, and work stops after Stage B.
