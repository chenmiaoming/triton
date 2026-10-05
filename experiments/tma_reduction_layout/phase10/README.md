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
