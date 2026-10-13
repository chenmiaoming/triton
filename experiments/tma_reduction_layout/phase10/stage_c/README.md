# Stage C — Frozen actual-patch validation

User continuation authorizes this stage. The native patch is exactly the
accepted Stage B patch on pinned upstream `20d9d98fead78c2f65e1b3ba5a7881d86a836dad`.
Baseline and patch use the same upstream and architecture-selected assembler.
Existing native images are reused with exact source/native SHA assertions; no
native build or `make` is needed for these Python-only collectors.

The frozen matrix contains 48 held-out pairs (8 unseen tile/warp identities ×
BF16/max, FP16/min, FP32/sum × host/device), 6 historical anchor pairs and
8 control pairs per target. It is an explicitly declared dtype/op subset,
not full-factorial coverage. Historical membership uses only structural
identities in baseline-tracked paths/JSON, including the early complete sweep;
no historical numeric performance field determines selection. The targets
remain strict H100/SM90, B200/SM100 and RTX PRO 6000/SM120.

All 372 compiler attempts, errors and full successful source/IR/PTX/CUBIN/SASS/
resource/occupancy outputs are retained. Fresh JIT objects and actual compile
hooks prevent the prior in-memory cache error. Both variants must pass complete
exact finite-value correctness, archived-binary occupancy and ABI checks.
All passing pairs enter timing regardless of observed resources/layouts.
Unavailable pairs stay unavailable, with no replacement or zero imputation.

Three timing calls per architecture use archived binaries without importing
Triton or spawning subprocesses. Ten rounds use five event samples per visit,
with alternating baseline/patch order and cyclic case/grid rotation. Each
binary is measured at its full grid and 1/4, 1/16 prefixes, after three checked
warmups per condition. ABI arguments, grid, module/function handles, per-launch
SHA, output extent and worker UUIDs are recorded. Every operation is a valid
logical block program; no source-level hardware IDs or inline assembly occur.

The primary effect is whole-kernel time decrease at the full grid. Three
invocation medians give the frozen ±3% practical decision and descriptive df2
band. Every secondary effect, unavailable case, resource stratum and guard
control remains reported. Identical-CUBIN variation cannot support a causal
compiler claim. No universal safety or GB300/SM103 coverage is inferred.

The old Stage A/B validator is replayed in an isolated checkout of `67bccffd`
because its historical scope stops before Stage C. Its source, byte manifests
and reports stay unchanged. Stage C independently protects all 12,278 earlier
files and validates its original-byte manifest after commit.

Run locally after GPU artifacts/timing are collected:

```sh
.venv/bin/python -m experiments.tma_reduction_layout.phase10.stage_c.audit_artifacts
.venv/bin/python -m experiments.tma_reduction_layout.phase10.stage_c.analyze
.venv/bin/python -m experiments.tma_reduction_layout.phase10.stage_c.validate_final --freeze
.venv/bin/python -m experiments.tma_reduction_layout.phase10.stage_c.validate_final --committed
```

GPU dispatch uses an immutable `/tmp/tma-phase10-stage-c-snapshot` source bundle,
`include_source=False`, and an explicitly selected authorized Modal profile.
If billing blocks dispatch, preserve the failure and ask the user for a funded
profile. Never silently switch profiles. No timing outcome can change the
frozen rule or cohort. Complete one Stage C commit and push, then stop; no Stage D,
PR creation/update or message to reviewers is authorized here.

Completed results are in
[completion.md](../../results/phase10/stage_c/completion.md) and
[scope.md](../../results/phase10/stage_c/scope.md). The source/native rule stays
frozen even where the measurements are unresolved. Structured exact inputs
verify the finite-value oracle; other distributions, floating edge cases and
GB300/SM103 are outside this evidence. Small-grid driver-event intervals can
also be sensitive to host submission gaps. No universal safety claim is made.
