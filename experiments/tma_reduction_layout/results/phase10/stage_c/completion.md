# Phase 10 Stage C completion

Stage C validates the unchanged, actual Stage B compiler patch against clean
upstream `20d9d98fead78c2f65e1b3ba5a7881d86a836dad`, using the same assembler
SHA between baseline and patch on every target. The patch SHA remains
`3641586e23d7b85f76eee00b6a04f8a940ba4904698348f515971e115e29b69e`.

The eight unseen tile/warp identities were frozen before dispatch, with the
declared BF16/max, FP16/min, FP32/sum cells on host/device paths. Six historical
anchor pairs and eight controls per target remain separate. Existing native
images and ccache were reused; no C++ source change or native rebuild occurred.

| Target | Compiler attempts | Exact-output passes | Eligible pairs | Samples |
| --- | --- | --- | --- | --- |
| H100 / SM90 | 124 | 124 | 62 | 55,800 |
| B200 / SM100 | 124 | 124 | 62 | 55,800 |
| RTX PRO 6000 / SM120 | 124 | 104 | 52 | 46,800 |
| Total | 372 | 352 | 176 | 158,400 |

Twenty SM120 attempts (ten pairs) exceed the 101,376-byte shared-memory limit
in both compiler variants. They retain their errors and complete compiler
outputs; no replacement or zero-effect imputation was used. All passing pairs,
including changed residency and local-memory cases, entered timing. Every
guard control preserves identical baseline/patch CUBIN bytes.

Nine single-use timing calls compare both frozen binaries on the same worker,
at full, quarter and sixteenth grids. Alternating variant order and cyclic
case/grid rotation were preregistered. Complete-output/prefix warmup checks,
actual ABI parameters, per-launch SHA guards, driver call counts, module/function
handles and GPU UUIDs pass. No Triton import or subprocess occurs in timing.
Independent medians and the three-process descriptive bands are recomputed.

The full-grid canonical M32/N64/8-warps device time decreases are 41.170% on
H100, 21.438% on B200 and 17.753% on SM120. Held-out full-grid benefits above
the frozen practical threshold number 2 on H100, 13 on B200 and 0 on SM120.
All three grids have zero classified regressions, but 222 case/grid effects
remain unresolved. Of these, 114 bands allow losses beyond 3%, including 50
with changed CUBINs. This does not establish universal non-regression. The
SM120 canonical host effect is unresolved despite a positive mean.

All measurements use the preregistered exact quarter-multiple input pattern;
they do not establish transfer to other input distributions or floating-point
edge cases. Event intervals enclose individually submitted kernel launches;
small-grid results can also be sensitive to submission gaps and event resolution.
The matrix is a declared subset rather than full dtype/op/warp-count coverage.
B200/SM100 does not replace GB300/SM103 evidence. No historical culprit or
bank-conflict mechanism is established by this timing contrast.

The October 6 billing-limit failure and pending-status record are preserved.
After the user's October 13 funding confirmation, both collection applications
completed and their GPU containers stopped. The protocol, worker source and
all 12,278 earlier files stay unchanged. No adverse-outcome reruns, policy
adjustment, PR creation/update or later phase occurred.

The final validator binds protocol/source/native SHA, all raw archives,
compiler/resource/correctness gates, launch ABI/order/grid/SHA and statistics;
four isolated tamper probes must be rejected. After the single Stage C commit,
it checks all original bytes against their first committed form:

```sh
.venv/bin/python -m experiments.tma_reduction_layout.phase10.stage_c.validate_final --committed
```

Work stops after the Stage C commit and push. A further confirmation or PR
packaging decision remains for the user; approval is required before any PR.

[Scope and limitations](scope.md) · [All primary effects](summary.md) ·
[Every case/grid result](results.json) · [Validation](validation.json) ·
[Frozen protocol](preregistration/protocol.json)
