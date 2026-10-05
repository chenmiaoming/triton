# Phase 10 Stage B — Checked compiler prototype

The actual prototype patch applies to clean upstream
`20d9d98fead78c2f65e1b3ba5a7881d86a836dad`. Its exact native source is compiled,
and the stored patch independently reconstructs the checked sources from the
archived upstream tar. It is stored as a patch on the experiment branch and was
tested in a separate current-upstream checkout.

The rule compares load vectorization and reduction communication using an
uncalibrated instruction-count proxy. It enumerates smaller contiguous vectors,
preserves default ownership at the output, and requires strict improvement
without more modeled synchronization or scalar input values per lane. It has
explicit single-use, single-combine, FP32, CUDA, one-CTA and layout scope.
The transformation preserves the block programming model and descriptor view;
no source-level hardware IDs, inline assembly or hidden divergent scalars occur.

| Check | Result |
| --- | --- |
| Native build | `make`, then `make triton-opt`; exact source/native SHA retained |
| Targeted lit | optimize-locality and coalesce pass; nine new layout cases |
| Full lit | 303 passed, 2 unsupported, 0 failed |
| H100 correctness | 54 passed: host/device, three tiles, BF16/FP16/FP32, max/min/sum |
| Full-pipeline artifacts | 54 fresh-cache source/TTIR/TTGIR/LLIR/PTX/CUBIN/metadata exports retained |
| Final patch binding | independently applied to archived clean source; checked CPU native/lit and final GPU test SHA match |
| Final native rebuild | one C++ file, one ccache miss; existing objects reused |
| Prototype timing / held-out performance | not performed |

The full GPU pipeline chooses vector1 for `(M32,N64,w8)` and vector2 for both
`(M32,N128,w4)` and `(M32,N128,w8)` in all 54 test cases. An eligible
`(M1024,N64,w4)` lit example keeps vector8 because modeled load cost dominates.
These choices are compiler outputs, not choices fitted to timing results.
Historical vector4 timing cannot establish the performance of this rule.

The original design, all failed builds/checks, source snapshots, Modal logs,
worker UUIDs and complete successful exports are retained under this stage.
The finite exact-input oracle verifies reduction values independently of tree
order; it does not establish floating-point edge-case behavior, cross-architecture
performance, actual register usage or universal TMA safety.

The next required gate is held-out timing of this frozen actual patch against
the same upstream and assembler, with applicability/fallback controls and
resource checks. No PR was created or updated. Work stops at Stage B.

[Patch](../../../phase10/compiler_prototype.patch) · [Full checked results](results.json) ·
[Accepted native/lit run](accepted.json) · [Final GPU check](final_python/result.json)
