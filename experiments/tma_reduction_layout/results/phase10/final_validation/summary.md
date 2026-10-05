# Phase 10 completion — Stages A and B

Stage A is committed as `88ed38b3f85c9774ca2bc95d89dcbe750997bfc5`; Stage B
is a separate commit. No PR was created or updated, and no later stage started.

The current-main baseline retains 54 actual compiler attempts and 8,100 valid
exact-binary event samples across H100, B200 and RTX PRO 6000. All 10,892
pre-existing experiment files are unchanged (10,890 baseline-tracked files
plus two already ignored NCU diagnostic JSONs). The first invalid JIT-cache
cohort retains all original files, with its 18 incorrectly labeled vector4
conditions independently rejected by their actual vector8 TTGIR.

Current upstream still shows substantial diagnostic opportunity for the
M32/N64/8-warps identity: host-path time decreases relative to the historical
vector4 compiler are 38.723% on H100, 33.668% on B200 and 5.345% on SM120.
The original M32/N128/4-warps host contrast is only 1.384% on H100, versus
18.669% on B200. All paths and three-process bands remain in Stage A's full
table. H100 also changes assembler version, and all contrasts change native
compiler; they are not causal patch measurements or proof of a historical
regression culprit. Neither B200 nor SM120 replaces GB300/SM103 evidence.

The actual instruction-count compiler prototype is built against the pinned
upstream, with nine added lit cases, 303 full-suite lit passes (2 unsupported)
and 54 exact-value H100 correctness passes. All 54 final correctness compiles
are archived. The stored patch reconstructs the source bytes checked by the
native/lit run and final GPU run. The complete pipeline selects vector1 or
vector2, so previous fixed-vector4 performance evidence does not validate it.
No prototype GPU timing was performed, and multi-architecture prototype
correctness/performance remains a later gate.

The final validator checks archive/export equality, frozen protocol/ABI/order,
per-launch SHA and event counts, independent medians/bands, original upstream
source and patch application, all 54 GPU compile identities/layouts, source/native
SHA binding, failed-run preservation and first-commit raw-byte equality.
Build logs excluded by the generic `build/` ignore rule were explicitly tracked.
Native ccache was reused; the final compiler edit rebuilt one C++ file, and the
formatted Python test reused the existing native image without a rebuild.

Run after both commits:

```sh
.venv/bin/python -m experiments.tma_reduction_layout.phase10.validate_final --committed
```

[Suite](suite.json) · [Baseline](../stage_a/summary.md) · [Prototype](../stage_b/summary.md) ·
[Patch](../../../phase10/compiler_prototype.patch)
