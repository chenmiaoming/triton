# A10: compiler-selected layouts on the ordinary load path

This experiment answers whether the column-wise partial reduction from
[issue #9703](https://github.com/triton-lang/triton/issues/9703) also changes
layout or regresses across Triton 3.5.1 → 3.6.0 when expressed using ordinary
`tl.load`. It also includes the locally installed 3.7.1 used in the preceding
A10 diagnostic. **It does not compare manually selected vector widths.**

Only physical A10 GPU 0 was used, identified by UUID
`GPU-69464998-e901-74ba-63ff-ae7022f8aed9`. No job was launched on GPU 1.
No TMA, native compiler build, Phase 11 C calibration, or PR was started.

## Actual Triton DSL

The complete kernel is in [kernels.py](kernels.py). Its function AST is identical
to the preceding diagnostic's **plain Triton** `local_a10.kernels.canonical`:

```python
@triton.jit
def canonical(x, out, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    pm, pn = pid // (8192 // BN), pid % (8192 // BN)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    offsets = tl.expand_dims(rows[:, None] * 8192 + cols[None, :], 0)
    values = tl.load(x + offsets).to(tl.float32)
    result = tl.max(values, axis=1)
    tl.store(out + pm * 8192 + cols, tl.reshape(result, [BN]))
```

Input is contiguous BF16 `[1,8192,8192]`. Each CTA consumes `[1,32,BN]`,
combines the 32 values within each column, and writes FP32 `[1,BN]`.
The full output is `[256,8192]`. `axis=1` is M because the leading batch axis
is retained. A two-dimensional `[BM,BN]` spelling would reduce `axis=0`, but
this experiment deliberately preserves the previous kernel's rank and source.

The source requests neither a register layout nor initial shared-memory
staging. The compiler chooses layouts for global loads, reduction, and stores.
The resulting SASS has direct global loads; shared-memory operations later
in the kernel implement reduction/output communication. Their presence does
not turn the initial load into a TMA or `cp.async` path.

## What the compilers selected

The load and the **input to the reduction** were checked separately in each
final TTGIR. Omitting the size-one batch dimension:

| Tile / warps | Versions | sizePerThread `[M,N]` | threadsPerWarp `[M,N]` | warpsPerCTA `[M,N]` | Contributors to one column |
|---|---|---|---|---|---|
| 32×128 / 4 | 3.5.1, 3.6.0, 3.7.1 | `[1,8]` | `[2,16]` | `[4,1]` | 8 threads across 4 warps |
| 32×64 / 8 | 3.5.1, 3.6.0, 3.7.1 | `[1,8]` | `[4,8]` | `[8,1]` | 32 threads across 8 warps |

For the original 32×128 tile, the base CTA tile covers 8×128. Covering all
32 rows repeats the same thread identities four times in M. Each contributor
owns four values in a column, and 32 total elements across eight columns.
There are still exactly one CTA, four warps, and 128 threads per tile.

The full rank-three `order` changes from `[2,0,1]` in 3.5.1/3.6.0 to
`[2,1,0]` in 3.7.1. The interchanged batch dimension has extent one and receives
no additional lanes or warps; the effective M/N ownership is the same. The
complete encodings, including `order`, are retained in `analysis.json` and IR.
The versions' instruction streams differ; equal ownership does not imply
identical generated code or performance.

This differs from the reported **TMA** transition: that path went from the
thread-local column layout to a vectorized, distributed reduction input.
For ordinary `tl.load`, the vectorized/distributed layout was already selected
in 3.5.1. The source context agrees: ordinary coalescing already existed in
[3.5.1 Coalesce.cpp](https://github.com/triton-lang/triton/blob/0add682/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp),
whereas the
[3.6.0 pass](https://github.com/triton-lang/triton/blob/7c56a5e/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp)
also selects layouts for descriptor operations.

Thus the load/reduction layout trade-off also exists on the ordinary path,
but the specific release-to-release layout change in the issue is absent in
these two ordinary-load configurations. This is not a claim that the default
layout is optimal or that all non-TMA kernels are free from regressions.

## Controlled version comparison and results

Evidence root:
`results/local_a10_versions/20261015_gpu0_r2` relative to this project's parent.
The derived [summary](../results/local_a10_versions/20261015_gpu0_r2/summary.md)
contains all layouts, timing ratios, and confidence intervals.

3.5.1 and 3.6.0 use official CPython 3.12 PyPI wheels, verified against the PyPI
SHA256 metadata and unpacked into isolated `/tmp` directories. 3.7.1 reuses the
existing `hexcute` installation. All compilers use the same Python 3.12,
Torch 2.13.0+cu130, and **the same CUDA 12.8 ptxas**. The compiler-resolved ptxas
path is checked, rather than merely recording an environment variable. These
are controlled compiler-version comparisons, not comparisons of the releases'
different bundled assemblers. Package URLs, native hashes, assembler hash,
cache paths, and exact subprocess environments are archived.

Each version compiles in a separate process. All six CUBINs then run in the
same driver-only timing harness, with no Triton import or JIT. Seven sequential
processes on the same physical GPU each perform 12 rounds × 5 single-launch
CUDA-event samples per binary. Rotated/reversed ordering pairs the versions
throughout the run. No profiler duration, CUDA graph, batch division, or
outcome-based repetition enters the table.

| Tile / warps | 3.5.1 (µs) | 3.6.0 (µs) | 3.7.1 (µs) | 3.6.0 / 3.5.1 time ratio, 95% CI |
|---|---:|---:|---:|---|
| 32×128 / 4 | 298.032 | 298.224 | 298.832 | 1.00012 [0.99882, 1.00141] |
| 32×64 / 8 | 436.224 | 436.208 | 436.976 | 0.99989 [0.99973, 1.00006] |

Times are medians of the seven process medians. Ratio estimates are geometric
means of the seven paired process-median ratios; intervals use Student t on
those log ratios (df=6). These summaries need not divide to the same number.
A practical 3% slowdown threshold was frozen before compiling/timing. Both
3.6.0 comparisons and both 3.7.1 comparisons have upper interval bounds below
1.03. **No such release regression was observed for these A10 cases.** This is
a conditional result for one GPU and workload, not independent-GPU inference
or a zero-regression proof for all shapes.

There are 2,520 timing samples, 12 collection-stage complete-output checks
(two finite patterns), and 42 additional driver-only complete-output checks
(the quarter-multiple timing pattern). Every result equals its reference.
Every launch is guarded by the archived CUBIN's SHA. All CUDA return codes are
zero and all binaries have zero spills/global scratch. None of the static
SASS counts is treated as hardware transactions or used to score layouts.

## Evidence, cache reuse, and validation

The first collector attempt at `20261015_gpu0` compiled the two 3.5.1 kernels
but passed only the two pointer arguments to the old `CompiledKernel` wrapper,
which also expects constexpr placeholders. It failed before correctness and
formal timing. Its source snapshot, traceback, and compiler outputs are
preserved. The corrected collector at `20261015_gpu0_r2` uses the archived-CUBIN
CUDA driver launcher for correctness as well as timing; the kernel AST is
unchanged. The failed and accepted roots are both included in the raw manifest.

The first attempt filled a dedicated 3.5.1 cache, which the accepted collector
reused (CUBIN count 2 before and after). The other versions use their own
kernel caches, and every timing process reuses only archived CUBINs. No native
build was needed, so there was no C++ compilation for ccache to accelerate.
The older A10 diagnostic's samples, compiler outputs, and NCU reports were
left untouched.

Offline validation, including committed bytes after committing:

```bash
.venv/bin/python -B -m experiments.tma_reduction_layout.local_a10_versions.analyze \
  --root experiments/tma_reduction_layout/results/local_a10_versions/20261015_gpu0_r2 \
  --validate --committed
```

For a fresh rerun, first create isolated package prefixes and a toolchain
provenance JSON as shown by the archived `toolchains.json`. `experiment prepare`
freezes sources/protocol into a **new** evidence root. `dispatch collect` then
`dispatch timing` run sequentially on physical GPU 0. Do not overwrite an
existing raw root. No later microbenchmark/model stage is authorized by this
diagnostic.
