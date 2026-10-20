# M3: prebound tests of M2-derived mechanisms

This stage freezes three new hypotheses before its own compilation/measurement:
the mixed-transposition conversion threshold, native store-sector request
amplification, and reduction-result replica demand under a different consumer.
M2 and earlier evidence remain immutable; no selector or blind score changes.

Thirty FULL/control programs are timed, including reused M2 anchors and new
vec4 programs. Four all-row consumers are count-only diagnostics at grid=1.
Their different output shape has a separate CPU oracle and they never enter
FULL service slopes or speedups. Source-level variation uses tensors/layouts.

Recompute from retained evidence without a GPU:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m experiments.tma_reduction_layout.phase11.m3.analyze services
.venv/bin/python -m experiments.tma_reduction_layout.phase11.m3.analyze counters
.venv/bin/python -m experiments.tma_reduction_layout.phase11.m3.offline_import
.venv/bin/python -m experiments.tma_reduction_layout.phase11.m3.diagnose
.venv/bin/python -m experiments.tma_reduction_layout.phase11.m3.report
.venv/bin/python -m experiments.tma_reduction_layout.phase11.m3.validate --committed
```

The final validator checks prior hashes, prebound static predictions, source/
CUBIN/ABI bindings, complete-output canaries, all retained samples, independent
equally spaced OLS slopes, same-worker exact-q contrasts, software-PC count
conservation, report re-import and count-only consumer exclusion from timing.
It does not require hypotheses to be true: mismatches are valid outcomes.

Paid Modal calls explicitly used chenmiaoming and the existing native/compiler
and Triton caches. Do not repeat successful cells or modify frozen evidence.
Finish this stage's commit/push and stop for review before another stage.
