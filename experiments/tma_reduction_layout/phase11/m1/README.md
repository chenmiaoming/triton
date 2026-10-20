# M1 CPU explanatory audit

This stage explains the frozen H100 models without changing them. D1 is exposed
diagnostic data in this audit. It is never used for fitting, selecting eta/ridge,
or publishing a replacement blind-test score. All prior evidence is read-only.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m experiments.tma_reduction_layout.phase11.m1.run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m experiments.tma_reduction_layout.phase11.m1.validate --committed
```

The protocol and manifest were bound before audit computations. `profile.py`
reuses the original CV scores and does ten development-only fits. `pipeline.py`
reproduces the frozen predictions and evaluates all four input substitutions.
`mechanisms.py` pairs same-vec layouts and reads the original CUBIN debug lines
using local nvdisasm; it does not compile or execute a GPU program. The six
original CPU disassembly command outputs are retained in the new results folder.

Results are under `results/phase11/stage_m1/`: REPORT.md, three detailed JSON
datasets, four CSV tables and four standalone PNG/PDF figure pairs. Missing
control-q observations and unresolved source attribution remain explicit.

The final validator independently verifies the response identity, exact-q and
worker matching, relative RMSE, source-PC instruction-count conservation and
prior hashes, then reruns the audit to check reproducibility. No GPU framework
is imported. A future matched FULL intervention requires a separate experiment;
this stage stops after validation, commit and push.
