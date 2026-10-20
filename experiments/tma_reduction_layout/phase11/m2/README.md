# M2 FULL output-policy experiment

H100 only, original descriptor route and pinned 20d9d98 compiler/native image.
All original C34/D1/M1 evidence stays read-only. The exposed geometries are
mechanism diagnostic data, not new blind tests or selector calibration.

The protocol preceded compilation. There are 20 new matched-source fixed/native
FULL programs plus 13 reused fixed/control CUBINs, complete-output correctness,
three worker timing calls and six independent NCU reports. Exact commands,
environments, samples and source snapshots are saved in stage_m2.

Recompute without allocating a GPU:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m experiments.tma_reduction_layout.phase11.m2.analyze services
.venv/bin/python -m experiments.tma_reduction_layout.phase11.m2.analyze counters
.venv/bin/python -m experiments.tma_reduction_layout.phase11.m2.offline_import
.venv/bin/python -m experiments.tma_reduction_layout.phase11.m2.diagnose
.venv/bin/python -m experiments.tma_reduction_layout.phase11.m2.report
.venv/bin/python -m experiments.tma_reduction_layout.phase11.m2.validate --committed
```

Paid acquisition was preregister → CPU preflight → compile gate → correctness/
resource census → artifact gate → timing_all + profiles_all. Every Modal call
set both MODAL_PROFILE=chenmiaoming and --profile chenmiaoming. GPU routines load
frozen binaries through the CUDA driver and cannot import Triton/JIT. Do not
rerun completed paid cells as a replication shortcut; reuse the saved evidence.
