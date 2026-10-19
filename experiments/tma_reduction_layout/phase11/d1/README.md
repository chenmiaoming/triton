# D1 frozen-policy held-out H100 validation

Use `.venv/bin/python` from the repository root. Results live under
`experiments/tma_reduction_layout/results/phase11/stage_d1/`.

The write-once protocol and original prediction bindings preceded all D1
compilation. The primary model is C34_resource_only. No parameter, decision,
model role or endpoint may be changed using these outcomes.

All paid jobs have completed. Reproduce the analysis locally without a GPU:

```bash
.venv/bin/python -m experiments.tma_reduction_layout.phase11.d1.analyze services
.venv/bin/python -m experiments.tma_reduction_layout.phase11.d1.analyze counters
.venv/bin/python -m experiments.tma_reduction_layout.phase11.d1.offline_import
.venv/bin/python -m experiments.tma_reduction_layout.phase11.d1.counter_summary
.venv/bin/python -m experiments.tma_reduction_layout.phase11.d1.evaluate
.venv/bin/python -m experiments.tma_reduction_layout.phase11.d1.report
.venv/bin/python -m experiments.tma_reduction_layout.phase11.d1.validate --committed
```

The final validator recomputes all grid slopes and score/coverage tables,
independently checks equally spaced grid slopes and CPU oracle checksums,
verifies graph structure, CUBIN loads, raw archives, offline NCU imports,
all earlier file hashes, source bindings and the final freeze.

The historical acquisition sequence was preregister, Modal preflight,
`analyze compile`, Modal collect, `analyze artifacts`, Modal timing_all,
Modal profiles_all, offline analysis, final validation/freeze and commit.
Each Modal invocation explicitly used both `MODAL_PROFILE=chenmiaoming`
and `--profile chenmiaoming`, on the pinned images and shared cache volume.
Existing output paths reject duplicate dispatches. Reproduce from retained
raw data; do not rerun paid D1 measurements or start another stage without
new authorization.
