# Gluon operator survey preparation

Eight candidate families are in [CANDIDATES.md](CANDIDATES.md); the comparison
rules and stop point are in [PLAN.md](PLAN.md). This preparation provides nine
canonical semantic specializations, paired Triton/Gluon functions, explicit
layout variants, independent CPU references and a CPU-only compilation gate.

These fixtures are not the original library binaries. Production adapters,
whole-graph same-layout bridges and GPU correctness remain required before any
claim about upstream performance. In particular, the FP8 fixture is not the
MXFP8 E8M0 recipe, and P·V is not a complete attention kernel.

Create a new, write-once preparation root:

```bash
.venv/bin/python -B -m experiments.tma_reduction_layout.phase11.operator_survey.prepare \
  --output /tmp/operator-survey-protocol
```

Pinned CPU compilation reuses the existing native image and kernel-cache
volume. There is no GPU-decorated function in this app:

```bash
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming \
  -m experiments.tma_reduction_layout.phase11.operator_survey.cpu_modal \
  --protocol /tmp/operator-survey-protocol/protocol.json \
  --output /tmp/operator-survey-pinned-cpu
```

For development-only local compilation, explicitly select `development` and
use the installed hexcute Python. The default compiler policy rejects a native
SHA different from Phase 11. No native/compiler build is needed.

```bash
/home/chenmiaoming/miniforge3/envs/hexcute/bin/python -B \
  -m experiments.tma_reduction_layout.phase11.operator_survey.cpu_compile \
  --protocol /tmp/operator-survey-protocol/protocol.json \
  --output /tmp/operator-survey-development-cpu \
  --cache-dir /tmp/triton-exp-operator-survey-cpu371-kernel-cache \
  --compiler-policy development
```

Validate an exported CPU attempt and its frozen sources, compiler artifacts,
layout coverage and CPU semantic references:

```bash
.venv/bin/python -B -m experiments.tma_reduction_layout.phase11.operator_survey.validate \
  --attempt /tmp/operator-survey-pinned-cpu
```

The validator explicitly reports `GPU_kernel_correctness_verified=false`.
No timing, NCU profiling, cost-model fitting or PR is started by these commands.
