# G4a evidence and reproduction

The active freeze is `preregistration/protocol_v2.json` and the active executable archive is `compiled_payload_v2.zip`. Version 1 remains write-once historical evidence: CPU feasibility passed, but actual Tensor JIT rejected the three auxiliary integer specializations before any formal timing. Its CPU archive, first bind failure, first gate and source snapshot are retained. V2 restores the original runtime ABI and reproduces the G2 U/G0/stage2 CUBINs exactly.

No native rebuild or C++ edits. The existing compiler image, native hash, kernel cache and funded Modal profile are recorded in the protocol. H100! is required; automatic substitution by H200 is not allowed.

Existing results can be checked entirely on CPU:

```bash
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g3_validate --stage all
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_validate --stage g4a
```

The original collection sequence below is for a fresh evidence directory only. Do not rerun the completed matrix in place. Restore the frozen `source_snapshot_v2.zip` to `/tmp/operator-gpu-g4-worker-snapshot-v1` before importing Modal modules; the protocol checks each worker source SHA. CPU compilation was attempt 002 and the matching actual JIT binding was attempt 002; complete correctness and profiles were attempt 001.

```bash
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_run --stage g4a --action cpu --attempt 2
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_gate g4a
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_run --stage g4a --action bind --attempt 2
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_run --stage g4a --action collect
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_run --stage g4a --action timing_all
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_profiles --stage g4a --action check --attempt 2
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_profiles --stage g4a --action collect
```

Offline analysis commands: `g4_analyze g4a timing`, `counters`, `code`, `attribution`, `plans` (one action per invocation). All twenty reports are locally re-imported with `/usr/local/cuda/bin/ncu`; the numeric metrics and dynamic PC counts must equal the remote exports. `g4_plot g4a` creates standalone figures and `g4_report g4a` renders all predeclared cells. Derived files are deterministic except the figure export metadata; existing evidence must not be replaced.

Software counts are warp instruction instances for one CTA with eight KV iterations, not per-thread arithmetic operations. Application replay uses eight same-target warmups, cache-control none, clock-control none. Timing is collected separately from NCU. Compiler debug inline attribution is an operation locator and does not assign cycle cost or prove the critical path.

The earlier CLI invocation with `--action timing-all` failed its local action assertion and dispatched no timing jobs; the successful value is `timing_all`. No outcome-dependent timing repeat occurred.
