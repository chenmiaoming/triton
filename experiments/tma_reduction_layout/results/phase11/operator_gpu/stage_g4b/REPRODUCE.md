# G4b evidence and reproduction

Active protocol: `preregistration/protocol_v2.json`; executable package: `compiled_payload_v2.zip`; CPU and binding attempts: 001. The initial protocol draft was superseded before any G4b compile/timing because G4a's actual Tensor JIT gate caught the shared preflight ABI problem. Both drafts and source snapshots remain unchanged.

This is per-workload JIT at the pinned source kernel entry points, using actual CUDA tensors. It does not install a vLLM/SGLang server or claim framework/whole-model acceleration. C06 reports merge only; C02 reports both stage1 and its unchanged two-stage device pipeline. All old B4 counterparts reuse G2 CUBINs and exact runtime ABIs; they are not recompiled.

Validate the completed evidence without allocating a GPU:

```bash
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_validate --stage g4b
```

For a fresh evidence namespace, restore the frozen source snapshot and protocol first. The sequence was CPU preflight, `g4_gate g4b`, actual Tensor JIT, full correctness, then three independent timing workers:

```bash
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_run --stage g4b --action cpu
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_gate g4b
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_run --stage g4b --action bind
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_run --stage g4b --action collect
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_run --stage g4b --action timing_all
```

Use explicit `MODAL_PROFILE=chenmiaoming` and `--profile chenmiaoming`; compiler/native/cache and H100! are fixed as in G4a. No new NCU collection is required by G4b's pre-registration. B8 counter evidence may only be reused where binary, ABI and workload/launch context match; it is not extrapolated to B1 or B64.

Offline commands: `g4_analyze g4b timing`, `code`, `plans`; `g4_plot g4b`; `g4_report g4b`. Timing aliases use CUBIN SHA, runtime ABI, specialized scalar values, warps, shared memory and the stage2 binding at the same workload. Identical bindings are measured once per worker and logical labels remain in the full tables. All raw samples and comparisons, including negative and below-3% results, are retained.
