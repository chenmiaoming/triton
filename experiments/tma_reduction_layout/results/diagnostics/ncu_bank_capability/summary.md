# NCU shared-memory bank-counter capability probe

One current Modal H100 worker successfully returned nonzero shared load/store bank-conflict and wavefront counters. This confirms these four counters are readable in this specific environment. Two requested request counters were absent from both the metric query and the collected report despite collection exit code 0; they are unavailable evidence, never zero-valued observations.

| Requested counter | Result | Count |
| --- | --- | ---: |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | COLLECTED | 29,817 |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum` | COLLECTED | 330,891 |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` | COLLECTED | 3,568,761 |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum` | COLLECTED | 3,689,611 |
| `l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum` | NOT_RETURNED | unavailable |
| `l1tex__t_requests_pipe_lsu_mem_shared_op_st.sum` | NOT_RETURNED | unavailable |

## Environment and execution

- Dispatch: `fc-01M4574YZX9080185N77XEEA3Y`; Modal profile `chenmiaoming`; [Modal app](https://modal.com/apps/chenmiaoming/main/ap-Ls0OwqspyF2R7elsFg8KAy).
- GPU: NVIDIA H100 80GB HBM3, compute capability 9.0; UUID `GPU-c06a13a8-8174-9c16-b68c-7a1c7d2725c7`; checked CUDA driver API version 13000.
- Remote NCU: 2024.3.2.0. UID 0; CapEff/CapPrm `00000000a80405fb` (CAP_SYS_ADMIN bit not set); observed driver policy `RmProfilingAdminOnly: 0`.
- Native core image `im-joNh6Ry3lq2oFqOues9VFh` reused; resolved image `im-HwskQEWPovCkEw73QhlVjX`. No native rebuild or new kernel compilation.
- Source snapshot time: `2026-10-05T05:02:55.656276+00:00`. All 1,881 uploaded source files matched the local manifest on the worker.

The probe loads the pre-existing Phase 7 archive `M128_N32_w16:canonical:4`, CUBIN SHA256 `c1bab922951f283f6413e6ddcb4bf7aa439368ee74f28d7a528827987b820d2e`. The existing ELF-only launcher uses fully initialized BF16 `[65536,128,32]` input, full output and scratch allocations, one warmup, then one explicitly delimited profile launch of 16,384 CTAs with 512 threads/CTA. Both launch SHA guards matched; the first four output rows passed the existing correctness check with max absolute difference 0. Triton was never imported by the launcher and compile calls were 0.

Collection uses kernel replay, cache-control `all`, clock-control `none`, and profile-from-start `off`; the report records one replay pass. NCU duration is incidental profiler output and is not used as formal GPU timing. This probe adds no new scientific phase and changes no Phase 7 inference.

## Limits on the conclusion

The previous Phase 7 profiler reports requested DRAM/instruction/duration metrics and did not collect bank-conflict counters. Their success alone did not establish bank-counter availability. This new probe supplies that missing direct observation on one worker; it does not explain the user's historical failure or guarantee availability on every Modal profile, GPU, worker, or NCU version.

The counts are aggregate profiler diagnostics for this one archived kernel. They do not identify individual instructions, isolate a descriptor/store/reduction mechanism, establish a layout comparison, or measure an intrinsic bank-conflict cost. NVIDIA describes shared requests, wavefronts and bank conflicts in its [profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#shared-memory); source-level ideal/excess wavefront analysis would require separate evidence.

## Evidence and validation

`original_export.zip` is the exact, unmodified remote return. It contains the complete original `version.json`, `query_metrics.json`, `collection.json`, `result.json`, and binary `bank_report.ncu-rep`. The very large query/result JSON extractions remain untouched locally and are stored losslessly inside that tracked ZIP rather than duplicated in Git. No query output was truncated. `raw_manifest.json` binds every original file and ZIP member by SHA256.

`request.json`, `dispatch.json`, `probe_source.py`, `modal_cli.log`, `export_binding.json`, the extracted small command records/report, and the offline import record are retained. Local NCU 2026.3.1 imported the saved report without GPU collection; all four requested, returned counter values matched the original CSV exactly. Missing request counters remained missing.

Run the offline validator from the repository root:

```sh
python3 -m experiments.tma_reduction_layout.diagnostics.validate_ncu_bank
```

`validation.json` records PASS for original-byte closure, source provenance, frozen Phase 7 CUBIN/PTX bindings, actual launcher flags and guards, metric presence/value checks, and the saved-report roundtrip. All **8,158** existing experiment files at `146e307cf34cbb55a48e531b22fa99f7f1456a7f` remain byte-identical. The validator reads large original members directly from the tracked ZIP, so it also works in a fresh checkout without those optional extractions.
