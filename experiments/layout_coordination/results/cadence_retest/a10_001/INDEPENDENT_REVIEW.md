# Independent A10 eligibility review

The frozen CPU and actual GPU eligibility gates passed. **Formal timing has not started:** all three temporal process cohorts and all 33,480 planned observations remain pending after the user's pause request. This file records eligibility, not a performance result.

- Protocol SHA256: `c76c2979e39a333a4109290c9a7db8ee59c28c10d86a185d7e5edd09facc137a`.
- Actual eligibility gate SHA256: `6380a8458f3f920e863467c4a2318ccb577df9c9e26cddc6000a1fc49e7a896c`.
- Independent CPU checks matched frozen source and snapshot hashes, all 360 protected original files, 28 archived programs, eight U-repeat controls, and two authentic latest-version U/G0 pairs. No compilation was performed. Older releases retain U-only controls.
- Actual hardware was physical GPU0, UUID `GPU-69464998-e901-74ba-63ff-ae7022f8aed9`, NVIDIA A10, compute capability 8.6, 72 SMs. It was not presented as H100 hardware.
- All 28 captured graphs contained eight untimed priming kernels, one target kernel, and two events. All 28 direct checks, 28 captured-graph checks before timing, and 28 captured-graph checks after timing matched the complete independent 2,097,152-element oracle and passed guards. Four read-only 128 MiB input checks also passed.
- All eight process checks before and after the four contexts passed. They establish boundary checks, not continuous proof of exclusive use. The 157 telemetry rows are sampled readbacks; clocks were not locked, and these are not per-kernel frequency measurements.
- The result recorded zero formal timing samples, zero Triton imports/compilations/native rebuilds, no profiler, and successful CUDA calls and cleanup.

The reviewer independently reran `cpu_gate(protocol, payload, snapshot)` and `a10_report.validate(root, eligibility=True)` against the retained files. Both passed. If authorized to resume, formal timing must retain this protocol and gate binding; the intended repetitions are sequential fresh processes on **one physical A10**, not independent GPUs.
