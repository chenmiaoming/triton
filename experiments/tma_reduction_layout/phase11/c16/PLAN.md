# C1.6 — Validate the proposed overlap explanation

User authorization: validate the proposed HBM/overlap interpretation. Stay on
H100 and the original host-created descriptor path, pinned upstream/native.
This is a bounded extension of C1.5, not the broader C2 or a compiler change.

Before GPU work, freeze the proposed `max(platform,26.6+C_onchip)` predictions:
platform 61.4/56.1 us, original C1.5 slopes and CTA multipliers unchanged.
The new cohort tests replication; the eight layouts are already observed,
so this is not an unseen-shape generalization test.

Reuse 8 complete-kernel CUBINs and 16 C1.5 microbench CUBINs exactly. Compile
only 2 controls, one per geometry: full original TMA tile, original wait and
descriptor, gather row zero from the original descriptor, write BN FP32 outputs.
This preserves logical input/output bytes, but changes register usage and
occupancy. Call it a transfer control, not pure hardware HBM latency.

Measure full kernels and controls at full, half and quarter grids, both warm
and after a 256 MiB cache eviction outside timing. On the same three workers,
remeasure micro slopes at grid 8448 using K=16/64/256. Each cell has 20
single-kernel event samples. Total 6480 samples; no result-dependent repeats.
Three workers do not imply three independent physical GPUs.

Collect 20 NCU hardware reports: full-grid 8 kernels + 2 controls, warm/cold.
Use application replay and cache-control none so each replay reconstructs
the requested cache state. Collect DRAM read/write bytes, sustained-peak
throughput percentage, L2 sector hit rate, duration, SM utilization/frequency
and DRAM clock. Profiler durations never enter event timing statistics.

Primary analysis keeps all proposed parameters fixed. A separate calibration
uses only N64 vec4 to estimate an empirical offset, independently measured
transfer-control platform, and same-worker micro slopes, predicting the other
7 full-kernel points. The fitted offset is not an isolated physical cost.
Report counter evidence and cache sensitivity even if timing fits well.
Smaller grids diagnose scaling; do not invent an offset scaling rule.

All raw samples, exports, compiler artifacts and reports are write-once.
Reuse persistent kernel/native caches and funded chenmiaoming profile.
Preserve any failed attempt. Finish with one stage commit and validation;
stop before C2 and do not create a PR.

Technical revisions before timing: preserve attempts 1 and 2. Reshape required
explicit output layout conversion; the pinned compiler disallows slicing
through the swizzle pattern. Revision 3 uses the supported descriptor gather
and requires CPU-only sm90 compilation to pass before another GPU allocation.

Revision 4 fixes a profiling dependency before any counter samples succeeded:
the established NCU tools image has no Torch, so its child uses CUDA Driver
allocations, copies and eviction. Kernels and measurement conditions remain
unchanged. Keep failed profiling attempt 1 and successful attempt 2.

A separate frozen submission diagnostic adds 800 paired direct/Graph event
samples on one worker, no new programs. Each graph is checked to have exactly
one original kernel and two external event record nodes. Its source snapshot
and protocol are separate; primary predictions and 6480-sample plan unchanged.
