# Stage C2c — Composition, fixed-R contrasts, and resource calibration

Authorized 2026-10-17. Original host-created BF16 TMA descriptor, H100 sm90,
pinned compiler/native, no C++ edits or native rebuild. Reuse immutable C2a
MC CUBINs where identical. Keep every prior stage and raw export unchanged.
Use the funded chenmiaoming profile and existing persistent compiler cache.

Freeze an empirical-exposure audit and D1 protection list before any new
compilation. Include every unexposed legal core geometry and all eight legacy
holdouts. No D1 candidate is compiled, loaded, profiled, or timed here.
The remaining pool cannot supply independent R=8 or BM=32/64 core holdouts;
document that coverage gap and separate retrospective coverage from blindness.

Composition calibration spans eight non-D1 geometries. Compare original
single-pass full reduction with fresh-origin repeated transfer/full programs,
resident two-slot MC, resident first-row read/write, and a direct first-row
scaffold. Measure K1/2/4 separately and K16/64/256 steady slopes, with multiple
grid sizes and full-footprint K1. Keep global launch/tail intercepts out of tile
parameters. Repeated-transfer slot-reuse barriers are explicit fixture work.
The 26.6 us historical residual is neither subtracted nor fitted as a known base.
Test the base-zero and additive-tile envelope against held-out calibration
geometries. If a common B_tile and pure transfer cannot be identified, report
that failure and bounds rather than inventing component latencies.

Fixed-R probes cover R4/8/16/32 and every power-of-two p divisor, with columns=R/p.
Use a logical owner tensor axis and no source hardware IDs. Each matched group
has native BF16-friendly max, an exact FP32-scale/max/unscale version, and two
corresponding max-deleted fixture controls. All observe every final BF16 state
and use the same per-group FP32 checksum. Across p, grouping changes as intended
for a local partial-reduction stage; no claim of identical full-kernel semantics
or naked instruction latency. Compare paired increments at q1/2/4/natural;
verify lowering and dynamic instruction scaling with NCU. Failed contrast
isolation, optimizer folding and zero differences are retained.

Record physical n_regs/local storage and spill instructions for all compiled
programs, including two separate R128/256 non-D1 resource calibration geometries.
Resource-only programs are not performance training targets. Do not infer
spill from logical BF16 element count.

Three fixed timing invocations, 12 graph-event samples per cell, random
interleaving, no outcome-dependent reruns. NCU reports are collected separately
from event timing. Validate raw bindings, correctness, counter imports, reserved
holdouts, cache/native identity and prior hashes; commit this stage separately.
Then C3.2 may update derived models and freeze stratified predictions. Stop
before D1 compilation/timing and do not create a PR.
