# Stage C2d — complete loop-free FULL development matrix

Authorized 2026-10-18. Preserve every pre-existing source, raw sample and compiler
artifact. No C++ change, native rebuild or D1 compilation/timing.

Before compilation move only (16,64,8) from code-only D1 to development. Retain
the other ten strict-zero-mention geometries (five core, five boundary), under
the historical cutoff used by C3.2. Predictions themselves do not constitute
experimental exposure. Freeze the split, source hashes and C3.3 methodology
before collecting any new measurements.

The original eight geometries have 55 unique legal FULL layouts, including 18
unmeasured within-warp layouts. Add the six unique layouts of (16,64,8), giving
61 FULL programs, nine loop-free row-output TMA controls and six resident MC
programs for the new geometry. Reuse exactly 37 FULL and eight controls from
C2c; compile only the remaining 31 programs, CPU-only against the pinned
compiler/native library and existing Triton cache. Export PTX, SASS, resource
usage and CUBIN, with content hashes. ccache statistics must be unchanged.

Use the original host-created descriptor path and rank-3 BF16, 128-byte swizzle.
All source scalars remain block-uniform. No hardware ID backdoor. FULL is the
genuine loop-free TMA/wait/load/max/convert/store program. Old CUBINs are not
rewritten. The host descriptor's allocation is enlarged to 24576x8192 BF16;
its row stride is unchanged. FULL grids cover contiguous 8192/16384/24576 rows
(128/256/384 MiB), using the existing binary's unwrapped row origin. This
avoids fitting quarter/half-input points that can fit in L2. Record the original
8192-row grid separately; descriptor extent is the documented host-side
protocol difference. Runtime grid is an experimental normalization variable,
never a static selector feature. MC retains its original 8192-row slot wrapping.

Measure the entire matrix in each of three fresh H100! workers: paired within
worker, pattern, q condition and visit. This is a necessary matched remeasurement
of frozen anchor binaries, not an outcome-dependent repeat. q conditions are
natural, 1, 2, 4 and 8; use extra dynamic shared reservation to limit occupancy,
skip infeasible caps and deduplicate physical aliases. Record actual q, n_regs,
spills and GPU UUID. FULL/control K=1 at three input footprints. MC uses
K={1,16,64,256}, grids={32,64,96}*132; fit only K>=16. For each cell take three
visits of four CUDA-event samples with two warmups and deterministic randomized
orders. Three independent worker cohorts; no performance-based retries.

Check quarter/random/boundary values against an independent CPU oracle. Every
timed cell checks first/middle/last CTA, including origins beyond row 8192.
Timing and profiling load checked CUBINs through the CUDA driver without
importing Triton/JIT. Preserve every dispatch, return, failed attempt and raw
sample. NCU is separate from timing. Collect all distinct binaries: software
K1 counts for FULL/control; K16/K64 affine counts for MC; hardware wavefront,
bank-conflict, occupancy and DRAM counters at natural and capped q=4. Report
whole-kernel counters without silently attributing output conversion traffic
to ML. Locally reimport every .ncu-rep and compare raw metrics/source counts.

Fit time versus grid/SM to obtain ns/tile/SM and retain the global launch
intercept and fit residual. Report actual 128-MiB times as well. Pair candidates
with the default within the same worker, never pair unrelated worker clocks.
Report all 61 measured candidates, control q responses, new R4 MC response and
the within-warp FULL discrepancy. Do not infer a universal scalar B or naked
packing latency from these measurements.
