# Stage M3 — conversion threshold, store sectors and reduction-result consumers

M2 is complete and frozen before this protocol. Three hypotheses follow from
its counterexamples. Keep M2 and all previous files unchanged. Use Gluon with
the pinned compiler/native image, funded chenmiaoming profile and H100 only.
No selector fitting, D1 re-scoring, C++ edits, PR, or D2.

H1: minimum warp-local reach is insufficient to predict output conversion.
The pinned cvtNeedsWarpShuffle also requires fewer than two mixed register/lane
transpositions. Implement its restricted contiguous-N permutation-cycle count,
freeze predictions, then test vec4 between existing vec2/vec8 anchors. Expected:
warp-local vec4 has two transpositions and uses shared, not shuffle. Cross-warp
conversions remain shared. A different observed lowering refutes this restricted
predictor; do not modify predictions to agree after collection.

H2: native vec8 store-sector amplification is a request granularity effect.
FP32 vec2/4 fit one <=128-bit store per owner; vec8 needs two, potentially
touching each sector twice. Freeze per-warp/per-instruction sector sets from
logical owner coordinates. Expected native vec4 uses the same sectors/CTA as
fixed, while vec8 uses twice as many; this is not a DRAM-byte prediction. Check
actual STG widths and NCU sectors, including zero/opposite outcomes.

H3: the extra LDS reduction in native output includes pruning of unused
reduction-result replicas. For N-first vec2/vec8 at the two development
geometries, add a diagnostic consumer that writes max(column)+logical_row for
all BM rows. Every row now needs the result. Logical row/column tensors use
valid SliceLayouts; no hardware-ID scalar, hidden divergence, or inline asm.
This different output is profiled at grid=1 only, never timed or counted as FULL
performance. Expected whole LDS counts restore fixed whole LDS minus explicit
MO LDS if max packing, input loads and reduction structure remain unchanged.
If upstream code changes, report the joint effect instead of isolating return
broadcast cost. Store bytes differ by design; never compare its timing to FULL.

Matrix: two existing development geometries (32x64/w8, 32x128/w4); vec2/4/8;
N-first/within; matched fixed/native output = 24 FULL cells. Reuse the 16 M2
vec2/8 binaries; compile eight vec4 FULL programs. Add four old vec4 fixed
binaries as source bridges and two existing controls = 30 timed programs.
Add four profile-only broadcast consumers = 34 logical programs, 12 new CPU
compiles and 22 reused binaries. No other shape or architecture is added.

Use the same correctness patterns, three-worker single-kernel CUDA Graph
protocol, natural/q=1/2/4/8 capacity conditions, and 128/256/384 MiB slopes as M2.
No performance-based repeats. All program outputs receive full 2 MiB input
correctness/canary checks; dense diagnostic outputs use their own CPU oracle.
NCU separately collects source/dynamic counts and hardware metrics. Broadcast
consumers use grid=1 for all counter conditions; FULL/control use original full
grid. Re-import every report on the local CPU host. Reuse caches; no native
rebuild. Preserve failed artifacts and register/spill changes.

Complete report and validator, commit and push this stage, then stop for review.
