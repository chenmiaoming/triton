# Stage C2 — Formal calibration

The user's 2026-10-16 instruction authorizes C2a and C2b, separate commits.
Do not create a PR or advance to C3. Pinned upstream 20d9, unchanged native,
H100!, funded chenmiaoming profile, original host-created descriptor path.

C2a: 15 unique layouts × (ML D1/D2, MR D1/D2, MO, MC) = 90 logical
programs. Reuse the eight N-first ML CUBINs from C1 and MR CUBINs from C1.5
exactly, with all 128 original NCU reports. Only 74 logical programs need
new compilation; exact aliases share physical execution with traceable IDs.

C2b: the 40 p/u/w/e D1/D2 probes in FOLLOWUP_PLAN section 3, followed by
feasible q=1/2/4/natural controls of the calibration programs. No source-level
lane/warp IDs, assembly, compiler changes or reduction-result broadcast.

MR uses the corrected BF16 private update before single-use widening and max.
The earlier draft's result-broadcast feedback is superseded by C1/C1.5.
ML retains two fixed shared slots and independent full-tile checksums. D2
has two independently observed outputs, not a merged checksum.

MO initializes a nonconstant FP32 vector in the native SliceLayout from
the actual reference reduction results, then converts and stores. Its input
load is fixture work. If the compiler eliminates or changes the conversion,
MO is diagnostic and the output subgraph remains jointly identified in MC.

MC alternates the two shared slots and executes shared-load → single-use
widening → max → fixed output conversion → store on every iteration. Each
iteration writes a distinct output tile; no checksum or private recurrence.
The observable output grows as K. K1 transfers only the first original tile
and performs the original complete computation. Its CUBIN still reserves
two slots for runtime K; this resource difference is explicitly recorded.
The old eight complete-kernel anchor CUBINs remain available unchanged.
No claim that MC K1 is byte-identical to an original lower-resource binary.

All new code compiles on CPU for explicit sm90 before any paid GPU check.
GPU jobs load frozen CUBINs through CUDA Driver, no Triton import or JIT.
Correctness covers complete small-grid output, finite quarter/random/boundary
inputs and K1/16/64/256. Source/hardware NCU gates precede formal timing.
Batch several K/repeat launches per report to amortize input preparation;
retain distinct launch IDs and per-launch source/metric exports.

Natural timing: grid1 and 32/64/96×actual SM count, K1/16/64/256 for repeated
fixtures, one-shot MO. Three workers, 20 samples/cell, one kernel between
two external event nodes in each CUDA Graph, no batch division. Existing
timing samples remain immutable; new timing is needed for matched D/q and
steady-state normalization. Full-output buffers are bounded before allocation.

q changes only unused dynamic shared reservation. Set/read back 100% preferred
shared carveout consistently where supported; record unsupported controls.
CUDA occupancy gives a capacity bound, NCU reports achieved occupancy.
Unattainable q and spill are retained as resource boundaries, not imputed.

Normalization: first fit T(K)=a+bK on K>=16 at each measurement grid. Then fit
b(G)=a_grid+(G/SM)*D*s_tile over three grids, retaining adjacent slopes and
nonlinearity. s_tile is an effective per-SM tile service interval; grid1 D1
gives a separate effective dependency-chain constraint. Neither is a naked
instruction latency. Reuse the independently checked C1.6 TMA controls, with
matched full-grid timing, for per-SM transfer service. Do not reuse the 26.6us
empirical offset. Record MC output-traffic and fixture/overlap residuals;
do not label a residual as pure overlap when fixture or codegen differs.

Export static features only from (M,N,W,dtype,layout); resource binaries and
measured grid are diagnostics/training provenance, not selector features.
HBM/on-chip quantities use the same tile/SM units. A rho proxy is explicitly
limited to identifiable measured subgraphs; no forced scalar when a component
remains unidentifiable. C3 needs a separate model and validation decision.

Raw samples, artifacts, NCU reports, source snapshots and failures write once.
Derived analyses may update. Protect all previously committed result bytes.
