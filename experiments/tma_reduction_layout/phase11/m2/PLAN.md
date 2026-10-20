# Stage M2 — FULL output-policy intervention

The user authorized H100 GPU experiments and additional hypothesis-driven
contrasts. Preserve all prior stages and frozen selector roles. This stage is
mechanism diagnosis on exposed geometries, not a new blind selector score.
Use the funded chenmiaoming Modal profile explicitly, pinned compiler 20d9d98,
the existing native library and cache. No C++ or native rebuild, PR, or D2.

Primary design: 32x64/w8 and 32x128/w4, vec2, N-first versus within, with fixed
versus native output. Extensions fixed before results: vec8 at those geometries,
and vec2 at the exposed 16x256/w8 failure. Ten logical reduction layouts total.
For each layout retain the old fixed-output CUBIN and compile matched fixed and
native output programs from the same new source. This source-rewrite bridge
prevents assuming new line information or a new function leaves code unchanged.
Add the three existing operational TMA controls: 33 logical programs, 20 new
CPU-only kernel compiles, 13 reused binaries. Deduplicate identical binaries.

Every FULL computes the identical BF16-input, FP32-output column maximum with
the original host descriptor route, input shared layout and single-use widening.
Native output uses the valid double SliceLayout and tensor offsets; compiler
masking must emit exactly one logical output value. No hidden varying scalars.

Hypotheses and failure criteria:

1. Output conversion changes communication topology. Derive owner-to-owner
   reach statically using blocked linear bases, and check conversion source
   instructions against old and new CUBINs/NCU. Native output should remove the
   explicit conversion, but store width/predication and reduction scheduling
   can also change. Do not call reach a timing or hardware transaction count.
2. Output policy contributes to the N-first/within FULL gap. Primary response
   is (within-fixed minus NF-fixed) minus (within-native minus NF-native), in
   ns/tile/SM, measured on the same worker and exact capacity. Zero/opposite
   interactions refute the proposed directional explanation. A nonzero value
   is a joint output-policy interaction unless upstream instructions, global
   stores, resources and capacity are demonstrably controlled.
3. The q=1 sign-error cases persist under matched source. Compare the original
   frozen models without refitting and record source-bridge differences. If
   output policy does not repair the direction, retain a FULL response failure.
4. Vec8 expansion tests whether the effect grows with local column width and
   communication plan. Report all directions, not just an agreeing case.

Timing follows C2d/D1: one-kernel CUDA Graph, three independent worker calls,
randomized three visits of four samples, two warmups per visit, 128/256/384 MiB
inputs and OLS time-versus-grid/SM slope. Raw 128 MiB time stays auxiliary.
Natural and controlled q=1,2,4,8 (aliases and infeasible conditions explicit).
q is driver occupancy capacity, not observed instantaneous CTA residency.
All variants are paired on the same worker; monitor any reservations that make
q controls infeasible. No performance-based reruns or removing adverse samples.

Driver-only execution loads SHA-checked frozen CUBINs, checks ABI, native SM90
132-SM H100 identity, resources/spills, and CPU oracle patterns. Before timing,
run a complete-output 2 MiB and sampled large-footprint correctness check.
NCU is separate: software InstructionStats/per-PC at grid=1; hardware at natural,
q=1 and q=4 on the full original grid. Collect shared wavefronts/conflicts,
global-store sectors, executed instructions, DRAM bytes/utilization and active
warps; retain reports and locally re-import. NCU durations are not formal timing.

Freeze code and protocol before CPU preflight. Compilation or infrastructure
fixes require append-only execution revisions preserving failures. Extra GPU
interventions after these outcomes require a separately frozen hypothesis and
matrix within this authorization. Stop for budget errors; ask the user to fund
or switch profile. Complete report, validator, commit and push before stopping.
