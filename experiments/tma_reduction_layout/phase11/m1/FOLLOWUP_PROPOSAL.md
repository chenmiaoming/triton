# Proposed M2 — identify the output-policy interaction

This is a review proposal, not a launched experiment. M1 allocated no GPU and
did not compile new kernels. Do not modify the frozen C34/D1 artifacts.

The M1 matched-pair audit found that the same coarse MO flags can lower to
different communication strategies. For 32x64/w8 vec2, the retained source-PC
counts attribute at least 16 SHFL warp instructions to the N-first output
conversion, versus 16 BAR, 2 LDS and 8 STS for within. The present FULL contrast
therefore includes an output-policy interaction, not an isolated MR cost.

First develop a static operation plan for SliceLayout-to-output conversion,
using the pinned compiler's LinearLayout/lowering semantics. Validate the plan
against retained CUBINs and PC counters. Report instruction grouping/fusion and
missing source attribution. Do not turn static register tile width into a
hardware load count.

Then consider a small Gluon FULL factorial diagnostic on development geometries
32x64/w8 and 32x128/w4, vec2 only:

| Reduction layout | Output policy |
|---|---|
| N-first | Existing fixed Blocked output |
| within | Existing fixed Blocked output |
| N-first | Store in the native reduction-result layout |
| within | Store in the native reduction-result layout |

This would add four logical programs. Reuse the four original fixed-output
CUBINs and the two independent operational-control CUBINs. Keep the original
descriptor, BF16 max semantics, loop-free one-pass path, compiler PIN, native
library SHA and shared layouts. No hidden lane/warp-varying scalars or C++ edits.
The native-output indexing must be a valid tensor layout and emit exactly the
same logical output. Validate the generated participation mask and all values.

Freeze choices, code, expected operation plans and failure handling before
compilation/timing. Keep load/cast/max on separate source lines in new programs
to improve provenance. A source rewrite is never assumed to preserve machine
code; compare actual instruction sections and resource assignments.

Pair all variants on the same workers using the established three-footprint
FULL service-slope protocol and common feasible q capacities. Inspect natural
q and controlled q=1,4; retain infeasible conditions. Use CPU oracle patterns,
CUBIN load hashes, n_regs/local storage, dynamic shared reservation and driver
occupancy capacity. Profile separately with NCU and locally reimport reports.
Reuse caches; no Triton native rebuild.

The target contrast is

    (FULL_within_fixed - FULL_Nfirst_fixed)
      - (FULL_within_native - FULL_Nfirst_native).

It measures how the output policy changes the layout gap. It is not
automatically pure MO latency: native output can change global-store width,
predication, register allocation, or upstream scheduling. Check ML/MR generated
code, dynamic instructions, output stores and matched capacity. If these change,
retain the result as a joint output-policy effect rather than subtracting an
isolated MO probe as a constant.

Before any new GPU launch, finalize the static legality gate, exact candidate
code and preregistration. Keep adverse/zero interactions. Do not use this
diagnostic or exposed D1 to claim a new held-out selector win. Model changes
need a separate development version and subsequent independent validation.
