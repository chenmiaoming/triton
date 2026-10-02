# Phase 4 Stage A — Pre-registration and cohort freeze

This directory performs performance-field-independent programmatic cohort
selection, frozen before any new Phase 4 mechanism measurements. Historical
canonical outcomes existed before this preregistration, so human-level complete
outcome blinding cannot be claimed. The unit is `(M,N,num_warps,default→cand4)`. It imports no GPU/runtime/compiler or
Modal package, collects no samples, and changes no kernel or production heuristic.
The trusted parent is `ce04afd7d9095349d54a007e61ec2b5d4fd557b6` on
`explore/tma-reduction-layout`.

```bash
python3 experiments/tma_reduction_layout/phase4/preregister.py
python3 experiments/tma_reduction_layout/phase4/validate_preregistration.py
```

`preregister.py` projects explicit structural keys from the mixed Phase 2 JSON.
Its pure selector does not receive performance or resource metadata. All 30
transitions and every exclusion are retained. `canonical_eligibility.json` is a
separate archival/resource layer; absent evidence is pending and cannot alter
the structural pool. Historical empty `reduce_ops` metadata is acknowledged;
source AST supplies the logical M-axis contract, and Stage B must verify actual
TTGIR. Existing anchors are labels, not automatic primary exclusions.

The validator independently recomputes membership, reasons, coverage and anchor
labels. It replaces and deletes performance fields with deterministic extreme
values and raises on any attempted access, checks resource independence, tests
structural failures, and verifies complete condition schedules. Synthetic tests
exercise generic artifact gates against existing text archives without compiling
or measuring anything. Source/code/protocol byte hashes are bound in
`source_bindings.json`.

`artifact_gate.py` is an offline future archive auditor, not a compiler or timing
runner. It checks complete body fingerprints, full LocalLoad sequences, actual
TTGIR axes/layouts, repeated-loop/memory effects, spills, copy observations,
actual CUBIN archives and exact-binary occupancy records. Canonical single
reproduction requires the entire exact LocalLoad opcode sequence; repeated
pre-loop `DIFFERENT_ENCODING` is recorded without automatic exclusion when
R=0, fixed binary and all isolation/resource gates pass. R=0 subtraction does
not prove absence of register-allocation/live-range/scheduling interactions. Future manifests must
include all fields specified in [PROTOCOL.md](PROTOCOL.md). Lack of complete
archives prevents readiness, including for existing anchors.

Stage A produces only preregistration outputs. After the local freeze commit,
stop for review: do not push, run Stage B, run Modal/H100, or collect R=0/1 data.
Any structural/gate/protocol amendment requires an explicit new preregistration
version and review before observations; never amend it based on performance.

`analysis_contract.py` freezes continuous-value Spearman average-rank ties,
leave-one-case-out sensitivity, and the df=2 Student-t SIGN_UNRESOLVED band.
Stage A exercises it only with synthetic numerical fixtures. PRIMARY n<5
cannot establish cross-shape generalization; n>=5 remains descriptive cohort
evidence. The original 20-transition/360-condition master schedule is retained
byte-for-byte and only gate filtering preserving relative order is allowed.
The 360 measurement conditions comprise 120 canonical (R=null) and 240
repeated (R=0/1) entries per invocation; candidates are separate scheduling
entries. Each condition receives 100 scalar timings per invocation. For n
eligible cases, repeated samples total 3600n and canonical samples total
1800n across three invocations. The validator derives counts from dimensions
and serialized orders and rejects deleted/duplicated entries or visits.
