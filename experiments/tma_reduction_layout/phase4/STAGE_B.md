# Phase 4 Stage B: structural admission before timing

Frozen start: `e430b24b31d484440cc10ca7c79576dec88b13a0` on
`explore/tma-reduction-layout`. All 20 included transitions receive two candidates
and three independent harness bundles. The ten structural exclusions remain frozen.

`kernels_stage_b.py` generalizes the fixed-binary canonical code path and the
existing single/repeated Gluon representation using shape-aware blocked storage
extent. It introduces no hardware-dependent source scalars. Single/repeated shared
layout parameters come from actual canonical TTGIR, while blocked layouts come
from frozen expectations; independent audits check both against actual outputs.

Run `.venv/bin/modal run -m experiments.tma_reduction_layout.phase4.run_artifact_gate`.
The existing Modal skill guides the persistent `triton-build-cache` image build.
Only H100 CC9.0 is accepted. This entry point has no benchmark/timing calls and
does not execute the frozen schedule. Each repeated kernel is warmed up once with
unspecialized runtime R, then the same CompiledKernel is called directly for R0/R1.

Exports are immutable. Actual CUBIN bytes are written first, read back into a
buffer, loaded using checked CUDA driver calls, and queried with
`cuOccupancyMaxActiveBlocksPerMultiprocessor` at actual and zero dynamic SMEM.
Resource attributes are checked against cuobjdump. The actual bundled ptxas and
native Triton library are hashed separately from PATH tools. Source bytes and
the uploaded subset are archived with frozen protocol/pool/schedule hashes.

Resource interpretation keeps the raw cuobjdump `SHARED` field separate from
`CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES`. This build reports 1024 and 0 respectively;
no equivalence is assumed. Both are retained against the same CUBIN SHA, and
residency always comes from the actual driver occupancy call. Raw metadata and
frozen helper outputs retain their legacy parser field names; the top-level case
resource records use driver static SMEM and explicitly name the cuobjdump field.
`derivation_provenance.json` binds the current offline auditors separately from
the uploaded compilation source snapshot. All 20 cases are re-audited after an
offline interpretation fix; raw exports and the kernel source remain unchanged.

Stage A's unchanged validator allows additions under `phase4/` but disallows the
suggested sibling artifact directory. Consequently outputs use
`phase4/results/artifact_gate/{canonical,single,repeated}`. The nested results tree
is excluded from the build upload and source manifest. Stage A's five master files
and all SHA-bound Stage A code/docs remain unchanged.

Derive with `python3 experiments/tma_reduction_layout/phase4/stage_b_audit.py`;
validate with `python3 experiments/tma_reduction_layout/phase4/validate_artifact_gate.py`.
The latter reopens bytes, checks all source/binary/function bindings and recomputes
all cases, classifications and stable schedule filtering. Thirteen mutations run
against in-memory copies of actual exports; original compiler artifacts are untouched.

The frozen complete fingerprint families/immediates, exact single LocalLoad rule,
repeated pre-loop encoding exception, spill and copy gates are unchanged. The
frozen no-explicit-loop-SASS-MOV gate and matched candidate residency for all three
harnesses are retained. No requirement is added for equal register counts. True
structural failures exclude; compiler/export failures retain full errors as PENDING.
Reason semantics are declared in `stage_b_audit.REASONS` before the GPU batch.

Canonical/single smoke uses four deterministic initialized prefix tiles of a
B_DESC=65536 BF16 allocation and checks float32 M-axis maxima. Repeated's trivial
zero output checks launch safety only. Neither the compiler barriers nor R0
subtraction prove absence of register allocation, live-range, scheduling or
compiler-decision interactions.

Future Stage C must compare the actual launched canonical/repeated CUBIN SHA to
the corresponding eligible Stage B archive SHA, aborting before timing on a
mismatch. The preview retains master relative order and is never executed here.
No latency, throughput, speedup, slope, association or hypothesis-status update
is produced. One local commit is allowed; no push and no Stage C.
