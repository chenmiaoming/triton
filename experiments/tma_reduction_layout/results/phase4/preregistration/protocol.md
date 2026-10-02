# Phase 4 — Cross-Shape Generalization: Stage A preregistration

Status: **STRUCTURAL_COHORT_FROZEN / ALL_NEW_TIMING_ELIGIBILITY_PENDING**.
Parent evidence commit: `ce04afd7d9095349d54a007e61ec2b5d4fd557b6`.
No new compiler execution, H100/Modal work or performance measurement is part
of Stage A. Stage B requires a separate authorization. Do not push this freeze
until the user reviews it.

## Population and protection against selection bias

This is performance-field-independent programmatic cohort selection, frozen
before any new Phase 4 mechanism measurements. Historical canonical outcomes
existed before this preregistration, so human-level complete outcome blinding
cannot be claimed. Poison tests establish implementation-level independence
of membership from performance fields.

The source is the entire archived Phase 2 sweep: M={32,64,128},
N={16,32,64,128,256}, warps={4,8}, five candidates per configuration. The
research unit is the transition `(M,N,num_warps,default→cand4)`, not shape alone.
No timing, slope, speedup, winner, regret, practical-significance label or
historical representative-selection rule enters membership.

The initial structural rule requires:

1. Both candidates legal.
2. Distinct supported rank-3 blocked layouts on `[1,M,N]`, with positive
   power-of-two factors, permutation order, 32 lanes/warp, one CTA and the
   declared warp count, supported rank-3 NVMMA shared representation.
3. Same logical max reduction along M (axis 1) and same `num_warps`.
4. `warpPart[M]` unchanged and `lanePart[M]` strictly decreased.

All source transitions are retained exactly once, with boolean membership and
machine-readable, nonexclusive reason codes. Missing logical representations
cannot silently pass. The entire source Cartesian domain is checked against the
source AST constants. Legality is independently checked against the existing
forced-vector rules; no new legal layouts are fabricated for illegal cand4.

The sweep's historical parsed `reduce_ops` lists are empty. Source AST verifies
its `[1,M,N]` tile and `tl.max(axis=1)` operation for logical selection. This is
not per-binary TTGIR closure: Stage B must verify actual reduction axis, shape,
warps and layouts from each archived artifact.

Resources, spills, residency and artifact availability form a separate layer.
They never remove cases from the structural population. Record every later
pending/failure reason and tier, including negative, near-zero and other
counterexample cases. Cases lacking artifacts stay pending for Stage B; do not
replace them with cases selected by speedup.

The two existing anchors, M32_N64_w8 and M32_N128_w4, remain in the structural
pool, labeled `ANCHOR_EXISTING`. Analyze them with other primary-eligible cases
if their new gates pass, and report a predeclared new-cases-only sensitivity.
Historical single-reduction reproduction is exact for both anchors; historical
N128 default repeated lowering is only opcode-multiset equivalent. Those
reference classifications do not grant fresh Phase 4 timing eligibility.
Both historical cand4 repeated harnesses load `ld.shared.v4.b16`, whereas
their canonical/single references load `ld.shared.v2.b32`. The entire opcode
lists are archived in the inventory. Record this as `DIFFERENT_ENCODING`,
not an automatic timing exclusion. The M32_N64_w8 anchor passes the generic
body/isolation contract for both candidates; no config-name exception is used.
It remains pending exact Phase 4 binary/resource/runtime provenance closure.

## Frozen structural and artifact gates

Use native Gluon TMA, matching NVMMA shared and blocked layouts, one explicit
shared-to-register load and float32 max(axis=1). Preserve Triton/Gluon's logical
block model; introduce no lane-varying scalar hardware IDs or source-level
backdoors. Do not implement source-level v6/v7 or a production heuristic.

**CANONICAL_SINGLE_REPRODUCTION_GATE** — for each case/candidate, compare:

- Exact blocked attributes: sizePerThread, threadsPerWarp, warpsPerCTA, order.
- Exact NVMMA shared attributes, logical shape, axis and warp/CTA geometry.
- Every canonical and single-reproduction initial LocalLoad opcode and width, in order.
  Store both full lists; first-opcode or count-only matching is insufficient.
- The entire filtered normalized canonical reduction-stage fingerprint and
  entire Gluon reduction fingerprint. Include all max.*, cvt.*, shfl.*,
  st.shared*, ld.shared*, ldmatrix*, bar.sync and selp.*. Retain shuffle and
  barrier immediates. Normalize register IDs, other addresses and predicates;
  this does not establish full PTX, dataflow, SASS or CUBIN identity.
- Actual resource text, LOCAL=0 and STACK=0. Static/dynamic shared memory and
  registers are recorded even though they do not affect structural membership.

Stage boundaries must be established structurally before timing. Canonical
convert/max bounds include terminal reduction exchanges and exclude the
output-store layout conversion. Future compile metadata binds the source file,
SHA and AST lines for convert/max; PTX source locations establish the whole
stage. An ambiguous/missing stage is pending, not a reason to search for a
matching subsequence. Minimal-output Gluon uses all instructions after initial
tile loads and before global store, retaining terminal reloads even when debug
locations attribute them to output. The repeated harness uses its entire
runtime loop. Stage A's archival canonical extraction is checked against the
frozen Phase 3 annotations; it is not a general substitute for Stage B source
and stage evidence.

Classification is frozen:

- **PRIMARY**: canonical single reproduction passes exact blocked/NVMMA
  layouts and complete initial LocalLoad sequences, with both candidates
  `EXACT_SEQUENCE_EQUIVALENT` in single and entire runtime-loop fingerprints.
  All repeated isolation, archive, runtime, spill/residency gates pass. Repeated
  pre-loop LocalLoad encoding need not equal the canonical single encoding.
- **SECONDARY**: all gates pass, no fingerprint mismatch, and at least one
  comparison is `PIPELINED_OPCODE_EQUIVALENT`. This compares the whole opcode
  multiset and does not prove dependency or address topology. Report separately.
- **EXCLUDE_FROM_TIMING**: a real gate failure: complete fingerprint mismatch,
  canonical-single LocalLoad mismatch, repeated LocalLoad semantic failure,
  changed binary/LocalLoad across R, missing required R=0, layout mismatch,
  spills, residency mismatch, invalid loop/barrier contract or other actual
  structural failure. Retain the record and reasons in cohort accounting.
- **PENDING**: insufficient archives/evidence, unresolved source-stage bounds or
  inability to audit a valid representation. Do not count pending as a measured
  failure, and never promote it to timing-ready.

`TOPOLOGY_EQUIVALENT_NONEXACT` is not assigned by the automatic auditor. A future
separate dependency/address proof and pre-observation amendment would be needed
before treating such a class as eligible; multiset equality is insufficient.

## Exact-binary residency and archival requirements

For both default/cand4 in each canonical and controlled harness, primary timing
eligibility requires equal theoretical blocks/SM and active warps/SM. This is
not achieved occupancy. Query the exact compiled timing binary using the
official occupancy API and archive the actual binary, resource text and query
result before timing. Do not reuse Phase 3's resource-matched recompile as an
exact measured-binary query.

A future case manifest has `logical_shape`, `num_warps`,
`repeated_source_path`, `repeated_source_sha256`, and three candidate mappings:
`canonical`, `gluon_reproduction`, `gluon_repeated`, each with `default` and `4`.
Each mapping contains `paths` and `SHA256` dictionaries for all of:

```
ttgir, ptx, sass, resource.txt, cubin, cubin.sha256,
occupancy.json, source_manifest.json, build_provenance.json,
compile_metadata.json, environment.json
```

The `.cubin` must be the actual compiler-emitted ELF binary, not a hash file.
All export hashes must agree with the compile/archive manifest. Occupancy records
contain `query_api=cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
`queried_cubin_sha256`, `resource_sha256`, `num_warps`, `dynamic_shared_bytes`,
`blocks_per_sm`, `active_warps_per_sm`, and resource fields matching actual text.
The query also records `block_threads=32*num_warps`, `gpu_uuid`, and
`build_identity`; `resources` uses `num_regs`, `local_bytes`, `stack_bytes`,
`static_smem_bytes` in both occupancy and compile metadata. GPU/build identities
must agree with the environment and build provenance.
Compile metadata contains `logical_shape`, `num_warps`, `dynamic_shared_bytes`,
resource fields and `reduction_stage` with `file_id`, `source_lines`,
`source_path`, `source_sha256`, `kernel_function`, and `ptx_source_path`.
The full convert/max source AST lines must agree with `source_lines`, and the
PTX `.file` entry must agree with the archived source-path mapping.

Bind those stage lines to the actual archived source AST and compilation
exports. Archive the source bundle alongside its complete source manifest;
manifest file paths and SHAs must refer to those exact source bytes.
Each source manifest entry records `path`, `SHA256` and the originating
`ptx_source_path` for a stage source. Paths refer to the archived source bundle,
not mutable external checkout files.
`build_provenance` records Modal image/build identity, source-manifest SHA,
compiler/toolchain version and exact export hashes.
The provenance keys are `modal_image`, `build_identity`, `toolchain`,
`source_manifest_sha256` and `export_SHA256`; the latter binds TTGIR, PTX,
SASS, resource text, CUBIN, CUBIN SHA file and compile metadata.
`environment` records GPU
UUID, driver, CUDA and toolchain. Logged occupancy identity must be authentic
to that invocation; matching a regenerated resource proxy is insufficient.
The offline gate validates the recorded bindings and does not itself call CUDA
or establish authenticity from hashes alone.

For every invocation retain full local provenance, uploaded/source manifest,
image/build provenance, pre/post telemetry and actual executed order. Historical
missing CUBINs, per-R SHA maps, complete provenance and rotation history remain
missing; Stage A does not manufacture them.

## REPEATED_HARNESS_ISOLATION_GATE

Use the term **canonical reduction-body-equivalent controlled harness**.
Do not describe it as an exact canonical full kernel.

Verify one TMA and one initial tile LocalLoad before the single runtime
reduction loop. R is an i32 runtime argument with `do_not_specialize`, and the
same per-harness binary is used across all R and B_RUN. No initial tile reload,
global store/atomic, accumulator artifact or spill is permitted in the loop.
Canonical reduction exchange loads and terminal reloads must remain inside it;
complete loop memory-op signatures prevent hiding extra operations in a
matching window. Verify matching theoretical candidate residency for that exact
binary. Enumerate all PTX/SASS backward/self edges; distinguish reduction from
TMA polling and compiler terminal edges.

Record `canonical_local_load_match` as `EXACT`, `DIFFERENT_ENCODING`, or
`MISMATCH`, alongside both complete opcode lists. A different packing/width
encoding with exact TTGIR tile/layout/shared semantics and matching complete
per-thread byte extent is `DIFFERENT_ENCODING`. Byte extent alone does not
prove PTX address/dataflow identity. It does not force EXCLUDE if R=0 exists,
the load is one-time/outside-loop, the binary is fixed across R, the runtime
body passes its gate, and residency is matched. Actual wrong/missing payload,
a LocalLoad moved inside the loop, changed load/binary across R, or unexpected
extra loop memory operation still fails.

Each future repeated candidate bundle additionally supplies
`runtime_conditions`: exactly the six `(R,B_RUN)` combinations of R={0,1} and
B_RUN={16384,32768,65536}. Each record has `R`, `B_RUN`, `cubin_sha256`,
`ptx_sha256`, and `pre_loop_localload_opcodes`. They must bind to the same
archived CUBIN/PTX and pre-loop sequence. This is a pre-timing execution plan;
future invocation logs must verify actual execution against it.

PTX tied copies may exist. Archive count, full opcode/operand sequence,
one-to-one mapping, candidate symmetry and explicit MOV/IMAD.MOV observations.
Require one `mov.b16` input copy per physical per-thread BF16 tile element
before the empty input asm, unique source/destination registers, and no copy
after the sink. Record all loop MOVs separately: BF16 packing/unpacking within
the reduction is not an input-barrier copy.
Empty input/sink asm and absent explicit loop MOV are observational pre-gates.
Copy symmetry does not prove zero total or differential barrier cost: allocation,
live ranges, scheduler and other indirect compiler effects remain unisolated.

## Future sampling protocol — not executed in Stage A

- B_DESC=65536, B_RUN={16384,32768,65536}, repeated R={0,1}.
- One binary per case/candidate/harness across runtime R and B_RUN; canonical
  and Gluon have separate binaries. Archive all actual binary hashes per condition.
- 10 rounds × 10 samples = 100 positive finite scalar samples per measurement
  condition **per invocation**, not 100 across all three invocations.
- Three benchmark invocations; same UUID means same-device temporal replication,
  not independent hardware replication.
- Save the complete master and executed deterministic rotation schedules,
  sample order, all raw samples and telemetry. Do not save only schedule length.
- Retain the original pre-artifact-gating master schedule byte-for-byte in
  `rotation_schedule.json`: it covers the 20 structurally included transitions
  and 360 combined measurement conditions, not all 30 source transitions
  (which would be 540 combined conditions). All 30 source transitions remain
  in cohort accounting.
  Obtain the actual timing schedule only by filtering ineligible transitions
  through artifact/resource gates while preserving relative order. Never
  regenerate a favorable order after Stage B gates are known. Record every
  pending/excluded condition and never retry/drop
  individual cases because of observed effect or model fit.
- No adaptive case choice, outcome-dependent stopping or threshold tuning.
  Protocol failure invalidates the whole affected invocation with a recorded
  reason; repair/repeat needs explicit authorization and complete disclosure.

### Schedule schema and cardinality

A **measurement condition** is exactly
`(case, harness, candidate, R, B_RUN)`. Each `master_conditions` dictionary
entry has fields `config_id`, `harness`, `candidate`, `R`, `B_RUN` and an
identifier encoding those fields. It contains one candidate, not a candidate
pair. The repeated harness has R={0,1}; the separate canonical harness has
R=null because it has no runtime repetition parameter. Canonical measurements
are required to estimate G_canonical; they are additional to repeated-harness
measurements used for delta_g1.

A **candidate-paired condition** would group default/cand4 at the same
case/harness/R/B/invocation. This schedule does not store or execute such a
paired scheduling unit: the two candidate conditions are separate entries.
Future analysis pairs their invocation-level differentials; that pairing does
not change the scheduling schema or the count of scalar measurements.

A **round order** is an ordered visit to every retained measurement condition
once. A visit requests 10 timed scalar **samples**. An **invocation** is one
temporal benchmark run containing 10 such round orders. An invocation-level
condition is `(invocation, measurement condition)`, counted once even though
it is visited in each of the ten rounds. Samples and round visits are distinct
from unique measurement conditions.

Inspection of the serialized frozen master gives 20 structural cases, two
candidates and three B_RUN values. It contains **240 repeated conditions**
`20*2*2*3` plus **120 canonical conditions** `20*2*1*3`, hence 360 total
measurement conditions per invocation. This is neither 360 paired units nor
a 30-source-case repeated-only schedule. The validator reconstructs both
Cartesian domains, identities and cardinalities from dimensions and checks
all entries, invocation orders, rounds, visits and sample counts.

For final eligible cohort size n, retain the original order and filter only
ineligible transitions; never regenerate a new favorable order:

| Future count | Repeated R={0,1} | Canonical R=null | Combined |
| --- | ---: | ---: | ---: |
| Measurement conditions per invocation | 12n | 6n | 18n |
| Invocation-level conditions across 3 invocations | 36n | 18n | 54n |
| Round-order visits across 3 invocations × 10 rounds | 360n | 180n | 540n |
| Scalar timing samples (10 samples per visit) | 3600n | 1800n | 5400n |
| Conditions per invocation if n=20 | 240 | 120 | 360 |
| Invocation-level conditions if n=20 | 720 | 360 | 1080 |
| Scalar timing samples if n=20 | 72,000 | 36,000 | 108,000 |

All of these are future plan counts, not collected measurements. The intended
Phase 3-compatible repeated-harness accounting is exactly 3600n scalar samples;
the additional 1800n canonical samples were already present in the frozen
master. `protocol.json` records mechanically derived per-harness cardinalities.
The original master domain and every round order remain byte-for-byte frozen.

R=0 subtraction removes fixed one-time harness differentials, but does not
prove absence of interaction through register allocation, live ranges, or
scheduling. It is a controlled measured-baseline subtraction, not proof that
all load/store/overlap effects are independent of R.

## Metrics and association

Per invocation and condition, take the sample median, then OLS fit
T(B)=a+bB on the three B points. Express b as ns/additional CTA.
Preserve intercept, R² and residuals as quality diagnostics, never as an
outcome-dependent case-removal rule. Poor fits can limit slope interpretation
and must be disclosed without changing the frozen population.

For each paired invocation:

```
G_canonical = b_default_canonical - b_cand4_canonical
g(R) = b_default_repeated(R) - b_cand4_repeated(R)
Δg1 = g(1) - g(0)
```

Average the three paired invocation differentials; report sample SD. These are
marginal grid-time slopes, not single-CTA latency. The association is not an
additive causal decomposition or a cost-model/production heuristic.

Primary association: Spearman rank correlation between mean Δg1 and mean
G_canonical across all future PRIMARY cases. Spearman is Pearson correlation
of one-based ranks of the continuous unrounded means, with average ranks for
exact ties. Do not discretize into POSITIVE / SIGN_UNRESOLVED / NEGATIVE before
ranking, round values, or replace unresolved numerical values with zero.
Secondary: Pearson on the same unrounded means. Report SECONDARY cases as a
separate stratum and PRIMARY new-generalization-only sensitivity. Keep anchors
in the primary analysis if gates pass and label them separately. Fewer than
three eligible pairs or a constant variable yields UNDEFINED, with no case or
threshold changes. The cohort is a structured factorial design, not an IID
population sample. Do not use correlation p-values as evidence of population-
level random-sample inference. Pearson remains secondary descriptive analysis.

For the PRIMARY cohort freeze leave-one-case-out Spearman sensitivity:
report every case i and `rho_minus_i`, then min/max/median of defined values.
Use the same continuous-value average ranks, recomputed over each remaining
cohort. Retain undefined omissions and report defined/undefined counts; an
omission with fewer than three pairs or a constant variable is UNDEFINED.
This is sensitivity only; do not delete outliers or replace the full-cohort
analysis with a selected omission.

If PRIMARY n<5, report per-case results and association metrics (including
UNDEFINED where applicable), but do not claim cross-shape generalization is
established. For n>=5, association remains descriptive evidence over this
defined cohort, not population inference. Do not change inclusion by size or
invent a required rho threshold.

## Student-t sign-resolution band frozen before new mechanism observations

A relative practical band rescales with a reference slope and needs an explicitly
chosen denominator. Phase 2's precedent is `phase2_benchmark.py:1134–1145`
(`run_sweep_remote`): `abs((candidate median-default median)/default median*100)
>3.0` triggers an extra within-run repeat. It is not a universal significance
criterion and not calibrated for marginal slopes or Δg1. Reusing 3% as the
primary near-zero threshold would add an unverified interpretation.

A fixed absolute ns/CTA band keeps the two signed differential metrics in the
same units but has no prior noise-floor calibration. Recommend instead an
**uncertainty-scaled absolute band**, separately for G_canonical and Δg1:

```
h = t(0.975, df=2) * SD(three paired invocation differentials) / sqrt(3)
t(0.975,df=2) = 4.302652729911275
POSITIVE: mean > h
NEGATIVE: mean < -h
SIGN_UNRESOLVED: -h <= mean <= h
```

This is a two-sided 95% Student-t sign-resolution band with df=2 for three
paired temporal values; sample SD uses ddof=1. The machine label is
`SIGN_UNRESOLVED`; `NEAR_ZERO` may be retained only as a display alias.
SIGN_UNRESOLVED does not mean practical equivalence. It only means that sign
is unresolved at the chosen temporal-replication precision. Its nominal
95% coverage relies on assumptions not established by sharing a GPU; report
those limits and quantization/drift explicitly. Do not zero out numeric values
or exclude uncertain cases from
correlations. No Phase 4 observations were used to set this rule. Sample SD
must be finite/nonnegative, invocation count exactly three; invalid inputs are
protocol failures, not case-selection opportunities.

Report three-category sign agreement over all eligible pairs. Additionally
report resolved-sign-only agreement with numerator, denominator and unresolved
coverage; never substitute it for the all-pairs accounting.

## Counterexamples and outliers

The two axes below are **Δg1 category / G_canonical category**:

| Pair | Frozen category |
| --- | --- |
| POSITIVE / POSITIVE | A: mechanism-aligned positive |
| SIGN_UNRESOLVED / SIGN_UNRESOLVED | B: mechanism-neutral / unresolved |
| POSITIVE / SIGN_UNRESOLVED | C: masked-local-advantage candidate |
| POSITIVE / NEGATIVE | D: opposed canonical outcome |
| SIGN_UNRESOLVED / POSITIVE | E: other-mechanism candidate |
| NEGATIVE / NEGATIVE | mechanism-aligned negative |
| NEGATIVE / POSITIVE | opposed canonical positive |
| NEGATIVE / SIGN_UNRESOLVED | masked-local-disadvantage candidate |
| SIGN_UNRESOLVED / NEGATIVE | canonical negative with unresolved body |

C/D/E and all other categories remain in the analysis. Labels suggest
explanations to investigate, not established causal diagnoses.

Freeze outlier score as
`abs(average_rank(Δg1)-average_rank(G_canonical))/(n-1)`.
Rank every case and discuss the top min(5,n), descending score and config_id
ascending to break ties. Label anchors. Do not remove outliers, redefine the
criterion, or refit after deletion. A one-case group is not ranked for this score.

## Coverage and stopping

The defined population covers a lane-partition-reduction class with unchanged
warpPart[M]. All N256 exclusions are consequences of those structural rules,
not performance filtering. Do not generalize results to all Triton partial-
reduction layouts or claim H2b (lane dominates warp) or H2c is verified.

Stage A ends after the preregistration freeze, validator runs and report/diff
review. It grants no authorization for push, Stage B, Modal/H100 compilation or
R=0/1 timing.
