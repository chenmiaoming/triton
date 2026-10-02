# Phase 3 evidence freeze

Status: **READY_TO_FREEZE_WITH_DOCUMENTED_ARCHIVAL_LIMITS**. This freeze implements
all must-fix items in section 7 of the takeover audit of
`adc35416d4b75a08da1e5b9626e4bfc39fbb4e72`. It contains offline tooling fixes and
regenerated derived reports only. No GPU timing, compiler build, or Phase 4 was
performed. It authorizes no next stage.

## Current quantitative lineage

The canonical denominator is derived from
`results/phase2/saturation/corrected_pilot_runs.json`, SHA256
`75087a0ea79d359e4b1820be729295db3f0d7cf392cdfde2a888b63dbb453e9b`,
last changed at `a57bff355124dd3d80e6ff4116e4796d1eeb48de`.
For `M32_N64_w8`, take the sample median at B={16384,32768,65536}, fit
T(B)=a+bB by OLS for each candidate and invocation, convert b to ns/additional
CTA, then average three invocation slopes. Default is 3.9449405340703723,
cand4 is 2.4853515610413437, and their gap is 1.4595889730290286.
Each report records the source path, raw SHA, last-change commit, configuration,
candidates, B domain, estimator, and individual run slopes.

Step E primary Δg(1) is 1.092889685120705 ± 0.011770658120654217 ns/additional
CTA. Its descriptive magnitude ratio to the current canonical gap is
0.7487653752636082 (74.88%). The former 76.5% used an older canonical raw
generation; it is not the current denominator. This is a cross-harness
descriptive comparison, not an additive causal decomposition or single-CTA
latency estimate. ± is the sample SD of three same-device temporal invocations,
not a confidence interval or multi-device replication.

R² of cross-run mean points is 0.9996471407591544; mean of individual run R²
is 0.999642049199822. They have distinct fields. The observed increments vary;
the data support approximately linear amplification, not constant repetition
cost.

The declared **retrospective** H2a decision rule requires all primary invocation
betas > 0, mean Δg(1) > 0.1 ns/CTA, mean invocation-fit R² >= 0.90, and
nondecreasing primary g(R) for mean points and each invocation. Thresholds are
preserved from the existing rule. H2a currently evaluates to
`SUPPORTED_AT_REDUCTION_BODY_LEVEL`; H2b/H2c remain `UNVERIFIED`. Integrity
validation checks derivation and decision consistency, and also accepts a
consistent `NOT_SUPPORTED` result. The regression probes demonstrate this with
synthetic observations; those probes are not experimental measurements.

## Structural contract and limits

C checks every initial LocalLoad opcode/width, a unique complete normalized
reduction fingerprint match, layouts, actual resource text, and recorded SHA.
D/E inspect actual TTGIR/PTX/SASS rather than trusting derived PASS flags.
They enumerate every backward/self edge, distinguish the runtime reduction loop
from TMA polling, verify one pre-loop TMA and LocalLoad, unspecialized runtime R,
complete loop reduction/memory signatures, terminal reload placement, absent
accumulator/global effects, empty asm blocks, one-to-one PTX copies, and spills.
Every structural condition is required by the archived D gate.

“Exact sequence” compares the entire filtered normalized reduction fingerprint:
max/cvt/shfl/shared-load/shared-store/ldmatrix/bar.sync/selp opcodes and
shuffle/barrier immediates. Most operands and predicates are omitted. It does
not establish dataflow, full PTX, SASS, or CUBIN equality. Primary N64 matches
this complete sequence for both candidates. Secondary N128 default has only
opcode multiset equality (`PIPELINED_OPCODE_EQUIVALENT`), which does not prove
register-dependency or address topology.

PTX includes 8 input copies per N64 candidate and 32 per N128 candidate. No
explicit MOV or IMAD.MOV is observed in the actual runtime SASS reduction
region. Equal copy counts and this observation do not establish zero total
barrier cost or zero candidate differential: live ranges, allocation, scheduling,
and other indirect compiler effects remain unisolated.

Shape-aware blocked-layout ownership counts vector width once and allows
replicated ownership for small logical axes. N64 default/cand4 totals are 8/8
logical elements per thread; N128 totals are 32/32. The historical v1–v5
rejection/confound records are preserved.

## Archival limitations preserved without invented metadata

- Canonical occupancy API records came from a **resource-matched recompiled
  baseline**, not the exact measured canonical CUBIN. Its recorded SHA differs
  from the measured SHA. REG/static/dynamic shared-memory evidence matches;
  recorded theoretical blocks/SM are 8/8 for w8 and 16/16 for w4, with 64 active
  warps/SM. This supports a resource/residency control, not exact measured-binary
  API closure or achieved occupancy measurement. Original CUBIN binaries and
  occupancy-query disassembly were not archived, so that closure cannot be
  restored offline.
- D/E PTX, TTGIR, SASS, and resource texts match byte-for-byte in all four cases.
  Their CUBIN SHA records differ. No CUBIN binary is archived; text/resource
  equality must not be described as CUBIN byte identity. E SHA records match
  each of its three raw invocations; E contains one SHA per candidate, not an
  archived R-by-R hash map. D contains R=1/2/4/8 SHA records. Unspecialized source
  and runtime artifacts support E's single-binary protocol without inventing
  missing historical per-R observations.
- E retains manifest verification summaries (PASS, 1815 files, matching
  local/subset digest), but not each invocation's full local source
  manifest/provenance archive. Rotation records retain length=600, not the full
  rotation log. Existing summaries cannot reconstruct the missing uploads or
  rotation history. Raw files receive no fabricated historical metadata.

Before any separately authorized future experiment, archive the measured
binaries/resources, exact-binary occupancy evidence, complete per-invocation
source provenance and rotation logs, and finish the structural gate before
timing. This is a future evidence requirement, not a Phase 4 implementation or
execution in this freeze.

## Offline verification and protected evidence

`evidence_inputs.json` inventories 791 original evidence files at the audited
baseline with byte SHA256 values. Validator binds inventory completeness and hashes to the audited Git tree and checks existence and finite
JSON numbers in this inventory. Raw samples and original compiler artifacts
remain byte-identical. Derived C/D/E JSON and Markdown plus the structural
comparison/hypotheses are regenerated; original annotations and historical
results are preserved.

Run from the repository root:

```bash
python experiments/tma_reduction_layout/validate_evidence.py --self-test
python experiments/tma_reduction_layout/validate_evidence.py
```

The 24-probe offline regression suite covers corrupted std/mean/CV, NaN/Inf and boolean
numbers, missing R/residuals, stability labels, R² semantics, stale denominators,
inconsistent H2a status, nonfinite samples, extra loop shared loads,
MOV/IMAD.MOV, later LocalLoad width, E resource mismatch and missing artifacts,
inventory omissions/hash corruption, a consistent negative decision, and duplicated small-shape ownership. It
intercepts reads in memory and never edits original evidence.

These checks establish consistency of the archived evidence and its stated
contracts; they do not fill archival gaps or validate unmeasured component
latencies. Final committed-HEAD validator output is retained outside the evidence
archive and reported with the pushed remote HEAD.
