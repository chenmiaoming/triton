# Stage M1 — Identifiability and mechanism audit

Authorized 2026-10-20 after D1. CPU only. This is post-hoc analysis of existing
development and exposed D1 evidence. Preserve every previous SHA, model role,
prediction, raw sample, compiler artifact and NCU report. No C++ changes, native
builds, GPU allocations or new timing. Write the method and prior hash manifest
before producing audit results. No replacement D1 scorecard or selector.

1. Reconstruct the 27-column FULL design using only the nine development
geometries. Inspect singular values, exact/near dependencies and active-column
rank. Distinguish numerical rank deficiency from uniqueness under nonnegativity
and ridge constraints. Do not interpret weighted NNLS residuals as raw ns.

2. Reuse frozen C34 scalar inner-CV loss profiles across eta/ridge, including
each outer training fold. Fit eta={0,.25,.5,.75,1} with fixed ridge=.1 on all
development data, then also with each eta's best pre-existing inner-CV ridge.
Retain coefficients, normalized contribution changes and prediction changes.
No D1 data enter fitting or parameter choice. This is a CV-error profile, not
a profile-likelihood confidence interval. Any collinearity/eta compensation
conclusion must be supported by the observed curves, not assumed in advance.

3. Evaluate frozen scalar resource-only and full:auto response models on C2d
development FULL curves and D1 FULL curves. Development uses geometry-held-out
models; D1 uses the originally frozen final models. Compute static predictions,
observed-q substitution, same-q observed T_control substitution, and both.
Matching requires the same geometry, cohort, worker and GPU UUID. Missing
measured q is unavailable, never interpolated. Show common-support metrics and
all available metrics separately, signed paired residuals and all four-way
interactions. Substitution is a computational diagnostic, not causal proof.
T_control is the operational row-output kernel, not pure transfer latency.

4. Enumerate all same-vec N-first/within layout pairs among 61 development and
59 D1 layouts. Compare static ML/local/MR/MO plans, physical resources, per-PC
dynamic SASS and hardware counters. Static subgraph counts must never be
equated to whole-kernel counts. Coarse matching does not guarantee matching
instruction dependencies or output conversions. For 32x64/w8 vec2 and 32x128/w4
vec2, inspect retained CUBIN debug-line disassembly when possible; retain raw
CPU tool output in the new directory and report unavailable attribution.

5. Plot actual and frozen-model FULL/control response against measured q for
32x64/w8, 32x128/w4 and 16x256/w8. Keep aliases visible but do not treat them as
new independent conditions; geometry is the validation unit, worker-paired
differences are formed before medians. For each frozen natural decision,
separate nominal gain, resource scenarios, performance upper tail and support
vetoes. A correct abstention is not a correct point prediction.

Deliver JSON/CSV diagnostics, standalone PNG/PDF scientific plots, REPORT.md,
CPU-only reproduction commands and a final validator/freeze. Recompute derived
metrics, verify the frozen models' natural predictions and all earlier hashes,
then commit/push and stop. If available evidence cannot isolate a mechanism,
state the missing matched FULL intervention; do not launch it automatically.
