# C3 — static interval selector and D1 freeze

Authorized 2026-10-17. CPU analysis only; no CUDA launch, GPU timing, NCU,
native rebuild, C++ edit, or D1 execution. Inputs are immutable C2a/C2b
calibration and compiler source at 20d9d98. The candidate predictor accepts
only geometry, dtype, axis, register/shared layouts and fixed target constants.

1. Enumerate blocked/128B-swizzled addresses, vector transactions and bank
   word multiplicities. Validate against all 15 development layouts. Derive
   reduction topology and compare PTX and dynamic counters independently.
2. Fit shared nonnegative effective-service coefficients, retaining probe
   fixture work. Use dependency/resource scheduling envelopes, q uncertainty,
   and global dimensionless composition bounds; never a per-layout correction.
   Report leave-one-structure-out errors and interval coverage, not a claim
   that calibration is independent validation.
3. Keep steady body, one-pass service, and single-CTA effective latency
   distinct. MO is a diagnostic upper constraint, not an additive repeated
   component. MC residuals remain diagnostics. No historical empirical offset.
4. Compare default/fixed-vector/fanout baselines and three ablations on the
   development set. Complete-kernel observations exist only for N-first;
   do not present the MC ranking as a complete-kernel oracle for all 15.
5. Audit all available Phase 1–11 files (including archives and source pools)
   for geometry exposure. Freeze eight eligible unseen combinations before
   running predictions. Freeze source, shared parameters, predictions, inputs
   and protocol in a C3-only commit. Stop before D1 GPU timing.

The practical threshold is 3%. A switch requires interval dominance over
default in both steady and one-pass service and no single-CTA/q reversal.
An unsupported topology/resource extrapolation is explicit abstention. All
abstention is permitted as an honest outcome, but is not evidence of useful
selection; report switch coverage and remaining identification gaps.

`e_vector=vec`, `elements_per_thread`, and the historical E-probe size are
separate quantities. NCU aggregate conflicts may include contention; static
bank conflicts are address-induced predictions, not all q-counter events.
