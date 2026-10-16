# Stage C3.1 — direct MC model and paired steady-state decisions

Authorized 2026-10-17. CPU-only. No D1 compilation/timing, GPU allocation,
native rebuild or C++ changes. Original C3 remains write-once at commit
6a850c0060611dd598466a086c2cf6ec926e33d7 and is referenced as
`c3_v0_overconservative`; new evidence lives under stage_c31/c3_v1.

Fit MC D1 steady ns/tile/SM directly, not sums/ratios of fixture timings.
Reuse the frozen static G_L generator. Features are nonnegative static
load, local-tree, shuffle, shared-reduction/barrier work and shared primitive
shape/residency transforms. Fit nonnegative coefficients; occupancy pressure
is nonincreasing in q. No per-layout/geometry identity coefficient. Primitive
costs are regressors with fitted weights, not an additive prediction target.

Calibration: 15 layouts, q1/q2/q4/natural, three workers. Fit cell medians
(60 cells); keep all 180 worker estimates for same-cohort/worker/q contrasts.
Evaluate leave-one-layout-out (all q/workers withheld) and leave-one-geometry-
out. Predict candidate and default with the same fold. Retain signed default
and paired-difference residuals jointly, grouped by occupancy regime, never
subtract two independent marginal extrema. No launch-as-independent-GPU CI.

Develop a small predeclared family of nonnegative linear/hinge models with
optional primitive-derived shapes. Choose hyperparameters by held-geometry
paired-difference error, not by obtaining a requested selection. Retain all
attempts, coefficients, errors and ablations. The two requested development
decisions are acceptance diagnostics, not hardcoded labels.

Use independent full-grid TMA controls in ns/tile/SM; distinguish C2a and
C1.6 cohorts and propagate a common transfer scenario for each comparison.
Version 1 uses a steady bottleneck envelope max(s_TMA,S_MC), i.e. s_base=0
as an explicit modelling hypothesis, rather than introducing an unidentified
one-shot offset. It predicts steady bottlenecks, not exact total latency.
Report old complete-kernel results as evaluation only. K1/MO/grid1 do not veto.

Switch only when the paired roofline improvement remains >3% over static
resource/q and paired residual scenarios. Preserve necessary shared/register
legality gates. Report extrapolation, covariance and support limitations.
Regime-stratified additive residuals are empirical error envelopes, not a
guaranteed non-regression interval or independently validated confidence band.

Compare default/fixed-vector/fanout baselines and refit Load-blind, Comm-blind,
Resource/Envelope-blind variants with the same protocol. Report selection,
service/ranking/error differences even if a decision does not flip. Do not
force every ablation to select a different layout on only two geometries.

Freeze the same eight D1 geometries, model/parameters and candidate predictions
in a separate C3.1 commit; stop before any D1 execution. No PR.
