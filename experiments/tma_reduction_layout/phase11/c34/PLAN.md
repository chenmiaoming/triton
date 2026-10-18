# Stage C3.4 — training-only resource calibration and overlap-family test

2026-10-18. CPU only. C2d, C3.3 and all earlier evidence are read-only. No GPU
allocation, native build, D1 compilation, timing, PTX inspection or NCU run.
The ten D1 shapes and the historical split remain fixed. This plan is frozen
before C3.4 fits, coverage measurements or policy scores are produced.

## Resource experiment (no timing input)

Keep the C3.3 physical-register point regression and its 74 distinct loop-free
FULL CUBINs: 61 development layouts on nine geometries plus 13 already compiled
resource-boundary layouts. Never use resident-loop/fixture registers as FULL
registers. Resource fitting receives only case metadata, CUBIN hashes, measured
n_regs, spill/shared attributes and driver occupancy capacities. Exclude the
entire outer held geometry, including every source/alias of that geometry.

Replace the all-scale fallback with progressive distance in log2(R). Start
with [R/2,2R]; if fewer than three geometries are available, add the nearest
remaining scale, including all ties, until three are present. Record the pool,
radius, geometries and whether expansion occurred. Placement-specific paired
residuals are used when at least three geometries exist in that same pool;
otherwise relax placement inside the pool before any further scale expansion.

The common residual is the cross-fitted **default-layout** register error;
the paired residual is candidate error minus its same-geometry default error.
Both intervals include zero. Preserve a common shift of candidate/default;
do not choose independent candidate/default extrema. Use a rectangular common
and paired-error envelope, explicitly an empirical superset of observed joint
errors. Enumerate all attainable q pairs through allocation boundaries inside
this continuous envelope, not only its corners. Static shared lower/upper
estimates are propagated too. q means occupancy capacity, not measured active
CTA count at every instant. Register upper bounds above 255 retain the spill
uncertainty veto; resource estimates are not a proof of no spill.

Jointly select the two added register margins from
{0,1,2,4,8,12,16,24,32,48,64}. For each outer training fold and target R/layout
family, choose the smallest total margin achieving 100% empirical coverage of
default/candidate n_regs upper bounds **and** observed joint q capacities in
the target's progressively selected training-geometries neighborhood. Calibration
observations are whole-geometry held-out predictions; their residual envelopes
are themselves built by LOGO on the remaining training geometries. Thus a
calibration observation cannot calibrate its own envelope. Ties prefer smaller
maximum individual margin, then smaller common margin. If no pair attains the
coverage target, freeze the maximum pair and report failure. No timings,
predicted gains, switches or target cases influence this choice.

Report nine-geometry outer n_regs-upper, marginal q-interval and joint q-pair
coverage, by geometry and as candidate counts (not independent samples).
Show C3.3, resource-only C3.4 (C3.3 time models/residual banks unchanged), and
full C3.4 separately. Outer R2/R4 and the three specified D1 cases receive
scenario decompositions regardless of outcome. Do not silently eliminate a
resource scenario or narrow performance-error bounds to release a winner.

## Independent eta(q) experiment

Keep the C3.3 FULL/pair relative loss, operation features, control and MC fits,
ridge grid {0,.001,.01,.1,1}, scalar eta grid {0,.25,.5,.75,1}, q point estimates
and whole-geometry holdouts. Compare scalar overlap with
eta(q)=clip(eta8 + b*log2(q/8),0,1), where eta8 uses the same grid and
b in {-.25,+.25}. Both signs are allowed; greater occupancy is not assumed to
imply greater overlap. b=0 is exactly the existing scalar family. The direct
FULL model remains a common competitor. Inner CV alone chooses the family,
eta and ridge; on numerical ties prefer the direct/scalar simpler family.
Also report separately restricted scalar and q-dependent nested-CV results,
their inner loss profiles and per-geometry outer errors. q-dependence is a
response-model hypothesis, not an independently identified physical eta.

Repeat Load-blind, Comm-blind, Resource/Envelope-blind and Fixed-R-blind with
the same inner search; Resource/Envelope-blind uses only the direct model.
Do not change C3.3's signed, structure/scale-conditioned performance upper-tail
rule, its support/extrapolation margins or its minimum distinct-geometry count.
Rebuild its banks from training-only cross-fits for each selected recipe.

## Evaluation, freeze and stop

Score all nine outer geometries and all 61 measured candidates; distinguish
full-data fits from outer predictions. Compare nominal/conservative with
upstream_default, fixed_vec1/2/4/8 and fanout_only. Primary units are FULL
128/256/384 MiB service slopes, ns/tile/SM; fixed-footprint raw time is a
different estimand. Keep adverse results and the limited advantage over strong
simple baselines. No tuning to pass known winners or to a D1 switch count.

Freeze source, input manifest, resource-only data/model/calibration trace,
inner/outer CV, policy/ablation results, five-case decompositions, theta and ten
static D1 predictions. Validator recomputes coverage, folds, selection rules,
errors, policy scoring, scenarios and all D1 predictions; checks prior SHA
manifests, input exclusion and CPU-only imports. Commit and push C3.4, verify
remote HEAD, then stop for review before D1.
