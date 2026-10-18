# Stage C3.3 — FULL-target model and stratified D1 freeze

Pre-measurement methodology, 2026-10-18. CPU-only model work. Leave C3/C3.1/
C3.2 and all prior evidence unchanged. Use C2d's nine complete geometries
(61 FULL candidates), C2c's matched fixed-R observations and C2a/b/c MC and
primitive data only as calibration. No D1 GPU compilation, timing or inspection.

Primary targets are loop-free FULL service (ns/tile/SM) and same-geometry,
same-worker candidate-minus-default FULL differences, including natural and
controlled q. Fit geometry-balanced dimensionless errors; publish both relative
and absolute errors. Fit time-versus-grid only to normalize experiments; the
selector receives BM,BN,W,dtype,reduce_axis,register_layout,shared_layout only.

Model structural input is the verified static G_L: ML wavefronts and conflicts,
local packed/scalar max and widening counts, warp shuffles, MR shared traffic,
barriers and output redistribution. Resource/q estimates use independently
calibrated physical FULL registers and static shared bytes, not MC fixture
register counts. Include q responses and within-vs-N-first contrasts. Historical
26.6 us and global-grid intercepts are prohibited model constants.

Compare a small preregistered model family: FULL structural service regression;
partial overlap T=H+max(0,A_L-eta*H), eta in {0,.25,.5,.75,1}; and old zero-base
max(control,MC) as a diagnostic baseline. H is a static-feature fitted transfer
control response; A_L is calibrated primarily on FULL, not forced equal to MC.
Nonnegative structural coefficients and ridge candidates {0,.001,.01,.1,1}
control complexity. An MC response prediction and fixed-R matched observations
may be structural covariates, with weight learned from FULL; their absence is
an explicit ablation. Inner whole-geometry CV selects recipe/hyperparameters;
outer leave-one-geometry-out excludes the held geometry from FULL, controls,
resource calibration and MC fit. Use geometry-balanced relative FULL and paired
errors jointly. All candidate truths are available for scoring. Report failure
if model support is inadequate; do not tune to a desired D1 selection count.

Evaluate independent upstream_default, fixed vec1/2/4/8, fanout-only, C3.2
zero-base, full nominal and conservative selectors. Ablations remove loading,
communication, resource/envelope and fixed-R terms. Compare observed oracle
gains with predicted FULL gains, explicitly checking old false 0% cases
64x64/w8 and 8x64/w8. No MC tie-break is called a positive FULL benefit.

Use signed out-of-geometry paired residuals normalized by the measured default
FULL scale. Condition upper-tail uncertainty by candidate/default layout family,
overlap branch (both platform, crossing, both on-chip) and q-change; inspect
R/scale support. Sparse strata back off to pooled strata with distinct-geometry
counts stated. No reflection of negative residuals. Add a positive-tail,
cross-validated scale margin and bounded leverage/extrapolation factor; do not
use an in-sample maximum as an unbuffered hard bound. For each outer-fold
decision build its uncertainty from cross-validation on the training geometries
only. Report coverage, adverse selections, abstentions and out-of-support
cases. Resource q intervals propagate as correlated candidate/default scenarios,
not independent extremes of a shared fixed base. A candidate switches only if
its worst supported predicted FULL gain exceeds 3%; otherwise keep default.

Freeze model code, parameters, uncertainty calibration, history audit and all
ten D1 prediction records in a separate C3.3 commit. The D1 list has five core
and five boundary shapes, all strict-zero-mention at the original historical
cutoff; 16x64/w8 is now calibration and excluded from all blind denominators.
Run validators, commit and push both stages. Stop before any D1 GPU work.
