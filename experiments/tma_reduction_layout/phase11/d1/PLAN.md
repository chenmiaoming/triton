# Stage D1 — pre-bound ten-geometry H100 held-out validation

Authorized 2026-10-19. Freeze this plan, protocol.json, all policy choices and
source/previous-evidence hashes before any D1 compile, CUBIN inspection or GPU
launch. The primary selector is C34_resource_only, selected using development
outer CV only. Its nine-geometry outer scores are consequently development
selection evidence, not an independent confirmation of this model choice.
D1 is the untouched confirmatory set. C34 full:auto is the pre-registered
automated-inner-CV comparator; C33 is historical. Do not change their roles or
predictions after D1 outcomes. No low-q refit is needed for this stage.

Bind byte-exact copies of C34 resource_only_D1_predictions.json, C34
D1_predictions.json and C33 D1_predictions.json. Evaluate conservative and
nominal choices for all three models, and upstream_default, fixed_vec1/2/4
(N-first, invalid vec falls back to default), fanout_only (minimum u*w,
larger vec, N-first). All eleven policies appear in each score table. Core five,
boundary five and all ten are fixed strata; no outcome-dependent subsets.

Generate all legal unique layouts with the unchanged C3 static generator:
59 FULL layouts and ten independent row-output TMA controls, 69 programs.
Reuse the exact C2c one_pass Gluon kernel, host-created rank-three BF16 128B
swizzled descriptor, output conversion, C2d driver ABI and graph timing.
All scalars are block-uniform. No C++ change or native rebuild. CPU-only
compilation uses pinned compiler 20d9d98fead78c2f65e1b3ba5a7881d86a836dad,
the native SHA in C2d and the same persistent Triton/ccache volume/images.
Save each compile attempt and all compiler artifacts; ccache stats unchanged.

Eligibility checks quarter/random/boundary values with the C2d independent CPU
oracle, CUBIN load hashes, ABI, physical n_regs/local bytes, dynamic shared and
driver occupancy. Spilled but executable layouts remain in the timing matrix;
spills are a result, not an exclusion criterion. A compile/resource/correctness
failure is preserved and explicitly marks an unavailable candidate. It never
changes a frozen policy choice. Report failed selections and valid denominators;
do not silently score a substituted layout. The oracle is the measured minimum
over eligible layouts with its finite-sample uncertainty, not known true optimum.

Primary measurement is the loop-free FULL time-versus-grid/SM slope at contiguous
128/256/384 MiB, ns/tile/SM. Host descriptor extent 24576x8192, unchanged row
stride; genuine unwrapped CTA origins. Keep 128 MiB raw time as auxiliary only.
Use three fresh H100! invocations, same C2d 3 visits x 4 event samples/cell,
2 warmups/visit, deterministic randomized order, q=natural,1,2,4,8, feasible caps
and physical aliases deduplicated. Every timed cell checks first/middle/last CTA.
The natural q slope is the sole policy-scoring endpoint. Controlled q curves
are diagnostics and do not tune the model or change policy choices.

The timing subprocess imports no Triton/JIT/compiler and loads only the frozen
CUBIN ZIP via the CUDA driver. NCU is separate: all unique FULL/control binaries,
K1 software instruction counts and natural/q4 hardware counters, groups of at
most 12 binaries, same cache/clock flags as C2d. Locally reimport every .ncu-rep.
Failed infrastructure dispatches/profiling attempts are retained. Allow at most
three attempts per infrastructure job; retry only execution/transport/profiler
failures, never because of observed performance. In particular no rerun to
improve an unfavorable candidate. Persistent account-budget errors stop paid
dispatches and are reported to the user, without switching accounts silently.

Report all 59 candidate truths and all ten controls. Compute n_regs upper,
marginal q interval (and discrete q set when available), and joint candidate/
default q-pair coverage from the frozen predictions and measured capacities.
Compute frozen point-prediction FULL relative/absolute RMSE, same-worker paired
absolute and ratio RMSE, signed nominal-q upper-tail coverage and full propagated
gain-bound coverage. Do not condition primary predictions on measured n_regs/q.
Do not retune any parameter after exposure. Out-of-support predictions and
abstentions are scored and labelled, never counted as proof of generalization.

Score each policy in core/boundary/all: switches, mean observed slope gain,
mean oracle regret, >3% slowdowns, and switches with observed gain <=3%; add
failure counts/denominators. Form gains/regrets within worker before aggregating.
Show worker variation and classify uncertain effects, not launches as independent
replicates. For three workers use median worker-paired per-case gains/regrets,
then equal-weight case means per stratum. Publish 95% t intervals across the
three worker-paired ratios as descriptive small-sample intervals; use t(2)=4.303.
3% is a practical threshold, not a guarantee or a significance threshold.

Preserve every older SHA, raw sample, CUBIN and NCU report. Freeze source,
protocol, predictions, dispatches, raw returns, artifacts, derived analysis and
REPORT.md. Run the final CPU validator, commit and push; stop for review.
