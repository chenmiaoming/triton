# Phase 6 — Execution-context and harness diagnostics

Starting branch: `explore/tma-reduction-layout`; baseline: `fbc9ae41eb1d79f196bb28b196039a4b3f9671e6`.

User authorizes staged execution, one commit per stage, immutable evidence,
supplementary measurements when needed, and switching Modal profiles on quota failure.
Timing stages separate raw freeze and analysis into distinct commits.

1. Stage A: rederive all 31 historical cases from raw samples, audit all 39 held-out
   prediction errors, and describe the five already-observed N64/w8 PRIMARY cases.
2. Stage B: outcome-blind archived-binary feasibility, exact launch bindings, and
   preregistration of supplementary canonical/single/repeated measurements.
3. Stage C: three independent exact-binary H100 invocations, raw freeze, then
   separate curve and cross-harness analysis.
4. Stage D: a previously unmeasured, structurally defined cohort; independent
   artifact admission before timing, raw freeze, and prospective predictor tests.

Historical raw samples, compiler artifacts, protocols, classifications, frozen
models and scientific conclusions remain byte-identical. New experiments are
stored under `results/phase6/`. H2b/H2c remain unverified absent a new valid
single-factor intervention. Cross-harness differences and G-D are descriptive.
No production heuristic or compiler/PR #11991 changes are authorized here.

Stage A:

```bash
python experiments/tma_reduction_layout/phase6/diagnose.py
python experiments/tma_reduction_layout/phase6/diagnose.py --validate
```

Frozen legacy validators include stage-scope assertions that reject later-stage
directories. They run unchanged in detached trusted-baseline checkouts, with
every current historical file checked against its Git blob. The original
current-checkout scope diagnostic is retained in Stage A. New validators run
against the current checkout and independently rederive new evidence.
