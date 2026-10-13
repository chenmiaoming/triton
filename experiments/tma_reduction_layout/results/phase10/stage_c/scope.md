# Stage C scope — Actual compiler patch

This is an actual-patch contrast with the same upstream and assembler. All 372 compiler attempts and 158,400 exact-binary samples are retained; 352 compiler attempts pass full-output correctness and 20 exceed SM120 shared-memory limits. No native rebuild, rule adjustment, outcome-selected exclusion or timing rerun occurred.

| Target | Eligible held-out pairs | Held-out benefit | Within ±3% | Unresolved | Anchor benefit |
| --- | --- | --- | --- | --- | --- |
| sm100 | 48 | 13 | 22 | 13 | 6 |
| sm120 | 40 | 0 | 38 | 2 | 1 |
| sm90 | 48 | 2 | 43 | 3 | 4 |

The fixed descriptive bands classify 0 regressions at any of the three grids. They leave 222 case/grid effects unresolved, including 114 bands allowing losses beyond 3% (50 with changed CUBINs). Zero classified regressions is not proof that all cases avoid a regression. These are three-process bands without familywise significance.

At the full grid, the canonical M32/N64/8-warps device path decreases time by 41.170% on H100, 21.438% on B200 and 17.753% on SM120; all three exceed the frozen practical benefit threshold. The host path is a benefit on H100 and B200 (45.801%, 45.099%), while the SM120 host effect is unresolved (mean 16.052%, band [-6.671%, 38.776%]). The original M32/N128/4-warps host case decreases time by 19.813% on B200; its H100 2.094% and SM120 0.409% contrasts remain within the practical band.

The held-out full-grid subset gives 2 H100 and 13 B200 benefit classifications, and no SM120 held-out benefit classifications. Most SM120 held-out effects are within the practical band; two are unresolved. Ten SM120 pairs remain unavailable (eight held-out FP32/sum paths and two large-input BF16 fallback probes), with the same resource failures in baseline and patch. All six guard controls per architecture preserve identical CUBINs. Resource-changing and local-memory cases remain included rather than screened out.

The rule has broader evidence than a fixed-vector shape heuristic: it selects from structural instruction costs, preserves guards and cost-dominated defaults, and shows benefit on several unseen cells and on the canonical device path across three architectures. The declared dtype/op subset, finite-value oracle, tested warp counts and resource availability do not cover all eligible programs or floating-point edge cases. B200/SM100 is not GB300/SM103. This phase does not attribute the original historical compiler regression to a particular commit or isolate bank-conflict causality.

Before a PR, review the model assumptions and remaining uncertain/resource-limited coverage, then decide whether another independently frozen confirmation is needed. No further phase, native change, PR creation/update or reviewer message is started. User approval remains required before any PR.

[All case/grid results](results.json) · [Full primary table](summary.md) · [Scope data](scope.json)
