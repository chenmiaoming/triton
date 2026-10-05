# Phase9 StageD — Architecture scope decision

All three requested architectures were allocated and measured with their own archived CUBINs. The exact native core was reused; no compiler/production change or PR was made.

| Target | Original M32_N64_w8 | Mean improvement % | Process band % | Samples | All primary decisions |
| --- | --- | --- | --- | --- | --- |
| sm90 | BENEFIT | 34.725083 | [34.093276,35.356891] | 32400 | {'load_copy': {'WITHIN_PRACTICAL_BAND': 3}, 'reduction': {'BENEFIT': 6, 'WITHIN_PRACTICAL_BAND': 6}, 'store_copy': {'WITHIN_PRACTICAL_BAND': 3}} |
| sm100 | BENEFIT | 16.453098 | [15.895916,17.010280] | 32400 | {'load_copy': {'WITHIN_PRACTICAL_BAND': 3}, 'reduction': {'BENEFIT': 8, 'WITHIN_PRACTICAL_BAND': 4}, 'store_copy': {'WITHIN_PRACTICAL_BAND': 3}} |
| sm120 | BENEFIT | 15.424337 | [11.630773,19.217901] | 28800 | {'load_copy': {'WITHIN_PRACTICAL_BAND': 3}, 'reduction': {'BENEFIT': 3, 'UNAVAILABLE': 2, 'WITHIN_PRACTICAL_BAND': 7}, 'store_copy': {'WITHIN_PRACTICAL_BAND': 3}} |

A global vector4 policy is not justified. Finite retrospective matrix and incomplete/uncertain safety coverage do not establish a universal vector4 policy. Whole-kernel benefits include descriptor setup, layout lowering, registers, residency and scheduling. Source-level candidate selection is not a production patch.

No primary pair crossed the preregistered3% regression threshold. This is finite tested coverage, not an all-TMA safety claim.

## Retained regression cases

| Target | Pair | Mean improvement % | Process band % |
| --- | --- | --- | --- |

0 unresolved and 2 unavailable pairs remain explicit in results.json. The two unavailable SM120 large tiles exceed actual shared-memory limits; they are not replaced by smaller tiles.

## Secondary grids with unresolved practical effects

| Target | Pair | Grid | Mean improvement % | Process band % |
| --- | --- | --- | --- | --- |
| sm90 | M32_N64_w8:store_copy | 16384 | 0.812823 | [-2.002818,3.628464] |
| sm90 | M32_N128_w4:reduction | 16384 | -0.348231 | [-3.224863,2.528402] |
| sm90 | M32_N16_w4:reduction | 16384 | -1.661167 | [-8.087178,4.764844] |
| sm100 | M32_N256_w4:reduction | 16384 | 1.595905 | [-0.560257,3.752066] |
| sm100 | M256_N64_w8:reduction | 16384 | -2.709546 | [-11.779412,6.360321] |
| sm100 | M512_N64_w8:reduction | 16384 | -1.262961 | [-6.628906,4.102984] |
| sm120 | M32_N64_w8:load_copy | 16384 | -2.134404 | [-5.025387,0.756579] |
| sm120 | M32_N128_w4:reduction | 16384 | -3.493237 | [-12.842197,5.855724] |
| sm120 | M32_N16_w4:reduction | 32768 | -1.313592 | [-4.922263,2.295079] |

9 secondary-grid effects remain unresolved; 0 meet the regression rule. Some bands include decreases beyond3%; the larger-grid result does not establish non-regression at these smaller grids. They remain in the same fixed cohort with no retiming or threshold change.

Next implementation should be scoped to justified target/IR contexts, preserve unrelated descriptor loads/stores, and receive separate fixed-patch validation. Original historical good/bad compiler attribution is still outstanding. Every old sample and artifact remains unchanged. No later phase starts automatically; user approval is mandatory before creating any PR, including draft.
