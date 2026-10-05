# Phase9 StageC — sm100 frozen-binary timing

18 pairs; 32400 event samples; three independent processes; 3 observed physical UUIDs. All valid pairs retained, including changed residency.

Primary effect is100*(T_default-T_cand4)/T_default at grid65536. Positive means candidate4 is faster. Fixed3% practical band and df2 descriptive process uncertainty; no multiple-comparison significance claim. Smaller grids and OLS slopes are secondary.

| Pair | Decision | Mean improvement % | Process band % | Resources |
| --- | --- | --- | --- | --- |
| M32_N64_w8:reduction | BENEFIT | 16.453098 | [15.895916,17.010280] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N64_w8:load_copy | WITHIN_PRACTICAL_BAND | -0.049425 | [-0.661148,0.562298] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N64_w8:store_copy | WITHIN_PRACTICAL_BAND | -0.115757 | [-0.472668,0.241155] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N128_w4:reduction | BENEFIT | 9.007633 | [8.241739,9.773527] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N16_w4:reduction | WITHIN_PRACTICAL_BAND | 0.095724 | [-0.506315,0.697764] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N256_w4:reduction | WITHIN_PRACTICAL_BAND | 1.007694 | [0.807547,1.207841] | CHANGED_RESIDENCY |
| M32_N256_w4:load_copy | WITHIN_PRACTICAL_BAND | -0.039562 | [-0.128308,0.049185] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N256_w4:store_copy | WITHIN_PRACTICAL_BAND | 0.046920 | [-0.191517,0.285357] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M256_N64_w8:reduction | WITHIN_PRACTICAL_BAND | 1.668519 | [1.086335,2.250703] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M512_N64_w8:reduction | WITHIN_PRACTICAL_BAND | 1.350046 | [1.171476,1.528616] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M128_N32_w16:reduction | BENEFIT | 30.433777 | [30.200527,30.667027] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M128_N32_w16:load_copy | WITHIN_PRACTICAL_BAND | -0.001615 | [-0.099409,0.096179] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M128_N32_w16:store_copy | WITHIN_PRACTICAL_BAND | 0.043016 | [-0.065675,0.151707] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M1024_N32_w16:reduction | BENEFIT | 9.859092 | [9.492764,10.225420] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M2048_N32_w16:reduction | BENEFIT | 5.485303 | [5.386602,5.584003] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M128_N64_w32:reduction | BENEFIT | 23.190457 | [22.991324,23.389589] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M512_N128_w32:reduction | BENEFIT | 5.822778 | [5.680117,5.965440] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N128_w8:reduction | BENEFIT | 12.826968 | [12.578077,13.075860] | MATCHED_ZERO_LOCAL_RESIDENCY |

All100-sample medians/statistics, individual invocation timings/effects, three-grid OLS intercepts/slopes/R²/residuals, exclusions and resource contexts are retained in results.json. No model fit or case removal. This comparison does not identify a historical bad compiler commit or verify a production fix.
