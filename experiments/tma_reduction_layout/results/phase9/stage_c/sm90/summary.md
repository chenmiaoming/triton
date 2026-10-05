# Phase9 StageC — sm90 frozen-binary timing

18 pairs; 32400 event samples; three independent processes; 3 observed physical UUIDs. All valid pairs retained, including changed residency.

Primary effect is100*(T_default-T_cand4)/T_default at grid65536. Positive means candidate4 is faster. Fixed3% practical band and df2 descriptive process uncertainty; no multiple-comparison significance claim. Smaller grids and OLS slopes are secondary.

| Pair | Decision | Mean improvement % | Process band % | Resources |
| --- | --- | --- | --- | --- |
| M32_N64_w8:reduction | BENEFIT | 34.725083 | [34.093276,35.356891] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N64_w8:load_copy | WITHIN_PRACTICAL_BAND | 0.180230 | [-1.226119,1.586580] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N64_w8:store_copy | WITHIN_PRACTICAL_BAND | 0.224945 | [-0.459695,0.909585] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N128_w4:reduction | WITHIN_PRACTICAL_BAND | 0.361074 | [-0.216853,0.939001] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N16_w4:reduction | WITHIN_PRACTICAL_BAND | -0.319448 | [-0.807817,0.168921] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N256_w4:reduction | WITHIN_PRACTICAL_BAND | 0.219303 | [0.008812,0.429794] | CHANGED_RESIDENCY |
| M32_N256_w4:load_copy | WITHIN_PRACTICAL_BAND | 0.047804 | [-0.077569,0.173177] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N256_w4:store_copy | WITHIN_PRACTICAL_BAND | 0.026877 | [-0.184187,0.237941] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M256_N64_w8:reduction | WITHIN_PRACTICAL_BAND | 0.124339 | [-0.105555,0.354232] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M512_N64_w8:reduction | WITHIN_PRACTICAL_BAND | -0.020071 | [-0.069318,0.029176] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M128_N32_w16:reduction | BENEFIT | 34.357759 | [34.198284,34.517234] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M128_N32_w16:load_copy | WITHIN_PRACTICAL_BAND | -0.133074 | [-0.594470,0.328323] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M128_N32_w16:store_copy | WITHIN_PRACTICAL_BAND | 0.124597 | [-0.078469,0.327663] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M1024_N32_w16:reduction | WITHIN_PRACTICAL_BAND | -0.240736 | [-0.647480,0.166007] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M2048_N32_w16:reduction | BENEFIT | 5.507284 | [5.113871,5.900697] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M128_N64_w32:reduction | BENEFIT | 29.953236 | [29.637609,30.268863] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M512_N128_w32:reduction | BENEFIT | 7.348277 | [7.084254,7.612300] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N128_w8:reduction | BENEFIT | 20.330791 | [19.047029,21.614554] | MATCHED_ZERO_LOCAL_RESIDENCY |

All100-sample medians/statistics, individual invocation timings/effects, three-grid OLS intercepts/slopes/R²/residuals, exclusions and resource contexts are retained in results.json. No model fit or case removal. This comparison does not identify a historical bad compiler commit or verify a production fix.
