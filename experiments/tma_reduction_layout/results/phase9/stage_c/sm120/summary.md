# Phase9 StageC — sm120 frozen-binary timing

16 pairs; 28800 event samples; three independent processes; 2 observed physical UUIDs. All valid pairs retained, including changed residency.

Primary effect is100*(T_default-T_cand4)/T_default at grid65536. Positive means candidate4 is faster. Fixed3% practical band and df2 descriptive process uncertainty; no multiple-comparison significance claim. Smaller grids and OLS slopes are secondary.

| Pair | Decision | Mean improvement % | Process band % | Resources |
| --- | --- | --- | --- | --- |
| M32_N64_w8:reduction | BENEFIT | 15.424337 | [11.630773,19.217901] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N64_w8:load_copy | WITHIN_PRACTICAL_BAND | -0.091010 | [-0.362718,0.180697] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N64_w8:store_copy | WITHIN_PRACTICAL_BAND | -0.060605 | [-0.145504,0.024294] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N128_w4:reduction | WITHIN_PRACTICAL_BAND | 0.063918 | [-0.191138,0.318974] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N16_w4:reduction | WITHIN_PRACTICAL_BAND | 0.033314 | [-0.203945,0.270573] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N256_w4:reduction | WITHIN_PRACTICAL_BAND | 0.023958 | [-0.050373,0.098288] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N256_w4:load_copy | WITHIN_PRACTICAL_BAND | -0.013618 | [-0.143201,0.115964] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M32_N256_w4:store_copy | WITHIN_PRACTICAL_BAND | -0.017991 | [-0.033281,-0.002702] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M256_N64_w8:reduction | WITHIN_PRACTICAL_BAND | 0.082125 | [-0.081212,0.245463] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M512_N64_w8:reduction | WITHIN_PRACTICAL_BAND | -0.013069 | [-0.030359,0.004221] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M128_N32_w16:reduction | BENEFIT | 27.441044 | [25.908913,28.973176] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M128_N32_w16:load_copy | WITHIN_PRACTICAL_BAND | 0.076086 | [-0.033118,0.185289] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M128_N32_w16:store_copy | WITHIN_PRACTICAL_BAND | 0.146767 | [-0.031860,0.325393] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M1024_N32_w16:reduction | WITHIN_PRACTICAL_BAND | -0.028848 | [-0.042793,-0.014903] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M2048_N32_w16:reduction | UNAVAILABLE | — | — | actual-target pre-timing artifact failure |
| M128_N64_w32:reduction | BENEFIT | 22.435701 | [20.941760,23.929642] | MATCHED_ZERO_LOCAL_RESIDENCY |
| M512_N128_w32:reduction | UNAVAILABLE | — | — | actual-target pre-timing artifact failure |
| M32_N128_w8:reduction | WITHIN_PRACTICAL_BAND | 0.092103 | [-0.171758,0.355964] | MATCHED_ZERO_LOCAL_RESIDENCY |

All100-sample medians/statistics, individual invocation timings/effects, three-grid OLS intercepts/slopes/R²/residuals, exclusions and resource contexts are retained in results.json. No model fit or case removal. This comparison does not identify a historical bad compiler commit or verify a production fix.
