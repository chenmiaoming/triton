# Fixed-cadence replacement: legacy

Data status: **OK**. Three workers × ten visits × 31 samples per physical condition; 552,420 target observations.
Each graph runs eight untimed semantic iterations, then exactly one timed iteration. Original pipeline zeroes remain inside each iteration; any input restoration precedes the target events. Between observations the requested minimum host gap is 1 ms; actual gaps remain recorded.

U/G0 equivalence: 0/0. Same-program sham equivalence: 189/189.
Those are logical workload comparisons; exact aliases can share data. Distinct physical U/G0 pairs: 0; distinct physical sham labels per worker: 189. They are not additional independent GPUs.
Equivalence requires the full paired CI95 to be inside ±3%. A candidate is marked strictly improved only when its fresh U and G0 lower bounds both exceed +3%, and the relevant controls pass. Legacy/anchor comparisons instead use their explicitly named Gluon reference.

All logical rows, physical aliases, visit medians, host-gap summaries, old/new descriptive comparisons and control decisions are retained in `analysis.json` and `timing.csv`. No unfavorable or inconclusive control is removed.

Recorded host gaps below the requested minimum: 0. Observed minimum/maximum: 1006.124 / 7428.961 µs. GPU-side primes are outside these host-gap durations and outside target timing.

## Control results

| Context | U/reference | G0 vs U gain % [CI95] | Same-program repeat gain % [CI95] | Status |
|---|---|---:|---:|---|
| M3|M32|N128|W4|vec2|fixed|q1|G16384 | M32_N128_w4:vec2:n_first:fixed:ONE_FULL|q1|G16384|K1 | unavailable | -0.004 [-0.028,0.021] | validated |
| M3|M32|N128|W4|vec2|fixed|q1|G32768 | M32_N128_w4:vec2:n_first:fixed:ONE_FULL|q1|G32768|K1 | unavailable | 0.002 [-0.020,0.025] | validated |
| M3|M32|N128|W4|vec2|fixed|q1|G49152 | M32_N128_w4:vec2:n_first:fixed:ONE_FULL|q1|G49152|K1 | unavailable | 0.001 [-0.011,0.012] | validated |
| M3|M32|N128|W4|vec2|fixed|q2|G16384 | M32_N128_w4:vec2:n_first:fixed:ONE_FULL|q2|G16384|K1 | unavailable | -0.047 [-0.170,0.087] | validated |
| M3|M32|N128|W4|vec2|fixed|q2|G32768 | M32_N128_w4:vec2:n_first:fixed:ONE_FULL|q2|G32768|K1 | unavailable | 0.004 [-0.067,0.077] | validated |
| M3|M32|N128|W4|vec2|fixed|q2|G49152 | M32_N128_w4:vec2:n_first:fixed:ONE_FULL|q2|G49152|K1 | unavailable | -0.008 [-0.087,0.072] | validated |
| M3|M32|N128|W4|vec2|fixed|q4|G16384 | M32_N128_w4:vec2:n_first:fixed:ONE_FULL|q4|G16384|K1 | unavailable | 0.116 [-0.440,0.669] | validated |
| M3|M32|N128|W4|vec2|fixed|q4|G32768 | M32_N128_w4:vec2:n_first:fixed:ONE_FULL|q4|G32768|K1 | unavailable | 0.015 [-0.003,0.031] | validated |
| M3|M32|N128|W4|vec2|fixed|q4|G49152 | M32_N128_w4:vec2:n_first:fixed:ONE_FULL|q4|G49152|K1 | unavailable | -0.057 [-0.270,0.196] | validated |
| M3|M32|N128|W4|vec2|fixed|q8|G16384 | M32_N128_w4:vec2:n_first:fixed:ONE_FULL|q8|G16384|K1 | unavailable | -0.747 [-1.728,0.295] | validated |
| M3|M32|N128|W4|vec2|fixed|q8|G32768 | M32_N128_w4:vec2:n_first:fixed:ONE_FULL|q8|G32768|K1 | unavailable | 0.006 [-0.027,0.039] | validated |
| M3|M32|N128|W4|vec2|fixed|q8|G49152 | M32_N128_w4:vec2:n_first:fixed:ONE_FULL|q8|G49152|K1 | unavailable | -0.085 [-0.376,0.243] | validated |
| M3|M32|N128|W4|vec2|fixed|qnatural|G16384 | M32_N128_w4:vec2:n_first:fixed:ONE_FULL|qnatural|G16384|K1 | unavailable | -0.328 [-1.128,0.590] | validated |
| M3|M32|N128|W4|vec2|fixed|qnatural|G32768 | M32_N128_w4:vec2:n_first:fixed:ONE_FULL|qnatural|G32768|K1 | unavailable | 0.022 [0.001,0.043] | validated |
| M3|M32|N128|W4|vec2|fixed|qnatural|G49152 | M32_N128_w4:vec2:n_first:fixed:ONE_FULL|qnatural|G49152|K1 | unavailable | 0.033 [-0.220,0.243] | validated |
| M3|M32|N128|W4|vec2|native|q1|G16384 | M32_N128_w4:vec2:n_first:native:ONE_FULL|q1|G16384|K1 | unavailable | -0.026 [-0.050,-0.003] | validated |
| M3|M32|N128|W4|vec2|native|q1|G32768 | M32_N128_w4:vec2:n_first:native:ONE_FULL|q1|G32768|K1 | unavailable | -0.010 [-0.034,0.009] | validated |
| M3|M32|N128|W4|vec2|native|q1|G49152 | M32_N128_w4:vec2:n_first:native:ONE_FULL|q1|G49152|K1 | unavailable | -0.002 [-0.016,0.011] | validated |
| M3|M32|N128|W4|vec2|native|q2|G16384 | M32_N128_w4:vec2:n_first:native:ONE_FULL|q2|G16384|K1 | unavailable | -0.014 [-0.130,0.107] | validated |
| M3|M32|N128|W4|vec2|native|q2|G32768 | M32_N128_w4:vec2:n_first:native:ONE_FULL|q2|G32768|K1 | unavailable | 0.047 [-0.035,0.130] | validated |
| M3|M32|N128|W4|vec2|native|q2|G49152 | M32_N128_w4:vec2:n_first:native:ONE_FULL|q2|G49152|K1 | unavailable | 0.009 [-0.069,0.097] | validated |
| M3|M32|N128|W4|vec2|native|q4|G16384 | M32_N128_w4:vec2:n_first:native:ONE_FULL|q4|G16384|K1 | unavailable | 0.409 [-0.194,1.047] | validated |
| M3|M32|N128|W4|vec2|native|q4|G32768 | M32_N128_w4:vec2:n_first:native:ONE_FULL|q4|G32768|K1 | unavailable | 0.001 [-0.015,0.017] | validated |
| M3|M32|N128|W4|vec2|native|q4|G49152 | M32_N128_w4:vec2:n_first:native:ONE_FULL|q4|G49152|K1 | unavailable | -0.181 [-0.399,0.050] | validated |
| M3|M32|N128|W4|vec2|native|q8|G16384 | M32_N128_w4:vec2:n_first:native:ONE_FULL|q8|G16384|K1 | unavailable | -0.425 [-1.318,0.506] | validated |
| M3|M32|N128|W4|vec2|native|q8|G32768 | M32_N128_w4:vec2:n_first:native:ONE_FULL|q8|G32768|K1 | unavailable | -0.004 [-0.031,0.017] | validated |
| M3|M32|N128|W4|vec2|native|q8|G49152 | M32_N128_w4:vec2:n_first:native:ONE_FULL|q8|G49152|K1 | unavailable | 0.061 [-0.237,0.320] | validated |
| M3|M32|N128|W4|vec2|native|qnatural|G16384 | M32_N128_w4:vec2:n_first:native:ONE_FULL|qnatural|G16384|K1 | unavailable | 0.069 [-0.556,0.726] | validated |
| M3|M32|N128|W4|vec2|native|qnatural|G32768 | M32_N128_w4:vec2:n_first:native:ONE_FULL|qnatural|G32768|K1 | unavailable | -0.014 [-0.049,0.025] | validated |
| M3|M32|N128|W4|vec2|native|qnatural|G49152 | M32_N128_w4:vec2:n_first:native:ONE_FULL|qnatural|G49152|K1 | unavailable | -0.095 [-0.256,0.108] | validated |
| M3|M32|N128|W4|vec4|fixed|q1|G16384 | M32_N128_w4:vec4:n_first:fixed:ONE_FULL|q1|G16384|K1 | unavailable | 0.003 [-0.037,0.043] | validated |
| M3|M32|N128|W4|vec4|fixed|q1|G32768 | M32_N128_w4:vec4:n_first:fixed:ONE_FULL|q1|G32768|K1 | unavailable | -0.016 [-0.042,0.009] | validated |
| M3|M32|N128|W4|vec4|fixed|q1|G49152 | M32_N128_w4:vec4:n_first:fixed:ONE_FULL|q1|G49152|K1 | unavailable | -0.001 [-0.014,0.013] | validated |
| M3|M32|N128|W4|vec4|fixed|q2|G16384 | M32_N128_w4:vec4:n_first:fixed:ONE_FULL|q2|G16384|K1 | unavailable | -0.030 [-0.140,0.083] | validated |
| M3|M32|N128|W4|vec4|fixed|q2|G32768 | M32_N128_w4:vec4:n_first:fixed:ONE_FULL|q2|G32768|K1 | unavailable | 0.016 [-0.051,0.081] | validated |
| M3|M32|N128|W4|vec4|fixed|q2|G49152 | M32_N128_w4:vec4:n_first:fixed:ONE_FULL|q2|G49152|K1 | unavailable | 0.017 [-0.040,0.073] | validated |
| M3|M32|N128|W4|vec4|fixed|q4|G16384 | M32_N128_w4:vec4:n_first:fixed:ONE_FULL|q4|G16384|K1 | unavailable | 0.163 [-0.519,0.841] | validated |
| M3|M32|N128|W4|vec4|fixed|q4|G32768 | M32_N128_w4:vec4:n_first:fixed:ONE_FULL|q4|G32768|K1 | unavailable | -0.006 [-0.025,0.012] | validated |
| M3|M32|N128|W4|vec4|fixed|q4|G49152 | M32_N128_w4:vec4:n_first:fixed:ONE_FULL|q4|G49152|K1 | unavailable | -0.038 [-0.253,0.174] | validated |
| M3|M32|N128|W4|vec4|fixed|q8|G16384 | M32_N128_w4:vec4:n_first:fixed:ONE_FULL|q8|G16384|K1 | unavailable | 0.644 [-0.262,1.512] | validated |
| M3|M32|N128|W4|vec4|fixed|q8|G32768 | M32_N128_w4:vec4:n_first:fixed:ONE_FULL|q8|G32768|K1 | unavailable | -0.014 [-0.038,0.008] | validated |
| M3|M32|N128|W4|vec4|fixed|q8|G49152 | M32_N128_w4:vec4:n_first:fixed:ONE_FULL|q8|G49152|K1 | unavailable | 0.038 [-0.253,0.315] | validated |
| M3|M32|N128|W4|vec4|fixed|qnatural|G16384 | M32_N128_w4:vec4:n_first:fixed:ONE_FULL|qnatural|G16384|K1 | unavailable | 0.211 [-0.482,0.846] | validated |
| M3|M32|N128|W4|vec4|fixed|qnatural|G32768 | M32_N128_w4:vec4:n_first:fixed:ONE_FULL|qnatural|G32768|K1 | unavailable | 0.004 [-0.018,0.028] | validated |
| M3|M32|N128|W4|vec4|fixed|qnatural|G49152 | M32_N128_w4:vec4:n_first:fixed:ONE_FULL|qnatural|G49152|K1 | unavailable | -0.048 [-0.197,0.087] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q1|G16384 | M32_N128_w4:vec4:n_first:legacy_fixed:ONE_FULL|q1|G16384|K1 | unavailable | 0.002 [-0.022,0.025] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q1|G32768 | M32_N128_w4:vec4:n_first:legacy_fixed:ONE_FULL|q1|G32768|K1 | unavailable | -0.009 [-0.024,0.008] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q1|G49152 | M32_N128_w4:vec4:n_first:legacy_fixed:ONE_FULL|q1|G49152|K1 | unavailable | -0.013 [-0.036,0.007] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q2|G16384 | M32_N128_w4:vec4:n_first:legacy_fixed:ONE_FULL|q2|G16384|K1 | unavailable | -0.076 [-0.142,-0.009] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q2|G32768 | M32_N128_w4:vec4:n_first:legacy_fixed:ONE_FULL|q2|G32768|K1 | unavailable | 0.019 [-0.134,0.169] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q2|G49152 | M32_N128_w4:vec4:n_first:legacy_fixed:ONE_FULL|q2|G49152|K1 | unavailable | 0.004 [-0.070,0.074] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q4|G16384 | M32_N128_w4:vec4:n_first:legacy_fixed:ONE_FULL|q4|G16384|K1 | unavailable | -0.036 [-0.480,0.434] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q4|G32768 | M32_N128_w4:vec4:n_first:legacy_fixed:ONE_FULL|q4|G32768|K1 | unavailable | -0.005 [-0.033,0.023] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q4|G49152 | M32_N128_w4:vec4:n_first:legacy_fixed:ONE_FULL|q4|G49152|K1 | unavailable | -0.017 [-0.180,0.149] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q8|G16384 | M32_N128_w4:vec4:n_first:legacy_fixed:ONE_FULL|q8|G16384|K1 | unavailable | 0.316 [-0.999,1.739] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q8|G32768 | M32_N128_w4:vec4:n_first:legacy_fixed:ONE_FULL|q8|G32768|K1 | unavailable | 0.011 [-0.011,0.031] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q8|G49152 | M32_N128_w4:vec4:n_first:legacy_fixed:ONE_FULL|q8|G49152|K1 | unavailable | 0.013 [-0.254,0.246] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|qnatural|G16384 | M32_N128_w4:vec4:n_first:legacy_fixed:ONE_FULL|qnatural|G16384|K1 | unavailable | -0.056 [-0.761,0.616] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|qnatural|G32768 | M32_N128_w4:vec4:n_first:legacy_fixed:ONE_FULL|qnatural|G32768|K1 | unavailable | -0.002 [-0.026,0.023] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|qnatural|G49152 | M32_N128_w4:vec4:n_first:legacy_fixed:ONE_FULL|qnatural|G49152|K1 | unavailable | 0.090 [-0.062,0.240] | validated |
| M3|M32|N128|W4|vec4|native|q1|G16384 | M32_N128_w4:vec4:n_first:native:ONE_FULL|q1|G16384|K1 | unavailable | 0.013 [-0.016,0.043] | validated |
| M3|M32|N128|W4|vec4|native|q1|G32768 | M32_N128_w4:vec4:n_first:native:ONE_FULL|q1|G32768|K1 | unavailable | 0.003 [-0.017,0.023] | validated |
| M3|M32|N128|W4|vec4|native|q1|G49152 | M32_N128_w4:vec4:n_first:native:ONE_FULL|q1|G49152|K1 | unavailable | 0.011 [0.001,0.021] | validated |
| M3|M32|N128|W4|vec4|native|q2|G16384 | M32_N128_w4:vec4:n_first:native:ONE_FULL|q2|G16384|K1 | unavailable | -0.032 [-0.114,0.052] | validated |
| M3|M32|N128|W4|vec4|native|q2|G32768 | M32_N128_w4:vec4:n_first:native:ONE_FULL|q2|G32768|K1 | unavailable | -0.001 [-0.070,0.063] | validated |
| M3|M32|N128|W4|vec4|native|q2|G49152 | M32_N128_w4:vec4:n_first:native:ONE_FULL|q2|G49152|K1 | unavailable | 0.022 [-0.026,0.068] | validated |
| M3|M32|N128|W4|vec4|native|q4|G16384 | M32_N128_w4:vec4:n_first:native:ONE_FULL|q4|G16384|K1 | unavailable | 0.176 [-0.271,0.626] | validated |
| M3|M32|N128|W4|vec4|native|q4|G32768 | M32_N128_w4:vec4:n_first:native:ONE_FULL|q4|G32768|K1 | unavailable | -0.000 [-0.021,0.020] | validated |
| M3|M32|N128|W4|vec4|native|q4|G49152 | M32_N128_w4:vec4:n_first:native:ONE_FULL|q4|G49152|K1 | unavailable | 0.096 [-0.191,0.336] | validated |
| M3|M32|N128|W4|vec4|native|q8|G16384 | M32_N128_w4:vec4:n_first:native:ONE_FULL|q8|G16384|K1 | unavailable | -0.251 [-1.915,1.275] | validated |
| M3|M32|N128|W4|vec4|native|q8|G32768 | M32_N128_w4:vec4:n_first:native:ONE_FULL|q8|G32768|K1 | unavailable | 0.021 [-0.003,0.042] | validated |
| M3|M32|N128|W4|vec4|native|q8|G49152 | M32_N128_w4:vec4:n_first:native:ONE_FULL|q8|G49152|K1 | unavailable | -0.019 [-0.214,0.204] | validated |
| M3|M32|N128|W4|vec4|native|qnatural|G16384 | M32_N128_w4:vec4:n_first:native:ONE_FULL|qnatural|G16384|K1 | unavailable | 0.008 [-0.617,0.601] | validated |
| M3|M32|N128|W4|vec4|native|qnatural|G32768 | M32_N128_w4:vec4:n_first:native:ONE_FULL|qnatural|G32768|K1 | unavailable | 0.013 [-0.006,0.031] | validated |
| M3|M32|N128|W4|vec4|native|qnatural|G49152 | M32_N128_w4:vec4:n_first:native:ONE_FULL|qnatural|G49152|K1 | unavailable | -0.089 [-0.254,0.039] | validated |
| M3|M32|N128|W4|vec8|fixed|q1|G16384 | M32_N128_w4:vec8:n_first:fixed:ONE_FULL|q1|G16384|K1 | unavailable | -0.012 [-0.033,0.010] | validated |
| M3|M32|N128|W4|vec8|fixed|q1|G32768 | M32_N128_w4:vec8:n_first:fixed:ONE_FULL|q1|G32768|K1 | unavailable | -0.000 [-0.020,0.023] | validated |
| M3|M32|N128|W4|vec8|fixed|q1|G49152 | M32_N128_w4:vec8:n_first:fixed:ONE_FULL|q1|G49152|K1 | unavailable | 0.004 [-0.014,0.021] | validated |
| M3|M32|N128|W4|vec8|fixed|q2|G16384 | M32_N128_w4:vec8:n_first:fixed:ONE_FULL|q2|G16384|K1 | unavailable | 0.009 [-0.100,0.109] | validated |
| M3|M32|N128|W4|vec8|fixed|q2|G32768 | M32_N128_w4:vec8:n_first:fixed:ONE_FULL|q2|G32768|K1 | unavailable | -0.014 [-0.062,0.032] | validated |
| M3|M32|N128|W4|vec8|fixed|q2|G49152 | M32_N128_w4:vec8:n_first:fixed:ONE_FULL|q2|G49152|K1 | unavailable | -0.017 [-0.056,0.019] | validated |
| M3|M32|N128|W4|vec8|fixed|q4|G16384 | M32_N128_w4:vec8:n_first:fixed:ONE_FULL|q4|G16384|K1 | unavailable | -0.265 [-0.625,0.114] | validated |
| M3|M32|N128|W4|vec8|fixed|q4|G32768 | M32_N128_w4:vec8:n_first:fixed:ONE_FULL|q4|G32768|K1 | unavailable | -0.014 [-0.037,0.011] | validated |
| M3|M32|N128|W4|vec8|fixed|q4|G49152 | M32_N128_w4:vec8:n_first:fixed:ONE_FULL|q4|G49152|K1 | unavailable | 0.064 [-0.033,0.167] | validated |
| M3|M32|N128|W4|vec8|fixed|q8|G16384 | M32_N128_w4:vec8:n_first:fixed:ONE_FULL|q8|G16384|K1 | unavailable | -0.381 [-1.248,0.536] | validated |
| M3|M32|N128|W4|vec8|fixed|q8|G32768 | M32_N128_w4:vec8:n_first:fixed:ONE_FULL|q8|G32768|K1 | unavailable | 0.003 [-0.011,0.018] | validated |
| M3|M32|N128|W4|vec8|fixed|q8|G49152 | M32_N128_w4:vec8:n_first:fixed:ONE_FULL|q8|G49152|K1 | unavailable | -0.127 [-0.391,0.124] | validated |
| M3|M32|N128|W4|vec8|fixed|qnatural|G16384 | M32_N128_w4:vec8:n_first:fixed:ONE_FULL|qnatural|G16384|K1 | unavailable | -0.030 [-0.525,0.469] | validated |
| M3|M32|N128|W4|vec8|fixed|qnatural|G32768 | M32_N128_w4:vec8:n_first:fixed:ONE_FULL|qnatural|G32768|K1 | unavailable | -0.004 [-0.056,0.044] | validated |
| M3|M32|N128|W4|vec8|fixed|qnatural|G49152 | M32_N128_w4:vec8:n_first:fixed:ONE_FULL|qnatural|G49152|K1 | unavailable | 0.001 [-0.147,0.115] | validated |
| M3|M32|N128|W4|vec8|native|q1|G16384 | M32_N128_w4:vec8:n_first:native:ONE_FULL|q1|G16384|K1 | unavailable | -0.001 [-0.026,0.021] | validated |
| M3|M32|N128|W4|vec8|native|q1|G32768 | M32_N128_w4:vec8:n_first:native:ONE_FULL|q1|G32768|K1 | unavailable | -0.000 [-0.016,0.020] | validated |
| M3|M32|N128|W4|vec8|native|q1|G49152 | M32_N128_w4:vec8:n_first:native:ONE_FULL|q1|G49152|K1 | unavailable | -0.001 [-0.013,0.010] | validated |
| M3|M32|N128|W4|vec8|native|q2|G16384 | M32_N128_w4:vec8:n_first:native:ONE_FULL|q2|G16384|K1 | unavailable | 0.068 [-0.004,0.134] | validated |
| M3|M32|N128|W4|vec8|native|q2|G32768 | M32_N128_w4:vec8:n_first:native:ONE_FULL|q2|G32768|K1 | unavailable | 0.036 [-0.024,0.097] | validated |
| M3|M32|N128|W4|vec8|native|q2|G49152 | M32_N128_w4:vec8:n_first:native:ONE_FULL|q2|G49152|K1 | unavailable | 0.024 [-0.015,0.063] | validated |
| M3|M32|N128|W4|vec8|native|q4|G16384 | M32_N128_w4:vec8:n_first:native:ONE_FULL|q4|G16384|K1 | unavailable | 0.086 [-0.489,0.558] | validated |
| M3|M32|N128|W4|vec8|native|q4|G32768 | M32_N128_w4:vec8:n_first:native:ONE_FULL|q4|G32768|K1 | unavailable | -0.004 [-0.030,0.026] | validated |
| M3|M32|N128|W4|vec8|native|q4|G49152 | M32_N128_w4:vec8:n_first:native:ONE_FULL|q4|G49152|K1 | unavailable | -0.061 [-0.197,0.055] | validated |
| M3|M32|N128|W4|vec8|native|q8|G16384 | M32_N128_w4:vec8:n_first:native:ONE_FULL|q8|G16384|K1 | unavailable | -0.323 [-1.017,0.463] | validated |
| M3|M32|N128|W4|vec8|native|q8|G32768 | M32_N128_w4:vec8:n_first:native:ONE_FULL|q8|G32768|K1 | unavailable | -0.023 [-0.068,0.011] | validated |
| M3|M32|N128|W4|vec8|native|q8|G49152 | M32_N128_w4:vec8:n_first:native:ONE_FULL|q8|G49152|K1 | unavailable | 0.031 [-0.204,0.267] | validated |
| M3|M32|N128|W4|vec8|native|qnatural|G16384 | M32_N128_w4:vec8:n_first:native:ONE_FULL|qnatural|G16384|K1 | unavailable | 0.034 [-0.436,0.554] | validated |
| M3|M32|N128|W4|vec8|native|qnatural|G32768 | M32_N128_w4:vec8:n_first:native:ONE_FULL|qnatural|G32768|K1 | unavailable | 0.002 [-0.023,0.026] | validated |
| M3|M32|N128|W4|vec8|native|qnatural|G49152 | M32_N128_w4:vec8:n_first:native:ONE_FULL|qnatural|G49152|K1 | unavailable | 0.025 [-0.199,0.262] | validated |
| M3|M32|N64|W8|vec2|fixed|q1|G32768 | M32_N64_w8:vec2:n_first:fixed:ONE_FULL|q1|G32768|K1 | unavailable | 0.001 [-0.020,0.024] | validated |
| M3|M32|N64|W8|vec2|fixed|q1|G65536 | M32_N64_w8:vec2:n_first:fixed:ONE_FULL|q1|G65536|K1 | unavailable | 0.001 [-0.009,0.011] | validated |
| M3|M32|N64|W8|vec2|fixed|q1|G98304 | M32_N64_w8:vec2:n_first:fixed:ONE_FULL|q1|G98304|K1 | unavailable | 0.004 [-0.004,0.011] | validated |
| M3|M32|N64|W8|vec2|fixed|q2|G32768 | M32_N64_w8:vec2:n_first:fixed:ONE_FULL|q2|G32768|K1 | unavailable | -0.005 [-0.035,0.024] | validated |
| M3|M32|N64|W8|vec2|fixed|q2|G65536 | M32_N64_w8:vec2:n_first:fixed:ONE_FULL|q2|G65536|K1 | unavailable | -0.004 [-0.024,0.018] | validated |
| M3|M32|N64|W8|vec2|fixed|q2|G98304 | M32_N64_w8:vec2:n_first:fixed:ONE_FULL|q2|G98304|K1 | unavailable | 0.001 [-0.013,0.014] | validated |
| M3|M32|N64|W8|vec2|fixed|q4|G32768 | M32_N64_w8:vec2:n_first:fixed:ONE_FULL|q4|G32768|K1 | unavailable | -0.059 [-0.138,0.015] | validated |
| M3|M32|N64|W8|vec2|fixed|q4|G65536 | M32_N64_w8:vec2:n_first:fixed:ONE_FULL|q4|G65536|K1 | unavailable | -0.018 [-0.070,0.036] | validated |
| M3|M32|N64|W8|vec2|fixed|q4|G98304 | M32_N64_w8:vec2:n_first:fixed:ONE_FULL|q4|G98304|K1 | unavailable | -0.034 [-0.094,0.023] | validated |
| M3|M32|N64|W8|vec2|fixed|qnatural|G32768 | M32_N64_w8:vec2:n_first:fixed:ONE_FULL|qnatural|G32768|K1 | unavailable | 0.075 [-0.616,0.689] | validated |
| M3|M32|N64|W8|vec2|fixed|qnatural|G65536 | M32_N64_w8:vec2:n_first:fixed:ONE_FULL|qnatural|G65536|K1 | unavailable | 0.022 [-0.013,0.058] | validated |
| M3|M32|N64|W8|vec2|fixed|qnatural|G98304 | M32_N64_w8:vec2:n_first:fixed:ONE_FULL|qnatural|G98304|K1 | unavailable | -0.031 [-0.226,0.145] | validated |
| M3|M32|N64|W8|vec2|native|q1|G32768 | M32_N64_w8:vec2:n_first:native:ONE_FULL|q1|G32768|K1 | unavailable | -0.002 [-0.017,0.013] | validated |
| M3|M32|N64|W8|vec2|native|q1|G65536 | M32_N64_w8:vec2:n_first:native:ONE_FULL|q1|G65536|K1 | unavailable | -0.000 [-0.008,0.008] | validated |
| M3|M32|N64|W8|vec2|native|q1|G98304 | M32_N64_w8:vec2:n_first:native:ONE_FULL|q1|G98304|K1 | unavailable | -0.004 [-0.010,0.003] | validated |
| M3|M32|N64|W8|vec2|native|q2|G32768 | M32_N64_w8:vec2:n_first:native:ONE_FULL|q2|G32768|K1 | unavailable | -0.031 [-0.096,0.032] | validated |
| M3|M32|N64|W8|vec2|native|q2|G65536 | M32_N64_w8:vec2:n_first:native:ONE_FULL|q2|G65536|K1 | unavailable | 0.004 [-0.018,0.026] | validated |
| M3|M32|N64|W8|vec2|native|q2|G98304 | M32_N64_w8:vec2:n_first:native:ONE_FULL|q2|G98304|K1 | unavailable | 0.009 [-0.009,0.027] | validated |
| M3|M32|N64|W8|vec2|native|q4|G32768 | M32_N64_w8:vec2:n_first:native:ONE_FULL|q4|G32768|K1 | unavailable | -0.063 [-0.167,0.026] | validated |
| M3|M32|N64|W8|vec2|native|q4|G65536 | M32_N64_w8:vec2:n_first:native:ONE_FULL|q4|G65536|K1 | unavailable | -0.012 [-0.094,0.069] | validated |
| M3|M32|N64|W8|vec2|native|q4|G98304 | M32_N64_w8:vec2:n_first:native:ONE_FULL|q4|G98304|K1 | unavailable | -0.004 [-0.049,0.045] | validated |
| M3|M32|N64|W8|vec2|native|qnatural|G32768 | M32_N64_w8:vec2:n_first:native:ONE_FULL|qnatural|G32768|K1 | unavailable | -0.087 [-0.608,0.395] | validated |
| M3|M32|N64|W8|vec2|native|qnatural|G65536 | M32_N64_w8:vec2:n_first:native:ONE_FULL|qnatural|G65536|K1 | unavailable | 0.002 [-0.017,0.022] | validated |
| M3|M32|N64|W8|vec2|native|qnatural|G98304 | M32_N64_w8:vec2:n_first:native:ONE_FULL|qnatural|G98304|K1 | unavailable | -0.044 [-0.297,0.195] | validated |
| M3|M32|N64|W8|vec4|fixed|q1|G32768 | M32_N64_w8:vec4:n_first:fixed:ONE_FULL|q1|G32768|K1 | unavailable | -0.002 [-0.017,0.015] | validated |
| M3|M32|N64|W8|vec4|fixed|q1|G65536 | M32_N64_w8:vec4:n_first:fixed:ONE_FULL|q1|G65536|K1 | unavailable | 0.002 [-0.007,0.008] | validated |
| M3|M32|N64|W8|vec4|fixed|q1|G98304 | M32_N64_w8:vec4:n_first:fixed:ONE_FULL|q1|G98304|K1 | unavailable | -0.002 [-0.008,0.007] | validated |
| M3|M32|N64|W8|vec4|fixed|q2|G32768 | M32_N64_w8:vec4:n_first:fixed:ONE_FULL|q2|G32768|K1 | unavailable | 0.006 [-0.015,0.028] | validated |
| M3|M32|N64|W8|vec4|fixed|q2|G65536 | M32_N64_w8:vec4:n_first:fixed:ONE_FULL|q2|G65536|K1 | unavailable | -0.003 [-0.018,0.010] | validated |
| M3|M32|N64|W8|vec4|fixed|q2|G98304 | M32_N64_w8:vec4:n_first:fixed:ONE_FULL|q2|G98304|K1 | unavailable | -0.007 [-0.018,0.005] | validated |
| M3|M32|N64|W8|vec4|fixed|q4|G32768 | M32_N64_w8:vec4:n_first:fixed:ONE_FULL|q4|G32768|K1 | unavailable | -0.006 [-0.068,0.060] | validated |
| M3|M32|N64|W8|vec4|fixed|q4|G65536 | M32_N64_w8:vec4:n_first:fixed:ONE_FULL|q4|G65536|K1 | unavailable | 0.012 [-0.031,0.057] | validated |
| M3|M32|N64|W8|vec4|fixed|q4|G98304 | M32_N64_w8:vec4:n_first:fixed:ONE_FULL|q4|G98304|K1 | unavailable | 0.004 [-0.039,0.040] | validated |
| M3|M32|N64|W8|vec4|fixed|qnatural|G32768 | M32_N64_w8:vec4:n_first:fixed:ONE_FULL|qnatural|G32768|K1 | unavailable | -0.278 [-0.719,0.095] | validated |
| M3|M32|N64|W8|vec4|fixed|qnatural|G65536 | M32_N64_w8:vec4:n_first:fixed:ONE_FULL|qnatural|G65536|K1 | unavailable | -0.012 [-0.138,0.096] | validated |
| M3|M32|N64|W8|vec4|fixed|qnatural|G98304 | M32_N64_w8:vec4:n_first:fixed:ONE_FULL|qnatural|G98304|K1 | unavailable | -0.001 [-0.056,0.053] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q1|G32768 | M32_N64_w8:vec4:n_first:legacy_fixed:ONE_FULL|q1|G32768|K1 | unavailable | -0.006 [-0.020,0.009] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q1|G65536 | M32_N64_w8:vec4:n_first:legacy_fixed:ONE_FULL|q1|G65536|K1 | unavailable | 0.002 [-0.007,0.012] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q1|G98304 | M32_N64_w8:vec4:n_first:legacy_fixed:ONE_FULL|q1|G98304|K1 | unavailable | 0.003 [-0.003,0.010] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q2|G32768 | M32_N64_w8:vec4:n_first:legacy_fixed:ONE_FULL|q2|G32768|K1 | unavailable | -0.012 [-0.035,0.014] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q2|G65536 | M32_N64_w8:vec4:n_first:legacy_fixed:ONE_FULL|q2|G65536|K1 | unavailable | -0.006 [-0.025,0.013] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q2|G98304 | M32_N64_w8:vec4:n_first:legacy_fixed:ONE_FULL|q2|G98304|K1 | unavailable | -0.001 [-0.008,0.005] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q4|G32768 | M32_N64_w8:vec4:n_first:legacy_fixed:ONE_FULL|q4|G32768|K1 | unavailable | 0.014 [-0.032,0.069] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q4|G65536 | M32_N64_w8:vec4:n_first:legacy_fixed:ONE_FULL|q4|G65536|K1 | unavailable | 0.002 [-0.034,0.040] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q4|G98304 | M32_N64_w8:vec4:n_first:legacy_fixed:ONE_FULL|q4|G98304|K1 | unavailable | 0.001 [-0.037,0.038] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|qnatural|G32768 | M32_N64_w8:vec4:n_first:legacy_fixed:ONE_FULL|qnatural|G32768|K1 | unavailable | -0.122 [-0.399,0.192] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|qnatural|G65536 | M32_N64_w8:vec4:n_first:legacy_fixed:ONE_FULL|qnatural|G65536|K1 | unavailable | 0.011 [-0.067,0.088] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|qnatural|G98304 | M32_N64_w8:vec4:n_first:legacy_fixed:ONE_FULL|qnatural|G98304|K1 | unavailable | 0.018 [-0.035,0.069] | validated |
| M3|M32|N64|W8|vec4|native|q1|G32768 | M32_N64_w8:vec4:n_first:native:ONE_FULL|q1|G32768|K1 | unavailable | -0.005 [-0.029,0.017] | validated |
| M3|M32|N64|W8|vec4|native|q1|G65536 | M32_N64_w8:vec4:n_first:native:ONE_FULL|q1|G65536|K1 | unavailable | -0.004 [-0.014,0.005] | validated |
| M3|M32|N64|W8|vec4|native|q1|G98304 | M32_N64_w8:vec4:n_first:native:ONE_FULL|q1|G98304|K1 | unavailable | -0.001 [-0.007,0.005] | validated |
| M3|M32|N64|W8|vec4|native|q2|G32768 | M32_N64_w8:vec4:n_first:native:ONE_FULL|q2|G32768|K1 | unavailable | 0.000 [-0.020,0.019] | validated |
| M3|M32|N64|W8|vec4|native|q2|G65536 | M32_N64_w8:vec4:n_first:native:ONE_FULL|q2|G65536|K1 | unavailable | 0.010 [-0.013,0.029] | validated |
| M3|M32|N64|W8|vec4|native|q2|G98304 | M32_N64_w8:vec4:n_first:native:ONE_FULL|q2|G98304|K1 | unavailable | 0.009 [-0.001,0.020] | validated |
| M3|M32|N64|W8|vec4|native|q4|G32768 | M32_N64_w8:vec4:n_first:native:ONE_FULL|q4|G32768|K1 | unavailable | 0.075 [0.015,0.131] | validated |
| M3|M32|N64|W8|vec4|native|q4|G65536 | M32_N64_w8:vec4:n_first:native:ONE_FULL|q4|G65536|K1 | unavailable | -0.039 [-0.088,0.013] | validated |
| M3|M32|N64|W8|vec4|native|q4|G98304 | M32_N64_w8:vec4:n_first:native:ONE_FULL|q4|G98304|K1 | unavailable | 0.025 [-0.014,0.061] | validated |
| M3|M32|N64|W8|vec4|native|qnatural|G32768 | M32_N64_w8:vec4:n_first:native:ONE_FULL|qnatural|G32768|K1 | unavailable | 0.059 [-0.258,0.369] | validated |
| M3|M32|N64|W8|vec4|native|qnatural|G65536 | M32_N64_w8:vec4:n_first:native:ONE_FULL|qnatural|G65536|K1 | unavailable | 0.012 [-0.035,0.061] | validated |
| M3|M32|N64|W8|vec4|native|qnatural|G98304 | M32_N64_w8:vec4:n_first:native:ONE_FULL|qnatural|G98304|K1 | unavailable | -0.011 [-0.066,0.047] | validated |
| M3|M32|N64|W8|vec8|fixed|q1|G32768 | M32_N64_w8:vec8:n_first:fixed:ONE_FULL|q1|G32768|K1 | unavailable | 0.004 [-0.007,0.014] | validated |
| M3|M32|N64|W8|vec8|fixed|q1|G65536 | M32_N64_w8:vec8:n_first:fixed:ONE_FULL|q1|G65536|K1 | unavailable | -0.001 [-0.008,0.005] | validated |
| M3|M32|N64|W8|vec8|fixed|q1|G98304 | M32_N64_w8:vec8:n_first:fixed:ONE_FULL|q1|G98304|K1 | unavailable | -0.000 [-0.004,0.004] | validated |
| M3|M32|N64|W8|vec8|fixed|q2|G32768 | M32_N64_w8:vec8:n_first:fixed:ONE_FULL|q2|G32768|K1 | unavailable | -0.003 [-0.032,0.027] | validated |
| M3|M32|N64|W8|vec8|fixed|q2|G65536 | M32_N64_w8:vec8:n_first:fixed:ONE_FULL|q2|G65536|K1 | unavailable | 0.017 [0.002,0.032] | validated |
| M3|M32|N64|W8|vec8|fixed|q2|G98304 | M32_N64_w8:vec8:n_first:fixed:ONE_FULL|q2|G98304|K1 | unavailable | -0.002 [-0.017,0.012] | validated |
| M3|M32|N64|W8|vec8|fixed|q4|G32768 | M32_N64_w8:vec8:n_first:fixed:ONE_FULL|q4|G32768|K1 | unavailable | 0.013 [-0.024,0.052] | validated |
| M3|M32|N64|W8|vec8|fixed|q4|G65536 | M32_N64_w8:vec8:n_first:fixed:ONE_FULL|q4|G65536|K1 | unavailable | 0.007 [-0.015,0.030] | validated |
| M3|M32|N64|W8|vec8|fixed|q4|G98304 | M32_N64_w8:vec8:n_first:fixed:ONE_FULL|q4|G98304|K1 | unavailable | 0.006 [-0.010,0.021] | validated |
| M3|M32|N64|W8|vec8|fixed|qnatural|G32768 | M32_N64_w8:vec8:n_first:fixed:ONE_FULL|qnatural|G32768|K1 | unavailable | -0.013 [-0.042,0.014] | validated |
| M3|M32|N64|W8|vec8|fixed|qnatural|G65536 | M32_N64_w8:vec8:n_first:fixed:ONE_FULL|qnatural|G65536|K1 | unavailable | -0.000 [-0.011,0.015] | validated |
| M3|M32|N64|W8|vec8|fixed|qnatural|G98304 | M32_N64_w8:vec8:n_first:fixed:ONE_FULL|qnatural|G98304|K1 | unavailable | -0.000 [-0.010,0.011] | validated |
| M3|M32|N64|W8|vec8|native|q1|G32768 | M32_N64_w8:vec8:n_first:native:ONE_FULL|q1|G32768|K1 | unavailable | 0.001 [-0.012,0.012] | validated |
| M3|M32|N64|W8|vec8|native|q1|G65536 | M32_N64_w8:vec8:n_first:native:ONE_FULL|q1|G65536|K1 | unavailable | -0.002 [-0.009,0.006] | validated |
| M3|M32|N64|W8|vec8|native|q1|G98304 | M32_N64_w8:vec8:n_first:native:ONE_FULL|q1|G98304|K1 | unavailable | -0.005 [-0.009,-0.001] | validated |
| M3|M32|N64|W8|vec8|native|q2|G32768 | M32_N64_w8:vec8:n_first:native:ONE_FULL|q2|G32768|K1 | unavailable | -0.019 [-0.040,0.003] | validated |
| M3|M32|N64|W8|vec8|native|q2|G65536 | M32_N64_w8:vec8:n_first:native:ONE_FULL|q2|G65536|K1 | unavailable | 0.007 [-0.015,0.026] | validated |
| M3|M32|N64|W8|vec8|native|q2|G98304 | M32_N64_w8:vec8:n_first:native:ONE_FULL|q2|G98304|K1 | unavailable | 0.004 [-0.011,0.019] | validated |
| M3|M32|N64|W8|vec8|native|q4|G32768 | M32_N64_w8:vec8:n_first:native:ONE_FULL|q4|G32768|K1 | unavailable | -0.004 [-0.053,0.046] | validated |
| M3|M32|N64|W8|vec8|native|q4|G65536 | M32_N64_w8:vec8:n_first:native:ONE_FULL|q4|G65536|K1 | unavailable | -0.020 [-0.061,0.015] | validated |
| M3|M32|N64|W8|vec8|native|q4|G98304 | M32_N64_w8:vec8:n_first:native:ONE_FULL|q4|G98304|K1 | unavailable | -0.024 [-0.066,0.009] | validated |
| M3|M32|N64|W8|vec8|native|qnatural|G32768 | M32_N64_w8:vec8:n_first:native:ONE_FULL|qnatural|G32768|K1 | unavailable | 0.015 [-0.030,0.064] | validated |
| M3|M32|N64|W8|vec8|native|qnatural|G65536 | M32_N64_w8:vec8:n_first:native:ONE_FULL|qnatural|G65536|K1 | unavailable | 0.013 [-0.011,0.037] | validated |
| M3|M32|N64|W8|vec8|native|qnatural|G98304 | M32_N64_w8:vec8:n_first:native:ONE_FULL|qnatural|G98304|K1 | unavailable | 0.004 [-0.012,0.020] | validated |

## Every context

The minimum candidate below is selected retrospectively for a compact inventory, not used as a deployed model. Full candidate and losing-layout data remain in the CSV/JSON.

| Context | Physical rows including sham | Descriptive minimum candidate | Target time µs [CI95] | Gain vs paired reference % [CI95] | Control |
|---|---:|---|---:|---:|---|
| M3|M32|N128|W4|control|control|q1|G16384 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N128|W4|control|control|q1|G32768 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N128|W4|control|control|q1|G49152 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N128|W4|control|control|q2|G16384 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N128|W4|control|control|q2|G32768 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N128|W4|control|control|q2|G49152 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N128|W4|control|control|q4|G16384 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N128|W4|control|control|q4|G32768 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N128|W4|control|control|q4|G49152 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N128|W4|control|control|q8|G16384 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N128|W4|control|control|q8|G32768 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N128|W4|control|control|q8|G49152 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N128|W4|control|control|qnatural|G16384 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N128|W4|control|control|qnatural|G32768 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N128|W4|control|control|qnatural|G49152 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N128|W4|vec2|fixed|q1|G16384 | 3 | M32_N128_w4:vec2:within:fixed:ONE_FULL|q1|G16384|K1 | 129.354 [129.170,129.675] | 3.645 [3.592,3.691] | validated |
| M3|M32|N128|W4|vec2|fixed|q1|G32768 | 3 | M32_N128_w4:vec2:within:fixed:ONE_FULL|q1|G32768|K1 | 253.102 [252.737,253.762] | 3.729 [3.684,3.775] | validated |
| M3|M32|N128|W4|vec2|fixed|q1|G49152 | 3 | M32_N128_w4:vec2:within:fixed:ONE_FULL|q1|G49152|K1 | 377.273 [376.871,378.017] | 3.547 [3.526,3.572] | validated |
| M3|M32|N128|W4|vec2|fixed|q2|G16384 | 3 | M32_N128_w4:vec2:within:fixed:ONE_FULL|q2|G16384|K1 | 78.252 [78.151,78.336] | 2.008 [1.896,2.119] | validated |
| M3|M32|N128|W4|vec2|fixed|q2|G32768 | 3 | M32_N128_w4:vec2:within:fixed:ONE_FULL|q2|G32768|K1 | 149.028 [148.924,149.113] | 2.498 [2.395,2.604] | validated |
| M3|M32|N128|W4|vec2|fixed|q2|G49152 | 3 | M32_N128_w4:vec2:within:fixed:ONE_FULL|q2|G49152|K1 | 221.093 [220.884,221.378] | 2.405 [2.361,2.447] | validated |
| M3|M32|N128|W4|vec2|fixed|q4|G16384 | 3 | M32_N128_w4:vec2:within:fixed:ONE_FULL|q4|G16384|K1 | 55.916 [55.678,56.133] | 0.321 [-0.240,0.904] | validated |
| M3|M32|N128|W4|vec2|fixed|q4|G32768 | 3 | M32_N128_w4:vec2:within:fixed:ONE_FULL|q4|G32768|K1 | 105.472 [105.438,105.506] | 0.222 [0.203,0.241] | validated |
| M3|M32|N128|W4|vec2|fixed|q4|G49152 | 3 | M32_N128_w4:vec2:within:fixed:ONE_FULL|q4|G49152|K1 | 155.170 [154.964,155.373] | 0.215 [-0.027,0.447] | validated |
| M3|M32|N128|W4|vec2|fixed|q8|G16384 | 3 | M32_N128_w4:vec2:within:fixed:ONE_FULL|q8|G16384|K1 | 52.294 [51.941,52.671] | -0.926 [-2.016,0.131] | validated |
| M3|M32|N128|W4|vec2|fixed|q8|G32768 | 3 | M32_N128_w4:vec2:within:fixed:ONE_FULL|q8|G32768|K1 | 98.582 [98.513,98.635] | 0.006 [-0.017,0.029] | validated |
| M3|M32|N128|W4|vec2|fixed|q8|G49152 | 3 | M32_N128_w4:vec2:within:fixed:ONE_FULL|q8|G49152|K1 | 144.542 [144.297,144.837] | -0.150 [-0.371,0.086] | validated |
| M3|M32|N128|W4|vec2|fixed|qnatural|G16384 | 3 | M32_N128_w4:vec2:within:fixed:ONE_FULL|qnatural|G16384|K1 | 52.678 [52.453,52.907] | -0.497 [-1.259,0.200] | validated |
| M3|M32|N128|W4|vec2|fixed|qnatural|G32768 | 3 | M32_N128_w4:vec2:within:fixed:ONE_FULL|qnatural|G32768|K1 | 99.261 [99.190,99.306] | -0.151 [-0.191,-0.112] | validated |
| M3|M32|N128|W4|vec2|fixed|qnatural|G49152 | 3 | M32_N128_w4:vec2:within:fixed:ONE_FULL|qnatural|G49152|K1 | 145.482 [145.298,145.722] | -0.115 [-0.252,0.031] | validated |
| M3|M32|N128|W4|vec2|native|q1|G16384 | 3 | M32_N128_w4:vec2:within:native:ONE_FULL|q1|G16384|K1 | 127.878 [127.683,128.213] | 2.395 [2.362,2.432] | validated |
| M3|M32|N128|W4|vec2|native|q1|G32768 | 3 | M32_N128_w4:vec2:within:native:ONE_FULL|q1|G32768|K1 | 249.918 [249.581,250.465] | 2.503 [2.474,2.530] | validated |
| M3|M32|N128|W4|vec2|native|q1|G49152 | 3 | M32_N128_w4:vec2:within:native:ONE_FULL|q1|G49152|K1 | 372.063 [371.687,372.752] | 2.536 [2.488,2.597] | validated |
| M3|M32|N128|W4|vec2|native|q2|G16384 | 3 | M32_N128_w4:vec2:within:native:ONE_FULL|q2|G16384|K1 | 77.312 [77.196,77.452] | 1.167 [1.060,1.292] | validated |
| M3|M32|N128|W4|vec2|native|q2|G32768 | 3 | M32_N128_w4:vec2:within:native:ONE_FULL|q2|G32768|K1 | 147.250 [147.139,147.366] | 1.515 [1.423,1.593] | validated |
| M3|M32|N128|W4|vec2|native|q2|G49152 | 3 | M32_N128_w4:vec2:within:native:ONE_FULL|q2|G49152|K1 | 218.259 [218.034,218.571] | 1.402 [1.326,1.473] | validated |
| M3|M32|N128|W4|vec2|native|q4|G16384 | 3 | M32_N128_w4:vec2:within:native:ONE_FULL|q4|G16384|K1 | 55.814 [55.421,56.153] | -0.019 [-0.662,0.684] | validated |
| M3|M32|N128|W4|vec2|native|q4|G32768 | 3 | M32_N128_w4:vec2:within:native:ONE_FULL|q4|G32768|K1 | 105.021 [104.997,105.047] | 0.245 [0.203,0.280] | validated |
| M3|M32|N128|W4|vec2|native|q4|G49152 | 3 | M32_N128_w4:vec2:within:native:ONE_FULL|q4|G49152|K1 | 154.490 [154.243,154.731] | 0.176 [-0.106,0.438] | validated |
| M3|M32|N128|W4|vec2|native|q8|G16384 | 3 | M32_N128_w4:vec2:within:native:ONE_FULL|q8|G16384|K1 | 52.222 [51.784,52.732] | -0.456 [-1.883,0.877] | validated |
| M3|M32|N128|W4|vec2|native|q8|G32768 | 3 | M32_N128_w4:vec2:within:native:ONE_FULL|q8|G32768|K1 | 98.674 [98.630,98.709] | -0.096 [-0.138,-0.056] | validated |
| M3|M32|N128|W4|vec2|native|q8|G49152 | 3 | M32_N128_w4:vec2:within:native:ONE_FULL|q8|G49152|K1 | 144.372 [144.159,144.605] | 0.215 [-0.018,0.446] | validated |
| M3|M32|N128|W4|vec2|native|qnatural|G16384 | 3 | M32_N128_w4:vec2:within:native:ONE_FULL|qnatural|G16384|K1 | 52.590 [52.371,52.801] | -0.045 [-0.637,0.545] | validated |
| M3|M32|N128|W4|vec2|native|qnatural|G32768 | 3 | M32_N128_w4:vec2:within:native:ONE_FULL|qnatural|G32768|K1 | 99.134 [99.077,99.201] | 0.002 [-0.027,0.030] | validated |
| M3|M32|N128|W4|vec2|native|qnatural|G49152 | 3 | M32_N128_w4:vec2:within:native:ONE_FULL|qnatural|G49152|K1 | 145.279 [145.145,145.454] | 0.037 [-0.164,0.276] | validated |
| M3|M32|N128|W4|vec4|fixed|q1|G16384 | 3 | M32_N128_w4:vec4:within:fixed:ONE_FULL|q1|G16384|K1 | 131.697 [131.465,132.089] | 4.113 [4.071,4.161] | validated |
| M3|M32|N128|W4|vec4|fixed|q1|G32768 | 3 | M32_N128_w4:vec4:within:fixed:ONE_FULL|q1|G32768|K1 | 257.568 [257.249,258.091] | 4.267 [4.245,4.289] | validated |
| M3|M32|N128|W4|vec4|fixed|q1|G49152 | 3 | M32_N128_w4:vec4:within:fixed:ONE_FULL|q1|G49152|K1 | 383.573 [383.102,384.421] | 4.325 [4.300,4.360] | validated |
| M3|M32|N128|W4|vec4|fixed|q2|G16384 | 3 | M32_N128_w4:vec4:within:fixed:ONE_FULL|q2|G16384|K1 | 78.935 [78.780,79.142] | 3.059 [2.969,3.143] | validated |
| M3|M32|N128|W4|vec4|fixed|q2|G32768 | 3 | M32_N128_w4:vec4:within:fixed:ONE_FULL|q2|G32768|K1 | 150.604 [150.397,150.806] | 3.233 [3.169,3.297] | validated |
| M3|M32|N128|W4|vec4|fixed|q2|G49152 | 3 | M32_N128_w4:vec4:within:fixed:ONE_FULL|q2|G49152|K1 | 223.257 [222.977,223.596] | 3.212 [3.155,3.270] | validated |
| M3|M32|N128|W4|vec4|fixed|q4|G16384 | 3 | M32_N128_w4:vec4:within:fixed:ONE_FULL|q4|G16384|K1 | 56.140 [55.853,56.427] | 0.375 [-0.163,0.963] | validated |
| M3|M32|N128|W4|vec4|fixed|q4|G32768 | 3 | M32_N128_w4:vec4:within:fixed:ONE_FULL|q4|G32768|K1 | 105.858 [105.807,105.920] | 0.520 [0.502,0.538] | validated |
| M3|M32|N128|W4|vec4|fixed|q4|G49152 | 3 | M32_N128_w4:vec4:within:fixed:ONE_FULL|q4|G49152|K1 | 155.659 [155.388,155.908] | 0.766 [0.563,0.969] | validated |
| M3|M32|N128|W4|vec4|fixed|q8|G16384 | 3 | M32_N128_w4:vec4:within:fixed:ONE_FULL|q8|G16384|K1 | 52.371 [51.996,52.741] | -0.098 [-1.066,0.797] | validated |
| M3|M32|N128|W4|vec4|fixed|q8|G32768 | 3 | M32_N128_w4:vec4:within:fixed:ONE_FULL|q8|G32768|K1 | 98.667 [98.570,98.742] | -0.039 [-0.059,-0.017] | validated |
| M3|M32|N128|W4|vec4|fixed|q8|G49152 | 3 | M32_N128_w4:vec4:within:fixed:ONE_FULL|q8|G49152|K1 | 144.298 [144.149,144.498] | 0.041 [-0.159,0.278] | validated |
| M3|M32|N128|W4|vec4|fixed|qnatural|G16384 | 3 | M32_N128_w4:vec4:within:fixed:ONE_FULL|qnatural|G16384|K1 | 52.817 [52.632,53.000] | 0.454 [0.074,0.963] | validated |
| M3|M32|N128|W4|vec4|fixed|qnatural|G32768 | 3 | M32_N128_w4:vec4:within:fixed:ONE_FULL|qnatural|G32768|K1 | 99.396 [99.355,99.449] | -0.023 [-0.058,0.014] | validated |
| M3|M32|N128|W4|vec4|fixed|qnatural|G49152 | 3 | M32_N128_w4:vec4:within:fixed:ONE_FULL|qnatural|G49152|K1 | 145.636 [145.412,145.900] | -0.131 [-0.285,0.018] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q1|G16384 | 3 | M32_N128_w4:vec4:within:legacy_fixed:ONE_FULL|q1|G16384|K1 | 131.735 [131.527,132.057] | 4.827 [4.751,4.884] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q1|G32768 | 3 | M32_N128_w4:vec4:within:legacy_fixed:ONE_FULL|q1|G32768|K1 | 257.639 [257.321,258.171] | 5.060 [5.041,5.078] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q1|G49152 | 3 | M32_N128_w4:vec4:within:legacy_fixed:ONE_FULL|q1|G49152|K1 | 383.639 [383.226,384.380] | 5.083 [5.054,5.108] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q2|G16384 | 3 | M32_N128_w4:vec4:within:legacy_fixed:ONE_FULL|q2|G16384|K1 | 78.933 [78.785,79.045] | 3.532 [3.418,3.629] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q2|G32768 | 3 | M32_N128_w4:vec4:within:legacy_fixed:ONE_FULL|q2|G32768|K1 | 150.503 [150.342,150.662] | 3.968 [3.838,4.092] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q2|G49152 | 3 | M32_N128_w4:vec4:within:legacy_fixed:ONE_FULL|q2|G49152|K1 | 223.156 [222.898,223.435] | 3.947 [3.881,4.018] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q4|G16384 | 3 | M32_N128_w4:vec4:within:legacy_fixed:ONE_FULL|q4|G16384|K1 | 55.935 [55.697,56.166] | 1.213 [0.664,1.717] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q4|G32768 | 3 | M32_N128_w4:vec4:within:legacy_fixed:ONE_FULL|q4|G32768|K1 | 105.763 [105.693,105.851] | 0.861 [0.824,0.893] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q4|G49152 | 3 | M32_N128_w4:vec4:within:legacy_fixed:ONE_FULL|q4|G49152|K1 | 155.897 [155.710,156.075] | 0.812 [0.607,1.001] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q8|G16384 | 3 | M32_N128_w4:vec4:within:legacy_fixed:ONE_FULL|q8|G16384|K1 | 52.214 [51.828,52.645] | 0.342 [-1.105,1.449] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q8|G32768 | 3 | M32_N128_w4:vec4:within:legacy_fixed:ONE_FULL|q8|G32768|K1 | 98.589 [98.505,98.704] | 0.152 [0.107,0.193] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|q8|G49152 | 3 | M32_N128_w4:vec4:within:legacy_fixed:ONE_FULL|q8|G49152|K1 | 144.390 [144.109,144.723] | -0.006 [-0.268,0.260] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|qnatural|G16384 | 3 | M32_N128_w4:vec4:within:legacy_fixed:ONE_FULL|qnatural|G16384|K1 | 52.783 [52.606,52.970] | 0.346 [-0.321,1.059] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|qnatural|G32768 | 3 | M32_N128_w4:vec4:within:legacy_fixed:ONE_FULL|qnatural|G32768|K1 | 99.430 [99.337,99.517] | -0.005 [-0.039,0.029] | validated |
| M3|M32|N128|W4|vec4|legacy_fixed|qnatural|G49152 | 3 | M32_N128_w4:vec4:within:legacy_fixed:ONE_FULL|qnatural|G49152|K1 | 145.657 [145.476,145.858] | -0.038 [-0.172,0.089] | validated |
| M3|M32|N128|W4|vec4|native|q1|G16384 | 3 | M32_N128_w4:vec4:within:native:ONE_FULL|q1|G16384|K1 | 130.028 [129.856,130.325] | 4.286 [4.231,4.333] | validated |
| M3|M32|N128|W4|vec4|native|q1|G32768 | 3 | M32_N128_w4:vec4:within:native:ONE_FULL|q1|G32768|K1 | 254.331 [254.042,254.841] | 4.251 [4.230,4.270] | validated |
| M3|M32|N128|W4|vec4|native|q1|G49152 | 3 | M32_N128_w4:vec4:within:native:ONE_FULL|q1|G49152|K1 | 378.672 [378.192,379.542] | 4.209 [4.179,4.245] | validated |
| M3|M32|N128|W4|vec4|native|q2|G16384 | 3 | M32_N128_w4:vec4:within:native:ONE_FULL|q2|G16384|K1 | 78.183 [78.087,78.272] | 2.893 [2.793,2.999] | validated |
| M3|M32|N128|W4|vec4|native|q2|G32768 | 3 | M32_N128_w4:vec4:within:native:ONE_FULL|q2|G32768|K1 | 148.933 [148.855,149.005] | 3.231 [3.110,3.361] | validated |
| M3|M32|N128|W4|vec4|native|q2|G49152 | 3 | M32_N128_w4:vec4:within:native:ONE_FULL|q2|G49152|K1 | 220.721 [220.422,221.162] | 3.232 [3.158,3.301] | validated |
| M3|M32|N128|W4|vec4|native|q4|G16384 | 3 | M32_N128_w4:vec4:within:native:ONE_FULL|q4|G16384|K1 | 55.763 [55.494,56.011] | 1.109 [0.530,1.761] | validated |
| M3|M32|N128|W4|vec4|native|q4|G32768 | 3 | M32_N128_w4:vec4:within:native:ONE_FULL|q4|G32768|K1 | 105.540 [105.495,105.598] | 0.544 [0.507,0.584] | validated |
| M3|M32|N128|W4|vec4|native|q4|G49152 | 3 | M32_N128_w4:vec4:within:native:ONE_FULL|q4|G49152|K1 | 155.109 [154.906,155.333] | 0.714 [0.412,0.994] | validated |
| M3|M32|N128|W4|vec4|native|q8|G16384 | 3 | M32_N128_w4:vec4:within:native:ONE_FULL|q8|G16384|K1 | 52.047 [51.750,52.386] | 0.296 [-0.676,1.268] | validated |
| M3|M32|N128|W4|vec4|native|q8|G32768 | 3 | M32_N128_w4:vec4:within:native:ONE_FULL|q8|G32768|K1 | 98.615 [98.561,98.673] | 0.133 [0.099,0.164] | validated |
| M3|M32|N128|W4|vec4|native|q8|G49152 | 3 | M32_N128_w4:vec4:within:native:ONE_FULL|q8|G49152|K1 | 144.393 [144.217,144.602] | -0.064 [-0.246,0.128] | validated |
| M3|M32|N128|W4|vec4|native|qnatural|G16384 | 3 | M32_N128_w4:vec4:within:native:ONE_FULL|qnatural|G16384|K1 | 52.736 [52.561,52.917] | -0.099 [-0.671,0.462] | validated |
| M3|M32|N128|W4|vec4|native|qnatural|G32768 | 3 | M32_N128_w4:vec4:within:native:ONE_FULL|qnatural|G32768|K1 | 99.343 [99.299,99.405] | -0.087 [-0.116,-0.059] | validated |
| M3|M32|N128|W4|vec4|native|qnatural|G49152 | 3 | M32_N128_w4:vec4:within:native:ONE_FULL|qnatural|G49152|K1 | 145.498 [145.349,145.663] | -0.078 [-0.234,0.088] | validated |
| M3|M32|N128|W4|vec8|fixed|q1|G16384 | 3 | M32_N128_w4:vec8:within:fixed:ONE_FULL|q1|G16384|K1 | 136.652 [136.472,136.902] | 7.752 [7.727,7.773] | validated |
| M3|M32|N128|W4|vec8|fixed|q1|G32768 | 3 | M32_N128_w4:vec8:within:fixed:ONE_FULL|q1|G32768|K1 | 267.716 [267.414,268.270] | 7.864 [7.849,7.879] | validated |
| M3|M32|N128|W4|vec8|fixed|q1|G49152 | 3 | M32_N128_w4:vec8:within:fixed:ONE_FULL|q1|G49152|K1 | 398.733 [398.261,399.622] | 7.855 [7.829,7.878] | validated |
| M3|M32|N128|W4|vec8|fixed|q2|G16384 | 3 | M32_N128_w4:vec8:within:fixed:ONE_FULL|q2|G16384|K1 | 82.133 [82.043,82.249] | 5.835 [5.707,5.955] | validated |
| M3|M32|N128|W4|vec8|fixed|q2|G32768 | 3 | M32_N128_w4:vec8:within:fixed:ONE_FULL|q2|G32768|K1 | 157.142 [156.930,157.411] | 5.986 [5.908,6.074] | validated |
| M3|M32|N128|W4|vec8|fixed|q2|G49152 | 3 | M32_N128_w4:vec8:within:fixed:ONE_FULL|q2|G49152|K1 | 232.991 [232.702,233.438] | 5.968 [5.881,6.034] | validated |
| M3|M32|N128|W4|vec8|fixed|q4|G16384 | 3 | M32_N128_w4:vec8:within:fixed:ONE_FULL|q4|G16384|K1 | 57.001 [56.729,57.264] | 2.496 [2.041,2.938] | validated |
| M3|M32|N128|W4|vec8|fixed|q4|G32768 | 3 | M32_N128_w4:vec8:within:fixed:ONE_FULL|q4|G32768|K1 | 107.455 [107.360,107.529] | 2.622 [2.597,2.652] | validated |
| M3|M32|N128|W4|vec8|fixed|q4|G49152 | 3 | M32_N128_w4:vec8:within:fixed:ONE_FULL|q4|G49152|K1 | 158.330 [158.092,158.553] | 2.670 [2.532,2.808] | validated |
| M3|M32|N128|W4|vec8|fixed|q8|G16384 | 3 | M32_N128_w4:vec8:within:fixed:ONE_FULL|q8|G16384|K1 | 52.351 [52.082,52.638] | 0.543 [-0.065,1.132] | validated |
| M3|M32|N128|W4|vec8|fixed|q8|G32768 | 3 | M32_N128_w4:vec8:within:fixed:ONE_FULL|q8|G32768|K1 | 98.766 [98.682,98.823] | 0.323 [0.308,0.339] | validated |
| M3|M32|N128|W4|vec8|fixed|q8|G49152 | 3 | M32_N128_w4:vec8:within:fixed:ONE_FULL|q8|G49152|K1 | 144.697 [144.417,145.020] | -0.023 [-0.318,0.269] | validated |
| M3|M32|N128|W4|vec8|fixed|qnatural|G16384 | 3 | M32_N128_w4:vec8:within:fixed:ONE_FULL|qnatural|G16384|K1 | 53.204 [53.021,53.398] | 0.772 [0.331,1.203] | validated |
| M3|M32|N128|W4|vec8|fixed|qnatural|G32768 | 3 | M32_N128_w4:vec8:within:fixed:ONE_FULL|qnatural|G32768|K1 | 99.727 [99.639,99.791] | 0.533 [0.504,0.568] | validated |
| M3|M32|N128|W4|vec8|fixed|qnatural|G49152 | 3 | M32_N128_w4:vec8:within:fixed:ONE_FULL|qnatural|G49152|K1 | 145.947 [145.798,146.128] | 0.305 [0.116,0.466] | validated |
| M3|M32|N128|W4|vec8|native|q1|G16384 | 3 | M32_N128_w4:vec8:within:native:ONE_FULL|q1|G16384|K1 | 133.877 [133.651,134.223] | 7.162 [7.075,7.243] | validated |
| M3|M32|N128|W4|vec8|native|q1|G32768 | 3 | M32_N128_w4:vec8:within:native:ONE_FULL|q1|G32768|K1 | 262.059 [261.678,262.705] | 7.321 [7.249,7.371] | validated |
| M3|M32|N128|W4|vec8|native|q1|G49152 | 3 | M32_N128_w4:vec8:within:native:ONE_FULL|q1|G49152|K1 | 390.221 [389.838,390.889] | 7.317 [7.301,7.334] | validated |
| M3|M32|N128|W4|vec8|native|q2|G16384 | 3 | M32_N128_w4:vec8:within:native:ONE_FULL|q2|G16384|K1 | 81.019 [80.942,81.077] | 5.465 [5.385,5.551] | validated |
| M3|M32|N128|W4|vec8|native|q2|G32768 | 3 | M32_N128_w4:vec8:within:native:ONE_FULL|q2|G32768|K1 | 154.918 [154.749,155.117] | 5.660 [5.588,5.729] | validated |
| M3|M32|N128|W4|vec8|native|q2|G49152 | 3 | M32_N128_w4:vec8:within:native:ONE_FULL|q2|G49152|K1 | 229.689 [229.439,229.971] | 5.641 [5.578,5.708] | validated |
| M3|M32|N128|W4|vec8|native|q4|G16384 | 3 | M32_N128_w4:vec8:within:native:ONE_FULL|q4|G16384|K1 | 56.859 [56.644,57.047] | 2.006 [1.565,2.373] | validated |
| M3|M32|N128|W4|vec8|native|q4|G32768 | 3 | M32_N128_w4:vec8:within:native:ONE_FULL|q4|G32768|K1 | 107.074 [107.054,107.095] | 2.000 [1.970,2.031] | validated |
| M3|M32|N128|W4|vec8|native|q4|G49152 | 3 | M32_N128_w4:vec8:within:native:ONE_FULL|q4|G49152|K1 | 157.612 [157.312,157.898] | 2.209 [2.064,2.337] | validated |
| M3|M32|N128|W4|vec8|native|q8|G16384 | 3 | M32_N128_w4:vec8:within:native:ONE_FULL|q8|G16384|K1 | 52.722 [52.347,53.057] | 0.147 [-1.223,1.433] | validated |
| M3|M32|N128|W4|vec8|native|q8|G32768 | 3 | M32_N128_w4:vec8:within:native:ONE_FULL|q8|G32768|K1 | 98.887 [98.780,99.066] | 0.103 [0.077,0.130] | validated |
| M3|M32|N128|W4|vec8|native|q8|G49152 | 3 | M32_N128_w4:vec8:within:native:ONE_FULL|q8|G49152|K1 | 144.338 [144.233,144.487] | 0.240 [-0.004,0.496] | validated |
| M3|M32|N128|W4|vec8|native|qnatural|G16384 | 3 | M32_N128_w4:vec8:within:native:ONE_FULL|qnatural|G16384|K1 | 53.118 [52.868,53.371] | 0.708 [-0.108,1.613] | validated |
| M3|M32|N128|W4|vec8|native|qnatural|G32768 | 3 | M32_N128_w4:vec8:within:native:ONE_FULL|qnatural|G32768|K1 | 99.733 [99.687,99.780] | 0.332 [0.295,0.357] | validated |
| M3|M32|N128|W4|vec8|native|qnatural|G49152 | 3 | M32_N128_w4:vec8:within:native:ONE_FULL|qnatural|G49152|K1 | 145.845 [145.725,146.009] | 0.364 [0.201,0.554] | validated |
| M3|M32|N64|W8|control|control|q1|G32768 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N64|W8|control|control|q1|G65536 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N64|W8|control|control|q1|G98304 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N64|W8|control|control|q2|G32768 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N64|W8|control|control|q2|G65536 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N64|W8|control|control|q2|G98304 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N64|W8|control|control|q4|G32768 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N64|W8|control|control|q4|G65536 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N64|W8|control|control|q4|G98304 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N64|W8|control|control|qnatural|G32768 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N64|W8|control|control|qnatural|G65536 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N64|W8|control|control|qnatural|G98304 | 1 | — | unavailable | unavailable | no layout comparison |
| M3|M32|N64|W8|vec2|fixed|q1|G32768 | 3 | M32_N64_w8:vec2:within:fixed:ONE_FULL|q1|G32768|K1 | 236.975 [236.608,237.619] | 6.358 [6.328,6.390] | validated |
| M3|M32|N64|W8|vec2|fixed|q1|G65536 | 3 | M32_N64_w8:vec2:within:fixed:ONE_FULL|q1|G65536|K1 | 468.394 [467.733,469.689] | 6.402 [6.386,6.416] | validated |
| M3|M32|N64|W8|vec2|fixed|q1|G98304 | 3 | M32_N64_w8:vec2:within:fixed:ONE_FULL|q1|G98304|K1 | 700.059 [698.852,702.240] | 6.360 [6.291,6.405] | validated |
| M3|M32|N64|W8|vec2|fixed|q2|G32768 | 3 | M32_N64_w8:vec2:within:fixed:ONE_FULL|q2|G32768|K1 | 128.698 [128.564,128.899] | 6.659 [6.607,6.702] | validated |
| M3|M32|N64|W8|vec2|fixed|q2|G65536 | 3 | M32_N64_w8:vec2:within:fixed:ONE_FULL|q2|G65536|K1 | 251.426 [251.089,251.969] | 6.781 [6.733,6.835] | validated |
| M3|M32|N64|W8|vec2|fixed|q2|G98304 | 3 | M32_N64_w8:vec2:within:fixed:ONE_FULL|q2|G98304|K1 | 374.276 [373.701,375.287] | 6.806 [6.781,6.829] | validated |
| M3|M32|N64|W8|vec2|fixed|q4|G32768 | 3 | M32_N64_w8:vec2:within:fixed:ONE_FULL|q4|G32768|K1 | 77.840 [77.749,77.926] | 5.914 [5.831,5.998] | validated |
| M3|M32|N64|W8|vec2|fixed|q4|G65536 | 3 | M32_N64_w8:vec2:within:fixed:ONE_FULL|q4|G65536|K1 | 148.541 [148.411,148.694] | 6.258 [6.189,6.328] | validated |
| M3|M32|N64|W8|vec2|fixed|q4|G98304 | 3 | M32_N64_w8:vec2:within:fixed:ONE_FULL|q4|G98304|K1 | 220.415 [220.254,220.637] | 6.152 [6.098,6.195] | validated |
| M3|M32|N64|W8|vec2|fixed|qnatural|G32768 | 3 | M32_N64_w8:vec2:within:fixed:ONE_FULL|q8|G32768|K1 | 57.206 [57.055,57.337] | 2.067 [1.584,2.565] | validated |
| M3|M32|N64|W8|vec2|fixed|qnatural|G65536 | 3 | M32_N64_w8:vec2:within:fixed:ONE_FULL|q8|G65536|K1 | 107.827 [107.715,107.957] | 2.373 [2.337,2.410] | validated |
| M3|M32|N64|W8|vec2|fixed|qnatural|G98304 | 3 | M32_N64_w8:vec2:within:fixed:ONE_FULL|q8|G98304|K1 | 159.117 [158.912,159.282] | 2.430 [2.256,2.600] | validated |
| M3|M32|N64|W8|vec2|native|q1|G32768 | 3 | M32_N64_w8:vec2:within:native:ONE_FULL|q1|G32768|K1 | 233.482 [233.052,234.282] | 6.138 [6.086,6.174] | validated |
| M3|M32|N64|W8|vec2|native|q1|G65536 | 3 | M32_N64_w8:vec2:within:native:ONE_FULL|q1|G65536|K1 | 461.657 [460.797,463.180] | 6.196 [6.158,6.224] | validated |
| M3|M32|N64|W8|vec2|native|q1|G98304 | 3 | M32_N64_w8:vec2:within:native:ONE_FULL|q1|G98304|K1 | 689.491 [688.341,691.488] | 6.250 [6.219,6.276] | validated |
| M3|M32|N64|W8|vec2|native|q2|G32768 | 3 | M32_N64_w8:vec2:within:native:ONE_FULL|q2|G32768|K1 | 126.946 [126.908,126.995] | 6.261 [6.152,6.415] | validated |
| M3|M32|N64|W8|vec2|native|q2|G65536 | 3 | M32_N64_w8:vec2:within:native:ONE_FULL|q2|G65536|K1 | 248.213 [247.938,248.701] | 6.089 [6.003,6.221] | validated |
| M3|M32|N64|W8|vec2|native|q2|G98304 | 3 | M32_N64_w8:vec2:within:native:ONE_FULL|q2|G98304|K1 | 369.857 [369.394,370.712] | 6.087 [6.071,6.102] | validated |
| M3|M32|N64|W8|vec2|native|q4|G32768 | 3 | M32_N64_w8:vec2:within:native:ONE_FULL|q4|G32768|K1 | 76.815 [76.713,76.920] | 5.310 [5.238,5.383] | validated |
| M3|M32|N64|W8|vec2|native|q4|G65536 | 3 | M32_N64_w8:vec2:within:native:ONE_FULL|q4|G65536|K1 | 146.514 [146.388,146.640] | 5.597 [5.529,5.668] | validated |
| M3|M32|N64|W8|vec2|native|q4|G98304 | 3 | M32_N64_w8:vec2:within:native:ONE_FULL|q4|G98304|K1 | 217.715 [217.589,217.890] | 5.461 [5.369,5.561] | validated |
| M3|M32|N64|W8|vec2|native|qnatural|G32768 | 3 | M32_N64_w8:vec2:within:native:ONE_FULL|q8|G32768|K1 | 56.898 [56.731,57.053] | 1.770 [1.359,2.190] | validated |
| M3|M32|N64|W8|vec2|native|qnatural|G65536 | 3 | M32_N64_w8:vec2:within:native:ONE_FULL|q8|G65536|K1 | 107.163 [107.130,107.210] | 2.045 [2.006,2.080] | validated |
| M3|M32|N64|W8|vec2|native|qnatural|G98304 | 3 | M32_N64_w8:vec2:within:native:ONE_FULL|q8|G98304|K1 | 158.176 [157.954,158.373] | 2.017 [1.783,2.260] | validated |
| M3|M32|N64|W8|vec4|fixed|q1|G32768 | 3 | M32_N64_w8:vec4:within:fixed:ONE_FULL|q1|G32768|K1 | 251.131 [250.779,251.755] | 9.673 [9.654,9.696] | validated |
| M3|M32|N64|W8|vec4|fixed|q1|G65536 | 3 | M32_N64_w8:vec4:within:fixed:ONE_FULL|q1|G65536|K1 | 496.597 [495.914,497.906] | 9.781 [9.738,9.820] | validated |
| M3|M32|N64|W8|vec4|fixed|q1|G98304 | 3 | M32_N64_w8:vec4:within:fixed:ONE_FULL|q1|G98304|K1 | 741.980 [740.975,743.934] | 9.845 [9.822,9.860] | validated |
| M3|M32|N64|W8|vec4|fixed|q2|G32768 | 3 | M32_N64_w8:vec4:within:fixed:ONE_FULL|q2|G32768|K1 | 137.344 [137.198,137.520] | 9.467 [9.431,9.500] | validated |
| M3|M32|N64|W8|vec4|fixed|q2|G65536 | 3 | M32_N64_w8:vec4:within:fixed:ONE_FULL|q2|G65536|K1 | 268.644 [268.305,269.223] | 9.521 [9.493,9.542] | validated |
| M3|M32|N64|W8|vec4|fixed|q2|G98304 | 3 | M32_N64_w8:vec4:within:fixed:ONE_FULL|q2|G98304|K1 | 400.157 [399.743,400.894] | 9.587 [9.481,9.652] | validated |
| M3|M32|N64|W8|vec4|fixed|q4|G32768 | 3 | M32_N64_w8:vec4:within:fixed:ONE_FULL|q4|G32768|K1 | 83.159 [83.091,83.221] | 9.814 [9.739,9.888] | validated |
| M3|M32|N64|W8|vec4|fixed|q4|G65536 | 3 | M32_N64_w8:vec4:within:fixed:ONE_FULL|q4|G65536|K1 | 159.049 [158.899,159.243] | 10.044 [9.982,10.107] | validated |
| M3|M32|N64|W8|vec4|fixed|q4|G98304 | 3 | M32_N64_w8:vec4:within:fixed:ONE_FULL|q4|G98304|K1 | 235.851 [235.680,236.087] | 10.082 [10.056,10.109] | validated |
| M3|M32|N64|W8|vec4|fixed|qnatural|G32768 | 3 | M32_N64_w8:vec4:within:fixed:ONE_FULL|q8|G32768|K1 | 58.940 [58.757,59.088] | 9.138 [8.830,9.507] | validated |
| M3|M32|N64|W8|vec4|fixed|qnatural|G65536 | 3 | M32_N64_w8:vec4:within:fixed:ONE_FULL|q8|G65536|K1 | 111.158 [111.110,111.195] | 8.915 [8.860,8.976] | validated |
| M3|M32|N64|W8|vec4|fixed|qnatural|G98304 | 3 | M32_N64_w8:vec4:within:fixed:ONE_FULL|q8|G98304|K1 | 164.341 [164.183,164.453] | 8.842 [8.780,8.927] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q1|G32768 | 3 | M32_N64_w8:vec4:within:legacy_fixed:ONE_FULL|q1|G32768|K1 | 251.142 [250.751,251.812] | 9.425 [9.414,9.437] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q1|G65536 | 3 | M32_N64_w8:vec4:within:legacy_fixed:ONE_FULL|q1|G65536|K1 | 496.605 [495.818,498.037] | 9.538 [9.519,9.559] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q1|G98304 | 3 | M32_N64_w8:vec4:within:legacy_fixed:ONE_FULL|q1|G98304|K1 | 742.076 [740.978,744.187] | 9.598 [9.581,9.613] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q2|G32768 | 3 | M32_N64_w8:vec4:within:legacy_fixed:ONE_FULL|q2|G32768|K1 | 137.467 [137.123,137.776] | 9.109 [9.016,9.185] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q2|G65536 | 3 | M32_N64_w8:vec4:within:legacy_fixed:ONE_FULL|q2|G65536|K1 | 268.683 [268.240,269.370] | 9.331 [9.259,9.385] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q2|G98304 | 3 | M32_N64_w8:vec4:within:legacy_fixed:ONE_FULL|q2|G98304|K1 | 400.275 [399.729,401.244] | 9.339 [9.325,9.356] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q4|G32768 | 3 | M32_N64_w8:vec4:within:legacy_fixed:ONE_FULL|q4|G32768|K1 | 83.068 [82.874,83.242] | 9.710 [9.618,9.817] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q4|G65536 | 3 | M32_N64_w8:vec4:within:legacy_fixed:ONE_FULL|q4|G65536|K1 | 159.068 [158.826,159.325] | 9.731 [9.670,9.803] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|q4|G98304 | 3 | M32_N64_w8:vec4:within:legacy_fixed:ONE_FULL|q4|G98304|K1 | 235.853 [235.665,236.130] | 9.657 [9.623,9.686] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|qnatural|G32768 | 3 | M32_N64_w8:vec4:within:legacy_fixed:ONE_FULL|q8|G32768|K1 | 59.061 [58.885,59.207] | 8.818 [8.486,9.149] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|qnatural|G65536 | 3 | M32_N64_w8:vec4:within:legacy_fixed:ONE_FULL|q8|G65536|K1 | 111.097 [111.023,111.164] | 8.927 [8.826,9.018] | validated |
| M3|M32|N64|W8|vec4|legacy_fixed|qnatural|G98304 | 3 | M32_N64_w8:vec4:within:legacy_fixed:ONE_FULL|q8|G98304|K1 | 164.209 [164.062,164.346] | 8.813 [8.740,8.897] | validated |
| M3|M32|N64|W8|vec4|native|q1|G32768 | 3 | M32_N64_w8:vec4:within:native:ONE_FULL|q1|G32768|K1 | 246.435 [246.048,247.135] | 6.429 [6.408,6.452] | validated |
| M3|M32|N64|W8|vec4|native|q1|G65536 | 3 | M32_N64_w8:vec4:within:native:ONE_FULL|q1|G65536|K1 | 487.450 [486.772,488.761] | 6.407 [6.396,6.417] | validated |
| M3|M32|N64|W8|vec4|native|q1|G98304 | 3 | M32_N64_w8:vec4:within:native:ONE_FULL|q1|G98304|K1 | 728.303 [727.455,729.891] | 6.496 [6.483,6.510] | validated |
| M3|M32|N64|W8|vec4|native|q2|G32768 | 3 | M32_N64_w8:vec4:within:native:ONE_FULL|q2|G32768|K1 | 132.825 [132.706,133.022] | 8.130 [8.101,8.155] | validated |
| M3|M32|N64|W8|vec4|native|q2|G65536 | 3 | M32_N64_w8:vec4:within:native:ONE_FULL|q2|G65536|K1 | 259.934 [259.639,260.446] | 8.104 [8.073,8.138] | validated |
| M3|M32|N64|W8|vec4|native|q2|G98304 | 3 | M32_N64_w8:vec4:within:native:ONE_FULL|q2|G98304|K1 | 387.130 [386.679,387.961] | 8.111 [8.094,8.129] | validated |
| M3|M32|N64|W8|vec4|native|q4|G32768 | 3 | M32_N64_w8:vec4:within:native:ONE_FULL|q4|G32768|K1 | 80.363 [80.313,80.406] | 8.572 [8.527,8.622] | validated |
| M3|M32|N64|W8|vec4|native|q4|G65536 | 3 | M32_N64_w8:vec4:within:native:ONE_FULL|q4|G65536|K1 | 153.558 [153.469,153.639] | 8.765 [8.699,8.831] | validated |
| M3|M32|N64|W8|vec4|native|q4|G98304 | 3 | M32_N64_w8:vec4:within:native:ONE_FULL|q4|G98304|K1 | 227.666 [227.547,227.822] | 8.795 [8.763,8.828] | validated |
| M3|M32|N64|W8|vec4|native|qnatural|G32768 | 3 | M32_N64_w8:vec4:within:native:ONE_FULL|q8|G32768|K1 | 57.831 [57.610,58.031] | 6.043 [5.694,6.414] | validated |
| M3|M32|N64|W8|vec4|native|qnatural|G65536 | 3 | M32_N64_w8:vec4:within:native:ONE_FULL|q8|G65536|K1 | 109.396 [109.364,109.426] | 5.470 [5.434,5.511] | validated |
| M3|M32|N64|W8|vec4|native|qnatural|G98304 | 3 | M32_N64_w8:vec4:within:native:ONE_FULL|q8|G98304|K1 | 161.552 [161.217,161.796] | 5.474 [5.345,5.642] | validated |
| M3|M32|N64|W8|vec8|fixed|q1|G32768 | 3 | M32_N64_w8:vec8:within:fixed:ONE_FULL|q1|G32768|K1 | 271.588 [271.273,272.162] | 12.578 [12.561,12.597] | validated |
| M3|M32|N64|W8|vec8|fixed|q1|G65536 | 3 | M32_N64_w8:vec8:within:fixed:ONE_FULL|q1|G65536|K1 | 537.586 [536.749,538.975] | 12.683 [12.645,12.728] | validated |
| M3|M32|N64|W8|vec8|fixed|q1|G98304 | 3 | M32_N64_w8:vec8:within:fixed:ONE_FULL|q1|G98304|K1 | 803.698 [802.480,805.832] | 12.699 [12.654,12.733] | validated |
| M3|M32|N64|W8|vec8|fixed|q2|G32768 | 3 | M32_N64_w8:vec8:within:fixed:ONE_FULL|q2|G32768|K1 | 150.176 [149.987,150.428] | 14.521 [14.489,14.555] | validated |
| M3|M32|N64|W8|vec8|fixed|q2|G65536 | 3 | M32_N64_w8:vec8:within:fixed:ONE_FULL|q2|G65536|K1 | 294.188 [294.026,294.446] | 14.641 [14.598,14.703] | validated |
| M3|M32|N64|W8|vec8|fixed|q2|G98304 | 3 | M32_N64_w8:vec8:within:fixed:ONE_FULL|q2|G98304|K1 | 438.551 [438.262,439.028] | 14.661 [14.625,14.710] | validated |
| M3|M32|N64|W8|vec8|fixed|q4|G32768 | 3 | M32_N64_w8:vec8:within:fixed:ONE_FULL|q4|G32768|K1 | 93.199 [93.106,93.295] | 17.280 [17.220,17.348] | validated |
| M3|M32|N64|W8|vec8|fixed|q4|G65536 | 3 | M32_N64_w8:vec8:within:fixed:ONE_FULL|q4|G65536|K1 | 179.021 [178.884,179.193] | 17.822 [17.781,17.865] | validated |
| M3|M32|N64|W8|vec8|fixed|q4|G98304 | 3 | M32_N64_w8:vec8:within:fixed:ONE_FULL|q4|G98304|K1 | 265.222 [265.051,265.447] | 17.789 [17.753,17.824] | validated |
| M3|M32|N64|W8|vec8|fixed|qnatural|G32768 | 3 | M32_N64_w8:vec8:within:fixed:ONE_FULL|q8|G32768|K1 | 83.362 [83.315,83.398] | 23.042 [23.006,23.085] | validated |
| M3|M32|N64|W8|vec8|fixed|qnatural|G65536 | 3 | M32_N64_w8:vec8:within:fixed:ONE_FULL|q8|G65536|K1 | 159.774 [159.629,159.923] | 23.772 [23.718,23.834] | validated |
| M3|M32|N64|W8|vec8|fixed|qnatural|G98304 | 3 | M32_N64_w8:vec8:within:fixed:ONE_FULL|q8|G98304|K1 | 236.543 [236.448,236.683] | 23.901 [23.885,23.918] | validated |
| M3|M32|N64|W8|vec8|native|q1|G32768 | 3 | M32_N64_w8:vec8:within:native:ONE_FULL|q1|G32768|K1 | 264.417 [263.983,265.129] | 12.295 [12.265,12.324] | validated |
| M3|M32|N64|W8|vec8|native|q1|G65536 | 3 | M32_N64_w8:vec8:within:native:ONE_FULL|q1|G65536|K1 | 523.282 [522.578,524.619] | 12.364 [12.336,12.383] | validated |
| M3|M32|N64|W8|vec8|native|q1|G98304 | 3 | M32_N64_w8:vec8:within:native:ONE_FULL|q1|G98304|K1 | 782.302 [781.191,784.364] | 12.379 [12.352,12.399] | validated |
| M3|M32|N64|W8|vec8|native|q2|G32768 | 3 | M32_N64_w8:vec8:within:native:ONE_FULL|q2|G32768|K1 | 145.515 [145.314,145.762] | 14.107 [14.078,14.140] | validated |
| M3|M32|N64|W8|vec8|native|q2|G65536 | 3 | M32_N64_w8:vec8:within:native:ONE_FULL|q2|G65536|K1 | 284.880 [284.603,285.385] | 14.241 [14.223,14.257] | validated |
| M3|M32|N64|W8|vec8|native|q2|G98304 | 3 | M32_N64_w8:vec8:within:native:ONE_FULL|q2|G98304|K1 | 424.447 [423.923,425.423] | 14.286 [14.240,14.322] | validated |
| M3|M32|N64|W8|vec8|native|q4|G32768 | 3 | M32_N64_w8:vec8:within:native:ONE_FULL|q4|G32768|K1 | 94.511 [94.403,94.673] | 10.623 [10.575,10.679] | validated |
| M3|M32|N64|W8|vec8|native|q4|G65536 | 3 | M32_N64_w8:vec8:within:native:ONE_FULL|q4|G65536|K1 | 181.929 [181.750,182.206] | 10.790 [10.755,10.826] | validated |
| M3|M32|N64|W8|vec8|native|q4|G98304 | 3 | M32_N64_w8:vec8:within:native:ONE_FULL|q4|G98304|K1 | 269.917 [269.642,270.323] | 10.618 [10.571,10.670] | validated |
| M3|M32|N64|W8|vec8|native|qnatural|G32768 | 3 | M32_N64_w8:vec8:within:native:ONE_FULL|q8|G32768|K1 | 83.915 [83.842,83.989] | 13.061 [13.037,13.085] | validated |
| M3|M32|N64|W8|vec8|native|qnatural|G65536 | 3 | M32_N64_w8:vec8:within:native:ONE_FULL|q8|G65536|K1 | 160.986 [160.851,161.166] | 13.377 [13.333,13.420] | validated |
| M3|M32|N64|W8|vec8|native|qnatural|G98304 | 3 | M32_N64_w8:vec8:within:native:ONE_FULL|q8|G98304|K1 | 238.454 [238.252,238.695] | 13.342 [13.305,13.378] | validated |

## Interpretation boundaries

- The new estimand is a single target semantic iteration after eight GPU-side primes, with input restoration outside its events and a fixed minimum host gap. It is not the historical unrestricted-cadence estimand or an idle-first-use measurement.
- U/G0 and same-program sham failures remain inconclusive controls; no sample/worker is dropped and no favorable equivalence is required by analysis.
- Exact aliases reuse a physical observation and never count as additional independent data. Each suite has three GPU allocations, not raw_sample_count independent GPUs.
- Every logical candidate is reported. Context minima are descriptive post-hoc summaries, not a pre-registered selector or a multiplicity-controlled ranking.
- Old/new comparisons are separate-cohort descriptive means and within-run gains. They are not paired causal differences, do not subtract a fixed idle penalty, and do not relabel old NCU reports.
- No model is refit and seen D1 geometries are not called new blind tests. Legacy and anchor defaults are explicit Gluon references, not invented source-aligned Triton U/G0 baselines.

Telemetry retains return codes and compressed raw samples. Its nominal clocks and hardware averaging windows do not establish per-kernel fixed frequency or identify a causal DVFS mechanism.
