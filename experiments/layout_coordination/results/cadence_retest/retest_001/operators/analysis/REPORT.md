# Fixed-cadence replacement: operators

Data status: **OK**. Three workers × ten visits × 31 samples per physical condition; 469,650 target observations.
Each graph runs eight untimed semantic iterations, then exactly one timed iteration. Original pipeline zeroes remain inside each iteration; any input restoration precedes the target events. Between observations the requested minimum host gap is 1 ms; actual gaps remain recorded.

U/G0 equivalence: 78/78. Same-program sham equivalence: 78/78.
Those are logical workload comparisons; exact aliases can share data. Distinct physical U/G0 pairs: 66; distinct physical sham labels per worker: 66. They are not additional independent GPUs.
Equivalence requires the full paired CI95 to be inside ±3%. A candidate is marked strictly improved only when its fresh U and G0 lower bounds both exceed +3%, and the relevant controls pass. Legacy/anchor comparisons instead use their explicitly named Gluon reference.

All logical rows, physical aliases, visit medians, host-gap summaries, old/new descriptive comparisons and control decisions are retained in `analysis.json` and `timing.csv`. No unfavorable or inconclusive control is removed.

Recorded host gaps below the requested minimum: 0. Observed minimum/maximum: 1004.115 / 67579.803 µs. GPU-side primes are outside these host-gap durations and outside target timing.

## Control results

| Context | U/reference | G0 vs U gain % [CI95] | Same-program repeat gain % [CI95] | Status |
|---|---|---:|---:|---|
| op__C01_p128_n4096__merge__M128 | C01_p128_n4096:upstream | -0.547 [-1.907,0.300] | -0.013 [-0.091,0.065] | validated |
| op__C01_p128_n4096__pipeline__M128 | C01_p128_n4096:upstream | 0.626 [0.362,1.019] | -0.011 [-0.071,0.055] | validated |
| op__C01_p128_n4096__pipeline__M2048 | C01_p128_n4096:upstream | -0.032 [-0.132,0.062] | 0.084 [-0.024,0.202] | validated |
| op__C01_p256_n1024__merge__M128 | C01_p256_n1024:upstream | 1.599 [1.408,1.797] | 0.020 [-0.039,0.078] | validated |
| op__C01_p256_n1024__pipeline__M128 | C01_p256_n1024:upstream | -0.851 [-1.616,-0.177] | -0.006 [-0.042,0.030] | validated |
| op__C01_p256_n1024__pipeline__M2048 | C01_p256_n1024:upstream | 0.258 [0.043,0.464] | 0.012 [-0.125,0.157] | validated |
| op__C01_p96_n8192__merge__M128 | C01_p96_n8192:upstream | -0.999 [-2.128,-0.329] | 0.014 [-0.071,0.128] | validated |
| op__C01_p96_n8192__pipeline__M128 | C01_p96_n8192:upstream | -1.205 [-1.377,-0.984] | -0.048 [-0.131,0.019] | validated |
| op__C01_p96_n8192__pipeline__M2048 | C01_p96_n8192:upstream | -1.144 [-1.260,-1.026] | 0.046 [-0.020,0.114] | validated |
| op__g2_C02_n64_d128__b1__l2048__decode | g2_C02_n64_d128:upstream | 0.271 [0.059,0.534] | -0.054 [-0.177,0.044] | validated |
| op__g2_C02_n64_d128__b1__l2048__stage1 | g2_C02_n64_d128:upstream | -0.256 [-0.495,0.061] | -0.011 [-0.134,0.150] | validated |
| op__g2_C02_n64_d128__b1__l256__decode | g2_C02_n64_d128:upstream | -0.011 [-0.079,0.045] | 0.011 [-0.034,0.056] | validated |
| op__g2_C02_n64_d128__b1__l256__stage1 | g2_C02_n64_d128:upstream | -0.270 [-0.919,0.227] | -0.000 [-0.135,0.136] | validated |
| op__g2_C02_n64_d128__b8__l2048__decode | g2_C02_n64_d128:upstream | -0.018 [-0.074,0.042] | -0.000 [-0.056,0.055] | validated |
| op__g2_C02_n64_d128__b8__l2048__stage1 | g2_C02_n64_d128:upstream | -0.091 [-0.135,-0.050] | 0.009 [-0.045,0.060] | validated |
| op__g2_C02_n64_d128__b8__l256__decode | g2_C02_n64_d128:upstream | -0.079 [-0.340,0.177] | -0.033 [-0.295,0.229] | validated |
| op__g2_C02_n64_d128__b8__l256__stage1 | g2_C02_n64_d128:upstream | -0.269 [-0.611,0.092] | -0.092 [-0.518,0.406] | validated |
| op__g2_C02_n64_d256__b1__l2048__decode | g2_C02_n64_d256:upstream | 0.095 [-0.025,0.229] | 0.110 [0.007,0.213] | validated |
| op__g2_C02_n64_d256__b1__l2048__stage1 | g2_C02_n64_d256:upstream | 0.128 [-0.014,0.297] | 0.043 [-0.082,0.188] | validated |
| op__g2_C02_n64_d256__b1__l256__decode | g2_C02_n64_d256:upstream | 0.461 [0.043,1.036] | -0.017 [-0.070,0.044] | validated |
| op__g2_C02_n64_d256__b1__l256__stage1 | g2_C02_n64_d256:upstream | -0.230 [-0.809,0.275] | -0.055 [-0.297,0.189] | validated |
| op__g2_C02_n64_d256__b8__l2048__decode | g2_C02_n64_d256:upstream | 0.037 [-0.014,0.096] | 0.024 [-0.026,0.080] | validated |
| op__g2_C02_n64_d256__b8__l2048__stage1 | g2_C02_n64_d256:upstream | -0.056 [-0.105,0.001] | -0.005 [-0.055,0.046] | validated |
| op__g2_C02_n64_d256__b8__l256__decode | g2_C02_n64_d256:upstream | 0.212 [0.076,0.343] | -0.025 [-0.131,0.087] | validated |
| op__g2_C02_n64_d256__b8__l256__stage1 | g2_C02_n64_d256:upstream | 0.018 [-0.193,0.192] | -0.003 [-0.137,0.150] | validated |
| op__g2_C02_n64_d64__b1__l2048__decode | g2_C02_n64_d64:upstream | 0.077 [0.014,0.148] | -0.000 [-0.070,0.070] | validated |
| op__g2_C02_n64_d64__b1__l2048__stage1 | g2_C02_n64_d64:upstream | -0.709 [-0.951,-0.311] | -0.041 [-0.173,0.041] | validated |
| op__g2_C02_n64_d64__b1__l256__decode | g2_C02_n64_d64:upstream | 0.221 [-0.122,0.697] | -0.012 [-0.123,0.086] | validated |
| op__g2_C02_n64_d64__b1__l256__stage1 | g2_C02_n64_d64:upstream | -0.136 [-0.347,0.017] | 0.000 [-0.068,0.069] | validated |
| op__g2_C02_n64_d64__b8__l2048__decode | g2_C02_n64_d64:upstream | -0.059 [-0.241,0.100] | -0.034 [-0.255,0.172] | validated |
| op__g2_C02_n64_d64__b8__l2048__stage1 | g2_C02_n64_d64:upstream | -0.205 [-0.428,0.012] | 0.077 [-0.059,0.230] | validated |
| op__g2_C02_n64_d64__b8__l256__decode | g2_C02_n64_d64:upstream | 0.378 [0.072,0.664] | 0.009 [-0.109,0.153] | validated |
| op__g2_C02_n64_d64__b8__l256__stage1 | g2_C02_n64_d64:upstream | 0.641 [0.287,1.041] | -0.023 [-0.240,0.206] | validated |
| op__g2_C06_s16_d128__b1__l8192__merge | g2_C06_s16_d128:upstream | 0.246 [-0.141,0.761] | -0.211 [-0.548,0.052] | validated |
| op__g2_C06_s16_d128__b64__l8192__merge | g2_C06_s16_d128:upstream | -1.081 [-1.803,-0.267] | -0.032 [-0.107,0.033] | validated |
| op__g2_C06_s16_d128__b8__l8192__merge | g2_C06_s16_d128:upstream | 0.824 [-0.081,2.108] | 0.016 [-0.081,0.113] | validated |
| op__g2_C06_s32_d128__b1__l8192__merge | g2_C06_s32_d128:upstream | -0.120 [-2.440,1.837] | 0.034 [-0.085,0.190] | validated |
| op__g2_C06_s32_d128__b64__l8192__merge | g2_C06_s32_d128:upstream | -0.126 [-0.403,0.140] | -0.237 [-0.513,0.022] | validated |
| op__g2_C06_s32_d128__b8__l8192__merge | g2_C06_s32_d128:upstream | -0.391 [-0.771,0.032] | -0.031 [-0.127,0.047] | validated |
| op__g2_C06_s32_d256__b1__l8192__merge | g2_C06_s32_d256:upstream | 0.338 [-0.129,1.044] | 0.048 [-0.083,0.190] | validated |
| op__g2_C06_s32_d256__b64__l8192__merge | g2_C06_s32_d256:upstream | -0.230 [-0.465,0.023] | -0.054 [-0.224,0.131] | validated |
| op__g2_C06_s32_d256__b8__l8192__merge | g2_C06_s32_d256:upstream | 0.070 [-0.086,0.221] | 0.042 [-0.071,0.152] | validated |
| op__g4a_C02_n64_d128__b8__l2048__decode | g4a_C02_n64_d128:upstream | -0.049 [-0.095,-0.002] | 0.010 [-0.036,0.052] | validated |
| op__g4a_C02_n64_d128__b8__l2048__stage1 | g4a_C02_n64_d128:upstream | 0.067 [0.023,0.108] | -0.016 [-0.063,0.032] | validated |
| op__g4a_C02_n64_d128__b8__l256__decode | g4a_C02_n64_d128:upstream | -0.219 [-0.522,0.084] | 0.097 [-0.272,0.465] | validated |
| op__g4a_C02_n64_d128__b8__l256__stage1 | g4a_C02_n64_d128:upstream | -0.645 [-1.003,-0.307] | -0.069 [-0.354,0.207] | validated |
| op__g4a_C02_n64_d64__b8__l2048__decode | g4a_C02_n64_d64:upstream | 0.026 [-0.128,0.176] | -0.063 [-0.219,0.100] | validated |
| op__g4a_C02_n64_d64__b8__l2048__stage1 | g4a_C02_n64_d64:upstream | -0.096 [-0.271,0.088] | 0.042 [-0.149,0.237] | validated |
| op__g4a_C02_n64_d64__b8__l256__decode | g4a_C02_n64_d64:upstream | -0.192 [-0.400,-0.027] | 0.082 [-0.073,0.227] | validated |
| op__g4a_C02_n64_d64__b8__l256__stage1 | g4a_C02_n64_d64:upstream | 0.322 [0.162,0.510] | 0.046 [-0.080,0.174] | validated |
| op__g4b_C02_n64_d128_b1_l256__b1__l256__decode | g4b_C02_n64_d128_b1_l256:B4:upstream | -0.191 [-0.444,0.203] | 0.045 [-0.090,0.234] | validated |
| op__g4b_C02_n64_d128_b1_l256__b1__l256__decode | g4b_C02_n64_d128_b1_l256:actual:upstream | -0.191 [-0.444,0.203] | 0.045 [-0.090,0.234] | validated |
| op__g4b_C02_n64_d128_b1_l256__b1__l256__stage1 | g4b_C02_n64_d128_b1_l256:B4:upstream | 0.000 [-1.268,0.855] | 0.015 [-0.044,0.075] | validated |
| op__g4b_C02_n64_d128_b1_l256__b1__l256__stage1 | g4b_C02_n64_d128_b1_l256:actual:upstream | 0.000 [-1.268,0.855] | 0.015 [-0.044,0.075] | validated |
| op__g4b_C02_n64_d128_b8_l2048__b8__l2048__decode | g4b_C02_n64_d128_b8_l2048:B4:upstream | -0.002 [-0.054,0.060] | -0.010 [-0.080,0.075] | validated |
| op__g4b_C02_n64_d128_b8_l2048__b8__l2048__decode | g4b_C02_n64_d128_b8_l2048:actual:upstream | -0.002 [-0.054,0.060] | -0.010 [-0.080,0.075] | validated |
| op__g4b_C02_n64_d128_b8_l2048__b8__l2048__stage1 | g4b_C02_n64_d128_b8_l2048:B4:upstream | 0.002 [-0.048,0.057] | 0.043 [-0.062,0.126] | validated |
| op__g4b_C02_n64_d128_b8_l2048__b8__l2048__stage1 | g4b_C02_n64_d128_b8_l2048:actual:upstream | 0.002 [-0.048,0.057] | 0.043 [-0.062,0.126] | validated |
| op__g4b_C02_n64_d64_b1_l256__b1__l256__decode | g4b_C02_n64_d64_b1_l256:B4:upstream | 0.379 [0.122,0.616] | 0.012 [-0.074,0.098] | validated |
| op__g4b_C02_n64_d64_b1_l256__b1__l256__decode | g4b_C02_n64_d64_b1_l256:actual:upstream | 0.379 [0.122,0.616] | 0.012 [-0.074,0.098] | validated |
| op__g4b_C02_n64_d64_b1_l256__b1__l256__stage1 | g4b_C02_n64_d64_b1_l256:B4:upstream | 1.249 [0.271,2.182] | 0.084 [-0.168,0.453] | validated |
| op__g4b_C02_n64_d64_b1_l256__b1__l256__stage1 | g4b_C02_n64_d64_b1_l256:actual:upstream | 1.249 [0.271,2.182] | 0.084 [-0.168,0.453] | validated |
| op__g4b_C02_n64_d64_b8_l2048__b8__l2048__decode | g4b_C02_n64_d64_b8_l2048:B4:upstream | 0.136 [0.014,0.268] | -0.006 [-0.142,0.135] | validated |
| op__g4b_C02_n64_d64_b8_l2048__b8__l2048__decode | g4b_C02_n64_d64_b8_l2048:actual:upstream | 0.136 [0.014,0.268] | -0.006 [-0.142,0.135] | validated |
| op__g4b_C02_n64_d64_b8_l2048__b8__l2048__stage1 | g4b_C02_n64_d64_b8_l2048:B4:upstream | -0.104 [-0.430,0.161] | 0.089 [-0.127,0.318] | validated |
| op__g4b_C02_n64_d64_b8_l2048__b8__l2048__stage1 | g4b_C02_n64_d64_b8_l2048:actual:upstream | -0.104 [-0.430,0.161] | 0.089 [-0.127,0.318] | validated |
| op__g4b_C06_s16_d128_b1_l8192__b1__l8192__merge | g4b_C06_s16_d128_b1_l8192:B4:upstream | 0.560 [-0.542,1.489] | 0.053 [-0.141,0.227] | validated |
| op__g4b_C06_s16_d128_b1_l8192__b1__l8192__merge | g4b_C06_s16_d128_b1_l8192:actual:upstream | 1.650 [1.025,2.075] | -0.071 [-0.494,0.338] | validated |
| op__g4b_C06_s16_d128_b64_l8192__b64__l8192__merge | g4b_C06_s16_d128_b64_l8192:B4:upstream | -0.773 [-1.560,-0.075] | -0.032 [-0.249,0.128] | validated |
| op__g4b_C06_s16_d128_b64_l8192__b64__l8192__merge | g4b_C06_s16_d128_b64_l8192:actual:upstream | -0.773 [-1.560,-0.075] | -0.032 [-0.249,0.128] | validated |
| op__g4b_C06_s16_d128_b8_l8192__b8__l8192__merge | g4b_C06_s16_d128_b8_l8192:B4:upstream | -0.975 [-2.097,-0.227] | -0.081 [-0.374,0.114] | validated |
| op__g4b_C06_s16_d128_b8_l8192__b8__l8192__merge | g4b_C06_s16_d128_b8_l8192:actual:upstream | -0.975 [-2.097,-0.227] | -0.081 [-0.374,0.114] | validated |
| op__g4b_C06_s32_d256_b1_l8192__b1__l8192__merge | g4b_C06_s32_d256_b1_l8192:B4:upstream | 1.298 [1.014,1.735] | -0.096 [-0.195,-0.016] | validated |
| op__g4b_C06_s32_d256_b1_l8192__b1__l8192__merge | g4b_C06_s32_d256_b1_l8192:actual:upstream | -0.528 [-1.894,0.252] | -0.048 [-0.191,0.064] | validated |
| op__g4b_C06_s32_d256_b64_l8192__b64__l8192__merge | g4b_C06_s32_d256_b64_l8192:B4:upstream | -0.152 [-0.329,0.067] | 0.047 [-0.174,0.250] | validated |
| op__g4b_C06_s32_d256_b64_l8192__b64__l8192__merge | g4b_C06_s32_d256_b64_l8192:actual:upstream | -0.152 [-0.329,0.067] | 0.047 [-0.174,0.250] | validated |
| op__g4b_C06_s32_d256_b8_l8192__b8__l8192__merge | g4b_C06_s32_d256_b8_l8192:B4:upstream | -0.503 [-1.447,0.014] | 0.042 [-0.055,0.138] | validated |
| op__g4b_C06_s32_d256_b8_l8192__b8__l8192__merge | g4b_C06_s32_d256_b8_l8192:actual:upstream | -0.503 [-1.447,0.014] | 0.042 [-0.055,0.138] | validated |

## Every context

The minimum candidate below is selected retrospectively for a compact inventory, not used as a deployed model. Full candidate and losing-layout data remain in the CSV/JSON.

| Context | Physical rows including sham | Descriptive minimum candidate | Target time µs [CI95] | Gain vs paired reference % [CI95] | Control |
|---|---:|---|---:|---:|---|
| op__C01_p128_n4096__merge__M128 | 11 | C01_p128_n4096:vec4:n_first | 6.547 [6.452,6.629] | 20.109 [19.390,21.075] | validated |
| op__C01_p128_n4096__pipeline__M128 | 11 | C01_p128_n4096:vec4:within | 17.057 [16.980,17.193] | 12.234 [11.386,13.217] | validated |
| op__C01_p128_n4096__pipeline__M2048 | 11 | C01_p128_n4096:vec4:n_first | 51.193 [51.078,51.279] | 4.227 [4.141,4.306] | validated |
| op__C01_p256_n1024__merge__M128 | 11 | C01_p256_n1024:vec4:n_first | 7.785 [7.726,7.851] | 28.835 [27.974,29.359] | validated |
| op__C01_p256_n1024__pipeline__M128 | 11 | C01_p256_n1024:vec4:n_first | 15.906 [15.790,15.968] | 11.280 [11.024,11.711] | validated |
| op__C01_p256_n1024__pipeline__M2048 | 11 | C01_p256_n1024:vec4:n_first | 31.212 [31.134,31.281] | 11.072 [10.896,11.260] | validated |
| op__C01_p96_n8192__merge__M128 | 11 | C01_p96_n8192:vec4:within | 6.331 [6.219,6.428] | 15.263 [14.525,16.236] | validated |
| op__C01_p96_n8192__pipeline__M128 | 11 | C01_p96_n8192:vec4:within | 20.567 [20.532,20.604] | 6.674 [6.459,6.920] | validated |
| op__C01_p96_n8192__pipeline__M2048 | 11 | C01_p96_n8192:vec2:within | 77.766 [77.498,77.952] | 1.839 [1.686,2.032] | validated |
| op__g2_C02_n64_d128__b1__l2048__decode | 9 | g2_C02_n64_d128:vec8:n_first | 21.596 [21.459,21.670] | 0.271 [0.059,0.534] | validated |
| op__g2_C02_n64_d128__b1__l2048__stage1 | 9 | g2_C02_n64_d128:vec8:n_first | 19.227 [19.119,19.356] | -0.256 [-0.495,0.061] | validated |
| op__g2_C02_n64_d128__b1__l256__decode | 9 | g2_C02_n64_d128:vec8:n_first | 9.454 [9.376,9.506] | -0.011 [-0.079,0.045] | validated |
| op__g2_C02_n64_d128__b1__l256__stage1 | 9 | g2_C02_n64_d128:vec8:n_first | 7.130 [7.028,7.264] | -0.270 [-0.919,0.227] | validated |
| op__g2_C02_n64_d128__b8__l2048__decode | 9 | g2_C02_n64_d128:vec8:n_first | 102.313 [101.246,104.352] | -0.018 [-0.074,0.042] | validated |
| op__g2_C02_n64_d128__b8__l2048__stage1 | 9 | g2_C02_n64_d128:vec8:n_first | 98.926 [97.845,100.974] | -0.091 [-0.135,-0.050] | validated |
| op__g2_C02_n64_d128__b8__l256__decode | 9 | g2_C02_n64_d128:vec8:n_first | 16.305 [16.236,16.356] | -0.079 [-0.340,0.177] | validated |
| op__g2_C02_n64_d128__b8__l256__stage1 | 9 | g2_C02_n64_d128:vec8:n_first | 13.897 [13.808,13.973] | -0.269 [-0.611,0.092] | validated |
| op__g2_C02_n64_d256__b1__l2048__decode | 8 | g2_C02_n64_d256:vec8:n_first | 71.803 [71.572,71.947] | 0.095 [-0.025,0.229] | validated |
| op__g2_C02_n64_d256__b1__l2048__stage1 | 8 | g2_C02_n64_d256:vec8:n_first | 69.110 [68.958,69.227] | 0.128 [-0.014,0.297] | validated |
| op__g2_C02_n64_d256__b1__l256__decode | 8 | g2_C02_n64_d256:vec8:n_first | 12.219 [12.119,12.377] | 0.461 [0.043,1.036] | validated |
| op__g2_C02_n64_d256__b1__l256__stage1 | 8 | g2_C02_n64_d256:vec8:n_first | 9.757 [9.648,9.831] | -0.230 [-0.809,0.275] | validated |
| op__g2_C02_n64_d256__b8__l2048__decode | 8 | g2_C02_n64_d256:vec8:n_first | 203.875 [201.381,208.745] | 0.037 [-0.014,0.096] | validated |
| op__g2_C02_n64_d256__b8__l2048__stage1 | 8 | g2_C02_n64_d256:vec8:n_first | 200.461 [198.013,205.282] | -0.056 [-0.105,0.001] | validated |
| op__g2_C02_n64_d256__b8__l256__decode | 8 | g2_C02_n64_d256:vec8:n_first | 38.103 [37.892,38.439] | 0.212 [0.076,0.343] | validated |
| op__g2_C02_n64_d256__b8__l256__stage1 | 8 | g2_C02_n64_d256:vec8:n_first | 35.087 [34.753,35.525] | 0.018 [-0.193,0.192] | validated |
| op__g2_C02_n64_d64__b1__l2048__decode | 10 | g2_C02_n64_d64:vec8:n_first | 15.150 [15.076,15.206] | 0.077 [0.014,0.148] | validated |
| op__g2_C02_n64_d64__b1__l2048__stage1 | 10 | g2_C02_n64_d64:vec8:n_first | 13.023 [12.953,13.063] | -0.709 [-0.951,-0.311] | validated |
| op__g2_C02_n64_d64__b1__l256__decode | 10 | g2_C02_n64_d64:vec2:within | 8.592 [8.519,8.662] | 1.069 [0.419,2.054] | validated |
| op__g2_C02_n64_d64__b1__l256__stage1 | 10 | g2_C02_n64_d64:vec2:within | 6.251 [6.144,6.336] | 0.034 [0.000,0.102] | validated |
| op__g2_C02_n64_d64__b8__l2048__decode | 10 | g2_C02_n64_d64:vec4:n_first | 53.947 [53.428,54.855] | 0.093 [-0.122,0.297] | validated |
| op__g2_C02_n64_d64__b8__l2048__stage1 | 10 | g2_C02_n64_d64:vec4:n_first | 51.219 [50.644,52.175] | 0.342 [0.083,0.545] | validated |
| op__g2_C02_n64_d64__b8__l256__decode | 10 | g2_C02_n64_d64:vec4:n_first | 11.603 [11.568,11.650] | 1.982 [1.648,2.295] | validated |
| op__g2_C02_n64_d64__b8__l256__stage1 | 10 | g2_C02_n64_d64:vec4:n_first | 9.107 [9.063,9.154] | 2.233 [2.021,2.518] | validated |
| op__g2_C06_s16_d128__b1__l8192__merge | 9 | g2_C06_s16_d128:vec4:within | 5.773 [5.744,5.802] | 4.936 [4.545,5.235] | validated |
| op__g2_C06_s16_d128__b64__l8192__merge | 9 | g2_C06_s16_d128:vec4:within | 9.151 [9.100,9.189] | 7.264 [6.638,8.009] | validated |
| op__g2_C06_s16_d128__b8__l8192__merge | 9 | g2_C06_s16_d128:vec2:within | 6.342 [6.272,6.433] | 3.942 [2.820,4.620] | validated |
| op__g2_C06_s32_d128__b1__l8192__merge | 9 | g2_C06_s32_d128:vec4:within | 5.998 [5.899,6.051] | 3.732 [1.220,5.952] | validated |
| op__g2_C06_s32_d128__b64__l8192__merge | 9 | g2_C06_s32_d128:vec4:within | 13.376 [13.278,13.461] | 7.235 [7.003,7.465] | validated |
| op__g2_C06_s32_d128__b8__l8192__merge | 9 | g2_C06_s32_d128:vec4:within | 6.623 [6.588,6.688] | 2.848 [1.073,4.566] | validated |
| op__g2_C06_s32_d256__b1__l8192__merge | 8 | g2_C06_s32_d256:vec4:within | 6.419 [6.272,6.499] | 3.216 [2.922,3.544] | validated |
| op__g2_C06_s32_d256__b64__l8192__merge | 8 | g2_C06_s32_d256:vec4:within | 31.035 [30.895,31.243] | 1.673 [1.474,1.908] | validated |
| op__g2_C06_s32_d256__b8__l8192__merge | 8 | g2_C06_s32_d256:vec4:within | 7.326 [7.175,7.505] | 3.917 [2.615,5.263] | validated |
| op__g4a_C02_n64_d128__b8__l2048__decode | 6 | g4a_C02_n64_d128:pv_only | 104.791 [103.775,106.694] | -2.409 [-2.513,-2.294] | validated |
| op__g4a_C02_n64_d128__b8__l2048__stage1 | 6 | g4a_C02_n64_d128:pv_only | 101.282 [100.224,103.307] | -2.315 [-2.385,-2.245] | validated |
| op__g4a_C02_n64_d128__b8__l256__decode | 6 | g4a_C02_n64_d128:pv_only | 16.805 [16.750,16.868] | -1.639 [-1.834,-1.450] | validated |
| op__g4a_C02_n64_d128__b8__l256__stage1 | 6 | g4a_C02_n64_d128:pv_only | 14.269 [14.206,14.321] | -2.726 [-2.975,-2.428] | validated |
| op__g4a_C02_n64_d64__b8__l2048__decode | 6 | g4a_C02_n64_d64:pv_only | 67.918 [67.451,68.747] | -25.866 [-26.364,-25.210] | validated |
| op__g4a_C02_n64_d64__b8__l2048__stage1 | 6 | g4a_C02_n64_d64:pv_only | 64.562 [64.059,65.435] | -25.909 [-26.395,-25.268] | validated |
| op__g4a_C02_n64_d64__b8__l256__decode | 6 | g4a_C02_n64_d64:pv_only | 12.604 [12.484,12.669] | -7.761 [-7.966,-7.508] | validated |
| op__g4a_C02_n64_d64__b8__l256__stage1 | 6 | g4a_C02_n64_d64:pv_only | 10.091 [10.016,10.144] | -8.949 [-9.157,-8.738] | validated |
| op__g4b_C02_n64_d128_b1_l256__b1__l256__decode | 4 | g4b_C02_n64_d128_b1_l256:B4:target | 9.697 [9.536,9.856] | -2.238 [-2.667,-1.687] | validated |
| op__g4b_C02_n64_d128_b1_l256__b1__l256__stage1 | 4 | g4b_C02_n64_d128_b1_l256:B4:target | 7.392 [7.232,7.548] | -2.773 [-4.366,-1.802] | validated |
| op__g4b_C02_n64_d128_b8_l2048__b8__l2048__decode | 4 | g4b_C02_n64_d128_b8_l2048:B4:target | 104.620 [103.634,106.494] | -2.189 [-2.251,-2.127] | validated |
| op__g4b_C02_n64_d128_b8_l2048__b8__l2048__stage1 | 4 | g4b_C02_n64_d128_b8_l2048:B4:target | 101.126 [100.109,103.075] | -2.251 [-2.313,-2.194] | validated |
| op__g4b_C02_n64_d64_b1_l256__b1__l256__decode | 4 | g4b_C02_n64_d64_b1_l256:B4:target | 8.911 [8.798,8.999] | -2.165 [-2.582,-1.564] | validated |
| op__g4b_C02_n64_d64_b1_l256__b1__l256__stage1 | 4 | g4b_C02_n64_d64_b1_l256:B4:target | 6.635 [6.449,6.748] | -4.961 [-6.647,-3.103] | validated |
| op__g4b_C02_n64_d64_b8_l2048__b8__l2048__decode | 4 | g4b_C02_n64_d64_b8_l2048:B4:target | 67.836 [67.391,68.562] | -25.396 [-25.840,-24.780] | validated |
| op__g4b_C02_n64_d64_b8_l2048__b8__l2048__stage1 | 4 | g4b_C02_n64_d64_b8_l2048:B4:target | 64.583 [64.131,65.412] | -25.765 [-26.244,-25.231] | validated |
| op__g4b_C06_s16_d128_b1_l8192__b1__l8192__merge | 8 | g4b_C06_s16_d128_b1_l8192:actual:target | 6.197 [6.084,6.269] | -3.051 [-3.890,-2.004] | validated |
| op__g4b_C06_s16_d128_b64_l8192__b64__l8192__merge | 4 | g4b_C06_s16_d128_b64_l8192:B4:target | 14.471 [14.240,14.656] | -45.569 [-46.795,-44.782] | validated |
| op__g4b_C06_s16_d128_b8_l8192__b8__l8192__merge | 4 | g4b_C06_s16_d128_b8_l8192:B4:target | 6.762 [6.688,6.816] | -2.989 [-3.980,-2.375] | validated |
| op__g4b_C06_s32_d256_b1_l8192__b1__l8192__merge | 8 | g4b_C06_s32_d256_b1_l8192:actual:target | 6.229 [6.112,6.308] | 6.500 [6.363,6.665] | validated |
| op__g4b_C06_s32_d256_b64_l8192__b64__l8192__merge | 4 | g4b_C06_s32_d256_b64_l8192:B4:target | 30.959 [30.838,31.132] | 1.949 [1.694,2.263] | validated |
| op__g4b_C06_s32_d256_b8_l8192__b8__l8192__merge | 4 | g4b_C06_s32_d256_b8_l8192:B4:target | 7.340 [7.236,7.424] | 3.803 [2.822,4.564] | validated |

## Interpretation boundaries

- The new estimand is a single target semantic iteration after eight GPU-side primes, with input restoration outside its events and a fixed minimum host gap. It is not the historical unrestricted-cadence estimand or an idle-first-use measurement.
- U/G0 and same-program sham failures remain inconclusive controls; no sample/worker is dropped and no favorable equivalence is required by analysis.
- Exact aliases reuse a physical observation and never count as additional independent data. Each suite has three GPU allocations, not raw_sample_count independent GPUs.
- Every logical candidate is reported. Context minima are descriptive post-hoc summaries, not a pre-registered selector or a multiplicity-controlled ranking.
- Old/new comparisons are separate-cohort descriptive means and within-run gains. They are not paired causal differences, do not subtract a fixed idle penalty, and do not relabel old NCU reports.
- No model is refit and seen D1 geometries are not called new blind tests. Legacy and anchor defaults are explicit Gluon references, not invented source-aligned Triton U/G0 baselines.

Telemetry retains return codes and compressed raw samples. Its nominal clocks and hardware averaging windows do not establish per-kernel fixed frequency or identify a causal DVFS mechanism.
