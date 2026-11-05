# Fixed-cadence replacement: anchors

Data status: **OK**. Three workers × ten visits × 31 samples per physical condition; 440,820 target observations.
Each graph runs eight untimed semantic iterations, then exactly one timed iteration. Original pipeline zeroes remain inside each iteration; any input restoration precedes the target events. Between observations the requested minimum host gap is 1 ms; actual gaps remain recorded.

U/G0 equivalence: 0/0. Same-program sham equivalence: 57/57.
Those are logical workload comparisons; exact aliases can share data. Distinct physical U/G0 pairs: 0; distinct physical sham labels per worker: 57. They are not additional independent GPUs.
Equivalence requires the full paired CI95 to be inside ±3%. A candidate is marked strictly improved only when its fresh U and G0 lower bounds both exceed +3%, and the relevant controls pass. Legacy/anchor comparisons instead use their explicitly named Gluon reference.

All logical rows, physical aliases, visit medians, host-gap summaries, old/new descriptive comparisons and control decisions are retained in `analysis.json` and `timing.csv`. No unfavorable or inconclusive control is removed.

Recorded host gaps below the requested minimum: 0. Observed minimum/maximum: 1007.284 / 51959.305 µs. GPU-side primes are outside these host-gap durations and outside target timing.

## Control results

| Context | U/reference | G0 vs U gain % [CI95] | Same-program repeat gain % [CI95] | Status |
|---|---|---:|---:|---|
| c2d|M16|N128|W4|qnatural|G32768 | c2d|M16_N128_w4:vec8:n_first:ONE_FULL|qnatural|G32768|K1 | unavailable | -0.134 [-0.446,0.167] | validated |
| c2d|M16|N128|W4|qnatural|G65536 | c2d|M16_N128_w4:vec8:n_first:ONE_FULL|qnatural|G65536|K1 | unavailable | 0.004 [-0.020,0.024] | validated |
| c2d|M16|N128|W4|qnatural|G98304 | c2d|M16_N128_w4:vec8:n_first:ONE_FULL|qnatural|G98304|K1 | unavailable | 0.024 [-0.056,0.096] | validated |
| c2d|M16|N64|W4|qnatural|G131072 | c2d|M16_N64_w4:vec8:n_first:ONE_FULL|qnatural|G131072|K1 | unavailable | 0.006 [-0.023,0.037] | validated |
| c2d|M16|N64|W4|qnatural|G196608 | c2d|M16_N64_w4:vec8:n_first:ONE_FULL|qnatural|G196608|K1 | unavailable | 0.014 [-0.005,0.033] | validated |
| c2d|M16|N64|W4|qnatural|G65536 | c2d|M16_N64_w4:vec8:n_first:ONE_FULL|qnatural|G65536|K1 | unavailable | -0.001 [-0.031,0.029] | validated |
| c2d|M16|N64|W8|qnatural|G131072 | c2d|M16_N64_w8:vec4:n_first:ONE_FULL|qnatural|G131072|K1 | unavailable | -0.008 [-0.026,0.012] | validated |
| c2d|M16|N64|W8|qnatural|G196608 | c2d|M16_N64_w8:vec4:n_first:ONE_FULL|qnatural|G196608|K1 | unavailable | -0.009 [-0.026,0.007] | validated |
| c2d|M16|N64|W8|qnatural|G65536 | c2d|M16_N64_w8:vec4:n_first:ONE_FULL|qnatural|G65536|K1 | unavailable | -0.005 [-0.074,0.053] | validated |
| c2d|M32|N128|W4|qnatural|G16384 | c2d|M32_N128_w4:vec8:n_first:ONE_FULL|qnatural|G16384|K1 | unavailable | 0.211 [-0.134,0.496] | validated |
| c2d|M32|N128|W4|qnatural|G32768 | c2d|M32_N128_w4:vec8:n_first:ONE_FULL|qnatural|G32768|K1 | unavailable | 0.000 [-0.021,0.025] | validated |
| c2d|M32|N128|W4|qnatural|G49152 | c2d|M32_N128_w4:vec8:n_first:ONE_FULL|qnatural|G49152|K1 | unavailable | -0.043 [-0.154,0.059] | validated |
| c2d|M32|N64|W8|qnatural|G32768 | c2d|M32_N64_w8:vec8:n_first:ONE_FULL|qnatural|G32768|K1 | unavailable | 0.002 [-0.019,0.023] | validated |
| c2d|M32|N64|W8|qnatural|G65536 | c2d|M32_N64_w8:vec8:n_first:ONE_FULL|qnatural|G65536|K1 | unavailable | 0.008 [-0.012,0.025] | validated |
| c2d|M32|N64|W8|qnatural|G98304 | c2d|M32_N64_w8:vec8:n_first:ONE_FULL|qnatural|G98304|K1 | unavailable | -0.001 [-0.013,0.010] | validated |
| c2d|M64|N64|W8|qnatural|G16384 | c2d|M64_N64_w8:vec8:n_first:ONE_FULL|qnatural|G16384|K1 | unavailable | 0.050 [-0.089,0.195] | validated |
| c2d|M64|N64|W8|qnatural|G32768 | c2d|M64_N64_w8:vec8:n_first:ONE_FULL|qnatural|G32768|K1 | unavailable | -0.035 [-0.067,0.000] | validated |
| c2d|M64|N64|W8|qnatural|G49152 | c2d|M64_N64_w8:vec8:n_first:ONE_FULL|qnatural|G49152|K1 | unavailable | -0.003 [-0.046,0.044] | validated |
| c2d|M8|N128|W4|qnatural|G131072 | c2d|M8_N128_w4:vec8:n_first:ONE_FULL|qnatural|G131072|K1 | unavailable | 0.001 [-0.015,0.015] | validated |
| c2d|M8|N128|W4|qnatural|G196608 | c2d|M8_N128_w4:vec8:n_first:ONE_FULL|qnatural|G196608|K1 | unavailable | -0.004 [-0.017,0.007] | validated |
| c2d|M8|N128|W4|qnatural|G65536 | c2d|M8_N128_w4:vec8:n_first:ONE_FULL|qnatural|G65536|K1 | unavailable | -0.007 [-0.034,0.017] | validated |
| c2d|M8|N256|W4|qnatural|G32768 | c2d|M8_N256_w4:vec8:n_first:ONE_FULL|qnatural|G32768|K1 | unavailable | -0.296 [-0.757,0.166] | validated |
| c2d|M8|N256|W4|qnatural|G65536 | c2d|M8_N256_w4:vec8:n_first:ONE_FULL|qnatural|G65536|K1 | unavailable | -0.031 [-0.089,0.031] | validated |
| c2d|M8|N256|W4|qnatural|G98304 | c2d|M8_N256_w4:vec8:n_first:ONE_FULL|qnatural|G98304|K1 | unavailable | 0.085 [-0.059,0.221] | validated |
| c2d|M8|N64|W8|qnatural|G131072 | c2d|M8_N64_w8:vec2:n_first:ONE_FULL|qnatural|G131072|K1 | unavailable | -0.001 [-0.035,0.037] | validated |
| c2d|M8|N64|W8|qnatural|G262144 | c2d|M8_N64_w8:vec2:n_first:ONE_FULL|qnatural|G262144|K1 | unavailable | -0.001 [-0.014,0.011] | validated |
| c2d|M8|N64|W8|qnatural|G393216 | c2d|M8_N64_w8:vec2:n_first:ONE_FULL|qnatural|G393216|K1 | unavailable | 0.010 [-0.003,0.022] | validated |
| d1|M128|N64|W2|qnatural|G16384 | d1|M128_N64_w2:vec8:n_first:ONE_FULL|qnatural|G16384|K1 | unavailable | 0.003 [-0.024,0.028] | validated |
| d1|M128|N64|W2|qnatural|G24576 | d1|M128_N64_w2:vec8:n_first:ONE_FULL|qnatural|G24576|K1 | unavailable | 0.024 [-0.041,0.092] | validated |
| d1|M128|N64|W2|qnatural|G8192 | d1|M128_N64_w2:vec8:n_first:ONE_FULL|qnatural|G8192|K1 | unavailable | -0.091 [-0.487,0.309] | validated |
| d1|M16|N128|W2|qnatural|G32768 | d1|M16_N128_w2:vec8:n_first:ONE_FULL|qnatural|G32768|K1 | unavailable | 0.058 [-0.400,0.520] | validated |
| d1|M16|N128|W2|qnatural|G65536 | d1|M16_N128_w2:vec8:n_first:ONE_FULL|qnatural|G65536|K1 | unavailable | -0.002 [-0.029,0.020] | validated |
| d1|M16|N128|W2|qnatural|G98304 | d1|M16_N128_w2:vec8:n_first:ONE_FULL|qnatural|G98304|K1 | unavailable | -0.067 [-0.281,0.155] | validated |
| d1|M16|N256|W8|qnatural|G16384 | d1|M16_N256_w8:vec8:n_first:ONE_FULL|qnatural|G16384|K1 | unavailable | -0.508 [-0.914,-0.038] | validated |
| d1|M16|N256|W8|qnatural|G32768 | d1|M16_N256_w8:vec8:n_first:ONE_FULL|qnatural|G32768|K1 | unavailable | -0.034 [-0.067,0.003] | validated |
| d1|M16|N256|W8|qnatural|G49152 | d1|M16_N256_w8:vec8:n_first:ONE_FULL|qnatural|G49152|K1 | unavailable | -0.161 [-0.508,0.167] | validated |
| d1|M32|N256|W2|qnatural|G16384 | d1|M32_N256_w2:vec8:n_first:ONE_FULL|qnatural|G16384|K1 | unavailable | -0.002 [-0.022,0.017] | validated |
| d1|M32|N256|W2|qnatural|G24576 | d1|M32_N256_w2:vec8:n_first:ONE_FULL|qnatural|G24576|K1 | unavailable | -0.045 [-0.203,0.099] | validated |
| d1|M32|N256|W2|qnatural|G8192 | d1|M32_N256_w2:vec8:n_first:ONE_FULL|qnatural|G8192|K1 | unavailable | -0.189 [-0.533,0.219] | validated |
| d1|M64|N128|W2|qnatural|G16384 | d1|M64_N128_w2:vec8:n_first:ONE_FULL|qnatural|G16384|K1 | unavailable | 0.021 [-0.002,0.043] | validated |
| d1|M64|N128|W2|qnatural|G24576 | d1|M64_N128_w2:vec8:n_first:ONE_FULL|qnatural|G24576|K1 | unavailable | -0.022 [-0.197,0.133] | validated |
| d1|M64|N128|W2|qnatural|G8192 | d1|M64_N128_w2:vec8:n_first:ONE_FULL|qnatural|G8192|K1 | unavailable | 0.351 [-0.056,0.765] | validated |
| d1|M64|N512|W4|qnatural|G2048 | d1|M64_N512_w4:vec8:n_first:ONE_FULL|qnatural|G2048|K1 | unavailable | 0.137 [-0.201,0.462] | validated |
| d1|M64|N512|W4|qnatural|G4096 | d1|M64_N512_w4:vec8:n_first:ONE_FULL|qnatural|G4096|K1 | unavailable | -0.011 [-0.044,0.021] | validated |
| d1|M64|N512|W4|qnatural|G6144 | d1|M64_N512_w4:vec8:n_first:ONE_FULL|qnatural|G6144|K1 | unavailable | -0.028 [-0.133,0.094] | validated |
| d1|M8|N128|W8|qnatural|G131072 | d1|M8_N128_w8:vec4:n_first:ONE_FULL|qnatural|G131072|K1 | unavailable | -0.015 [-0.059,0.026] | validated |
| d1|M8|N128|W8|qnatural|G196608 | d1|M8_N128_w8:vec4:n_first:ONE_FULL|qnatural|G196608|K1 | unavailable | -0.009 [-0.032,0.013] | validated |
| d1|M8|N128|W8|qnatural|G65536 | d1|M8_N128_w8:vec4:n_first:ONE_FULL|qnatural|G65536|K1 | unavailable | -0.004 [-0.055,0.048] | validated |
| d1|M8|N512|W4|qnatural|G16384 | d1|M8_N512_w4:vec8:n_first:ONE_FULL|qnatural|G16384|K1 | unavailable | -0.080 [-0.315,0.131] | validated |
| d1|M8|N512|W4|qnatural|G32768 | d1|M8_N512_w4:vec8:n_first:ONE_FULL|qnatural|G32768|K1 | unavailable | 0.007 [-0.027,0.037] | validated |
| d1|M8|N512|W4|qnatural|G49152 | d1|M8_N512_w4:vec8:n_first:ONE_FULL|qnatural|G49152|K1 | unavailable | -0.009 [-0.138,0.145] | validated |
| d1|M8|N512|W8|qnatural|G16384 | d1|M8_N512_w8:vec8:n_first:ONE_FULL|qnatural|G16384|K1 | unavailable | 0.267 [-0.144,0.710] | validated |
| d1|M8|N512|W8|qnatural|G32768 | d1|M8_N512_w8:vec8:n_first:ONE_FULL|qnatural|G32768|K1 | unavailable | -0.019 [-0.051,0.012] | validated |
| d1|M8|N512|W8|qnatural|G49152 | d1|M8_N512_w8:vec8:n_first:ONE_FULL|qnatural|G49152|K1 | unavailable | -0.006 [-0.221,0.203] | validated |
| d1|M8|N64|W4|qnatural|G131072 | d1|M8_N64_w4:vec4:n_first:ONE_FULL|qnatural|G131072|K1 | unavailable | -0.016 [-0.048,0.017] | validated |
| d1|M8|N64|W4|qnatural|G262144 | d1|M8_N64_w4:vec4:n_first:ONE_FULL|qnatural|G262144|K1 | unavailable | 0.014 [-0.010,0.038] | validated |
| d1|M8|N64|W4|qnatural|G393216 | d1|M8_N64_w4:vec4:n_first:ONE_FULL|qnatural|G393216|K1 | unavailable | 0.010 [-0.004,0.025] | validated |

## Every context

The minimum candidate below is selected retrospectively for a compact inventory, not used as a deployed model. Full candidate and losing-layout data remain in the CSV/JSON.

| Context | Physical rows including sham | Descriptive minimum candidate | Target time µs [CI95] | Gain vs paired reference % [CI95] | Control |
|---|---:|---|---:|---:|---|
| c2d|M16|N128|W4|qnatural|G32768 | 9 | c2d|M16_N128_w4:vec4:within:ONE_FULL|qnatural|G32768|K1 | 54.153 [53.860,54.466] | 9.325 [8.737,9.859] | validated |
| c2d|M16|N128|W4|qnatural|G65536 | 9 | c2d|M16_N128_w4:vec4:within:ONE_FULL|qnatural|G65536|K1 | 102.943 [102.802,103.146] | 8.732 [8.630,8.878] | validated |
| c2d|M16|N128|W4|qnatural|G98304 | 9 | c2d|M16_N128_w4:vec2:within:ONE_FULL|qnatural|G98304|K1 | 151.302 [150.802,151.859] | 8.847 [8.657,9.019] | validated |
| c2d|M16|N64|W4|qnatural|G131072 | 10 | c2d|M16_N64_w4:vec2:within:ONE_FULL|qnatural|G131072|K1 | 110.271 [110.137,110.523] | 37.907 [37.761,37.992] | validated |
| c2d|M16|N64|W4|qnatural|G196608 | 10 | c2d|M16_N64_w4:vec2:within:ONE_FULL|qnatural|G196608|K1 | 162.611 [162.370,162.912] | 38.056 [37.933,38.156] | validated |
| c2d|M16|N64|W4|qnatural|G65536 | 10 | c2d|M16_N64_w4:vec2:within:ONE_FULL|qnatural|G65536|K1 | 58.121 [57.856,58.384] | 37.007 [36.730,37.282] | validated |
| c2d|M16|N64|W8|qnatural|G131072 | 8 | c2d|M16_N64_w8:vec1:within:ONE_FULL|q8|G131072|K1 | 156.996 [156.883,157.100] | 25.167 [25.125,25.210] | validated |
| c2d|M16|N64|W8|qnatural|G196608 | 8 | c2d|M16_N64_w8:vec1:within:ONE_FULL|q8|G196608|K1 | 232.461 [232.369,232.553] | 25.455 [25.411,25.490] | validated |
| c2d|M16|N64|W8|qnatural|G65536 | 8 | c2d|M16_N64_w8:vec1:within:ONE_FULL|q8|G65536|K1 | 81.491 [81.462,81.522] | 24.953 [24.926,24.981] | validated |
| c2d|M32|N128|W4|qnatural|G16384 | 9 | c2d|M32_N128_w4:vec1:n_first:ONE_FULL|qnatural|G16384|K1 | 52.397 [52.192,52.592] | 2.216 [1.760,2.608] | validated |
| c2d|M32|N128|W4|qnatural|G32768 | 9 | c2d|M32_N128_w4:vec2:n_first:ONE_FULL|qnatural|G32768|K1 | 98.754 [98.397,99.410] | 1.178 [1.087,1.240] | validated |
| c2d|M32|N128|W4|qnatural|G49152 | 9 | c2d|M32_N128_w4:vec1:n_first:ONE_FULL|qnatural|G49152|K1 | 144.592 [144.037,145.532] | 0.745 [0.621,0.898] | validated |
| c2d|M32|N64|W8|qnatural|G32768 | 10 | c2d|M32_N64_w8:vec2:within:ONE_FULL|q8|G32768|K1 | 56.765 [56.343,57.266] | 47.557 [47.061,47.945] | validated |
| c2d|M32|N64|W8|qnatural|G65536 | 10 | c2d|M32_N64_w8:vec2:within:ONE_FULL|q8|G65536|K1 | 107.345 [106.998,107.885] | 48.619 [48.335,48.772] | validated |
| c2d|M32|N64|W8|qnatural|G98304 | 10 | c2d|M32_N64_w8:vec2:within:ONE_FULL|q8|G98304|K1 | 158.266 [157.729,158.971] | 48.912 [48.676,49.103] | validated |
| c2d|M64|N64|W8|qnatural|G16384 | 10 | c2d|M64_N64_w8:vec2:within:ONE_FULL|q8|G16384|K1 | 51.187 [50.979,51.412] | 19.927 [19.428,20.418] | validated |
| c2d|M64|N64|W8|qnatural|G32768 | 10 | c2d|M64_N64_w8:vec1:within:ONE_FULL|q8|G32768|K1 | 96.836 [96.732,97.012] | 19.375 [18.995,19.627] | validated |
| c2d|M64|N64|W8|qnatural|G49152 | 10 | c2d|M64_N64_w8:vec2:n_first:ONE_FULL|q8|G49152|K1 | 141.350 [141.182,141.621] | 19.704 [19.363,19.932] | validated |
| c2d|M8|N128|W4|qnatural|G131072 | 9 | c2d|M8_N128_w4:vec1:n_first:ONE_FULL|qnatural|G131072|K1 | 116.335 [116.212,116.436] | 40.207 [40.170,40.261] | validated |
| c2d|M8|N128|W4|qnatural|G196608 | 9 | c2d|M8_N128_w4:vec1:n_first:ONE_FULL|qnatural|G196608|K1 | 172.335 [172.170,172.434] | 40.150 [40.123,40.201] | validated |
| c2d|M8|N128|W4|qnatural|G65536 | 9 | c2d|M8_N128_w4:vec1:n_first:ONE_FULL|qnatural|G65536|K1 | 61.491 [61.250,61.687] | 38.867 [38.661,39.115] | validated |
| c2d|M8|N256|W4|qnatural|G32768 | 8 | c2d|M8_N256_w4:vec4:within:ONE_FULL|qnatural|G32768|K1 | 59.363 [59.097,59.643] | 2.587 [2.047,3.114] | validated |
| c2d|M8|N256|W4|qnatural|G65536 | 8 | c2d|M8_N256_w4:vec4:n_first:ONE_FULL|qnatural|G65536|K1 | 113.100 [112.945,113.312] | 1.344 [1.206,1.455] | validated |
| c2d|M8|N256|W4|qnatural|G98304 | 8 | c2d|M8_N256_w4:vec1:n_first:ONE_FULL|qnatural|G98304|K1 | 166.788 [166.547,167.026] | 1.701 [1.573,1.860] | validated |
| c2d|M8|N64|W8|qnatural|G131072 | 6 | c2d|M8_N64_w8:vec1:within:ONE_FULL|q8|G131072|K1 | 143.534 [143.303,143.678] | 9.929 [9.891,9.965] | validated |
| c2d|M8|N64|W8|qnatural|G262144 | 6 | c2d|M8_N64_w8:vec1:within:ONE_FULL|q8|G262144|K1 | 281.749 [281.174,282.086] | 10.125 [10.089,10.172] | validated |
| c2d|M8|N64|W8|qnatural|G393216 | 6 | c2d|M8_N64_w8:vec1:within:ONE_FULL|q8|G393216|K1 | 419.646 [418.891,420.073] | 10.198 [10.180,10.219] | validated |
| d1|M128|N64|W2|qnatural|G16384 | 9 | d1|M128_N64_w2:vec4:n_first:ONE_FULL|qnatural|G16384|K1 | 96.518 [95.794,97.868] | 0.119 [0.077,0.165] | validated |
| d1|M128|N64|W2|qnatural|G24576 | 9 | d1|M128_N64_w2:vec4:n_first:ONE_FULL|qnatural|G24576|K1 | 141.358 [140.213,143.531] | 0.124 [0.049,0.205] | validated |
| d1|M128|N64|W2|qnatural|G8192 | 9 | d1|M128_N64_w2:vec4:n_first:ONE_FULL|qnatural|G8192|K1 | 51.129 [50.828,51.529] | 0.517 [-0.094,1.086] | validated |
| d1|M16|N128|W2|qnatural|G32768 | 8 | d1|M16_N128_w2:vec2:n_first:ONE_FULL|qnatural|G32768|K1 | 54.796 [54.455,55.100] | 1.212 [0.625,1.790] | validated |
| d1|M16|N128|W2|qnatural|G65536 | 8 | d1|M16_N128_w2:vec1:n_first:ONE_FULL|qnatural|G65536|K1 | 102.998 [102.576,103.703] | 0.784 [0.727,0.833] | validated |
| d1|M16|N128|W2|qnatural|G98304 | 8 | d1|M16_N128_w2:vec4:within:ONE_FULL|qnatural|G98304|K1 | 151.115 [150.404,152.241] | 0.552 [0.378,0.734] | validated |
| d1|M16|N256|W8|qnatural|G16384 | 9 | d1|M16_N256_w8:vec4:within:ONE_FULL|q8|G16384|K1 | 54.530 [54.222,54.842] | 3.066 [2.487,3.659] | validated |
| d1|M16|N256|W8|qnatural|G32768 | 9 | d1|M16_N256_w8:vec2:n_first:ONE_FULL|q8|G32768|K1 | 103.020 [102.877,103.219] | 2.094 [1.976,2.246] | validated |
| d1|M16|N256|W8|qnatural|G49152 | 9 | d1|M16_N256_w8:vec2:within:ONE_FULL|q8|G49152|K1 | 151.309 [150.839,151.809] | 1.609 [1.348,1.867] | validated |
| d1|M32|N256|W2|qnatural|G16384 | 7 | d1|M32_N256_w2:vec4:n_first:ONE_FULL|qnatural|G16384|K1 | 99.186 [98.633,100.261] | 0.378 [0.338,0.421] | validated |
| d1|M32|N256|W2|qnatural|G24576 | 7 | d1|M32_N256_w2:vec2:n_first:ONE_FULL|qnatural|G24576|K1 | 145.217 [144.292,146.773] | 0.370 [0.267,0.520] | validated |
| d1|M32|N256|W2|qnatural|G8192 | 7 | d1|M32_N256_w2:vec2:n_first:ONE_FULL|qnatural|G8192|K1 | 52.649 [52.134,53.466] | 0.788 [0.227,1.348] | validated |
| d1|M64|N128|W2|qnatural|G16384 | 8 | d1|M64_N128_w2:vec4:within:ONE_FULL|qnatural|G16384|K1 | 97.011 [96.457,98.082] | 0.347 [0.325,0.368] | validated |
| d1|M64|N128|W2|qnatural|G24576 | 8 | d1|M64_N128_w2:vec4:within:ONE_FULL|qnatural|G24576|K1 | 142.285 [141.353,143.899] | 0.124 [-0.017,0.271] | validated |
| d1|M64|N128|W2|qnatural|G8192 | 8 | d1|M64_N128_w2:vec4:within:ONE_FULL|qnatural|G8192|K1 | 51.582 [51.275,51.958] | 0.619 [0.139,1.058] | validated |
| d1|M64|N512|W4|qnatural|G2048 | 7 | d1|M64_N512_w4:vec8:within:ONE_FULL|qnatural|G2048|K1 | 51.801 [51.413,52.315] | 0.624 [0.253,0.978] | validated |
| d1|M64|N512|W4|qnatural|G4096 | 7 | d1|M64_N512_w4:vec8:within:ONE_FULL|qnatural|G4096|K1 | 97.310 [96.828,98.243] | 0.089 [0.048,0.122] | validated |
| d1|M64|N512|W4|qnatural|G6144 | 7 | d1|M64_N512_w4:vec4:n_first:ONE_FULL|qnatural|G6144|K1 | 142.331 [141.531,143.717] | 0.206 [0.123,0.308] | validated |
| d1|M8|N128|W8|qnatural|G131072 | 8 | d1|M8_N128_w8:vec1:within:ONE_FULL|q8|G131072|K1 | 161.585 [161.221,161.846] | 21.841 [21.711,21.967] | validated |
| d1|M8|N128|W8|qnatural|G196608 | 8 | d1|M8_N128_w8:vec1:within:ONE_FULL|q8|G196608|K1 | 238.449 [237.862,238.887] | 22.146 [22.001,22.278] | validated |
| d1|M8|N128|W8|qnatural|G65536 | 8 | d1|M8_N128_w8:vec1:within:ONE_FULL|q8|G65536|K1 | 84.127 [83.782,84.319] | 21.452 [21.379,21.541] | validated |
| d1|M8|N512|W4|qnatural|G16384 | 7 | d1|M8_N512_w4:vec4:n_first:ONE_FULL|qnatural|G16384|K1 | 59.932 [59.450,60.608] | 1.764 [1.352,2.197] | validated |
| d1|M8|N512|W4|qnatural|G32768 | 7 | d1|M8_N512_w4:vec4:n_first:ONE_FULL|qnatural|G32768|K1 | 113.050 [112.609,113.889] | 1.031 [0.972,1.081] | validated |
| d1|M8|N512|W4|qnatural|G49152 | 7 | d1|M8_N512_w4:vec2:n_first:ONE_FULL|qnatural|G49152|K1 | 166.980 [166.189,168.341] | 0.574 [0.463,0.686] | validated |
| d1|M8|N512|W8|qnatural|G16384 | 8 | d1|M8_N512_w8:vec1:n_first:ONE_FULL|q8|G16384|K1 | 60.098 [59.483,60.825] | 1.860 [1.305,2.661] | validated |
| d1|M8|N512|W8|qnatural|G32768 | 8 | d1|M8_N512_w8:vec1:n_first:ONE_FULL|q8|G32768|K1 | 113.033 [112.866,113.315] | 0.860 [0.784,0.981] | validated |
| d1|M8|N512|W8|qnatural|G49152 | 8 | d1|M8_N512_w8:vec4:within:ONE_FULL|q8|G49152|K1 | 166.836 [166.642,167.019] | 0.680 [0.450,0.903] | validated |
| d1|M8|N64|W4|qnatural|G131072 | 8 | d1|M8_N64_w4:vec2:within:ONE_FULL|qnatural|G131072|K1 | 85.680 [85.539,85.814] | 19.766 [19.447,19.974] | validated |
| d1|M8|N64|W4|qnatural|G262144 | 8 | d1|M8_N64_w4:vec2:within:ONE_FULL|qnatural|G262144|K1 | 165.065 [164.916,165.198] | 20.033 [19.709,20.249] | validated |
| d1|M8|N64|W4|qnatural|G393216 | 8 | d1|M8_N64_w4:vec1:within:ONE_FULL|qnatural|G393216|K1 | 244.242 [244.075,244.485] | 20.313 [19.984,20.550] | validated |

## Anchor service slopes

OLS is applied to the same worker/visit medians across three grids. The slope in µs/CTA is multiplied by 132 SM × 1000 to obtain ns/tile/SM. Shared bootstrap draws preserve paired grid/default differences; negative or uncertain slopes are not truncated.

| Geometry | Program | Role | Service ns/tile/SM [CI95] | Gain vs default % [CI95] |
|---|---|---|---:|---:|
| M128_N64_w2 | M128_N64_w2:control:ONE_TRANSFER | control | 726.877 [719.063,739.956] | unavailable |
| M128_N64_w2 | M128_N64_w2:vec1:n_first:ONE_FULL | candidate | 727.349 [720.199,739.896] | -0.155 [-0.416,0.109] |
| M128_N64_w2 | M128_N64_w2:vec2:n_first:ONE_FULL | candidate | 726.421 [719.157,738.598] | -0.027 [-0.237,0.198] |
| M128_N64_w2 | M128_N64_w2:vec2:within:ONE_FULL | candidate | 727.813 [721.617,738.452] | -0.219 [-0.558,0.116] |
| M128_N64_w2 | M128_N64_w2:vec4:n_first:ONE_FULL | candidate | 726.945 [718.764,740.970] | -0.099 [-0.435,0.274] |
| M128_N64_w2 | M128_N64_w2:vec4:within:ONE_FULL | candidate | 729.644 [723.456,740.755] | -0.471 [-0.744,-0.191] |
| M128_N64_w2 | M128_N64_w2:vec8:n_first:ONE_FULL | default | 726.223 [718.822,738.755] | 0.000 [0.000,0.000] |
| M128_N64_w2 | M128_N64_w2:vec8:n_first:ONE_FULL | default_repeat | 725.570 [718.309,737.937] | 0.090 [-0.166,0.347] |
| M128_N64_w2 | M128_N64_w2:vec8:within:ONE_FULL | candidate | 726.739 [719.409,739.492] | -0.071 [-0.289,0.192] |
| M16_N128_w2 | M16_N128_w2:control:ONE_TRANSFER | control | 194.502 [192.775,197.297] | unavailable |
| M16_N128_w2 | M16_N128_w2:vec1:n_first:ONE_FULL | candidate | 194.408 [192.878,196.537] | -0.035 [-0.458,0.423] |
| M16_N128_w2 | M16_N128_w2:vec2:n_first:ONE_FULL | candidate | 194.335 [193.121,195.940] | 0.002 [-0.473,0.464] |
| M16_N128_w2 | M16_N128_w2:vec4:n_first:ONE_FULL | candidate | 194.229 [193.078,195.818] | 0.056 [-0.301,0.435] |
| M16_N128_w2 | M16_N128_w2:vec4:within:ONE_FULL | candidate | 193.982 [192.756,195.491] | 0.184 [-0.249,0.653] |
| M16_N128_w2 | M16_N128_w2:vec8:n_first:ONE_FULL | default | 194.339 [193.076,195.925] | 0.000 [0.000,0.000] |
| M16_N128_w2 | M16_N128_w2:vec8:n_first:ONE_FULL | default_repeat | 194.610 [193.396,196.048] | -0.139 [-0.601,0.267] |
| M16_N128_w2 | M16_N128_w2:vec8:within:ONE_FULL | candidate | 194.444 [193.406,195.540] | -0.054 [-0.504,0.403] |
| M16_N128_w4 | M16_N128_w4:control:ONE_TRANSFER | control | 195.433 [194.633,196.215] | unavailable |
| M16_N128_w4 | M16_N128_w4:vec1:n_first:ONE_FULL | candidate | 194.827 [193.076,196.451] | 8.975 [8.505,9.473] |
| M16_N128_w4 | M16_N128_w4:vec2:n_first:ONE_FULL | candidate | 194.376 [193.572,195.240] | 9.186 [8.591,9.771] |
| M16_N128_w4 | M16_N128_w4:vec2:within:ONE_FULL | candidate | 194.270 [193.039,195.794] | 9.235 [8.833,9.655] |
| M16_N128_w4 | M16_N128_w4:vec4:n_first:ONE_FULL | candidate | 195.104 [193.685,196.672] | 8.845 [8.070,9.557] |
| M16_N128_w4 | M16_N128_w4:vec4:within:ONE_FULL | candidate | 195.843 [194.932,196.666] | 8.500 [7.852,9.150] |
| M16_N128_w4 | M16_N128_w4:vec8:n_first:ONE_FULL | default | 214.036 [213.026,215.129] | 0.000 [0.000,0.000] |
| M16_N128_w4 | M16_N128_w4:vec8:n_first:ONE_FULL | default_repeat | 213.795 [212.891,214.952] | 0.112 [-0.081,0.315] |
| M16_N128_w4 | M16_N128_w4:vec8:within:ONE_FULL | candidate | 194.597 [193.690,195.483] | 9.082 [8.691,9.453] |
| M16_N256_w8 | M16_N256_w8:control:ONE_TRANSFER | control | 390.517 [387.926,393.311] | unavailable |
| M16_N256_w8 | M16_N256_w8:vec1:n_first:ONE_FULL | candidate | 389.963 [388.506,391.390] | 0.742 [0.202,1.227] |
| M16_N256_w8 | M16_N256_w8:vec2:n_first:ONE_FULL | candidate | 391.699 [389.855,393.457] | 0.300 [-0.172,0.831] |
| M16_N256_w8 | M16_N256_w8:vec2:within:ONE_FULL | candidate | 388.511 [386.633,390.328] | 1.111 [0.515,1.715] |
| M16_N256_w8 | M16_N256_w8:vec4:n_first:ONE_FULL | candidate | 390.921 [388.695,393.138] | 0.498 [-0.044,1.115] |
| M16_N256_w8 | M16_N256_w8:vec4:within:ONE_FULL | candidate | 390.887 [388.618,392.807] | 0.506 [0.056,1.032] |
| M16_N256_w8 | M16_N256_w8:vec8:n_first:ONE_FULL | default | 392.876 [391.634,393.993] | 0.000 [0.000,0.000] |
| M16_N256_w8 | M16_N256_w8:vec8:n_first:ONE_FULL | default_repeat | 392.721 [390.526,395.206] | 0.039 [-0.627,0.648] |
| M16_N256_w8 | M16_N256_w8:vec8:within:ONE_FULL | candidate | 389.224 [386.683,391.970] | 0.930 [0.296,1.577] |
| M16_N64_w4 | M16_N64_w4:control:ONE_TRANSFER | control | 103.948 [103.554,104.342] | unavailable |
| M16_N64_w4 | M16_N64_w4:vec1:n_first:ONE_FULL | candidate | 105.424 [104.996,105.791] | 38.512 [38.304,38.744] |
| M16_N64_w4 | M16_N64_w4:vec1:within:ONE_FULL | candidate | 105.756 [105.474,106.058] | 38.319 [38.148,38.474] |
| M16_N64_w4 | M16_N64_w4:vec2:n_first:ONE_FULL | candidate | 106.426 [105.931,106.987] | 37.928 [37.593,38.219] |
| M16_N64_w4 | M16_N64_w4:vec2:within:ONE_FULL | candidate | 105.230 [104.969,105.501] | 38.625 [38.470,38.769] |
| M16_N64_w4 | M16_N64_w4:vec4:n_first:ONE_FULL | candidate | 113.603 [113.233,114.044] | 33.742 [33.476,33.969] |
| M16_N64_w4 | M16_N64_w4:vec4:within:ONE_FULL | candidate | 106.878 [106.485,107.245] | 37.664 [37.441,37.912] |
| M16_N64_w4 | M16_N64_w4:vec8:n_first:ONE_FULL | default | 171.455 [171.380,171.544] | 0.000 [0.000,0.000] |
| M16_N64_w4 | M16_N64_w4:vec8:n_first:ONE_FULL | default_repeat | 171.417 [171.348,171.505] | 0.022 [-0.014,0.058] |
| M16_N64_w4 | M16_N64_w4:vec8:within:ONE_FULL | candidate | 151.038 [150.962,151.114] | 11.908 [11.877,11.938] |
| M16_N64_w8 | M16_N64_w8:control:ONE_TRANSFER | control | 139.044 [138.917,139.223] | unavailable |
| M16_N64_w8 | M16_N64_w8:vec1:n_first:ONE_FULL | candidate | 156.057 [156.006,156.115] | 23.759 [23.723,23.796] |
| M16_N64_w8 | M16_N64_w8:vec1:within:ONE_FULL | candidate | 152.038 [151.935,152.145] | 25.723 [25.658,25.776] |
| M16_N64_w8 | M16_N64_w8:vec2:n_first:ONE_FULL | candidate | 169.608 [169.546,169.667] | 17.139 [17.103,17.175] |
| M16_N64_w8 | M16_N64_w8:vec2:within:ONE_FULL | candidate | 153.838 [153.752,153.926] | 24.844 [24.790,24.889] |
| M16_N64_w8 | M16_N64_w8:vec4:n_first:ONE_FULL | default | 204.690 [204.634,204.745] | 0.000 [0.000,0.000] |
| M16_N64_w8 | M16_N64_w8:vec4:n_first:ONE_FULL | default_repeat | 204.712 [204.648,204.775] | -0.010 [-0.035,0.013] |
| M16_N64_w8 | M16_N64_w8:vec4:within:ONE_FULL | candidate | 171.556 [171.476,171.634] | 16.188 [16.147,16.232] |
| M32_N128_w4 | M32_N128_w4:control:ONE_TRANSFER | control | 371.491 [368.873,374.754] | unavailable |
| M32_N128_w4 | M32_N128_w4:vec1:n_first:ONE_FULL | candidate | 371.392 [368.912,374.856] | -0.111 [-0.438,0.233] |
| M32_N128_w4 | M32_N128_w4:vec2:n_first:ONE_FULL | candidate | 371.177 [368.479,374.733] | -0.053 [-0.519,0.358] |
| M32_N128_w4 | M32_N128_w4:vec2:within:ONE_FULL | candidate | 371.508 [368.745,375.104] | -0.142 [-0.562,0.243] |
| M32_N128_w4 | M32_N128_w4:vec4:n_first:ONE_FULL | candidate | 370.825 [368.758,373.411] | 0.042 [-0.277,0.365] |
| M32_N128_w4 | M32_N128_w4:vec4:within:ONE_FULL | candidate | 370.532 [367.992,373.984] | 0.120 [-0.283,0.501] |
| M32_N128_w4 | M32_N128_w4:vec8:n_first:ONE_FULL | default | 370.979 [369.286,373.270] | 0.000 [0.000,0.000] |
| M32_N128_w4 | M32_N128_w4:vec8:n_first:ONE_FULL | default_repeat | 371.688 [369.384,374.443] | -0.191 [-0.440,0.078] |
| M32_N128_w4 | M32_N128_w4:vec8:within:ONE_FULL | candidate | 370.210 [368.105,372.853] | 0.207 [-0.074,0.472] |
| M32_N256_w2 | M32_N256_w2:control:ONE_TRANSFER | control | 746.333 [739.750,756.252] | unavailable |
| M32_N256_w2 | M32_N256_w2:vec1:n_first:ONE_FULL | candidate | 746.445 [740.781,755.012] | 0.043 [-0.301,0.361] |
| M32_N256_w2 | M32_N256_w2:vec2:n_first:ONE_FULL | candidate | 745.791 [741.434,751.498] | 0.130 [-0.301,0.616] |
| M32_N256_w2 | M32_N256_w2:vec4:n_first:ONE_FULL | candidate | 747.373 [741.477,755.365] | -0.082 [-0.509,0.314] |
| M32_N256_w2 | M32_N256_w2:vec8:n_first:ONE_FULL | default | 746.763 [740.678,756.259] | 0.000 [0.000,0.000] |
| M32_N256_w2 | M32_N256_w2:vec8:n_first:ONE_FULL | default_repeat | 746.487 [740.601,755.616] | 0.037 [-0.225,0.327] |
| M32_N256_w2 | M32_N256_w2:vec8:within:ONE_FULL | candidate | 744.923 [739.982,752.182] | 0.246 [-0.151,0.660] |
| M32_N64_w8 | M32_N64_w8:control:ONE_TRANSFER | control | 200.617 [199.962,201.450] | unavailable |
| M32_N64_w8 | M32_N64_w8:vec1:n_first:ONE_FULL | candidate | 207.926 [207.201,208.627] | 48.781 [48.614,48.953] |
| M32_N64_w8 | M32_N64_w8:vec1:within:ONE_FULL | candidate | 205.633 [204.675,206.618] | 49.346 [49.104,49.586] |
| M32_N64_w8 | M32_N64_w8:vec2:n_first:ONE_FULL | candidate | 208.869 [207.460,209.896] | 48.549 [48.298,48.905] |
| M32_N64_w8 | M32_N64_w8:vec2:within:ONE_FULL | candidate | 204.439 [203.669,205.049] | 49.640 [49.487,49.836] |
| M32_N64_w8 | M32_N64_w8:vec4:n_first:ONE_FULL | candidate | 231.758 [231.352,232.173] | 42.910 [42.810,43.015] |
| M32_N64_w8 | M32_N64_w8:vec4:within:ONE_FULL | candidate | 211.050 [210.216,211.881] | 48.011 [47.810,48.223] |
| M32_N64_w8 | M32_N64_w8:vec8:n_first:ONE_FULL | default | 405.954 [405.853,406.047] | 0.000 [0.000,0.000] |
| M32_N64_w8 | M32_N64_w8:vec8:n_first:ONE_FULL | default_repeat | 405.967 [405.894,406.035] | -0.003 [-0.019,0.012] |
| M32_N64_w8 | M32_N64_w8:vec8:within:ONE_FULL | candidate | 308.707 [308.625,308.791] | 23.955 [23.931,23.981] |
| M64_N128_w2 | M64_N128_w2:control:ONE_TRANSFER | control | 731.311 [726.241,739.286] | unavailable |
| M64_N128_w2 | M64_N128_w2:vec1:n_first:ONE_FULL | candidate | 731.852 [726.120,741.290] | -0.310 [-0.604,0.019] |
| M64_N128_w2 | M64_N128_w2:vec2:n_first:ONE_FULL | candidate | 729.670 [724.436,738.205] | -0.011 [-0.281,0.232] |
| M64_N128_w2 | M64_N128_w2:vec4:n_first:ONE_FULL | candidate | 732.102 [727.117,739.857] | -0.344 [-0.702,0.031] |
| M64_N128_w2 | M64_N128_w2:vec4:within:ONE_FULL | candidate | 730.761 [724.754,740.465] | -0.160 [-0.437,0.133] |
| M64_N128_w2 | M64_N128_w2:vec8:n_first:ONE_FULL | default | 729.592 [723.514,739.716] | 0.000 [0.000,0.000] |
| M64_N128_w2 | M64_N128_w2:vec8:n_first:ONE_FULL | default_repeat | 731.320 [725.983,739.762] | -0.237 [-0.528,0.090] |
| M64_N128_w2 | M64_N128_w2:vec8:within:ONE_FULL | candidate | 729.738 [723.963,738.858] | -0.020 [-0.271,0.249] |
| M64_N512_w4 | M64_N512_w4:control:ONE_TRANSFER | control | 2926.825 [2906.234,2954.023] | unavailable |
| M64_N512_w4 | M64_N512_w4:vec1:n_first:ONE_FULL | candidate | 2916.650 [2900.872,2938.169] | -0.006 [-0.362,0.409] |
| M64_N512_w4 | M64_N512_w4:vec2:n_first:ONE_FULL | candidate | 2951.231 [2933.150,2976.092] | -1.192 [-1.571,-0.854] |
| M64_N512_w4 | M64_N512_w4:vec4:n_first:ONE_FULL | candidate | 2915.206 [2895.853,2944.288] | 0.044 [-0.215,0.310] |
| M64_N512_w4 | M64_N512_w4:vec8:n_first:ONE_FULL | default | 2916.478 [2897.091,2948.412] | 0.000 [0.000,0.000] |
| M64_N512_w4 | M64_N512_w4:vec8:n_first:ONE_FULL | default_repeat | 2920.088 [2901.834,2946.606] | -0.124 [-0.376,0.139] |
| M64_N512_w4 | M64_N512_w4:vec8:within:ONE_FULL | candidate | 2919.572 [2901.766,2946.495] | -0.106 [-0.361,0.181] |
| M64_N64_w8 | M64_N64_w8:control:ONE_TRANSFER | control | 363.821 [362.393,365.376] | unavailable |
| M64_N64_w8 | M64_N64_w8:vec1:n_first:ONE_FULL | candidate | 363.327 [362.175,364.719] | 19.550 [19.126,19.966] |
| M64_N64_w8 | M64_N64_w8:vec1:within:ONE_FULL | candidate | 362.278 [361.015,363.486] | 19.782 [19.321,20.176] |
| M64_N64_w8 | M64_N64_w8:vec2:n_first:ONE_FULL | candidate | 361.982 [360.882,363.090] | 19.848 [19.468,20.199] |
| M64_N64_w8 | M64_N64_w8:vec2:within:ONE_FULL | candidate | 363.245 [362.192,364.212] | 19.568 [19.288,19.820] |
| M64_N64_w8 | M64_N64_w8:vec4:n_first:ONE_FULL | candidate | 362.364 [361.090,363.468] | 19.763 [19.414,20.075] |
| M64_N64_w8 | M64_N64_w8:vec4:within:ONE_FULL | candidate | 362.321 [361.015,363.774] | 19.773 [19.270,20.249] |
| M64_N64_w8 | M64_N64_w8:vec8:n_first:ONE_FULL | default | 451.619 [449.272,453.174] | 0.000 [0.000,0.000] |
| M64_N64_w8 | M64_N64_w8:vec8:n_first:ONE_FULL | default_repeat | 451.769 [449.041,453.445] | -0.033 [-0.127,0.072] |
| M64_N64_w8 | M64_N64_w8:vec8:within:ONE_FULL | candidate | 388.622 [386.865,390.216] | 13.949 [13.801,14.114] |
| M8_N128_w4 | M8_N128_w4:control:ONE_TRANSFER | control | 111.140 [110.793,111.481] | unavailable |
| M8_N128_w4 | M8_N128_w4:vec1:n_first:ONE_FULL | candidate | 111.629 [111.390,111.896] | 40.838 [40.692,40.964] |
| M8_N128_w4 | M8_N128_w4:vec2:n_first:ONE_FULL | candidate | 114.225 [113.978,114.493] | 39.462 [39.315,39.598] |
| M8_N128_w4 | M8_N128_w4:vec2:within:ONE_FULL | candidate | 112.136 [111.866,112.389] | 40.569 [40.438,40.710] |
| M8_N128_w4 | M8_N128_w4:vec4:n_first:ONE_FULL | candidate | 117.641 [117.413,117.927] | 37.651 [37.493,37.774] |
| M8_N128_w4 | M8_N128_w4:vec4:within:ONE_FULL | candidate | 113.302 [113.107,113.522] | 39.951 [39.835,40.057] |
| M8_N128_w4 | M8_N128_w4:vec8:n_first:ONE_FULL | default | 188.682 [188.648,188.719] | 0.000 [0.000,0.000] |
| M8_N128_w4 | M8_N128_w4:vec8:n_first:ONE_FULL | default_repeat | 188.688 [188.639,188.736] | -0.003 [-0.024,0.019] |
| M8_N128_w4 | M8_N128_w4:vec8:within:ONE_FULL | candidate | 140.780 [140.629,141.035] | 25.388 [25.241,25.477] |
| M8_N128_w8 | M8_N128_w8:control:ONE_TRANSFER | control | 144.111 [143.891,144.341] | unavailable |
| M8_N128_w8 | M8_N128_w8:vec1:n_first:ONE_FULL | candidate | 160.530 [160.266,160.741] | 19.969 [19.882,20.052] |
| M8_N128_w8 | M8_N128_w8:vec1:within:ONE_FULL | candidate | 155.415 [155.181,155.681] | 22.519 [22.335,22.671] |
| M8_N128_w8 | M8_N128_w8:vec2:n_first:ONE_FULL | candidate | 177.939 [177.824,178.043] | 11.290 [11.184,11.430] |
| M8_N128_w8 | M8_N128_w8:vec2:within:ONE_FULL | candidate | 156.527 [156.237,156.727] | 21.965 [21.848,22.058] |
| M8_N128_w8 | M8_N128_w8:vec4:n_first:ONE_FULL | default | 200.585 [200.343,200.941] | 0.000 [0.000,0.000] |
| M8_N128_w8 | M8_N128_w8:vec4:n_first:ONE_FULL | default_repeat | 200.607 [200.392,200.952] | -0.011 [-0.052,0.024] |
| M8_N128_w8 | M8_N128_w8:vec4:within:ONE_FULL | candidate | 167.448 [167.100,167.679] | 16.520 [16.365,16.632] |
| M8_N256_w4 | M8_N256_w4:control:ONE_TRANSFER | control | 215.785 [215.202,216.395] | unavailable |
| M8_N256_w4 | M8_N256_w4:vec1:n_first:ONE_FULL | candidate | 215.542 [214.934,216.185] | 1.583 [1.221,1.944] |
| M8_N256_w4 | M8_N256_w4:vec2:n_first:ONE_FULL | candidate | 215.669 [215.059,216.268] | 1.525 [1.158,1.858] |
| M8_N256_w4 | M8_N256_w4:vec4:n_first:ONE_FULL | candidate | 215.480 [214.657,216.367] | 1.612 [1.087,2.053] |
| M8_N256_w4 | M8_N256_w4:vec4:within:ONE_FULL | candidate | 216.487 [215.705,217.295] | 1.152 [0.774,1.540] |
| M8_N256_w4 | M8_N256_w4:vec8:n_first:ONE_FULL | default | 219.010 [218.459,219.557] | 0.000 [0.000,0.000] |
| M8_N256_w4 | M8_N256_w4:vec8:n_first:ONE_FULL | default_repeat | 218.356 [217.798,218.868] | 0.298 [-0.052,0.664] |
| M8_N256_w4 | M8_N256_w4:vec8:within:ONE_FULL | candidate | 215.834 [215.258,216.410] | 1.450 [1.059,1.845] |
| M8_N512_w4 | M8_N512_w4:control:ONE_TRANSFER | control | 430.697 [428.639,433.606] | unavailable |
| M8_N512_w4 | M8_N512_w4:vec1:n_first:ONE_FULL | candidate | 431.050 [428.386,434.934] | -0.064 [-0.359,0.222] |
| M8_N512_w4 | M8_N512_w4:vec2:n_first:ONE_FULL | candidate | 430.783 [428.476,433.973] | -0.002 [-0.254,0.272] |
| M8_N512_w4 | M8_N512_w4:vec4:n_first:ONE_FULL | candidate | 431.943 [429.528,435.085] | -0.271 [-0.549,0.017] |
| M8_N512_w4 | M8_N512_w4:vec8:n_first:ONE_FULL | default | 430.775 [428.278,434.137] | 0.000 [0.000,0.000] |
| M8_N512_w4 | M8_N512_w4:vec8:n_first:ONE_FULL | default_repeat | 430.637 [428.695,433.520] | 0.032 [-0.220,0.284] |
| M8_N512_w4 | M8_N512_w4:vec8:within:ONE_FULL | candidate | 430.543 [428.282,433.911] | 0.054 [-0.206,0.315] |
| M8_N512_w8 | M8_N512_w8:control:ONE_TRANSFER | control | 430.852 [429.249,432.584] | unavailable |
| M8_N512_w8 | M8_N512_w8:vec1:n_first:ONE_FULL | candidate | 430.809 [428.458,433.246] | -0.191 [-0.743,0.332] |
| M8_N512_w8 | M8_N512_w8:vec2:n_first:ONE_FULL | candidate | 429.584 [428.381,430.870] | 0.094 [-0.278,0.407] |
| M8_N512_w8 | M8_N512_w8:vec4:n_first:ONE_FULL | candidate | 430.586 [429.442,431.982] | -0.139 [-0.536,0.190] |
| M8_N512_w8 | M8_N512_w8:vec4:within:ONE_FULL | candidate | 429.576 [427.801,431.316] | 0.096 [-0.341,0.521] |
| M8_N512_w8 | M8_N512_w8:vec8:n_first:ONE_FULL | default | 429.988 [429.133,430.843] | 0.000 [0.000,0.000] |
| M8_N512_w8 | M8_N512_w8:vec8:n_first:ONE_FULL | default_repeat | 430.684 [428.978,432.388] | -0.162 [-0.577,0.282] |
| M8_N512_w8 | M8_N512_w8:vec8:within:ONE_FULL | candidate | 430.805 [428.391,432.940] | -0.190 [-0.695,0.350] |
| M8_N64_w4 | M8_N64_w4:control:ONE_TRANSFER | control | 79.748 [79.708,79.791] | unavailable |
| M8_N64_w4 | M8_N64_w4:vec1:n_first:ONE_FULL | candidate | 79.905 [79.858,79.952] | 20.543 [20.219,20.772] |
| M8_N64_w4 | M8_N64_w4:vec1:within:ONE_FULL | candidate | 79.830 [79.785,79.889] | 20.617 [20.292,20.853] |
| M8_N64_w4 | M8_N64_w4:vec2:n_first:ONE_FULL | candidate | 80.567 [80.507,80.637] | 19.884 [19.589,20.107] |
| M8_N64_w4 | M8_N64_w4:vec2:within:ONE_FULL | candidate | 79.852 [79.799,79.912] | 20.595 [20.268,20.836] |
| M8_N64_w4 | M8_N64_w4:vec4:n_first:ONE_FULL | default | 100.564 [100.135,100.815] | 0.000 [0.000,0.000] |
| M8_N64_w4 | M8_N64_w4:vec4:n_first:ONE_FULL | default_repeat | 100.540 [100.124,100.775] | 0.023 [-0.006,0.057] |
| M8_N64_w4 | M8_N64_w4:vec4:within:ONE_FULL | candidate | 80.439 [80.304,80.543] | 20.012 [19.813,20.137] |
| M8_N64_w8 | M8_N64_w8:control:ONE_TRANSFER | control | 125.149 [124.780,125.351] | unavailable |
| M8_N64_w8 | M8_N64_w8:vec1:n_first:ONE_FULL | candidate | 142.294 [141.926,142.509] | 8.235 [8.196,8.271] |
| M8_N64_w8 | M8_N64_w8:vec1:within:ONE_FULL | candidate | 139.033 [138.774,139.187] | 10.337 [10.305,10.367] |
| M8_N64_w8 | M8_N64_w8:vec2:n_first:ONE_FULL | default | 155.063 [154.724,155.251] | 0.000 [0.000,0.000] |
| M8_N64_w8 | M8_N64_w8:vec2:n_first:ONE_FULL | default_repeat | 155.038 [154.689,155.237] | 0.016 [-0.004,0.035] |
| M8_N64_w8 | M8_N64_w8:vec2:within:ONE_FULL | candidate | 141.237 [140.871,141.437] | 8.917 [8.884,8.953] |

## Interpretation boundaries

- The new estimand is a single target semantic iteration after eight GPU-side primes, with input restoration outside its events and a fixed minimum host gap. It is not the historical unrestricted-cadence estimand or an idle-first-use measurement.
- U/G0 and same-program sham failures remain inconclusive controls; no sample/worker is dropped and no favorable equivalence is required by analysis.
- Exact aliases reuse a physical observation and never count as additional independent data. Each suite has three GPU allocations, not raw_sample_count independent GPUs.
- Every logical candidate is reported. Context minima are descriptive post-hoc summaries, not a pre-registered selector or a multiplicity-controlled ranking.
- Old/new comparisons are separate-cohort descriptive means and within-run gains. They are not paired causal differences, do not subtract a fixed idle penalty, and do not relabel old NCU reports.
- No model is refit and seen D1 geometries are not called new blind tests. Legacy and anchor defaults are explicit Gluon references, not invented source-aligned Triton U/G0 baselines.

Telemetry retains return codes and compressed raw samples. Its nominal clocks and hardware averaging windows do not establish per-kernel fixed frequency or identify a causal DVFS mechanism.
