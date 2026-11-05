# Fixed-cadence replacement: nonreduction

Data status: **OK**. Three workers × ten visits × 31 samples per physical condition; 139,500 target observations.
Each graph runs eight untimed semantic iterations, then exactly one timed iteration. Original pipeline zeroes remain inside each iteration; any input restoration precedes the target events. Between observations the requested minimum host gap is 1 ms; actual gaps remain recorded.

U/G0 equivalence: 15/16. Same-program sham equivalence: 16/16.
Those are logical workload comparisons; exact aliases can share data. Distinct physical U/G0 pairs: 16; distinct physical sham labels per worker: 16. They are not additional independent GPUs.
Equivalence requires the full paired CI95 to be inside ±3%. A candidate is marked strictly improved only when its fresh U and G0 lower bounds both exceed +3%, and the relevant controls pass. Legacy/anchor comparisons instead use their explicitly named Gluon reference.

All logical rows, physical aliases, visit medians, host-gap summaries, old/new descriptive comparisons and control decisions are retained in `analysis.json` and `timing.csv`. No unfavorable or inconclusive control is removed.

Recorded host gaps below the requested minimum: 0. Observed minimum/maximum: 1006.581 / 200420.996 µs. GPU-side primes are outside these host-gap durations and outside target timing.

## Control results

| Context | U/reference | G0 vs U gain % [CI95] | Same-program repeat gain % [CI95] | Status |
|---|---|---:|---:|---|
| N01_m11008_n128 | N01_m11008_n128:U | 1.084 [0.713,1.407] | 0.012 [-0.197,0.221] | validated |
| N01_m4096_n128 | N01_m4096_n128:U | 0.877 [0.115,2.045] | 0.049 [-0.196,0.294] | validated |
| N01_m4096_n8192 | N01_m4096_n8192:U | 0.085 [0.056,0.118] | 0.003 [-0.034,0.038] | validated |
| N02_k1024_p128_g128 | N02_k1024_p128_g128:U | -2.072 [-3.138,-0.193] | -0.035 [-0.227,0.156] | inconclusive_U_G0 |
| N02_k3584_p448_g128 | N02_k3584_p448_g128:U | 0.056 [-0.137,0.256] | 0.136 [0.016,0.263] | validated |
| N02_k8192_p1024_g128 | N02_k8192_p1024_g128:U | -0.156 [-0.371,0.028] | 0.047 [-0.287,0.317] | validated |
| N03_FA_D128 | N03_FA_D128:U | 0.101 [-0.390,0.576] | -0.233 [-0.800,0.363] | validated |
| N03_FA_D64 | N03_FA_D64:U | -0.023 [-0.488,0.384] | 0.076 [-0.152,0.513] | validated |
| N03_MROPE_D128 | N03_MROPE_D128:U | -0.372 [-0.539,-0.211] | 0.007 [-0.160,0.162] | validated |
| N03_MROPE_D64 | N03_MROPE_D64:U | 0.621 [0.339,0.845] | 0.102 [-0.181,0.395] | validated |
| N04_h32_s4096_d128 | N04_h32_s4096_d128:U | -0.126 [-0.422,0.169] | -0.073 [-0.335,0.183] | validated |
| N04_h8_s2048_d128 | N04_h8_s2048_d128:U | 0.436 [0.238,0.663] | 0.000 [-0.109,0.129] | validated |
| N05_t2048_h8_d128 | N05_t2048_h8_d128:U | 1.046 [0.314,1.681] | 0.027 [-0.082,0.136] | validated |
| N05_t2048_h8_d64 | N05_t2048_h8_d64:U | -0.300 [-1.066,0.217] | 0.033 [-0.083,0.150] | validated |
| N06_h32_d128 | N06_h32_d128:U | -0.086 [-0.212,0.031] | -0.015 [-0.175,0.134] | validated |
| N06_h8_d64 | N06_h8_d64:U | -0.557 [-1.075,0.263] | -0.028 [-0.125,0.069] | validated |

## Every context

The minimum candidate below is selected retrospectively for a compact inventory, not used as a deployed model. Full candidate and losing-layout data remain in the CSV/JSON.

| Context | Physical rows including sham | Descriptive minimum candidate | Target time µs [CI95] | Gain vs paired reference % [CI95] | Control |
|---|---:|---|---:|---:|---|
| N01_m11008_n128 | 11 | N01_m11008_n128:GL_output_fully_coalesced | 8.143 [8.075,8.204] | 5.927 [5.747,6.110] | validated |
| N01_m4096_n128 | 11 | N01_m4096_n128:GL_load_s1x2_t16x2_o10 | 6.220 [6.106,6.300] | 5.295 [4.431,6.299] | validated |
| N01_m4096_n8192 | 4 | N01_m4096_n8192:GL_coordinated_fully_coalesced | 73.101 [72.999,73.195] | 7.887 [7.846,7.930] | validated |
| N02_k1024_p128_g128 | 11 | N02_k1024_p128_g128:GL_packed_compute_vec1 | 5.893 [5.836,5.952] | 3.796 [3.537,4.025] | inconclusive_U_G0 |
| N02_k3584_p448_g128 | 11 | N02_k3584_p448_g128:GL_packed_compute_vec1 | 12.345 [12.258,12.412] | 7.126 [6.852,7.426] | validated |
| N02_k8192_p1024_g128 | 4 | N02_k8192_p1024_g128:GL_packed_compute_vec1 | 65.011 [64.782,65.180] | -1.910 [-2.195,-1.604] | validated |
| N03_FA_D128 | 8 | N03_FA_D128:GL_input_vec4 | 50.691 [50.476,50.932] | 0.281 [-0.154,0.737] | validated |
| N03_FA_D64 | 8 | N03_FA_D64:GL_paired_vec4 | 28.035 [27.949,28.107] | 0.334 [0.098,0.800] | validated |
| N03_MROPE_D128 | 8 | N03_MROPE_D128:GL_QK_vec2 | 14.467 [14.402,14.515] | 1.115 [0.575,1.918] | validated |
| N03_MROPE_D64 | 8 | N03_MROPE_D64:GL_K_vec2 | 9.388 [9.365,9.413] | 0.621 [0.339,0.845] | validated |
| N04_h32_s4096_d128 | 11 | N04_h32_s4096_d128:GL_v_input_vec8 | 61.097 [60.991,61.203] | 0.103 [-0.182,0.381] | validated |
| N04_h8_s2048_d128 | 11 | N04_h8_s2048_d128:GL_v_input_vec8 | 10.774 [10.688,10.833] | -0.079 [-0.804,0.546] | validated |
| N05_t2048_h8_d128 | 11 | N05_t2048_h8_d128:GL_feature_vec4 | 7.531 [7.511,7.545] | 4.102 [3.948,4.216] | validated |
| N05_t2048_h8_d64 | 11 | N05_t2048_h8_d64:GL_feature_vec4 | 6.290 [6.266,6.313] | 1.799 [1.415,2.188] | validated |
| N06_h32_d128 | 11 | N06_h32_d128:GL_e4_heads1 | 28.561 [28.442,28.701] | 0.138 [-0.141,0.317] | validated |
| N06_h8_d64 | 11 | N06_h8_d64:GL_e2_heads4 | 7.812 [7.710,7.898] | -1.963 [-3.104,-1.232] | validated |

## Previously frozen N4 choices

These candidates were fixed by the earlier N4 protocol. They are not reselected using the new matrix.

| Case | Frozen candidate | Gain vs U % [CI95] | Gain vs G0 % [CI95] | Strict dual-control >3% |
|---|---|---:|---:|---|
| N01_m11008_n128 | N01_m11008_n128:GL_coordinated_fully_coalesced | 5.816 [5.388,6.291] | 4.784 [4.393,5.191] | True |
| N01_m4096_n128 | N01_m4096_n128:GL_coordinated_fully_coalesced | 3.898 [2.732,5.346] | 3.048 [2.444,3.547] | False |
| N02_k1024_p128_g128 | N02_k1024_p128_g128:GL_packed_compute_vec1 | 3.796 [3.537,4.025] | 5.749 [4.187,6.592] | False |
| N02_k3584_p448_g128 | N02_k3584_p448_g128:GL_packed_compute_vec1 | 7.126 [6.852,7.426] | 7.074 [6.920,7.262] | True |
| N03_FA_D128 | N03_FA_D128:GL_paired_vec4 | -0.034 [-0.622,0.637] | -0.134 [-0.731,0.462] | False |
| N03_FA_D64 | N03_FA_D64:GL_paired_vec4 | 0.334 [0.098,0.800] | 0.356 [0.072,0.897] | False |
| N03_MROPE_D128 | N03_MROPE_D128:GL_QK_vec2 | 1.115 [0.575,1.918] | 1.482 [0.882,2.336] | False |
| N03_MROPE_D64 | N03_MROPE_D64:GL_K_vec2 | 0.621 [0.339,0.845] | 0.000 [0.000,0.000] | False |
| N04_h32_s4096_d128 | N04_h32_s4096_d128:GL_v_input_vec8 | 0.103 [-0.182,0.381] | 0.228 [-0.003,0.451] | False |
| N04_h8_s2048_d128 | N04_h8_s2048_d128:GL_v_input_vec8 | -0.079 [-0.804,0.546] | -0.517 [-1.055,0.040] | False |
| N05_t2048_h8_d128 | N05_t2048_h8_d128:GL_feature_vec4 | 4.102 [3.948,4.216] | 3.089 [2.511,3.667] | False |
| N05_t2048_h8_d64 | N05_t2048_h8_d64:GL_feature_vec4 | 1.799 [1.415,2.188] | 2.092 [1.642,2.471] | False |
| N06_h32_d128 | N06_h32_d128:GL_e4_heads1 | 0.138 [-0.141,0.317] | 0.224 [-0.060,0.413] | False |
| N06_h8_d64 | N06_h8_d64:GL_e2_heads4 | -1.963 [-3.104,-1.232] | -1.398 [-2.166,-0.291] | False |

## Interpretation boundaries

- The new estimand is a single target semantic iteration after eight GPU-side primes, with input restoration outside its events and a fixed minimum host gap. It is not the historical unrestricted-cadence estimand or an idle-first-use measurement.
- U/G0 and same-program sham failures remain inconclusive controls; no sample/worker is dropped and no favorable equivalence is required by analysis.
- Exact aliases reuse a physical observation and never count as additional independent data. Each suite has three GPU allocations, not raw_sample_count independent GPUs.
- Every logical candidate is reported. Context minima are descriptive post-hoc summaries, not a pre-registered selector or a multiplicity-controlled ranking.
- Old/new comparisons are separate-cohort descriptive means and within-run gains. They are not paired causal differences, do not subtract a fixed idle penalty, and do not relabel old NCU reports.
- No model is refit and seen D1 geometries are not called new blind tests. Legacy and anchor defaults are explicit Gluon references, not invented source-aligned Triton U/G0 baselines.

Telemetry retains return codes and compressed raw samples. Its nominal clocks and hardware averaging windows do not establish per-kernel fixed frequency or identify a causal DVFS mechanism.
