# 固定预热与采样间隙复测：旧证据与新证据汇总

状态：**OK**；469,650 个目标计时样本。此文件只汇总已有冻结分析，不调参、不重新选模型、不改变旧数据。

| Suite | Contexts | 原始样本 | U/G0 实用等价（逻辑比较） | Sham 实用等价（逻辑比较） |
|---|---:|---:|---:|---:|
| operators | 64 | 469,650 | 78/78 | 78/78 |

## 完整 G8 候选矩阵

361 条逻辑候选比较的双控制 CI 下界均超过 +3%：旧 44 条，新 40 条。别名仍共享原始物理观测；这些不是独立 GPU 数，也不是家族错误率保证。

全部 361 条及未通过的新控制均保留在 summary.json。下列是旧 G8 报告已经讨论的固定问题列表，不根据新结果筛选。

| Context / candidate | 旧相对 U % [CI95] | 新相对 U % [CI95] | 新相对 G0 % [CI95] | 状态 |
|---|---:|---:|---:|---|
| op__C01_p128_n4096__merge__M128 / C01_p128_n4096:vec4:n_first | 19.699 [18.834,20.644] | 20.109 [19.390,21.075] | 20.544 [19.618,21.107] | retained |
| op__C01_p128_n4096__merge__M128 / C01_p128_n4096:vec4:within | 19.311 [18.255,20.229] | 18.600 [17.108,19.583] | 19.042 [18.685,19.378] | retained |
| op__C01_p128_n4096__pipeline__M2048 / C01_p128_n4096:vec4:n_first | 4.556 [4.219,4.866] | 4.227 [4.141,4.306] | 4.257 [4.163,4.343] | retained |
| op__C01_p128_n4096__pipeline__M2048 / C01_p128_n4096:vec4:within | 4.524 [4.204,4.878] | 2.536 [2.456,2.614] | 2.567 [2.478,2.655] | not_resolved_in_new_context |
| op__C01_p256_n1024__merge__M128 / C01_p256_n1024:vec4:n_first | 27.876 [27.095,28.509] | 28.835 [27.974,29.359] | 27.678 [26.916,28.226] | retained |
| op__C01_p256_n1024__merge__M128 / C01_p256_n1024:vec4:within | 26.325 [25.213,27.119] | 27.011 [26.338,27.877] | 25.825 [25.221,26.725] | retained |
| op__C01_p256_n1024__pipeline__M2048 / C01_p256_n1024:vec4:n_first | 11.364 [11.009,11.737] | 11.072 [10.896,11.260] | 10.841 [10.723,10.950] | retained |
| op__C01_p256_n1024__pipeline__M2048 / C01_p256_n1024:vec4:within | 10.110 [9.866,10.321] | 10.023 [9.473,10.407] | 9.790 [9.436,10.059] | retained |
| op__C01_p96_n8192__merge__M128 / C01_p96_n8192:vec4:n_first | 15.726 [14.644,16.832] | 13.507 [12.766,13.923] | 14.362 [14.176,14.583] | retained |
| op__C01_p96_n8192__merge__M128 / C01_p96_n8192:vec4:within | 15.755 [14.976,16.475] | 15.263 [14.525,16.236] | 16.101 [15.353,16.660] | retained |
| op__C01_p96_n8192__pipeline__M2048 / C01_p96_n8192:vec4:n_first | 2.158 [1.898,2.431] | 1.350 [1.205,1.551] | 2.467 [2.369,2.572] | neither_resolved |
| op__C01_p96_n8192__pipeline__M2048 / C01_p96_n8192:vec4:within | 2.265 [2.029,2.493] | 1.205 [1.131,1.280] | 2.323 [2.202,2.431] | neither_resolved |
| op__g2_C06_s16_d128__b64__l8192__merge / g2_C06_s16_d128:vec1:n_first | -45.365 [-46.717,-43.477] | -46.146 [-46.822,-45.183] | -44.583 [-44.898,-44.196] | neither_resolved |
| op__g2_C06_s16_d128__b64__l8192__merge / g2_C06_s16_d128:vec2:within | -10.100 [-10.710,-9.550] | -11.145 [-11.992,-9.869] | -9.956 [-10.373,-9.563] | neither_resolved |
| op__g2_C06_s16_d128__b64__l8192__merge / g2_C06_s16_d128:vec4:within | 7.501 [7.132,7.948] | 7.264 [6.638,8.009] | 8.256 [8.060,8.453] | retained |
| op__g2_C06_s16_d128__b8__l8192__merge / g2_C06_s16_d128:vec1:n_first | -2.610 [-3.580,-1.520] | -2.439 [-2.959,-1.900] | -3.290 [-4.078,-2.834] | neither_resolved |
| op__g2_C06_s16_d128__b8__l8192__merge / g2_C06_s16_d128:vec2:within | 4.233 [3.801,4.709] | 3.942 [2.820,4.620] | 3.144 [2.376,4.468] | neither_resolved |
| op__g2_C06_s16_d128__b8__l8192__merge / g2_C06_s16_d128:vec4:within | 3.883 [3.237,4.606] | 3.554 [2.448,4.330] | 2.753 [2.006,3.981] | neither_resolved |
| op__g2_C06_s32_d128__b64__l8192__merge / g2_C06_s32_d128:vec1:n_first | -74.739 [-76.417,-72.117] | -71.630 [-72.162,-71.168] | -71.415 [-71.746,-71.068] | neither_resolved |
| op__g2_C06_s32_d128__b64__l8192__merge / g2_C06_s32_d128:vec2:within | -17.491 [-18.382,-16.412] | -16.815 [-17.132,-16.503] | -16.668 [-16.869,-16.482] | neither_resolved |
| op__g2_C06_s32_d128__b64__l8192__merge / g2_C06_s32_d128:vec4:within | 7.241 [6.894,7.576] | 7.235 [7.003,7.465] | 7.351 [7.224,7.476] | retained |
| op__g2_C06_s32_d128__b8__l8192__merge / g2_C06_s32_d128:vec1:n_first | -23.634 [-24.886,-22.007] | -24.143 [-25.476,-22.374] | -23.660 [-24.694,-21.892] | neither_resolved |
| op__g2_C06_s32_d128__b8__l8192__merge / g2_C06_s32_d128:vec2:within | -0.311 [-0.699,0.093] | -0.391 [-0.673,0.000] | -0.000 [-0.657,0.413] | neither_resolved |
| op__g2_C06_s32_d128__b8__l8192__merge / g2_C06_s32_d128:vec4:within | 4.037 [3.348,4.813] | 2.848 [1.073,4.566] | 3.226 [1.026,4.942] | not_resolved_in_new_context |
| op__g2_C06_s32_d256__b64__l8192__merge / g2_C06_s32_d256:vec1:n_first | -7.448 [-7.915,-6.993] | -7.016 [-7.353,-6.595] | -6.771 [-6.982,-6.559] | neither_resolved |
| op__g2_C06_s32_d256__b64__l8192__merge / g2_C06_s32_d256:vec4:within | 1.627 [1.373,1.916] | 1.673 [1.474,1.908] | 1.898 [1.761,2.054] | neither_resolved |
| op__g2_C06_s32_d256__b8__l8192__merge / g2_C06_s32_d256:vec1:n_first | -7.727 [-9.955,-5.974] | -7.834 [-9.286,-6.742] | -7.910 [-9.286,-6.964] | neither_resolved |
| op__g2_C06_s32_d256__b8__l8192__merge / g2_C06_s32_d256:vec4:within | 3.849 [2.879,4.840] | 3.917 [2.615,5.263] | 3.850 [2.413,5.263] | neither_resolved |
| op__g4a_C02_n64_d128__b8__l2048__stage1 / g4a_C02_n64_d128:coordinated | -9.749 [-9.976,-9.539] | -9.296 [-9.638,-8.895] | -9.369 [-9.706,-8.985] | neither_resolved |
| op__g4a_C02_n64_d128__b8__l2048__stage1 / g4a_C02_n64_d128:pv_only | -2.418 [-2.541,-2.307] | -2.315 [-2.385,-2.245] | -2.383 [-2.451,-2.326] | neither_resolved |
| op__g4a_C02_n64_d128__b8__l2048__stage1 / g4a_C02_n64_d128:qk_only | -44.673 [-45.890,-42.480] | -44.036 [-45.075,-42.136] | -44.132 [-45.169,-42.240] | neither_resolved |
| op__g4a_C02_n64_d64__b8__l2048__stage1 / g4a_C02_n64_d64:coordinated | -146.857 [-150.564,-140.147] | -148.434 [-151.390,-143.124] | -148.196 [-151.227,-142.802] | neither_resolved |
| op__g4a_C02_n64_d64__b8__l2048__stage1 / g4a_C02_n64_d64:pv_only | -25.183 [-25.938,-24.043] | -25.909 [-26.395,-25.268] | -25.789 [-26.321,-25.097] | neither_resolved |
| op__g4a_C02_n64_d64__b8__l2048__stage1 / g4a_C02_n64_d64:qk_only | -152.428 [-156.393,-145.401] | -154.179 [-157.241,-148.757] | -153.936 [-156.997,-148.407] | neither_resolved |

## 解释边界

- This is derived postprocessing of frozen measurements, not a newly selected cost model or sampling protocol.
- A support transition means a difference in pointwise CI resolution between two execution contexts. It does not prove the old gain was caused by idle or that a new gain would hold under every real workload.
- Old N4/G8 CI-supported thresholds use their actual stored confidence intervals. Legacy/anchor published point slopes do not acquire invented old confidence intervals.
- All losing, unchanged and control-inconclusive outcomes remain in the source analyses. Aliases do not create additional independent observations.
- Each suite has three independently allocated formal GPU workers. Raw sample counts and logical comparison counts are not independent GPU counts.
