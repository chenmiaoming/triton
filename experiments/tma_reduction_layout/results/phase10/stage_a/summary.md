# Phase10 Stage A — Current-main residual audit

Clean upstream `20d9d98fead78c2f65e1b3ba5a7881d86a836dad` was built with `make`. All 54 compiler attempts and 8100 exact-binary event samples are retained. Each target has three independently dispatched worker processes; each worker compares all 18 frozen binaries.

The comparison uses the original `[1,8192,8192]` BF16→FP32 max geometry on host and device descriptor paths. The vector4 reference uses the historical compiler; its contrast with current main is a diagnostic opportunity, not current-main patch validation.

| Target | Case/path | Current main mean median µs | Research vector4 mean median µs | Diagnostic time decrease % | Process band % | Main/default CUBIN equal |
| --- | --- | --- | --- | --- | --- | --- |
| sm100 | M32_N128_w4:host | 35.3920 | 28.8107 | 18.669 | [14.883, 22.454] | False |
| sm100 | M32_N128_w4:device | 40.7253 | 37.0080 | 9.149 | [7.624, 10.674] | False |
| sm100 | M32_N64_w8:host | 84.2080 | 55.8720 | 33.668 | [31.470, 35.866] | False |
| sm100 | M32_N64_w8:device | 100.6560 | 87.2533 | 13.327 | [11.372, 15.281] | False |
| sm100 | M32_N128_w8:host | 45.0933 | 37.2160 | 17.524 | [13.708, 21.340] | False |
| sm100 | M32_N128_w8:device | 57.6587 | 50.9707 | 11.608 | [8.399, 14.817] | False |
| sm120 | M32_N128_w4:host | 101.5733 | 100.8640 | 0.702 | [-0.428, 1.832] | False |
| sm120 | M32_N128_w4:device | 103.0080 | 103.3600 | -0.345 | [-1.132, 0.442] | False |
| sm120 | M32_N64_w8:host | 106.9333 | 101.2160 | 5.345 | [3.840, 6.850] | False |
| sm120 | M32_N64_w8:device | 130.5227 | 109.1893 | 16.346 | [15.462, 17.230] | False |
| sm120 | M32_N128_w8:host | 101.6747 | 101.2373 | 0.429 | [0.049, 0.810] | False |
| sm120 | M32_N128_w8:device | 105.6907 | 104.0373 | 1.567 | [0.650, 2.484] | False |
| sm90 | M32_N128_w4:host | 56.0640 | 55.2907 | 1.384 | [0.269, 2.499] | False |
| sm90 | M32_N128_w4:device | 57.8240 | 57.4187 | 0.703 | [-0.399, 1.804] | False |
| sm90 | M32_N64_w8:host | 111.0133 | 68.0267 | 38.723 | [38.256, 39.190] | False |
| sm90 | M32_N64_w8:device | 142.7680 | 91.0240 | 36.244 | [35.903, 36.585] | False |
| sm90 | M32_N128_w8:host | 62.3200 | 54.5387 | 12.488 | [11.609, 13.366] | False |
| sm90 | M32_N128_w8:device | 76.8213 | 61.8293 | 19.517 | [18.178, 20.857] | False |

No historical culprit is inferred from this audit. B200/SM100 and RTX PRO6000/SM120 do not replace the original GB300/SM103 system. The next stage freezes an instruction-count compiler prototype and validates structure/correctness; held-out patch timing remains a separate stage. H100 current main uses ptxas 13.4.59 while its historical compiler uses 12.9.86; this cross-compiler contrast therefore also includes assembler changes. B200 and SM120 use the same assembler SHA on both compiler builds. The analysis retains the within-historical-compiler default/vector4 comparison separately. All prior evidence is unchanged.
