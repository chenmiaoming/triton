# Phase10 Stage A — Current-main residual audit

Clean upstream `20d9d98fead78c2f65e1b3ba5a7881d86a836dad` was built with `make`. All54 compiler attempts and 8100 exact-binary event samples are retained. Each target has three independently dispatched worker processes; each worker compares all18 frozen binaries.

The comparison uses the original `[1,8192,8192]` BF16→FP32 max geometry on host and device descriptor paths. The vector4 reference uses the historical compiler; its contrast with current main is a diagnostic opportunity, not current-main patch validation.

| Target | Case/path | Current main mean median µs | Research vector4 mean median µs | Diagnostic time decrease % | Process band % | Main/default CUBIN equal |
| --- | --- | --- | --- | --- | --- | --- |
| sm100 | M32_N128_w4:host | 35.9520 | 36.1813 | -0.657 | [-2.171, 0.857] | False |
| sm100 | M32_N128_w4:device | 40.6133 | 40.6880 | -0.177 | [-0.920, 0.566] | False |
| sm100 | M32_N64_w8:host | 84.2667 | 84.2187 | 0.050 | [-0.634, 0.734] | False |
| sm100 | M32_N64_w8:device | 100.1653 | 102.1227 | -1.955 | [-2.349, -1.562] | False |
| sm100 | M32_N128_w8:host | 45.1413 | 45.3707 | -0.468 | [-2.603, 1.667] | False |
| sm100 | M32_N128_w8:device | 57.6000 | 57.1467 | 0.798 | [-1.997, 3.592] | False |
| sm120 | M32_N128_w4:host | 100.3627 | 100.2027 | 0.158 | [-0.645, 0.961] | False |
| sm120 | M32_N128_w4:device | 102.2453 | 102.6613 | -0.407 | [-1.051, 0.237] | False |
| sm120 | M32_N64_w8:host | 105.0027 | 104.9173 | 0.082 | [-0.472, 0.637] | False |
| sm120 | M32_N64_w8:device | 128.5653 | 128.3360 | 0.181 | [-0.631, 0.993] | False |
| sm120 | M32_N128_w8:host | 100.9920 | 100.6293 | 0.358 | [-0.208, 0.924] | False |
| sm120 | M32_N128_w8:device | 104.8427 | 104.4107 | 0.411 | [-1.031, 1.853] | False |
| sm90 | M32_N128_w4:host | 56.1813 | 56.2880 | -0.191 | [-0.583, 0.201] | False |
| sm90 | M32_N128_w4:device | 57.8933 | 57.9307 | -0.066 | [-1.276, 1.144] | False |
| sm90 | M32_N64_w8:host | 110.7947 | 110.3413 | 0.409 | [0.285, 0.534] | False |
| sm90 | M32_N64_w8:device | 142.7520 | 140.3147 | 1.708 | [1.339, 2.076] | False |
| sm90 | M32_N128_w8:host | 62.2720 | 62.0320 | 0.382 | [-0.498, 1.262] | False |
| sm90 | M32_N128_w8:device | 76.8480 | 76.3093 | 0.701 | [0.461, 0.942] | False |

No historical culprit is inferred from this audit. B200/SM100 and RTX PRO6000/SM120 do not replace the original GB300/SM103 system. The next stage freezes an instruction-count compiler prototype and validates structure/correctness; held-out patch timing remains a separate stage. All prior evidence is unchanged.
