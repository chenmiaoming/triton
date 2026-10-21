# 全部硬件计数摘要

WF/BC 是全 grid `.sum/CTA数`；SHFL/BAR 是独立单 CTA software 的 warp 指令实例数。
q condition 是请求档，q_cap 是该二进制 natural 最大容量；具体 padding 容量见 artifact_gate.json。

| 程序 | q condition | natural q_cap | load WF/CTA | load BC/CTA | SHFL/CTA | BAR/CTA | barrier % | short SB % | long SB % | MIO % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M32_N128_w4:control:ONE_TRANSFER | natural | 16 | 4.05 | 0.05 | 0 | 8 | 4.31 | 0.88 | 86.36 | 3.88 |
| M32_N128_w4:copy_control:ONE_COPY | natural | 16 | 73.86 | 9.86 | 0 | 8 | 12.42 | 5.57 | 21.69 | 39.52 |
| M32_N128_w4:vec1:n_first:sw128:broadcast:ONE_FULL | natural | 16 | 137.50 | 9.50 | 0 | 8 | 3.12 | 1.86 | 38.4 | 48.7 |
| M32_N128_w4:vec1:n_first:sw128:centered:ONE_FULL | natural | 10 | 137.15 | 9.15 | 0 | 8 | 4.13 | 3.71 | 49.23 | 31.34 |
| M32_N128_w4:vec1:n_first:sw128:native:ONE_FULL | natural | 16 | 128.98 | 0.98 | 0 | 8 | 3.91 | 0.87 | 87.84 | 3.44 |
| M32_N128_w4:vec2:n_first:sw128:broadcast:ONE_FULL | natural | 16 | 90.70 | 10.70 | 8 | 24 | 5.03 | 14.19 | 47.36 | 22.97 |
| M32_N128_w4:vec2:n_first:sw128:centered:ONE_FULL | natural | 10 | 90.34 | 10.34 | 8 | 24 | 4.21 | 14.97 | 44.55 | 21.6 |
| M32_N128_w4:vec2:n_first:sw128:native:ONE_FULL | natural | 16 | 76.70 | 0.70 | 8 | 24 | 4.82 | 1.84 | 83.04 | 3.15 |
| M32_N128_w4:vec2:within:sw128:broadcast:ONE_FULL | natural | 16 | 138.25 | 74.25 | 8 | 8 | 3.37 | 7.66 | 52.02 | 26.45 |
| M32_N128_w4:vec2:within:sw128:centered:ONE_FULL | natural | 9 | 138.18 | 74.18 | 8 | 8 | 2.42 | 14.82 | 48.89 | 16.0 |
| M32_N128_w4:vec2:within:sw128:native:ONE_FULL | natural | 16 | 128.78 | 64.78 | 8 | 8 | 3.57 | 2.18 | 84.27 | 3.24 |
| M32_N128_w4:vec4:n_first:sw128:broadcast:ONE_FULL | natural | 12 | 106.73 | 10.73 | 32 | 24 | 6.03 | 17.34 | 41.78 | 22.77 |
| M32_N128_w4:vec4:n_first:sw128:centered:ONE_FULL | natural | 9 | 105.46 | 9.46 | 32 | 24 | 2.1 | 19.47 | 48.48 | 13.38 |
| M32_N128_w4:vec4:n_first:sw128:native:ONE_FULL | natural | 16 | 84.87 | 0.87 | 32 | 24 | 4.95 | 2.0 | 84.75 | 3.21 |
| M32_N128_w4:vec4:within:sw128:broadcast:ONE_FULL | natural | 16 | 137.77 | 73.77 | 32 | 8 | 4.04 | 8.33 | 43.52 | 35.05 |
| M32_N128_w4:vec4:within:sw128:centered:ONE_FULL | natural | 12 | 137.99 | 73.99 | 32 | 8 | 3.82 | 24.7 | 39.22 | 19.7 |
| M32_N128_w4:vec4:within:sw128:native:ONE_FULL | natural | 16 | 128.80 | 64.80 | 32 | 8 | 3.63 | 2.06 | 85.55 | 3.19 |
| M32_N128_w4:vec8:n_first:sw128:broadcast:ONE_FULL | natural | 12 | 144.54 | 16.54 | 96 | 40 | 8.59 | 17.17 | 31.25 | 33.66 |
| M32_N128_w4:vec8:n_first:sw128:centered:ONE_FULL | natural | 8 | 143.78 | 15.78 | 96 | 40 | 5.17 | 21.16 | 22.46 | 35.18 |
| M32_N128_w4:vec8:n_first:sw128:native:ONE_FULL | natural | 16 | 115.77 | 1.77 | 96 | 40 | 5.38 | 3.61 | 83.63 | 3.14 |
| M32_N128_w4:vec8:within:sw128:broadcast:ONE_FULL | natural | 16 | 138.24 | 74.24 | 96 | 8 | 3.41 | 6.78 | 33.44 | 48.53 |
| M32_N128_w4:vec8:within:sw128:centered:ONE_FULL | natural | 12 | 137.44 | 73.44 | 96 | 8 | 1.97 | 44.65 | 25.38 | 18.75 |
| M32_N128_w4:vec8:within:sw128:native:ONE_FULL | natural | 16 | 128.90 | 64.90 | 96 | 8 | 3.13 | 2.68 | 86.63 | 3.27 |
| M32_N64_w8:control:ONE_TRANSFER | natural | 8 | 2.08 | 0.08 | 0 | 16 | 3.65 | 2.5 | 80.06 | 4.64 |
| M32_N64_w8:copy_control:ONE_COPY | natural | 8 | 37.43 | 5.43 | 0 | 16 | 1.22 | 26.6 | 42.27 | 4.15 |
| M32_N64_w8:vec1:n_first:sw128:broadcast:ONE_FULL | natural | 8 | 84.38 | 4.38 | 16 | 48 | 3.62 | 3.08 | 76.62 | 3.55 |
| M32_N64_w8:vec1:n_first:sw128:centered:ONE_FULL | natural | 8 | 84.36 | 4.36 | 16 | 48 | 4.42 | 3.1 | 77.49 | 3.96 |
| M32_N64_w8:vec1:n_first:sw128:native:ONE_FULL | natural | 8 | 75.21 | 1.21 | 16 | 48 | 6.48 | 5.76 | 69.46 | 3.12 |
| M32_N64_w8:vec1:within:sw128:broadcast:ONE_FULL | natural | 8 | 67.94 | 3.94 | 16 | 16 | 2.31 | 2.48 | 78.09 | 3.67 |
| M32_N64_w8:vec1:within:sw128:centered:ONE_FULL | natural | 8 | 67.30 | 3.30 | 16 | 16 | 2.44 | 3.01 | 78.02 | 4.46 |
| M32_N64_w8:vec1:within:sw128:native:ONE_FULL | natural | 8 | 65.00 | 1.00 | 16 | 16 | 2.74 | 4.51 | 74.03 | 3.34 |
| M32_N64_w8:vec2:n_first:sw128:broadcast:ONE_FULL | natural | 8 | 69.95 | 5.95 | 48 | 48 | 3.92 | 4.41 | 77.29 | 3.05 |
| M32_N64_w8:vec2:n_first:sw128:centered:ONE_FULL | natural | 8 | 69.82 | 5.82 | 48 | 48 | 3.93 | 4.32 | 76.77 | 2.97 |
| M32_N64_w8:vec2:n_first:sw128:native:ONE_FULL | natural | 8 | 51.12 | 1.12 | 48 | 48 | 5.83 | 8.23 | 69.99 | 2.88 |
| M32_N64_w8:vec2:within:sw128:broadcast:ONE_FULL | natural | 8 | 35.92 | 3.92 | 48 | 16 | 2.05 | 4.09 | 78.34 | 2.76 |
| M32_N64_w8:vec2:within:sw128:centered:ONE_FULL | natural | 8 | 35.99 | 3.99 | 48 | 16 | 2.05 | 4.35 | 77.81 | 2.93 |
| M32_N64_w8:vec2:within:sw128:native:ONE_FULL | natural | 8 | 32.87 | 0.87 | 48 | 16 | 3.1 | 5.98 | 76.0 | 3.38 |
| M32_N64_w8:vec4:n_first:sw128:broadcast:ONE_FULL | natural | 8 | 103.58 | 7.58 | 128 | 48 | 4.18 | 5.64 | 75.37 | 2.99 |
| M32_N64_w8:vec4:n_first:sw128:centered:ONE_FULL | natural | 8 | 103.62 | 7.62 | 128 | 48 | 4.18 | 5.99 | 76.27 | 2.99 |
| M32_N64_w8:vec4:n_first:sw128:native:ONE_FULL | natural | 8 | 67.31 | 1.31 | 128 | 48 | 7.66 | 12.65 | 61.51 | 4.09 |
| M32_N64_w8:vec4:within:sw128:broadcast:ONE_FULL | natural | 8 | 38.46 | 6.46 | 128 | 16 | 1.89 | 5.03 | 75.37 | 3.11 |
| M32_N64_w8:vec4:within:sw128:centered:ONE_FULL | natural | 8 | 38.63 | 6.63 | 128 | 16 | 1.97 | 5.25 | 76.27 | 3.23 |
| M32_N64_w8:vec4:within:sw128:native:ONE_FULL | natural | 8 | 32.83 | 0.83 | 128 | 16 | 3.49 | 7.64 | 70.9 | 3.42 |
| M32_N64_w8:vec8:n_first:sw128:broadcast:ONE_FULL | natural | 8 | 176.64 | 16.64 | 320 | 80 | 9.18 | 10.4 | 33.8 | 29.98 |
| M32_N64_w8:vec8:n_first:sw128:centered:ONE_FULL | natural | 8 | 177.13 | 17.13 | 320 | 80 | 10.58 | 11.91 | 33.72 | 28.03 |
| M32_N64_w8:vec8:n_first:sw128:native:ONE_FULL | natural | 8 | 131.20 | 2.20 | 320 | 80 | 17.75 | 11.3 | 27.66 | 25.05 |
| M32_N64_w8:vec8:within:sw128:broadcast:ONE_FULL | natural | 8 | 40.68 | 8.68 | 320 | 16 | 4.3 | 6.1 | 17.5 | 56.34 |
| M32_N64_w8:vec8:within:sw128:centered:ONE_FULL | natural | 8 | 40.07 | 8.07 | 320 | 16 | 5.25 | 9.09 | 17.58 | 52.55 |
| M32_N64_w8:vec8:within:sw128:native:ONE_FULL | natural | 8 | 32.68 | 0.68 | 320 | 16 | 7.79 | 6.19 | 33.59 | 28.42 |
