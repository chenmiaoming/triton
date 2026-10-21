# M4B — 真实二维消费者与布局选择

主要结果与解释见 [FINDINGS.md](FINDINGS.md)，下文保留完整设计、表格与限制。

本阶段为开发域机制实验，旧选择器与旧报告保持不变，没有新选择器拟合或盲测记分。
固定 `20d9d98fead78c2f65e1b3ba5a7881d86a836dad`；复用既有 compiler/tools image、ccache 与 kernel cache，native rebuild=0。
49 个逻辑程序，新增 31、复用 18，CPU 拒绝 0。
三种输入、完整 2 MiB 输入所对应的全部输出与尾部 canary 共 147 项通过。
三名独立 H100 worker，23544 个正式 CUDA Graph 事件样本；NCU 原始报告 10 份，全部本地 CPU 重导入核对。

主指标是 128/256/384 MiB 连续 BF16 输入对应的 loop-free FULL 服务斜率，单位 ns/tile/SM；128 MiB 原始 µs 单列。
3 visits，每次 2 warmup+4 samples，配对随机顺序，不因输赢追加重复。每个 worker 内先以中位数作三点斜率，再比较同 worker 条件。
下文区间是三个配对 worker 的均值 ± t(2,0.975)×SEM，属于探索性逐项区间，未校正多重比较。
q 是占用 API 给出的 padding 驻留容量，不能当作实际同时活跃的 CTA 数。

## 1. 本轮消费者

native：每列写一个 FP32 max；broadcast：保留原诊断 max(column)+logical_row；centered：真实 `wide=x.to(float32); wide-max(wide,rows)`，写出 FP32 二维 tile。
centered 是实际规约后广播减法子图，不冒充完整 Column-Softmax/LayerNorm。15 个合法唯一布局均测试这三种消费者；另有两个旧 1D T_control 与两个匹配输入/二维写出的 copy 控制。
二维消费者每 CTA 写 BM×BN 个 FP32，输出字节是 BF16 输入的两倍；输出缓冲按真实 stride 分配。
不同消费者输出量不同，各自在同语义内记分；未把三种消费者时间混为一类加速。

## 2. 自然驻留下的实测最快候选

正 gain 表示相对该消费者自身的默认 vec8:N-first / fixed_vec1:N-first 耗时下降。
“最快”是本轮事后观察，不是冻结选择器，也不把简单基线未输的情况隐藏。

| geometry | consumer | 实测最快布局 | ns/tile/SM | gain vs default | gain vs fixed_vec1 | vs vec1 paired t95 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| (32, 64, 8) | broadcast | M32_N64_w8:vec1:within | 544.038 | +22.91% | +0.72% | [0.65, 0.79] |
| (32, 64, 8) | centered | M32_N64_w8:vec2:within | 544.489 | +22.68% | +0.62% | [0.05, 1.19] |
| (32, 64, 8) | native | M32_N64_w8:vec2:within | 204.080 | +42.98% | +2.00% | [1.11, 2.88] |
| (32, 128, 4) | broadcast | M32_N128_w4:vec4:n_first | 1106.596 | +7.35% | +1.98% | [1.5, 2.47] |
| (32, 128, 4) | centered | M32_N128_w4:vec4:n_first | 1101.354 | +24.74% | +1.38% | [1.1, 1.65] |
| (32, 128, 4) | native | M32_N128_w4:vec4:n_first | 372.775 | +0.25% | +0.22% | [-2.01, 2.45] |

## 3. 全部候选、实际寄存器、驻留与 max lowering

PTX opcode 站点不是动态执行次数；动态计数另见下一节。宽化输入有两个消费者后，是否仍能打包由实际编译产物决定；未为保留 packed max 改写算子语义。

| geometry | consumer | 布局 | q_cap | regs | service ns | 128 MiB µs | gain vs default | gain vs vec1 | PTX max sites |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (32, 64, 8) | broadcast | M32_N64_w8:vec1:n_first | 8 | 32 | 547.991 | 141.296 | +22.35% | +0.00% | {'max.bf16': 7, 'max.f32': 2} |
| (32, 64, 8) | broadcast | M32_N64_w8:vec1:within | 8 | 32 | 544.038 | 139.701 | +22.91% | +0.72% | {'max.bf16': 7, 'max.f32': 2} |
| (32, 64, 8) | broadcast | M32_N64_w8:vec2:n_first | 8 | 22 | 547.143 | 141.067 | +22.47% | +0.15% | {'max.bf16x2': 3, 'max.f32': 6} |
| (32, 64, 8) | broadcast | M32_N64_w8:vec2:within | 8 | 21 | 545.821 | 139.637 | +22.66% | +0.40% | {'max.bf16': 6, 'max.f32': 6} |
| (32, 64, 8) | broadcast | M32_N64_w8:vec4:n_first | 8 | 21 | 547.110 | 141.120 | +22.48% | +0.16% | {'max.bf16x2': 2, 'max.f32': 16} |
| (32, 64, 8) | broadcast | M32_N64_w8:vec4:within | 8 | 18 | 546.058 | 139.691 | +22.63% | +0.35% | {'max.bf16': 4, 'max.f32': 16} |
| (32, 64, 8) | broadcast | M32_N64_w8:vec8:n_first | 8 | 29 | 705.740 | 182.827 | +0.00% | -28.79% | {'max.f32': 40} |
| (32, 64, 8) | broadcast | M32_N64_w8:vec8:within | 8 | 23 | 1168.632 | 296.640 | -65.59% | -113.26% | {'max.f32': 40} |
| (32, 64, 8) | centered | M32_N64_w8:vec1:n_first | 8 | 26 | 547.873 | 141.397 | +22.20% | +0.00% | {'max.f32': 9} |
| (32, 64, 8) | centered | M32_N64_w8:vec1:within | 8 | 22 | 545.155 | 139.840 | +22.58% | +0.50% | {'max.f32': 9} |
| (32, 64, 8) | centered | M32_N64_w8:vec2:n_first | 8 | 24 | 548.260 | 140.800 | +22.14% | -0.07% | {'max.f32': 12} |
| (32, 64, 8) | centered | M32_N64_w8:vec2:within | 8 | 24 | 544.489 | 140.064 | +22.68% | +0.62% | {'max.f32': 12} |
| (32, 64, 8) | centered | M32_N64_w8:vec4:n_first | 8 | 29 | 547.078 | 141.259 | +22.31% | +0.15% | {'max.f32': 20} |
| (32, 64, 8) | centered | M32_N64_w8:vec4:within | 8 | 26 | 546.047 | 139.867 | +22.46% | +0.33% | {'max.f32': 20} |
| (32, 64, 8) | centered | M32_N64_w8:vec8:n_first | 8 | 32 | 704.193 | 182.517 | +0.00% | -28.53% | {'max.f32': 40} |
| (32, 64, 8) | centered | M32_N64_w8:vec8:within | 8 | 32 | 1168.814 | 296.347 | -65.98% | -113.34% | {'max.f32': 40} |
| (32, 64, 8) | native | M32_N64_w8:vec1:n_first | 8 | 16 | 208.237 | 57.333 | +41.82% | +0.00% | {'max.bf16': 7, 'max.f32': 2} |
| (32, 64, 8) | native | M32_N64_w8:vec1:within | 8 | 15 | 204.617 | 56.032 | +42.83% | +1.74% | {'max.bf16': 7, 'max.f32': 2} |
| (32, 64, 8) | native | M32_N64_w8:vec2:n_first | 8 | 19 | 207.904 | 57.413 | +41.91% | +0.16% | {'max.bf16x2': 3, 'max.f32': 6} |
| (32, 64, 8) | native | M32_N64_w8:vec2:within | 8 | 14 | 204.080 | 56.000 | +42.98% | +2.00% | {'max.bf16': 6, 'max.f32': 6} |
| (32, 64, 8) | native | M32_N64_w8:vec4:n_first | 8 | 21 | 221.182 | 60.608 | +38.20% | -6.22% | {'max.bf16x2': 2, 'max.f32': 16} |
| (32, 64, 8) | native | M32_N64_w8:vec4:within | 8 | 18 | 208.603 | 57.317 | +41.71% | -0.18% | {'max.bf16': 4, 'max.f32': 16} |
| (32, 64, 8) | native | M32_N64_w8:vec8:n_first | 8 | 27 | 357.897 | 96.549 | +0.00% | -71.87% | {'max.f32': 40} |
| (32, 64, 8) | native | M32_N64_w8:vec8:within | 8 | 26 | 310.428 | 83.803 | +13.26% | -49.07% | {'max.f32': 40} |
| (32, 128, 4) | broadcast | M32_N128_w4:vec1:n_first | 16 | 32 | 1128.961 | 144.651 | +5.48% | +0.00% | {'max.bf16': 31} |
| (32, 128, 4) | broadcast | M32_N128_w4:vec2:n_first | 16 | 32 | 1113.771 | 142.411 | +6.75% | +1.35% | {'max.bf16x2': 15, 'max.f32': 2} |
| (32, 128, 4) | broadcast | M32_N128_w4:vec2:within | 16 | 24 | 1130.465 | 145.280 | +5.35% | -0.13% | {'max.bf16': 30, 'max.f32': 2} |
| (32, 128, 4) | broadcast | M32_N128_w4:vec4:n_first | 12 | 36 | 1106.596 | 142.128 | +7.35% | +1.98% | {'max.bf16x2': 14, 'max.f32': 8} |
| (32, 128, 4) | broadcast | M32_N128_w4:vec4:within | 16 | 31 | 1132.979 | 145.019 | +5.14% | -0.36% | {'max.bf16': 28, 'max.f32': 8} |
| (32, 128, 4) | broadcast | M32_N128_w4:vec8:n_first | 12 | 35 | 1194.381 | 155.936 | +0.00% | -5.79% | {'max.bf16x2': 12, 'max.f32': 24} |
| (32, 128, 4) | broadcast | M32_N128_w4:vec8:within | 16 | 32 | 1185.916 | 154.427 | +0.71% | -5.04% | {'max.bf16': 24, 'max.f32': 24} |
| (32, 128, 4) | centered | M32_N128_w4:vec1:n_first | 10 | 48 | 1116.715 | 143.413 | +23.69% | +0.00% | {'max.f32': 31} |
| (32, 128, 4) | centered | M32_N128_w4:vec2:n_first | 10 | 47 | 1103.846 | 141.547 | +24.57% | +1.15% | {'max.f32': 32} |
| (32, 128, 4) | centered | M32_N128_w4:vec2:within | 9 | 53 | 1114.910 | 142.997 | +23.81% | +0.16% | {'max.f32': 32} |
| (32, 128, 4) | centered | M32_N128_w4:vec4:n_first | 9 | 54 | 1101.354 | 141.013 | +24.74% | +1.38% | {'max.f32': 36} |
| (32, 128, 4) | centered | M32_N128_w4:vec4:within | 12 | 40 | 1125.287 | 143.824 | +23.11% | -0.77% | {'max.f32': 36} |
| (32, 128, 4) | centered | M32_N128_w4:vec8:n_first | 8 | 60 | 1463.516 | 188.923 | +0.00% | -31.05% | {'max.f32': 48} |
| (32, 128, 4) | centered | M32_N128_w4:vec8:within | 12 | 38 | 1773.428 | 224.245 | -21.18% | -58.81% | {'max.f32': 48} |
| (32, 128, 4) | native | M32_N128_w4:vec1:n_first | 16 | 32 | 373.592 | 51.968 | +0.03% | +0.00% | {'max.bf16': 31} |
| (32, 128, 4) | native | M32_N128_w4:vec2:n_first | 16 | 22 | 373.549 | 52.021 | +0.05% | +0.01% | {'max.bf16x2': 15, 'max.f32': 2} |
| (32, 128, 4) | native | M32_N128_w4:vec2:within | 16 | 22 | 373.592 | 52.133 | +0.03% | -0.00% | {'max.bf16': 30, 'max.f32': 2} |
| (32, 128, 4) | native | M32_N128_w4:vec4:n_first | 16 | 24 | 372.775 | 52.453 | +0.25% | +0.22% | {'max.bf16x2': 14, 'max.f32': 8} |
| (32, 128, 4) | native | M32_N128_w4:vec4:within | 16 | 24 | 374.301 | 52.176 | -0.16% | -0.19% | {'max.bf16': 28, 'max.f32': 8} |
| (32, 128, 4) | native | M32_N128_w4:vec8:n_first | 16 | 29 | 373.721 | 53.045 | +0.00% | -0.03% | {'max.bf16x2': 12, 'max.f32': 24} |
| (32, 128, 4) | native | M32_N128_w4:vec8:within | 16 | 25 | 373.184 | 52.736 | +0.14% | +0.11% | {'max.bf16': 24, 'max.f32': 24} |

## 4. 静态输出计划的前向核对

在本轮 45 个 candidate/consumer 条件中，store sector 请求预测匹配 45/45。
全部命中/未命中记录保存在 diagnostics.json 的 forward_store_predictions，不事后覆盖预测。
返回读取需求是结构条件，实际 LDS 可能同时受 FP32 多消费者、MR lowering 和资源影响；不能只靠 whole LDS 差值分配独立广播延迟。
完整 q=1/2/4/8/natural 曲线、零收益和退化候选保存在 analysis.json，未只保留自然驻留下的赢家。

## 5. 原始证据、复核与停点

[全部计数摘要](COUNTERS.md)；完整 FULL 三点斜率与各 worker 原始时间见 analysis.json，全部原始事件见 timing/invocation_*/raw.json。
compile_gate.json/compiled_payload.zip 固定实际 CUBIN；artifact_gate.json 绑定真实寄存器、local bytes、q 与正确性。
profiles/*/attempt_*/export 保存 ncu-rep、source/raw CSV、worker UUID 与实际加载 SHA；offline_import_validation.json 保存独立本地重导入核对。
诊断只使用本阶段冻结数据，不把历史 26.6 µs 或人工总分引入模型。

M4a 首次 CPU preflight 在编译前发现上传快照遗漏 c16/kernels.py；失败 dispatch/retrieval 记录保留，CPU-only packaging revision 增补依赖，36 个程序与指标清单未改，未产生 GPU 样本。
本阶段通过 validator 后生成 freeze.json、commit/push。只继续用户已授权的 M4a/M4b，全部完成后停下；没有新模型拟合、C++ 修改、PR 或跨架构实验。
