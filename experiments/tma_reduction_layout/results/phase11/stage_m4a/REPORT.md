# M4A — 加载服务与同步依赖对照

主要结果与解释见 [FINDINGS.md](FINDINGS.md)，下文保留完整设计、表格与限制。

本阶段为开发域机制实验，旧选择器与旧报告保持不变，没有新选择器拟合或盲测记分。
固定 `20d9d98fead78c2f65e1b3ba5a7881d86a836dad`；复用既有 compiler/tools image、ccache 与 kernel cache，native rebuild=0。
36 个逻辑程序，新增 27、复用 9，CPU 拒绝 0。
三种输入、完整 2 MiB 输入所对应的全部输出与尾部 canary 共 108 项通过。
三名独立 H100 worker，7776 个正式 CUDA Graph 事件样本；NCU 原始报告 10 份，全部本地 CPU 重导入核对。

主指标是 128/256/384 MiB 连续 BF16 输入对应的 loop-free FULL 服务斜率，单位 ns/tile/SM；128 MiB 原始 µs 单列。
3 visits，每次 2 warmup+4 samples，配对随机顺序，不因输赢追加重复。每个 worker 内先以中位数作三点斜率，再比较同 worker 条件。
下文区间是三个配对 worker 的均值 ± t(2,0.975)×SEM，属于探索性逐项区间，未校正多重比较。
q 是占用 API 给出的 padding 驻留容量，不能当作实际同时活跃的 CTA 数。

## 1. H4 的检验口径

检验“加载冲突/吞吐与跨 warp 同步依赖有不同驻留响应”，不是提前认定 H4 已成立。
寄存器布局、native 输出与 max 源程序固定，只改变 NVMMASharedLayout 的 128/64/32/0B swizzle，另配同 swizzle 的 T_control。
36 个程序都通过编译和 GPU 正确性，原 q=2/4/8 曲线直接引用 M2/M3；新配对计时只做 q=1/natural，NCU 补 q=1/2/4/natural。
静态 64 个 ML 冲突的消除预测，见 CPU 准备的 operation_plans_v2.json；真实指令分组、MR 和资源变化在本阶段核对。

## 2. within−N-first 的完整程序差分

负数表示 within 更快。只在双方实际容量相同的 q 档配对；全部 natural 时间与各自容量见 analysis.json。

| geometry | shared swizzle B | q_cap | within−NF ns | paired t95 ns |
| --- | --- | --- | --- | --- |
| [32, 64, 8] | 128 | 1 | -62.176 | [-64.908, -59.444] |
| [32, 64, 8] | 128 | 8 | -4.458 | [-6.155, -2.761] |
| [32, 64, 8] | 64 | 1 | -73.520 | [-75.093, -71.946] |
| [32, 64, 8] | 64 | 8 | -7.240 | [-7.574, -6.907] |
| [32, 64, 8] | 32 | 1 | -101.954 | [-105.158, -98.750] |
| [32, 64, 8] | 32 | 8 | -30.132 | [-31.267, -28.997] |
| [32, 64, 8] | 0 | 1 | +43.098 | [+40.912, +45.283] |
| [32, 64, 8] | 0 | 8 | +61.112 | [+58.925, +63.299] |
| [32, 128, 4] | 128 | 1 | -25.738 | [-29.809, -21.667] |
| [32, 128, 4] | 128 | 16 | +0.537 | [-1.796, +2.870] |
| [32, 128, 4] | 64 | 1 | -92.598 | [-96.566, -88.629] |
| [32, 128, 4] | 64 | 16 | +2.943 | [+1.544, +4.342] |
| [32, 128, 4] | 32 | 1 | -127.230 | [-131.530, -122.931] |
| [32, 128, 4] | 32 | 16 | -0.236 | [-2.238, +1.766] |
| [32, 128, 4] | 0 | 1 | -16.758 | [-17.912, -15.603] |
| [32, 128, 4] | 0 | 16 | +0.172 | [-4.958, +5.302] |
| [16, 256, 8] | 128 | 1 | -32.484 | [-39.026, -25.943] |
| [16, 256, 8] | 128 | 8 | -0.902 | [-6.682, +4.878] |
| [16, 256, 8] | 64 | 1 | -101.170 | [-106.467, -95.873] |
| [16, 256, 8] | 64 | 8 | -1.289 | [-9.261, +6.683] |
| [16, 256, 8] | 32 | 1 | -137.994 | [-141.193, -134.796] |
| [16, 256, 8] | 32 | 8 | +0.924 | [-2.604, +4.451] |
| [16, 256, 8] | 0 | 1 | -35.514 | [-41.334, -29.694] |
| [16, 256, 8] | 0 | 8 | +1.354 | [-5.880, +8.587] |

## 3. 改变 swizzle 的配对效应与隔离程度

下表是同一寄存器布局、同一消费者相对 128B swizzle 的服务时间差。
MR 相同列只核对源行归属的 opcode 类及动态次数，不代表寄存器操作数、排程或所有依赖均相同。
寄存器不同时也保留数据，在受控 q=1 下消除容量差异，但不将其伪装成机器码完全相同的单变量延迟。

| 程序 | q_cap | 相对 sw128 ns | paired t95 ns | MR opcode/count 相同 | regs 相同 |
| --- | --- | --- | --- | --- | --- |
| M32_N64_w8:vec2:n_first:sw64:native:ONE_FULL | 1 | +16.446 | [+15.047, +17.845] | False | True |
| M32_N64_w8:vec2:n_first:sw64:native:ONE_FULL | 8 | +3.910 | [+2.468, +5.352] | False | True |
| M32_N64_w8:vec2:n_first:sw32:native:ONE_FULL | 1 | +51.283 | [+48.363, +54.203] | False | False |
| M32_N64_w8:vec2:n_first:sw32:native:ONE_FULL | 8 | +28.134 | [+26.883, +29.384] | False | False |
| M32_N64_w8:vec2:n_first:sw0:native:ONE_FULL | 1 | +5.726 | [+2.234, +9.218] | True | False |
| M32_N64_w8:vec2:n_first:sw0:native:ONE_FULL | 8 | +0.580 | [-0.652, +1.813] | True | False |
| M32_N64_w8:vec2:within:sw64:native:ONE_FULL | 1 | +5.103 | [+4.224, +5.981] | True | True |
| M32_N64_w8:vec2:within:sw64:native:ONE_FULL | 8 | +1.128 | [-0.471, +2.727] | True | True |
| M32_N64_w8:vec2:within:sw32:native:ONE_FULL | 1 | +11.505 | [+9.612, +13.398] | True | True |
| M32_N64_w8:vec2:within:sw32:native:ONE_FULL | 8 | +2.460 | [+0.761, +4.159] | True | True |
| M32_N64_w8:vec2:within:sw0:native:ONE_FULL | 1 | +110.999 | [+109.575, +112.423] | True | False |
| M32_N64_w8:vec2:within:sw0:native:ONE_FULL | 8 | +66.150 | [+64.090, +68.211] | True | False |
| M32_N128_w4:vec2:n_first:sw64:native:ONE_FULL | 1 | +30.615 | [+27.465, +33.765] | False | False |
| M32_N128_w4:vec2:n_first:sw64:native:ONE_FULL | 16 | +0.881 | [-0.481, +2.243] | False | False |
| M32_N128_w4:vec2:n_first:sw32:native:ONE_FULL | 1 | +145.363 | [+144.183, +146.543] | False | False |
| M32_N128_w4:vec2:n_first:sw32:native:ONE_FULL | 16 | +27.930 | [+26.173, +29.686] | False | False |
| M32_N128_w4:vec2:n_first:sw0:native:ONE_FULL | 1 | -25.051 | [-30.619, -19.482] | False | True |
| M32_N128_w4:vec2:n_first:sw0:native:ONE_FULL | 16 | +0.172 | [-0.585, +0.929] | False | True |
| M32_N128_w4:vec2:within:sw64:native:ONE_FULL | 1 | -36.244 | [-38.621, -33.867] | True | True |
| M32_N128_w4:vec2:within:sw64:native:ONE_FULL | 16 | +3.287 | [+0.791, +5.783] | True | True |
| M32_N128_w4:vec2:within:sw32:native:ONE_FULL | 1 | +43.871 | [+42.100, +45.642] | True | True |
| M32_N128_w4:vec2:within:sw32:native:ONE_FULL | 16 | +27.156 | [+25.794, +28.518] | True | True |
| M32_N128_w4:vec2:within:sw0:native:ONE_FULL | 1 | -16.070 | [-18.799, -13.342] | True | True |
| M32_N128_w4:vec2:within:sw0:native:ONE_FULL | 16 | -0.193 | [-2.985, +2.599] | True | True |
| M16_N256_w8:vec2:n_first:sw64:native:ONE_FULL | 1 | +25.975 | [+21.933, +30.016] | False | False |
| M16_N256_w8:vec2:n_first:sw64:native:ONE_FULL | 8 | +1.182 | [-3.837, +6.200] | False | False |
| M16_N256_w8:vec2:n_first:sw32:native:ONE_FULL | 1 | +150.541 | [+142.128, +158.954] | False | False |
| M16_N256_w8:vec2:n_first:sw32:native:ONE_FULL | 8 | +40.111 | [+33.884, +46.338] | False | False |
| M16_N256_w8:vec2:n_first:sw0:native:ONE_FULL | 1 | -25.824 | [-29.682, -21.967] | False | False |
| M16_N256_w8:vec2:n_first:sw0:native:ONE_FULL | 8 | -0.387 | [-6.792, +6.018] | False | False |
| M16_N256_w8:vec2:within:sw64:native:ONE_FULL | 1 | -42.711 | [-45.314, -40.108] | True | False |
| M16_N256_w8:vec2:within:sw64:native:ONE_FULL | 8 | +0.795 | [-0.247, +1.837] | True | False |
| M16_N256_w8:vec2:within:sw32:native:ONE_FULL | 1 | +45.031 | [+41.174, +48.889] | True | True |
| M16_N256_w8:vec2:within:sw32:native:ONE_FULL | 8 | +41.938 | [+37.578, +46.297] | True | True |
| M16_N256_w8:vec2:within:sw0:native:ONE_FULL | 1 | -28.854 | [-29.098, -28.609] | True | True |
| M16_N256_w8:vec2:within:sw0:native:ONE_FULL | 8 | +1.869 | [-8.958, +12.696] | True | True |

## 4. 硬件计数与结论边界

计数表见下节及 diagnostics.json：包含全部 shared load/store 波前、bank conflict、动态 LDS/STS/SHFL/BAR.SYNC、寄存器与 q，以及 barrier、short/long scoreboard、MIO 等可用 warp-state 指标。
源行/逐 PC 归属保存在 source_code_audits，不能把整个 kernel 的 shared traffic 都算成 ML。
shared swizzle 也改变 TensorMap 的 box 宽度和 TMA 指令分组；T_control 是匹配诊断，不是可以无条件线性扣除的已知底座。
因此本轮可以检验 H4 的方向、加载冲突是否有可测代价、其驻留响应和隔离程度；不能从这些共变数据直接求出唯一 MR 同步延迟系数。
若全 kernel barrier 百分比与叙事不一致，应保留该反例，而不能只看有利指标。
按 [NVIDIA NCU 文档](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)，short scoreboard/MIO 涉及 shared、特殊函数或分支等多种操作；warp-state 百分比不能直接相加为阶段纳秒。

## 5. 原始证据、复核与停点

[全部计数摘要](COUNTERS.md)；完整 FULL 三点斜率与各 worker 原始时间见 analysis.json，全部原始事件见 timing/invocation_*/raw.json。
compile_gate.json/compiled_payload.zip 固定实际 CUBIN；artifact_gate.json 绑定真实寄存器、local bytes、q 与正确性。
profiles/*/attempt_*/export 保存 ncu-rep、source/raw CSV、worker UUID 与实际加载 SHA；offline_import_validation.json 保存独立本地重导入核对。
诊断只使用本阶段冻结数据，不把历史 26.6 µs 或人工总分引入模型。

M4a 首次 CPU preflight 在编译前发现上传快照遗漏 c16/kernels.py；失败 dispatch/retrieval 记录保留，CPU-only packaging revision 增补依赖，36 个程序与指标清单未改，未产生 GPU 样本。
本阶段通过 validator 后生成 freeze.json、commit/push。只继续用户已授权的 M4a/M4b，全部完成后停下；没有新模型拟合、C++ 修改、PR 或跨架构实验。

## 6. TensorMap 身份去重修正

初版去重只看 CUBIN，错误合并了 32×64/w8 的 sw0/sw128 T_control；两者 TensorMap 不同。
初版原始样本、全部报告、源码、派生产物、freeze 和初版 validator 记录完整保留于原路径与 audit_revisions/1/frozen_state.zip。
初版 validator 未检查 TensorMap 身份，属于本轮测量工具缺陷；不将其通过状态当作最终有效性证明。
revision3 将 TensorMap metadata、extent、dtype 和 geometry 纳入执行身份，正式计时由新三个 worker 完整重做，避免混配不同采集批次。
初版 7,560 个样本仅作历史证据，不进入本报告主统计；新主统计是 7,776 个样本、216 个条件/worker。
原有 8 份有效 NCU 报告复用，只补缺少控制的 2 份报告。最终 36 个不同执行上下文虽然只有 35 个不同 CUBIN，均有独立真实计数。
修正由身份核查触发，36 个案例、假说和指标集合没有改变，未按性能输赢选择重复。
