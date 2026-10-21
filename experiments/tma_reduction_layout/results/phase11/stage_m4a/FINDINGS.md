# M4a 主要结果（TensorMap 身份修正后的正式批次）

H4 的方向得到支持，但“ML 只是吞吐、MR 只是串行延迟”的二分解释需要收紧。
以下仅使用 revision3 的 7,776 个样本；初版 7,560 个样本、初版 validator 与全部冻结状态保留，未混入主统计。

## 1. q=1 的原反例仍成立

128B swizzle 下 within−N-first 的配对差分，负数表示 within 更快：

| geometry | q_cap | within−NF ns/tile/SM | paired worker t95 ns |
| --- | --- | --- | --- |
| [32, 128, 4] | 1 | -25.74 | [-29.81, -21.67] |
| [32, 128, 4] | 16 | +0.54 | [-1.8, 2.87] |
| [16, 256, 8] | 1 | -32.48 | [-39.03, -25.94] |
| [16, 256, 8] | 8 | -0.90 | [-6.68, 4.88] |

## 2. 合法 shared-layout 干预

同一 vec2:within、同一 native 消费者，128B→64B swizzle。两个几何在 q=1 下均精确命中 load WF 128→64、load BC 64→0 的事前预测。

| 程序 | q_cap | FULL 差分 ns | paired t95 ns | regs 相同 | MR opcode/count 相同 |
| --- | --- | --- | --- | --- | --- |
| M32_N128_w4:vec2:within:sw64:native:ONE_FULL | 1 | -36.24 | [-38.62, -33.87] | True | True |
| M32_N128_w4:vec2:within:sw64:native:ONE_FULL | 16 | +3.29 | [0.79, 5.78] | True | True |
| M16_N256_w8:vec2:within:sw64:native:ONE_FULL | 1 | -42.71 | [-45.31, -40.11] | False | True |
| M16_N256_w8:vec2:within:sw64:native:ONE_FULL | 8 | +0.79 | [-0.25, 1.84] | False | True |

32×128/w4 的 regs 22→22、natural 容量 16→16；16×256/w8 的 regs 14→15、容量 8→8。
两项 MR 的动态 SHFL 分别保持 8 和 16。输入加载实际为 LDSM.16.M88.4，不能按 vec2 字面值猜 LDS 指令宽度。
融合 max 可能归属 raw.load 源行，主要 MR 则归属固定编译器 standard.py::max 的 core.reduce；所有 PC/opcode/来源均保留。

## 3. 仍然存在的混杂与 H4 边界

swizzle 同时改变 TensorMap box、地址计算和 TMA 搬运布局；匹配 T_control 的 64B−128B 差分为：

| geometry | q condition | T_control 差分 ns |
| --- | --- | --- |
| [32, 128, 4] | 1 | -9.32 |
| [32, 128, 4] | natural | +3.95 |
| [16, 256, 8] | 1 | -14.24 |
| [16, 256, 8] | natural | +0.79 |

因此不能把 FULL 时间变化全部分配给 bank conflict，也不能把 FULL−control 直接当作已辨识的可加性阶段成本。
32×128/w4 q=1 的 NF/within whole BAR.SYNC warp 指令实例为 24/8；barrier 比例 3.55%/1.37%。
16×256/w8 对应 barrier 比例为 5.89%/1.84%。这些计数与跨 warp 同步路径方向一致，仍属全 kernel 相关诊断。
消除冲突后，within 的 short-scoreboard 比例从 8.90%→5.80%、9.82%→5.98%，而 q=1 MIO throttle 均为 0。
加载冲突也体现为依赖等待，不能简单称为完全无依赖的吞吐开销；这些百分比不等于独立阶段纳秒。
自然驻留时差异被压到较小量：32×128 出现小幅反向变化（区间为正），16×256 的区间跨零。
这支持暴露成本随并发/传输/排程变化，不等于一律完全掩盖为零；没有建立唯一 ML/MR 延迟分解，也没有校准新选择器。

36 个合法执行上下文的正确性 108/108，通过 3 个新 worker 的完整正式配对批次；10 份 NCU 报告全部本地复读。
同一 CUBIN 配不同 TensorMap 也必须独立采样，此身份约束已加入最终 validator。
