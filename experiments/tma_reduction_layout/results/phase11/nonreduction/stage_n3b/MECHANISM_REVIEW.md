# Stage N3b 机制复核：KDA gate 的所有权匹配

本补充由已见 N2/N4 结果触发，是**后验机制诊断**，不并入原 N3 静态分母、不追加正式计时。16/16 报告覆盖 D64/D128 的 U/G0/feature_vec2/feature_vec4；两几何 U/G0 opcode 直方图完全一致。

证据为 `analysis/summary.json`（SHA256 `64d8ab56592b8baf6ea961551a7fb093ed9ed24c6135ac0f260a04938859a8eb`），并保留 N2/N4 触发 SHA、binding、原始报告。BT=32、W=8、stages=2 来自先行 U-only autotune，算术与 FP32 输出不变。

## 生成程序与硬件证据

U/G0 在 BF16 load 后、FP32 拓宽前改变所有权。feature_vec4 直接采用与 softplus/输出兼容的布局；剩余 convert_layout 是恒等编码，实际共享交换消失。表中为动态 warp 指令，BAR 不等于 CTA 同步阶段次数。[NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

| D | Variant | 总指令 | LDS/STS/BAR | SHFL | n_regs | q 容量 |
|---:|---|---:|---|---:|---:|---:|
| 128 | U/G0 | 2264 | 16/16/8 | 0 | 48 | 5 |
| 128 | feature_vec2 | 2792 | 32/32/24 | 32 | 48 | 5 |
| 128 | feature_vec4 | 2016 | 0/0/0 | 0 | 36 | 6 |
| 64 | U/G0 | 1344 | 16/16/8 | 0 | 30 | 8 |
| 64 | feature_vec2 | 1728 | 16/16/8 | 32 | 34 | 6 |
| 64 | feature_vec4 | 1144 | 0/0/0 | 0 | 25 | 8 |

feature_vec4 的 shared-load/store wavefront：D128 从 64/64→0/0，D64 从 32/32→0/0。原先 bank-conflict 已为 0，故减少的是交换本身。全部 local allocation 为 0，不能称为消除 spill；MUFU 数保持 D128=264、D64=136，非线性工作未被删去。

| D | G0→feature_vec4 load 请求/CTA | LD/ST sectors/CTA |
|---:|---|---|
| 128 | 32→48 | 始终 392/512 |
| 64 | 24→32 | 始终 200/256 |

load 指令/请求增加而 sector 数不变，不能用指令数推断流量。warm DRAM 利用率低于 0.4%，不解释为 HBM 收益；略高于 100% 的派生命中率保留原值，不用作精确概率。

**容量变化不等于实现了更高占用率。** grid 仅为 ceil(2048/32)×8=512 CTA，在 132 SM 上平均 3.879，低于容量 5 和 6。D128 的 q=5→6 是真实资源属性，却不能据此认定 achieved occupancy 上升或按 6/5 预测加速；交换消除与寄存器下降尚未分离归因。[CUDA Best Practices：Occupancy](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-best-practices-guide/index.html#occupancy)

## 原计时边界保持不变

下表沿用既有 N4；确认要求两个参考的 95% 下界均严格高于 +3%。

| D | vs U 收益 / 95% CI (%) | vs G0 收益 / 95% CI (%) | 严格双基线确认 |
|---:|---|---|---|
| 128 | 4.273504 / [3.301488, 5.202754] | 3.407324 / [2.988585, 3.875151] | 否，provisional |
| 64 | 2.710738 / [2.361741, 3.077620] | 1.904426 / [1.241909, 2.722273] | 否，实用门槛未确认 |

D64 保留同变换的实用无收益对照，不称严格零效应；D128 的精确 G0 下界为 **2.988585172510577%**，不能四舍五入后判定通过。机制成立不改变该统计结论。

本补充证明所有权匹配可消除交换并改变资源，尚未证明 >3% 收益必然发生或跨工作集/设备成立。没有给 LDS/STS/BAR、sector、寄存器赋予任意时间权重。原始数据、summary、REPORT 未改动；未读取 N5 数据，也未新增 GPU 实验。
