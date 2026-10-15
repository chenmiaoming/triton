# Phase 11 Stage B — H100 Gluon layout sweep

固定upstream、未修改native、同一个Gluon kernel，通过BlockedLayout控制vec1/2/4/8。数学工作、TMA/shared布局和输出布局固定，实际lowering与资源共同变化保留。

| Condition | vec1 μs | vec2 μs | vec4 μs | vec8 μs | 观测最低 | 仍可能最优 |
|---|---:|---:|---:|---:|---|---|
| M32_N128_w4:bfloat16:max:device | 58.528 | 58.384 | 58.752 | 58.992 | fixed_vec2 | fixed_vec1, fixed_vec2, fixed_vec4, fixed_vec8 |
| M32_N128_w4:bfloat16:max:host | 56.048 | 56.160 | 56.464 | 57.296 | fixed_vec1 | fixed_vec1, fixed_vec2, fixed_vec4, fixed_vec8 |
| M32_N128_w4:float32:max:device | 102.688 | 103.296 | 103.216 | 104.256 | fixed_vec1 | fixed_vec1, fixed_vec2, fixed_vec4, fixed_vec8 |
| M32_N128_w4:float32:max:host | 101.360 | 101.520 | 101.728 | 103.248 | fixed_vec1 | fixed_vec1, fixed_vec2, fixed_vec4, fixed_vec8 |
| M32_N64_w8:bfloat16:max:device | 85.120 | 85.824 | 92.256 | 143.440 | fixed_vec1 | fixed_vec1, fixed_vec2 |
| M32_N64_w8:bfloat16:max:host | 61.392 | 62.048 | 68.528 | 112.096 | fixed_vec1 | fixed_vec1, fixed_vec2 |
| M32_N64_w8:float32:max:device | 110.000 | 110.752 | 113.696 | 158.512 | fixed_vec1 | fixed_vec1, fixed_vec2, fixed_vec4 |
| M32_N64_w8:float32:max:host | 99.328 | 99.440 | 99.920 | 132.160 | fixed_vec1 | fixed_vec1, fixed_vec2, fixed_vec4 |

完整保留22,400个单kernel CUDA-event样本，7次独立worker调用；物理GPU可重复，不能把launch数或worker数当成独立设备数。表中为worker中位数的中位数。主口径是full grid，1/16 grid只作规模诊断。六个布局对照使用同条件内Bonferroni/t df6区间，预声明实用差异3%；不合成成本总分，不拟合权重。

| Canonical↔matching-layout Gluon | Gluon时间减少 % | 95%描述区间 % | 完整过滤指令序列相同 |
|---|---:|---|---|
| M32_N128_w4:bfloat16:max:device | 0.224 | [-0.198, 0.645] | True |
| M32_N128_w4:bfloat16:max:host | 0.225 | [-0.221, 0.668] | True |
| M32_N128_w4:float32:max:device | 0.113 | [-0.131, 0.357] | True |
| M32_N128_w4:float32:max:host | -0.192 | [-0.476, 0.090] | True |
| M32_N64_w8:bfloat16:max:device | 0.126 | [-0.043, 0.295] | True |
| M32_N64_w8:bfloat16:max:host | -0.140 | [-0.353, 0.073] | True |
| M32_N64_w8:float32:max:device | 0.163 | [0.009, 0.316] | True |
| M32_N64_w8:float32:max:host | -0.048 | [-0.338, 0.241] | True |

Triton锚点与Gluon使用不同pipeline；相同输入layout不足以证明等价。对应binary的代码生成、资源与独立NCU报告一并保留。过滤指令签名覆盖完整kernel，包括初始加载与输出，不称为纯规约耗时或纯规约计数。跨pipeline差异不作为布局因果收益。

本阶段没有进行C1/C2机制干预、跨架构性能复制或创建PR；在阶段B停止，等待讨论是否调整下一步。
