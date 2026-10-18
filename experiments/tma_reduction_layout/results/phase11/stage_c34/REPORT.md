# Stage C3.4 — 资源区间联合校准与 η(q) 独立检验

## 1. 范围、数据与主要结果

纯 CPU 分析；C2d、C3.3 和更早证据逐文件 SHA 保持不变。无 GPU 分配、无 native rebuild、无 D1 GPU 编译/计时、无新 PTX/NCU。继续使用固定 host descriptor 路径的已有实验。十个 D1 几何及历史分层不变；本阶段只更新静态预测，不读取其性能或寄存器真值。

资源点回归保留 C3.3 的 74 个去重 loop-free FULL CUBIN：九个开发几何 61 个候选，加两个已暴露资源边界几何的 13 个候选。后者是资源标定，不是 D1。resident MC/局部 fixture 的寄存器不进入 FULL 资源拟合。运行时间标签仍只有九个开发几何；本阶段不新增任何标签。

九折资源连续寄存器上界覆盖 **60/61**（C3.3 61/61），边际 q 区间覆盖 **61/61**，候选/默认联合 q 场景覆盖 **52/52**。q 为 CUDA occupancy capacity，不等于每时刻实际活跃 CTA 数。

η 族由内层 FULL/pair 联合误差选取。最终全数据 recipe：`partial_overlap_q; eta8=1.0, b=0.25, ridge=0.01`。完整自动选择的外层 FULL 相对 RMSE **7.71%**，paired RMSE **31.091 ns**；标量限制族为 **7.45% / 31.769 ns**，q 相关限制族为 **7.64% / 30.168 ns**。不能用内层选中次数替代外层检验。

完整 C3.4 外层保守改选 6/9；D1 预先预测改选 2/10。这些计数是结果，不参与资源裕量或 η 参数选择。开发集上的简单基线已很强，不能以放行已知赢家或复杂模型胜过 fixed_vec1 为校准目标。

关键不利结果：自动 η 族的性能上尾覆盖仅 **42/52**，旧 C3.3 为 49/52；外层 conservative 平均实测收益 **18.27%**，低于仅资源修正的 **20.20%** 和 fixed_vec1 的 **21.36%**。η(q) 不能据此认定改善了选择器；最终仍按预注册的内层选择结果冻结，未用外层结果重新挑族。

## 2. 资源近邻、两层裕量与校准排除

先选 `[R/2,2R]`；不足三个不同几何时按 `|log₂(R′/R)|` 从近到远扩展，并保留同尺度所有并列几何。低 R 的池不再直接跳到包含 R128/R256 的全尺度池。配对残差优先同 placement；不够三个几何时先在当前尺度池内放宽 placement。pool、扩展半径、来源几何全部保存在 resource_calibration.json。

公共误差明确为默认布局的 register residual；配对误差为同几何候选 residual 减默认 residual。两者区间包含零，并构造共同偏移的矩形包络。它仍是经验上界，可包含未实际共同出现的残差组合；不宣称是精确联合分布。相比旧版，还修正了将所有布局误差都当公共默认误差的口径。

附加裕量 `(m_common,m_pair)` 从预注册离散网格联合选择。仅使用外层训练集的 n_regs/q 真值；每条校准观测来自内层 whole-geometry 留出，而该观测的误差包络又在剩余几何上 cross-fit，不将自身误差加入自己的包络。达到训练校准集 100% 寄存器上界及联合 q 覆盖的组合中，先最小化两裕量之和，再最小化单项最大值，再选较小公共裕量。训练校准目标不是外层覆盖保证；无计时、收益、改选率输入。

q 场景遍历连续寄存器矩形内所有可达分配档，并传播静态 shared 上下界。只枚举四角可能漏掉中间 q 档，本版保留这些中间档。寄存器 >255、shared 非法、W/R 超出 FULL 支持等原有门限仍记录并保留。

| 留出几何 | R | N-first 附加裕量 公共/配对 | within 附加裕量 公共/配对 | 寄存器上界覆盖 前→后 | q 区间覆盖 前→后 | 联合 q 对覆盖 前→后 |
|---|---:|---|---|---|---|---|
| 8x64w8 | 2 | 4/0 | 4/0 | 4/4 → 4/4 | 4/4 → 4/4 | 3/3 → 3/3 |
| 8x128w4 | 8 | 0/0 | 0/0 | 7/7 → 7/7 | 7/7 → 7/7 | 6/6 → 6/6 |
| 8x256w4 | 16 | 0/0 | 0/0 | 6/6 → 6/6 | 6/6 → 6/6 | 5/5 → 5/5 |
| 16x64w4 | 8 | 0/0 | 0/0 | 8/8 → 8/8 | 8/8 → 8/8 | 7/7 → 7/7 |
| 16x128w4 | 16 | 0/0 | 0/0 | 7/7 → 7/7 | 7/7 → 7/7 | 6/6 → 6/6 |
| 32x64w8 | 8 | 0/0 | 0/0 | 8/8 → 8/8 | 8/8 → 8/8 | 7/7 → 7/7 |
| 32x128w4 | 32 | 0/0 | 0/1 | 7/7 → 6/7 | 7/7 → 7/7 | 6/6 → 6/6 |
| 64x64w8 | 16 | 0/0 | 0/0 | 8/8 → 8/8 | 8/8 → 8/8 | 7/7 → 7/7 |
| 16x64w8 | 4 | 2/0 | 2/0 | 6/6 → 6/6 | 6/6 → 6/6 | 5/5 → 5/5 |

连续上界漏覆项（不在看见结果后加裕量修补）：

- `M32_N128_w4:vec1:n_first`：实测 32，连续上界 31.924457，差 0.075543 个寄存器；实测 q=16，预测 q 集合 [16]。

寄存器是整数，但这里保留原连续模型的上界覆盖口径；不靠事后向外取整把漏覆改记为覆盖。61/61 或 52/52 的容量覆盖仍只来自九个相关几何，不是 D1 的安全保证。低 R 训练集的裕量结果包含零，表示既定 coverage 规则选到零，不能解读为先删裕量再证明改选合理。

## 3. 五个重点案例：旧版、仅资源更新、完整更新

下表始终跟踪 **旧 C3.3 的 nominal 布局**，从而避免通过更换比较对象制造改善；新 nominal 如不同另见 diagnostics.json。开发案例用各自外层留出模型，D1 用最终全开发集拟合。仅资源更新保持旧时间模型及旧性能误差库逐项不变；完整更新才采用内层选定的新 η 族和相应 cross-fit 性能残差。

| 案例 / 锚定布局 | 版本 | nominal 预测收益 | 固定 q 下界 | 仅 q 场景最差收益 | 最差 q(候选/默认) | 加性能上界后下界 |
|---|---|---:|---:|---:|---|---:|
| development 16×64/w8 / vec1:within | C3.3 | 29.73% | 12.16% | 10.24% | 6/8 | -6.35% |
| development 16×64/w8 / vec1:within | 仅资源更新 | 29.73% | 12.16% | 29.73% | 8/8 | 12.16% |
| development 16×64/w8 / vec1:within | 完整 C3.4 | 26.76% | -3.15% | 26.76% | 8/8 | -3.15% |
| development 8×64/w8 / vec1:within | C3.3 | 20.34% | 11.59% | -100.20% | 2/5 | -118.04% |
| development 8×64/w8 / vec1:within | 仅资源更新 | 20.34% | 11.59% | 20.34% | 8/8 | 11.59% |
| development 8×64/w8 / vec1:within | 完整 C3.4 | 25.37% | 11.71% | 25.37% | 8/8 | 11.71% |
| D1 8×64/w4 / vec1:within | C3.3 | 26.89% | 11.44% | 8.80% | 12/16 | -6.08% |
| D1 8×64/w4 / vec1:within | 仅资源更新 | 26.89% | 11.44% | 26.89% | 16/16 | 11.44% |
| D1 8×64/w4 / vec1:within | 完整 C3.4 | 19.97% | 9.78% | 19.97% | 16/16 | 9.78% |
| D1 8×128/w8 / vec2:within | C3.3 | 25.32% | 9.87% | 5.06% | 6/8 | -9.83% |
| D1 8×128/w8 / vec2:within | 仅资源更新 | 25.32% | 9.87% | 25.32% | 8/8 | 9.87% |
| D1 8×128/w8 / vec2:within | 完整 C3.4 | 23.93% | 13.74% | 23.93% | 8/8 | 13.74% |
| D1 8×512/w8 / vec1:n_first | C3.3 | 14.77% | -10.43% | 13.77% | 8/8 | -10.43% |
| D1 8×512/w8 / vec1:n_first | 仅资源更新 | 14.77% | -10.43% | 14.77% | 8/8 | -10.43% |
| D1 8×512/w8 / vec1:n_first | 完整 C3.4 | 10.76% | -12.71% | 6.50% | 6/8 | -13.05% |

完整更新后的新 nominal（与上表旧布局锚点区分）：

| 案例 | 新 nominal | 预测收益 | 新 nominal 最差下界 | 最终选择 | 新 nominal 否决原因 |
|---|---|---:|---:|---|---|
| 16×64/w8 | vec2:within | 28.58% | -1.33% | vec4:within | no_robust_FULL_gain |
| 8×64/w8 | vec1:within | 25.37% | 11.71% | vec2:n_first | FULL_R_outside_support |
| 8×64/w4 | vec2:within | 24.34% | 14.15% | vec2:within | 无 |
| 8×128/w8 | vec2:within | 23.93% | 13.74% | vec2:within | 无 |
| 8×512/w8 | vec1:n_first | 10.76% | -13.05% | vec8:n_first | no_robust_FULL_gain |

资源池与附加裕量（同一锚定布局）：

| 案例 | 资源更新后的池 | 公共/配对附加裕量 | 资源更新 q 对集合 | 旧选择 → 仅资源选择 → 完整选择 |
|---|---|---|---|---|
| 16×64/w8 | 16x64w4, 32x64w8, 8x128w4, 8x64w8 | 2/0 | 8/8 | vec4:n_first → vec1:within → vec4:within |
| 8×64/w8 | 16x64w4, 16x64w8, 32x64w8, 8x128w4 | 4/0 | 8/8 | vec2:n_first → vec2:n_first → vec2:n_first |
| 8×64/w4 | 16x64w4, 16x64w8, 32x64w8, 8x128w4, 8x64w8 | 0/0 | 16/16 | vec4:n_first → vec1:n_first → vec2:within |
| 8×128/w8 | 16x64w4, 16x64w8, 32x64w8, 8x128w4, 8x64w8 | 0/0 | 8/8 | vec4:n_first → vec1:n_first → vec2:within |
| 8×512/w8 | 16x128w4, 16x64w4, 32x128w4, 32x64w8, 64x64w8, 8x128w4, 8x256w4 | 0/1 | 6/6, 6/8, 8/6, 8/8 | vec8:n_first → vec8:n_first → vec8:n_first |

D1 的 n_regs、FULL 收益和是否回归均未知；表中寄存器/q 与下界是编译前静态预测。`8×512/w8` 的性能上尾没有人工缩减；其每版正残差范围、附加 margin、conditioning_keys、R/几何来源全部保留。外层 R4 缺少精确 R4 的性能 ×1.25 与两个 D1 R4 不触发该乘数的区别继续存在。

此前审计漏列了 R2 的额外硬门：外层留出唯一的 `8×64/w8` 后，FULL 训练 R 支持为 `[4,32]`，所以旧 C3.3 本来就同时有 `FULL_R_outside_support` 与性能下界否决。仅资源修正后下界为 +11.59%，剩下的 FULL 支持域门仍要求保留默认；完整 C3.4 同样保留该门。这个案例不能在移除资源污染后就记作外层成功改选。最初五案例预检误用了全开发集支持范围，错误地把 R2 选择记成改选；数值分解不受影响。原预检文件保留，纠正与 SHA 绑定见 execution/resource_only_focus_correction.json，正式外层 JSON 始终使用留出折自己的支持范围。

## 4. η(q) 的内层与外层比较

标量族与 q 相关族共用直接 FULL 竞争模型、eta8/ridge 网格、同一 FULL/pair 联合相对目标和静态 q 点预测。q 相关族为 `clip(eta8+b*log₂(q/8),0,1)`，b∈{−.25,+.25}，标量等价于 b=0。两类菜单全部进入同一内层 CV；restricted 表是从同一个已评分菜单提取各族最优项后重新拟合，不重复改变评分协议。自动选择仅取内层误差最小的 recipe，不用外层误差再选族。

eta 本身不是独立测出的物理重叠比例；原有 H 和 A 已含 q 特征，η(q) 的增量需由外层误差判定。相同 q 下不同几何的误差方向相反，不预设提高 η 可以修复所有点。

| 留出几何 | scalar 最优内层 loss×10⁴ | η(q) 最优内层 loss×10⁴ | 自动 recipe | 外层 FULL 相对 RMSE scalar / η(q) / auto |
|---|---:|---:|---|---|
| 16x128w4 | 71.102 | 54.038 | partial_overlap_q; eta8=0.75, b=0.25, ridge=0.0 | 7.35% / 4.59% / 4.59% |
| 16x64w4 | 71.579 | 60.559 | partial_overlap_q; eta8=1.0, b=0.25, ridge=0.1 | 7.35% / 8.42% / 8.42% |
| 16x64w8 | 73.043 | 59.068 | partial_overlap_q; eta8=1.0, b=0.25, ridge=0.01 | 6.34% / 9.96% / 9.96% |
| 32x128w4 | 74.562 | 63.213 | partial_overlap_q; eta8=1.0, b=0.25, ridge=0.01 | 3.36% / 3.29% / 3.29% |
| 32x64w8 | 65.781 | 61.467 | partial_overlap_q; eta8=1.0, b=0.25, ridge=0.1 | 7.10% / 8.96% / 8.96% |
| 64x64w8 | 60.821 | 57.369 | partial_overlap_q; eta8=0.75, b=0.25, ridge=0.0 | 10.20% / 6.81% / 6.81% |
| 8x128w4 | 57.141 | 61.979 | partial_overlap; eta8=0.25, b=0.0, ridge=0.0 | 7.17% / 6.46% / 7.17% |
| 8x256w4 | 66.673 | 57.024 | partial_overlap_q; eta8=1.0, b=0.25, ridge=0.01 | 9.77% / 8.46% / 8.46% |
| 8x64w8 | 68.358 | 56.425 | partial_overlap_q; eta8=1.0, b=0.25, ridge=0.1 | 4.59% / 10.49% / 10.49% |

全数据冻结曲线 η(q) 为 `{1: 0.25, 2: 0.5, 4: 0.75, 8: 1.0, 16: 1.0}`。本轮最终 η(8) 与 η(16) 均饱和为 1，q 相关变化主要发生在更低驻留档；这不支持“已识别 q=16 比 q=8 有更大的自然态重叠比例”的解释。每个外层折有自己的 recipe，上表与 JSON 分别记录。

内层 loss 是几何均衡的 FULL 相对平方误差与配对差分相对平方误差各占一半，不等于表中只按 natural FULL 计算的外层 RMSE。完整 inner_losses_by_geometry、inner_scores、holdout 排除与训练几何均已冻结。

为核对比较公平性，五种特征模式中共复算 1125 个既有 scalar inner score，与 C3.3 已冻结评分的最大绝对差为 0.000e+00。新族没有替换旧族的数据、损失函数或 q 点估计。

| 族/特征消融 | FULL 相对 RMSE | paired RMSE ns | paired 比例 RMSE pp | 外层 nominal 平均实测收益 | >3% 恶化 |
|---|---:|---:|---:|---:|---:|
| Resource/Envelope-blind:auto | 22.21% | 40.174 | 12.41 | 22.08% | 0 |
| Comm-blind:auto | 8.62% | 37.669 | 9.97 | 20.50% | 0 |
| Load-blind:auto | 6.71% | 27.360 | 7.56 | 21.84% | 0 |
| full:auto | 7.71% | 31.091 | 9.05 | 20.82% | 0 |
| full:scalar | 7.45% | 31.769 | 8.27 | 21.93% | 0 |
| full:q_dependent | 7.64% | 30.168 | 8.70 | 20.82% | 0 |
| Fixed-R-blind:auto | 7.30% | 26.706 | 8.46 | 20.86% | 0 |

消融均独立重新拟合和内层选参；Resource/Envelope-blind 只允许 direct 模型。某项消融改善误差时按原值保留，不为证明某特征必要而删除结果。受限于九个几何，拟合系数不等同独立硬件指令时延，外层表现也不能证明跨硬件有效。

自动模型 nominal-q 性能上尾覆盖 42/52（worker 中位；逐候选失败见 diagnostics.json）。性能库仍使用 C3.3 的 signed、结构/尺度条件化上尾规则；新模型残差会改变数值，但未改其最小几何数、leverage/R 裕量、正尾 margin 或 3% 门槛。

## 5. 九个开发几何的完整外层评估

| 几何 | nominal 布局 | nominal 预测 FULL 收益 | 同布局实测 | oracle 收益 | conservative 布局 | conservative 实测 |
|---|---|---:|---:|---:|---|---:|
| 16×128/w4 | vec8:n_first | 0.00% | 0.00% | 9.68% | vec8:n_first | 0.00% |
| 16×64/w4 | vec2:within | 31.49% | 38.78% | 38.78% | vec2:n_first | 37.82% |
| 16×64/w8 | vec2:within | 28.58% | 25.20% | 26.20% | vec4:within | 16.59% |
| 32×128/w4 | vec8:n_first | 0.00% | 0.00% | 0.40% | vec8:n_first | 0.00% |
| 32×64/w8 | vec2:within | 42.66% | 49.68% | 49.68% | vec2:within | 49.68% |
| 64×64/w8 | vec1:within | 27.22% | 20.52% | 21.13% | vec2:within | 20.63% |
| 8×128/w4 | vec1:n_first | 29.92% | 41.00% | 41.00% | vec4:n_first | 37.32% |
| 8×256/w4 | vec1:n_first | 11.69% | 1.92% | 2.39% | vec4:within | 2.39% |
| 8×64/w8 | vec1:within | 25.37% | 10.28% | 10.28% | vec2:n_first | 0.00% |

所有 nominal 与 conservative 所指布局均在 61/61 完整开发矩阵中有真值，未使用 oracle 训练选择器。主指标为 128/256/384 MiB 的 FULL 服务斜率 ns/tile/SM；128 MiB 原始事件时间另有启动/尾波截距，不能混用两种收益。例：32×128/w4 的 vec1 斜率收益 −0.67%，但其 128 MiB 原始时间为 51.824 µs，快于默认 52.992 µs。也不能将 ML 2×波前孤立地称为整核性能浪费。

另一个未解决的表示失败：`16×128/w4` 的外层新模型又给出 nominal 收益 0%，而该几何实测 oracle 收益为 9.68%。这与资源门无关，说明新 η 族仍可能把部分真实布局差异压到平台支路；不能把 FULL 预测机制称作已经闭合。

| 策略 | 改选/9 | 平均实测收益 | 平均 oracle regret | >3% 恶化 | 改选但实测收益≤3% |
|---|---:|---:|---:|---:|---:|
| upstream_default | 0 | 0.00% | 35.43% | 0 | 0 |
| fixed_vec1 | 9 | 21.36% | 1.05% | 0 | 2 |
| fixed_vec2 | 8 | 19.47% | 3.48% | 0 | 2 |
| fixed_vec4 | 7 | 16.28% | 8.48% | 0 | 2 |
| fixed_vec8 | 0 | 0.00% | 35.43% | 0 | 0 |
| fanout_only | 9 | 21.36% | 1.04% | 0 | 2 |
| C33_outer_nominal | 9 | 21.93% | 0.26% | 0 | 2 |
| C33_outer_conservative | 5 | 17.59% | 5.86% | 0 | 0 |
| C34_outer_nominal | 7 | 20.82% | 1.53% | 0 | 1 |
| C34_outer_conservative | 6 | 18.27% | 4.89% | 0 | 1 |
| C34_resource_only_nominal | 9 | 21.93% | 0.26% | 0 | 2 |
| C34_resource_only_conservative | 6 | 20.20% | 2.43% | 0 | 0 |
| C34_in_sample_nominal | 8 | 21.88% | 0.37% | 0 | 1 |
| C34_in_sample_conservative | 6 | 20.38% | 2.13% | 0 | 0 |

完整自动 η 族在 `8×256/w4` 触发保守改选，但同布局实测收益不足 3%；它没有造成 >3% 恶化，仍属于未兑现预测实用收益的改选。不能把“零大幅回归”写成“零误改选”。仅资源更新保留此处默认。

fixed_vec1 与 fanout_only 在开发集平均收益约 21.36%，已是强基线。复杂模型在这九个几何上的额外收益即使为正也有限；几何少、候选相关、部分差异低于 3% 实用门槛。nominal 与 conservative 的选择收益、FULL 时间预测准确率及资源覆盖率是不同问题。本阶段不宣称复杂选择器有统计显著优势或已证明合入价值。全数据拟合结果标记为 in_sample，仅 outer 作为留出几何评估。

## 6. 十个 strict_zero_mention D1 的最终静态冻结

| 几何 | 分层 | nominal | 预测收益 | 最终选择 | 选择的最差预测收益 | 改选 |
|---|---|---|---:|---|---:|---|
| 8×64/w4 | empirically_unseen_core | vec2:within | 24.34% | vec2:within | 14.15% | True |
| 8×128/w8 | empirically_unseen_core | vec2:within | 23.93% | vec2:within | 13.74% | True |
| 8×512/w4 | empirically_unseen_core | vec8:n_first | 0.00% | vec8:n_first | 0.00% | False |
| 8×512/w8 | empirically_unseen_core | vec1:n_first | 10.76% | vec8:n_first | 0.00% | False |
| 16×128/w2 | legacy_boundary | vec8:n_first | 0.00% | vec8:n_first | 0.00% | False |
| 16×256/w8 | empirically_unseen_core | vec1:n_first | 12.18% | vec8:n_first | 0.00% | False |
| 32×256/w2 | legacy_boundary | vec8:n_first | 0.00% | vec8:n_first | 0.00% | False |
| 64×128/w2 | legacy_boundary | vec8:n_first | 0.00% | vec8:n_first | 0.00% | False |
| 64×512/w4 | legacy_boundary | vec8:n_first | 0.00% | vec8:n_first | 0.00% | False |
| 128×64/w2 | legacy_boundary | vec8:n_first | 0.00% | vec8:n_first | 0.00% | False |

十个几何的 strict_zero_mention 指原始历史审计 cutoff；当前静态预测文本不算实际实验暴露。核心五个、边界五个分别记录，超出 FULL 的 R/W 支持或资源校准失败均在候选 reasons 中保留。保留默认的 selected_worst_gain=0 不代表每个备选也被预测为 0；每个备选完整场景、上尾及弃权原因见 D1_predictions.json。

## 7. 校验、复现与停点

预注册 PLAN.md 与 protocol.json 在 C3.4 拟合前写入；prior_manifest.json 绑定全部已跟踪旧证据。resource_inputs.json 仅含资源字段；resource_calibration.json 保存嵌套排除、全部尝试过的裕量组合及其覆盖失败。theta_H100.json、cross_validation.json、development.json、baselines.json、diagnostics.json 和 D1_predictions.json 保存最终模型与全部评分。

CPU 搜索按四个独立进程并行执行不同特征模式，BLAS 单线程；并行只改变调度，不改变 CV 分割、参数菜单、目标函数或选择规则。最初顺序执行在四个外层折后中止以调整调度，原始日志和当时源代码保存在 execution/initial_sequential.*；重跑完整结果采用同一冻结协议。

最终 validator 重算资源模型/裕量覆盖、63 个时间模型外层折及 inner winner loss、outer 排除、训练-only 性能误差库、全部政策算术、五案例拆解和十个 D1 预测；不通过重复完整超参数搜索制造新一轮模型选择。结果见 validation.json；source 与 derived artifacts 冻结于 freeze.json。

复现命令（从仓库根目录，已有 Python 依赖；不触发 native/GPU 编译）：

```bash
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1
.venv/bin/python -m experiments.tma_reduction_layout.phase11.c34.validate --committed
```

完成 commit 与 push 后停止。Stage D1 实际 GPU 编译和计时等待单独复核授权。
