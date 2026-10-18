# Stage C3.2 — 组合响应校准、资源门更新与分层预测冻结

日期：2026-10-18。仅 H100 sm90、BF16 输入/FP32 输出、host-created descriptor、沿 M max；没有修改 C++。

**这是开发集校准与 D1 预测冻结，不是盲测成功报告。** 在当前探针/对照口径下，C2c 的公共底座加稳态 MC 简单组合未能解释全部已测布局；本阶段比较三个可复算的静态组合响应模型，保留未闭合的误差，不把历史 26.6 µs、全图截距或每布局经验常数放入选择器。

## 1. 输入、固定 R 与标定边界

选择器只接受 `(BM,BN,W,dtype,reduce_axis,register_layout,shared_layout)`；计时数据先离线归一为 ns/tile/SM，运行时 grid 与全图尺寸不进入选择。保留 C3 的静态 ML 地址/波前与 MR 操作计划；旧 `2R+2columns+48` 寄存器上界和 P_shape_q 不再用于决策。

局部特征使用 C2c 固定 R 的 `native−fixture` 与 `wide−fixture_wide` 配对净成本，加上静态局部 max 工作量、树深、拓宽 partial 数。有符号净成本保留；不再把 `(p−1)/p` 乘变 R 探针时间作为独立惩罚。BF16 p=2 标量干预在后端被重新打包，NCU 已证实动态 max 次数与 native 相同，因此尚未独立辨识 packing 单项；不把其差值命名为裸指令延迟，也不宣称可以将 packing、conversion 和调度交互唯一分解。

同一组非负系数共享于全部几何；ML 波前/warp 指令、MR shuffle/波前/barrier/通信列数、输出线程/warp 重分布与 q 响应同时进入组合 MC 校准。这是结构约束的经验调度响应原型，并不保证跨特征共线性下每个系数都可作物理解释。固定 R 表按逆 q 插值；局部 R 支持为 4/8/16/32，MC 支持与外推标记分别保留。模型实现细节在 C2c 开发证据上完成；方法及模型家族已于 GPU 补测之前绑定，不声称每一个实现特征都已预注册。

## 2. 组合检验与嵌套留几何验证

比较 `zero_base=max(control,MC)`、非负共享特征的 `shared_tile_base` 和 floor-aware 的 `shared_full_service`。control 由独立 loop-free TMA/首行/写回对照拟合为 bulk 与有限驻留响应的 max；它包含等待和输出工作，仍不是纯 HBM 平台。shared_full_service 直接预测单次完整 tile 服务，再与 control 取 max；这是操作性修正形式，不声称辨识了独立 B_tile。

六个 MC recipe 为 direct/warp-normalized q × ridge {0,0.001,0.01}。八几何外层各留出一个完整几何，内层选择 MC recipe 和组合模型；外层预测的 q 也由排除该几何的资源模型估计，未偷用该候选 cubin 的实测寄存器。主评估为自然 q，受控 q 仅校准响应。

| 全开发集 recipe | 组合候选 | 内层几何均衡配对 RMSE ns |
|---|---|---:|
| {'local_q_scaling': 'warp_normalized', 'ridge': 0.01} | zero_base | 33.95 |
| {'local_q_scaling': 'warp_normalized', 'ridge': 0.01} | shared_tile_base | 33.95 |
| {'local_q_scaling': 'warp_normalized', 'ridge': 0.01} | shared_full_service | 48.30 |

最终选择 `zero_base`；嵌套外层配对 RMSE **34.59 ns**。完整单次 tile 服务绝对相对误差 median 4.56%，最大 37.69%。这些是跨几何开发验证，不等于 D1 泛化误差。

| 外层留出几何 | 内层组合选择 | 完整服务绝对相对误差 median / max |
|---|---|---:|
| 16x128w4 | zero_base | 2.12% / 5.55% |
| 16x64w4 | zero_base | 4.42% / 28.77% |
| 32x128w4 | zero_base | 1.66% / 3.33% |
| 32x64w8 | zero_base | 5.86% / 24.96% |
| 64x64w8 | zero_base | 4.55% / 11.27% |
| 8x128w4 | zero_base | 2.25% / 20.68% |
| 8x256w4 | zero_base | 9.44% / 10.70% |
| 8x64w8 | zero_base | 34.87% / 37.69% |

在同样八个历史 FULL 点上，C3.1 最大绝对相对误差 18.80%，本轮静态点预测为 21.99%。旧全图单点与本轮 grid slope 的 cohort、缓存与截距口径不同，分别保存，不据此宣称组合物理机制已闭合。若 zero_base 获选，只表示它在这几个候选模型中配对外推误差最小；不能解释为物理底座 B=0。

## 3. 单侧配对不确定性与资源门

静态预测 q 的外层整体差分残差为 [-78.63, 25.24] ns，包含资源预测误差，作为整体验证保留。条件于离线实测 q 的外层 FULL 成本差分残差为 **[-78.63, 25.24] ns**，用于成本不确定性，未做零点反射。额外单侧裕度为 max(5 ns, 几何均衡正向残差 RMS) = **5.00 ns**，再按候选减默认特征差分的 leverage 与 R 外推放大。不是置信上界，也不是样本内最大值直接充当硬保证。

每个误差场景保留同一外层记录的 `(default residual, delta residual)`；公共底座不以候选上界减默认下界重复放大。条件成本残差与下述资源场景分开，避免把静态 q 误差计两遍；实测 q 只用于离线标定，不提供给选择器。驻留场景来自配对寄存器残差、公共 ±8-register leverage margin 与候选差分余量，共享同一 smem 边界；未独立遍历候选/默认 q 区间的全部极值。仅所有联合场景仍预测 >3% 收益时改选。

通过安全门后，若候选的 nominal 完整服务严格相同（传输 floor，数值容差 1e-8 ns），按预测 MC 服务更低者打破并列，保留片上余量；不按亚纳秒的 margin 差异排序，也不使用实测赢家。该规则是本阶段开发实现，不声称已在 GPU 补测前逐项预注册。

资源普查按单次完整/驻留循环/局部 fixture 分族，452 条记录、341 个唯一 cubin/族。单次完整族 50 个，实测 n_regs 范围 14–80；留几何寄存器残差 [-18.22, 24.24]。

R128/256 已有单次完整 cubin 的实际分配依据，不再因 `2R` 自动判 >255。资源区间是经验校准，不能证明所有候选无 spill；未测过的 W=2 驻留响应与超出 MC 性能域的 R128/256 仍明确标记外推，不能以“寄存器门已修正”代替性能验证。

## 4. 开发集选择与消融

下列实际收益来自本轮 C2c loop-free 程序的自然 q、三 worker median grid slope；与旧 Stage B 的全图单点百分比不混用。选择器先给出静态决定，随后只用于评分；完整开发结果不能算成未见验证。

| 开发几何 | 默认 | nominal | 保守选择 | 实测所选收益 | 已测候选最优收益 |
|---|---|---|---|---:|---:|
| 16×128/w4 | vec8:N-first | vec1:N-first | vec8:N-first | 0.00% | 3.26% |
| 16×64/w4 | vec8:N-first | vec1:within | vec8:N-first | 0.00% | 36.68% |
| 32×128/w4 | vec8:N-first | vec1:N-first | vec8:N-first | 0.00% | 0.21% |
| 32×64/w8 | vec8:N-first | vec1:within | vec1:within | 45.36% | 45.47% |
| 64×64/w8 | vec8:N-first | vec1:within | vec8:N-first | 0.00% | 16.04% |
| 8×128/w4 | vec8:N-first | vec1:N-first | vec2:within | 该候选未实测 | 39.86% |
| 8×256/w4 | vec8:N-first | vec2:N-first | vec8:N-first | 0.00% | 0.05% |
| 8×64/w8 | vec2:N-first | vec1:within | vec2:N-first | 0.00% | 8.24% |

开发几何的新增 within 候选并非每个都已编译/计时；“该候选未实测”只是一条静态预测，不能算作成功改选。已测候选最优值只对本轮观测子集定义，未测候选不以插值或 nominal 预测替代实际结果。

| 消融 | 开发集改选数 | 留几何配对 RMSE ns |
|---|---:|---:|
| Comm-blind | 2/8 | 40.99 |
| Envelope-blind | 8/8 | 仅诊断 bypass |
| Fixed-R-blind | 1/8 | 34.27 |
| Load-blind | 3/8 | 34.50 |

三项特征消融固定完整模型选定的 recipe 后重新拟合并留几何验证，未独立嵌套选参；Envelope-blind 仅移除 control floor，未独立校准。消融若不改变最终选择，也如实保留，不人为制造差异。固定 vec 与 fanout-only 基线、逐布局误差、被拒原因和 q 场景见 development.json。

本轮 Load-blind 与 Fixed-R-blind 的配对 RMSE 未变差，尚未显示加入这些特征可改善跨几何预测；静态操作计划匹配与局部探针有效，不等于组合模型的每项参数已独立辨识。

## 5. 历史分层与 D1 冻结

共 **11** 个保护几何：{'empirically_unseen_core': 6, 'legacy_boundary': 5}；历史类别 {'strict_zero_mention': 10, 'unexecuted_code_only': 1}。strict_zero_mention 指 pre-C3 历史切点，从 C3 开始的设计/静态预测文字不算 GPU 暴露。code-only 子集保留此前静态规则预测标记。

No empirically unexposed R=8 or BM=32/64 core combinations remain under the recorded history. Balanced retrospective coverage must be reported separately, never called blind.

冻结所有 6 个未暴露核心域几何，同时保留全部旧边界，避免因默认弃权而删掉不利 case。BM32/64、R8 的均衡覆盖单列为已暴露的 retrospective panel；不并入盲测分母。D1 以后必须分别报告历史 strict 子集、全部未见核心域、边界集；这些统计有交集，不能相加当作独立样本。

| D1 几何 | R | 历史类别 / 层 | nominal | 保守选择 |
|---|---:|---|---|---|
| 8×64/w4 | 4 | strict_zero_mention / empirically_unseen_core | vec1:within | vec4:N-first |
| 8×128/w8 | 4 | strict_zero_mention / empirically_unseen_core | vec1:within | vec4:N-first |
| 8×512/w4 | 32 | strict_zero_mention / empirically_unseen_core | vec4:N-first | vec8:N-first |
| 8×512/w8 | 16 | strict_zero_mention / empirically_unseen_core | vec2:N-first | vec8:N-first |
| 16×64/w8 | 4 | unexecuted_code_only / empirically_unseen_core | vec1:within | vec4:N-first |
| 16×128/w2 | 32 | strict_zero_mention / legacy_boundary | vec2:N-first | vec8:N-first |
| 16×256/w8 | 16 | strict_zero_mention / empirically_unseen_core | vec1:N-first | vec8:N-first |
| 32×256/w2 | 128 | strict_zero_mention / legacy_boundary | vec4:N-first | vec8:N-first |
| 64×128/w2 | 128 | strict_zero_mention / legacy_boundary | vec2:N-first | vec8:N-first |
| 64×512/w4 | 256 | strict_zero_mention / legacy_boundary | vec4:N-first | vec8:N-first |
| 128×64/w2 | 128 | strict_zero_mention / legacy_boundary | vec2:N-first | vec8:N-first |

D1 预报改选 0/11。这是冻结预测数量，不是有效性结论。每个候选的预测、资源区间、配对单侧区间、联合 q 场景、外推标记和回退原因均已保存；不因看见 GPU 结果再改冻结选择。

**全部 D1 保守选择仍回退默认：模型尚未取得未见域的有效选择分辨率，不能宣称 C3.2 已构成有效选择模型。** nominal 候选、数值区间和回退原因仍冻结，便于复核判断下一步应修订哪些可辨识假设；没有自动进入 D1。

## 6. 当前结论与停点

已完成：C2c 原始证据引用、固定 R 代理替换、共享组合模型比较、嵌套配对残差、物理寄存器分族校准、三级历史审计与所有候选预测冻结。尚未证明：唯一物理底座/纯传输分解、跨设备适用性、D1 胜率及 PR 可合入性。若开发集改选不足或外层误差仍大，按模型限制报告，不能以全部回退默认宣称模型有效。

**D1 的候选编译、加载、NCU 与 GPU 计时均未启动；此处停止等待复核。没有创建 PR。**

[参数与外层残差](theta_H100.json) · [开发/消融](development.json) · [资源校准](resource_calibration.json) · [历史审计](history_audit.json) · [D1 全候选冻结预测](D1_predictions.json) · [回顾性覆盖](retrospective_coverage_predictions.json) · [冻结绑定](freeze.json) · [验证](validation.json)
