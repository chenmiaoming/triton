# Stage G8：修正生产 Gluon 基线后的完整重测

本轮重测改变了此前生产算子的结论：**源码对齐的 G0 已与原始 Triton U 对齐，但旧 Gluon 程序上的候选收益不能沿用。** SGLang 两个较大 head dimension 的既有收益未在新程序上保留；vLLM 仍有小幅收益，较好的候选转为 `vec2/vec4:within`；LayerNorm 梯度合并仍存在明显收益。与此同时，减小 vector width 能造成很大的退化，不能从 TMA/max 的结论推广出“普遍选 vec1”。

本阶段只替换受影响的 G2/G3a/G3b/G4a/G4b 生产算子比较。独立 TMA、D1、M2–M4、A10 证据不因本次缺陷重跑；它们各自的证据边界继续保留。旧样本、原始编译产物与旧报告均不覆盖，逐项传播审计见 [IMPACT_AUDIT.md](IMPACT_AUDIT.md) 与 [impact_audit.json](impact_audit.json)。

## 1. 冻结、编译和测量口径

- 使用生产来源 kernel 的合成输入复现与冻结设备 pipeline；未运行整个 vLLM/SGLang 框架或真实 LLM 模型，不能将子核收益写成在线服务吞吐收益。
- 固定编译器 `20d9d98fead78c2f65e1b3ba5a7881d86a836dad`，标准编译流水线，无 C++/native 修改。复用既有 native/build cache；CPU 编译前后 ccache 计数相同，native rebuild 为 0。
- 新编译 62 个唯一 Gluon 程序，复用 19 个经 SHA 校验的 U/stage2/dX 程序，组成 168 个逻辑编译记录。全部 156 个非 helper 的有效输入/中间/输出布局检查通过；31 对 G0/U 的 ELF allocated sections、ABI 与 launch resource 配置一致。
- 正确性门包含 798 项 inference 检查与 270 项训练检查，全部通过。包括完整输出、输入保留、canary、三种输入模式；训练 merge 参考为精确 BF16 partial 的 FP64 求和后舍入，完整 backward 的 dX 用独立公式核对，最终梯度按实际生成的 partial 核对。它不要求不同浮点累加顺序逐位相同。
- 517 个逻辑计时条件：G2 294、G3b 99、G4a 40、G4b 84。每个 worker 实测 439 个物理条件；3 个 H100 worker × 5 次随机次序 visit × 31 个样本，共 **204,135 个原始时间样本**。精确 CUBIN/ABI 别名仅共享一次物理采样，不重复统计；逻辑展开后的等价样本数为 240,405。
- 正式计时仅用 CUDA driver 加载冻结 CUBIN，GPU 进程无 Triton import/编译，所有 driver 返回码为 0。NCU 在正式计时结束后单独采集。
- 主估计为 15 个 worker/visit 中位数的均值；4,096 次配对分层 bootstrap。收益是 `100×(U−candidate)/U`，均使用本轮同 workload 的 U。负值表示候选耗时增加，例如 −100% 表示约 2 倍用时。CI 为逐点探索性区间，未做多重比较修正。下表的“最优”是事后描述，不是盲测选择器。

78 个 G0/U workload 对照的点估计范围为 **-0.636%～+0.870%**，全部 CI95 位于 ±3% 内（所有区间端点范围 -1.738%～+1.807%）。这与 G6/G7 的源码、机器代码及实际绑定门禁一致，新的默认基线没有复现旧的 U/G0 偏差。

协议与全量数据：[preregistration/protocol.json](preregistration/protocol.json)、[compile_gate.json](compile_gate.json)、[timing_analysis.json](timing_analysis.json)、[all_timings.csv](all_timings.csv)、[完整 workload 摘要](MATRIX.md)。

## 2. SGLang：旧收益不再成立，部分布局干预显著恶化

G2 覆盖 batch 1/8、context 256/2048，并同时测 stage1 与含原 stage2 的 decode。对 `d128/d256`，16 个相关 workload/模式中的最好候选均接近原默认 `vec8:n_first`；没有出现 >3% 的收益。`d64` 仅在短 context、batch 8 有小幅变化，长 context 也未出现实用收益。这里的“不成立”限定在本轮新源图的完整矩阵，不能解释为所有 SGLang shape 或所有硬件均无布局优化空间。

| head dim | batch / context | stage1 最优候选与收益 [CI95] | decode 最优候选与收益 [CI95] |
| --- | --- | --- | --- |
| d128 | 1 / 256 | `vec8:n_first` -0.18% [-0.89, +0.38] | `vec8:n_first` +0.07% [-0.50, +0.71] |
| d128 | 1 / 2048 | `vec8:n_first` -0.33% [-0.79, +0.05] | `vec8:n_first` -0.06% [-0.33, +0.24] |
| d128 | 8 / 256 | `vec8:n_first` +0.58% [+0.25, +1.04] | `vec8:n_first` -0.36% [-0.90, +0.16] |
| d128 | 8 / 2048 | `vec8:n_first` -0.03% [-0.11, +0.04] | `vec8:n_first` +0.01% [-0.11, +0.11] |
| d256 | 1 / 256 | `vec8:n_first` -0.17% [-0.53, +0.35] | `vec8:n_first` +0.33% [-0.30, +0.97] |
| d256 | 1 / 2048 | `vec8:n_first` +0.01% [-0.25, +0.25] | `vec8:n_first` +0.20% [+0.02, +0.37] |
| d256 | 8 / 256 | `vec8:n_first` -0.10% [-0.29, +0.09] | `vec8:n_first` -0.02% [-0.26, +0.25] |
| d256 | 8 / 2048 | `vec8:n_first` +0.01% [-0.03, +0.07] | `vec8:n_first` +0.06% [-0.02, +0.14] |
| d64 | 1 / 256 | `vec2:within` +0.20% [-0.59, +1.20] | `vec8:n_first` +0.39% [-0.19, +1.01] |
| d64 | 1 / 2048 | `vec8:n_first` +0.03% [-0.16, +0.28] | `vec8:n_first` +0.29% [-0.11, +0.78] |
| d64 | 8 / 256 | `vec4:n_first` +2.53% [+1.91, +3.20] | `vec4:n_first` +1.89% [+1.41, +2.31] |
| d64 | 8 / 2048 | `vec4:n_first` +0.31% [+0.12, +0.49] | `vec4:n_first` +0.23% [-0.04, +0.45] |

G4a 的 PV-only、QK-only 与协同干预也必须按新图重读。batch 8、context 2048 的 stage1 收益如下；负向 interaction 仅说明协同干预比简单相加损失小，不表示协同程序比 U 更快。全部 8 个 factorial interaction 与 CI 见 `timing_analysis.json` 的 `contrasts`。

| head dim | PV-only | QK-only | coordinated |
| --- | --- | --- | --- |
| d128 | -2.42% [-2.54, -2.31] | -44.67% [-45.89, -42.48] | -9.75% [-9.98, -9.54] |
| d64 | -25.18% [-25.94, -24.04] | -152.43% [-156.39, -145.40] | -146.86% [-150.56, -140.15] |

尤其是 `d64 / b8 / context2048 / qk_only`，耗时增加 **152.43%**（约 2.52 倍 U），协同干预仍增加 146.86%。中间布局/转换的资源和指令归因需与本轮 NCU 对应，不能仅凭 fanout 下降或旧 PC 行号解释。

## 3. vLLM：保留部分收益，但 vec1 的旧结论反转

三个 split/head-dim 几何均保留 batch 1/8/64。下表列出原输出语义下的矩阵最优，以及同一 workload 的 `vec1:n_first`，防止只展示 batch 8 的获益点。

| splits / dim | batch | 最优候选 | 收益 [CI95] | vec1:n_first 收益 [CI95] |
| --- | ---: | --- | --- | --- |
| s16_d128 | 1 | `vec4:within` | +4.28% [+3.45, +5.00] | -3.14% [-4.22, -2.15] |
| s16_d128 | 8 | `vec2:within` | +4.23% [+3.80, +4.71] | -2.61% [-3.58, -1.52] |
| s16_d128 | 64 | `vec4:within` | +7.50% [+7.13, +7.95] | -45.37% [-46.72, -43.48] |
| s32_d128 | 1 | `vec4:within` | +3.98% [+3.28, +4.60] | -19.14% [-20.46, -18.00] |
| s32_d128 | 8 | `vec4:within` | +4.04% [+3.35, +4.81] | -23.63% [-24.89, -22.01] |
| s32_d128 | 64 | `vec4:within` | +7.24% [+6.89, +7.58] | -74.74% [-76.42, -72.12] |
| s32_d256 | 1 | `vec4:within` | +3.63% [+3.06, +4.17] | -9.93% [-10.65, -9.14] |
| s32_d256 | 8 | `vec4:within` | +3.85% [+2.88, +4.84] | -7.73% [-9.95, -5.97] |
| s32_d256 | 64 | `vec4:within` | +1.63% [+1.37, +1.92] | -7.45% [-7.91, -6.99] |

`s32/d256/b8` 的最优点估计为 +3.85%，但 CI 下界为 +2.88%，因此不能称为“已解析出超过 3% 门槛的收益”。`s32/d128/b64` 的 vec1 耗时增加 74.74%，而 vec4:within 降低 7.24%；对这一点不能把低通信 fanout 等同于低完整程序成本。

G4b 保留原 actual ABI 与 B4 replay 对照。SGLang 对应实际条件在新程序上两者一致，target 没有隐藏收益。vLLM 的 batch 1 specialization 仍能影响结果：`s32/d256/b1` 的 target 在 actual 下 +5.84% [5.34, 6.34]，B4 下 +3.80% [3.15, 4.58]；两者不能混为一个结论。`s16/d128/b64` 的旧 target 在 actual 下为 **−45.10%**，而新矩阵较好的布局是 vec4:within，不能只复测旧赢家。42 个 actual/B4 配对差分见 `contrasts`。

对 G4b batch64 的额外 JIT 属性差异，另做了纯 CPU 审计：把 G7 真实 JIT 观察到的 `num_seqs % 16 == 0` 属性带入 `s16/d128`、`s32/d256` 的 G0 与 target 共 4 个程序。4/4 的完整 CUBIN、allocated sections、ABI 和编译资源与 G8 generic 形式相同，另外 2 对 U 也与 G7 真实 JIT 的执行字段一致。见 [actual_abi_audit/analysis.json](actual_abi_audit/analysis.json)。这支持复用本轮冻结二进制的 batch64 测量，但不是新增的候选实际 JIT 调用或 GPU 样本。原审计脚本写出分析后在打印摘要时出现缺失 `json` import 的错误，保留了 trace；独立离线 verifier 重新核验 4 个 Gluon 与 2 个 U 比较，未重新编译或运行 GPU。复现细节见 `REPRODUCE.md`。

## 4. LayerNorm backward：合并子核收益保留，完整图收益随占比缩小

训练候选包含每个几何全部 7 个合法 layout，以及 vec1/vec8 的 native-output 诊断。下表最优候选均保持原输出语义，不把 native-output 诊断程序当成生产优化。

| partial rows / N | merge 最优 / 收益 [CI95] | full backward M128 | full backward M2048 |
| --- | --- | --- | --- |
| p256_n1024 | `vec4:n_first` +27.88% [+27.10, +28.51] | `vec4:n_first` +11.97% [+11.40, +12.48] | `vec4:n_first` +11.36% [+11.01, +11.74] |
| p128_n4096 | `vec4:n_first` +19.70% [+18.83, +20.64] | `vec4:n_first` +10.74% [+10.17, +11.30] | `vec4:n_first` +4.56% [+4.22, +4.87] |
| p96_n8192 | `vec4:within` +15.75% [+14.98, +16.47] | `vec4:within` +6.83% [+6.24, +7.39] | `vec4:within` +2.26% [+2.03, +2.49] |

完整 backward 指原 dX/stage1 + 新 merge 的设备图，并含三个清零节点；因此 merge 的 15.75%～27.88% 不能直接写成端到端 backward 的收益。M2048 时三几何的实测最优分别为 +11.36%、+4.56%、+2.26%；最后一个 CI 位于 3% 门槛之下。

## 5. 旧结论的保留与撤回

以下数值均是各自实验内部、对各自同轮 U 归一化的收益。旧图和新图不是只有一个受控变量的同会话实验，因此本表只说明旧证据能否作为修正后布局结论继续引用，不把两列差值解释成源码修正的独立因果贡献。全量 517 条 lineage 见 [historical_gain_directions.json](historical_gain_directions.json)。

| 条件 / 原候选 | 旧程序内部收益 | 新程序内部收益 | 证据处理 |
| --- | ---: | ---: | --- |
| `C02_n64_d256` b8, stage1, `vec4:n_first` | +7.58% | -19.61% | 撤回旧 +7.58% 布局收益 |
| `C06_s16_d128` b8, merge, `vec1:n_first` | +7.45% | -2.61% | 旧 vec1 获益不保留 |
| `C06_s16_d128` b64, merge, `vec1:n_first` | +14.09% | -45.37% | 明显方向反转 |
| `C06_s32_d128` b64, merge, `vec1:n_first` | +16.79% | -74.74% | 明显方向反转 |
| `C06_s32_d256` b8, merge, `vec4:within` | +5.49% | +3.85% | 部分收益保留，仍需保留 CI 边界 |

LayerNorm 的“收益保留”指优化空间仍在，不表示旧最优布局排序都得到确认。以下先按旧实验的点估计选定旧最优候选，再查该同一候选的新结果；不把新最优与旧最优直接当成同一个程序。候选间的点估计排序也不构成经多重比较验证的排名。

| LN 几何 / 范围 | 旧点估计最优候选 | 旧程序内部收益 | 同一候选的新收益 [CI95] |
| --- | --- | ---: | --- |
| `C01_p256_n1024`, merge M128 | `vec4:n_first` | +27.62% | +27.88% [+27.10, +28.51] |
| `C01_p256_n1024`, pipeline M2048 | `vec4:n_first` | +11.33% | +11.36% [+11.01, +11.74] |
| `C01_p128_n4096`, merge M128 | `vec4:within` | +18.45% | +19.31% [+18.26, +20.23] |
| `C01_p128_n4096`, pipeline M2048 | `vec4:n_first` | +4.38% | +4.56% [+4.22, +4.87] |
| `C01_p96_n8192`, merge M128 | `vec2:within` | +16.07% | +14.44% [+13.88, +14.92] |
| `C01_p96_n8192`, pipeline M2048 | `vec4:within` | +2.13% | +2.26% [+2.03, +2.49] |

例如 `p96/n8192` 的旧 merge 最优点为 `vec2:within`（+16.07%），同一候选在新程序上为 +14.44%；新矩阵最优点则是 `vec4:within`（+15.75%）。`p128/n4096` 的旧 merge 最优点为 `vec4:within`，新最优点为 `vec4:n_first`。这些差别应连同完整矩阵和 CI 保留，不能仅写一句“旧赢家已确认”。

本轮不据此更新 C3.4，也不将生产算子重测冒充 D1 的新盲测。TMA/max、消费者机制实验与这些生产程序的布局变化不是同一测量图。

## 6. NCU：减少通信不保证完整程序更快

共 **173 份报告**：原计划 167 份，另有 6 份 post-timing 探索性诊断（为新观察到的 `s16/d128`、`s32/d128` 的 `vec4:within` 补计数）。三种模式分别为 software 69、warm application 69、cold isolated 35。全部已在本地从 `.ncu-rep` 重导入；software 报告的执行 PC 已与各自当前 CUBIN 的反汇编和 debug source 记录逐项匹配，未沿用旧报告硬编码的源码行号。见 [counter_analysis.json](counter_analysis.json)。

9 对 U/G0 的 software 指令族计数逐项相同，进一步支持新基线对齐。硬件 wavefront、bank conflict 等计数来自分别运行的报告，少量浮动不应被解释成等价 U/G0 的源程序差异。

下表 `SHFL/LDS/STS/BAR` 是 software 的动态 warp 指令次数/CTA；global-load sectors 与 shared-load wavefronts 是 warm application 的实际网格总和除以 CTA 数。两者来自不同 replay 条件，列在一起用于检查程序结构和请求变化，不能相加作时间成本。`q` 是 driver 查询的容量上限，不是实际达到的 occupancy。

| vLLM 几何 / 候选 | registers / q capacity | SHFL / LDS / STS / BAR | warm global-load sectors / CTA | warm shared-load WF / CTA |
| --- | ---: | ---: | ---: | ---: |
| s16_d128 / `upstream` | 32 / 16 | 64/12/24/20 | 324.50 | 36.00 |
| s16_d128 / `vec1:n_first` | 56 / 9 | 32/0/0/0 | 420.50 | 0.00 |
| s16_d128 / `vec2:within` | 48 / 10 | 48/0/0/0 | 356.50 | 0.00 |
| s16_d128 / `vec4:within` | 32 / 16 | 64/4/16/0 | 324.50 | 4.00 |
| s32_d128 / `upstream` | 56 / 9 | 72/12/24/20 | 628.50 | 36.00 |
| s32_d128 / `vec1:n_first` | 96 / 5 | 40/0/0/0 | 820.50 | 0.00 |
| s32_d128 / `vec4:within` | 56 / 9 | 72/4/16/0 | 628.50 | 4.00 |
| s32_d256 / `upstream` | 96 / 5 | 56/12/16/20 | 1204.50 | 36.00 |
| s32_d256 / `vec4:within` | 96 / 5 | 64/0/0/0 | 1204.50 | 0.00 |

`s16/d128` 与 `s32/d128` 的新 `vec4:within` 诊断保留了 U 的寄存器、容量 q 和 global-load sector 请求，却把 BAR 从 20 减为 0、warm shared-load wavefront 从 36 减为 4。`s32/d256` 的 vec4:within 也保留了这些资源/请求量并消除 shared-memory 往返，但 SHFL 从 56 增为 64；并非所有通信计数都同时下降。这些 post-timing 诊断解释了为何需要检查新赢家的完整代码结构，不能沿用旧 vec1 的解释。

`s16/d128` 的 vec1 确实删掉了表中的 shared-memory 通信，但寄存器从 32 升到 56、容量 q 从 16 降到 9，warm global-load sector 请求也增加；`s32/d128` 的 vec1 也在删通信的同时把寄存器从 56 升到 96、q 从 9 降到 5。它们说明“通信少了”与“完整 kernel 快了”不是同一个判断。这里尚未通过独立干预把寄存器、加载请求与依赖链各自造成的时间份额辨识出来，不能把任何一个计数直接乘权重解释全部退化。

| SGLang 几何 / 干预 | registers / q capacity | SHFL / LDS / STS / BAR | warm global-load sectors / CTA | warm shared-load WF / CTA |
| --- | ---: | ---: | ---: | ---: |
| n64_d64 / `upstream` | 56 / 9 | 1664/132/136/264 | 4252.00 | 325.22 |
| n64_d64 / `vec1:n_first` | 79 / 6 | 672/896/224/448 | 4252.00 | 1664.43 |
| n64_d64 / `qk_only` | 142 / 3 | 7232/488/1192/588 | 5148.00 | 1224.20 |
| n64_d64 / `coordinated` | 130 / 3 | 6240/388/416/388 | 5148.00 | 644.98 |
| n64_d128 / `upstream` | 83 / 5 | 1984/228/204/396 | 8468.00 | 612.14 |
| n64_d128 / `vec1:n_first` | 128 / 4 | 1216/1632/260/324 | 8468.00 | 3168.51 |
| n64_d128 / `qk_only` | 96 / 5 | 3456/404/588/588 | 8724.00 | 980.15 |
| n64_d128 / `coordinated` | 95 / 5 | 2944/148/136/268 | 8724.00 | 340.09 |

例如 `d64` 的 vec1 把 SHFL 从 1,664 降到 672，却把 LDS 从 132 增到 896、STS 从 136 增到 224、BAR 从 264 增到 448，且寄存器 56→79、q 9→6。这与其完整程序退化相伴，足以否定“只看归约 shuffle/fanout 就能预测该图收益”的解释；它尚不是这些变化各自贡献多少 ns 的因果分解。PC→source 表保留了实际中间转换、归约和消费者处的指令，不再用旧源图的行号给新图归因。

| LayerNorm merge 几何 / 候选 | registers / q capacity | SHFL / LDS / STS / BAR | warm global-load sectors / CTA | warm shared-load WF / CTA |
| --- | ---: | ---: | ---: | ---: |
| p256_n1024 / `upstream` | 96 / 5 | 192/40/96/76 | 4096.00 | 136.00 |
| p256_n1024 / `vec4:n_first` | 125 / 4 | 64/24/48/44 | 4096.00 | 72.00 |
| p128_n4096 / `upstream` | 96 / 5 | 192/40/96/76 | 2048.00 | 136.00 |
| p128_n4096 / `vec4:n_first` | 125 / 4 | 64/24/48/44 | 2048.00 | 72.00 |
| p96_n8192 / `upstream` | 96 / 5 | 192/40/96/76 | 1536.00 | 136.00 |
| p96_n8192 / `vec4:within` | 128 / 4 | 64/8/32/4 | 1536.00 | 8.02 |

LayerNorm 的表格同时给出另一个边界：vec4 候选虽然把容量 q 从 5 降到 4，merge 仍更快。因此 q 降档本身也不足以判定完整程序会变慢；本轮未构造把各因素独立分离的因果时间模型。

173 份 profile 绑定的 local-memory 分配均为 0，69 份 software 报告也未执行 LDL/STL；不能把这里的寄存器增加或退化直接称为 spill。这一结论仅覆盖本轮实际 profile 的程序。

计数边界必须保留：inference software 使用 batch1/grid1，warm/cold 使用 batch8（SGLang context2048、vLLM length8192），没有 batch64 的独立 NCU。本轮 LN profiles 只覆盖 standalone merge（M=P），不能直接分解完整 backward，尤其不能直接套到 M128 pipeline 中 P 被截为 min(P,M) 的 merge。

vLLM warm 报告的 DRAM 利用率仅 0.00%～0.83%；这些合成 batch8 工作集可落在 L2，不能照搬此前大 TMA 图的 HBM roofline 解释。L1TEX global sector 请求数不等于 DRAM 传输字节或 DRAM transaction 次数。

software patch 计数、warm application replay 与 cold isolated replay 分开保留，正式时间来自无 profiler 的 CUDA Graph 测量。NVIDIA 文档说明软件 patch 会引入运行开销，kernel replay 默认刷新缓存，而 application replay 配合关闭 cache control 可保留应用自己的预热状态；因此不能把这些不同条件的时延混成同一个性能估计。[NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

## 7. 复现与交付校验

执行顺序、缓存复用、补充 actual-attribute 审计与解释边界见 [REPRODUCE.md](REPRODUCE.md)。本轮正式 timing 不做按结果决定的重复采样。旧证据受冻结协议中 5,213 个保护 SHA 约束。报告与派生矩阵由 `g8_report.py` 从冻结 JSON 再生；最终独立校验记录见 [final_validation.json](final_validation.json)。

本轮在受影响的生产程序重测范围内结束，不继续启动下一阶段或提交 PR。
