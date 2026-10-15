# Phase 11 — Explicit Gluon layout experiments

阶段 A、B、C1 和用户单独批准的 C1.5 均已完成，每阶段单独 commit。当前停在 C1.5；C2/C3/D 未执行，PR 仍需用户批准。

[C1.5 方案](c15/PLAN.md)、[结果](../results/phase11/stage_c15/summary.md)和[最终验证](../results/phase11/stage_c15/validation.json)记录 BF16 MR 与稳态区间拟合修正。
MR 在 BF16 状态上先私有步进，再 single-use widening 供给 max；8/8 个布局的 PTX max opcode 种类/静态数量匹配原完整 kernel。
需要 BF16 局部 combine 的 7 个布局恢复 `max.bf16`/`max.bf16x2`；32×64/vec8 原本没有这一步。
新编译 8 个 MR、复用 8 个原 ML CUBIN，新增 96 项完整输出检查、64 份 MR NCU 报告和 7,680 个同轮 ML/MR event 样本。
所有主斜率统一使用 K=16/64/256，单列 T1、T1−b、拟合截距与相邻区间斜率；旧 C1 同区间重拟合只写在 C1.5 derived output。
动态 K 倍检查通过，BF16 真实 SASS 的 `HMNMX2`/`VHMNMX` 计数均保留。ML 仍存在区间依赖，CTA 线性缩放和 45 µs 底座可加性尚未实测验证。
C1.5 完成后停止，等待用户决定 C2。

2026-10-15：[后续实验 review draft](FOLLOWUP_PLAN.md)保存 C/D 的讨论草案；用户随后只批准了 C1。
实际 C1 以 [C1 方案](c1/PLAN.md)和结果目录的冻结协议为准：两种几何各覆盖 N-first vec1/2/4/8，
MR 改为私有输入步进和结果寄存器单向累加，去掉草案中的归约结果反向广播反馈。
原始 `PLAN.md`、review draft 和 A/B 冻结记录保留。

[C1 结果](../results/phase11/stage_c1/summary.md)、[完整分析](../results/phase11/stage_c1/analysis.json)和[最终验证](../results/phase11/stage_c1/validation.json)覆盖 H100 的 8 布局、16 程序、192 项完整输出检查、3 个计时 worker 的 7,680 个单 kernel event 样本和 128 份 NCU 报告。
所有目标 opcode 的逻辑 body 合计动态计数严格随 K=1/16/64/256 增长；四路展开/尾部副本的逐 PC 原值与技术解释修订保留。
MR 循环 IR 没有结果广播、反向转换或显式 shared 读写，底层 scratch 属于实际规约 lowering。native/ccache 未变，原始证据及失败记录保持 write-once。

C1 在 32×64 上测到 MR 的通信/同步成本随 vec2→4→8 增长，组合斜率复现退化方向；但当前 ML+MR 的量级不匹配完整 kernel，且不能解释 32×128 的平坦表现。
MR 的私有更新让 7/8 个布局失去原 kernel 的 BF16 packed-max 优化；ML 在小 K 有非线性、完整 tile checksum 输出及资源扰动。
这些结果支持继续修正探针，尚不足以启动正式校准或宣称 cost model 已成立。C1 后等待用户审阅。

阶段A已完成。三种架构上的普通shared load无冲突/32路冲突对照，以及ldmatrix.x4对照均通过输出正确性和独立NCU report-import核对。三种架构上还分别核对两个归档kernel的真实动态shared指令与wavefront计数。工具校准没有正式性能样本，没有重建Triton native。

直接shared Requests硬件指标不可用，保留为缺失。每PC动态warp指令由software/source报告测得，wavefront与bank-conflict由单独硬件报告测得。matrix的有冲突bank计数正对照尚未测试；无冲突matrix结果不能代替它。

[阶段A结果](../results/phase11/stage_a/summary.md)和[完整分析](../results/phase11/stage_a/analysis.json)保留三架构校准和归档kernel的计数。早期PTX探针错误及旧NCU不支持Blackwell的失败完整保留。阶段A复用了已有CUDA13.1 / NCU2025.4工具镜像，阶段B将复用固定upstream对应的未修改baseline native和持久编译缓存。

`PLAN.md`与`draft_protocol.json`保存最初的review方案；执行时以结果目录中的冻结protocol、source snapshot和dispatch为准。

阶段B在H100上完成40个artifact与120项完整输出正确性检查，以及7次独立worker调用的22,400个单kernel CUDA-event样本。最终40次kernel编译查询全部命中持久缓存，native和ccache统计未变。8组默认Triton↔相同布局Gluon对照另有96份NCU硬件/软件报告；计数与profiler耗时不进入正式计时。

[阶段B性能结果](../results/phase11/stage_b/summary.md)、[结果解释与下一步建议](../results/phase11/stage_b/interpretation.md)和[完整验证记录](../results/phase11/stage_b/validation.json)给出全部条件、区间与限制。32×64/8-warps的BF16中vec1/vec2明显快于vec8，但两者尚未分出唯一最优；原始32×128/4-warps的H100差异较小，四个候选都还在可能最优集合中。

执行协议修订按目录编号追加保存，`protocol.load_b()`选择编号最大的冻结修订。最终执行的是revision5。早期runner调用、NaN oracle和Gluon JIT审计hook的错误完整保留；NaN oracle按固定upstream的`PropagateNan.NONE`/`arith.maxnumf`修正。kernel、布局矩阵和统计方案未因性能结果调整，正式计时没有按结果重跑。

阶段A离线验证：

```sh
.venv/bin/python -m experiments.tma_reduction_layout.phase11.validate_a --validate
.venv/bin/python -m experiments.tma_reduction_layout.phase11.artifact_audit --validate
.venv/bin/python -m experiments.tma_reduction_layout.phase11.analyze_b --validate
.venv/bin/python -m experiments.tma_reduction_layout.phase11.validate_b
```

阶段A/B validator增加`--committed`可核对提交内的原始字节。旧Phase1–10的16,106个tracked实验文件受prior manifest保护。
