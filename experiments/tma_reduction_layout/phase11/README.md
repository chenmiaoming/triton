# Phase 11 — Explicit Gluon layout experiments

阶段A、B均已完成，每阶段单独commit。工作停在阶段B，等待讨论是否调整后续方案；阶段C、D未执行，PR仍需用户批准。

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
