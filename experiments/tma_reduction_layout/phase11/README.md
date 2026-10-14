# Phase 11 — Explicit Gluon layout experiments

本轮已授权先完成阶段A、B，每阶段单独commit；阶段B后停下来汇报。阶段C、D暂不执行，PR仍需用户批准。

阶段A已完成。三种架构上的普通shared load无冲突/32路冲突对照，以及ldmatrix.x4对照均通过输出正确性和独立NCU report-import核对。三种架构上还分别核对两个归档kernel的真实动态shared指令与wavefront计数。工具校准没有正式性能样本，没有重建Triton native。

直接shared Requests硬件指标不可用，保留为缺失。每PC动态warp指令由software/source报告测得，wavefront与bank-conflict由单独硬件报告测得。matrix的有冲突bank计数正对照尚未测试；无冲突matrix结果不能代替它。

[阶段A结果](../results/phase11/stage_a/summary.md)和[完整分析](../results/phase11/stage_a/analysis.json)保留三架构校准和归档kernel的计数。早期PTX探针错误及旧NCU不支持Blackwell的失败完整保留。阶段A复用了已有CUDA13.1 / NCU2025.4工具镜像，阶段B将复用固定upstream对应的未修改baseline native和持久编译缓存。

`PLAN.md`与`draft_protocol.json`保存最初的review方案；执行时以结果目录中的冻结protocol、source snapshot和dispatch为准。

阶段A离线验证：

```sh
.venv/bin/python -m experiments.tma_reduction_layout.phase11.validate_a --validate
```

提交后增加`--committed`核对提交内的原始字节。旧Phase1–10的16,106个tracked实验文件受prior manifest保护。
