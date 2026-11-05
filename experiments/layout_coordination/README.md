# Layout coordination

研究对象：Triton SSA 数据流中，输入访存、数据重排、计算/广播、输出访存与资源之间的布局协调。

- [重新规划](REPLAN.md)：候选生成、选择信息、目标 lowering 与验证步骤。
- [源码及证据核查](reference_audit.json)：固定版本、文件指纹和 AWQ 信息边界实例。
- [分支更名记录](branch_rename.json)：旧分支与新分支的对应关系。
- [已完成的非规约实验](../tma_reduction_layout/results/phase11/nonreduction/SUMMARY.md)：N1–N5 的全部结果入口。

本目录是新的规划入口。旧实验目录、原始样本、compiler artifacts、协议和历史名称保留原样。
本次仅完成分支更名、只读源码审计和方案重写；没有运行新 GPU 实验、修改 C++ 或提交 PR。
