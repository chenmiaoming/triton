# Layout coordination

研究对象：Triton SSA 数据流中，输入访存、数据重排、计算/广播、输出访存与资源之间的布局协调。

- [重新规划](REPLAN.md)：候选生成、选择信息、目标 lowering 与验证步骤。
- [源码及证据核查](reference_audit.json)：固定版本、文件指纹和 AWQ 信息边界实例。
- [分支更名记录](branch_rename.json)：旧分支与新分支的对应关系。
- [已完成的非规约实验](../tma_reduction_layout/results/phase11/nonreduction/SUMMARY.md)：N1–N5 的全部结果入口。
- [Modal H100 锁频权限实测](results/clock_control/probe_001/REPORT.md)：三台 worker 的实际拒绝结果；[匹配复测草案](clock_control/MATCHED_RETEST_PLAN.md) 尚未执行。

本目录是新的规划入口。旧实验目录、原始样本、compiler artifacts、协议和历史名称保留原样。
改名/重规划提交仅做只读源码审计；之后按用户要求新增了 GPU 锁频权限探针。探针未执行 kernel 计时；没有修改 C++ 或提交 PR。
