# 源码复现与 ABI 重放的边界

本批源码来自两个固定公开 commit，提取的 Triton 函数体保持不变，外围 serving
库的 import/dispatch 环境被最小依赖模块替代。完整包未安装，不能称为 vLLM
或 SGLang 服务端端到端结果。

GPU JIT 绑定使用 batch=4、length=129 的实际 CUDA tensor，捕获 dtype、指针
对齐与整数特化，并将导出的同一 ABI 在各工作负载上重放。标量 stride 相同，
输入地址保持对齐，运行时形状与序列参数合法；所有 partial 与最终输出均另行
检查。每个 workload 没有重新走一次 serving 库的自动 JIT/cache 分派。

特别是 C06 的 num_seqs 在绑定时为普通动态 i32；batch=1 的实际 Python JIT
调用通常还可能将值 1 特化，batch=64 也可能获得更强整数整除属性。因此本批
小 batch 时间属于固定 ABI 下的源码级对照，不能精确代替这些调用各自的默认
二进制延迟。该限制同时作用于 U/G0/GL，不改变源程序正确性，但会影响可外推
的绝对时间与百分比。C02 的 batch/context 从指针数据读取，stride 在本批固定。

G0 的 Q/K/V 和输出编码与 U 均已逐项核对；静态 SASS 与寄存器并非全部相同。
所以 G0 是 source/encoding anchor，尚未证明完全的 lowering bridge。正式报告
必须同时列出 U↔G0 与 G0↔GL；相对 U 的改进不能全部归为单一布局干预。

所有时间均为 warm CUDA Graph 单次 kernel 或两个 kernel 的设备时间，不含
Python 调用、调度、Q/K/V 准备、前置 attention stage 或整个 LLM 的其他层。
缓存内小图与大于 L2 的工作负载分别报告，不外推为整个模型的加速比。
