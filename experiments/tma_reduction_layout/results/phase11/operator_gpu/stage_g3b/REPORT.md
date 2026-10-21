# G3b：真实 LayerNorm 训练入口的布局对照

固定 Triton 官方教程的原始两段 backward 函数体，研究参数梯度归并的默认布局优化空间。这里没有版本对照，不称为性能回归。
H100 上，三组真实归并尺度的较好候选为 vec2/vec4；归并单 kernel 相对真实 Triton U 的本批最小均值改善约 16%–28%，部分收益延伸至完整 backward 设备流程。不能由此声称所有训练算子或完整训练 step 都有同样收益。

## 1. 实际入口与受控条件

- [固定源码](https://github.com/triton-lang/triton/blob/20d9d98fead78c2f65e1b3ba5a7881d86a836dad/python/tutorials/05-layer-norm.py)：`_layer_norm_bwd_dwdb` 与 `_layer_norm_bwd_dx_fused`，函数 AST 与固定版本完全相同。没有改写原始算法、归并策略或 C++。
- 只在 Gluon 中改变归并寄存器布局；BM=32、BN=128、4 warps 固定。主对照固定最终输出 encoding；另外预注册 vec1/vec8 的 native-output 诊断。
- N=1024/4096/8192 时，教程实际 GROUP_SIZE_M=256/128/96。没有把部分结果矩阵扩大成 8192×8192。输入与 partial、最终梯度为 BF16，累加为 FP32，与教程 wrapper 一致。
- U 为原始 Triton；G0 为默认输入/输出 encoding 下的 Gluon 锚点。U/G0 的 SASS 不是完全相同（544 vs 592 个静态 site），因此同时列 vs U 和纯 G0→GL，不能混为一种收益。
- 完整 backward 设备流程包含 Lock、DW、DB 清零，原始 dx_fused 与候选归并；均值/rstd 作为已有的 backward 输入。包括两 kernel 与三个 memset，不包括前向、框架调度、优化器或整个训练 step。
- pipeline M=128/2048；真实 effective P=min(GROUP_SIZE_M,M)。N1024/M128 的 P=128，和 standalone P256 不同，因此二者不能简单用一个 Amdahl 比例相加外推。

## 2. 编译、正确性、计时与 NCU

- 36 个逻辑程序、13 个不同 CUBIN；36/36 与真实 CUDA Tensor JIT 生成的 CUBIN SHA 完全匹配。native SHA 保持 `fc5f2ad73f9f8c8ab31c3a01b8124329ccec391cdc5dbf2dafb2d42778cf6adb`，CPU ccache 前后统计相同，native rebuild=0。
- 270 个去重后的实际执行检查全部通过，展开逻辑别名为 297 个条件。含三类输入、完整输出、输入不变、两侧 canary；独立 FP64 merge 参考及 dX 公式。
- pipeline 最终梯度对原始 stage1 实际写出的 BF16 partial 做独立 CPU 求和校验；这保留原教程 partial 舍入语义，不能冒充对无中间 BF16 舍入的理想梯度逐位等价。
- 3 个独立 worker 进程 × 5 visits × 31 samples；共 41,850 原始样本、99 个逻辑计时条件。固定顺序随机种子，8 次预热；无按胜负重测。
- 全部归并候选均有单 CTA 动态指令及真实 grid 硬件报告，共 60 份，均本地 CPU 重新导入并比对。最初预注册 24 份；看到计时后透明地扩展到全部剩余候选，另冻结 extension protocol。新增 36 份属于探索性机制诊断，没有新增/挑选正式时间。
- 第一轮 CPU tools-image 缺 Triton 的失败保留在 attempt_001；v2 在任何训练 GPU 运行前切换到已有 compiler image，attempt_002 成功。

## 3. 全部九个工作负载的本批最小均值

最小值是全候选事后观测，不是预冻结选择器结果。部分 vec2/vec4 相互差异小于 3%，不声称最小点与次优点已被稳定区分。完整 99 行见 [MATRIX.md](MATRIX.md)。

| N / nominal P | mode / full rows M | U µs | G0 µs | 本批最小均值候选 | µs | gain vs U [CI95] | gain vs G0 [CI95] |
|---|---|---:|---:|---|---:|---|---|
| 4096 / 128 | merge / n/a | 7.9765 | 8.0469 | vec4:within | 6.5045 | +18.45% [+17.91, +18.98] | +19.17% [+18.60, +19.85] |
| 4096 / 128 | pipeline / 128 | 19.0933 | 19.1296 | vec4:within | 17.0112 | +10.91% [+10.12, +11.70] | +11.07% [+10.46, +11.68] |
| 4096 / 128 | pipeline / 2048 | 53.9157 | 54.0053 | vec4:n_first | 51.5563 | +4.38% [+4.11, +4.61] | +4.53% [+4.22, +4.81] |
| 1024 / 256 | merge / n/a | 10.6261 | 10.6965 | vec4:n_first | 7.6907 | +27.62% [+27.09, +28.28] | +28.10% [+27.62, +28.64] |
| 1024 / 256 | pipeline / 128 | 17.5296 | 17.4891 | vec4:n_first | 15.5371 | +11.37% [+10.83, +11.89] | +11.16% [+10.70, +11.62] |
| 1024 / 256 | pipeline / 2048 | 35.0357 | 34.7776 | vec4:n_first | 31.0677 | +11.33% [+10.63, +12.51] | +10.67% [+10.08, +11.17] |
| 8192 / 96 | merge / n/a | 7.2768 | 7.3451 | vec2:within | 6.1077 | +16.07% [+15.07, +17.19] | +16.85% [+16.22, +17.52] |
| 8192 / 96 | pipeline / 128 | 21.8325 | 21.8581 | vec4:n_first | 20.4011 | +6.56% [+5.91, +7.27] | +6.67% [+6.36, +6.96] |
| 8192 / 96 | pipeline / 2048 | 79.9680 | 80.0405 | vec4:within | 78.2635 | +2.13% [+1.90, +2.37] | +2.22% [+2.05, +2.50] |

## 4. 硬件证据支持的解释与反例

- 以 N1024/P256 为例，G0→vec4:N-first 的 SHFL 192→64、BAR.SYNC 100→68、shared-load WF 156→92；global sectors 都是 4096/CTA，requests 则 256→512。这与减少部分通信工作、同时保留一定加载粒度的折中一致。
- vec1:N-first 虽达到 SHFL=0、BAR=0、WF=0，但 global-load requests 是 2048/CTA，比默认的 256 多 8 倍，sector 仍是 4096。其归并收益仅 +2.69% vs G0，而 vec4:N-first 为 +28.10%。单纯最小化 fanout 不能得到本批最好的布局。
- 更少同步也不是充分条件：N1024 的 vec4:within 只有 BAR=16、WF=28，比 vec4:N-first 的 68/92 更少，但耗时 7.8592 vs 7.6907 µs；该差异不足 3%，只能记录方向，不能宣称已经分辨其唯一原因。
- vec8 native-output 比固定输出减少 BAR 100→60、WF 156→128，但三个归并尺度的点收益只有约 1.1%–2.4% vs G0，不能把本批主要收益全部归于最终输出转换。
- 归并 grid 只有 8/32/64 CTA，均小于 H100 的 132 SM。虽然寄存器使 q_cap 从 5 降到 4，实际 active-cycle occupancy 仍约 6.25%；不能把 q_cap 降档直接当作这里变慢的原因。所有归并候选 local bytes=0，本批无 spill 证据。
- 新计数已描述真实工作与调度现象，但尚未将加载吞吐、依赖长度及每种通信操作各自分配独立时间权重。最优集合会随完整 pipeline 的 M 改变；不恢复旧的无权指令计分。

## 5. 边界与下一步

这是固定编译器版本和合成但真实 dispatch 尺度上的源码级对照。证明了至少一个真实训练入口存在有用的布局改选空间，不证明所有历史版本一直慢、所有 LayerNorm 实现都慢、或某个新选择模型已经有效。
下一轮优先为 SGLang vec4 拆解依赖/转换开销，并用完整模型路径确认训练和推理收益占比；其余 C03/C04/C05/C07/C08 仍需核查实际 recipe、dispatch 和执行后端后再逐项推广。
官方解释依据：[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)、[NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)、[NVIDIA Compute Triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html)。访存合并、资源容量、实际并发、发射及 stall 比例分别报告。

最终验证：`.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g3_validate --stage all`。