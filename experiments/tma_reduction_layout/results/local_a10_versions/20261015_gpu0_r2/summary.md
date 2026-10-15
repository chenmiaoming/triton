# A10 普通 Triton 默认布局的版本对比

仅物理卡 0，输入 BF16 `[1,8192,8192]`，每个 tile 的 M=32，沿 axis=1 取 max，输出 FP32。
同一普通 `tl.load` kernel，AST 与上一轮 canonical anchor 相同；无 Gluon、手动布局或 descriptor。
3.5.1 / 3.6.0 官方二进制 wheel 与已安装 3.7.1 分别在独立进程编译，
使用同一 Python/Torch 和同一个 ptxas；没有重建 Triton。实际版本及 SHA 见 toolchains.json。

## 编译器实际选择

去掉大小为 1 的 batch 维，下面数组的次序均为 `[M,N]`；order 为 `[N,M]`。
同时读取最终 TTGIR 中 global-load 和 reduction 输入的 encoding，不由 num_warps 或性能倒推。

| tile / warps | Triton | load SPT / TPW / WPC | reduction SPT / TPW / WPC | 每列贡献线程 / warp |
|---|---|---|---|---|
| 32×128 / 4 | 3.5.1 | [1, 8] / [2, 16] / [4, 1] | [1, 8] / [2, 16] / [4, 1] | 8 / 4 |
| 32×128 / 4 | 3.6.0 | [1, 8] / [2, 16] / [4, 1] | [1, 8] / [2, 16] / [4, 1] | 8 / 4 |
| 32×128 / 4 | 3.7.1 | [1, 8] / [2, 16] / [4, 1] | [1, 8] / [2, 16] / [4, 1] | 8 / 4 |
| 32×64 / 8 | 3.5.1 | [1, 8] / [4, 8] / [8, 1] | [1, 8] / [4, 8] / [8, 1] | 32 / 8 |
| 32×64 / 8 | 3.6.0 | [1, 8] / [4, 8] / [8, 1] | [1, 8] / [4, 8] / [8, 1] | 32 / 8 |
| 32×64 / 8 | 3.7.1 | [1, 8] / [4, 8] / [8, 1] | [1, 8] / [4, 8] / [8, 1] | 32 / 8 |

这些数字描述元素归属和规约通信需求；sizePerThread 的 N 分量不等于硬件访存次数。
完整 TTGIR、LLVM IR、PTX、CUBIN、SASS 与资源信息均保留，SASS 数量仅是静态指令位置。

## 完整 kernel 时间

冻结 CUBIN 后，统一 driver-only harness 配对计时：7 个顺序进程、每项 60 个单 kernel event 样本，
共 2,520 个样本。每次只执行一次 kernel，不用 graph 或批次时间除以次数。
下表为 7 个进程中位数的中位数，单位微秒。

| tile / warps | 3.5.1 | 3.6.0 | 3.7.1 |
|---|---:|---:|---:|
| 32×128 / 4 | 298.032 | 298.224 | 298.832 |
| 32×64 / 8 | 436.224 | 436.208 | 436.976 |

比值以同一进程内的中位数配对，取 log-ratio 的均值和 95% t 区间（df=6）。
3% 实用回归界限在编译和 timing 前冻结；不是跨设备置信区间，也不证明所有 shape 无回归。

| tile / warps | 比较 | 时间比 | 95% CI | 判定 |
|---|---|---:|---|---|
| M32_N128_w4 | 3.6.0 / 3.5.1 | 1.00012 | [0.99882, 1.00141] | excludes_3_percent_slowdown_for_this_workload |
| M32_N128_w4 | 3.7.1 / 3.5.1 | 1.00087 | [0.99945, 1.00230] | excludes_3_percent_slowdown_for_this_workload |
| M32_N64_w8 | 3.6.0 / 3.5.1 | 0.99989 | [0.99973, 1.00006] | excludes_3_percent_slowdown_for_this_workload |
| M32_N64_w8 | 3.7.1 / 3.5.1 | 1.00121 | [0.99951, 1.00292] | excludes_3_percent_slowdown_for_this_workload |

## 边界与证据

12 项编译阶段完整输出检查和 42 项冻结 CUBIN 的完整输出检查通过，覆盖两个有限输入模式。
每次 launch 前核对 CUBIN SHA，所有 timing 进程不导入 Triton、不调用 JIT。
原始样本、源代码快照、编译产物均 write-once；raw_manifest.json 固定原始文件哈希。
此轮直接回答普通 load 的默认布局及版本变化，不比较手工 vec2/vec8 优劣。
A10 没有运行 TMA；此结果不能代替原 issue 的 TMA 路径或其他 GPU 的版本验证。
Phase 11 C/model 仍未启动，未创建 PR。
