# 跨算子 Gluon 对照实验：准备阶段

日期：2026-10-21。父提交：`48acaa5dee5ba653ed47bb7136043aa03ea16031`。

八类候选已全部纳入，并完成成对语义程序、合法布局枚举、独立 CPU 参考与
sm90 目标的 CPU 编译检查。正式编译 **78/78 通过**，对应 **69 个不同编译键**；
最终 validator 通过。没有分配 GPU，没有 GPU 正确性、计时或 NCU 结果。

本轮回答“这些语义程序及布局干预是否可以编译、证据是否可复核”，尚不能
回答“哪个生产算子的默认布局有性能问题”。真实库入口适配及完整 Gluon
桥接仍未完成，不将这些探针冒充原库二进制。

## 1. 候选与可执行准备

候选来源与具体分支限制见 [CANDIDATES.md](../../../../phase11/operator_survey/CANDIDATES.md)，
对照顺序与 GPU 停点见 [PLAN.md](../../../../phase11/operator_survey/PLAN.md)。
下表是编译探针形状，不是声称从模型运行轨迹采集到的形状。

| ID | 候选 | 当前语义程序 | 逻辑 M×N / tile / warps | CPU 程序数 |
|---|---|---|---|---:|
| C01 | LayerNorm / RMSNorm 参数梯度归并 | 两个部分梯度矩阵沿 M 求和 | 128×4096 / 32×128 / 4 | 9 |
| C02 | decode attention P·V | FP32 概率加权 BF16 V，沿 token 求和 | 64×128 / 32×128 / 4 | 9 |
| C03 | columnwise FP8/MXFP8 量化 | 动态列 amax、FP32 scale、E4M3FN、列主序写出 | 32×128 / 32×128 / 4 | 9 |
| C04 | 融合 RMSNorm backward | 双方向规约、dX 与分块 dγ | 19×128 / 16×128 / 4 | 9 |
| C05 | linear/MLP bias gradient | 沿 token 的 BF16→FP32 求和 | 257×1024 / 32×128 / 4 | 9 |
| C06 | 分段 attention 并行合并 | 按 maxima/sums 重新缩放，沿 segment 合并 | 13×128 / 16×128 / 4 | 9 |
| C07 | Gated DeltaNet 递归状态更新 | V-first 状态、四步更新、全部输出与最终状态 | 128×128 / 8×128 / 1 | 6 |
| C08 | embedding token pooling | masked mean 与 max 两个配置 | 193×768 / 32×128 / 4 | 18 |

九个配置使用六对 Triton/Gluon 函数。八个多 warp 配置各包含 canonical
Triton、一个解析得到的 Gluon 输入布局锚点、七个去重后的布局；C07 包含
canonical、输入锚点、四个单 warp 布局。每个锚点与一个枚举布局重复编译，
共九个相同编译键别名，因此 78 次检查对应 69 个不同编译键。

布局覆盖合法的 `vec1/2/4/8` 与 `N-first/within` 分布，并逐一检查元素所有权
完整且不重复。C07 保留单 warp；其两个 placement 标签实际相同时合并为别名，
没有人为增加跨 warp 通信。原五类没有删减，新增的三个推理候选为 C06–C08。

## 2. 对照的含义与当前限制

正式比较需要依次建立：原库真实 dispatch 的 Triton 程序 U、与 U 在完整图
布局及精度上对齐的 Gluon 程序 G0、再对 G0 改变布局的 GL。本轮提供的是
canonical 语义程序 F，用于准备与后续子图分析；`production_adapter_verified`
和 `whole_graph_bridge_verified` 均保持 `false`。

CPU 编译解析到的九个 canonical 输入均为 `sizePerThread=[1,1]`、
`threadsPerWarp=[1,32]`；八个多 warp 配置的 `warpsPerCTA=[1,4]`，C07 为
`[1,1]`。这是本次 `ASTSource` 固定编译输入下的结果：该入口**没有注入
运行时 JIT 的指针对齐/特化属性**。不能据此推断生产调用默认选 vec1，
也不能与此前普通 load 的 vec8 实测混为一谈。正式 GPU 桥接需要记录真实
调用的 specialization、stride、对齐属性与完整中间表示。

另外保留以下语义限制：

- C02 只覆盖 P·V 子图，不含 Q·K、在线 softmax、分页和完整 attention。
- C03 当前是 FP32 scale 的动态 E4M3FN，**不是 E8M0 MXFP8 recipe**；
  原 TorchAO MXFP8 桥接待做，不能将该 fixture 的结果归给原 recipe。
- C04 尚未包含整个 backward 的全部 launch；C05/C08 仍需确认实际执行后端。
- C07 使用预处理后的 q/k/gate/beta，保留实际 V-first 和一个 warp；
  生产 layer 的归一化/激活与接口成本尚未进入比较。
- C08 的全 mask mean=0、max=-inf 是当前探针的显式语义，真实库若采用
  不同 sentinel 或 mask 策略，适配时必须忠实保留。

## 3. 编译身份、缓存与证据

正式 CPU 检查运行于 `chenmiaoming` 的 CPU 函数，目标为 `cuda:90`。
Modal app：[ap-5MewEYKkTkAV26M0F0TyIJ](https://modal.com/apps/chenmiaoming/main/ap-5MewEYKkTkAV26M0F0TyIJ)，
function call：`fc-01M5D5FZQ42RBKBY5JR5D0GVGX`。入口没有 `gpu=` 参数，
设置 `CUDA_VISIBLE_DEVICES=''`，没有启动 CUDA context 或 kernel。

| 项目 | 记录 |
|---|---|
| 复用的 Phase 11 编译器源码身份 | `20d9d98fead78c2f65e1b3ba5a7881d86a836dad` |
| Python Triton version | `3.9.0` |
| native SHA256，前后相同 | `fc5f2ad73f9f8c8ab31c3a01b8124329ccec391cdc5dbf2dafb2d42778cf6adb` |
| ptxas | CUDA 13.4 / V13.4.59 |
| ptxas SHA256 | `9c2c084df7eb9be48f0502b6938ce2571679b2a78cad1c564b583bd7d9ea3c50` |
| kernel cache | `/cache/phase11-upstream20d9-kernel-cache` |
| cache CUBIN 数 | 613 → 682，即新增 69 个 |
| native rebuild | 0；native SHA 与 ccache 全部统计前后一致 |

不因 Python/Gluon 改动重建 Triton。新的 kernel 需要正常编译，既有 native、
工具链和缓存继续复用；九个重复锚点复用了相同编译键。这不等于声称本轮
所有 kernel 都是缓存命中。

冻结协议及源文件快照位于 [preregistration](preregistration/protocol.json)。
正式导出目录为 [pinned_cpu_attempt_001](pinned_cpu_attempt_001/result.json)，包含
每个程序的 IR、PTX、CUBIN、SASS、cuobjdump 静态资源信息和 SHA。
这些都是静态编译产物，不提供动态指令数、bank conflict 或实际并发计数。

本地 Triton 3.7.1 的开发尝试另外保存在 [development/manifest.json](development/manifest.json)：
第一次 canonical 9 项成功，但 Gluon 69 项因 `GluonASTSource` 的导出位置变化
失败；加入版本 API 兼容后第二次 78/78 成功。两次原始输出分别按字节打包
保存，不覆盖失败证据，不与正式固定编译器结果混算。

## 4. 校验结果

[validation.json](validation.json) 记录最终 `PASS`：

- 六对函数的 AST 表达式、cast 与控制流检查一致；该检查不证明 GPU lowering
  或数值等价，也不将布局语法差异当作算法差异。
- 78/78 个 CPU 编译程序成功；九个 canonical 输入布局均解析成功。
  全部合法布局的元素所有权检查和编译键别名检查通过。
- 源码快照与协议、IR/PTX/CUBIN/SASS 哈希检查通过。
- 27 项 CPU 参考检查通过，覆盖九个配置及三类输入；RMS 由 CPU autograd
  复核，递归状态由另一种 CPU 张量实现复核。
- 509 个 FP8 codebook 检查通过；CPU Torch 未初始化 CUDA。

CPU 参考检查用于验证将来的正确性判据，**没有执行 GPU kernel**。完整输出、
mask/canary、FP8 数据与 scale、每步递归状态的 GPU 正确性仍是下一阶段门槛。

复核命令（只读，不生成 GPU 工作）：

```bash
.venv/bin/python -B -m experiments.tma_reduction_layout.phase11.operator_survey.validate \
  --attempt experiments/tma_reduction_layout/results/phase11/operator_survey_preparation/20261021/pinned_cpu_attempt_001
```

## 5. 下一批实验与停点

优先绑定 C06 的真实并行 segment merge 与 C02 的非 Tensor Core P·V 入口，
固定库 commit、dispatch、实际 workload 形状和精度，然后完成 U→G0 桥接与
GPU 正确性。之后才计时 GL；完整 attention 与子 kernel 分开汇报。C08 先
确认实际编译后端，C07 保留单 warp 的机制边界，其余候选继续按清单分批推进。

生产尺度的单次算子时间是主指标，不人为扩大 grid 来代表小 batch 推理。
保留默认胜出、持平与退化结果；动态硬件成本后续由相应 worker 的 NCU
计数核对，不用 `32/vec` 或静态指令条数替代 transaction。

本轮停在 CPU 准备与归档。未运行新 GPU 实验，未修改旧模型、旧 raw samples
或原始 compiler artifacts，未开始新的 cost model 拟合或创建 PR。
