# Gluon 跨算子布局实验准备

当前范围是准备与 CPU 编译门，不运行 GPU。八类候选全部保留，先做推理入口
C06（并行 segment merge）和 C02（decode P·V），再分批覆盖其余候选。
不能据本轮编译结果宣布原库存在性能缺陷，不能修改历史模型或旧证据。

## 对照层级

| 层级 | 程序 | 能回答的问题 |
|---|---|---|
| U | 冻结库源码、真实 dispatch 的原始 Triton | 实际实现的默认选择与实际耗时 |
| G0 | 与 U 在精度、算法、tile/warp、输入/规约/输出布局上对齐的 Gluon | DSL 改写是否引入了额外差异 |
| GL | G0 的合法布局变体 | 元素归属变化能否解释并改善代价 |
| F | 本轮 canonical Triton / Gluon 语义程序 | 编译可行性与独立子图探针，不能冒充 U |

本轮 F 已覆盖八类。自动解析 F 的输入 load layout 可构造 Gluon 输入锚点，
但这不等于整个图已与 U 对齐。正式实验先完成 U→G0 桥接；若未对齐，保留
结果并将比较标为联合变化，不能全部计作纯布局收益。

## 本轮可执行准备

- 九个语义配置：八类各一项，加 C08 max pooling。枚举去重后的 vec1/2/4/8
  N-first / within 布局；C07 保留实际 V-first、一个 warp，且检查多步状态。
- 用 AST 检查六对 Triton/Gluon 函数的数学表达式、cast 和控制流一致，只移除
  Gluon 的布局语法；这是源码检查，不是 GPU 正确性证明。
- CPU target=sm90 编译 canonical、输入锚点和所有布局变体，归档 IR/PTX/CUBIN
  及 assembler/native SHA。入口没有 GPU 函数，不初始化 CUDA context。
- 独立 NumPy 参考包含完整输出、padding/all-mask、部分梯度与更新状态；RMS
  用 CPU autograd 复核，GDN 用另一种 CPU 张量实现复核，FP8 用枚举 codebook
  与 CPU Torch 核对。所有 GPU 正确性门仍保持未通过状态，等待实测。
- 本地 3.7.1 只作开发检查。正式 CPU 门复用 Phase 11 镜像、20d9d98 对应的
  native SHA 与 `/cache/phase11-upstream20d9-kernel-cache`；不运行 make。

## 开始 GPU 实验后的顺序

1. **绑定真实入口**。固定库 commit、最小 source snapshot、dispatch、模型形状
   和 stride。C06 保留实际段数/softmax 统计格式；C02 的 P·V 子图与完整在线
   attention 分开；C08/C05 先确认 PyTorch 图的实际后端。
2. **完整正确性与桥接**。U、G0、GL 使用相同输入和明确的精度语义；完整输出
   与 canary 先过门，再做性能。GDN 同时检查每步输出和状态；量化检查数据与
   scale。检查 cast/packing、MR/MO、全局写出与资源，不只比较 load encoding。
3. **分批单次计时**。每批不超过 72 个唯一 binary；首先 C06/C02。按生产尺度
   的 batch/head/context 计时，原始单次算子时间为主指标。使用冻结 CUBIN、
   同 worker 配对轮换顺序、预先固定的三组进程采样；不按输赢追加重复。
   不为小 batch 伪造铺满显卡的 grid，不从 K=64 推回 K=1，不统一套用全图截距。
4. **硬件解释**。对预先定义的收益、退化、持平代表点采集 NCU。查询该 worker
   的可用 metrics，记录动态 shuffle/barrier、shared wavefront/conflict、global
   load/store sectors、DRAM、寄存器/spill 与 occupancy capacity；不把静态指令
   数或 vector width 当硬件 transaction 数，也不把容量 q 当实时驻留数。
5. **完整算子验证与阶段提交**。分别报告子 kernel 与整个 attention/backward
   或量化程序的耗时、默认胜出/持平/退化、置信区间和至少 3% 的实用差异。
   原始样本和 compiler artifacts 写入新目录，失败记录保留，validator 后按
   完成的阶段 commit。未验证的模型改选与 PR 不在本阶段范围内。

## 必须单列的语义边界

- C03 当前 F 是 FP32 scale 的动态 per-column FP8，**不是 E8M0 MXFP8**。
  其 MXFP8 原库桥接尚未实现。不能拿 F 的数值或性能冒充 TorchAO 原 recipe。
- C02 当前 F 只有加权 P·V，没有 Q·K、分页查表、softmax 和在线累加状态。
  C04 F 输出 dX 和分块 dγ，还不是整个 RMSNorm backward 的所有 launch。
- C07 F 使用预处理后的 q/k/gate/beta；真实 layer 的归一化与激活融合需在 U/G0
  中匹配。保持 V-first 格式，不用人为换成 K-first 来制造默认缺陷。
- C08 的 mean/max、空集合、变长和 mask 策略随模型而定；当前 F 的 all-mask
  mean=0、max=-inf 是其显式语义，不能覆盖具有不同 sentinel 的生产实现。
- 同算法内的布局收益、tile/warp 调参、状态格式、融合/拆分和算法重写分别记账。
  不以放行已知赢家、让复杂模型击败 fixed_vec1 或证明 Triton 普遍选错为目标。

当前停点：CPU 准备完成后汇报，GPU 运行留待下一次启动实验。
