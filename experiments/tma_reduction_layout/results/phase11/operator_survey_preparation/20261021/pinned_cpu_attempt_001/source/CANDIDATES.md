# 跨算子布局候选清单

源码筛选日期：2026-10-20；CPU 准备更新：2026-10-21。
成对语义程序已进入 CPU 编译检查；原库程序尚未完成桥接和 GPU 实测。
准备范围及停点见 [PLAN.md](PLAN.md)。

原先讨论的五类全部保留为候选；本轮另外加入三个推理候选。
这里记录的是测试入口和可检验假说，不是性能问题清单，也不是已冻结的
GPU 实验协议。历史阶段、模型和原始证据保持不动。

| ID | 候选 | 用途 | 源码核对后的定位 |
|---|---|---|---|
| C01 | LayerNorm / RMSNorm 参数梯度归并 | 训练 | 连续特征维加载，沿部分结果维归约；RMSNorm 最终归并需核对后端 |
| C02 | 非 Tensor Core decode attention 的 P·V | 推理 | 概率广播、沿 token 归约、在线状态累加 |
| C03 | Columnwise FP8/MXFP8 量化 | 取决于量化 recipe | 列 amax 后广播量化及二维写出；不假定所有推理都走 columnwise |
| C04 | 融合 RMSNorm backward | 训练 | 两个方向的规约，加二维 dX 与一维参数梯度输出 |
| C05 | Linear / MLP bias gradient | 训练 | 仅限有 bias 且执行后端可确认的路径 |
| C06 | 分段 attention 的并行结果合并 | 推理 | 已找到当前 vLLM 的二维普通 load + axis=0 sum |
| C07 | Gated DeltaNet 递归状态更新 | 推理 | 状态读改写与两次规约；实际 layer 已采用 V-first、单 warp，作为相关机制候选 |
| C08 | Sentence embedding 的 token pooling | 推理 | 沿 token 归约、保留连续 hidden 维；现有 PyTorch 路径需确认编译后端 |

## 原五类的测试入口

### C01：归一化参数梯度归并

部分梯度 P[P,H] 沿 P 求和，输出 [H]。Triton 教程的
`_layer_norm_bwd_dwdb` 是普通 `tl.load` 加二维 axis=0 规约的直接入口。
先保持实际分组数和算法，比较不同 register layout，再评估完整 backward。
不能把部分结果缓冲区放大成无关的大矩阵来代表模型收益；sum 的累加精度
也不能套用 BF16 packed-max 的解释。

来源：[Triton LayerNorm](https://github.com/triton-lang/triton/blob/main/python/tutorials/05-layer-norm.py)。
Liger RMSNorm 的最终 `_dW.sum(dim=0)` 属于 PyTorch 调用，执行后端需要另行确认。

### C02：decode attention 的 P·V

对 V[token,head_dim] 计算加权 token 求和。SGLang 的普通 decode 路径同时包含
Q·K 的特征维求和与 P·V 的 token 维求和。保留在线 softmax、分页 stride、
split 数、数据类型及精度语义，只改变布局。P·V 子图与完整 decode 分别计分。
使用 tl.dot 的 GQA 路径另列，不能据此声称所有 attention 都存在相同问题。

来源：[SGLang decode_attention.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/attention/decode_attention.py)。

### C03：columnwise 量化

每个 32×1 组的 amax 生成尺度，再广播给组内全部元素。TorchAO 的
`to_mxfp8_dim1_kernel` 包含普通加载、转置/reshape、尺度计算和列主序写出。
测试完整量化程序，rowwise 路径作为方向对照。先确认模型 recipe 是否实际
需要 columnwise 数据；量化准备 kernel 与原生 MXFP8 GEMM 的设备支持分别记录。

来源：[TorchAO MX kernels](https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/kernels.py)，
[MXFP8 方向与分组](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html)。

### C04：融合 RMSNorm backward

同一程序沿 hidden 计算 dX 所需统计量，沿 token 计算 dγ。Liger 的 block
实现存在这两个方向；当前调用条件为特征块不超过 256、行数至少 32768、
且未强制 row mode。较大 hidden 的另一条路径不能混入这条程序的统计。

来源：[Liger RMSNorm](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py)。

### C05：bias gradient

db[h] = sum_t dY[t,h]，可单独执行或与激活反向融合。先选择真实含 bias 的
模型路径。Liger fused linear cross-entropy 的 bias 累加使用 PyTorch sum；
若生成物不是 Triton，就保留为生产实现基线，不能归因于 Triton 布局。

来源：[Liger fused linear cross entropy](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)。

## 本轮补充的推理候选

### C06：分段 attention 的并行合并，优先级高

vLLM `triton_unified_attention.py::reduce_segments` 的输入为
`partial[token,head,segment,head_dim]`。它读取二维 segment×head_dim tile，
按每段 softmax 统计量缩放，再沿 segment 求和、归一化并写出向量。
其自身是普通 pointer load 路径，不依赖前一阶段是否使用 descriptor。

假说：加载连续 head_dim 的布局可能增加 segment 方向规约通信；同时
segment 统计量的布局和广播也可能影响代价。应保持段数、部分输出格式与
softmax 合并语义不变，比较原始 Triton 与 Gluon 布局，并分别测合并阶段和
完整分段 attention。短 query 的实际 3D 分支与 padded/空段必须覆盖。

入口：[当前 vLLM reduce_segments](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_unified_attention.py#L665)。
该文件 `unified_attention` 仅在 `use_3d` 分支调用它，不能当作所有请求的必经阶段。

另有固定版本的直接参照：[xFormers v0.0.29 _splitK_reduce](https://github.com/facebookresearch/xformers/blob/v0.0.29/xformers/ops/fmha/_triton/splitk_kernels.py#L730)。
这是历史版本，不标为当前 xFormers 默认。它不是额外一个算子候选。

特别区分：[vLLM triton_decode_attention.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/triton_decode_attention.py)
中的另一个 stage2 逐 split 在线累加，不能混称为同一并行 kernel。

### C07：Gated DeltaNet 的状态更新，相关机制候选

每个 decode step 读取状态 S[K,V]，先计算沿 K 的预测值，利用该向量更新
二维状态，再沿 K 计算输出。需要同时检查输出和更新后的状态，多步误差
也属于正确性检查范围。

源码提供 K-first 与 V-first 两种外部状态格式。但实际 FLA layer 的无梯度、
eval、短 query 分支显式采用 V-first，kernel launcher 使用一个 warp。
因此不把 K-first 的潜在冲突冒充当前 layer 默认缺陷，也不宣称它已有跨 warp
同步开销。保持实际 V-first 和原始 tile/warp 数，先查 warp 内通信及资源；
另一种状态格式只作配套格式/算法对照，计入转换和缓存接口成本。

来源：[FLA recurrent kernel](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/fused_recurrent.py)，
[实际 layer 分支](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/gated_deltanet.py)。

### C08：embedding 的 mean / weighted-mean / max pooling

输入 E[B,S,H] 沿 S 汇聚，保留 H；实际应用是配置了该 pooling 的 embedding
推理，不是所有生成式模型都会运行。Sentence Transformers 的固定版本中
已有这些语义。mean 的 mask/权重乘法也必须保留。

与原问题一样，行主序加载沿 H 连续、规约沿 S。先确认 torch.compile 是否
为该图生成 Triton，再比较同图、同分段策略的 Gluon 布局。原生 eager CUDA
实现是另一条基线；从 index_add 改成树形 reduction 的收益必须单列算法变化，
不能全记在 layout 上。必须保留 padding、变长、空/全 mask 的原始语义。

来源：[Sentence Transformers v3.4.1 Pooling](https://github.com/UKPLab/sentence-transformers/blob/v3.4.1/sentence_transformers/models/Pooling.py)。
当前 [vLLM MeanPool](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/pooler/seqwise/methods.py)
使用分块 index_add_，不是已经确认的 Triton 规约实现。

## 暂不计入同类问题的推理对照

- MoE expert combine：语义为 expert 维加权求和，但核对到的 TE Triton
  `_unpermute_kernel` 逐 expert 累加一个特征向量；vLLM 的一个接口还调用
  `_moe_C`。保留为可选对照，不把数学式中的 sum 当成跨线程规约。
  [TE kernel](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/triton/permutation.py#L297)，
  [vLLM wrapper](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/moe_permute_unpermute.py#L310)。
- Mamba selective state update：有二维状态读改写与规约，但标准连续状态下
  dstate 是连续且被规约的维度；作为方向一致的相关对照，不优先声称与原
  列规约相同。[源码](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/selective_state_update.py)。

## 后续测试边界

1. 原五类全部进入候选池。推理优先顺序为 C06、C02，再 C08、C07；C03 依
   实际 recipe 决定用途，不为平衡数量而标成常见推理路径。
2. 正式采集前固定库 commit、原始程序、dispatch 条件、精度、shape/stride、
   compiler/native/ptxas 身份和测量协议。本文 main 链接会移动，不能代替
   即将冻结的源码快照。先在 CPU 核对固定编译器的兼容性，不静默修改库算法。
3. 先比较原始 Triton 默认与同算法、同 tile、同 warp 数的合法 Gluon 布局。
   tile、warp 数、state 格式、分段数或融合边界的变化单列，不归入纯布局收益。
   保留默认胜出、差异不显著、编译后布局相同的结果。
4. 初筛使用真实 workload 尺度。小 batch decode、缓存内归并、带宽受限与
   launch 主导情况分别报告，不机械套用历史 128/256/384 MiB 服务斜率。
5. 完整输出、mask/canary 与状态正确性先通过，再按配对顺序计时。状态程序
   的初始状态和真实多步 decode 条件分别固定，不能让候选反复更新不同状态。
6. 对有稳定差异的代表点采集动态 NCU 指令、shared wavefront/conflict、
   global request sectors、寄存器/spill 和实际并发信息；PTX/SASS 用于定位，
   不把静态条数直接当硬件成本。分别报告子 kernel 和完整算子的效果。
7. 后续复用现有工具链及缓存，不因 Python/Gluon 改动重建 Triton；本轮未
   启动 GPU、未改变旧模型、未创建 PR，也未改写旧阶段的 raw/compiler 文件。
