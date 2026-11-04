# 无规约内核候选与 Gluon 对照方案（待复核）

状态：仅完成公开源码筛选。未编译候选，未运行 GPU、计时或 NCU，尚未证明任何候选存在性能问题。此文不是已启动的实验协议，不修改既有阶段证据。

## 1. 要验证的命题

在没有轴归约的真实公开 Triton 内核中，输入访存、数据重排、广播消费者、输出访存及寄存器资源之间的布局需求，是否也会使默认布局留下可重复的优化空间？

必须区分“不同布局的时间不同”与“原始 Triton 默认布局存在优化缺口”。后者需要源图对齐的 U/G0 和同轮合法候选比较，不能通过人为构造坏布局证明。

无规约门：目标 kernel 及其可达 JIT helper、编译后目标 IR 均没有轴归约、scan、dot/矩阵收缩、atomic accumulation。逐元素 maximum/minimum 用于 clamp 不属于轴归约。所在文件或完整上游 pipeline 可以包含规约，但不得据此把整个 pipeline 称为无规约。

## 2. 已读固定版本源码的候选

### N01：torchao MX scale swizzle（首批）

- 固定源码：[kernels.py / triton_scale_swizzle](https://github.com/pytorch/ao/blob/b3fc9544fd014daf04410aed233f8640d4db482a/torchao/prototype/mx_formats/kernels.py#L317-L420)。入口 `triton_mx_block_rearrange`。
- 已有 scale 的 128×4 byte tile 普通 load，输出按 `(r % 32)*16 + (r // 32)*4 + c` 重排；有实际 row/column strides、边界 mask 和完整 padding 输出，无规约 helper。
- 可疑点：输入的连续访问与 swizzled 输出的线程分组不一致；需要观察编译器选择的寄存器分布和转换是否协调两端。
- 后端边界：[to_blocked](https://github.com/pytorch/ao/blob/b3fc9544fd014daf04410aed233f8640d4db482a/torchao/prototype/mx_formats/utils.py#L31-L70) 需 `use_triton_kernel=True`，不是该 wrapper 默认 Torch 路径。目标 kernel 没有 TMA/TMEM 指令，不能把消费该格式的后续 GEMM 纳入本实验。
- 拟议输入：逻辑 scale shape `[4096,128]`、`[11008,128]`，另设 tail/padding 与合法非连续 stride 正确性诊断。这是方案形状，不是已捕获模型 trace。
- 干预：128×4 tile 固定，枚举合法 axis order、每线程 M/N 元素数、lane/warp 分布。BN=4，不能机械沿 N 枚举 vec8/vec16。
- 输出需逐字节与独立索引置换 oracle 相同。

### N02：vLLM AWQ 解包反量化（首批）

- 固定源码：[awq_triton.py / awq_dequantize_kernel](https://github.com/vllm-project/vllm/blob/67e6ea21a7bb9dd342511c9ebe200821af41b6e8/vllm/model_executor/layers/quantization/awq_triton.py#L12)。入口 `awq_dequantize_triton`。
- packed int32 `[K,N/8]` 经三次 interleave、位移/掩码解包，配合 group scales/zeros 广播，输出 `[K,N]` FP16；该目标函数没有 helper 或规约。
- 原 wrapper 默认 packed tile 为 32×32，对应展开输出 32×256。保持 group、tile、W/stages 的原实际 JIT 取值，不用改变 tile/warp 伪装成布局收益。
- 可疑点：解包前后元素数量与相邻关系不同，scales/zeros 又跨行广播，可能在加载、解包后分布、写出及资源之间产生冲突。逻辑展开 8 倍不代表硬件加载次数增加 8 倍。
- 后端边界：[vLLM 调用入口](https://github.com/vllm-project/vllm/blob/67e6ea21a7bb9dd342511c9ebe200821af41b6e8/vllm/_custom_ops.py#L565-L579) 由 `VLLM_USE_TRITON_AWQ` 选择；不能称为默认 CUDA AWQ 后端。排除同文件 `awq_gemm_kernel`。
- 仓库测试锚点：[test_awq_triton.py](https://github.com/vllm-project/vllm/blob/67e6ea21a7bb9dd342511c9ebe200821af41b6e8/tests/kernels/quantization/test_awq_triton.py#L77-L121)。初筛拟选 packed `[3584,448]`、`[1024,128]`，固定合法 group=128；group=32 另列扩展，不能混入第一轮 layout 主效应。
- 干预：packed-input、展开值、输出的合法布局；scales/zeros 的源图布局和计算次序先保持不变。
- 独立 oracle 同时核对 nibble 顺序、zero/scale 的分组、实际运算 dtype 与最终 FP16 舍入，不能只比较整数解包阶段。

### N03：RoPE 家族（首批先普通 RoPE，MRoPE 为同族扩展）

- 固定源码：[FlashAttention rotary_kernel](https://github.com/Dao-AILab/flash-attention/blob/5f7b05a78bf05cb33fb8ed07f74d2b82c9839dcd/flash_attn/ops/triton/rotary.py#L13)。原 caller BLOCK_H=2，rotary_dim≤128 时 BLOCK_M=8，否则为 4。
- 三维 `[head,token,dimension]` 输入与二维 cos/sin 广播；interleaved 分支 `reshape→split→join` 连接配对元素；无规约或 JIT helper。conjugate backward 是同族参数，不能计为另一独立算子族。
- 可疑点：配对元素的线程归属、跨 head 的 trig 广播、连续读写和活跃寄存器需求未必偏好同一布局。源码关于 ROTARY_DIM constexpr 与 LDG 位宽的注释只是作者线索，不能当作本轮性能实测。
- 初筛固定非 varlen、out-of-place（原 API 合法模式）、interleaved=True、rotary_dim=head_dim；拟议 `(B,S,H,D)=(4,2048,32,64)`、`(4,2048,32,128)`。NeoX、varlen、backward、head/tail 后续独立扩展。
- 同族扩展：[SGLang _triton_mrope_forward_fused](https://github.com/sgl-project/sglang/blob/fd19cfb660f42bd3ceb6df7598bd343572e30d2c/python/sglang/kernels/ops/attention/rotary_triton.py#L12)，[CUDA caller](https://github.com/sgl-project/sglang/blob/fd19cfb660f42bd3ceb6df7598bd343572e30d2c/python/sglang/srt/layers/rotary_embedding/mrope.py#L255)。每 token 的 Q/K 二维矩阵共用多轴 trig 广播，不含规约。可分别干预 Q、K、协同布局。
- MRoPE G0 保留各轴 masked trig loads、相加次序、cast 和原地读写顺序；使用 CUDA caller 的 cache dtype，不混用 HIP 表达。
- 原地版本每次 warmup/计时/NCU launch 前都必须恢复输入；恢复在 timing event 外且对 U/G0/GL 一致，记录它对缓存的影响。不能把原地 kernel 改成非原地并宣称同一基线。

### N04：SGLang FP8 QKV pack（强结构候选，架构边界单列）

- 固定源码：[fused_prep.py / _pack_qkv](https://github.com/sgl-project/sglang/blob/fd19cfb660f42bd3ceb6df7598bd343572e30d2c/python/sglang/kernels/ops/attention/fp8_fa_sm120/fused_prep.py)。目标及 `_load_tile`、`_quantize`、`_divide_rn` 均无规约；`div.rn.f32` 是纯逐元素 tensor asm，无隐藏 lane/warp scalar。
- 64×128 tile，原 caller W=8。Q/K 按 token-major 输出；V 经 token gather、FP8 量化和转置输出，多个消费者需求不同。
- 必须固定外部 scales，只测 pack。完整 `fused_prepare` 有独立 amax/atomic-max pass，不能称为无规约。保留 RN division、clamp、FP8 舍入和 padding，FP8 输出逐字节检查。
- 源码属于 SM120 attention 的准备路径。先检查固定编译器/目标兼容性；H100 若能编译，只证明可移植 pack 模式，不称为 H100 原生生产调用。SM120 实测需另列硬件层，不混算。

### N05：FLA KDA gate forward（广播/计算对照）

- 固定源码：[gate.py / kda_gate_fwd_kernel](https://github.com/fla-org/flash-linear-attention/blob/9e82a80fdaefcd7c9cc3da51ea36d6a9e1587bd1/fla/ops/kda/gate.py#L117)，[layer caller](https://github.com/fla-org/flash-linear-attention/blob/9e82a80fdaefcd7c9cc3da51ea36d6a9e1587bd1/fla/layers/precond_kda.py#L318)。
- `[BT,BD]` g、按 feature 广播 dt_bias、按 head 广播 A_log，逐元素 softplus/sigmoid，默认 FP32 输出。forward 及 exp/softplus helpers 无规约；同文件 backward/chunk-cumsum 必须排除。
- 原 autotune 含 BT32/64/128 和不同 W/stages。先让 U 按原逻辑选择，冻结实际 winner/config，再以相同配置比较 G0/GL；不把手改 BT/W 收益归入布局效果。
- 相比前三项，两端 D 都连续，冲突怀疑较弱；作为“存在广播但未必需要重分布”的真实对照。冻结 FLA_USE_FAST_OPS 和实际 helper，不替换近似数学。

### N06：SGLang HND KV scatter（真实输入/输出对照）

- 固定源码：[cache_ops.py / reshape_and_cache_flash](https://github.com/sgl-project/sglang/blob/fd19cfb660f42bd3ceb6df7598bd343572e30d2c/python/sglang/kernels/ops/kvcache/cache_ops.py#L6)，[CUDA HND caller](https://github.com/sgl-project/sglang/blob/fd19cfb660f42bd3ceb6df7598bd343572e30d2c/python/sglang/srt/mem_cache/memory_pool.py#L2944)。
- HEAD_BLOCK=4、BLOCK_D=nextpow2(D)，K/V token-major 读取、slot scatter 写入 HND paged cache；目标无规约。
- 两侧 D 都连续，不能预判有布局缺陷；保持真实 cache stride/slot 分布，作为中低怀疑度对照。只用唯一有效 slot，正确处理原实现支持的 -1 padding，避免写竞争。
- 不把 ROCm AITER-only 的 shuffle_5d_cache 路径冒充 NVIDIA 生产默认。

## 3. 首轮规模与门禁

首轮仅 N01/N02/N03，每族两个冻结配置。每配置 U、G0 加最多八个合法唯一 GL，故编译记录上限 60；真实编译和计时按完全等价 CUBIN/ABI/launch 条件去重，保留逻辑别名。正确性尾部用例与相同 JIT specialization 共享编译，不按结果加测赢家。MRoPE、N04–N06 不自动并入首轮。

1. 固定 repo commit、原函数/helper 与 caller；区分实际源码分支、仓库测试形状、拟议性能形状。核对可达 helper 和编译 IR 均无归约，排除 host pipeline 中其他 kernel 的时间。
2. 沿用已验证编译器与缓存；先检查源码兼容性。若需要不同编译器，单列新 stratum，让 U/G0/GL 同版本，不能静默改原算子。此阶段不改 C++、不禁用优化 pass。
3. 用真实 Tensor JIT 导出 U 的 signature、specialization/attributes、全部中间布局、输出编码和 aliasing。G0 按完整源图重建，不只复制输入 layout。
4. 默认 G0 与 U 的 allocated ELF sections、参数 ABI、warp/stages、register/shared/local/global-scratch 配置必须匹配；真实 GPU 加载/输出/driver 检查后，同轮计时 CI95 须落在预定 ±3% 等价带内。过不了先修 G0，不能进入候选收益比较，也不能把 U→G0 差值算作布局收益。
5. GL 只改变冻结边界处的合法布局；第一次保持 tile/grid/W、算法、数学表达、mask/stride、dtype 转换、输出语义不变。不能将 reshape/interleave 改写为另一算法。由最终 TTGIR 检查干预确实生效，记录其引起的其他转换与资源变化。
6. 纯重排/整数解包/FP8 输出做完整逐字节与独立 oracle 检查。U/G0 要求逐位一致；浮点 GL 的容差须在正式计时前按原语义预注册并记录逐位结果，不能事后放宽容差掩盖表达差异。包含 padding、输入不变/原地 alias 检查、canaries。

候选布局不是照搬 reduction 的 N-first/within：按内核物理连续维度、pair/unpack/swizzle 的轴映射，枚举输入友好、输出友好及中间折中分布。SPT 的每轴大小、order、TPW/WPC 均需按形状和 dtype 合法；逻辑向量宽度不解释为硬件 load 次数。

## 4. 计时、NCU 与成功标准

- 首轮计划 H100；不需要 TMA。SM120 来源路径单列，不跨架构混算。这里只提出计划，当前未申请 GPU。
- 同轮配对随机 U/G0/GL；沿用 3 workers×5 visits×31 samples、4096 次配对 bootstrap、3% 实用门槛。保留每个候选的收益/退化及全部样本；逐点探索性 CI 不冒充独立选优验证。
- 对小图与大图、cache-resident 与超 L2 工作集分别报告；kernel 时间是主口径，不把可选子核称为整个框架或模型吞吐提升。scale swizzle 的输入尺寸较小，不能为了制造结果称其为 HBM 受限。
- 正式时间使用无 profiler 的冻结 CUBIN；NCU 另采。预先选定 U/G0 和有明确结构差异的 GL，不只采事后赢家；新增诊断单独标注。
- 记录动态 LDG/STG、SHFL、LDS/STS、BAR、global sector requests、shared wavefront/bank conflicts、local traffic、真实 n_regs、local allocation、capacity q 与 achieved occupancy；warm application 与 cold replay 分开。没有 shared 指令时报告零，不凭空归因到 smem；不把请求数等同 DRAM 字节或直接加权成总分。
- 对新增转换使用当前 CUBIN 的 PC→source 对应，不复用旧源行硬编码。资源、缓存和通信只作为机制证据，不把相关性当独立因果时间分解。
- 若合法候选在基线门通过后相对 U/G0 可重复降低 >3%，且误差区间支持该门槛，记为该配置的默认布局优化缺口；不推出全算子族普遍次优。若默认接近最优或所有干预恶化，完整保留零结果。
- 只有源图一致、无规约门通过的正结果，才可支撑“超出规约的布局选择问题”；仅构造慢布局、手工 retile、改变算法或将含规约 pipeline 拆名均不够。

## 5. 不作为主要候选的源码

- Liger 普通 SwiGLU 的 CTA tensor 主要是一维连续 arange；二维 grid 不等于二维逻辑 tile，优先作为低重分布对照。
- Megatron MoE permutation 流程有 atomic-add/scan，unpermute 有 weighted atomic accumulation，融合量化还可能有 tl.max；不能整体列成无规约。
- TransformerEngine padding 虽接受任意 strides，实际调用前可能已经由 C++ materialize 为连续 tensor，不能只凭 API 推断方向冲突。
- SGLang qknorm+rope、动态 amax 量化、AWQ GEMM 都有规约，不进入本分支。

停点：先复核此候选与方案；随后才冻结正式协议并编译/运行。现阶段不启动下一阶段、GPU、commit/push 或 PR。
