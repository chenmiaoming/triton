# Stage N1：无规约算子的 U / G0 基线对齐

本阶段建立可复核的实验基线：六个算子族、七个计算子类、14 个固定 case。U 为冻结的公开 Triton 程序；G0 按 U 的实际编译结果重建。本阶段仅比较 U/G0，不据此判断候选布局收益或跨算子的共同性能机制。

## 1. 研究范围与公开源码

| 族/子类 | 独立 kernel | 本轮适用范围 |
|---|---|---|
| N01 | 量化 scale swizzle | torchao 的可选 Triton scale 重排；不包含 scale 计算或量化归约。 |
| N02 | AWQ 反量化 | vLLM 可选 Triton AWQ 解包/反量化；不是 AWQ GEMM，也不代表默认后端。 |
| N03/FA | 普通 RoPE | FlashAttention interleaved、非原地、完整 rotary dimension；本轮不涵盖 varlen/partial rotary 等分支。 |
| N03/MRoPE | 三轴 MRoPE | SGLang CUDA NeoX-style、BF16 cache、Q/K 原地更新；未覆盖 GLM/interleaved 的其他分支。 |
| N04 | 固定 scale 的 FP8 QKV pack | 提取自 SGLang SM120 attention preparation；仅迁移独立 pack 到 H100，不是该后端在 H100 的生产默认。 |
| N05 | KDA forward gate | 固定 A/bias，未启用 beta/lower-bound 分支；不包含 chunk-cumsum、反向或完整 KDA 流水。 |
| N06 | KV cache scatter | SGLang CUDA HND 物理布局，唯一 slot 映射；未启用 SWA/scale，也不包含 attention。 |

所有被测 kernel 的源码与生成 IR 均检查了 reduction、scan、dot 和原子累积。逐元素 maximum、三轴系数逐元素相加及位解包不属于轴规约。完整公开文件可能还包含其他未执行的 kernel；检查范围为所选函数及其本地 helper 闭包。

| 函数/依赖 | 冻结 commit | 原始 Git blob |
|---|---|---|
| [triton_scale_swizzle](https://github.com/pytorch/ao/blob/b3fc9544fd014daf04410aed233f8640d4db482a/torchao/prototype/mx_formats/kernels.py#L316-L365) | `b3fc9544fd01` | `89ea5759e2345e2f604bfa10d5cea7c9733e6374` |
| [awq_dequantize_kernel](https://github.com/vllm-project/vllm/blob/67e6ea21a7bb9dd342511c9ebe200821af41b6e8/vllm/model_executor/layers/quantization/awq_triton.py#L11-L105) | `67e6ea21a7bb` | `67b4dbbfd4d82ec89d605f5352b24e142507c221` |
| [rotary_kernel](https://github.com/Dao-AILab/flash-attention/blob/5f7b05a78bf05cb33fb8ed07f74d2b82c9839dcd/flash_attn/ops/triton/rotary.py) | `5f7b05a78bf0` | `ff4017fda3e4a6e18cf3a51b34f0fb073d8f678a` |
| [_triton_mrope_forward_fused](https://github.com/sgl-project/sglang/blob/fd19cfb660f42bd3ceb6df7598bd343572e30d2c/python/sglang/kernels/ops/attention/rotary_triton.py) | `fd19cfb660f4` | `3e066bb070501ca5c1ca847d32c5bc4aa8ee4bf1` |
| [pack_fused_prep_pinned.py](https://github.com/sgl-project/sglang/blob/fd19cfb660f42bd3ceb6df7598bd343572e30d2c/python/sglang/kernels/ops/attention/fp8_fa_sm120/fused_prep.py) | `fd19cfb660f4` | `19d96269f912ae59a99a78d2edccbccd22b80621` |
| [gate_gate_pinned.py](https://github.com/fla-org/flash-linear-attention/blob/9e82a80fdaefcd7c9cc3da51ea36d6a9e1587bd1/fla/ops/kda/gate.py) | `9e82a80fdaef` | `5c1df57fd5a3f2baaacae82ff6e2520139c309c3` |
| [gate_softplus_pinned.py](https://github.com/fla-org/flash-linear-attention/blob/9e82a80fdaefcd7c9cc3da51ea36d6a9e1587bd1/fla/ops/utils/softplus.py) | `9e82a80fdaef` | `2052828995fa1f80bb595bff8232fcf78465e964` |
| [gate_op_pinned.py](https://github.com/fla-org/flash-linear-attention/blob/9e82a80fdaefcd7c9cc3da51ea36d6a9e1587bd1/fla/ops/utils/op.py) | `9e82a80fdaef` | `c62c0b3d5d0225dcb67278a8c96ec6db291cd233` |
| [reshape_and_cache_flash](https://github.com/sgl-project/sglang/blob/fd19cfb660f42bd3ceb6df7598bd343572e30d2c/python/sglang/kernels/ops/kvcache/cache_ops.py) | `fd19cfb660f4` | `c8496856a666fb7d7744473e7d79ffd8bbe30ab0` |

原始完整源码、独立提取文件与 AST/Git-blob 指纹分别保留。N04 的 `_pack_qkv` 与 helper 保持原始函数体/签名；N05 保留 NVIDIA softplus 与 `FLA_USE_FAST_OPS=0` 的 exp 语义，仅把原来的宿主 autotune/heuristic 包装拆开记录。N06 的绑定文件与原始完整文件只差末尾换行，kernel 和 wrapper AST 相同。

## 2. 配置、编译与正确性门

- 编译器 commit：`20d9d98fead78c2f65e1b3ba5a7881d86a836dad`；native SHA256：`fc5f2ad73f9f8c8ab31c3a01b8124329ccec391cdc5dbf2dafb2d42778cf6adb`。
- 原始 U 使用实际 Torch Tensor JIT 绑定，保留真实 dtype、stride、storage alias、constexpr 与运行时整数的 specialization。
- revision 008：14/14 个 CPU G0 与 U 的全部 allocated ELF section、运行 ABI/launch 及资源使用相同。比较不只覆盖 PTX/SASS 计数，也不要求调试节或非 allocated 节的文件字节相同。
- revision 009：实际 Tensor JIT 的 G0、CPU G0 与 U 三方 allocated ELF/ABI/资源检查全部通过；G0 共 42 项 finite/structured/edge 正确性检查通过。
- 独立冻结二进制 driver 检查：28 个 U/G0 程序，84 项完整输出/只读与未激活区域/guard 检查通过。三个计时 worker 另各有 28, 28, 28 项有限值检查；交错采样协议下，这些检查位于首个 visit 的 warmup 后、正式采样轮之前。
- MRoPE 的 q/k 是原地输出。每次 warmup 与每个计时样本之前都恢复原始物理存储；恢复操作位于 graph 的两个计时事件之外。该 kernel 时间不含恢复/宿主搬运成本。

| Case | 逻辑形状/参数 | Warps | Stages |
|---|---|---:|---:|
| `N01_m4096_n128` | 4096×128 | 4 | 3 |
| `N01_m11008_n128` | 11008×128 | 4 | 3 |
| `N02_k3584_p448_g128` | packed 3584×448 → 3584×3584 | 4 | 3 |
| `N02_k1024_p128_g128` | packed 1024×128 → 1024×1024 | 4 | 3 |
| `N03_FA_D64` | 4×2048×32×64 | 4 | 3 |
| `N03_FA_D128` | 4×2048×32×128 | 4 | 3 |
| `N03_MROPE_D64` | T=2048, QH/KH=32/8, D=64 | 4 | 3 |
| `N03_MROPE_D128` | T=2048, QH/KH=32/8, D=128 | 4 | 3 |
| `N04_h8_s2048_d128` | H=8, S=2048, D=128 | 8 | 3 |
| `N04_h32_s4096_d128` | H=32, S=4096, D=128 | 8 | 3 |
| `N05_t2048_h8_d64` | T=2048, H=8, D=64, BT=32 | 8 | 2 |
| `N05_t2048_h8_d128` | T=2048, H=8, D=128, BT=32 | 8 | 2 |
| `N06_h8_d64` | T=2048, H=8, D=64, block=16 | 4 | 3 |
| `N06_h32_d128` | T=2048, H=32, D=128, block=16 | 4 | 3 |

N05 先只用 U 遍历原公开的 `BT∈{32,64,128} × W∈{4,8,16,32} × stages∈{2,3}`，即每个 case 24 个配置。每配置 3 个 visit、每 visit 9 个 graph 样本，按 visit 中位数的均值选取。这是保留原配置菜单的独立 U-only 选择，不冒充完整 FLA autotuner/cache 实现，也未利用 Gluon 布局速度选参。
- `N05_t2048_h8_d64` 选定：`BT=32, W=8, stages=2`。
- `N05_t2048_h8_d128` 选定：`BT=32, W=8, stages=2`。

## 3. 同轮 U/G0 计时

正式估计量为 3 个 worker × 每 worker 10 个 visit；每 visit 31 个原始样本先取中位数，再对 worker/visit 中位数取算术平均。共 26,040 个正式原始样本。每次 graph 仅一个 kernel 与两个外部计时事件，未附加 profiler。

采样方案修订：初轮采用逐程序成块采样，观察到部分 visit 间漂移及时间离群；查看全部 U/G0 控制后，对全部 14 个 case 统一改为同一 case 内逐 round 随机交错 U/G0，并统一增加为 10 个 visit。每 round 每个物理程序各获得一个观测，完整顺序随原始记录保留。kernel、单 kernel graph、输入与冻结二进制保持相同；原轮未删除、未与新轮混合。这是查看初轮控制后的自适应采样修订，不是初轮预注册方案的确认性结果。

| 采样轮 | 顺序与 visit 数 | 原始样本数 | 等价 / 不确定 |
|---|---|---:|---|
| [r2 初轮](../stage_n1_baseline_r2/analysis/analysis.json) | 逐程序成块，5 visits | 13,020 | 7 / 7 |
| 当前冻结轮 | `interleaved_rounds`，10 visits | 26,040 | 14 / 0 |

新轮的统计稳定性与更均衡的时间采样一致；没有用 profiler 对初轮离群的硬件原因进行归因，也不把这一采样比较解释为某条 GPU 指令或布局机制的因果实验。

U/G0 在每个 worker 独立测量。4096 次配对分层 bootstrap 共用 worker/visit 抽样索引；95% CI 完全落在 ±3% 内才分类为 practical equivalence，其余分类按观察值保留。机器代码等价门与带噪声的计时分类分别报告；完整保留每轮数据及采样方案修订。

| Case | U (µs) | G0 (µs) | G0 相对 U 收益 (%) | 配对 95% CI (%) | 分类 |
|---|---:|---:|---:|---|---|
| `N01_m4096_n128` | 6.382 | 6.452 | -1.10 | [-1.35, -0.85] | practically_equivalent |
| `N01_m11008_n128` | 8.462 | 8.415 | +0.55 | [+0.37, +0.73] | practically_equivalent |
| `N02_k3584_p448_g128` | 13.051 | 13.137 | -0.66 | [-1.44, -0.10] | practically_equivalent |
| `N02_k1024_p128_g128` | 6.355 | 6.279 | +1.19 | [+0.78, +1.74] | practically_equivalent |
| `N03_FA_D64` | 27.914 | 27.991 | -0.28 | [-0.60, -0.07] | practically_equivalent |
| `N03_FA_D128` | 50.698 | 50.765 | -0.13 | [-0.30, +0.04] | practically_equivalent |
| `N03_MROPE_D64` | 11.478 | 11.500 | -0.19 | [-0.47, +0.10] | practically_equivalent |
| `N03_MROPE_D128` | 16.805 | 16.745 | +0.36 | [+0.20, +0.55] | practically_equivalent |
| `N04_h8_s2048_d128` | 11.009 | 11.041 | -0.29 | [-0.55, +0.04] | practically_equivalent |
| `N04_h32_s4096_d128` | 60.901 | 60.803 | +0.16 | [+0.04, +0.29] | practically_equivalent |
| `N05_t2048_h8_d64` | 6.301 | 6.357 | -0.90 | [-2.15, +0.27] | practically_equivalent |
| `N05_t2048_h8_d128` | 7.773 | 7.716 | +0.73 | [+0.56, +0.91] | practically_equivalent |
| `N06_h8_d64` | 7.645 | 7.652 | -0.10 | [-0.56, +0.46] | practically_equivalent |
| `N06_h32_d128` | 28.430 | 28.452 | -0.08 | [-0.33, +0.13] | practically_equivalent |

分类计数：`practically_equivalent`=14。

| 计时 worker | GPU UUID | GPU / SM 数 |
|---|---|---|
| 1 | `GPU-bcd9981e-5d1a-69f2-08f2-5c5d5021674d` | NVIDIA H100 80GB HBM3 / 132 |
| 2 | `GPU-6bcf9c0c-834d-b41e-10ae-40173c6b9bbe` | NVIDIA H100 80GB HBM3 / 132 |
| 3 | `GPU-a1267821-5aa8-5381-f8da-d1d33a6fd49b` | NVIDIA H100 80GB HBM3 / 132 |

## 4. 缓存、失败记录与证据边界

- 各 worker 的 native SHA 固定，记录的 native rebuild 数均为 0；复用既有 Triton image、内核编译缓存与 Modal cache volume，没有重新构建 Triton C++。保留的 13 组 ccache 前后快照中，13 组计数未变；这些快照用于检查 native 构建活动，不据此推导未记录的 Triton kernel-cache 命中率。
- revision 001 的冻结依赖缺失、首个 baseline worker 导入时读取本地主机协议路径的失败均原样保留。两者未形成正式计时样本；后者改用新增的 `stage_n1_baseline_r2`，不覆盖原协议。
- 各次 G0 编译失败或不匹配也保留在独立 revision。最终接受 revision 008 与实际绑定 revision 009；不要求历史失败尝试变成通过。
- H100 上的独立 kernel 证据不能外推为 SM120/B200、完整 LLM 或框架默认路径的收益。本阶段尚未证明哪些候选布局更快，也未证明存在版本性能回归。后续布局比较需要同轮 U/G0 与完整候选真值。

## 5. 可复核产物

- [接受清单](accepted_manifest.json)，SHA256：`7009ffea2a5dea5b38193580ffddcd40e997e489573c015bce69de62d62dd2fb`。
- [正式协议](../stage_n1_baseline_r3/preregistration/protocol.json)，SHA256：`bc478215fd1cb2ede2cfc8388ba7a54252e6e4a34fa9457c4aeecfe60cb45810`。
- [计时分析](../stage_n1_baseline_r3/analysis/analysis.json)；全部原始记录由其输入 SHA 列表与接受清单关联。
- [实际 G0 绑定](revision_009/g0bind/attempt_001/export/result.json)；保留三方完整 IR/PTX/SASS/CUBIN、资源与检查记录。
- [源码 provenance](../../../../phase11/nonreduction_survey/sources)；各 revision 保留协议与 source snapshot ZIP。
- [最终 CPU validator](validation_final.json)：`OK`。

历史尝试索引（状态不作改写）：

| 记录 | 状态 |
|---|---|
| `revision_002/binding/attempt_001/export/result.json` | OK |
| `revision_003/autotune/attempt_001/export/result.json` | OK |
| `revision_004/g0/attempt_001/export/result.json` | NEEDS_ALIGNMENT |
| `revision_005/g0/attempt_001/export/result.json` | NEEDS_ALIGNMENT |
| `revision_006/g0/attempt_001/export/result.json` | NEEDS_ALIGNMENT |
| `revision_007/g0/attempt_001/export/result.json` | NEEDS_ALIGNMENT |
| `revision_008/g0/attempt_001/export/result.json` | OK |
| `revision_009/g0bind/attempt_001/export/result.json` | OK |
| `revision_101/g0/attempt_001/export/result.json` | NEEDS_ALIGNMENT |
| `revision_102/g0/attempt_001/export/result.json` | OK |
| `revision_103/g0/attempt_001/export/result.json` | OK |
| `revision_201/g0/attempt_001/export/result.json` | OK |
| `revision_301/g0/attempt_001/export/result.json` | OK |
