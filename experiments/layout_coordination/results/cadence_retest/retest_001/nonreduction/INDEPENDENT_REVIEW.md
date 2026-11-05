# Non-reduction 复测：独立原始数据复核

本复核只读取已冻结协议、原始 GPU 结果及旧证据，没有修改冻结源码、原始样本或 compiler artifacts，也没有追加 GPU 采样。独立实现了 visit 中位数、均值与共享 worker/visit bootstrap，并与冻结分析器的结果核对。

## 1. 完整性与实际 graph 正确性

- 覆盖 16 个 case、144 个原逻辑程序、134 个原物理程序，以及 16 个独立观测 U-repeat；正式数据为 3 个独立 H100 × 10 visits × 31 samples，共 **139,500 个目标样本**。
- eligibility worker 的 402 个新 finite 检查全部通过：134 个直接执行检查、134 个实际捕获 graph 前置读回、134 个 graph 后置读回。后两种检查不重置输出、不另行执行目标，直接检查已 replay graph 的最终输出。
- 134 个实际 graph 中，120 个为 9 个 kernel 加两个 event；14 个 MRoPE graph 各为 9 个 kernel、18 个 D2D restore 节点加两个 event。每次 prime 和最后的 target 前均恢复两个原地输入物理 storage；目标恢复发生在 start event 前。
- 14 个 MRoPE 物理程序 × 三个检查阶段 × 两个恢复源，共 **84 个 immutable GPU source 哈希读回**全部与初始输入一致。没有把原地旋转连续执行九次后当成一次原始输入计算。
- 全部 16 个 U/G0 对在新 eligibility H100 上的 CUDA ABI、n_regs、local/static shared 与 occupancy-capacity 属性相同。
- 独立逐文件重新计算 SHA：**50,693 个受保护旧文件未变，437 个当前/快照源码文件均与冻结协议一致**。

旧的 finite/structured/edge 正确性证明另行保留并绑定 SHA：N2 的 384 个与 N5 的 18 个原物理检查均通过，覆盖全部原逻辑别名。它们与上述 402 个新 finite/graph 检查是不同证据，不重复计为独立 GPU 计时。

## 2. 控制结果：15/16 U/G0 等价，16/16 U-repeat 等价

等价要求整个配对 95% 区间落入预注册 ±3% 带内。新数据有 **15/16 个 U/G0 对实用等价**，全部 **16/16 个 U-repeat 对实用等价**；一个 U/G0 对保留为不确定，不能写成全部基线均通过。

不确定项为 `N02_k1024_p128_g128`：

| 同一 worker 内均值 | U | G0 | U-repeat | G0 相对 U 变慢 |
|---|---:|---:|---:|---:|
| 1 | 6.1696 µs | 6.3616 µs | 6.1728 µs | 约 3.11% |
| 2 | 6.0768 µs | 6.0864 µs | 6.0800 µs | 约 0.16% |
| 3 | 6.1312 µs | 6.3104 µs | 6.1312 µs | 约 2.92% |

聚合的 G0 相对 U 收益为 **−2.072%，CI [−3.138%,−0.193%]**。区间既未完整落入 ±3%，也未完整低于 −3%，因此是 `inconclusive_U_G0`，不是已证明的 >3% G0 退化。U-repeat 为 −0.035% [−0.227%,+0.156%]。

这个差异不是一个孤立慢 visit：worker 1、3 的 G0 偏移在十个 visit 中均持续存在，worker 2 接近无偏移。它也不是 U/G0 一起移动的共同漂移；U-repeat 与 U 基本重合。

### 这个基线问题排查到了哪一步

- U/G0 两份 CUBIN 均为 36,200 bytes；完整 SHA 不同。完整 ELF 头相同；所有 section 的 alignment/type/flags/link/info/entry-size 相同。
- 执行 `.text`、constant 等 allocated section 的内容及文件位置相同；所有非调试 section 的内容也相同，包括 `.nv.info`、函数 `.nv.info`、callgraph 和 compat 信息。区别限于 debug 内容、debug 大小及由此产生的部分后续 section 文件偏移。
- 新 worker 查询到的 ABI 均为相同九个参数类型；资源均为 72 registers、local 0、static shared 0、capacity q=7；动态 shared、warps、运行参数名称顺序与 specialization 相同。
- 冻结 source 与共享 fixture 规定相同输入/输出指针、标量和 grid `(4,32,1)`。本次未导出逐 graph 的 GPU function 地址或序列化参数地址，因此不把源码一致性表述成独立取得了所有 GPU 地址的现场证明。
- U 与 G0 的 graph 均为 9 kernel + 2 event。由于 U-repeat 与 U 共享同一个 graph，U graph 共 replay **652** 次，G0 共 **342** 次，这是预注册设计：每个观测都含八次 GPU prime，U-repeat 额外贡献 310 次真实观测。不能声称模块/graph 身份或这种活动频率差异的影响已被排除，也没有证据据此确定它们是本次偏移的原因。
- 三台在该 case 的 timing 时窗分别有 81/80/83 次 NVML 查询，均为 SM 1980 MHz、memory 2619 MHz、P0、clock-event mask 0；温度 32–34°C、driver 均为 610.57.04。50 ms 采样不能证明每个约 6 µs kernel 的瞬时时钟相同，更不能由这些读数断定偏移的硬件原因。

目前没有找到源码计算图、执行代码、ABI 或资源不一致的证据。模块/graph 身份、代码映射、调度或其他执行历史因素仍未独立辨识。结果保留为不确定，不追加采样直到通过。

## 3. 原冻结候选的同轮评估

前 14 项使用原 N4 在本轮之前选定的候选；最后两项使用 N5 原定大 footprint 候选。下表不按新数据换赢家。所有区间都是逐比较的 95% 配对 percentile CI，未作多重比较校正。

严格通过还要求该 case 的 U/G0 与 U-repeat 控制均等价，且候选相对 U、G0 的区间下界都超过 +3%。

| Case | 原冻结候选 | 相对 U 收益 % [CI] | 相对 G0 收益 % [CI] | 严格通过 |
|---|---|---:|---:|---|
| N01_m4096_n128 | coordinated_fully_coalesced | +3.898 [2.732,5.346] | +3.048 [2.444,3.547] | 否 |
| N01_m11008_n128 | coordinated_fully_coalesced | +5.816 [5.388,6.291] | +4.784 [4.393,5.191] | 是 |
| N02_k3584_p448_g128 | packed_compute_vec1 | +7.126 [6.852,7.426] | +7.074 [6.920,7.262] | 是 |
| N02_k1024_p128_g128 | packed_compute_vec1 | +3.796 [3.537,4.025] | +5.749 [4.187,6.592] | 否：U/G0 不确定 |
| N03_FA_D64 | paired_vec4 | +0.334 [0.098,0.800] | +0.356 [0.072,0.897] | 否 |
| N03_FA_D128 | paired_vec4 | −0.034 [−0.622,0.637] | −0.134 [−0.731,0.462] | 否 |
| N03_MROPE_D64 | K_vec2 | +0.621 [0.339,0.845] | 0.000 [0.000,0.000] | 否；与 G0 同一物理程序 |
| N03_MROPE_D128 | QK_vec2 | +1.115 [0.575,1.918] | +1.482 [0.882,2.336] | 否 |
| N04_h8_s2048_d128 | v_input_vec8 | −0.079 [−0.804,0.546] | −0.517 [−1.055,0.040] | 否 |
| N04_h32_s4096_d128 | v_input_vec8 | +0.103 [−0.182,0.381] | +0.228 [−0.003,0.451] | 否 |
| N05_t2048_h8_d64 | feature_vec4 | +1.799 [1.415,2.188] | +2.092 [1.642,2.471] | 否 |
| N05_t2048_h8_d128 | feature_vec4 | +4.102 [3.948,4.216] | +3.089 [2.511,3.667] | 否 |
| N06_h8_d64 | e2_heads4 | −1.963 [−3.104,−1.232] | −1.398 [−2.166,−0.291] | 否 |
| N06_h32_d128 | e4_heads1 | +0.138 [−0.141,0.317] | +0.224 [−0.060,0.413] | 否 |
| N01_m4096_n8192 | coordinated_fully_coalesced | +7.887 [7.846,7.930] | +7.809 [7.769,7.851] | 是 |
| N02_k8192_p1024_g128 | packed_compute_vec1 | −1.910 [−2.195,−1.604] | −1.751 [−2.093,−1.400] | 否；±3% 内等价 |

原冻结候选为 **2/14 个 N4 case 严格通过**；加上两个 N5 大 case，共 **3/16**。AWQ 小 case 的候选本身两个收益区间下界都超过 3%，但控制门未过，不能计为受支持的布局改进。N05 D128 仍只有相对 U 下界超过 3%，相对 G0 下界没有超过。

## 4. 完整候选矩阵与旧结论的关系

全部 112 个逻辑 GL 候选中，10 个在四个 case 上达到本轮严格标准：

- `N01_m11008_n128`：6 个候选。
- `N01_m4096_n128`：2 个候选，分别为 `load_s1x1_t8x4_o10`、`load_s1x2_t16x2_o10`。
- `N01_m4096_n8192`：原定 coordinated 候选。
- `N02_k3584_p448_g128`：原定 packed-compute vec1 候选。

完整矩阵的最低耗时项是事后描述，不是新的预注册 selector。特别是小 N01 的两个候选通过，不能用来替换未通过的原 N4 冻结选择后，再宣称原选择已经确认。

当前仍支持 N01 较大 scale-swizzle、较大 AWQ 的候选收益，以及大 footprint N01 的收益；大 footprint AWQ 仍约慢 2%，在 ±3% 内实用等价。RoPE/MRoPE、FP8 pack、KDA gate 与 KV scatter 没有得到本轮双控制支持的 >3% 冻结候选收益。

新旧样本没有混算。新协议是每个目标之前八次 GPU 内语义 prime、1 ms **最小** host 间隙；MRoPE 从不可变 GPU snapshot 恢复输入。它与旧的 Volume 文件 guard、host 恢复及不受控样本间隙是不同执行情境。因此，绝对耗时变化或收益百分比变化不能唯一归因于先前发现的 20 ms idle 效应，也不把旧 NCU 的缓存/带宽/停顿百分比改称为新协议的计数结果。

参考：[冻结协议](../preregistration/protocol.json)、[完整分析](analysis/REPORT.md)、[eligibility 校验](check/attempt_001/validation.json)。
