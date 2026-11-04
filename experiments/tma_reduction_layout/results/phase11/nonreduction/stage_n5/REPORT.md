# Stage N5 — footprint 与尾块/步幅跟进

固定候选在新 footprint 上表现不同：N01 scale swizzle 相对 U 提高 **7.9134%**（95% CI [+7.8003%, +8.0364%]）；AWQ dequant 候选则慢 **1.9550%**（收益 CI [-2.1127%, -1.7949%]）。两项 U/G0 基线均实测等价。AWQ 的方向为小幅变慢，但整个区间仍在预注册 ±3% 实用等价带内；不能称为 >3% 的实用退化。

这轮是已知 kernel 的条件跟进。两个大 case 各只有一个事先绑定的候选：`N01 coordinated_fully_coalesced` 与 `N02 packed_compute_vec1`；未根据本轮计时或计数重新选择、修改候选。编译器版本固定，因此这里讨论布局选择与尺寸的关系，不把它称为版本升级引起的性能回归。

## 1. 程序、footprint 与正确性

| Case | 数据 storage 体积 | 固定候选 | 正确性覆盖 |
| --- | --- | --- | --- |
| N01_m4096_n8192 | uint8 input 32 MiB + output 32 MiB = **64 MiB** | coordinated_fully_coalesced | U/G0/GL × 3 patterns，9/9 |
| N02_k8192_p1024_g128 | packed int32 weight 32 MiB + FP16 scale 1 MiB + zero 0.25 MiB + FP16 output 128 MiB = **161.25 MiB** | packed_compute_vec1 | U/G0/GL × 3 patterns，9/9 |
| N01_tail_stride_m131_n7 | input logical 131×7，stride=(19,2)；完整 padded output=256×8 | 原八个声明布局全部保留，包括默认 alias | U/G0/8 GL × 3 patterns，30/30 |

以上体积来自真实绑定 fixture 的 shape/stride/element size 对应的数据 storage，不含每个 allocation 的 256B 前后 guard。大 case 的数据 tensor 均为完整连续 storage；尾块输入覆盖 2483B 物理跨度，其中 917B 为逻辑元素、1566B 为只读间隙。输出的 2048 个 uint8 位置全部验证，包含 1131 个由 padding 产生的零值位置。

**总计 16 个逻辑程序、48/48 个完整检查通过**，覆盖 finite/structured/edge、独立 oracle、guard、只读输入及存储间隙。三个 case 的真实 Tensor U/G0 JIT 均通过 allocated ELF、运行 ABI/launch 与资源一致性门。尾块仅用于正确性检查，不进入计时或这六份 NCU 报告。

编译器 commit：`20d9d98fead78c2f65e1b3ba5a7881d86a836dad`；native SHA256：`fc5f2ad73f9f8c8ab31c3a01b8124329ccec391cdc5dbf2dafb2d42778cf6adb`。复用现有编译镜像及 kernel/compiler 缓存；绑定阶段 ccache 前后统计相同，native rebuild=0。正式计时与 NCU 阶段均加载冻结 CUBIN，不执行 JIT。

## 2. 无 profiler 的正式计时

3 个独立 worker，每个 10 visits、每个 visit 31 个原始样本，共 5580 个样本。每轮随机交错测量各物理程序；单 kernel CUDA Graph 配两个外部计时事件。点估计为 worker×visit 中位数的算术平均；4096 次配对 worker/visit 层次 bootstrap，U/G0/GL 共享抽样。区间为逐点 95% percentile CI，未作多重比较校正。

| Case | U (µs) | G0 (µs) | 固定 GL (µs) | GL vs U 收益与 CI | GL vs G0 收益与 CI |
| --- | --- | --- | --- | --- | --- |
| N01_m4096_n8192 | 79.4464 | 79.5061 | 73.1595 | +7.9134% [+7.8003%, +8.0364%] | +7.9826% [+7.8695%, +8.1454%] |
| N02_k8192_p1024_g128 | 63.6181 | 63.6597 | 64.8619 | -1.9550% [-2.1127%, -1.7949%] | -1.8884% [-2.0873%, -1.6763%] |

| Case | G0 vs U 收益 | 95% CI | 基线结论 |
| --- | --- | --- | --- |
| N01_m4096_n8192 | -0.0752% | [-0.1731%, +0.0215%] | ±3% 内实用等价，U/G0 独立物理测量 |
| N02_k8192_p1024_g128 | -0.0654% | [-0.2454%, +0.1335%] | ±3% 内实用等价，U/G0 独立物理测量 |

N01 在这一较大 footprint 上仍有超过 3% 的收益。AWQ 的相同候选没有获得实用收益，且观察到约 2% 的小幅反向。因此，减少某条转换路径或增加资源容量都不能单独当作端到端收益判据。此处没有重扫其他 AWQ 布局来替换该结果。

## 3. 当前 footprint 的 NCU 硬件观测

六份报告均为完整 grid、`application replay / cache none / clock none`，八次同程序 warmup 后采集一个 kernel。每份报告独立加载一个冻结二进制，全部 replay 的 loaded SHA、完整 oracle/guard、原始参数、GPU UUID 和 CUDA q 已验证。**N5 没有采集 InstructionStats**，不生成 CTA0 软件指令表，也没有复用小尺寸 N3 的全图缓存百分比。以下只使用本轮大 footprint 的真实计数。

| Case / Variant | CUDA REG / capacity q | L2 hit (%) | DRAM read / write (MiB) | DRAM % peak | active warps % peak |
| --- | --- | --- | --- | --- | --- |
| N01_m4096_n8192 / U | 14 / 16 | 91.5400 | 32.0818 / 33.5688 | 25.0800 | 87.0100 |
| N01_m4096_n8192 / G0 | 14 / 16 | 91.5600 | 32.0845 / 34.0591 | 25.3800 | 87.0000 |
| N01_m4096_n8192 / GL_coordinated_fully_coalesced | 12 / 16 | 89.3400 | 32.0139 / 33.3772 | 27.3900 | 88.2200 |
| N02_k8192_p1024_g128 / U | 72 / 7 | 79.6600 | 33.2642 / 126.3135 | 82.9600 | 40.7200 |
| N02_k8192_p1024_g128 / G0 | 72 / 7 | 79.6200 | 33.2668 / 126.2439 | 82.1000 | 40.8000 |
| N02_k8192_p1024_g128 / GL_packed_compute_vec1 | 40 / 12 | 79.5900 | 33.2639 / 126.4136 | 82.2900 | 66.8500 |

| Case / Variant | Global load requests / sectors per CTA | Global store requests / sectors per CTA | Shared load / store wavefronts per CTA |
| --- | --- | --- | --- |
| N01_m4096_n8192 / U | 4.0000 / 128.0000 | 4.0000 / 64.0000 | 0.0000 / 0.0000 |
| N01_m4096_n8192 / G0 | 4.0000 / 128.0000 | 4.0000 / 64.0000 | 0.0000 / 0.0000 |
| N01_m4096_n8192 / GL_coordinated_fully_coalesced | 4.0000 / 128.0000 | 4.0000 / 16.0000 | 0.0000 / 0.0000 |
| N02_k8192_p1024_g128 / U | 16.0000 / 160.0000 | 32.0000 / 512.0000 | 152.8262 / 133.7455 |
| N02_k8192_p1024_g128 / G0 | 16.0000 / 160.0000 | 32.0000 / 512.0000 | 152.5729 / 133.6149 |
| N02_k8192_p1024_g128 / GL_packed_compute_vec1 | 40.0000 / 160.0000 | 32.0000 / 512.0000 | 17.7172 / 4.0856 |

`capacity q` 来自 CUDA occupancy API，是给定函数与 block/shared 配置的最大 CTA 容量；它不是实际运行时始终驻留的 CTA 数。active-warps 百分比是另一项实测指标，不能与 q 混写。只对 additive 流量/指令 sum 按真实 CTA 数归一；百分比及比值保持原单位。

## 4. 正向与反向结果的条件解释

### N01_m4096_n8192

冻结候选的正式收益为 +7.9134% [+7.8003%, +8.0364%]。本轮 U→GL 的全局 store sectors/CTA 为 64.0000→16.0000；global load requests/CTA 为 4.0000→4.0000；shared load wavefronts/CTA 为 0.0000→0.0000。REG / capacity q 从 14 / 16 变为 12 / 16。

大 footprint 下测得的 L2 hit 为 91.5400%→89.3400%；DRAM read/write 为 32.0818/33.5688 MiB→32.0139/33.3772 MiB。这些是当前二进制与当前运行条件的观测，不能从 storage 大小预先推出。

该对照同时协调输入与输出所有权，保留原始字节置换。请求分组的变化与本轮实用收益可以并列观察；本轮没有独立扫描加载、写回和缓存并发，因而不把全部收益唯一归因于某一项计数。

### N02_k8192_p1024_g128

冻结候选的正式收益为 -1.9550% [-2.1127%, -1.7949%]。本轮 U→GL 的全局 store sectors/CTA 为 512.0000→512.0000；global load requests/CTA 为 16.0000→40.0000；shared load wavefronts/CTA 为 152.8262→17.7172。REG / capacity q 从 72 / 7 变为 40 / 12。

大 footprint 下测得的 L2 hit 为 79.6600%→79.5900%；DRAM read/write 为 33.2642/126.3135 MiB→33.2639/126.4136 MiB。这些是当前二进制与当前运行条件的观测，不能从 storage 大小预先推出。

该对照保留原始 AWQ 解包、零点、scale 和 FP16 输出语义。即使通信或资源指标改善，也必须保留这次小幅变慢的正式结果；加载指令的宽度/数量、请求、展开计算与并发掩盖会共同决定净时间，本轮没有分量计时来唯一分解它们。

L2 hit 是缓存层汇总比例，不能据此证明全部 tensor 的驻留状态；DRAM 读写也受缓存及写策略影响。这轮没有独立测定纯 HBM 带宽上界、流量下限和重叠边界，因此不宣称“超过 L2 容量就处于纯 HBM roof”。NCU 下的时间也不替代第 2 节无 profiler 的正式计时。

## 5. 失败与派生修订均保留

- `revision_001`：两个大 case 已通过 18 个完整检查；尾块 U/G0 的编译一致性也通过。旧 adapter 跳过与新默认相同的输入布局，随后触发固定八个候选的枚举断言，尾块执行前终止。完整 raw export 保留。`revision_002` 仅在独立 N5 协议中绑定原先八个候选定义并保留 alias，未修改冻结 adapter 或内核。
- 首份派生 `BINDING_REPORT.md` 误用展开宽度填写 AWQ 输入 stride；保留原文件，`BINDING_REPORT_v2.md` 按真实 fixture 元数据改为 packed qweight shape=`8192×1024`、stride=`(1024,1)`。原始实验与验证未变。
- 正式表同时保留 N01 的收益与 AWQ 的小幅反向，不用新的候选或新的指标替换不利结果。

## 6. 证据入口与停点

- [真实 JIT 绑定、48 项检查和资源门](revision_002/BINDING_REPORT_v2.md)
- [绑定 CPU 最终校验](revision_002/BINDING_VALIDATION.json)
- [正式计时分析及所有配对区间](../stage_n5_timing/analysis/analysis.json)
- [六份硬件报告的独立数值分析](../stage_n5_profiles/analysis/summary.json)
- [NCU 口径与完整指标长表](../stage_n5_profiles/analysis/REPORT.md)

本轮补齐的是当前 kernel、当前设备和两个新 footprint 下的证据，以及一项尾块/步幅正确性压力测试。它不证明所有模型尺寸或其他架构上的收益，也没有启动新的布局调优或编译器修改。

| 输入 | SHA256 |
| --- | --- |
| binding_protocol | `24140786e9a01148c11274582291a1367ef8767c374e25f644467a93817d4f28` |
| binding | `1cd559931dffcd1e99c7ed1f27e38915c8aaef514e12172da863e34f8162df45` |
| binding_validation | `c9e0a5946ebe2702983afdaed92b9f66c070b7b3d672023844d64e8cb2eae179` |
| timing_protocol | `568bc67acf2943684adcb491315121a8709eecc02e1484f87b61c76c73611b70` |
| timing | `433caa07d44685dabd9f8b60ee1e48456a269987f278c16e20f63d6f182f6244` |
| profiles_protocol | `9c95900ff5202b553b1fde1c1e8153656b4838ac997b8b8c9fd20b468440eaef` |
| profiles | `1cf31622c27956170a9787642c51f14cf0fa1cb00236ff7adfb9c03414102191` |
