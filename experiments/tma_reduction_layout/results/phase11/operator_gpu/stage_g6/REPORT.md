# Stage G6 — 默认编译流水线下的 Gluon 正式基线

## 1. 已解决的问题

用户判断成立：旧 Gluon 的数学目标与 U 相同，但没有完整表达 U 经优化后实际执行的中间布局、转换顺序及循环位置，所以其生成程序不是原始 Triton 的等价基线。这个问题应先在代码与证据层解决，不能把旧 G0 的额外变化混入布局加速。

G5 已通过临时编译后缀证明可以对齐。G6 将其收敛为三个独立的 Gluon 源码入口，**只使用未修改的默认 Gluon 流水线**，不再依赖 RMLC/LICM 后缀、Python 后端替换或 C++ 改动：

- 9 个默认几何 + 4 种额外真实调用 ABI，共 **13/13 种编译形式**全部通过。
- 与原始 U 比较：全部 ELF `SHF_ALLOC` 区段逐字节相同；运行时参数类型/顺序/名称相同；寄存器、local/stack、静态/动态 shared memory 和 launch 资源相同。
- 显式映射回 G4b 的 **10/10 实际工作负载**，避免将默认 ABI 的对齐扩大宣称为真实调用的对齐。
- 沿用固定编译器 `20d9d98fead78c2f65e1b3ba5a7881d86a836dad`、原生库 SHA `fc5f2ad73f9f8c8ab31c3a01b8124329ccec391cdc5dbf2dafb2d42778cf6adb` 与 `cuda:90`。

因此，这一版可以作为当前已验证路径的对齐 Gluon 默认基线。源文件路径和 debug 元数据不同，所以不要求整个 CUBIN 容器文件 SHA 相同；比较的是运行程序及相关 ABI/资源。机器代码相同也不代表测量中的随机噪声必须为零。本阶段没有新 GPU 正确性运行或性能计时，不报告新加速。

正式绑定见 [baseline_manifest.json](baseline_manifest.json)，独立离线校验见 [final_validation.json](final_validation.json)。

## 2. 具体改动

| 算子 | 新入口 | 源码中显式表达的操作 |
| --- | --- | --- |
| vLLM segment merge | [g6_vllm.py](../../../../phase11/operator_gpu/g6_vllm.py) | 在统计规约布局和矩阵行切片布局中分别加载/重算统计量；输出索引在目标布局重建；BF16 窄化在输出布局转换之前；保留 U 的地址计算次序。 |
| SGLang decode stage1 | [g6_sglang.py](../../../../phase11/operator_gpu/g6_sglang.py) | 各输入/消费者分别构造索引和 mask；Query 转换、广播及循环不变量明确放在 KV 循环外；输出地址在 LOUT 构造，仅转换实际值；避免 identity conversion。 |
| LayerNorm dw/db merge | [g6_layernorm.py](../../../../phase11/operator_gpu/g6_layernorm.py) | 循环不变量的 range、广播、指针在循环外预构造；输出指针/mask 在 LOUT 重建；两个结果先窄化到输出 dtype 再转换布局。 |

这是 Gluon 程序的表达修正，未改变原始 U、旧 G0 或旧实验结果。没有添加硬件 ID、线程分歧或隐藏 lane/warp 标量；各操作均是合法的块级张量程序。

SGLang 的 d64 还需要对齐索引计算位置：初版虽然数学、通信和资源数相同，但机器字节不一致。核查 LLVM/PTX 后，将 U 已在入口产生的行索引及输出列索引宽化，在源代码中放回对应位置。d128/d256 维持其相应位置，最终所有编译形式一起复核通过。

## 3. 验证矩阵

表中 conversion 为最终 TTGIR 的静态操作数，用于辅助解释；验收依据是全部可执行区段、ABI 与资源，不是 conversion 数或寄存器数相同。

| 编译形式 | n_regs：新 Gluon / U | conversion：新 Gluon / U | 全部区段、ABI、资源 |
| --- | --- | --- | --- |
| `C06_s16_d128` | 32 / 32 | 1 / 1 | 一致 |
| `C06_s32_d128` | 56 / 56 | 1 / 1 | 一致 |
| `C06_s32_d256` | 96 / 96 | 1 / 1 | 一致 |
| `C02_n64_d64` | 56 / 56 | 1 / 1 | 一致 |
| `C02_n64_d128` | 83 / 83 | 2 / 2 | 一致 |
| `C02_n64_d256` | 127 / 127 | 2 / 2 | 一致 |
| `C01_p256_n1024` | 96 / 96 | 2 / 2 | 一致 |
| `C01_p128_n4096` | 96 / 96 | 2 / 2 | 一致 |
| `C01_p96_n8192` | 96 / 96 | 2 / 2 | 一致 |
| `C06_s16_d128_b1_l8192` | 32 / 32 | 1 / 1 | 一致 |
| `C06_s32_d256_b1_l8192` | 96 / 96 | 1 / 1 | 一致 |
| `C02_n64_d128_b8_l2048` | 83 / 83 | 2 / 2 | 一致 |
| `C02_n64_d64_b8_l2048` | 56 / 56 | 1 / 1 | 一致 |

另见 [attempt_003/comparison.json](attempt_003/comparison.json) 中每个原始编译产物路径与完整比对；`baseline_manifest.json` 的 `actual_G4b_workloads` 保存了十个实际工作负载逐项绑定，不是仅按形状推断。

## 4. 保留的调查过程

| 轮次 | CPU 编译请求数 | 结果 | 处理 |
| --- | --- | --- | --- |
| attempt_001 | 5 | vLLM 5/5 对齐 | 保留原始包、源码快照与逐项比对。 |
| attempt_002 | 13 | 11/13 对齐；SGLang d64 的两种 ABI 未对齐 | 不作为正式基线，保留失败事实。 |
| attempt_003 | 13 | 13/13 对齐 | 冻结为 `G0_source_aligned_v1`。 |

共 31 次编译 API 请求，允许命中隔离的内核缓存；不能将这个数字说成 31 次新原生编译。所有任务均复用既有原生镜像和 ccache，`ccache_before == ccache_after`；原生重编译为 0。

本阶段使用 `chenmiaoming` profile，只分配 Modal CPU。GPU 分配、新计时样本、新 NCU 报告均为 0。所有历史证据均保持原字节；当前冻结清单保护 G2–G5 共 4,671 个文件。

## 5. 后续如何使用

入口选择与参数适配统一在 [g6_compile.py](../../../../phase11/operator_gpu/g6_compile.py)：

- `function_for_family(family)` 返回新的源码基线函数。
- `arguments(item, gl)` 按冻结的实际 signature、attributes、specialization 和默认 layout 构造编译参数。
- `arguments(item, gl, candidate_layout=...)` 保留布局干预入口。候选不能自动继承默认基线的等价或性能结论，必须另做实际生效布局与正确性检查。

`baseline_manifest.json` 指明了版本、源文件 SHA、编译器、目标、13 种实际 ABI/编译形式、产物和工作负载对应关系。后续实验应使用这个新入口，旧 G0/G5 文件用于复核历史，不再把旧 G0 称作原 U 的等价默认基线。旧运行脚本没有被静默改写，避免改变历史实验的复现含义。

范围限定为冻结的 BF16/FP32 路径、编译器、目标、tile/warp 参数与调用特化。未验证 FP8、可选 score modifier 或其他架构/编译器版本；新增形式应重新走门禁。这个范围不妨碍当前基线使用，但不能扩大到整个算子全部配置。

接下来的性能实验应比较 `U / G0_source_aligned / GL`。默认两者现在执行同一程序；从该源图出发改布局，才有条件重新评估收益。旧 GL 相对 U 的总收益仍不能重新贴标签为纯布局收益；新候选尚未在本阶段测量。

## 6. 校验与停点

原始导出 ZIP、协议、快照与所有失败尝试保留。最终 validator 重读原始包、全部 artifact、真实调用绑定及受保护旧文件，并检查完整 case 集合、无后端干预、固定 native/backend SHA 与 ccache 统计。

前序 G4/G5 validator 已复跑通过；G6 最终结果见 [final_validation.json](final_validation.json)。复核步骤与接入示例见 [REPRODUCE.md](REPRODUCE.md)。交付停在源码基线修复和 CPU 验证，没有启动后续 GPU sweep 或论文/PR 工作。
