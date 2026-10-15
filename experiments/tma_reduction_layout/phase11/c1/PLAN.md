# C1 — H100 上的 ML/MR 可行性检查

本阶段由用户单独批准。它覆盖两种几何的 N-first vec1/2/4/8，共 8 个布局、
16 个逻辑程序。它替代旧 review draft 的 C1 布局列表和 MR 广播反馈设计，
不授权进入 C2/C3/D、修改编译器或创建/更新 PR。

输入为 BF16，固定 upstream `20d9d98fead78c2f65e1b3ba5a7881d86a836dad` 的未修改 native。
沿用原 issue 的 host-created descriptor，保留 NVMMASharedLayout 的 128-byte swizzle。
H100 使用 `H100!`，记录真实 CC 和 Driver UUID。继续使用 `chenmiaoming` profile。

## 矩阵

数组省略 size=1 的 batch 维，按 `[M,N]` 表示。SPT 为 `[1,vec]`。

| 配置 | vec | TPW | WPC |
|---|---:|---|---|
| 32×64/w8 | 1 | [1,32] | [4,2] |
| 32×64/w8 | 2 | [1,32] | [8,1] |
| 32×64/w8 | 4 | [2,16] | [8,1] |
| 32×64/w8 | 8 | [4,8] | [8,1] |
| 32×128/w4 | 1 | [1,32] | [1,4] |
| 32×128/w4 | 2 | [1,32] | [2,2] |
| 32×128/w4 | 4 | [1,32] | [4,1] |
| 32×128/w4 | 8 | [2,16] | [4,1] |

## 循环与观测

`K=1,16,64,256` 为同一 CUBIN 的 runtime 参数，禁止把四个 K 编译为四份程序。

ML 在循环前用 TMA 填充两个固定 shared tile，wait 后通过 `slot=i&1` 轮换 memdesc view；
每次读数进入 FP32 elementwise accumulator，最终写出完整 tile，没有 max。
两槽位使用当前 tile 和 M-tile 坐标偏移 128 的 tile，所有布局的 TMA 字节数相同。

MR 在循环前只读一次输入。每轮 max 沿 M 生成结果，累加到原生结果 SliceLayout 的寄存器中。
每个输入元素独立执行有界步进：值 >=8 时减 16，否则加行坐标决定的 quarter step，再舍入 BF16。
归约结果不反馈给输入，无显式 shared store、反向布局转换或广播；只有一次最终 global store。
MR 的输入状态递推与结果 accumulator 各有依赖，但不是归约结果反馈给下一轮输入的延迟链。

先核对有限 quarter、确定性随机 BF16、有限浮点边界三种输入；四个 K、grid=2 检查完整输出。
固定 CUBIN 导出 GLIR/TTGIR/LLVM IR/PTX/SASS、native/assembler/source SHA、资源与 occupancy API 结果。
新程序首次编译允许 cache miss；计时与 NCU 仅加载导出 binary，不再 JIT，不重编译 native。

NCU 分开采集 16×4 个软件和硬件报告，共最多 128 个成功报告。
软件侧保留逐 PC 计数，按逻辑 opcode 的全部展开/尾部副本合计，验证循环内 shared load/store、shuffle、bar.sync、max 与转换的动态执行为 K 倍；
初始化与最终写出的 PC 应为 K 不变，异常、循环控制及谓词计数全部保留。
硬件侧采集 shared load/store wavefront、bank-conflict 和 executed instruction 计数。
Requests 在阶段 A 不可用，继续保留为缺失。

计数解释的技术修订：编译器自动进行四路展开，K=1 走尾部代码。因此需要合计同一逻辑 body
在各 PC 上的副本，而不是要求单个展开副本的 PC 自己执行 K 次。原始逐 PC 数据保留，
revision 4 在正式 event 计时前冻结该解释；opcode 组动态计数仍严格等于 K 倍，未修改 CUBIN。

计数门通过后，3 次 worker 对 grid=1 与 grid=64×实际 SM 数分别计时；
每个 K/cell 固定 3 次预热、20 个单 kernel CUDA-event 样本，使用随机平衡访问顺序。
先在 worker 内以四个 K 的中位数拟合截距与斜率，再汇总三次斜率及相邻 ΔT/ΔK、残差。
截距包括初始化/TMA/写出；斜率包含循环内 fixture 工作，均不直接命名为裸指令延迟。

## 趋势检验及停点

比较 ML/MR 的相邻布局斜率变化，并检查二者斜率之和在 32×64 的 vec2→4→8 上是否增加、
在 32×128 上是否较平坦。完整 kernel 的参照仅使用已归档的阶段 B H100/host/BF16 数据。
不重新测完整 kernel，不按结果挑选 K、布局或拟合任意权重。

ML 双缓冲、MR 私有更新/转换/结果累加，以及两者资源与原 kernel 的模板差异全部报告。
特别核对私有更新是否让原先可以移动到 widening 之前的 BF16 packed max 变成 FP32 max。
输出布局转换不属于这次 MR，组合斜率只是待验证假设，不能据此宣称模型已成立。
如果未保留真实动态重复，或合法程序无法表达预期干预，保留结果并停下汇报。

原始证据与失败记录 write-once，技术修订逐次冻结；完成 C1 后单独 commit 并向用户汇报。
