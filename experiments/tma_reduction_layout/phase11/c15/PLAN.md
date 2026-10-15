# C1.5 — BF16 状态更新与稳态区间斜率

用户仅批准在当前 8 个布局上修正探针、验证 packed-max lowering 并汇报，再决定 C2。
沿用 H100、host-created descriptor、BF16 输入、固定 upstream 20d9 和原矩阵：
32×64/w8 与 32×128/w4，各有 N-first vec1/2/4/8。

MR 每轮先在 BF16 状态上做私有步进；step、threshold、wrap 均为 BF16。
更新后的输入再转换 FP32，仅由 max 消费。结果仅累加到原生 SliceLayout 寄存器，
不反馈给输入、不添加反向 shared-memory 广播。该改动也使 reference 的顺序变为先更新再 max。

代码生成门要求 GLIR 中完整输入的 widening 为 single-use，私有 add/sub 为 BF16，
8 个布局的 PTX max opcode 类型和静态数量均与原阶段 B 完整 kernel 一致。
32×64/vec8 每线程每列只有一个值，原本没有线程内 BF16 combine；不强行插入 BF16 max。

仅新编译 8 个 MR；8 个 ML 逐字节复用原 C1 CUBIN、compiler artifacts 和对应的 64 份 NCU 报告。
MR 的 quarter、确定性随机 BF16、有限边界三种输入及四个 K 用 CPU reference 验证完整输出，
共 96 项新检查；原 ML 的 96 项完整输出检查保留来源绑定。
无 native 重编译，继续使用已有 Modal 镜像、kernel cache 和 `chenmiaoming` profile。

新 MR 使用 K=1/16/64/256 的同一 CUBIN，采集 64 份硬件/软件 NCU 报告。
沿用 C1 的函数偏移 PC、展开/尾部副本分组方法，要求循环目标 opcode 的动态计数严格 K 倍，
初始化与最终输出的固定 PC 单独记录；无 spill。该门通过后才开始计时。

三个新 worker 在同一轮中测量全部 16 个 CUBIN，grid=1 和 64×实际 SM 数，
每 cell 固定 3 次预热、20 个单 kernel CUDA event 样本，不 JIT、不使用 profiler duration。
因此 ML/MR 组合使用同一批新 worker，未把新旧 cohort 任意配对。

所有布局和两类 fixture 的主斜率统一拟合 K=16/64/256。
单独报告 K1 实测时间 T1、固定开销参考 T1−b、稳态拟合截距 a，
以及 16→64、64→256 两段增量、残差和三个 worker 的范围。
T1 包含一次 body，不能称为纯截距或纯 HBM/TMA 延迟；排除 K1 不自动证明其余点完全线性。
旧 C1 数据的同区间重拟合只写入 C1.5 的 derived output，禁止改写 C1。

对用户提出的全图 CTA 缩放和 45 µs HBM 底座做运行前算术核对；
倍率成立，但旧 C1 在 N128 上算出的缩短为 5.60%，并非低于 3%。
保留 CTA 线性放大和固定底座可加性为待验证假设，本轮不增加完整 kernel 或 TMA-only 实验。
不按结果调整底座、权重、布局或 K。

原始样本、compiler artifacts、NCU reports 和失败记录 write-once。
完成本阶段后独立 commit、运行最终 validator 并汇报，停在 C2 之前。
