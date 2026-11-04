# Stage N3 机制复核：非规约 kernel 的多阶段布局约束

本复核限定于固定编译器 `20d9d98fead78c2f65e1b3ba5a7881d86a836dad`、H100、预注册几何与工作集。N3 静态预选覆盖 7 个子类型、60/60 份报告；7/7 的 U/G0 独立软件采集具有相同 opcode 直方图。既有 ELF、ABI、资源和正确性门已通过。

证据为 `analysis/summary.json`（SHA256 `abc7eac8f6f363c586d1af8bcab0779ff17bad0849a6619d780ff20c6e434537`）及其指向的原始报告、binding、raw/source 导出。NCU 为 2024.3.2.0；后验 N3b 不并入上述分母。

## 计数口径

指令列是 CTA0 的动态 **warp 指令**，BAR 数不等于 CTA 同步阶段数。sector 为 32B 请求单位；L1/TEX sectors、shared wavefront 和 DRAM bytes 不可互替。软件单 CTA 与全图硬件采集分开解释，不采用 profiler 时间评价加速。[NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

`q` 是 driver 返回的最大驻留 CTA 容量，不是 achieved occupancy；寄存器减少或容量提高不保证等比例加速。[CUDA Best Practices：Occupancy](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-best-practices-guide/index.html#occupancy)

## 主要对照

下表为 G0→指定候选。指令、交换及 sector 按 CTA 统计；完整对照见既有 REPORT。

| 子类型 / 对照 | 总 warp 指令 | LDS/STS/BAR | Global LD/ST sectors | n_regs；q |
|---|---|---|---|---|
| N01 coordinated_fully_coalesced | 128→136 | 始终 0/0/0 | 128/64→128/16 | 14→12；16→16 |
| N02 packed_compute_vec1 | 2168→1928 | 36/36/68→4/4/4 | 始终 160/512 | 72→40；7→12 |
| Flash RoPE input_vec1 | 660→1060 | 0/0/0→16/16/12 | 始终 192/128 | 32→34；16→12 |
| MRoPE Q_vec1 | 764→1440 | 0/0/0→24/80/28 | 始终 412/320 | 40→44；12→10 |
| FP8 pack v_input_vec2 | 12288→13248 | 32/32/8→32/32/24 | 3096/768→2584/768 | 71→69；3→3 |
| KDA gate feature_vec2 | 2264→2792 | 16/16/8→32/32/24 | 始终 392/512 | 48→48；5→5 |
| KV cache e2_heads1 | 320→536 | 始终 0/0/0 | 68/64→132/64 | 20→26；16→16 |

**N01：输出请求分组与中间转换存在权衡。** 只改输出也将 store sectors 64→16，却新增 LDS/STS/BAR 各 4，总指令达 180；协调输入输出保留 16 sectors，并消除该交换。bank-conflict 均为 0、q 均为 16。warm DRAM read 为 0，不能称为 HBM 带宽收益。访存合并应依据实际地址与参与线程，而非 vec 倒数。[CUDA Best Practices：Coalescing](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

**N02：较窄 load 不等于更多实际流量。** load 指令/请求 16→40，load sectors 却仍为 160；shared-load wavefront 154.045→16.226、store wavefront 133.327→4，且寄存器和容量同时变化。不能将其中任一项单独换算为纳秒。小几何只有 128 CTA/132 SM，无法用容量 7→12 单独解释其收益。

**无收益对照限定结论。** RoPE/MRoPE 默认无共享交换，候选反而引入交换；pack 的 L1 sectors 减少时 DRAM read 仍约 49,171 B/CTA；KV cache 的候选也未减少交换。KV 候选软件 STG=30，而硬件 store requests/CTA=16，进一步说明两种计数不可互替；本轮未将差异归因为指令融合。没有证据表明所有候选算子都存在布局缺陷。

## Binary 桥接与性能边界

已从共同 payload 直接比较 CUBIN 字节、ABI、标量 specialization 和资源：N01 两尺寸的 U/G0/coordinated，以及 N02 两尺寸的 U/G0/packed_compute_vec1，分别完全一致；两者 CTA0 均为完整内部 tile。此桥接支持相同 binary 的软件机制，不允许跨 grid 照搬 cache/stall 百分比。

| 家族 | 对比几何 | 既有 N4 固定选择结果 |
|---|---|---|
| N01 | 4096×128 / 11008×128 | 小尺寸 vs U +2.47%，未过严格双基线门；大尺寸 +5.75%，通过 |
| N02 | k1024/p128 / k3584/p448 | vs U +4.22% / +6.42%，均通过严格双基线门 |

N05 exact feature_vec4 由单列 N3b 诊断，仍保留其 N4 边界状态。这些证据支持**非 TMA、非规约数据流也存在所有权、请求分组与转换/资源的耦合**；不证明跨工作集、跨设备或完整模型的普遍收益，不验证统一代价模型。由于没有版本升级对照，这里称默认布局的改进空间，不称新的性能回归。

本文件仅为新增复核；原始证据、summary、REPORT 未改动，未读取后续 N5 数据。
