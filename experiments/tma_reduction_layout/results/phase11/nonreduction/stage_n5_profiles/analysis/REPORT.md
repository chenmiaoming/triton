# N5 较大 footprint 的硬件计数（warm application）

本阶段沿用 N5 已冻结的两个大 case、U/G0 与各自唯一候选，只验证流量、请求分组及缓存行为。共六份报告，每份一个完整 grid 的冻结 CUBIN；不读取正式计时，不重新挑选布局。

`application replay / cache none / clock none`，每轮八次相同程序 warmup；warmup/target 前恢复原地输入，恢复位于 profiler 范围之外。全部 replay 的完整 oracle、guard、UUID、CUBIN SHA、运行参数及 CUDA 驻留容量已验证。

**未采集 InstructionStats**：没有 CTA0 软件指令表，也不伪造 U/G0 软件计数一致性检验。`smsp__inst_executed.sum` 仅作为实际硬件指标保留。

协议 SHA256：`9c95900ff5202b553b1fde1c1e8153656b4838ac997b8b8c9fd20b468440eaef`；payload SHA256：`7fcc58f9e4c163a97b69abbaf0d0ebb8791be9bf33b0024e8c38c938447a3805`。

| Case | Variant | CTA | REG / CUDA q | L2 hit (%) | DRAM read (MiB) | DRAM write (MiB) | Global load sectors/CTA | Global store sectors/CTA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N01_m4096_n8192 | U | 65536 | 14 / 16 | 91.5400 | 32.0818 | 33.5688 | 128.0000 | 64.0000 |
| N01_m4096_n8192 | G0 | 65536 | 14 / 16 | 91.5600 | 32.0845 | 34.0591 | 128.0000 | 64.0000 |
| N01_m4096_n8192 | GL_coordinated_fully_coalesced | 65536 | 12 / 16 | 89.3400 | 32.0139 | 33.3772 | 128.0000 | 16.0000 |
| N02_k8192_p1024_g128 | U | 8192 | 72 / 7 | 79.6600 | 33.2642 | 126.3135 | 160.0000 | 512.0000 |
| N02_k8192_p1024_g128 | G0 | 8192 | 72 / 7 | 79.6200 | 33.2668 | 126.2439 | 160.0000 | 512.0000 |
| N02_k8192_p1024_g128 | GL_packed_compute_vec1 | 8192 | 40 / 12 | 79.5900 | 33.2639 | 126.4136 | 160.0000 | 512.0000 |

表中 MiB 仅为原始 byte counter 的显示换算；全部 26 项指标的原始名称、单位及数值保留在 summary.json/metrics_long.csv。只有 additive 流量/指令 sum 按实际 CTA 数归一，百分比及比值不归一。

L2 命中率是缓存层汇总，不能单凭一个比例断言所有 buffer 均驻留 L2。DRAM 流量、全局请求/sector、输出写策略与不同 footprint 必须共同解释；这些单次 profiler 数据不替代无 profiler 的正式计时。

## 采集与完整性

| Group | 报告数 | GPU UUID | 全部 replay 检查 |
| --- | --- | --- | --- |
| N5_N01_scale_footprint | 3 | GPU-d09facb1-64d5-98a4-4311-a87b4baa9208 | True |
| N5_N02_awq_footprint | 3 | GPU-d194cf1b-e85c-5560-1574-caca10564b6e | True |
