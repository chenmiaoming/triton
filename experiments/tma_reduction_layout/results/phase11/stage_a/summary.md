# Phase 11 Stage A — NCU coverage

工具校准完成；没有正式性能样本，也没有重建或加载Triton native。所有早期源码/工具失败保留。

| Target | 普通LDS动态warp指令 | 无冲突wavefront | 32路冲突wavefront / conflict | LDSM动态warp指令 | LDSM wavefront / conflict |
|---|---:|---:|---|---:|---|
| sm90 | 128 | 128.0 | 4096.0 / 3968.0 | 128 | 512.0 / 0.0 |
| sm100 | 128 | 128.0 | 4096.0 / 3968.0 | 128 | 512.0 / 0.0 |
| sm120 | 128 | 128.0 | 4096.0 / 3968.0 | 128 | 512.0 / 0.0 |

每个fixture固定128个单warp CTA，输出已验证。LDSM的wavefront覆盖单独检查；无冲突LDSM不能验证有冲突LDSM的bank计数。直接shared Requests硬件指标未提供，后续使用NCU实际测得的每PC warp指令数并明确口径，不能用32/vec代替。

NCU 2024.3不识别SM100/SM120；复用已有CUDA13.1 / NCU2025.4工具镜像修复工具覆盖。每次失败、工具版本、query-metrics、query-sections、Driver UUID、PCI映射、完整命令、PTX/CUBIN/SASS、原始ncu-rep及独立report-import均归档。硬件计数与software指令插桩分开采集。

阶段B继续复用固定upstream的baseline native；工具校准不构成布局性能或回归因果结论。
