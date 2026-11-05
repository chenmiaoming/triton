# H100 同程序、同尺寸时间变化：未锁频诊断

全部观测执行同一份冻结 CUBIN、同一 Loaded function、相同输入、参数、shape 和单目标 graph。A/B 只是独立随机观测标签。仅改变正式计时事件之外的 guard、空闲或预负载条件；本轮不比较布局收益或跨尺寸差异。

固定 case：`N02_k1024_p128_g128`。所有 GPU 保持未锁频；本轮不能单独证明 DVFS 的因果作用。

完整样本：3 workers，24 cells，8,928 个原始计时。
每个 visit 先取 31 个样本的中位数，再对 worker × visit 均值。条件主估计对独立 A/B 观测等权平均。8192 次分层 bootstrap 在全部八个 cell 间共用重采样索引；区间为逐项 95% percentile CI。

| Condition | Alias | 时间 µs [95% CI] | Worker 均值范围 µs | Visit 中位数范围 µs |
|---|---|---:|---:|---:|
| fast_gap | A | 5.784 [5.622, 6.006] | [5.621, 6.008] | [5.504, 6.048] |
| fast_gap | B | 5.783 [5.621, 5.998] | [5.619, 6.000] | [5.472, 6.016] |
| legacy_gap | A | 5.786 [5.642, 5.985] | [5.643, 5.987] | [5.472, 6.016] |
| legacy_gap | B | 5.793 [5.646, 6.003] | [5.648, 6.005] | [5.504, 6.048] |
| idle_20ms | A | 6.964 [6.115, 7.904] | [6.107, 7.917] | [5.920, 8.128] |
| idle_20ms | B | 6.911 [6.029, 7.920] | [6.024, 7.933] | [5.920, 8.160] |
| gpu_preload | A | 5.796 [5.638, 5.993] | [5.637, 5.995] | [5.472, 6.016] |
| gpu_preload | B | 5.815 [5.675, 5.993] | [5.672, 5.995] | [5.536, 6.048] |

A/B sham：同一程序的两组独立观测。B 相对 A 时间降低为 100×(A−B)/A，正值代表 B 更短；practically_equivalent 要求整个区间落入 ±3%。分类名 practical_improvement/regression 仅表示时间方向，不能解释为代码或布局改进。

| Condition | B−A µs [95% CI] | B 相对 A 时间降低 % [95% CI] | ±3% 分类 |
|---|---:|---:|---|
| fast_gap | -0.001 [-0.018, 0.015] | 0.015 [-0.264, 0.302] | practically_equivalent |
| legacy_gap | 0.007 [-0.007, 0.024] | -0.123 [-0.410, 0.123] | practically_equivalent |
| idle_20ms | -0.052 [-0.165, 0.044] | 0.753 [-0.595, 2.481] | practically_equivalent |
| gpu_preload | 0.020 [-0.002, 0.042] | -0.337 [-0.733, 0.030] | practically_equivalent |

下表比较同一程序在不同样本间条件下的 kernel event 时间。差分为 condition−fast_gap；时间降低为 100×(fast−condition)/fast。两种口径符号相反。

| Condition | 时间 µs [95% CI] | condition−fast µs [95% CI] | 相对 fast 时间降低 % [95% CI] |
|---|---:|---:|---:|
| fast_gap | 5.784 [5.622, 6.003] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| legacy_gap | 5.789 [5.644, 5.995] | 0.006 [-0.013, 0.032] | -0.100 [-0.568, 0.221] |
| idle_20ms | 6.937 [6.072, 7.915] | 1.154 [0.348, 1.910] | -19.949 [-31.807, -6.073] |
| gpu_preload | 5.805 [5.656, 5.993] | 0.022 [-0.010, 0.055] | -0.377 [-0.974, 0.163] |

全部区间直接来自共用重采样中的配对差分，未相减两个独立区间。以下保留每 worker 的匹配条件差分，不能当作额外独立 GPU。

| Condition | Worker | condition−fast µs | 相对 fast 时间降低 % |
|---|---|---:|---:|
| fast_gap | 0 | 0.000 | 0.000 |
| fast_gap | 1 | 0.000 | 0.000 |
| fast_gap | 2 | 0.000 | 0.000 |
| legacy_gap | 0 | -0.000 | 0.000 |
| legacy_gap | 1 | 0.025 | -0.451 |
| legacy_gap | 2 | -0.008 | 0.133 |
| idle_20ms | 0 | 0.339 | -5.914 |
| idle_20ms | 1 | 1.201 | -21.376 |
| idle_20ms | 2 | 1.921 | -32.001 |
| gpu_preload | 0 | 0.040 | -0.698 |
| gpu_preload | 1 | 0.035 | -0.617 |
| gpu_preload | 2 | -0.009 | 0.155 |

每 worker / alias / condition 的原始样本 min/p05/median/p95/max 与 visit 中位数范围保存于 `analysis.json` 的 audit；guard、host gaps、launch/completion timestamp 和 graph node 类型单列，不加入 kernel event 时间。

遥测按 worker / phase / case / condition 汇总。NVML 查询返回码、缺失值、采样跨度、SM/graphics/memory clocks、功率、温度、利用率与 clock-event mask 均保留。

| Worker | Phase | Case | Condition | Samples | SM MHz median [min,max] |
|---|---|---|---|---:|---:|
| 0 | after | N02_k1024_p128_g128 | idle_20ms | 1 | 1980.0 [1980.0,1980.0] |
| 0 | block_guard | N02_k1024_p128_g128 | gpu_preload | 1 | 1980.0 [1980.0,1980.0] |
| 0 | prewarm | — | — | 201 | 1980.0 [1980.0,1980.0] |
| 0 | timing | N02_k1024_p128_g128 | fast_gap | 1 | 1980.0 [1980.0,1980.0] |
| 0 | timing | N02_k1024_p128_g128 | gpu_preload | 23 | 1980.0 [1980.0,1980.0] |
| 0 | timing | N02_k1024_p128_g128 | idle_20ms | 1533 | 1980.0 [1980.0,1980.0] |
| 0 | timing | N02_k1024_p128_g128 | legacy_gap | 8 | 1980.0 [1980.0,1980.0] |
| 0 | warmup | N02_k1024_p128_g128 | gpu_preload | 2 | 1980.0 [1980.0,1980.0] |
| 0 | warmup | N02_k1024_p128_g128 | idle_20ms | 198 | 1980.0 [1980.0,1980.0] |
| 1 | after | N02_k1024_p128_g128 | fast_gap | 1 | 1980.0 [1980.0,1980.0] |
| 1 | block_guard | N02_k1024_p128_g128 | gpu_preload | 1 | 1980.0 [1980.0,1980.0] |
| 1 | prewarm | — | — | 201 | 1980.0 [1980.0,1980.0] |
| 1 | timing | N02_k1024_p128_g128 | fast_gap | 1 | 1980.0 [1980.0,1980.0] |
| 1 | timing | N02_k1024_p128_g128 | gpu_preload | 23 | 1980.0 [1980.0,1980.0] |
| 1 | timing | N02_k1024_p128_g128 | idle_20ms | 1535 | 1980.0 [1980.0,1980.0] |
| 1 | timing | N02_k1024_p128_g128 | legacy_gap | 4 | 1980.0 [1980.0,1980.0] |
| 1 | warmup | N02_k1024_p128_g128 | gpu_preload | 4 | 1980.0 [1980.0,1980.0] |
| 1 | warmup | N02_k1024_p128_g128 | idle_20ms | 199 | 1980.0 [1980.0,1980.0] |
| 1 | warmup | N02_k1024_p128_g128 | legacy_gap | 2 | 1980.0 [1980.0,1980.0] |
| 2 | after | N02_k1024_p128_g128 | fast_gap | 2 | 1980.0 [1980.0,1980.0] |
| 2 | block_guard | N02_k1024_p128_g128 | fast_gap | 2 | 1980.0 [1980.0,1980.0] |
| 2 | prewarm | — | — | 201 | 1980.0 [1980.0,1980.0] |
| 2 | timing | N02_k1024_p128_g128 | fast_gap | 1 | 1980.0 [1980.0,1980.0] |
| 2 | timing | N02_k1024_p128_g128 | gpu_preload | 24 | 1980.0 [1980.0,1980.0] |
| 2 | timing | N02_k1024_p128_g128 | idle_20ms | 1543 | 1980.0 [1980.0,1980.0] |
| 2 | timing | N02_k1024_p128_g128 | legacy_gap | 7 | 1980.0 [1980.0,1980.0] |
| 2 | warmup | N02_k1024_p128_g128 | gpu_preload | 3 | 1980.0 [1980.0,1980.0] |
| 2 | warmup | N02_k1024_p128_g128 | idle_20ms | 199 | 1980.0 [1980.0,1980.0] |
| 2 | warmup | N02_k1024_p128_g128 | legacy_gap | 2 | 1980.0 [1980.0,1980.0] |

另在 `telemetry_associations` 按记录的 visit 时间窗匹配 NVML 样本，报告各 worker 内 condition−fast 时间差与遥测差的描述性关联。缺采样窗保留为空，不插值，不把相关性解释为 DVFS 因果证据。NVML telemetry 包含 host gaps，且功率/利用率自身具有硬件平均窗；clock-event mask 次数不是限频持续时间。

- All measurements are unlocked. Sampled NVML clocks do not establish fixed frequency or per-kernel frequency and cannot isolate a causal DVFS effect.
- NVML power/utilization readings have hardware averaging windows that may extend beyond a visit/context label; query timestamps do not make them instantaneous per-kernel measurements.
- Between-sample guard, idle and preload interventions may affect several mechanisms including cache state, temperature, frequency and scheduling. Kernel timing events exclude their host/preload duration.
- Three workers provide limited evidence about the platform population. Repeated within-worker observations are not independent GPU allocations.
- A failed A/B sham equivalence check remains a reported result; no worker or condition is dropped to obtain a favorable outcome.
- Raw-sample, visit and worker ranges/quantiles are descriptive variability measures; extreme-value ranges are not confidence intervals.
