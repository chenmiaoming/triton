# H100 同程序、同尺寸时间变化：未锁频诊断

全部观测执行同一份冻结 CUBIN、同一 Loaded function、相同输入、参数、shape 和单目标 graph。A/B 只是独立随机观测标签。仅改变正式计时事件之外的 guard、空闲或预负载条件；本轮不比较布局收益或跨尺寸差异。

固定 case：`N02_k1024_p128_g128`。所有 GPU 保持未锁频；本轮不能单独证明 DVFS 的因果作用。

完整样本：3 workers，24 cells，8,928 个原始计时。
每个 visit 先取 31 个样本的中位数，再对 worker × visit 均值。条件主估计对独立 A/B 观测等权平均。8192 次分层 bootstrap 在全部八个 cell 间共用重采样索引；区间为逐项 95% percentile CI。

| Condition | Alias | 时间 µs [95% CI] | Worker 均值范围 µs | Visit 中位数范围 µs |
|---|---|---:|---:|---:|
| fast_gap | A | 5.976 [5.836, 6.070] | [5.835, 6.069] | [5.824, 6.240] |
| fast_gap | B | 5.975 [5.841, 6.078] | [5.840, 6.083] | [5.824, 6.208] |
| idle_20ms | A | 7.515 [6.697, 8.202] | [6.677, 8.179] | [6.336, 10.176] |
| idle_20ms | B | 7.543 [6.625, 8.237] | [6.611, 8.221] | [6.336, 9.568] |
| idle_then_same_prime | A | 5.970 [5.892, 6.056] | [5.891, 6.061] | [5.856, 6.208] |
| idle_then_same_prime | B | 5.973 [5.868, 6.095] | [5.867, 6.099] | [5.824, 6.144] |
| idle_then_disjoint_prime | A | 6.011 [5.943, 6.091] | [5.941, 6.096] | [5.856, 6.240] |
| idle_then_disjoint_prime | B | 6.017 [5.947, 6.099] | [5.949, 6.104] | [5.888, 6.240] |

A/B sham：同一程序的两组独立观测。B 相对 A 时间降低为 100×(A−B)/A，正值代表 B 更短；practically_equivalent 要求整个区间落入 ±3%。分类名 practical_improvement/regression 仅表示时间方向，不能解释为代码或布局改进。

| Condition | B−A µs [95% CI] | B 相对 A 时间降低 % [95% CI] | ±3% 分类 |
|---|---:|---:|---|
| fast_gap | -0.001 [-0.020, 0.016] | 0.015 [-0.267, 0.325] | practically_equivalent |
| idle_20ms | 0.028 [-0.085, 0.149] | -0.379 [-1.928, 1.200] | practically_equivalent |
| idle_then_same_prime | 0.004 [-0.025, 0.038] | -0.060 [-0.635, 0.422] | practically_equivalent |
| idle_then_disjoint_prime | 0.006 [-0.009, 0.022] | -0.104 [-0.369, 0.149] | practically_equivalent |

下表比较同一程序在不同样本间条件下的 kernel event 时间。差分为 condition−fast_gap；时间降低为 100×(fast−condition)/fast。两种口径符号相反。

| Condition | 时间 µs [95% CI] | condition−fast µs [95% CI] | 相对 fast 时间降低 % [95% CI] |
|---|---:|---:|---:|
| fast_gap | 5.976 [5.838, 6.073] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| idle_20ms | 7.529 [6.658, 8.219] | 1.553 [0.581, 2.349] | -25.995 [-39.999, -9.566] |
| idle_then_same_prime | 5.972 [5.880, 6.076] | -0.004 [-0.056, 0.040] | 0.067 [-0.685, 0.924] |
| idle_then_disjoint_prime | 6.014 [5.946, 6.095] | 0.038 [-0.015, 0.104] | -0.640 [-1.789, 0.251] |

全部区间直接来自共用重采样中的配对差分，未相减两个独立区间。以下保留每 worker 的匹配条件差分，不能当作额外独立 GPU。

| Condition | Worker | condition−fast µs | 相对 fast 时间降低 % |
|---|---|---:|---:|
| fast_gap | 0 | 0.000 | 0.000 |
| fast_gap | 1 | 0.000 | 0.000 |
| fast_gap | 2 | 0.000 | 0.000 |
| idle_20ms | 0 | 2.363 | -40.475 |
| idle_20ms | 1 | 0.568 | -9.348 |
| idle_20ms | 2 | 1.729 | -28.758 |
| idle_then_same_prime | 0 | 0.041 | -0.708 |
| idle_then_same_prime | 1 | 0.004 | -0.066 |
| idle_then_same_prime | 2 | -0.057 | 0.953 |
| idle_then_disjoint_prime | 0 | 0.108 | -1.850 |
| idle_then_disjoint_prime | 1 | 0.024 | -0.395 |
| idle_then_disjoint_prime | 2 | -0.017 | 0.288 |

每 worker / alias / condition 的原始样本 min/p05/median/p95/max 与 visit 中位数范围保存于 `analysis.json` 的 audit；guard、host gaps、launch/completion timestamp 和 graph node 类型单列，不加入 kernel event 时间。

遥测按 worker / phase / case / condition 汇总。NVML 查询返回码、缺失值、采样跨度、SM/graphics/memory clocks、功率、温度、利用率与 clock-event mask 均保留。

| Worker | Phase | Case | Condition | Samples | SM MHz median [min,max] |
|---|---|---|---|---:|---:|
| 0 | after | N02_k1024_p128_g128 | idle_then_same_prime | 4 | 1980.0 [1980.0,1980.0] |
| 0 | prewarm | N02_k1024_p128_g128 | — | 200 | 1980.0 [1980.0,1980.0] |
| 0 | timing | N02_k1024_p128_g128 | fast_gap | 3 | 1980.0 [1980.0,1980.0] |
| 0 | timing | N02_k1024_p128_g128 | idle_20ms | 1523 | 1980.0 [1980.0,1980.0] |
| 0 | timing | N02_k1024_p128_g128 | idle_then_disjoint_prime | 1534 | 1980.0 [1980.0,1980.0] |
| 0 | timing | N02_k1024_p128_g128 | idle_then_same_prime | 1541 | 1980.0 [1980.0,1980.0] |
| 0 | warmup | N02_k1024_p128_g128 | idle_20ms | 202 | 1980.0 [1980.0,1980.0] |
| 0 | warmup | N02_k1024_p128_g128 | idle_then_disjoint_prime | 200 | 1980.0 [1980.0,1980.0] |
| 0 | warmup | N02_k1024_p128_g128 | idle_then_same_prime | 199 | 1980.0 [1980.0,1980.0] |
| 1 | after | N02_k1024_p128_g128 | fast_gap | 3 | 1980.0 [1980.0,1980.0] |
| 1 | block_guard | N02_k1024_p128_g128 | idle_20ms | 1 | 1980.0 [1980.0,1980.0] |
| 1 | prewarm | N02_k1024_p128_g128 | — | 200 | 1980.0 [1980.0,1980.0] |
| 1 | timing | N02_k1024_p128_g128 | fast_gap | 1 | 1980.0 [1980.0,1980.0] |
| 1 | timing | N02_k1024_p128_g128 | idle_20ms | 1536 | 1980.0 [1980.0,1980.0] |
| 1 | timing | N02_k1024_p128_g128 | idle_then_disjoint_prime | 1536 | 1980.0 [1980.0,1980.0] |
| 1 | timing | N02_k1024_p128_g128 | idle_then_same_prime | 1535 | 1980.0 [1980.0,1980.0] |
| 1 | warmup | N02_k1024_p128_g128 | fast_gap | 1 | 1980.0 [1980.0,1980.0] |
| 1 | warmup | N02_k1024_p128_g128 | idle_20ms | 200 | 1980.0 [1980.0,1980.0] |
| 1 | warmup | N02_k1024_p128_g128 | idle_then_disjoint_prime | 202 | 1980.0 [1980.0,1980.0] |
| 1 | warmup | N02_k1024_p128_g128 | idle_then_same_prime | 201 | 1980.0 [1980.0,1980.0] |
| 2 | after | N02_k1024_p128_g128 | fast_gap | 4 | 1980.0 [1980.0,1980.0] |
| 2 | block_guard | N02_k1024_p128_g128 | fast_gap | 1 | 1980.0 [1980.0,1980.0] |
| 2 | block_guard | N02_k1024_p128_g128 | idle_20ms | 1 | 1980.0 [1980.0,1980.0] |
| 2 | prewarm | N02_k1024_p128_g128 | — | 201 | 1980.0 [1980.0,1980.0] |
| 2 | timing | N02_k1024_p128_g128 | fast_gap | 4 | 1980.0 [1980.0,1980.0] |
| 2 | timing | N02_k1024_p128_g128 | idle_20ms | 1537 | 1980.0 [1980.0,1980.0] |
| 2 | timing | N02_k1024_p128_g128 | idle_then_disjoint_prime | 1537 | 1980.0 [1980.0,1980.0] |
| 2 | timing | N02_k1024_p128_g128 | idle_then_same_prime | 1539 | 1980.0 [1980.0,1980.0] |
| 2 | warmup | N02_k1024_p128_g128 | idle_20ms | 199 | 1980.0 [1980.0,1980.0] |
| 2 | warmup | N02_k1024_p128_g128 | idle_then_disjoint_prime | 200 | 1980.0 [1980.0,1980.0] |
| 2 | warmup | N02_k1024_p128_g128 | idle_then_same_prime | 198 | 1980.0 [1980.0,1980.0] |

另在 `telemetry_associations` 按记录的 visit 时间窗匹配 NVML 样本，报告各 worker 内 condition−fast 时间差与遥测差的描述性关联。缺采样窗保留为空，不插值，不把相关性解释为 DVFS 因果证据。NVML telemetry 包含 host gaps，且功率/利用率自身具有硬件平均窗；clock-event mask 次数不是限频持续时间。

- All measurements are unlocked. Sampled NVML clocks do not establish fixed frequency or per-kernel frequency and cannot isolate a causal DVFS effect.
- NVML power/utilization readings have hardware averaging windows that may extend beyond a visit/context label; query timestamps do not make them instantaneous per-kernel measurements.
- Between-sample guard, idle and preload interventions may affect several mechanisms including cache state, temperature, frequency and scheduling. Kernel timing events exclude their host/preload duration.
- Three workers provide limited evidence about the platform population. Repeated within-worker observations are not independent GPU allocations.
- A failed A/B sham equivalence check remains a reported result; no worker or condition is dropped to obtain a favorable outcome.
- Raw-sample, visit and worker ranges/quantiles are descriptive variability measures; extreme-value ranges are not confidence intervals.
- The disjoint primer has independent allocations and a different frozen CUBIN but exercises similar memory/computation resources. It can affect shared cache state as well as GPU execution state; it is not a cache-free intervention.
- A recovery percentage is an unstable ratio when idle-minus-fast approaches or crosses zero. Its interval is omitted unless every bootstrap denominator is positive; no draws are discarded and no recovery outcome is required.

## 自适应 wake-up 跟进的额外配对对照

这一轮在首轮结果之后设计，独立采集，不与首轮样本混算。四种条件均计时同一目标 graph；primer 位于目标事件外。same-prime 使用目标程序和目标数据，disjoint-prime 使用另一冻结 CUBIN 与独立分配的数据。

| Condition | Reference | condition−reference µs [95% CI] | 时间降低 % [95% CI] |
|---|---|---:|---:|
| idle_then_same_prime | idle_20ms | -1.557 [-2.325, -0.577] | 20.685 [8.659, 28.309] |
| idle_then_disjoint_prime | idle_20ms | -1.515 [-2.264, -0.555] | 20.124 [8.335, 27.562] |
| idle_then_same_prime | idle_then_disjoint_prime | -0.042 [-0.073, -0.018] | 0.702 [0.291, 1.227] |

恢复比例为 100×(idle−primed)/(idle−fast)，允许小于 0% 或超过 100%；它不是额外的独立观察。所有分子与分母直接共用同一 bootstrap 重采样。若任何重采样的 idle−fast 分母不为正，保留点值与分母诊断，但不报告恢复比例区间，也不删除这些抽样。

| Primer condition | 恢复比例 % [95% CI] | 正分母抽样比例 |
|---|---:|---:|
| idle_then_same_prime | 100.258 [97.787, 103.322] | 1.000000 |
| idle_then_disjoint_prime | 97.539 [94.527, 100.904] | 1.000000 |

若 disjoint-prime 也恢复目标时间，只能说明精确重用目标输入并非恢复的必要条件。两种 primer 都会改变 GPU 与缓存的运行历史，不能由这一对照独断地判定 DVFS、clock gating、调度或 event 时间路径中的哪一种是原因。采样时钟仍不能充当每个 kernel 的瞬时频率证明。
