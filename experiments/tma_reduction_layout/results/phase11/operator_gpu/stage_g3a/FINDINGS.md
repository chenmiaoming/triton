# G3a：新增证据说明了什么

这一批补的是默认布局的优化机会及其解释，不是版本性能回归。正式时间全部复用 G2，新增 60 份硬件报告，无新编译、无按结果重计时。

## 1. 确实需要同时观察 request 与 sector

vLLM `s16_d128`，G0→vec1：global-load requests 从 44.5 增至 92.5 / CTA，但 sectors 都是 292.5 / CTA。较窄的访问增加了指令/请求数，却没有增加同一层次的 sector 请求总量。不能从逻辑 vec 或指令数量直接推导流量。

同一对照的 SHFL 为 64→32，BAR.SYNC 为 52→12，shared-load WF 为 72→40.43；G2 的 merge 耗时为 6.377→5.961 µs（相对 G0 +6.52%）。这支持通信变化与收益相关，但没有给每项通信分配独立延迟。

## 2. 减少 SHFL 和 stall 比例仍可能变慢

SGLang `d64`，G0→vec1：SHFL 1664→672，但 BAR 280→448、shared-load WF 337.14→1664.50，global-load requests 400→1296 / CTA，而 sectors 仍为 4252 / CTA。寄存器 56→79，容量 q 9→6，实测 occupancy 46.99%→29.94%。G2 两阶段 decode 为 54.953→68.320 µs，即相对 G0 慢 24.33%。

这个较慢变体的 issue-active 比例反而 30.86%→34.37%，barrier 比例 17.43%→14.44%，long-scoreboard 67.89%→58.21%。比例下降和发射率提高不能替代绝对时间：它还执行了更多工作。

## 3. `d128 vec4` 的收益不能只归因于“通信全面减少”

新增该候选专门的报告后，G0→vec4 的 SHFL 是 1984→1472，但 BAR 为 600→664、shared-load WF 为 944.20→1456.17，regs 为 80→96、q 为 6→5。global sectors 均为 8468 / CTA，global-load requests 为 784→1040 / CTA。

单 CTA 动态 FADD 为 6280→5768，与较少的规约合并工作一致；同时存在更多地址、加载、转换或同步工作。G2 的完整 decode 仍有 +4.05% 的纯 Gluon 对照收益。这是需要进一步拆解依赖链/子图组合的点，不能据此宣布逐项时间机制已闭合，也不能直接把各类指令无权相加。

这个 batch8/context2048 点相对真实 U 的收益只有 +1.70%，未超过 3% 实用门槛。G2 在 context256 下的对应收益是 +5.65% vs U / +4.26% vs G0，但本轮 context2048 的 NCU 计数不能直接当作 context256 的逐项机制证明。

## 4. 缓存对照把两类实际环境分开了

- C06 `s16_d128` 的 G0 在 application replay + 自身预热下，DRAM read 为 0，L2 hit 为 99.94%；冷缓存 kernel replay 的 read 为约 2.14 MB、L2 hit 为 32.79%。该 merge 本身主要访问缓存中的部分结果，不能套用大型 HBM 流式矩阵的屋顶。
- C02 `d128` 的两种 cache 设置都读取约 268.7 MB DRAM，G0 的 L2 hit 约 1%–2%，与此工作负载的较大 KV footprint 一致。
- 少量 C06 L2 hit 读数略高于 100%（例如 100.24%），原样保留。小 kernel、多 pass 的比值读数存在采集误差；不截断、不将它当成物理上超过 100% 的命中率，也不据此排除案例。

缓存对照没有额外做非 profiler 正式计时，因此只检验计数及状态解释，不能报告冷/热缓存的独立时间收益。

## 5. 官方资料与边界

与本轮判断直接相关的官方定义和方法见 [NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)、[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)、[Compute Triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html)。本轮查阅日期为 2026-10-21。

这些文档支持区分请求、波前、sector、依赖等待及实际发射；不提供足以把它们直接相加得到整个 Gluon 子图时间的通用权重。U↔G0 的不同 lowering 也继续单列，未重新解释成纯布局收益。
