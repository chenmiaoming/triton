# 频率与运行条件的匹配复测方案

状态：待执行的后续协议草案。当前 Modal 三台 H100 均拒绝设置 SM clocks，因此没有执行固定频率复测；以下内容不代表已取得新计时。

目标是区分同一对 AWQ 程序在不同运行规模下的收益变化与 DVFS、缓存及采样顺序的干扰。

1. 同一 H100 worker、UUID、context、stream 内，两种尺寸共享同一个已加载 U function、同一个已加载 GL function。G0 使用注明来源的共同 allocated payload。原有文件保留不动，GPU 上只做冻结 CUBIN 的 driver execution。
2. 每尺寸保留 U/G0/GL，并增加 `U_repeat` 独立观测。它与 U 使用同一 Loaded function、相同指针/参数/grid，专门检验“同程序、同输入”的时间稳定性，不复用 U 的样本充当重复。
3. 固定两种尺寸分别为 packed 3584×448、8192×1024，grid 分别 (14,112,1)、(32,256,1)，warps=4。完整 oracle、guard、只读输入和 ABI 校验先通过。
4. 大尺寸可能驱逐小尺寸缓存。采用按 visit 随机交错的 size block，每个 block 重新 prime 当前尺寸，再交错 U/G0/GL/U_repeat；或采用每个目标前固定 8 次同程序 prime 的 graph。若选择后一方案，明确记录为“8 untimed kernel → start event → 1 target → end event”，不与旧的一 kernel graph 混算。
5. 实验开始前冻结 priming、warmup、顺序、样本数、bootstrap 和频率阶段。默认频率与锁频阶段使用完全相同的缓存/采样协议；按 worker 平衡阶段顺序，避免把新测量方法的变化记成锁频效果。
6. 每个 visit/phase 前后检查文件与 loaded-buffer SHA；正式样本循环不反复读磁盘、哈希文件或插入 HtoD/oracle 检查。AWQ 没有原地输入，无需每样本恢复输入。记录 host 间隙和 graph 结构变化。
7. 负载期间使用 NVML 采样实际 SM/graphics/memory MHz、功率/上限、温度、pstate、利用率与 clock-event reasons，保留 monotonic timestamp。采样 telemetry 不是每个微秒 kernel 的精确频率证明。
8. 固定频率阶段只有在设置命令成功、同卡负载 telemetry 支持有效锁定后才成立。H100 的 memory lock 单列；不做 deferred memory writes 或宿主机 reset。仅在 finally 中恢复本次成功修改的配置并验证结果。
9. 报告 U/U_repeat、U/G0 等价性、每尺寸 GL 相对默认的收益，以及 `[gain_small−gain_large]` 的交互项。跨尺寸/频率阶段共用 worker/visit bootstrap 索引；不能相减两个独立 CI 代替交互项区间。
10. 如果仍无锁频权限，可以另行执行有频率监控的同卡诊断，但必须标记为未锁频。严格锁频因果对照需要具备设备管理权限的 H100 环境；A10 等其他架构的实验不能替代该 H100 对照。

上述后续计时尚未执行。本轮只保存权限探针，不将 nvidia-smi 查询、持续负载、默认 boost 或 NCU 的 clock-control 当成成功取得无 profiler 锁频测试条件。
