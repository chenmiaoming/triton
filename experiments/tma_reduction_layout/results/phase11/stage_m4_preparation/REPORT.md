# M4 CPU 准备：扩充操作计划与实验预绑定

新增 operation_plans_v2.json；M3 的计划和全部历史证据保持不变。本阶段无 GPU
编译/计时，无选择器拟合。新计划包含转换混合置换门限、warp/指令级 store 分组、
消费者需要的结果副本、单次返回读取的剪枝条件，以及真实二维消费者对宽化输入
活跃值的影响。完整实验边界见 phase11/m4/PLAN.md。

M4a 在寄存器布局和 native 消费者固定时改变 128/64/32/0B shared swizzle；
下面是编译前地址预测（wavefronts / bank conflicts）。预测以逻辑 vec 分组为条件，
必须由 SASS 和 NCU 检查真实指令分组，不能直接用计数除频率当延迟。

| 布局 | 128B | 64B | 32B | 0B |
|---|---|---|---|---|
| M32_N64_w8:vec2:n_first | 32 / 0 | 64 / 32 | 128 / 96 | 32 / 0 |
| M32_N64_w8:vec2:within | 32 / 0 | 32 / 0 | 32 / 0 | 256 / 224 |
| M32_N128_w4:vec2:n_first | 64 / 0 | 128 / 64 | 256 / 192 | 64 / 0 |
| M32_N128_w4:vec2:within | 128 / 64 | 64 / 0 | 128 / 64 | 128 / 64 |
| M16_N256_w8:vec2:n_first | 64 / 0 | 128 / 64 | 256 / 192 | 64 / 0 |
| M16_N256_w8:vec2:within | 128 / 64 | 64 / 0 | 128 / 64 | 128 / 64 |

M4b 在两个原开发几何上测全部合法唯一布局，保留 native 1D 与原广播诊断，新增
真实 x.to(float32)-column_max(x.to(float32))。后者宽化输入存在第二个消费者，
BF16 packed-max、寄存器、驻留容量都可能改变，不能提前把变化只归到 MO。
不同消费者有不同输出字节数，分别在同语义内比较候选；不会把二维程序与一维
max 的时间直接作布局加速比。输出缓冲区按 BM*BN stride 分配。

加载侧支持 12 个 shared-address 双射检查，并与旧 128B 地址模型逐布局一致；
dense store 请求预测保留 vec8 的每指令 sector 重复，未将其当作 DRAM 写字节数。
旧 M3 H1/H2/H3 是已有结构证据；新增 shared-layout 和真实消费者的性能仍待测。

H4 拟检验加载吞吐与同步依赖的不同驻留响应，但 stall 百分比是相关诊断，不能
孤立证明某个 barrier 的延迟。根据 NVIDIA 的 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)，
short scoreboard 和 MIO 涉及多种操作；需联合源行、指令和 wavefront 检查。
若合法 swizzle 无法独立改变冲突，或同时改变 MR/寄存器，保留该负结果。

后续停点：完成 M4a、M4b 的报告/validator/分阶段 commit 与 push，等待复核。
