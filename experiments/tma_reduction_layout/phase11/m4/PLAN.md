# M4：加载服务、同步依赖与真实消费者

本轮按用户截图授权推进 CPU 准备、M4a、M4b，分别提交并 push，随后停下复核。
继续使用 chenmiaoming / H100!、固定 20d9 native、现有 image/ccache/kernel cache。
旧 C3.4、M2、M3、D1 和跨算子 G1/G2 的源代码、原始样本与编译产物全部保持不动。
不拟合选择器，不修改 C++，不创建 PR，不启动 D2。

CPU 准备：新增受限操作计划，纳入混合置换门限、按 warp/指令分组的 store
sector 请求、消费者需要的结果副本与返回读取剪枝条件、FP32 输入多消费者的
活跃值变化。128/64/32/0B shared swizzle 地址预测来自固定编译器的 LinearLayout
构造。载入宽度是条件假设，必须核对生成 SASS；结构计划不输出人为总分。

M4a：H4 是待检验假说，不能由 64 波前/频率直接换算同步停顿时间。
复用 M2/M3 三个几何 32x64/w8、32x128/w4、16x256/w8 的 q 曲线。
对每个几何 vec2:N-first/within、native 输出，保持寄存器布局和消费者源程序，
测试 shared swizzle 128/64/32/0；每个 swizzle 有匹配的 T_control。
共 36 个逻辑程序，其中 9 个 128B 程序复用。CPU 拒绝的布局单列；正确性失败
不进入计时。新的配对实验只测 q=1 和 natural，原 q=2/4/8 曲线直接复用；
硬件计数补齐 q=1/2/4/natural 的 barrier、short/long scoreboard、MIO、issue。
无计时结果驱动的 NCU 扩容或重复。

M4a 只能在源行归属的 MR/输出指令、寄存器与实际驻留容量均核对后说明隔离程度。
shared swizzle 同时改变 TMA descriptor 和加载指令时，使用匹配控制并报告联合效应，
不能从全 kernel stall 百分比直接提取独立 MR/barrier 延迟。保留方向相反或零结果。
NCU warp-state 百分比有各自分母，不能直接相减或与动态指令数相加为纳秒。

M4b：两个原开发几何 32x64/w8 与 32x128/w4 的全部 15 个合法唯一布局，
保留 native 1D max 与既有 broadcast(max+logical_row) 诊断消费者，新增真实
`wide = x.to(float32); y = wide - max(wide, axis=rows)`，写出 FP32 二维 tile。
这三种消费者各 15 个程序；另有两个旧 1D T_control 和两个匹配输入/二维写出的
copy 控制，共 49 个逻辑程序。旧 fixed 输出程序/q 曲线保留作历史上下文，不重复
整个 fixed 矩阵。新旧不同输出字节数的消费者各自记分，不混为同语义加速。

M4b 报告每个消费者内相对默认 vec8:N-first、fixed_vec1 的差分和全部 vec/placement。
不假定 vec2/4 必胜，也不把事后最快候选当作冻结选择器验证。真实 centered 消费者
会使宽化输入存在第二个用户，允许 max packing、寄存器和 q 改变；这些都属于
消费者耦合机制，必须报告。copy 对照不是可以无条件减去的固定底座。

共同测量：先 CPU 编译与 native/ccache 检查，再独立 driver 进程加载 SHA 绑定
CUBIN，三种输入、完整 2 MiB 输入对应输出和 canary 通过后才计时。主指标沿用
128/256/384 MiB 连续 BF16 输入的 loop-free FULL 服务斜率 ns/tile/SM；128 MiB
原始 µs 单列。3 worker、3 visits、每次 4 samples、2 warmup，种子冻结、配对顺序。
输出缓冲区按实际 stride 分配，二维消费者最多写出输入字节数的两倍，严禁复用
原 1D 输出分配公式。q 是 padding 控制的容量，不是实测活跃 CTA 数。
NCU source/dynamic 与 hardware/stall 独立采集，报告本地 CPU 复读。
最终逐文件 validator、原始证据 SHA、分阶段 commit/push 后停下。
