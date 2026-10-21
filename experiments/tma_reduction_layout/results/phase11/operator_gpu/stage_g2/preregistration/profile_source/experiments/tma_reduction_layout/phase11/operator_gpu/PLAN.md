# G1/G2：跨算子 GPU 首批验证

本批已获 GPU、Modal 外发、阶段 commit/push 授权。继续使用 chenmiaoming，
固定 Phase 11 的 20d9 native 与已有编译缓存，不改 C++、旧模型或历史原始证据。

G1 先直接加载既有 69 个不同 CUBIN，对全部八类语义程序完成输出/输入/canary
检查。原始失败记录保留；FP8 CPU 参考修订必须单列，不以放宽容限掩盖差异。

G2 只对本批优先的两个生产入口做性能主实验，其他候选尚无生产性能结论：

- C06：固定 commit 的 vLLM `reduce_segments`；保留二分序列查找、零长度与
  padding 早退、segment 统计量、FP32 scratch 和 BF16 输出。属于开启 use_3d
  且准备对应 scratch 的分支，不代表所有 vLLM attention。源代码依赖做独立
  提取，在固定 Triton 下重现 kernel，未安装或测量整个 serving package。
- C02：固定 commit 的 SGLang non-GQA、普通 pointer、PAGE_SIZE=1 的 normal
  decode stage1；保留 BF16 乘积精度、QK、在线 softmax、PV、分页索引与原始
  stage2。只改变 PV 矩阵布局，Q/K、softmax 所在布局和输出布局保持固定。

## 预固定矩阵

C06：segment×head_dim 为 16×128、32×128、32×256，4 warps。每项 batch 为
1/8/64，32 query heads，序列长度 8192，测量一次 merge。使用单 query decode
且配置 use_3d threshold=64 的可达条件；不伪造大 grid 代表小 batch。

C02：BLOCK_N=64，head_dim=64/128/256，4 warps，32 Q/KV heads、4 splits。
batch=1/8，context=256/2048；分别测原 stage1 和同一 stage2 下的两阶段 decode。
这些是合成的 serving 尺度输入，不冒充捕获的模型轨迹。GQA/tl.dot、其他分页
大小、FP8 KV、sink、score_mod 与完整服务器端到端不纳入本批结论。

每项包括原始 Triton、Gluon canonical 输入布局锚点及合法唯一的 vec1/2/4/8
N-first/within。按真实 JIT 的 dtype、指针对齐与整数特化编译并导出，再由另一个
无 Triton 进程加载 CUBIN 做正确性和计时。最多 72 个不同 binary；完全相同的
CUBIN/ABI 在计时中合并别名，保留全部逻辑标签。

## 桥接、正确性与计时

从真实基线 IR 读取 Q/K/V、序列统计量和输出 encoding；不预设统一 vec8。
G0 检查数学表达式/控制流一致、输入/输出编码及资源/指令结构，并在计时中
单列 U↔G0 差异。未完整对齐时只能称 source/encoding anchor，不能把 U↔GL
差值全解释成纯布局收益。纯布局对照使用同一个受控 Gluon 程序的 G0↔GL。

正确性先检查三类输入、完整输出、未写出的 padding/inactive split、输入字节与
两侧 canary，C02 同时检查 partial、LSE 和最终输出。CPU 高精度参考保持原算法
的 BF16 乘积；FP32 rtol=2e-4/atol=2e-5，BF16 rtol=.008/atol=1e-4。失败记录
保留，未经正确性门的程序不计时。

正式时间为一次调用 CUDA Graph 的设备事件时间（单 kernel 或 stage1+stage2），
单位 µs；每一组 3 个独立进程、5 次访问、每次 8 warmup + 31 样本，访问顺序按
固定种子配对随机化。不依输赢追加重复；以 worker/visit 为统计单位报告不确定性，
不把全部样本当独立 worker。>=3% 是实用差异门槛，不保证一定有收益。

## 硬件计数与停点

预指定每个 geometry 的 U、G0、vec1:N-first、vec2:within、vec8:within（合法
别名解析后去重）做计数。software 为单 CTA，hardware 为 batch8 的实际 grid；
SG 使用 context2048。保留 ncu-rep、raw/source CSV、worker UUID、指令/波前/
bank conflict/global sectors、寄存器/local 与 occupancy capacity。NCU 的时间
不参与正式计时，静态 vec 不替代硬件访存次数。

本批完成 G1/G2 的报告、validator、分阶段提交与 push 后汇报；不自行拟合新
cost model、修改编译器、创建 PR 或跨架构迁移。其余六类生产入口留在候选清单。
