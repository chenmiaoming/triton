# N5 绑定与正确性补测（未计时）

这轮检查较大的内存 footprint，以及 N01 的尾块、padding 和非连续输入步幅。它不判定 L2/DRAM 主导区间，也不声称这些形状来自生产模型。

本 v2 修正首份派生报告的 AWQ stride 表述：packed 权重输入为 `8192×1024`、stride=`(1024,1)`；`8192×8192` 是展开后的逻辑/输出形状。所有 stride 现直接来自真实绑定 fixture 元数据，单位为各自 tensor 的元素。原始实验及验证结果未变。

## 冻结条件与结果

编译器继续固定为 `20d9d98fead78c2f65e1b3ba5a7881d86a836dad`；native SHA256 为 `fc5f2ad73f9f8c8ab31c3a01b8124329ccec391cdc5dbf2dafb2d42778cf6adb`。GPU 为 `NVIDIA H100 80GB HBM3`，CUDA UUID 为 `c58d5bcb-6149-a892-b8ee-0811dba8046f`。复用现有 kernel/compiler 缓存，`ccache_before == ccache_after`，没有 native rebuild、CUDA timing events 或正式计时。

每个 case 的 U 与 G0 均通过真实 Tensor-JIT 绑定后的 allocated ELF、ABI/launch、资源一致性门。随后所有 U/G0/候选程序在 finite、structured、edge 三种输入上执行完整独立 oracle、256B 前后 guard、只读字节及输入存储间隙校验。

| case | 逻辑 shape | 输入 row/col stride | 逻辑程序 | 完整检查 | ELF / ABI / resources |
| --- | --- | --- | --- | --- | --- |
| N01_m4096_n8192 | 4096×8192 | scale_ptr: 8192/1 | 3 | 9/9 | True / True / True |
| N02_k8192_p1024_g128 | 8192×8192 | qweight_ptr: 1024/1 | 3 | 9/9 | True / True / True |
| N01_tail_stride_m131_n7 | 131×7 | scale_ptr: 19/2 | 10 | 30/30 | True / True / True |

总计 **16 个逻辑程序、48/48 个完整检查通过**。大 N01 仅比较已冻结的 `coordinated_fully_coalesced`，大 AWQ 仅比较 `packed_compute_vec1`。尾块仅做正确性检查，覆盖原先八个 N01 声明，包括与新默认重合的 alias；实际输入 stride 为 `(19,2)`，完整 padded 输出 shape 为 `256×8`，grid 为 `(2,2,1)`。

N01 大点的 uint8 输入和输出各为 32 MiB；N02 大点的 FP16 输出为 128 MiB、packed int32 权重为 32 MiB，另有 scale/zero。这里仅报告分配体积，尚不据此标记缓存命中率或 DRAM 流量。

## 保留的失败尝试

revision_001 两个大 case 已通过 18 个完整检查。尾块的 U/G0 编译一致性也通过，但旧开发 adapter 会跳过等同新默认的输入布局，再触发固定“八个候选”的枚举断言，因此尾块在执行前终止。原始失败完整保留。revision_002 没有修改冻结 adapter 或内核，只在 N5 协议中绑定 N2_r2 已声明的八个候选定义，保留 alias 后重新执行全套检查。

## 下一步接口与停点

[完整原始结果](binding/attempt_001/export/result.json) 与 [raw_export.zip](binding/attempt_001/raw_export.zip) 保持原样。通过的两个大 case 及其六个 U/G0/候选程序另封装为 [accepted_timing_payload.zip](binding/attempt_001/accepted_timing_payload.zip)，可供独立冻结的 driver 测量阶段使用；尾块不进入该计时包。

[CPU 最终验证](BINDING_VALIDATION.json) 校验 16 份 artifact hash 清单、48 个检查、source snapshot，以及之前 N2_r2 的 1113 个受保护文件均未改变。**本阶段停在正确性与基线门之后，未开始正式计时。**
