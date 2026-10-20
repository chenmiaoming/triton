# G1：八类语义程序的 H100 正确性门

2026-10-21。本阶段直接加载准备阶段的 78 个程序（69 个不同编译键），没有
重新编译、修改原 CUBIN 或进行计时。全部九个语义配置、三种输入均已执行，
共 234 组完整输出检查。固定编译器 native SHA 仍为
`fc5f2ad73f9f8c8ab31c3a01b8124329ccec391cdc5dbf2dafb2d42778cf6adb`。

## GPU 结果与数值边界

有效运行的设备为 NVIDIA H100 80GB HBM3，UUID
`GPU-8599e318-819d-a2d3-e9c3-95da68535714`。
原始结果完整保存在 [attempt_002/raw.json](fixtures/attempt_002/raw.json)，
不能将其中的 `CORRECTNESS_FAILED` 改写为 `OK`：原 CPU 参考有 216/234 通过，
另 18 组均为 C03 的 FP8 code 在舍入边界上相差一个编码单位。

| 配置 | 原参考通过 | 物理寄存器范围 | occupancy capacity q |
|---|---:|---:|---|
| C01 参数梯度归并 | 27/27 | 139–168 | 3 |
| C02 P·V 子图 | 27/27 | 71–80 | 6、7 |
| C03 FP32-scale FP8 探针 | 9/27 | 45–58 | 8、9、10 |
| C04 RMS backward 子图 | 27/27 | 64–128 | 4、6、7、8 |
| C05 bias gradient | 27/27 | 70–77 | 6、7 |
| C06 segment merge 探针 | 27/27 | 31–69 | 7、12、16 |
| C07 V-first 递归状态 | 18/18 | 72–96 | 20、28 |
| C08 mean pooling | 27/27 | 72–151 | 3、4、5、7 |
| C08 max pooling | 27/27 | 70–91 | 5、6、7 |

所有输入的完整字节均未被改写，所有分配两侧的 256-byte canary 均保持。
运行进程未导入 Triton，CUDA driver 返回码均为 0。`q` 是 API 查询的驻留
容量，不是实时驻留计数；这些资源数属于当前语义程序，不代表原库程序。

C03 差异已由独立 CPU 运算顺序诊断精确解释。原参考使用
`amax / float32(448)`，归档 PTX 则明确执行
`div.approx.f32(1,448)` 后 `mul.rn.f32(amax, reciprocal)`。CPU 按该顺序计算
FP32 scale，再用原先独立的 FP8 codebook 编码后，**九个布局 × 三种输入的
quantized 和 scale_out 全部与 GPU 的完整输出 SHA 精确相同（54/54）**。
未放宽 FP8 容限、未修改 kernel、未为了参考修订额外申请 GPU。

[correctness_gate.json](correctness_gate.json) 将其明确标为
`PASS_WITH_EXPLICIT_FP8_REFERENCE_REVISION`，并保留原始 216/234 的判定。
这支持当前输入下布局变体的数值检查，不将近似除法解释成精确 CPU 除法，
也不将这个动态 E4M3FN 探针冒充 E8M0 MXFP8。

## 失败尝试与修复

[attempt_001](fixtures/attempt_001/retrieval_failure.json) 完整保留第一次调用的
异常。启动器误用了排序后的 JSON 对象键顺序来构造指针 ABI，导致首个 C01
读取了错误的参数地址并触发 CUDA 700；失败后清理又遇到 sticky error，
最初掩盖了归档结果。修复后从原函数签名显式冻结参数名列表，校验集合与
顺序，并将清理异常单独归档。没有性能样本产生。

两个协议与各自源码快照均保留在 preregistration：
[初始协议](preregistration/fixtures_protocol.json)、
[ABI 修复后的协议](preregistration/fixtures_protocol_v2.json)。
第二次调用完整执行 234 组检查，作为本阶段的有效 GPU 数据。

## 复核与下一步

```bash
.venv/bin/python -B -m experiments.tma_reduction_layout.phase11.operator_gpu.validate_g1
```

该 validator 检查源文件快照、payload 和 CUBIN SHA、真实加载的 CUBIN、全部
canary/输入、原参考失败集合及 54 个修订参考的精确哈希。

本阶段没有生产库性能结论。下一阶段优先绑定固定 commit 的 vLLM
`reduce_segments` 与 SGLang 普通 decode stage1（保留 Q·K、在线 softmax、
P·V 和原 stage2），使用真实 JIT 参数特化、同算法/同 tile/同 warp 的
Gluon 对照；完整图未对齐的比较必须标明。旧阶段证据保持不动。
