# 本地 A10：非 TMA 加载与 partial reduction

用户授权仅使用物理 GPU 0。设备为 A10 / SM86，UUID
`GPU-69464998-e901-74ba-63ff-ae7022f8aed9`；使用 UUID 限定 CUDA 可见设备。

此诊断回答两个问题：普通 load 后的规约是否必须先把输入放进 shared，
以及非 TMA 的 shared → registers 路径是否也对布局敏感。
它独立于仍待审阅的 Phase 11 C/model 方案。

完整程序保持 `[1,8192,8192]` BF16 输入、FP32 max/output、32×128/w4 和 32×64/w8，
每种几何有普通 Triton anchor，以及同一 Gluon kernel 的两条显式路径：

- direct：`gl.load` 直接得到寄存器 tensor，随后 max；编译器仍可为规约/输出生成 shared 交换。
- cp_async：固定 writer layout 的 `cp.async` 写入 shared，wait 后按指定消费者布局读入寄存器，再 max。

每条 Gluon 路径测试自然 vec1/2/4/8 及同 vec8 的 within-warp 分配。
布局均通过合法 tensor 表达；没有 source-level lane/warp ID、内联访存汇编或 compiler 修改。

复用 `hexcute` 中已安装的 Triton 3.7.1 / PyTorch 2.13.0+cu130，native 与 assembler SHA 已归档。
这不是 Phase 11 固定 upstream/native 的替代验证；也没有运行 TMA 或复现历史 release 回归。
没有 make 或 Triton native rebuild。新 kernel 使用持久的 `/tmp/triton-exp-a10-native371-kernel-cache`。

[完整结果](../results/local_a10/20261015_gpu0_r2/summary.md)：22 个 artifact，44 项完整输出检查，
3 次同卡独立进程、1,980 个单 kernel event 样本，以及 8 份独立 NCU reports。
冻结 binary 的计时与 profile 进程不导入 Triton、不触发 JIT。
结果包括所有布局，性能只作同一 A10 上的描述性比较，不作唯一赢家或普遍非回归声明。

早期采集器兼容性失败保存在 `results/local_a10/20261015_gpu0`，修正后使用新的 `..._r2` 目录；
kernel 源码 SHA 相同。NCU 2026.3 的 `.ncu-repz` 扩展名识别修正也保留了原 dispatch，
并复用了已经成功的报告。原始 evidence write-once，derived analysis 可以版本化重算。

离线 validator 不需要 GPU：

```sh
.venv/bin/python -B -m experiments.tma_reduction_layout.local_a10.analyze \
  --root experiments/tma_reduction_layout/results/local_a10/20261015_gpu0_r2 \
  --validate --committed
```

`prepare/collect/timing/profile` 均要求新的输出身份，不能覆盖原 raw。
只有用户授权的设备才可运行。Phase 11 C1 与后续 cost model 仍需按审阅后的方案推进。
