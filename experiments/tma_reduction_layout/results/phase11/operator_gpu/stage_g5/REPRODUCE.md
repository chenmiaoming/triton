# G5 复核与复现

在仓库根目录运行，下列命令全部为 CPU/离线处理，不申请 GPU：

```bash
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g5_analyze
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g5_actual_analyze
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g5_validate
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g3_validate --stage all
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_validate --stage all
```

分析器对已有派生产物只允许字节一致的重生成；不会覆盖不同内容。最终 validator 重读原始压缩包、解压文件、源码快照、所有编译产物、pass trace、ELF 区段、缓存恢复 key，以及旧阶段保护清单。`final_validation.json` 在交付前以 `g5_validate --write` 创建，此后严格核对内容与清单 SHA。

## 原始 CPU 任务

| 任务 | 协议/源码快照 | 原始结果 |
| --- | --- | --- |
| pilot 1 | `protocol.json` / `source_snapshot.zip` | `pilot/attempt_001` |
| 共享 cache key 恢复 | `protocol_v2.json` / `source_snapshot_v2.zip` | `pilot/attempt_002` |
| 其余六个默认几何 | 同 V2 | `matrix/attempt_001` |
| vLLM literal 1 | `literal_protocol.json` / `literal_source_snapshot.zip` | `literal/attempt_001` |
| vLLM literal 2 | `literal2_protocol.json` / `literal2_source_snapshot.zip` | `literal2/attempt_001` |
| vLLM literal 3 | `literal3_protocol.json` / `literal3_source_snapshot.zip` | `literal3/attempt_001` |
| 真实调用 ABI | `actual_protocol.json` / `actual_source_snapshot.zip` | `actual/attempt_001` |

协议与 ZIP 均在 `preregistration/`；每次 dispatch 保留 Modal function call ID、账户与协议 SHA；每个子程序的 `command.json` 保留命令、退出状态和输出。源码改动前的失败对照有独立版本，没有覆盖。V1 后端干预发生在共享 kernel cache，V2 已验证恢复，**不要重新在共享缓存执行 V1 干预**。

## 如需重新做远端 CPU 编译

当前工作树的 `g5_worker.py` 是 V2。必须从对应快照在独立临时目录重建源码，而不能将 V1 协议与 V2 源码混用。后端 native 镜像固定为 `im-qzTUZumWu2JXjWic8K82cZ`，原生库 SHA 与编译器版本见协议。所有 Python 后端干预均应使用新的独立 `TRITON_CACHE_DIR`；不重建原生库。

入口为 `g5_run.py`（pilot/matrix）、`g5_align_run.py`、`g5_align2_run.py`、`g5_align3_run.py`、`g5_actual_run.py`。它们只声明 CPU Modal function，未声明 GPU。账户环境和参数必须同时固定：

```bash
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g5_actual_run --attempt 2
```

该示例会写一个新的 `actual/attempt_002`，不替换冻结的 attempt_001，分析器也不会自动挑选最新一次作为赢家。普通离线复核无须运行此命令、支付远端费用或重新编译。

## 等价的单位

`equivalence_gate.json` 比较九组默认程序；`actual_ABI_analysis.json` 比较四组额外 ABI，并映射十个 G4b workload。比较覆盖：全部 ELF SHF_ALLOC 区段（仅规范化区段名中的入口符号）、运行时参数类型/次序/名称、静态及动态 shared memory、物理寄存器/stack/local 分配。源文件/debug 元数据可以不同。相同 conversion 数、相同 n_regs、接近的计时都不足以代替该门禁。

诊断后缀不是编译器全局补丁；后续候选必须另做实际生效 layout 审核。此交付没有新候选计时。
