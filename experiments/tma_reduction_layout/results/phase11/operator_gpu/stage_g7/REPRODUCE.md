# G7 复核与复现

## 无 GPU 离线复核

仓库根目录运行：

```bash
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g7_analyze
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g7_validate
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g6_validate
```

以上命令不会申请 GPU 或访问 Modal。分析器允许字节一致地重生成 timing_analysis.json/timing.csv；不同内容禁止覆盖。最终 validator 以 --write 创建一次冻结结果，之后严格核对 inventory/source/report SHA。

## 证据组织

- `preregistration/protocol.json`、`source_snapshot.zip`：原始预注册与源码。
- `protocol_v2.json`、`source_snapshot_v2.zip`：计时前修正调用脚本，冻结仅补齐8次SGLang绑定的协议；原始记录原样保留。
- `compiled_payload.zip`、`compile_gate*.json`：48个逻辑artifact（含7个stage2、3个dX helper），来自既有冻结U和G6。没有新候选编译。
- `binding/attempt_001`：保留首轮完整失败/成功产物。
- `binding/attempt_002`：只补齐8个缺失真实CUDA tensor warmup。
- `binding_gate.json`：26次完整实际绑定的逐项汇总、属性字面差异和兼容依据。
- `correctness/attempt_001/raw.json`：完整正确性与同一partial位等检查。
- `timing/invocation_{1,2,3}/raw.json`：预先固定的三次独立H100计时，32条件×2角色×5visit×31samples。
- `timing_analysis.json`、`timing.csv`：派生统计与每worker明细。

原始ZIP与解压件逐字节核对；每次dispatch保存Modal function call ID和协议SHA，原始结果包含实际GPU UUID、driver、所有输出SHA及CUDA driver返回码。完整backward的非位等诊断与same-partial严格位等分别保留，不混为一项。

## 远端运行的原命令

已完成，不需要重复以复核；结果驱动重跑禁止。历史命令为：

```bash
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g7_run --action bind --attempt 2
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g7_run --action check
MODAL_PROFILE=chenmiaoming .venv/bin/modal run --profile chenmiaoming -m experiments.tma_reduction_layout.phase11.operator_gpu.g7_run --action timing
```

首轮bind使用原始快照；当前代码对应V2，不能拿当前代码覆盖或重放首轮成新“成功”。重新实验必须另立输出目录与协议。运行账户为chenmiaoming，指定H100!避免自动替换H200；native和kernel缓存复用，性能进程本身不导入Triton、不编译。

统计标准在计时前冻结：paired CI95完全落入[-3%,+3%]即practically_equivalent，其余uncertain。validator只校验事实/证据完整性，不以这个结果为PASS条件。
