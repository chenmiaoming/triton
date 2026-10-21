# CPU 复核与证据位置

```bash
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g3_validate --stage a
```

- `preregistration/protocol.json`：问题、20 个旧二进制、三种采集模式及原 G2 全文件 SHA。
- `preregistration/source_snapshot.zip`：采集前冻结的源码。恢复到 `/tmp/operator-gpu-g3a-snapshot-v1` 后，`g3_profile` 可读取对应源；复采需要新 attempt，不能覆盖原证据。
- `profiles/*/attempt_001/export/`：60 份原始 `.ncu-rep`，对应命令、job、远端导入与执行绑定。
- `offline_imports/`：本地 NCU 的 CPU 重导入结果。
- `counter_analysis.json`：全部数值、动态逐 PC 指令、UUID、寄存器与驻留容量。
- `REPORT.md` / `FINDINGS.md`：完整表和解释边界；正式时间引用原 G2。

新增采集使用 `MODAL_PROFILE=chenmiaoming` 及 CLI `--profile chenmiaoming`，没有新 native build，没有修改 G2 的原始样本或编译产物。
