# 离线复核

```bash
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g3_validate --stage all
```

该命令只在 CPU 校验原始源码 AST、源码快照、CUBIN/ABI、真实 JIT 比对、正确性、原始计时来源、NCU 本地重导入与 G2 文件 SHA，不启动 GPU。

重新计算派生表使用 `g3b_analyze timing`、`g3b_analyze counters`、`g3b_report`。这些脚本对已有派生文件要求结果相同，不覆盖原始结果。全量表见 `MATRIX.md` 和 `COUNTERS.md`。

- `preregistration/protocol.json` 与 `source_snapshot.zip`：初次 CPU 预检前的冻结内容。
- `protocol_v2.json` 与 `source_snapshot_v2.zip`：切换到已有 compiler image 的修订，发生在训练 GPU 编译身份校验和计时之前。
- `cpu/attempt_001`：CPU tools-image 缺 Triton 的原始失败记录；`cpu/attempt_002`：成功的 36 程序编译与全部 compiler artifacts。
- `binding/attempt_001`：36/36 实际 CUDA Tensor JIT CUBIN SHA 比对。
- `correctness/attempt_001`：270 个去重执行检查；`timing/invocation_1..3`：三份原始计时。
- `profile_protocol.json` 与 `profile_extension_protocol.json`：初始及完整候选补证协议；对应两份独立源码快照。
- `profiles` / `profiles_extension`：共 60 份原始 NCU 报告及远端导入，`offline_imports` 保存本地 CPU 重导入。

Gluon anchor 与 vec8:N-first 等精确执行别名的去重依据见 `BINARY_ALIASES.md`。所有 GPU 使用明确的 chenmiaoming profile；复采须新 attempt，不覆盖现有证据。
