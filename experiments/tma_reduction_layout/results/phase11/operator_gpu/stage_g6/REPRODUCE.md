# G6 离线复核与后续接入

## 默认 Gluon 基线

已接受版本为 `baseline_manifest.json` 中的 `G0_source_aligned_v1`（attempt_003）。三个入口为：

```python
from experiments.tma_reduction_layout.phase11.operator_gpu.g6_compile import arguments
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.backends.compiler import GPUTarget
import triton

# item 来自冻结 protocol.json 的 cases；保持其 ABI/attributes/specialization。
fn, constants, attrs = arguments(item, gl)
compiled = triton.compile(
    gluon.GluonASTSource(fn, dict(item['record']['JIT_signature']), constants, attrs),
    target=GPUTarget('cuda', 90, 32),
    options={'num_warps': item['case']['W'], 'num_stages': item['case']['num_stages']},
)
```

实际接入必须使用 manifest 的 compiler/native/backend/target 与源文件 SHA。函数里没有安装编译器后缀，也没有 monkeypatch。`function_for_family` 可以取得普通 JIT 入口；不同实际调用特化产生的编译形式要重新对 U 校验。

候选布局可传给 `arguments(item, gl, candidate_layout=layout_spec)`。该接口只表示可构造候选，未表示已验证候选性能。继续计时前应冻结候选代码、审核实际生效布局并验证完整输出。

## 离线验证

仓库根目录：

```bash
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g6_analyze --attempt 1
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g6_analyze --attempt 2
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g6_analyze --attempt 3
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g6_validate
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g5_validate
.venv/bin/python -m experiments.tma_reduction_layout.phase11.operator_gpu.g4_validate --stage all
```

这些命令没有 GPU 或远端调用。分析器只允许字节一致地重建派生 JSON，保留 attempt_002 的两个非等价项。最终 validator 在交付前用 `--write` 创建冻结结果，此后只检查内容一致。

## 原始 CPU 运行

各 `attempt_00N/` 下包含：`preregistration/protocol.json`、`source_snapshot.zip`、`dispatch.json`、原始 `raw_export.zip`、完整解压 `export/`、`comparison.json`。源码在每次采集前冻结，历史快照保留旧版本，不要求它们与最新源文件 SHA 相同。

远端 CPU 入口 `g6_run.py` 使用既有镜像 `im-qzTUZumWu2JXjWic8K82cZ`，只声明 CPU function，独立内核缓存目录 `/cache/phase11-upstream20d9-kernel-cache-g6-source-only`。调用必须同时设置 `MODAL_PROFILE=chenmiaoming` 与 `--profile chenmiaoming`。原生构建不重跑，ccache 统计在调用前后不变。

如需另一次实验，先用新的 attempt 冻结协议与源码，再运行对应 attempt；不要覆盖既有证据。普通复核无需付费重编译。
