# 开题报告修改版与布局选择源码审计

修改用户上传的 `tma_reduction_layout_opening_report.pptx`，保持原有白底学术版式与可编辑布局图。
修改版为 17 页正文、3 页附录。正文新增当前编译器机制与已有实验结果，附录补普通 Triton 完整代码、版本对照和参考索引。

- `tma_reduction_layout_opening_report_revised.pptx`：可编辑 PowerPoint。
- `tma_reduction_layout_opening_report_preview.pdf`：依据 PPTX 形状与文本重绘的预览；字体使用 Noto SC，不是 PowerPoint 原生导出。
- `mechanism_audit.md`：TMA、普通 load、blocked 构造器、布局传播与 thread-locality guards 的源码说明和不可变链接。
- `speaker_notes.md`：20 页同步更新后的讲稿。
- `audit.json`：源码与输入 SHA、CPU 布局重建、六份 A10 和两份 H100 已有最终 TTGIR 的对照，以及已有实验数字。
- `source_audit/`：固定 upstream、v3.7.1 与本次观察的 main 源码原文及获取信息。
- `input/`：上传原稿与所引用方案草案的快照。

**主要修改**

修正了笼统的“36%–46% 退化”、混合测量批次的 A10 收益、未注明 descriptor 路径的 H100 数字、默认 PTX 的 BF16/FP32 顺序、“模型高估 4 倍”，以及尚未完成工作的状态描述。
现有三种布局的数学关系和完整 32 行元素归属图保留；早期 vec2 原型明确标成示例，不再作为已验证的现行模型。

回答新问题的核心结论：默认启发式有时已接近所测较快候选。TMA 路径以每线程元素量和 128-bit 向量化目标构造布局；普通 load 根据指针连续性、对齐和相关内存操作构造布局。后续有布局传播和特定模式优化，但没有为本例做经过校准的完整候选成本比较。

2026-10-15 读取的 main 为 `4eda6cc91e86553a2517d1c764ea2967985553ad`。其五个关键布局 transform 文件与实验 upstream `20d9d98fead78c2f65e1b3ba5a7881d86a836dad` 逐字相同。源码审计不能当作在新 main 上完成了性能测试。

**构建与验证**

从仓库根目录运行：

```bash
.venv/bin/python -B experiments/tma_reduction_layout/reports/opening_report_2026_10_15/audit.py
.venv/bin/python -B experiments/tma_reduction_layout/reports/opening_report_2026_10_15/build_report.py
.venv/bin/python -B experiments/tma_reduction_layout/reports/opening_report_2026_10_15/render_preview.py
.venv/bin/python -B experiments/tma_reduction_layout/reports/opening_report_2026_10_15/validate.py
```

依赖：python-pptx、reportlab、Noto SC 常规/粗体字体及 DejaVu Sans Mono；本机使用已有 `/tmp/tma-ppt-fonts/`。PDF 页数验证调用 `pdfinfo`。修改版 PPTX 保留 Microsoft YaHei / Consolas，与上传原稿一致。

`audit.py` 的公式复现是 CPU 只读分析，核对已有最终 IR，不是新增 instrumented compiler trace。本次没有 native 编译或 GPU 采样；没有修改旧 raw samples、原始 compiler artifacts 或 NCU reports；新的 C1/model 阶段仍未开始。

原始实验位置分别是 `results/phase11/stage_b/`、`results/local_a10/20261015_gpu0_r2/`、`results/local_a10_versions/20261015_gpu0_r2/`。H100 配对百分比与表内跨 worker 中位数采用不同估计量；A10 两轮时间来自不同批次，报告已明确区分。
