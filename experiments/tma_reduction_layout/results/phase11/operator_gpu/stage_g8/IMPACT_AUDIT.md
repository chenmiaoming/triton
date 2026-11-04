# 旧生产 Gluon 基线影响审计

本轮用户已授权：发现受影响项目直接重测。重测应另存 G8，使用已通过 G6/G7 的源码对齐基线；旧 raw samples、原始 compiler artifacts 和旧报告原样保留。

## 需要重新测量的部分

| 旧阶段 | 依赖 | 原计时逻辑/物理条件；样本 | 原 NCU 文件 | G8处理 |
| --- | --- | --- | --- | --- |
| G2 | vLLM/SGLang controlled Gluon | 294 / 261；121,365 | 12（每文件多个程序，共60个kernel结果） | 重做六几何合法布局候选的性能与关键计数 |
| G3a | 直接复用G2 CUBIN | 无新计时 | 60 | 对新源图重采对应机制计数；不当成另一份性能矩阵重复计费 |
| G3b | 旧 g3b_gluon.merge | 99 / 90；41,850 | 60 | 重做三个LN几何的merge、两档完整backward，以及native输出对照 |
| G4a | 旧SGLang PV/QK受控图 | 40 / 40；18,600 | 20 | 重新测PV-only、QK-only、协同factorial及计数归因 |
| G4b | 旧actual/B4子集与target | 84 / 48；22,320 | 0 | 与G8完整矩阵合并覆盖；保留actual ABI和原B4差异的边界，不重复计费或挑旧结果 |

候选矩阵细节：G2六几何共有41个合法layout候选加6个G0 anchor（47个Gl标签），再加6个U与3个stage2，共56个逻辑编译记录；G3b每几何11个标签（U、anchor、7个layout、vec1/vec8 native）乘三个运行范围；G4a为两个SG几何、五个程序、两种context、stage1/decode。

源入口是 `production_compile.py`、`g3b_compile.py`、`g4_compile.py`；G3a从G2 compile gate直接绑定。逐项协议、原计时/计数JSON及SHA见 [impact_audit.json](impact_audit.json)。新源图可能改变布局排序、最优候选与NCU结构，因此不能只复测旧赢家，也不能把旧GL/U总收益沿用成纯布局收益。

## 不因本次缺陷重测的部分

- Phase11 B、C1/C1.5/C1.6、C2a/b/c/d、D1、M2/M3/M4a/M4b：使用各自TMA/max或消费者kernel。冻结worker清单不包含五个旧生产Gluon模块；13个阶段的源码ZIP均按协议SHA逐项核验通过，共20、12、14、14、14、14、18、17、22、27、32、40、41个成员。
- C3–C3.4与M1：在上述独立TMA证据上做CPU模型、交叉验证与审计。模型已有的证据/外推限制继续保留，但不是本次旧生产G0缺陷的传播结果，不因此自动重拟合。
- A10：`local_a10/kernels.py`使用自有direct/cp.async Gluon；SHA与原freeze一致。`local_a10_versions/kernels.py`是纯Triton语言，没有Gluon，版本回归比较不受影响。
- operator_survey与G1：使用独立`kernels_gluon.py`语义fixture。survey明确`whole_graph_bridge_verified=False`，G1仅234项完整正确性检查，无生产性能结论。其正确性数据保留，不因生产G0变化重跑。
- 早期Phase4–10源码未引用这些旧生产模块。全仓非results Python AST扫描的旧模块直接import均局限在operator_gpu；这项扫描不替代对其他历史U/Gluon程序的独立等价证明。

## 两个容易误判的依赖

`stage_m4_preparation/prior_manifest.json`等保护清单收录旧生产模块，是“保留历史文件SHA”，不等于执行它们。M4 validator只复用了通用`layout_parsing.encodings`解析器。

G6 vLLM仍导入旧模块里的`find_seq_idx`、`cdiv_fn`辅助函数，未调用旧`reduce_segments`。不能仅按文件名出现就把通过G6/G7机器代码与实际运行门禁的新基线也判为受影响。

本审计严格指“是否受此次生产G0表达偏差影响”。独立实验不受这次缺陷影响，不等于替它们证明所有默认Triton与手写Gluon天然相同；其原始适用范围与方法限制仍然有效。
