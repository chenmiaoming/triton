# Stage N3b — 后验 KDA gate 机制诊断

覆盖 2 个 kernel 子类型；16 份实际报告，alias 展开后 16 个逻辑报告。本补充由 N2 的 feature_vec4 结果和 N4 固定选择复测触发，明确属于后验计数器诊断；与原 N3 静态预选分母分开，不作为新增确认性计时。

协议 SHA256：`abcd5a29e3d332db30ebd038783714025aa48a483ba11b4978f6699222267ed7`。Payload SHA256：`715b5b20b539ebe0bd06bd5a332a669b6ad4383593b56ee5787a3ec54a035fec`。

## 1. 口径

- `software`：保持原始尺寸、stride、标量及二进制，grid 改为一个 CTA，读取 InstructionStats/SASS 动态执行数；检查该 CTA 的完整输出、其余 inactive bytes、输入和 canaries。
- `warm_application`：完整 grid，application replay / cache none，八次同程序 warmup。
- 如存在 `cold_isolated`：完整 grid，kernel replay / cache all；与 warm_application 分开解释。
- 每次 warmup 和 target 前恢复原地输入，恢复位于 profiler 范围之外；不导入 Triton，不执行 JIT。
- 软件计数的单位是动态 warp instruction instances；一次指令并不等同于一次 shared/global 事务。硬件 additive counter 可按实际 CTA 数归一；百分比、比值及时间不按 CTA 数相除。
- 单次 NCU 采集没有计时置信区间；profiler 下的时间不能替代无 profiler 的正式计时，也不把指令数加权成成本分数。

## 2. U / G0 动态计数交叉验证

| Case | 分别采集 | Opcode counts 相同 | 指令类别相同 | U 动态指令 | G0 动态指令 |
|---|---|---|---|---:|---:|
| N05_t2048_h8_d128 | True | True | True | 2264 | 2264 |
| N05_t2048_h8_d64 | True | True | True | 1344 | 1344 |

## 3. CTA0 的软件动态指令

| Case | Variant（含 alias） | 总指令 | LDG | STG | LDS/LDSM | STS/STSM | SHFL | BAR | LDL+STL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N05_t2048_h8_d128 | U | 2264 | 32 | 32 | 16 | 16 | 0 | 8 | 0 |
| N05_t2048_h8_d128 | G0 | 2264 | 32 | 32 | 16 | 16 | 0 | 8 | 0 |
| N05_t2048_h8_d128 | GL_feature_vec2 | 2792 | 80 | 32 | 32 | 32 | 32 | 24 | 0 |
| N05_t2048_h8_d128 | GL_feature_vec4 | 2016 | 48 | 32 | 0 | 0 | 0 | 0 | 0 |
| N05_t2048_h8_d64 | U | 1344 | 24 | 16 | 16 | 16 | 0 | 8 | 0 |
| N05_t2048_h8_d64 | G0 | 1344 | 24 | 16 | 16 | 16 | 0 | 8 | 0 |
| N05_t2048_h8_d64 | GL_feature_vec2 | 1728 | 48 | 16 | 16 | 16 | 32 | 8 | 0 |
| N05_t2048_h8_d64 | GL_feature_vec4 | 1144 | 32 | 16 | 0 | 0 | 0 | 0 | 0 |

## 4. 完整 grid 的请求与共享内存服务

下表仅展示所选 additive counter 的每 CTA 值。完整 raw 名称、单位、数值、stall 百分比、DRAM/L2 利用率及资源数据保留于 `summary.json`、`summary.csv`、`metrics.csv`。这些列不做延迟加权。

| Case | Variant（含 alias） | Mode | Global LD sectors/CTA | Global ST sectors/CTA | Shared LD WF/CTA | Shared LD conflicts/CTA | n_regs | q |
|---|---|---|---:|---:|---:|---:|---:|---:|
| N05_t2048_h8_d128 | U | warm_application | 392.000 | 512.000 | 64.000 | 0.000 | 48 | 5 |
| N05_t2048_h8_d128 | G0 | warm_application | 392.000 | 512.000 | 64.000 | 0.000 | 48 | 5 |
| N05_t2048_h8_d128 | GL_feature_vec2 | warm_application | 392.000 | 512.000 | 128.295 | 0.295 | 48 | 5 |
| N05_t2048_h8_d128 | GL_feature_vec4 | warm_application | 392.000 | 512.000 | 0.000 | 0.000 | 36 | 6 |
| N05_t2048_h8_d64 | U | warm_application | 200.000 | 256.000 | 32.000 | 0.000 | 30 | 8 |
| N05_t2048_h8_d64 | G0 | warm_application | 200.000 | 256.000 | 32.000 | 0.000 | 30 | 8 |
| N05_t2048_h8_d64 | GL_feature_vec2 | warm_application | 200.000 | 256.000 | 64.000 | 0.000 | 34 | 6 |
| N05_t2048_h8_d64 | GL_feature_vec4 | warm_application | 200.000 | 256.000 | 0.000 | 0.000 | 25 | 8 |

## 5. 后验诊断的预绑定对照范围

- `N05_kda_gate_d128` / `N05_t2048_h8_d128`：Post-timing diagnostic: compare exact feature_vec4 with U/G0 and the preselected feature_vec2 contrast. Feature_vec4 aligns input/compute with output ownership. D64 is a practical-null control for the same transformation; no timing is rerun.
- `N05_kda_gate_d64` / `N05_t2048_h8_d64`：Post-timing diagnostic: compare exact feature_vec4 with U/G0 and the preselected feature_vec2 contrast. Feature_vec4 aligns input/compute with output ownership. D64 is a practical-null control for the same transformation; no timing is rerun.

原始 `.ncu-rep`、raw/source 导出、NCU 版本、绑定与正确性记录均按 SHA 保留。U/G0 的软件计数一致与否如实展示；任何不一致需要单独调查，不在分析器中改写。对完整框架、LLM 端到端收益、跨设备泛化或版本回归均不作结论。

## 6. 保留 N4 原始统计结论

此次只补计数器，不改写或追加正式计时。严格确认要求相对 U 与 G0 的 95% 区间下界均严格高于 +3%；计数器机制成立也不能将未过门槛的计时升级为确认收益。

| Case | N4 收益 vs U (%) | N4 CI vs U (%) | N4 收益 vs G0 (%) | N4 CI vs G0 (%) | 两基线严格 +3% 确认 |
|---|---:|---|---:|---|---|
| N05_t2048_h8_d64 | 2.710738 | [2.361741153761765, 3.0776195954437413] | 1.904426 | [1.2419093955825729, 2.7222730880348536] | False |
| N05_t2048_h8_d128 | 4.273504 | [3.301488176419555, 5.20275404199978] | 3.407324 | [2.988585172510577, 3.8751511589264385] | False |

N2 触发分析 SHA256：`12fc2cba6f0ce5b8b21473688be25245850a964bb6cdd09fdbba2661a3a6a9d7`。
N4 固定选择复测分析 SHA256：`eafd435668c44e6eb41118b19fa38ddfc65abc57a9d8d50f55dfdb9ed03be39f`。

D64 保留为同变换的实用无收益对照；D128 保留其严格门槛下的 provisional 状态。寄存器和驻留容量来自真实 binary binding；容量变化不等同于实际占用率或可直接换算的加速比例。
