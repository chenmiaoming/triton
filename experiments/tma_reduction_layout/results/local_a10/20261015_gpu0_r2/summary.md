# 本地 A10 非 TMA 诊断

GPU：物理卡 0，NVIDIA A10 / SM86。复用已安装 Triton 3.7.1；不是 Phase 11 的固定 native。
22 个 artifact，44 项有限输入完整输出验证，3 次独立进程共 1,980 个单 kernel event 样本。
同一张卡的时间重复，不宣称覆盖多个设备或唯一最优；没有做历史 release 回归 A/B。

## 数据路径与直接证据

普通 global load 的值直接进入寄存器，max 在寄存器上计算。
32×128/w4 的 direct vec1 没有 LDS、STS、shuffle 或 barrier；SASS 和 NCU 均支持。
direct vec8 的 global load 仍直接进入寄存器，后续 shared 访问来自规约/输出转换，不能叫初始 shared-load。
cp.async 使用 LDGSTS 先填充 shared，再由 LDS/LDSM 读入寄存器；它没有 TMA。
相同几何内五种 cp.async 消费者布局使用固定复制布局，LDGSTS 的静态 opcode/数量相同。

## 完整 kernel 时间（微秒）

表中为三次进程中位数的中位数，完整逐次结果见 analysis.json；不由此做机制归因。

| 几何 / 路径 | vec1 | vec2 | vec4 | vec8 | vec8 within-warp |
|---|---:|---:|---:|---:|---:|
| 32×128/w4, direct | 313.264 | 297.984 | 296.960 | 298.880 | 324.608 |
| 32×128/w4, cp_async | 295.936 | 295.936 | 297.872 | 300.032 | 295.936 |
| 32×64/w8, direct | 279.440 | 285.696 | 368.640 | 421.888 | 315.392 |
| 32×64/w8, cp_async | 279.328 | 282.624 | 395.648 | 455.680 | 301.056 |

普通 Triton 的两个 canonical anchor 实际使用 vec8 的输入布局；
它们的时间与代码单独保存，不把跨 Triton/Gluon pipeline 的差异当纯布局因果效果。

## NCU

四个 32×128/w4 binary 各有硬件与 source 采集，合计 8 份报告。
硬件指标与软件插桩分开，profiler duration 不进入性能表。下列动态指令计数忽略谓词，
另存 predicated-on thread 计数；不能等同为硬件 requests 或阶段时间。

| 路径 / 布局 | 动态 global LDG | 动态 shared-load | shared-load wavefronts |
|---|---:|---:|---:|
| M32_N128_w4:direct:vec1 | 2097152 | 0 | 0 |
| M32_N128_w4:direct:vec8 | 262144 | 851968 | 8.72705e+06 |
| M32_N128_w4:cp_async:vec1 | 0 | 2097152 | 2.09829e+06 |
| M32_N128_w4:cp_async:vec8 | 0 | 1114112 | 9.77966e+06 |

这里证明了无需 TMA 也会出现依赖布局的规约/重排成本；不证明原 issue 的版本回归同样发生于普通 load。
A10 没有执行 TMA，不能替代 H100/B200 上初始 shared-load 的成本校准。

## 修正与证据保留

先前 20261015_gpu0 采集器因未初始化 CompiledKernel handles、使用一维 launch tuple 失败；
完整错误、源代码和已导出文件保留，未产生正式性能数据。修正仅涉及采集器，kernel SHA 不变。
成功采集前后 cache CUBIN 文件数均为 22，native SHA 未变；没有重建 Triton。
未安装逐次 cache listener，因此不把文件数量当作精确的逐条 cache-hit 统计。
NCU 2026.3 默认生成 .ncu-repz。首次 dispatch 对旧扩展名的判断错误保留；
dispatch_v2 识别并复用已成功的原报告，没有重跑这次 GPU profile。
所有 raw、原始 compiler artifacts、原始 NCU reports 和失败记录保持 write-once。
Phase 11 C1/C2/model 仍未执行；本轮没有修改 native compiler 或创建 PR。
