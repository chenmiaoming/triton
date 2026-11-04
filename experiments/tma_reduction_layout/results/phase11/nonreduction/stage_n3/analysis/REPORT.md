# Stage N3 — 静态预选布局对照的 NCU 证据

覆盖 7 个 kernel 子类型；60 份实际报告，alias 展开后 60 个逻辑报告。所有对照由布局机制静态预选，未按候选耗时挑选。

协议 SHA256：`1288a7e4c2625cbeb5fd9c349afa3909c0518a6f42997745b2dfc90b44850d78`。Payload SHA256：`715b5b20b539ebe0bd06bd5a332a669b6ad4383593b56ee5787a3ec54a035fec`。

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
| N01_m4096_n128 | True | True | True | 128 | 128 |
| N02_k3584_p448_g128 | True | True | True | 2168 | 2168 |
| N03_FA_D128 | True | True | True | 660 | 660 |
| N03_MROPE_D128 | True | True | True | 764 | 764 |
| N04_h32_s4096_d128 | True | True | True | 12288 | 12288 |
| N05_t2048_h8_d128 | True | True | True | 2264 | 2264 |
| N06_h32_d128 | True | True | True | 320 | 320 |

## 3. CTA0 的软件动态指令

| Case | Variant（含 alias） | 总指令 | LDG | STG | LDS/LDSM | STS/STSM | SHFL | BAR | LDL+STL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N01_m4096_n128 | U | 128 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| N01_m4096_n128 | G0 | 128 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| N01_m4096_n128 | GL_output_fully_coalesced | 180 | 4 | 4 | 4 | 4 | 0 | 4 | 0 |
| N01_m4096_n128 | GL_coordinated_fully_coalesced | 136 | 4 | 4 | 0 | 0 | 0 | 0 | 0 |
| N02_k3584_p448_g128 | U | 2168 | 16 | 32 | 36 | 36 | 0 | 68 | 0 |
| N02_k3584_p448_g128 | G0 | 2168 | 16 | 32 | 36 | 36 | 0 | 68 | 0 |
| N02_k3584_p448_g128 | GL_packed_compute_vec1 | 1928 | 40 | 32 | 4 | 4 | 0 | 4 | 0 |
| N02_k3584_p448_g128 | GL_output_vec4 | 2308 | 16 | 64 | 48 | 36 | 0 | 36 | 0 |
| N03_FA_D128 | U | 660 | 16 | 8 | 0 | 0 | 0 | 0 | 0 |
| N03_FA_D128 | G0 | 660 | 16 | 8 | 0 | 0 | 0 | 0 | 0 |
| N03_FA_D128 | GL_input_vec1 | 1060 | 72 | 8 | 16 | 16 | 0 | 12 | 0 |
| N03_FA_D128 | GL_paired_vec4 | 916 | 24 | 8 | 24 | 24 | 0 | 20 | 0 |
| N03_MROPE_D128 | U | 764 | 84 | 24 | 0 | 0 | 0 | 0 | 0 |
| N03_MROPE_D128 | G0 | 764 | 84 | 24 | 0 | 0 | 0 | 0 | 0 |
| N03_MROPE_D128 | GL_Q_vec1 | 1440 | 196 | 24 | 24 | 80 | 0 | 28 | 0 |
| N03_MROPE_D128 | GL_QK_vec4 | 1044 | 100 | 24 | 16 | 16 | 32 | 12 | 0 |
| N04_h32_s4096_d128 | U | 12288 | 120 | 48 | 32 | 32 | 0 | 8 | 0 |
| N04_h32_s4096_d128 | G0 | 12288 | 120 | 48 | 32 | 32 | 0 | 8 | 0 |
| N04_h32_s4096_d128 | GL_v_input_vec2 | 13248 | 216 | 48 | 32 | 32 | 0 | 24 | 0 |
| N04_h32_s4096_d128 | GL_v_compute_input | 13176 | 120 | 48 | 64 | 256 | 0 | 8 | 0 |
| N05_t2048_h8_d128 | U | 2264 | 32 | 32 | 16 | 16 | 0 | 8 | 0 |
| N05_t2048_h8_d128 | G0 | 2264 | 32 | 32 | 16 | 16 | 0 | 8 | 0 |
| N05_t2048_h8_d128 | GL_feature_vec2 | 2792 | 80 | 32 | 32 | 32 | 32 | 24 | 0 |
| N05_t2048_h8_d128 | GL_token_vec2 | 2880 | 144 | 32 | 144 | 160 | 0 | 24 | 0 |
| N06_h32_d128 | U | 320 | 12 | 8 | 0 | 0 | 0 | 0 | 0 |
| N06_h32_d128 | G0 | 320 | 12 | 8 | 0 | 0 | 0 | 0 | 0 |
| N06_h32_d128 | GL_e2_heads1 | 536 | 36 | 30 | 0 | 0 | 0 | 0 | 0 |
| N06_h32_d128 | GL_e2_heads4 | 404 | 20 | 16 | 0 | 0 | 0 | 0 | 0 |

## 4. 完整 grid 的请求与共享内存服务

下表仅展示所选 additive counter 的每 CTA 值。完整 raw 名称、单位、数值、stall 百分比、DRAM/L2 利用率及资源数据保留于 `summary.json`、`summary.csv`、`metrics.csv`。这些列不做延迟加权。

| Case | Variant（含 alias） | Mode | Global LD sectors/CTA | Global ST sectors/CTA | Shared LD WF/CTA | Shared LD conflicts/CTA | n_regs | q |
|---|---|---|---:|---:|---:|---:|---:|---:|
| N01_m4096_n128 | U | warm_application | 128.000 | 64.000 | 0.000 | 0.000 | 14 | 16 |
| N01_m4096_n128 | G0 | warm_application | 128.000 | 64.000 | 0.000 | 0.000 | 14 | 16 |
| N01_m4096_n128 | GL_output_fully_coalesced | warm_application | 128.000 | 16.000 | 4.000 | 0.000 | 12 | 16 |
| N01_m4096_n128 | GL_coordinated_fully_coalesced | warm_application | 128.000 | 16.000 | 0.000 | 0.000 | 12 | 16 |
| N02_k3584_p448_g128 | U | warm_application | 160.000 | 512.000 | 154.126 | 10.126 | 72 | 7 |
| N02_k3584_p448_g128 | G0 | warm_application | 160.000 | 512.000 | 154.045 | 10.045 | 72 | 7 |
| N02_k3584_p448_g128 | GL_packed_compute_vec1 | warm_application | 160.000 | 512.000 | 16.226 | 0.226 | 40 | 12 |
| N02_k3584_p448_g128 | GL_output_vec4 | warm_application | 160.000 | 512.000 | 148.318 | 4.318 | 72 | 7 |
| N03_FA_D128 | U | warm_application | 192.000 | 128.000 | 0.000 | 0.000 | 32 | 16 |
| N03_FA_D128 | G0 | warm_application | 192.000 | 128.000 | 0.000 | 0.000 | 32 | 16 |
| N03_FA_D128 | GL_input_vec1 | warm_application | 192.000 | 128.000 | 65.091 | 1.091 | 34 | 12 |
| N03_FA_D128 | GL_paired_vec4 | warm_application | 192.000 | 128.000 | 65.667 | 1.667 | 32 | 16 |
| N03_MROPE_D128 | U | warm_application | 412.000 | 320.000 | 0.000 | 0.000 | 40 | 12 |
| N03_MROPE_D128 | G0 | warm_application | 412.000 | 320.000 | 0.000 | 0.000 | 40 | 12 |
| N03_MROPE_D128 | GL_Q_vec1 | warm_application | 412.000 | 320.000 | 75.300 | 3.300 | 44 | 10 |
| N03_MROPE_D128 | GL_QK_vec4 | warm_application | 412.000 | 320.000 | 67.813 | 3.813 | 48 | 10 |
| N04_h32_s4096_d128 | U | warm_application | 3096.000 | 768.000 | 128.170 | 0.170 | 71 | 3 |
| N04_h32_s4096_d128 | U | cold_isolated | 3096.000 | 768.000 | 128.206 | 0.206 | 71 | 3 |
| N04_h32_s4096_d128 | G0 | warm_application | 3096.000 | 768.000 | 128.418 | 0.418 | 71 | 3 |
| N04_h32_s4096_d128 | G0 | cold_isolated | 3096.000 | 768.000 | 128.275 | 0.275 | 71 | 3 |
| N04_h32_s4096_d128 | GL_v_input_vec2 | warm_application | 2584.000 | 768.000 | 129.868 | 1.868 | 69 | 3 |
| N04_h32_s4096_d128 | GL_v_input_vec2 | cold_isolated | 2584.000 | 768.000 | 129.200 | 1.200 | 69 | 3 |
| N04_h32_s4096_d128 | GL_v_compute_input | warm_application | 3096.000 | 768.000 | 64.608 | 0.608 | 74 | 3 |
| N04_h32_s4096_d128 | GL_v_compute_input | cold_isolated | 3096.000 | 768.000 | 64.523 | 0.523 | 74 | 3 |
| N05_t2048_h8_d128 | U | warm_application | 392.000 | 512.000 | 64.000 | 0.000 | 48 | 5 |
| N05_t2048_h8_d128 | G0 | warm_application | 392.000 | 512.000 | 64.000 | 0.000 | 48 | 5 |
| N05_t2048_h8_d128 | GL_feature_vec2 | warm_application | 392.000 | 512.000 | 128.229 | 0.229 | 48 | 5 |
| N05_t2048_h8_d128 | GL_token_vec2 | warm_application | 2184.000 | 512.000 | 160.000 | 0.000 | 42 | 5 |
| N06_h32_d128 | U | warm_application | 68.000 | 64.000 | 0.000 | 0.000 | 20 | 16 |
| N06_h32_d128 | G0 | warm_application | 68.000 | 64.000 | 0.000 | 0.000 | 20 | 16 |
| N06_h32_d128 | GL_e2_heads1 | warm_application | 132.000 | 64.000 | 0.000 | 0.000 | 26 | 16 |
| N06_h32_d128 | GL_e2_heads4 | warm_application | 68.000 | 64.000 | 0.000 | 0.000 | 22 | 16 |

## 5. 静态对照的适用范围

- `N01_scale_swizzle` / `N01_m4096_n128`：Compare output-friendly ownership alone with coordinated input/output ownership; isolate the conversion cost versus store request grouping.
- `N02_awq_dequantize` / `N02_k3584_p448_g128`：Contrast packed-input/expanded-compute ownership with output-only vec4 ownership; inspect load, unpacking, layout conversion and store instructions separately.
- `N03_flash_rope` / `N03_FA_D128`：Compare input-only vec1 ownership with coordinated input/paired-compute vec4; retain the original rotary arithmetic, output layout and launch configuration.
- `N03_mrope` / `N03_MROPE_D128`：Compare Q-only vec1 with coordinated Q/K vec4 input ownership; inspect Q/K handling without changing the in-place rotary operator or trig/output layouts.
- `N04_fp8_pack` / `N04_h32_s4096_d128`：Change V input ownership alone versus keeping V computation in input ownership; inspect BF16 conversion, FP8 quantization, transpose and store grouping. Q/K are fixed.
- `N05_kda_gate` / `N05_t2048_h8_d128`：Hold vec2 fixed and reverse feature-versus-token ownership of input/compute; output/bias ownership, exact elementwise arithmetic and U-only tuned BT/W/stages remain fixed.
- `N06_kv_cache` / `N06_h32_d128`：Hold sizePerThread vec2 fixed and redistribute lanes over one versus four heads; inspect token-major loads and HND-strided paged cache stores.

原始 `.ncu-rep`、raw/source 导出、NCU 版本、绑定与正确性记录均按 SHA 保留。U/G0 的软件计数一致与否如实展示；任何不一致需要单独调查，不在分析器中改写。对完整框架、LLM 端到端收益、跨设备泛化或版本回归均不作结论。
