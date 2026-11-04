# Stage N2 — Non-reduction layout survey

本轮只评价公开源码中固定配置的独立 kernel。U 是冻结公开 Triton DSL 程序，G0 是按 U 实际编译结果重建的 Gluon 基线；候选仅改变显式布局。

## 1. 冻结输入与覆盖

- 协议 SHA256：`23bdfdefb5e4fcb5086baa167feebb8760170801caae299472a23a0123beb20e`。
- 运行 payload SHA256：`715b5b20b539ebe0bd06bd5a332a669b6ad4383593b56ee5787a3ec54a035fec`。
- 14 个 case / 6 个算子族；每个逻辑程序均覆盖 3 个 worker、每 worker 10 个 visit、每 visit 31 个原始样本。
- 逻辑程序 138 个；全部 worker 的实际计时 cell 共 384 个，原始计时样本 119,040 个；展开 alias 后逻辑等效样本 128,340 个。物理样本与 alias 展开数量分别统计。
- U 与 G0 在每个 worker 都单独计时。与其他候选共用同一实际程序的 alias 继承相同观测，不伪装成额外重复实验。
- 已接受编译基线清单：`experiments/tma_reduction_layout/results/phase11/nonreduction/stage_n1/accepted_manifest.json`；SHA256：`7009ffea2a5dea5b38193580ffddcd40e997e489573c015bce69de62d62dd2fb`。编译等价性及 ABI/资源检查以该独立清单和原始编译产物为准。

## 2. 估计量与证据边界

- 主时间：先对每 visit 的 31 个样本取中位数，再对 3×10 个 worker/visit 中位数取算术平均；同一 case 内采用 `interleaved_rounds` 顺序。
- 收益：`100 × (mean(reference) − mean(candidate)) / mean(reference)`；正值表示耗时下降。
- 置信区间：4096 次分层 bootstrap，先重采样 worker、再重采样该 worker 内的 visit；同一 case 的 U/G0/全部候选共用同一组抽样索引。不额外重采样 visit 内原始样本。
- `±3%` 是实际差异门槛。95% 区间完整落入该区间才记为 practical equivalence；完整高于 `+3%` / 低于 `−3%` 才记为 practical improvement / regression，其余为 inconclusive。
- 区间为逐比较区间，未作多重比较校正。按本轮结果选出的 best observed 候选是探索性结果，不能当作未见数据上的确认性胜率。
- 不把固定 shape 的独立 kernel 结果外推为框架、完整 LLM 或整条 attention/KDA 流水的性能；也不凭布局改进直接认定存在版本性能回归。

## 3. U / G0 同轮计时

| Case | U (µs) | G0 (µs) | G0 相对 U 收益 (%) | 95% CI (%) | 分类 |
|---|---:|---:|---:|---|---|
| N01_m4096_n128 | 6.331 | 6.388 | -0.91 | [-1.14, -0.71] | practically_equivalent |
| N01_m11008_n128 | 8.457 | 8.441 | +0.19 | [-0.00, +0.35] | practically_equivalent |
| N02_k3584_p448_g128 | 13.383 | 13.452 | -0.51 | [-0.78, -0.19] | practically_equivalent |
| N02_k1024_p128_g128 | 6.089 | 6.100 | -0.19 | [-0.49, +0.09] | practically_equivalent |
| N03_FA_D64 | 28.141 | 28.157 | -0.06 | [-0.27, +0.16] | practically_equivalent |
| N03_FA_D128 | 50.701 | 50.772 | -0.14 | [-0.29, +0.01] | practically_equivalent |
| N03_MROPE_D64 | 10.753 | 10.758 | -0.05 | [-0.45, +0.36] | practically_equivalent |
| N03_MROPE_D128 | 16.206 | 16.283 | -0.47 | [-0.90, -0.03] | practically_equivalent |
| N04_h8_s2048_d128 | 11.072 | 11.053 | +0.17 | [-0.12, +0.37] | practically_equivalent |
| N04_h32_s4096_d128 | 61.155 | 61.254 | -0.16 | [-0.28, -0.03] | practically_equivalent |
| N05_t2048_h8_d64 | 6.385 | 6.403 | -0.28 | [-0.96, +0.61] | practically_equivalent |
| N05_t2048_h8_d128 | 7.847 | 7.774 | +0.94 | [+0.34, +1.34] | practically_equivalent |
| N06_h8_d64 | 7.506 | 7.515 | -0.11 | [-0.34, +0.13] | practically_equivalent |
| N06_h32_d128 | 28.703 | 28.573 | +0.45 | [+0.12, +0.83] | practically_equivalent |

分类计数：`practically_equivalent` = 14。计时等价是统计分类，不作为预先保证的结果。

## 4. 各 case 的候选结果

以下按观测均值列出最快候选。所有合法候选及相对 U、G0 的配对区间见 `timing.csv` 与 `analysis.json`；最快候选选择不删除其他结果。

| Case | 候选数 | Best observed | 相对 U 收益 (%) | 相对 G0 收益 (%) | vs U 超过 +3% 的候选数 | vs U 低于 −3% 的候选数 |
|---|---:|---|---:|---:|---:|---:|
| N01_m4096_n128 | 8 | GL_coordinated_fully_coalesced | 4.03 | 4.89 | 3 | 0 |
| N01_m11008_n128 | 8 | GL_coordinated_fully_coalesced | 6.03 | 5.85 | 6 | 0 |
| N02_k3584_p448_g128 | 8 | GL_packed_compute_vec1 | 5.48 | 5.96 | 1 | 1 |
| N02_k1024_p128_g128 | 8 | GL_packed_compute_vec1 | 5.73 | 5.91 | 1 | 1 |
| N03_FA_D64 | 7 | GL_paired_vec4 | 0.28 | 0.34 | 0 | 1 |
| N03_FA_D128 | 7 | GL_paired_vec4 | 0.37 | 0.51 | 0 | 1 |
| N03_MROPE_D64 | 8 | GL_K_vec2 | -0.05 | 0.00 | 0 | 2 |
| N03_MROPE_D128 | 8 | GL_QK_vec2 | 2.07 | 2.54 | 0 | 1 |
| N04_h8_s2048_d128 | 8 | GL_v_input_vec8 | 0.62 | 0.44 | 0 | 2 |
| N04_h32_s4096_d128 | 8 | GL_v_input_vec8 | -0.11 | 0.05 | 0 | 1 |
| N05_t2048_h8_d64 | 8 | GL_feature_vec4 | 2.34 | 2.62 | 0 | 6 |
| N05_t2048_h8_d128 | 8 | GL_feature_vec4 | 4.43 | 3.53 | 1 | 5 |
| N06_h8_d64 | 8 | GL_e2_heads4 | -0.63 | -0.51 | 0 | 2 |
| N06_h32_d128 | 8 | GL_e4_heads1 | 0.70 | 0.25 | 0 | 3 |

## 5. 算子族汇总

族内收益均值仅为各 case 等权的描述性汇总，不混合不同 kernel 的绝对耗时，也不代表真实模型调用频率加权的加速。

| 族 | 独立 kernel 范围 | Case 数 | G0 计时等价数 | 有 practical improvement 候选的 case 数（vs U） | Best observed 平均收益（vs U，%） |
|---|---|---:|---:|---:|---:|
| N01 | 量化 scale 的 swizzle / 排列转换 | 2 | 2 | 2 | 5.03 |
| N02 | AWQ 反量化 | 2 | 2 | 2 | 5.60 |
| N03 | RoPE / MRoPE | 4 | 4 | 0 | 0.67 |
| N04 | 固定外部 scale 的 FP8 QKV pack | 2 | 2 | 0 | 0.25 |
| N05 | KDA forward gate | 2 | 2 | 1 | 3.38 |
| N06 | KV cache 写入 | 2 | 2 | 0 | 0.04 |

## 6. 全部候选的逐比较结果

| Case | Variant | 时间 (µs) | vs U 收益 / CI (%) | vs G0 收益 / CI (%) | vs U 分类 |
|---|---|---:|---|---|---|
| N01_m4096_n128 | GL_coordinated_fully_coalesced | 6.076 | +4.03 / [+3.69, +4.39] | +4.89 / [+4.57, +5.24] | practical_improvement |
| N01_m4096_n128 | GL_coordinated_partially_coalesced | 6.117 | +3.37 / [+2.72, +3.88] | +4.24 / [+3.67, +4.66] | inconclusive |
| N01_m4096_n128 | GL_load_s1x1_t8x4_o10 | 6.115 | +3.40 / [+2.96, +4.02] | +4.27 / [+3.83, +4.94] | inconclusive |
| N01_m4096_n128 | GL_load_s1x2_t16x2_o10 | 6.100 | +3.64 / [+3.23, +4.02] | +4.51 / [+4.07, +4.87] | practical_improvement |
| N01_m4096_n128 | GL_load_s2x2_t16x2_o10 | 6.397 | -1.04 / [-1.67, -0.55] | -0.13 / [-0.72, +0.32] | practically_equivalent |
| N01_m4096_n128 | GL_load_s4x1_t8x4_o01 | 6.242 | +1.40 / [+0.77, +2.08] | +2.29 / [+1.67, +2.93] | practically_equivalent |
| N01_m4096_n128 | GL_output_fully_coalesced | 6.167 | +2.58 / [+2.28, +2.86] | +3.46 / [+3.12, +3.82] | practically_equivalent |
| N01_m4096_n128 | GL_output_partially_coalesced | 6.101 | +3.62 / [+3.40, +3.86] | +4.49 / [+4.25, +4.69] | practical_improvement |
| N01_m11008_n128 | GL_coordinated_fully_coalesced | 7.947 | +6.03 / [+5.81, +6.24] | +5.85 / [+5.70, +6.00] | practical_improvement |
| N01_m11008_n128 | GL_coordinated_partially_coalesced | 8.025 | +5.11 / [+4.65, +5.60] | +4.93 / [+4.59, +5.31] | practical_improvement |
| N01_m11008_n128 | GL_load_s1x1_t8x4_o10 | 7.967 | +5.79 / [+5.40, +6.19] | +5.61 / [+5.31, +5.96] | practical_improvement |
| N01_m11008_n128 | GL_load_s1x2_t16x2_o10 | 7.966 | +5.80 / [+5.36, +6.19] | +5.62 / [+5.30, +5.97] | practical_improvement |
| N01_m11008_n128 | GL_load_s2x2_t16x2_o10 | 8.445 | +0.14 / [-0.19, +0.50] | -0.05 / [-0.33, +0.26] | practically_equivalent |
| N01_m11008_n128 | GL_load_s4x1_t8x4_o01 | 8.532 | -0.90 / [-1.14, -0.64] | -1.09 / [-1.29, -0.88] | practically_equivalent |
| N01_m11008_n128 | GL_output_fully_coalesced | 7.953 | +5.95 / [+5.71, +6.20] | +5.78 / [+5.45, +6.05] | practical_improvement |
| N01_m11008_n128 | GL_output_partially_coalesced | 8.020 | +5.16 / [+4.54, +5.69] | +4.98 / [+4.49, +5.47] | practical_improvement |
| N02_k3584_p448_g128 | GL_output_lane_rows2 | 13.520 | -1.02 / [-1.58, -0.54] | -0.51 / [-1.13, +0.04] | practically_equivalent |
| N02_k3584_p448_g128 | GL_output_vec2 | 13.573 | -1.42 / [-2.01, -0.95] | -0.90 / [-1.63, -0.36] | practically_equivalent |
| N02_k3584_p448_g128 | GL_output_vec4 | 13.212 | +1.28 / [+0.65, +1.75] | +1.78 / [+1.07, +2.26] | practically_equivalent |
| N02_k3584_p448_g128 | GL_packed_compute_vec1 | 12.651 | +5.48 / [+4.99, +5.87] | +5.96 / [+5.27, +6.43] | practical_improvement |
| N02_k3584_p448_g128 | GL_packed_compute_vec2 | 13.472 | -0.66 / [-0.96, -0.29] | -0.15 / [-0.49, +0.22] | practically_equivalent |
| N02_k3584_p448_g128 | GL_scale_load_output_encoding | 13.599 | -1.61 / [-1.98, -1.26] | -1.09 / [-1.60, -0.71] | practically_equivalent |
| N02_k3584_p448_g128 | GL_weight_load_vec1 | 16.384 | -22.42 / [-23.08, -21.82] | -21.80 / [-22.41, -21.14] | practical_regression |
| N02_k3584_p448_g128 | GL_weight_load_vec2 | 13.538 | -1.16 / [-1.50, -0.86] | -0.64 / [-1.20, -0.25] | practically_equivalent |
| N02_k1024_p128_g128 | GL_output_lane_rows2 | 6.115 | -0.44 / [-0.78, -0.09] | -0.24 / [-0.57, +0.07] | practically_equivalent |
| N02_k1024_p128_g128 | GL_output_vec2 | 6.042 | +0.77 / [+0.33, +1.14] | +0.96 / [+0.52, +1.51] | practically_equivalent |
| N02_k1024_p128_g128 | GL_output_vec4 | 6.049 | +0.65 / [+0.10, +1.06] | +0.84 / [+0.28, +1.28] | practically_equivalent |
| N02_k1024_p128_g128 | GL_packed_compute_vec1 | 5.740 | +5.73 / [+5.45, +5.99] | +5.91 / [+5.62, +6.21] | practical_improvement |
| N02_k1024_p128_g128 | GL_packed_compute_vec2 | 6.156 | -1.10 / [-1.61, -0.69] | -0.91 / [-1.43, -0.35] | practically_equivalent |
| N02_k1024_p128_g128 | GL_scale_load_output_encoding | 6.069 | +0.32 / [-0.24, +0.76] | +0.51 / [-0.09, +1.04] | practically_equivalent |
| N02_k1024_p128_g128 | GL_weight_load_vec1 | 6.542 | -7.45 / [-7.67, -7.19] | -7.24 / [-7.61, -6.88] | practical_regression |
| N02_k1024_p128_g128 | GL_weight_load_vec2 | 6.118 | -0.49 / [-0.76, -0.27] | -0.30 / [-0.52, -0.12] | practically_equivalent |
| N03_FA_D64 | GL_input_vec1 | 31.826 | -13.10 / [-13.59, -12.73] | -13.03 / [-13.59, -12.64] | practical_regression |
| N03_FA_D64 | GL_input_vec2 | 28.086 | +0.19 / [-0.08, +0.47] | +0.25 / [-0.04, +0.53] | practically_equivalent |
| N03_FA_D64 | GL_input_vec4 | 28.145 | -0.02 / [-0.38, +0.28] | +0.04 / [-0.36, +0.36] | practically_equivalent |
| N03_FA_D64 | GL_input_vec8 | 28.157 | -0.06 / [-0.27, +0.16] | +0.00 / [+0.00, +0.00] | practically_equivalent |
| N03_FA_D64 | GL_paired_vec2 | 28.124 | +0.06 / [-0.29, +0.36] | +0.12 / [-0.29, +0.47] | practically_equivalent |
| N03_FA_D64 | GL_paired_vec4 | 28.061 | +0.28 / [+0.00, +0.61] | +0.34 / [+0.06, +0.68] | practically_equivalent |
| N03_FA_D64 | GL_paired_vec8 | 28.157 | -0.06 / [-0.27, +0.16] | +0.00 / [+0.00, +0.00] | practically_equivalent |
| N03_FA_D128 | GL_input_vec1 | 53.095 | -4.72 / [-4.92, -4.52] | -4.58 / [-4.77, -4.39] | practical_regression |
| N03_FA_D128 | GL_input_vec2 | 50.818 | -0.23 / [-0.39, -0.08] | -0.09 / [-0.26, +0.09] | practically_equivalent |
| N03_FA_D128 | GL_input_vec4 | 50.525 | +0.35 / [+0.18, +0.50] | +0.49 / [+0.29, +0.71] | practically_equivalent |
| N03_FA_D128 | GL_input_vec8 | 50.772 | -0.14 / [-0.29, +0.01] | +0.00 / [+0.00, +0.00] | practically_equivalent |
| N03_FA_D128 | GL_paired_vec2 | 51.001 | -0.59 / [-0.80, -0.43] | -0.45 / [-0.67, -0.25] | practically_equivalent |
| N03_FA_D128 | GL_paired_vec4 | 50.515 | +0.37 / [+0.16, +0.52] | +0.51 / [+0.24, +0.76] | practically_equivalent |
| N03_FA_D128 | GL_paired_vec8 | 50.772 | -0.14 / [-0.29, +0.01] | +0.00 / [+0.00, +0.00] | practically_equivalent |
| N03_MROPE_D64 | GL_K_vec2 | 10.758 | -0.05 / [-0.45, +0.36] | +0.00 / [+0.00, +0.00] | practically_equivalent |
| N03_MROPE_D64 | GL_K_vec4 | 10.858 | -0.97 / [-1.37, -0.66] | -0.92 / [-1.37, -0.52] | practically_equivalent |
| N03_MROPE_D64 | GL_QK_vec2 | 10.813 | -0.56 / [-1.34, +0.31] | -0.51 / [-1.35, +0.14] | practically_equivalent |
| N03_MROPE_D64 | GL_QK_vec4 | 11.303 | -5.12 / [-5.65, -4.55] | -5.07 / [-5.47, -4.67] | practical_regression |
| N03_MROPE_D64 | GL_Q_vec1 | 10.837 | -0.78 / [-1.05, -0.49] | -0.73 / [-1.09, -0.42] | practically_equivalent |
| N03_MROPE_D64 | GL_Q_vec2 | 10.813 | -0.56 / [-1.34, +0.31] | -0.51 / [-1.35, +0.14] | practically_equivalent |
| N03_MROPE_D64 | GL_Q_vec4 | 11.215 | -4.30 / [-4.94, -3.42] | -4.24 / [-4.71, -3.67] | practical_regression |
| N03_MROPE_D64 | GL_Q_vec8 | 10.758 | -0.05 / [-0.45, +0.36] | +0.00 / [+0.00, +0.00] | practically_equivalent |
| N03_MROPE_D128 | GL_K_vec2 | 16.619 | -2.55 / [-2.96, -2.24] | -2.06 / [-2.46, -1.73] | practically_equivalent |
| N03_MROPE_D128 | GL_K_vec4 | 16.283 | -0.47 / [-0.90, -0.03] | +0.00 / [+0.00, +0.00] | practically_equivalent |
| N03_MROPE_D128 | GL_QK_vec2 | 15.870 | +2.07 / [+1.39, +2.66] | +2.54 / [+1.92, +3.14] | practically_equivalent |
| N03_MROPE_D128 | GL_QK_vec4 | 16.063 | +0.88 / [-0.16, +1.68] | +1.35 / [+0.44, +2.08] | practically_equivalent |
| N03_MROPE_D128 | GL_Q_vec1 | 16.378 | -1.06 / [-2.35, +0.02] | -0.58 / [-2.01, +0.61] | practically_equivalent |
| N03_MROPE_D128 | GL_Q_vec2 | 17.228 | -6.31 / [-6.75, -5.85] | -5.80 / [-6.12, -5.48] | practical_regression |
| N03_MROPE_D128 | GL_Q_vec4 | 16.063 | +0.88 / [-0.16, +1.68] | +1.35 / [+0.44, +2.08] | practically_equivalent |
| N03_MROPE_D128 | GL_Q_vec8 | 16.283 | -0.47 / [-0.90, -0.03] | +0.00 / [+0.00, +0.00] | practically_equivalent |
| N04_h8_s2048_d128 | GL_v_compute_input | 11.510 | -3.96 / [-4.31, -3.49] | -4.14 / [-4.44, -3.84] | practical_regression |
| N04_h8_s2048_d128 | GL_v_compute_token_vec32 | 11.245 | -1.56 / [-1.78, -1.35] | -1.74 / [-2.05, -1.29] | practically_equivalent |
| N04_h8_s2048_d128 | GL_v_compute_token_vec4 | 11.321 | -2.24 / [-2.46, -2.06] | -2.42 / [-2.60, -2.22] | practically_equivalent |
| N04_h8_s2048_d128 | GL_v_compute_token_vec8 | 11.237 | -1.49 / [-1.80, -1.23] | -1.67 / [-2.05, -1.25] | practically_equivalent |
| N04_h8_s2048_d128 | GL_v_input_vec1 | 11.701 | -5.68 / [-6.00, -5.40] | -5.87 / [-6.33, -5.49] | practical_regression |
| N04_h8_s2048_d128 | GL_v_input_vec2 | 11.422 | -3.16 / [-3.49, -2.86] | -3.34 / [-3.83, -2.92] | inconclusive |
| N04_h8_s2048_d128 | GL_v_input_vec4 | 11.187 | -1.04 / [-1.29, -0.76] | -1.22 / [-1.44, -1.00] | practically_equivalent |
| N04_h8_s2048_d128 | GL_v_input_vec8 | 11.004 | +0.62 / [+0.39, +0.83] | +0.44 / [+0.18, +0.70] | practically_equivalent |
| N04_h32_s4096_d128 | GL_v_compute_input | 61.977 | -1.34 / [-1.48, -1.21] | -1.18 / [-1.29, -1.08] | practically_equivalent |
| N04_h32_s4096_d128 | GL_v_compute_token_vec32 | 62.109 | -1.56 / [-1.72, -1.40] | -1.39 / [-1.53, -1.27] | practically_equivalent |
| N04_h32_s4096_d128 | GL_v_compute_token_vec4 | 61.936 | -1.28 / [-1.46, -1.08] | -1.11 / [-1.26, -0.96] | practically_equivalent |
| N04_h32_s4096_d128 | GL_v_compute_token_vec8 | 62.213 | -1.73 / [-1.93, -1.50] | -1.57 / [-1.75, -1.36] | practically_equivalent |
| N04_h32_s4096_d128 | GL_v_input_vec1 | 64.647 | -5.71 / [-5.90, -5.52] | -5.54 / [-5.70, -5.39] | practical_regression |
| N04_h32_s4096_d128 | GL_v_input_vec2 | 62.126 | -1.59 / [-1.74, -1.41] | -1.42 / [-1.55, -1.27] | practically_equivalent |
| N04_h32_s4096_d128 | GL_v_input_vec4 | 61.370 | -0.35 / [-0.46, -0.24] | -0.19 / [-0.33, -0.03] | practically_equivalent |
| N04_h32_s4096_d128 | GL_v_input_vec8 | 61.222 | -0.11 / [-0.26, +0.06] | +0.05 / [-0.08, +0.19] | practically_equivalent |
| N05_t2048_h8_d64 | GL_feature_vec1 | 6.773 | -6.08 / [-7.04, -4.80] | -5.78 / [-6.15, -5.42] | practical_regression |
| N05_t2048_h8_d64 | GL_feature_vec2 | 6.740 | -5.56 / [-6.07, -4.89] | -5.26 / [-5.61, -4.94] | practical_regression |
| N05_t2048_h8_d64 | GL_feature_vec4 | 6.236 | +2.34 / [+1.71, +3.31] | +2.62 / [+2.39, +2.86] | inconclusive |
| N05_t2048_h8_d64 | GL_feature_vec8 | 6.489 | -1.62 / [-2.21, -0.73] | -1.33 / [-1.66, -1.03] | practically_equivalent |
| N05_t2048_h8_d64 | GL_token_vec1 | 10.138 | -58.77 / [-59.20, -58.26] | -58.32 / [-59.26, -57.50] | practical_regression |
| N05_t2048_h8_d64 | GL_token_vec2 | 8.030 | -25.76 / [-26.20, -25.41] | -25.40 / [-26.29, -24.45] | practical_regression |
| N05_t2048_h8_d64 | GL_token_vec4 | 7.031 | -10.12 / [-10.49, -9.78] | -9.81 / [-10.81, -8.81] | practical_regression |
| N05_t2048_h8_d64 | GL_token_vec8 | 7.045 | -10.34 / [-10.97, -9.54] | -10.03 / [-10.33, -9.80] | practical_regression |
| N05_t2048_h8_d128 | GL_feature_vec1 | 8.180 | -4.24 / [-4.75, -3.89] | -5.23 / [-5.41, -5.04] | practical_regression |
| N05_t2048_h8_d128 | GL_feature_vec2 | 8.154 | -3.90 / [-4.27, -3.60] | -4.88 / [-5.17, -4.55] | practical_regression |
| N05_t2048_h8_d128 | GL_feature_vec4 | 7.500 | +4.43 / [+4.09, +4.73] | +3.53 / [+3.31, +3.84] | practical_improvement |
| N05_t2048_h8_d128 | GL_feature_vec8 | 7.692 | +1.98 / [+1.57, +2.39] | +1.06 / [+0.78, +1.31] | practically_equivalent |
| N05_t2048_h8_d128 | GL_token_vec1 | 15.284 | -94.77 / [-95.35, -94.01] | -96.61 / [-97.63, -95.67] | practical_regression |
| N05_t2048_h8_d128 | GL_token_vec2 | 11.299 | -43.99 / [-44.39, -43.59] | -45.35 / [-46.14, -44.21] | practical_regression |
| N05_t2048_h8_d128 | GL_token_vec4 | 8.893 | -13.32 / [-13.88, -12.63] | -14.39 / [-15.23, -13.81] | practical_regression |
| N05_t2048_h8_d128 | GL_token_vec8 | 8.108 | -3.32 / [-4.65, -2.27] | -4.29 / [-5.00, -3.60] | inconclusive |
| N06_h8_d64 | GL_e1_heads1 | 7.730 | -2.98 / [-3.25, -2.76] | -2.87 / [-3.00, -2.71] | inconclusive |
| N06_h8_d64 | GL_e1_heads4 | 8.042 | -7.13 / [-7.52, -6.58] | -7.01 / [-7.51, -6.41] | practical_regression |
| N06_h8_d64 | GL_e2_heads1 | 7.699 | -2.57 / [-2.83, -2.30] | -2.46 / [-2.90, -2.11] | practically_equivalent |
| N06_h8_d64 | GL_e2_heads4 | 7.553 | -0.63 / [-0.92, -0.36] | -0.51 / [-0.72, -0.35] | practically_equivalent |
| N06_h8_d64 | GL_e4_heads1 | 7.684 | -2.37 / [-2.74, -2.08] | -2.26 / [-2.50, -1.98] | practically_equivalent |
| N06_h8_d64 | GL_e4_heads2 | 7.555 | -0.65 / [-1.09, -0.31] | -0.54 / [-0.83, -0.31] | practically_equivalent |
| N06_h8_d64 | GL_e8_heads1 | 8.194 | -9.17 / [-9.69, -8.60] | -9.04 / [-9.76, -8.39] | practical_regression |
| N06_h8_d64 | GL_e8_heads2 | 7.709 | -2.70 / [-3.03, -2.44] | -2.58 / [-2.76, -2.39] | inconclusive |
| N06_h32_d128 | GL_e1_heads1 | 30.073 | -4.77 / [-5.02, -4.51] | -5.25 / [-5.52, -4.99] | practical_regression |
| N06_h32_d128 | GL_e1_heads4 | 34.695 | -20.88 / [-21.15, -20.56] | -21.43 / [-21.84, -20.96] | practical_regression |
| N06_h32_d128 | GL_e2_heads1 | 28.514 | +0.66 / [+0.38, +1.05] | +0.21 / [-0.07, +0.49] | practically_equivalent |
| N06_h32_d128 | GL_e2_heads4 | 28.934 | -0.81 / [-1.14, -0.47] | -1.27 / [-1.68, -0.95] | practically_equivalent |
| N06_h32_d128 | GL_e4_heads1 | 28.501 | +0.70 / [+0.30, +1.03] | +0.25 / [-0.20, +0.76] | practically_equivalent |
| N06_h32_d128 | GL_e4_heads2 | 28.531 | +0.60 / [+0.31, +0.93] | +0.15 / [-0.04, +0.36] | practically_equivalent |
| N06_h32_d128 | GL_e8_heads1 | 30.774 | -7.22 / [-7.37, -7.07] | -7.71 / [-8.04, -7.39] | practical_regression |
| N06_h32_d128 | GL_e8_heads2 | 28.623 | +0.28 / [+0.01, +0.53] | -0.18 / [-0.43, +0.06] | practically_equivalent |

## 7. 各 case 的适用范围

- `N01_m4096_n128`：Optional torchao Triton scale-rearrangement backend; no scale calculation.
- `N01_m11008_n128`：Optional torchao Triton scale-rearrangement backend; no scale calculation.
- `N02_k3584_p448_g128`：Optional VLLM_USE_TRITON_AWQ dequantization backend; not AWQ GEMM.
- `N02_k1024_p128_g128`：Optional VLLM_USE_TRITON_AWQ dequantization backend; not AWQ GEMM.
- `N03_FA_D64`：Standalone pinned source kernel, fixed case and configuration.
- `N03_FA_D128`：Standalone pinned source kernel, fixed case and configuration.
- `N03_MROPE_D64`：Standalone pinned source kernel, fixed case and configuration.
- `N03_MROPE_D128`：Standalone pinned source kernel, fixed case and configuration.
- `N04_h8_s2048_d128`：Standalone fixed-external-scale SM120-origin FP8 pack. H100 is a portability experiment, not its original attention backend.
- `N04_h32_s4096_d128`：Standalone fixed-external-scale SM120-origin FP8 pack. H100 is a portability experiment, not its original attention backend.
- `N05_t2048_h8_d64`：Forward standalone KDA gate with A/bias, no beta, no chunk-cumsum or backward. FP32 output.
- `N05_t2048_h8_d128`：Forward standalone KDA gate with A/bias, no beta, no chunk-cumsum or backward. FP32 output.
- `N06_h8_d64`：Standalone pinned source kernel, fixed case and configuration.
- `N06_h32_d128`：Standalone pinned source kernel, fixed case and configuration.

## 8. 保留证据

- 原始计时、编译产物、失败尝试与自适应决策记录保留原样；本报告只新增 derived analysis。
- `analysis.json`：输入文件 SHA、精确覆盖审计、环境、每 worker/visit 中位数、配对 bootstrap 区间与实际物理代表程序。
- `timing.csv`：全部逻辑程序的可复核结果表。
- 硬件机制须结合独立的 NCU/SASS/IR 证据逐项讨论；本时间分析本身不把快慢归因于某一计数器或 pass。
