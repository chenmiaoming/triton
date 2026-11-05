# 固定预热与采样间隙复测：旧证据与新证据汇总

状态：**OK**；139,500 个目标计时样本。此文件只汇总已有冻结分析，不调参、不重新选模型、不改变旧数据。

| Suite | Contexts | 原始样本 | U/G0 实用等价（逻辑比较） | Sham 实用等价（逻辑比较） |
|---|---:|---:|---:|---:|
| nonreduction | 16 | 139,500 | 15/16 | 16/16 |

## 原 N4 冻结选择及 N5 大尺寸对照

这里沿用原来预先冻结的候选，没有从新矩阵挑选替代赢家。旧/新两套 CI 属于各自采样条件，百分点变化不是同卡配对因果量。

| Case / frozen candidate | 旧相对 U % [CI95] | 新相对 U % [CI95] | 新相对 G0 % [CI95] | 门槛状态 |
|---|---:|---:|---:|---|
| N01_m11008_n128:GL_coordinated_fully_coalesced | 5.750 [5.021,6.347] | 5.816 [5.388,6.291] | 4.784 [4.393,5.191] | retained |
| N01_m4096_n128:GL_coordinated_fully_coalesced | 2.474 [1.943,3.090] | 3.898 [2.732,5.346] | 3.048 [2.444,3.547] | neither_resolved |
| N02_k1024_p128_g128:GL_packed_compute_vec1 | 4.224 [3.442,4.906] | 3.796 [3.537,4.025] | 5.749 [4.187,6.592] | new_controls_inconclusive |
| N02_k3584_p448_g128:GL_packed_compute_vec1 | 6.422 [6.093,6.793] | 7.126 [6.852,7.426] | 7.074 [6.920,7.262] | retained |
| N03_FA_D128:GL_paired_vec4 | 0.753 [0.554,0.958] | -0.034 [-0.622,0.637] | -0.134 [-0.731,0.462] | neither_resolved |
| N03_FA_D64:GL_paired_vec4 | -0.154 [-0.368,0.096] | 0.334 [0.098,0.800] | 0.356 [0.072,0.897] | neither_resolved |
| N03_MROPE_D128:GL_QK_vec2 | 1.224 [0.593,1.838] | 1.115 [0.575,1.918] | 1.482 [0.882,2.336] | neither_resolved |
| N03_MROPE_D64:GL_K_vec2 | -0.988 [-2.186,-0.108] | 0.621 [0.339,0.845] | 0.000 [0.000,0.000] | neither_resolved |
| N04_h32_s4096_d128:GL_v_input_vec8 | -0.041 [-0.225,0.134] | 0.103 [-0.182,0.381] | 0.228 [-0.003,0.451] | neither_resolved |
| N04_h8_s2048_d128:GL_v_input_vec8 | -0.849 [-1.207,-0.550] | -0.079 [-0.804,0.546] | -0.517 [-1.055,0.040] | neither_resolved |
| N05_t2048_h8_d128:GL_feature_vec4 | 4.274 [3.301,5.203] | 4.102 [3.948,4.216] | 3.089 [2.511,3.667] | neither_resolved |
| N05_t2048_h8_d64:GL_feature_vec4 | 2.711 [2.362,3.078] | 1.799 [1.415,2.188] | 2.092 [1.642,2.471] | neither_resolved |
| N06_h32_d128:GL_e4_heads1 | 0.774 [0.633,0.907] | 0.138 [-0.141,0.317] | 0.224 [-0.060,0.413] | neither_resolved |
| N06_h8_d64:GL_e2_heads4 | -0.547 [-1.566,0.436] | -1.963 [-3.104,-1.232] | -1.398 [-2.166,-0.291] | neither_resolved |
| N01_m4096_n8192:GL_coordinated_fully_coalesced | 7.913 [7.800,8.036] | 7.887 [7.846,7.930] | 7.809 [7.769,7.851] | retained |
| N02_k8192_p1024_g128:GL_packed_compute_vec1 | -1.955 [-2.113,-1.795] | -1.910 [-2.195,-1.604] | -1.751 [-2.093,-1.400] | neither_resolved |

## 解释边界

- This is derived postprocessing of frozen measurements, not a newly selected cost model or sampling protocol.
- A support transition means a difference in pointwise CI resolution between two execution contexts. It does not prove the old gain was caused by idle or that a new gain would hold under every real workload.
- Old N4/G8 CI-supported thresholds use their actual stored confidence intervals. Legacy/anchor published point slopes do not acquire invented old confidence intervals.
- All losing, unchanged and control-inconclusive outcomes remain in the source analyses. Aliases do not create additional independent observations.
- Each suite has three independently allocated formal GPU workers. Raw sample counts and logical comparison counts are not independent GPU counts.
