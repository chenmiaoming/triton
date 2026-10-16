# Stage C2b — H100 calibration

90 core programs plus 40 parameter programs; 40 new logical compilations in this batch.
New graph event samples: 221,700; 3695 physical cells per worker; three worker allocations.
New NCU reports: 26; C1/C1.5 original 128 reports and CUBINs remain unchanged.
Resource boundaries excluded from regular timing/fit: M32_N128_w4:vec2:n_first:ML:D2, M32_N128_w4:vec2:within:ML:D2. Correctness and artifacts are retained for these programs.

## Measurement and units

Repeated subgraphs use K16/64/256 OLS, with K1 retained separately. Three grids fit steady service per SM per tile; grid is calibration provenance, absent from static predictor features. MO is one-shot and uses T versus grid. All samples are single-kernel CUDA Graph event intervals. NCU duration is not used.
Independent unchanged TMA controls run at full grids. Their time is normalized in the same tile/SM units. No 26.6 us offset appears in the analysis.
The preferred shared carveout is set and read back as 100% for controlled q comparisons; these fresh curves have their own measurement conditions. Old timing samples are not pooled into them.
Maximum steady-K fit residual: 3.86% of the largest measured point. Maximum grid-fit residual: 1.04%. Adjacent slopes and all worker estimates remain in analysis.json.

## Natural service curves

| Program | Median ns/tile/SM | Worker range ns |
|---|---:|---:|
| M32_N128_w4:vec1:n_first:MC:D1 | 68.265 | 67.608–68.336 |
| M32_N128_w4:vec1:n_first:ML:D1 | 64.982 | 64.594–65.102 |
| M32_N128_w4:vec1:n_first:ML:D2 | 65.381 | 65.173–65.470 |
| M32_N128_w4:vec1:n_first:MO:D1 | 79.500 | 77.500–80.500 |
| M32_N128_w4:vec1:n_first:MR:D1 | 93.235 | 93.097–94.007 |
| M32_N128_w4:vec1:n_first:MR:D2 | 96.379 | 95.595–96.393 |
| M32_N128_w4:vec2:n_first:MC:D1 | 75.199 | 74.577–75.266 |
| M32_N128_w4:vec2:n_first:ML:D1 | 49.249 | 49.220–49.720 |
| M32_N128_w4:vec2:n_first:MO:D1 | 80.500 | 79.500–81.500 |
| M32_N128_w4:vec2:n_first:MR:D1 | 84.334 | 83.909–84.447 |
| M32_N128_w4:vec2:n_first:MR:D2 | 84.285 | 83.693–84.659 |
| M32_N128_w4:vec2:within:MC:D1 | 76.358 | 75.836–76.584 |
| M32_N128_w4:vec2:within:ML:D1 | 65.333 | 63.895–65.992 |
| M32_N128_w4:vec2:within:MO:D1 | 80.250 | 79.000–81.250 |
| M32_N128_w4:vec2:within:MR:D1 | 87.330 | 87.137–87.569 |
| M32_N128_w4:vec2:within:MR:D2 | 80.836 | 80.609–81.523 |
| M32_N128_w4:vec4:n_first:MC:D1 | 107.727 | 107.089–107.754 |
| M32_N128_w4:vec4:n_first:ML:D1 | 52.994 | 52.721–53.340 |
| M32_N128_w4:vec4:n_first:ML:D2 | 51.434 | 51.423–51.619 |
| M32_N128_w4:vec4:n_first:MO:D1 | 80.000 | 78.500–80.250 |
| M32_N128_w4:vec4:n_first:MR:D1 | 96.470 | 95.785–96.775 |
| M32_N128_w4:vec4:n_first:MR:D2 | 95.474 | 94.908–95.595 |
| M32_N128_w4:vec4:within:MC:D1 | 97.350 | 96.738–97.869 |
| M32_N128_w4:vec4:within:ML:D1 | 63.771 | 63.608–63.795 |
| M32_N128_w4:vec4:within:ML:D2 | 63.576 | 63.185–63.764 |
| M32_N128_w4:vec4:within:MO:D1 | 79.000 | 78.000–81.500 |
| M32_N128_w4:vec4:within:MR:D1 | 100.994 | 100.462–101.240 |
| M32_N128_w4:vec4:within:MR:D2 | 95.061 | 94.780–95.068 |
| M32_N128_w4:vec8:n_first:MC:D1 | 191.273 | 189.556–191.414 |
| M32_N128_w4:vec8:n_first:ML:D1 | 55.078 | 54.665–55.350 |
| M32_N128_w4:vec8:n_first:ML:D2 | 44.051 | 43.902–44.119 |
| M32_N128_w4:vec8:n_first:MO:D1 | 94.250 | 90.250–95.000 |
| M32_N128_w4:vec8:n_first:MR:D1 | 155.247 | 154.361–156.013 |
| M32_N128_w4:vec8:n_first:MR:D2 | 149.116 | 148.697–149.766 |
| M32_N128_w4:vec8:within:MC:D1 | 137.927 | 137.091–138.083 |
| M32_N128_w4:vec8:within:ML:D1 | 62.927 | 62.697–63.139 |
| M32_N128_w4:vec8:within:ML:D2 | 64.629 | 64.077–64.776 |
| M32_N128_w4:vec8:within:MO:D1 | 79.000 | 79.000–81.750 |
| M32_N128_w4:vec8:within:MR:D1 | 116.137 | 115.321–116.153 |
| M32_N128_w4:vec8:within:MR:D2 | 118.018 | 117.922–118.168 |
| M32_N64_w8:vec1:n_first:MC:D1 | 71.154 | 70.889–71.188 |
| M32_N64_w8:vec1:n_first:ML:D1 | 31.922 | 31.826–31.988 |
| M32_N64_w8:vec1:n_first:ML:D2 | 21.327 | 21.292–21.369 |
| M32_N64_w8:vec1:n_first:MO:D1 | 79.500 | 78.750–80.250 |
| M32_N64_w8:vec1:n_first:MR:D1 | 59.203 | 58.659–59.241 |
| M32_N64_w8:vec1:n_first:MR:D2 | 58.802 | 58.690–58.814 |
| M32_N64_w8:vec1:within:MC:D1 | 58.250 | 58.135–58.372 |
| M32_N64_w8:vec1:within:ML:D1 | 32.199 | 32.194–32.222 |
| M32_N64_w8:vec1:within:ML:D2 | 21.432 | 21.405–21.507 |
| M32_N64_w8:vec1:within:MO:D1 | 79.250 | 79.250–80.750 |
| M32_N64_w8:vec1:within:MR:D1 | 53.995 | 53.578–54.193 |
| M32_N64_w8:vec1:within:MR:D2 | 52.674 | 52.353–52.863 |
| M32_N64_w8:vec2:n_first:MC:D1 | 93.971 | 93.245–94.404 |
| M32_N64_w8:vec2:n_first:ML:D1 | 27.962 | 27.804–28.096 |
| M32_N64_w8:vec2:n_first:ML:D2 | 28.826 | 28.684–28.912 |
| M32_N64_w8:vec2:n_first:MO:D1 | 78.500 | 77.500–80.750 |
| M32_N64_w8:vec2:n_first:MR:D1 | 73.909 | 73.755–74.152 |
| M32_N64_w8:vec2:n_first:MR:D2 | 71.607 | 71.392–71.904 |
| M32_N64_w8:vec2:within:MC:D1 | 61.357 | 61.162–61.722 |
| M32_N64_w8:vec2:within:ML:D1 | 28.455 | 28.413–28.535 |
| M32_N64_w8:vec2:within:ML:D2 | 26.912 | 26.876–26.956 |
| M32_N64_w8:vec2:within:MO:D1 | 79.500 | 78.250–80.250 |
| M32_N64_w8:vec2:within:MR:D1 | 60.012 | 59.882–60.474 |
| M32_N64_w8:vec2:within:MR:D2 | 58.344 | 58.236–58.348 |
| M32_N64_w8:vec4:n_first:MC:D1 | 186.942 | 185.536–187.452 |
| M32_N64_w8:vec4:n_first:ML:D1 | 25.922 | 25.881–25.973 |
| M32_N64_w8:vec4:n_first:ML:D2 | 27.914 | 27.855–28.208 |
| M32_N64_w8:vec4:n_first:MO:D1 | 80.750 | 80.250–81.000 |
| M32_N64_w8:vec4:n_first:MR:D1 | 142.757 | 141.783–143.053 |
| M32_N64_w8:vec4:n_first:MR:D2 | 143.246 | 142.941–143.945 |
| M32_N64_w8:vec4:within:MC:D1 | 111.391 | 111.061–111.716 |
| M32_N64_w8:vec4:within:ML:D1 | 26.153 | 26.147–26.174 |
| M32_N64_w8:vec4:within:ML:D2 | 27.860 | 27.831–28.117 |
| M32_N64_w8:vec4:within:MO:D1 | 80.250 | 75.500–80.500 |
| M32_N64_w8:vec4:within:MR:D1 | 85.535 | 85.336–85.916 |
| M32_N64_w8:vec4:within:MR:D2 | 83.307 | 83.250–83.748 |
| M32_N64_w8:vec8:n_first:MC:D1 | 377.211 | 374.776–378.439 |
| M32_N64_w8:vec8:n_first:ML:D1 | 28.147 | 27.991–28.168 |
| M32_N64_w8:vec8:n_first:ML:D2 | 29.011 | 28.885–29.044 |
| M32_N64_w8:vec8:n_first:MO:D1 | 113.750 | 113.750–117.250 |
| M32_N64_w8:vec8:n_first:MR:D1 | 314.403 | 311.861–314.569 |
| M32_N64_w8:vec8:n_first:MR:D2 | 312.901 | 311.953–314.621 |
| M32_N64_w8:vec8:within:MC:D1 | 227.864 | 227.449–227.954 |
| M32_N64_w8:vec8:within:ML:D1 | 27.085 | 26.903–27.150 |
| M32_N64_w8:vec8:within:ML:D2 | 24.623 | 24.482–24.678 |
| M32_N64_w8:vec8:within:MO:D1 | 82.000 | 81.500–83.000 |
| M32_N64_w8:vec8:within:MR:D1 | 162.787 | 162.637–163.030 |
| M32_N64_w8:vec8:within:MR:D2 | 165.333 | 164.166–165.358 |
| probe_E2:D1 | 8.138 | 8.137–8.149 |
| probe_E2:D2 | 8.042 | 8.037–8.049 |
| probe_E32:D1 | 110.181 | 110.169–110.342 |
| probe_E32:D2 | 108.913 | 108.897–108.926 |
| probe_E8:D1 | 29.502 | 29.495–29.515 |
| probe_E8:D2 | 29.398 | 29.385–29.407 |
| probe_P16:D1 | 47.849 | 47.847–47.881 |
| probe_P16:D2 | 46.069 | 46.064–46.077 |
| probe_P1:D1 | 3.502 | 3.490–3.512 |
| probe_P1:D2 | 3.096 | 3.083–3.096 |
| probe_P2:D1 | 7.714 | 7.702–7.720 |
| probe_P2:D2 | 7.140 | 7.139–7.145 |
| probe_P32:D1 | 95.098 | 95.077–95.120 |
| probe_P32:D2 | 93.769 | 93.763–93.784 |
| probe_P4:D1 | 11.791 | 11.788–11.792 |
| probe_P4:D2 | 12.230 | 12.220–12.232 |
| probe_P64:D1 | 203.638 | 203.628–203.663 |
| probe_P64:D2 | 204.047 | 204.031–204.060 |
| probe_P8:D1 | 24.563 | 24.549–24.567 |
| probe_P8:D2 | 24.619 | 24.615–24.628 |
| probe_U16:D1 | 162.564 | 162.522–162.786 |
| probe_U16:D2 | 170.724 | 170.710–170.725 |
| probe_U1:D1 | 3.502 | 3.490–3.512 |
| probe_U1:D2 | 3.096 | 3.083–3.096 |
| probe_U2:D1 | 10.532 | 10.514–10.541 |
| probe_U2:D2 | 10.192 | 10.192–10.196 |
| probe_U32:D1 | 415.753 | 415.740–415.765 |
| probe_U32:D2 | 364.587 | 364.564–364.595 |
| probe_U4:D1 | 25.088 | 25.084–25.090 |
| probe_U4:D2 | 24.789 | 24.787–24.797 |
| probe_U8:D1 | 59.399 | 59.360–59.400 |
| probe_U8:D2 | 68.525 | 68.522–68.536 |
| probe_W1:D1 | 6.981 | 6.980–6.983 |
| probe_W1:D2 | 6.184 | 6.183–6.189 |
| probe_W2:D1 | 47.818 | 47.814–47.847 |
| probe_W2:D2 | 47.794 | 47.778–47.834 |
| probe_W4:D1 | 109.249 | 109.244–109.311 |
| probe_W4:D2 | 108.804 | 108.790–108.868 |
| probe_W8:D1 | 248.698 | 248.675–248.810 |
| probe_W8:D2 | 248.094 | 248.008–248.105 |

## Counter and fixture boundaries

Dynamic software counts and hardware shared-load wavefront/bank-conflict measurements are exported by tag in counter_gate.json. K-invariant setup is separated from K-dependent body counts. Within-warp MR tests zero loop shared-load wavefronts, not zero total initial loads.
MO includes an input load. A compiler-elided conversion does not identify standalone output-conversion latency. MC has no private recurrence or checksum, but repeated output writes and two-buffer resources differ from the original kernel.
Therefore MC − (ML + MR + MO) is a measured composition/fixture residual, not pure overlap. The MC/HBM ratio is a service proxy, not a complete kernel prediction. C3 cannot silently turn this proxy or the residual into an unisolated empirical constant.
One-shot MO includes per-CTA setup and input-load work that repeated MC amortizes across K. Consequently its grid slope cannot be added as though it were an isolated repeated conversion cost.

No compiler change, model selector, C3, architecture transfer or PR was started. Raw evidence, technical failures and source snapshots are preserved.

## Parameter probes and q residency controls

Unit convention: the service quantity is s = (1/D) d(dT/dK)/d(G/S), where S is the SM count. It is nanoseconds for one tile serviced on one SM. MO uses dT/d(G/S) because it is one-shot. Grid belongs to calibration provenance, not the selector feature schema.
All 40 new logical probes passed 480 new complete-output checks; 807 C2a checks were reused. Two U1 binaries alias P1 exactly, leaving 38 new physical binaries. 373 q interventions preserve every target opcode count; 499 achieved-occupancy observations are within their CUDA capacity bounds. NCU launch IDs, function names, grid/block sizes and each measured source PC opcode were matched to the frozen SASS.
Core natural curves and TMA controls are reused from C2a. Their comparisons to C2b q curves are unpaired across cohorts. The new parameter probes have natural and q measurements in the same C2b workers. Cohort and GPU UUID are retained on every service/curve row.

| Probe | Natural D1 ns/tile/SM | Natural D2 ns/tile/SM (per stream) |
|---|---:|---:|
| P1 | 3.502 | 3.096 |
| P2 | 7.714 | 7.140 |
| P4 | 11.791 | 12.230 |
| P8 | 24.563 | 24.619 |
| P16 | 47.849 | 46.069 |
| P32 | 95.098 | 93.769 |
| P64 | 203.638 | 204.047 |
| U1 | 3.502 | 3.096 |
| U2 | 10.532 | 10.192 |
| U4 | 25.088 | 24.789 |
| U8 | 59.399 | 68.525 |
| U16 | 162.564 | 170.724 |
| U32 | 415.753 | 364.587 |
| W1 | 6.981 | 6.184 |
| W2 | 47.818 | 47.794 |
| W4 | 109.249 | 108.804 |
| W8 | 248.698 | 248.094 |
| E2 | 8.138 | 8.042 |
| E8 | 29.502 | 29.398 |
| E32 | 110.181 | 108.913 |

| D1 program | Natural ns | q1 ns | q2 ns | q4 ns |
|---|---:|---:|---:|---:|
| probe_P1:D1 | 3.502 | 11.390 | 6.154 | 3.948 |
| probe_P64:D1 | 203.638 | 213.766 | 204.189 | 203.644 |
| probe_U32:D1 | 415.753 | 2538.093 | 1280.070 | 675.014 |
| probe_W8:D1 | 248.698 | 316.598 | 263.205 | 249.750 |
| probe_E32:D1 | 110.181 | 125.672 | 112.323 | 110.091 |
| M32_N64_w8:vec8:n_first:MC:D1 | 377.211 | 443.272 | 379.918 | 377.027 |
| M32_N64_w8:vec8:within:MC:D1 | 227.864 | 260.796 | 229.977 | 228.090 |

## Aggregate-counter diagnostic

An extra, non-preregistered diagnostic assumed raw shared wavefront/bank-conflict counters would remain identical under q. This hypothesis failed in 262 of 373 contrasts. Every contrast preserves the arithmetic quantity wavefronts minus reported conflicts. This difference is not a separately collected request or ideal-wavefront metric.
The official [NVIDIA Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/#symptom-shared-memory-bank-conflicts-are-significant) explains that aggregate bank counters can include shared/L1 cross-client contention. Our observations are consistent with this scope; the experiment does not causally isolate all extra events. The largest relative raw-wavefront difference is about 8.57%, in a one-shot MO load case. No affected q point was deleted, normalized away, or rerun.
The initial failed extra check is archived in audits/aggregate_invariance_initial_gate.zip, with a counter-scope audit. The final gate follows the preregistered target-instruction invariance and measured occupancy requirements and retains all hardware-counter differences.

## Setup and composition identification

analysis.json additionally fits the steady-K intercept versus grid in per-CTA/per-SM units and fits K1 versus grid separately. These are measured fixture/setup combinations, including transfer and final-output work. MC K1 uses one transfer while steady K uses two. They are not universal TMA constants.
The 1341 setup fits include 1 negative estimate, retained without clamping. K1 grid fits have up to 7.27% residual. These bounds prevent interpreting fitted intercepts as independently identified positive hardware latencies.
For 32×64/vec8 D1, the median MC K1 service is about 463.25 ns/tile/SM for N-first versus 363.50 ns for within, whereas steady-body service is 377.21 versus 227.86 ns. The steady-body improvement is therefore not the whole single-pass improvement. K1 is measured over the registered cropped grids, with the recorded cache/resource differences; it is not a new full-grid original-kernel benchmark.
The two retained local-memory boundaries belong to the ML D2 checksum fixture; they do not establish spilling in the corresponding original complete kernel.
This stage identifies effective p/u/w/e and residency-dependent subgraph costs and composition residuals. It does not uniquely separate every naked load/max/convert latency. In particular, one-shot MO preparation and repeated-MC output traffic differ. No residual is silently substituted for 26.6 us, and no static selector was trained or evaluated.

## Review artifacts

[Parameter curves](figures/parameters.png), [all service curves](figures/services.csv), [full derived calibration](analysis.json), [counter and q evidence](counter_gate.json).

Final offline checks: `.venv/bin/python -m experiments.tma_reduction_layout.phase11.c2.validate --batch a --committed` and the same command with `--batch b --committed`.
