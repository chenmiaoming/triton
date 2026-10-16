# Stage C2a — H100 calibration

90 core programs plus 0 parameter programs; 74 new logical compilations in this batch.
New graph event samples: 73,800; 1230 physical cells per worker; three worker allocations.
New NCU reports: 16; C1/C1.5 original 128 reports and CUBINs remain unchanged.
Resource boundaries excluded from regular timing/fit: M32_N128_w4:vec2:n_first:ML:D2, M32_N128_w4:vec2:within:ML:D2. Correctness and artifacts are retained for these programs.

## Measurement and units

Repeated subgraphs use K16/64/256 OLS, with K1 retained separately. Three grids fit steady service per SM per tile; grid is calibration provenance, absent from static predictor features. MO is one-shot and uses T versus grid. All samples are single-kernel CUDA Graph event intervals. NCU duration is not used.
Independent unchanged TMA controls run at full grids. Their time is normalized in the same tile/SM units. No 26.6 us offset appears in the analysis.
The preferred shared carveout is set and read back as 100% for controlled q comparisons; these fresh curves have their own measurement conditions. Old timing samples are not pooled into them.
Maximum steady-K fit residual: 3.48% of the largest measured point. Maximum grid-fit residual: 1.04%. Adjacent slopes and all worker estimates remain in analysis.json.

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

## Counter and fixture boundaries

Dynamic software counts and hardware shared-load wavefront/bank-conflict measurements are exported by tag in counter_gate.json. K-invariant setup is separated from K-dependent body counts. Within-warp MR tests zero loop shared-load wavefronts, not zero total initial loads.
MO includes an input load. A compiler-elided conversion does not identify standalone output-conversion latency. MC has no private recurrence or checksum, but repeated output writes and two-buffer resources differ from the original kernel.
Therefore MC − (ML + MR + MO) is a measured composition/fixture residual, not pure overlap. The MC/HBM ratio is a service proxy, not a complete kernel prediction. C3 cannot silently turn this proxy or the residual into an unisolated empirical constant.
One-shot MO includes per-CTA setup and input-load work that repeated MC amortizes across K. Consequently its grid slope cannot be added as though it were an isolated repeated conversion cost.

No compiler change, model selector, C3, architecture transfer or PR was started. Raw evidence, technical failures and source snapshots are preserved.

## Placement intervention: measured D1 counters

| Geometry | vec | Placement | ML load WF/K | ML load bank conflicts/K | MR load WF/K | MR BAR.SYNC warp inst/K |
|---|---:|---|---:|---:|---:|---:|
| 32×64/w8 | 1 | n_first | 64 | 0 | 16 | 32 |
| 32×64/w8 | 1 | within | 64 | 0 | 0 | 0 |
| 32×64/w8 | 2 | n_first | 32 | 0 | 32 | 32 |
| 32×64/w8 | 2 | within | 32 | 0 | 0 | 0 |
| 32×64/w8 | 4 | n_first | 32 | 0 | 64 | 32 |
| 32×64/w8 | 4 | within | 32 | 0 | 0 | 0 |
| 32×64/w8 | 8 | n_first | 32 | 0 | 128 | 64 |
| 32×64/w8 | 8 | within | 32 | 0 | 0 | 0 |
| 32×128/w4 | 1 | n_first | 128 | 0 | 0 | 0 |
| 32×128/w4 | 2 | n_first | 64 | 0 | 16 | 16 |
| 32×128/w4 | 2 | within | 128 | 64 | 0 | 0 |
| 32×128/w4 | 4 | n_first | 64 | 0 | 32 | 16 |
| 32×128/w4 | 4 | within | 128 | 64 | 0 | 0 |
| 32×128/w4 | 8 | n_first | 64 | 0 | 64 | 32 |
| 32×128/w4 | 8 | within | 128 | 64 | 0 | 0 |

## Preserved technical failures

The first CPU preflight rejected a nonconstant mbarrier byte argument. A uniform K branch with constant byte counts repaired the API use.
Initial MR D1 for 32×64/vec8/within failed 12 checks, affecting columns 1 modulo 8. Its PTX packed step construction read an undefined half-register. The mathematically identical step was factored as (13*m)%7 before lowering. Only this one logical program was recompiled; the other 89 core binaries stayed fixed. The repaired binary passed all 12 checks; all 807 total checks pass. Failed binaries, exact mismatch values, source snapshots and the unsupported broadcast helper attempt remain archived.

## Review artifacts

[Core service figure](figures/core_services.png), [tidy service table](figures/services.csv), [full derived data](analysis.json), [dynamic counters](counter_gate.json).
