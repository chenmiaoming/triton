# Phase 3 Step B: Reduction-Communication Amplification Microbenchmark Summary

> [!WARNING]
> Step B v1 is retained as an exploratory recurrent-composite amplification experiment.
> It does NOT isolate repeated copies of the canonical reduction body, because codegen and register allocation change with K.

## 1. Experimental Overview & Environment

- **Hardware Platform**: NVIDIA H100 80GB HBM3 (SM90, Compute Capability 9.0)
- **Device UUID**: `GPU-7c0d9bcd-6359-8d5f-9fc3-e596ee0b424d`
- **Driver / CUDA**: Driver 580.95.05 / CUDA 13.0
- **Software Environment**: Python 3.12.1, Triton 3.9.0
- **Replication Mode**: Same-device temporal replication (3 sequential remote invocations: `run_1`, `run_2`, `run_3`)
- **Evaluated Iteration Counts**: $K \in [1, 2, 4, 8]$

## 2. Codegen Invariance & Structural Verification

| Configuration | Candidate | K | Physical REG | LOCAL Spill | STACK Spill | TMA Count | Initial LocalLoad | Shuffles (shfl.bfly) | Barriers (bar.sync) | max.f32 | max.bf16x2 | LocalLoad Invariant |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `M32_N64_w8` | `4` | 1 | 23 | 0 B | 0 B | 1 | `ld.shared.v2.b32` | 16 | 10 | 16 | 2 | Yes |
| `M32_N64_w8` | `4` | 2 | 32 | 0 B | 0 B | 1 | `ld.shared.v4.b16` | 32 | 14 | 36 | 2 | No (signature mismatch) |
| `M32_N64_w8` | `4` | 4 | 32 | 0 B | 0 B | 1 | `ld.shared.v4.b16` | 64 | 22 | 76 | 2 | No (signature mismatch) |
| `M32_N64_w8` | `4` | 8 | 40 | 0 B | 0 B | 1 | `ld.shared.v4.b16` | 128 | 38 | 156 | 2 | No (signature mismatch) |
| `M32_N64_w8` | `default` | 1 | 29 | 0 B | 0 B | 1 | `ld.shared.v4.b32` | 40 | 14 | 40 | 0 | Yes |
| `M32_N64_w8` | `default` | 2 | 32 | 0 B | 0 B | 1 | `ld.shared.v4.b32` | 80 | 22 | 80 | 0 | Yes |
| `M32_N64_w8` | `default` | 4 | 40 | 0 B | 0 B | 1 | `ld.shared.v4.b32` | 160 | 38 | 160 | 0 | Yes |
| `M32_N64_w8` | `default` | 8 | 48 | 0 B | 0 B | 1 | `ld.shared.v4.b32` | 320 | 70 | 320 | 0 | Yes |
| `M32_N128_w4` | `4` | 1 | 25 | 0 B | 0 B | 1 | `ld.shared.v2.b32` | 8 | 10 | 8 | 14 | Yes |
| `M32_N128_w4` | `4` | 2 | 55 | 0 B | 0 B | 1 | `ld.shared.v4.b16` | 16 | 14 | 44 | 14 | No (signature mismatch) |
| `M32_N128_w4` | `4` | 4 | 71 | 0 B | 0 B | 1 | `ld.shared.v4.b16` | 32 | 22 | 116 | 14 | No (signature mismatch) |
| `M32_N128_w4` | `4` | 8 | 86 | 0 B | 0 B | 1 | `ld.shared.v4.b16` | 64 | 38 | 260 | 14 | No (signature mismatch) |
| `M32_N128_w4` | `default` | 1 | 31 | 0 B | 0 B | 1 | `ld.shared.v4.b32` | 24 | 14 | 24 | 12 | Yes |
| `M32_N128_w4` | `default` | 2 | 56 | 0 B | 0 B | 1 | `ld.shared.v4.b32` | 48 | 22 | 72 | 12 | No (signature mismatch) |
| `M32_N128_w4` | `default` | 4 | 72 | 0 B | 0 B | 1 | `ld.shared.v4.b32` | 96 | 38 | 168 | 12 | No (signature mismatch) |
| `M32_N128_w4` | `default` | 8 | 116 | 0 B | 0 B | 1 | `ld.shared.v4.b32` | 192 | 70 | 360 | 12 | No (signature mismatch) |

> [!CAUTION]
> **Structural Confounds in Step B v1**:
> 1. **Initial LocalLoad Signature Mismatch**: In `M32_N64_w8 cand4`, the initial load lowered to `2x ld.shared.v2.b32` at $K=1$, but switched to `2x ld.shared.v4.b16` for $K \in \{2, 4, 8\}$. In `M32_N128_w4 default`, initial loads dropped from 4 to 1 vector load.
> 2. **Non-Identical Incremental Reduction Body**: In `M32_N64_w8 cand4`, `max.bf16x2` count was 2 across all $K \in \{1, 2, 4, 8\}$; incremental iterations ($K > 1$) did not repeat packed BF16 reductions, executing exclusively `max.f32` (+20 per iteration).
> 3. **Material Register Pressure Growth**: Registers increased from 29 to 48 in `M32_N64_w8 default`, and from 31 to 116 in `M32_N128_w4 default`, introducing potential scheduling and residency confounds.
> Across all tested conditions, `is_valid_for_isolation = false`.

## 3. Marginal Slope and Gap Scaling vs K

### M32_N64_w8 (positive_case)

| K | default Slope (ns/CTA) | cand4 Slope (ns/CTA) | Gap $b_{\text{def}} - b_{\text{cand4}}$ (ns/CTA) | Advantage (%) | 3-Run Gap CV (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 3.949 | 2.531 | +1.418 | -35.90% | 1.28% |
| 2 | 5.866 | 3.389 | +2.477 | -42.22% | 1.81% |
| 4 | 10.567 | 5.291 | +5.276 | -49.93% | 1.15% |
| 8 | 20.023 | 9.923 | +10.099 | -50.44% | 0.12% |

- **Linear Gap Fit**: $\text{gap}(K) = 0.121 + (+1.252) \times K$ ($R^2 = 0.9990$)
  - Note on intercept (0.121): fitted K-axis intercept; no physical attribution.
- **Default Amplification Slope ($\Delta b / \Delta K$)**: `+2.317` ns/additional CTA/body
- **Cand4 Amplification Slope ($\Delta b / \Delta K$)**: `+1.064` ns/additional CTA/body
- **Empirical Differential Amplification ($\Delta \text{gap} / \Delta K$)**: `+1.252` ns/additional CTA/body
- **Classification**: `CONFOUNDED_BY_CODEGEN_AND_REGISTER_PRESSURE`

### M32_N128_w4 (negative_control)

| K | default Slope (ns/CTA) | cand4 Slope (ns/CTA) | Gap $b_{\text{def}} - b_{\text{cand4}}$ (ns/CTA) | Advantage (%) | 3-Run Gap CV (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 3.016 | 3.033 | -0.017 | +0.58% | 169.03% |
| 2 | 3.581 | 3.255 | +0.326 | -9.10% | 4.52% |
| 4 | 5.452 | 4.448 | +1.004 | -18.41% | 1.09% |
| 8 | 11.321 | 8.715 | +2.605 | -23.02% | 0.91% |

- **Linear Gap Fit**: $\text{gap}(K) = -0.429 + (+0.376) \times K$ ($R^2 = 0.9983$)
  - Note on intercept (-0.429): fitted K-axis intercept; no physical attribution.
- **Default Amplification Slope ($\Delta b / \Delta K$)**: `+1.214` ns/additional CTA/body
- **Cand4 Amplification Slope ($\Delta b / \Delta K$)**: `+0.839` ns/additional CTA/body
- **Empirical Differential Amplification ($\Delta \text{gap} / \Delta K$)**: `+0.376` ns/additional CTA/body
- **Pruned Butterfly Shuffle Scaling**: In `M32_N128_w4`, default issues $24 \times K$ butterfly shuffles and cand4 issues $8 \times K$ (difference $= 16 \times K$). In `M32_N64_w8`, default issues $40 \times K$ and cand4 issues $16 \times K$ (difference $= 24 \times K$). The ratio of pruned butterfly shuffles is $24 / 16 = 1.5\times$ (NOT $3.3\times$). The $3.33\times$ ratio ($1.252 / 0.376$) is the empirical gap amplification slope ratio.
- **Classification**: `CONFOUNDED_BY_CODEGEN_AND_REGISTER_PRESSURE`

## 4. Hypothesis Evaluation & Interpretation

### Hypothesis 2 (H2: Reduction-Communication Cost):
- **Status**: `UNVERIFIED`
- **Reclassification**: Step B v1 is reclassified as `CONFOUNDED_BY_CODEGEN_AND_REGISTER_PRESSURE`.
- **Observation**: The default-vs-cand4 gap amplifies in this recurrent composite workload, but the source of amplification is not isolated.

### Negative Control Contrast (`M32_N128_w4`):
- **Status**: `CONFOUNDED_BY_CODEGEN_AND_REGISTER_PRESSURE`
- **Observation**: Although the throughput gap widens from -0.017 ns/CTA at $K=1$ to +2.605 ns/CTA at $K=8$, this workload is confounded by register pressure growth (31 -> 116 regs) and non-invariant load lowerings.

### Non-Claims & Methodological Boundaries:
- **No causal proof of individual instruction latency**: We report only empirical incremental slope per additional compiler-generated reduction body ($\Delta b / \Delta K$), not single-instruction latencies.
- **H3 (LocalLoad cost) & H4 (Epilogue layout conversion)**: Remain `UNVERIFIED / PENDING_DIFFERENTIAL_MICROBENCH` as they were not isolated in this microbenchmark.
- **No production heuristics**: No compiler heuristics or threshold rules are proposed.
