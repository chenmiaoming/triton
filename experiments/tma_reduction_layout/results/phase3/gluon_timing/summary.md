# Phase 3 Step E: Controlled Gluon Repeated-Reduction Timing Report

**Hypothesis H2 Status**: `SUPPORTED_AT_REDUCTION_BODY_LEVEL`
**Replication Mode**: `same-device temporal replication`

## 1. Executive Summary

Phase 3 Step E conducts controlled repeated-reduction timing across repetition counts
`R in {0, 1, 2, 4, 8}` using the single-binary fixed-descriptor protocol on NVIDIA H100 (SM90).
Initial TMA loading and shared-to-register LocalLoads are paid exactly once before the loop.
Residency is matched (8 blocks/SM for w8, 16 blocks/SM for w4) with 0 spills across all conditions.

- **Primary Mechanism Metric**: Isolated one-reduction differential $\Delta g(1) = g(1) - g(0)$ is **`1.0929 ± 0.0118` ns/additional CTA**.
- **Descriptive Attribution Ratio**: The isolated one-reduction differential has a magnitude equal to 74.88% of the canonical default-vs-cand4 marginal-slope gap. (Canonical gap = 1.459589 ns/additional CTA; isolated delta = 1.0929 ns/additional CTA).
  - *Attribution Note*: This is a **cross-harness descriptive magnitude comparison, not an additive causal decomposition**.
- **Amplification Trend Summary**: Fitted linear slope $\beta = `1.2040 ± 0.0022` ns/(CTA · rep)** serves strictly as an **amplification-trend summary** across $R \in \{0, 1, 2, 4, 8\}$ ($R^2 = 0.9996).
- **Linearity Quality**: Incremental repetition deltas demonstrate **strong approximately linear amplification over R=0..8** without constant per-repetition cost assumptions.

## 2. Hardware & Device Provenance

| Run ID | Device Name | GPU UUID | Replication Type | PState | SM Clock | Power Draw | Temperature |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :-: |
| `1` | `NVIDIA H100 80GB HBM3` | `GPU-e5b3d39b-9788-144e-eaa1-b6ba6a67b65b` | `same-device temporal replication` | `P0` | `345 MHz` | `67.91 W` | `29 °C` |
| `2` | `NVIDIA H100 80GB HBM3` | `GPU-e5b3d39b-9788-144e-eaa1-b6ba6a67b65b` | `same-device temporal replication` | `P0` | `1980 MHz` | `115.12 W` | `31 °C` |
| `3` | `NVIDIA H100 80GB HBM3` | `GPU-e5b3d39b-9788-144e-eaa1-b6ba6a67b65b` | `same-device temporal replication` | `P0` | `1980 MHz` | `116.03 W` | `31 °C` |

> [!NOTE] Replication Mode Binding
> All 3 benchmark invocations executed on identical GPU UUID (`GPU-e5b3d39b-9788-144e-eaa1-b6ba6a67b65b`), establishing
> **same-device temporal replication** across distinct container invocations on the same physical SM90 processor.

## 3. PRIMARY Specialization: `M32_N64_w8` (Exact Canonical Subsequence Equivalent)

> [!NOTE] Primary Experiment
> Both candidates match the complete filtered normalized reduction fingerprint sequence
> with the canonical reduction fingerprint inside the runtime loop.

### Repetition Sweep Breakdown ($R \in \{0, 1, 2, 4, 8\}$)

| R | $b_{\text{default}}(R)$ (ns/CTA) | $b_{\text{cand4}}(R)$ (ns/CTA) | $g(R) = b_{\text{def}} - b_4$ (ns/CTA) | $\Delta g(R) = g(R) - g(0)$ (ns/CTA) | Cross-Run CV |
| :-: | :---: | :---: | :---: | :---: | :---: |
| **0** | 1.3060 | 1.3099 | **-0.0039** | **+0.0000** | N/A |
| **1** | 2.5407 | 1.4517 | **+1.0890** | **+1.0929** | 0.91% |
| **2** | 4.7262 | 2.2324 | **+2.4938** | **+2.4977** | 0.26% |
| **4** | 9.0008 | 4.1982 | **+4.8025** | **+4.8064** | 0.25% |
| **8** | 17.6050 | 8.0016 | **+9.6034** | **+9.6073** | 0.19% |

### Incremental Repetition Deltas (ns / additional CTA / additional reduction rep)

| Interval | Formula | Delta Mean ± Std (ns/CTA/rep) | Cross-Run CV | Linearity Assessment |
| :--- | :--- | :---: | :---: | :--- |
| **R: 0 -> 1** | `d_01 = g(1) - g(0)` | **1.0929 ± 0.0118** | 1.08% | Isolated initial repetition |
| **R: 1 -> 2** | `d_12 = g(2) - g(1)` | **1.4048 ± 0.0043** | 0.31% | Second repetition step |
| **R: 2 -> 4** | `d_24 = (g(4) - g(2)) / 2` | **1.1544 ± 0.0069** | 0.60% | Two-repetition average step |
| **R: 4 -> 8** | `d_48 = (g(8) - g(4)) / 4` | **1.2002 ± 0.0070** | 0.58% | Four-repetition average step |

The observed increments vary; the fitted trend is approximately linear over R=0..8, not an exact constant cost per repetition.

### Primary Amplification Model Fits ($g(R) = \alpha + \beta \cdot R$)

- **Measured Baseline $g(0)$**: `-0.0039 ± 0.0065` ns/CTA (CV: `N/A`, stability: `near_zero_or_sign_unstable`)
- **Primary Metric $\Delta g(1)$**: `1.0929 ± 0.0118` ns/CTA
- **Amplification Slope $\beta$**: `1.2040 ± 0.0022` ns/(CTA · rep) (amplification-trend summary)
- **Fit Intercept $\alpha$**: `-0.0150 ± 0.0064` ns/CTA (CV: `N/A`, stability: `near_zero_or_sign_unstable`)
- **Mean per-run fit $R^2$**: `0.9996`
- **Canonical Positive Gap**: `1.459589` ns/additional CTA
- **Descriptive Attribution Ratio**: **`74.9%`** (`1.0929 / 1.459589`)

### Linear Model Fit Residuals

| R | Observed $g(R)$ (ns/CTA) | Fitted $g(R) = \alpha + \beta R$ (ns/CTA) | Residual (ns/CTA) |
| :-: | :---: | :---: | :---: |
| **0** | -0.0039 | -0.0150 | +0.0111 |
| **1** | +1.0890 | +1.1890 | -0.1000 |
| **2** | +2.4938 | +2.3930 | +0.1008 |
| **4** | +4.8025 | +4.8009 | +0.0016 |
| **8** | +9.6034 | +9.6169 | -0.0135 |

## 4. SECONDARY CONTROL Specialization: `M32_N128_w4` (Pipelined Opcode Equivalent)

> [!IMPORTANT] Secondary Control Interpretation
> `M32_N128_w4` is classified as `SECONDARY_CONTROL` and exhibits `PIPELINED_OPCODE_EQUIVALENT` lowering.
> Results show a **nonlinear / threshold-like emergence of reduction cost under repeated execution** (R=0..2 near zero, surging at R=4..8),
> consistent with an **overlap/roof regime interpretation**. We do not claim to have proven HBM saturation or hardware bandwidth limits.

| R | $b_{\text{default}}(R)$ (ns/CTA) | $b_{\text{cand4}}(R)$ (ns/CTA) | $g(R) = b_{\text{def}} - b_4$ (ns/CTA) | $\Delta g(R) = g(R) - g(0)$ (ns/CTA) | Cross-Run CV |
| :-: | :---: | :---: | :---: | :---: | :---: |
| **0** | 2.6407 | 2.6293 | **+0.0114** | **+0.0000** | 26.63% |
| **1** | 2.6189 | 2.6360 | **-0.0171** | **-0.0285** | N/A |
| **2** | 2.6170 | 2.6354 | **-0.0184** | **-0.0298** | N/A |
| **4** | 4.0923 | 2.6084 | **+1.4840** | **+1.4725** | 0.72% |
| **8** | 7.5981 | 3.5937 | **+4.0045** | **+3.9930** | 0.17% |

### Secondary Amplification Model Fits ($g(R) = \alpha + \beta \cdot R$)

- **Measured Baseline $g(0)$**: `0.0114 ± 0.0030` ns/CTA
- **Isolated $\Delta g(1)$**: `-0.0285 ± 0.0168` ns/CTA
- **Linear Slope $\beta$**: `0.5381 ± 0.0016` ns/(CTA · rep)
- **Mean per-run fit $R^2$**: `0.9442`

## 5. Compiler Barrier Attribution & SASS Verification

- The input compiler barrier induces equal PTX copy counts within each shape. No explicit MOV or IMAD.MOV was observed in the runtime SASS reduction region.
- Indirect allocation, live-range, and scheduler effects remain possible; equal copy counts do not establish zero differential.

## 6. Hypothesis H2 Evaluation & Lineage Decomposition

- **Overall H2 Status**: **`SUPPORTED_AT_REDUCTION_BODY_LEVEL`**
- **Evaluation Verdict**: Hypothesis H2 is decomposed into sub-hypotheses: H2a is SUPPORTED AT REDUCTION BODY LEVEL. The isolated one-reduction differential has a magnitude equal to 74.9% of the canonical default-vs-cand4 marginal-slope gap (cross-harness descriptive magnitude comparison, not an additive causal decomposition). The primary mechanism metric is Δg(1) = 1.0929 ± 0.0118 ns/additional CTA. Linear slope beta = 1.2040 ± 0.0022 ns/(CTA·rep) serves strictly as an amplification-trend summary. Incremental repetition deltas exhibit strong approximately linear amplification over R=0..8. Sub-hypotheses H2b (lane-partition dominance) and H2c (intra-warp communication dominance) remain UNVERIFIED.

### Formal Sub-Hypothesis Lineage

1. **Hypothesis 2a (H2a)**: Composite reduction-body structure materially contributes to positive default-vs-cand4 throughput separation.
   - **Status**: **`SUPPORTED_AT_REDUCTION_BODY_LEVEL`**
   - **Evidence**: Isolated one-reduction differential $\Delta g(1) = 1.0929 ± 0.0118$ ns/additional CTA.
   - **Attribution Scope**: The isolated one-reduction differential has a magnitude equal to 74.88% of the canonical default-vs-cand4 marginal-slope gap (cross-harness descriptive magnitude comparison, not an additive causal decomposition).

2. **Hypothesis 2b (H2b)**: Lane-partition pruning dominates warp-partition pruning.
   - **Status**: **`UNVERIFIED`**
   - **Rationale**: No matched orthogonal lane-only vs warp-only intervention has been isolated without confounding.

3. **Hypothesis 2c (H2c)**: Intra-warp communication dominates cross-warp communication.
   - **Status**: **`UNVERIFIED`**
   - **Rationale**: Step E isolates the composite reduction body, but does not isolate thread-local vs shuffle vs smem vs barrier components separately.


The decision rule is retrospective: all primary run betas > 0, mean Δg(1) > 0.1 ns/CTA, mean per-run R² >= 0.90, and mean/each-run g(R) nondecreasing. It is not an integrity PASS condition.
± denotes sample SD of three same-device temporal invocations, not a confidence interval.
R² of cross-run mean points: 0.999647141; mean of per-run R²: 0.999642049.
Canonical raw binding: `experiments/tma_reduction_layout/results/phase2/saturation/corrected_pilot_runs.json`, SHA256 `75087a0ea79d359e4b1820be729295db3f0d7cf392cdfde2a888b63dbb453e9b`, last-change commit `a57bff355124dd3d80e6ff4116e4796d1eeb48de`.
Fingerprint equality ignores most operands and predicates; it does not prove dataflow, full PTX, SASS, or CUBIN identity. See ../freeze.md for archival limits.