# Phase 3 Step E: Controlled Gluon Repeated-Reduction Timing Report

**Hypothesis H2 Status**: `SUPPORTED_AT_REDUCTION_BODY_LEVEL`
**Replication Mode**: `same-device temporal replication`

## 1. Executive Summary

Phase 3 Step E conducts controlled repeated-reduction timing across repetition counts
`R in {0, 1, 2, 4, 8}` using the single-binary fixed-descriptor protocol on NVIDIA H100 (SM90).
Initial TMA loading and shared-to-register LocalLoads are paid exactly once before the loop.
Residency is matched (8 blocks/SM for w8, 16 blocks/SM for w4) with 0 spills across all conditions.

The isolated one-reduction differential $\Delta g(1) = g(1) - g(0)$ is **`1.0929` ns/additional CTA**.
Descriptive attribution ratio against canonical positive gap (1.4279 ns/CTA): **`76.5%`**.

## 2. Hardware & Device Provenance

| Run ID | Device Name | GPU UUID | Replication Type | PState | SM Clock | Power Draw | Temperature |
| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :-: |
| `1` | `NVIDIA H100 80GB HBM3` | `GPU-e5b3d39b-9788-144e-eaa1-b6ba6a67b65b` | `same-device temporal replication` | `P0` | `345 MHz` | `67.91 W` | `29 °C` |
| `2` | `NVIDIA H100 80GB HBM3` | `GPU-e5b3d39b-9788-144e-eaa1-b6ba6a67b65b` | `same-device temporal replication` | `P0` | `1980 MHz` | `115.12 W` | `31 °C` |
| `3` | `NVIDIA H100 80GB HBM3` | `GPU-e5b3d39b-9788-144e-eaa1-b6ba6a67b65b` | `same-device temporal replication` | `P0` | `1980 MHz` | `116.03 W` | `31 °C` |

## 3. PRIMARY Specialization: `M32_N64_w8` (Exact Canonical Subsequence Equivalent)

> [!NOTE] Primary Experiment
> Both default (vec=8) and cand4 (vec=4) exhibit 100% exact contiguous subsequence equivalence
> with the canonical reduction fingerprint inside the runtime loop.

| R | $b_{\text{default}}(R)$ (ns/CTA) | $b_{\text{cand4}}(R)$ (ns/CTA) | $g(R) = b_{\text{def}} - b_4$ (ns/CTA) | $\Delta g(R) = g(R) - g(0)$ (ns/CTA) | Cross-Run CV |
| :-: | :---: | :---: | :---: | :---: | :---: |
| **0** | 1.3060 | 1.3099 | **-0.0039** | **+0.0000** | -165.47% |
| **1** | 2.5407 | 1.4517 | **+1.0890** | **+1.0929** | 0.91% |
| **2** | 4.7262 | 2.2324 | **+2.4938** | **+2.4977** | 0.26% |
| **4** | 9.0008 | 4.1982 | **+4.8025** | **+4.8064** | 0.25% |
| **8** | 17.6050 | 8.0016 | **+9.6034** | **+9.6073** | 0.19% |

### Primary Amplification Model Fits ($g(R) = \alpha + \beta \cdot R$)

- **Measured Baseline $g(0)$**: `-0.0039 ± 0.0065` ns/CTA
- **Isolated $\Delta g(1)$**: `1.0929 ± 0.0118` ns/CTA
- **Linear Slope $\beta$**: `1.2040 ± 0.0022` ns/(CTA · rep)
- **Fit Intercept $\alpha$**: `-0.0150` ns/CTA
- **Model Fit $R^2$**: `0.9996`
- **Canonical Positive Gap**: `1.4279` ns/additional CTA
- **Descriptive Attribution Ratio**: **`76.5%`** (`1.0929 / 1.4279`)

## 4. SECONDARY CONTROL Specialization: `M32_N128_w4` (Pipelined Opcode Equivalent)

> [!IMPORTANT] Secondary Control Interpretation
> `M32_N128_w4 default` exhibits pipelined opcode equivalence (independent column slices scheduled
> by LLVM to reduce live registers from 32 to 20), rather than exact contiguous sequence equivalence.
> Results are presented as secondary control under the Gluon loop lowering.

| R | $b_{\text{default}}(R)$ (ns/CTA) | $b_{\text{cand4}}(R)$ (ns/CTA) | $g(R) = b_{\text{def}} - b_4$ (ns/CTA) | $\Delta g(R) = g(R) - g(0)$ (ns/CTA) | Cross-Run CV |
| :-: | :---: | :---: | :---: | :---: | :---: |
| **0** | 2.6407 | 2.6293 | **+0.0114** | **+0.0000** | 26.63% |
| **1** | 2.6189 | 2.6360 | **-0.0171** | **-0.0285** | -111.08% |
| **2** | 2.6170 | 2.6354 | **-0.0184** | **-0.0298** | -121.00% |
| **4** | 4.0923 | 2.6084 | **+1.4840** | **+1.4725** | 0.72% |
| **8** | 7.5981 | 3.5937 | **+4.0045** | **+3.9930** | 0.17% |

### Secondary Amplification Model Fits ($g(R) = \alpha + \beta \cdot R$)

- **Measured Baseline $g(0)$**: `0.0114 ± 0.0030` ns/CTA
- **Isolated $\Delta g(1)$**: `-0.0285 ± 0.0168` ns/CTA
- **Linear Slope $\beta$**: `0.5381 ± 0.0016` ns/(CTA · rep)
- **Model Fit $R^2$**: `0.9442`

## 5. Hypothesis H2 Evaluation

- **Updated H2 Status**: **`SUPPORTED_AT_REDUCTION_BODY_LEVEL`**
- **Evaluation Verdict**: Hypothesis H2 is SUPPORTED AT REDUCTION BODY LEVEL. Repeated canonical reduction amplification exhibits a strictly positive marginal slope (beta = 1.2040 ns/(CTA*rep), R^2 = 0.9996) with an isolated one-reduction gap Δg(1) = 1.0929 ns/additional CTA (76.5% of canonical gap). The layout-induced reduction-body structure is a material source of the positive-case throughput separation.
- **Attribution Scope**: The layout-induced reduction-body structure is a material source of the positive-case throughput separation.
- **Guardrail Constraint**: Does NOT attribute separation to individual shuffle vs barrier instructions in isolation; attribution applies to the composite reduction core.
