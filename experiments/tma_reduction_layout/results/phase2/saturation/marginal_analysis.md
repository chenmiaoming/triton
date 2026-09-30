# Offline Marginal Analysis of Phase 2 Extended Saturation Data

## Methodology & Affine Model Rationale

> [!NOTE]
> When kernel duration satisfies $T(B) = a + b \cdot B$, the amortized grid time $T(B)/B = a/B + b$ continues to decrease towards $b$ as $a/B \to 0$.
> Consequently, evaluating consecutive doublings of $T(B)/B$ conflates fixed device-side/grid costs with incremental throughput.
> 
> The true steady-state throughput of a candidate layout is governed by the incremental marginal cost per additional CTA:
> $$\text{Marginal Slope } b = \frac{\Delta T}{\Delta B} \quad (\text{ns/CTA})$$
> or the linear slope of the affine regression model $T(B) = \text{intercept\_us} + \text{slope\_us} \cdot B$ over large $B$.

- **Fitted Fixed-Time Intercept**: Labeled as `intercept_us`. Its physical origin is **UNKNOWN** (consistent with fixed device-side costs / initialization; source not isolated). It is not claimed to be CUDA kernel launch overhead.
- **Operational Criterion for Marginal Linear Regime**:
  - Change between the final two adjacent slope intervals $\left| \frac{(\Delta T/\Delta B)_{\text{last}} - (\Delta T/\Delta B)_{\text{prev}}}{(\Delta T/\Delta B)_{\text{prev}}} \right| < 5.0\%$
  - Affine fit $R^2 \ge 0.99$
  - When both are met: `marginal_linear_regime_observed = true`.

## Execution Environment
- **GPU**: `NVIDIA H100 80GB HBM3` (CC: `[9, 0]`, Driver: `580.95.05`)
- **SM Count**: `132`
- **PyTorch / CUDA**: `2.14.0+cu130` / CUDA `13.0`
- **Evaluated Grid Sizes $B$**: `[4096, 8192, 16384, 32768, 65536]`

## 1. Interval Incremental Marginal Slopes (ΔT / ΔB)

### Configuration: `M32_N16_w8` (`M=32, N=16, num_warps=8`)

| Candidate | 4096 -> 8192 | 8192 -> 16384 | 16384 -> 32768 | 32768 -> 65536 | Last 2 Intervals Delta (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `default` | 0.2539 ns | 1.4453 ns | 2.3389 ns | 2.2700 ns | **-2.94%** |
| 8 | - | - | - | - | INVALID: forcedVec 8 > numElemsPerThread (2) |
| 4 | - | - | - | - | INVALID: forcedVec 4 > numElemsPerThread (2) |
| `2` | 1.6172 ns | 1.0273 ns | 2.3232 ns | 2.2798 ns | **-1.87%** |
| `1` | 0.2500 ns | 1.5762 ns | 2.2363 ns | 2.2422 ns | **+0.26%** |

### Configuration: `M32_N64_w8` (`M=32, N=64, num_warps=8`)

| Candidate | 4096 -> 8192 | 8192 -> 16384 | 16384 -> 32768 | 32768 -> 65536 | Last 2 Intervals Delta (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `default` | 3.6367 ns | 4.2715 ns | 3.9990 ns | 3.8604 ns | **-3.47%** |
| `8` | 3.6797 ns | 4.2930 ns | 3.9600 ns | 3.9023 ns | **-1.45%** |
| `4` | 2.4297 ns | 2.7832 ns | 2.4961 ns | 2.4517 ns | **-1.78%** |
| `2` | 2.0195 ns | 2.6602 ns | 2.2959 ns | 2.2368 ns | **-2.57%** |
| `1` | 1.9570 ns | 2.6641 ns | 2.2383 ns | 2.2271 ns | **-0.50%** |

### Configuration: `M32_N128_w4` (`M=32, N=128, num_warps=4`)

| Candidate | 4096 -> 8192 | 8192 -> 16384 | 16384 -> 32768 | 32768 -> 65536 | Last 2 Intervals Delta (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `default` | 3.7930 ns | 2.9258 ns | 2.9326 ns | 2.9907 ns | **+1.98%** |
| `8` | 3.7930 ns | 2.9727 ns | 2.8945 ns | 2.9805 ns | **+2.97%** |
| `4` | 3.8594 ns | 2.9434 ns | 2.9248 ns | 2.9775 ns | **+1.80%** |
| `2` | 3.9258 ns | 2.9180 ns | 2.9121 ns | 2.9604 ns | **+1.66%** |
| `1` | 3.9258 ns | 2.9609 ns | 2.8965 ns | 2.9893 ns | **+3.20%** |

## 2. Affine Regression Model & Marginal Cost Comparison

Affine fit performed over the large-$B$ points: `B in [16384, 32768, 65536]`.

| Configuration | Candidate | Legal | Fitted Intercept (µs) | Marginal Slope (ns/CTA) | vs Default Slope (%) | Model R² | Residuals (µs) [16k, 32k, 64k] | Marginal Linear Regime? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `M32_N16_w8` | `default` | YES | 8.9040 | **2.2897** | **0.00% (base)** | 0.999945 | `[-0.322, +0.483, -0.161]` | **YES** |
| `M32_N16_w8` | `8` | NO | - | - | - | - | - | INVALID |
| `M32_N16_w8` | `4` | NO | - | - | - | - | - | INVALID |
| `M32_N16_w8` | `2` | YES | 8.7120 | **2.2922** | **+0.11%** | 0.999978 | `[-0.203, +0.305, -0.102]` | **YES** |
| `M32_N16_w8` | `1` | YES | 8.3040 | **2.2405** | **-2.15%** | 1.000000 | `[+0.027, -0.041, +0.014]` | **YES** |
| `M32_N64_w8` | `default` | YES | 18.6560 | **3.9000** | **0.00% (base)** | 0.999923 | `[-0.649, +0.974, -0.325]` | **YES** |
| `M32_N64_w8` | `8` | YES | 18.4960 | **3.9188** | **+0.48%** | 0.999987 | `[-0.270, +0.405, -0.135]` | **YES** |
| `M32_N64_w8` | `4` | YES | 18.4240 | **2.4644** | **-36.81%** | 0.999980 | `[-0.208, +0.312, -0.104]` | **YES** |
| `M32_N64_w8` | `2` | YES | 18.4560 | **2.2537** | **-42.21%** | 0.999958 | `[-0.277, +0.415, -0.138]` | **YES** |
| `M32_N64_w8` | `1` | YES | 18.2800 | **2.2303** | **-42.81%** | 0.999998 | `[-0.053, +0.079, -0.026]` | **YES** |
| `M32_N128_w4` | `default` | YES | 17.6240 | **2.9741** | **0.00% (base)** | 0.999977 | `[+0.272, -0.408, +0.136]` | **YES** |
| `M32_N128_w4` | `8` | YES | 18.1280 | **2.9559** | **-0.61%** | 0.999948 | `[+0.402, -0.603, +0.201]` | **YES** |
| `M32_N128_w4` | `4` | YES | 17.3120 | **2.9625** | **-0.39%** | 0.999981 | `[+0.247, -0.370, +0.123]` | **YES** |
| `M32_N128_w4` | `2` | YES | 17.5280 | **2.9466** | **-0.92%** | 0.999984 | `[+0.226, -0.339, +0.113]` | **YES** |
| `M32_N128_w4` | `1` | YES | 17.2480 | **2.9628** | **-0.38%** | 0.999940 | `[+0.434, -0.651, +0.217]` | **YES** |

## 3. Findings on Marginal Slope Separation

### A. `M32_N64_w8`: Substantial Marginal Slope Separation
The empirical marginal cost per additional CTA exhibits significant separation across layout candidates:
- `default`: **3.9000 ns/CTA** (baseline)
- `cand 8`:  **3.9188 ns/CTA** (`+0.48%` vs default, parity)
- `cand 4`:  **2.4644 ns/CTA** (`-36.81%` vs default)
- `cand 2`:  **2.2537 ns/CTA** (`-42.21%` vs default)
- `cand 1`:  **2.2303 ns/CTA** (`-42.81%` vs default)

**Key Observations**:
1. `default marginal cost >> cand 4 > cand 2 ≈ cand 1` is strongly confirmed by the data.
2. Candidates 1 and 2 achieve a **>42% reduction in marginal steady-state runtime per CTA** compared to default.
3. Candidate 8 remains in exact parity with default across all intervals and in marginal slope (`+0.48%`).
4. The fitted fixed-time intercept is nearly identical across all candidates (~18.3 to 18.7 µs). The performance divergence between default and narrow candidates is almost exclusively driven by the marginal slope term ($b$), proving that layout efficiency differences compound linearly with grid size rather than dissipating.

### B. `M32_N16_w8`: Marginal Slopes in Parity
- `default`: **2.2897 ns/CTA** (baseline)
- `cand 2`:  **2.2922 ns/CTA** (`+0.11%` vs default)
- `cand 1`:  **2.2405 ns/CTA** (`-2.15%` vs default)

**Observation**: Marginal costs across all legal candidates are within ~2% of default. There is no large layout separation for `M32_N16_w8` in steady state.

### C. `M32_N128_w4`: Marginal Slopes in Complete Parity
- `default`: **2.9741 ns/CTA** (baseline)
- `cand 8`:  **2.9559 ns/CTA** (`-0.61%` vs default)
- `cand 4`:  **2.9625 ns/CTA** (`-0.39%` vs default)
- `cand 2`:  **2.9466 ns/CTA** (`-0.92%` vs default)
- `cand 1`:  **2.9628 ns/CTA** (`-0.38%` vs default)

**Observation**: All candidates have identical marginal costs within $\pm 0.9\%$. Layout choice does not alter marginal steady-state throughput in this configuration.

### D. Marginal Linear Regime Verification
- In all three configurations, the change between the last two slope intervals ($16384 \to 32768$ vs $32768 \to 65536$) is strictly $< 3.5\%$, well below the $< 5.0\%$ threshold.
- In all cases, the affine model goodness-of-fit $R^2 \ge 0.9999$.
- Therefore, **all three configurations have entered the marginal linear steady-state regime** for $B \ge 16384$.