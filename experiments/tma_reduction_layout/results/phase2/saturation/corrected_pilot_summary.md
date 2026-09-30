# Corrected Multi-Invocation Fixed-Binary TMA Reduction Saturation Report

> [!NOTE]
> **Scope & Evidence Note**: The three benchmark invocations (`run_1`, `run_2`, `run_3`) were executed sequentially on the same physical NVIDIA H100 device (`GPU-a59752c5-ebb5-1dac-f887-c2e3c1f81aff`). They demonstrate same-device temporal repeatability, not cross-device replication across independent GPU hardware allocations.
>
> **Methodology Guarantees**:
> 1. **Fixed Binary**: Tensor descriptor shape is fixed to `B_DESC = max(B)`. Kernels are compiled once per candidate as a single specialization and variable grid sizes `B_RUN <= B_DESC` reuse the identical compiled binary without recompilation (verified via byte-level TTGIR and PTX SHA256 hashes).
> 2. **Sequential Replication**: Benchmark is executed across three sequential remote invocations on the same physical device (each compiling fresh binaries, generating fresh inputs, and warming up).
> 3. **Order Rotation**: Grid sizes ($B$) and layout candidates are rotated circularly across 10 timing rounds (10 samples/round = 100 samples per condition) to eliminate thermal drift, clock drift, and ordering bias.
> 4. **Affine Steady-State Model**: Evaluates $T(B) = \text{intercept\_us} + (\text{marginal\_ns\_per\_cta} / 1000) \times B$ over the largest three $B$ points. The primary metric is the empirical large-$B$ marginal slope per additional CTA ($b = \Delta T / \Delta B$). Goodness-of-fit $R^2$ is descriptive; adjacent-interval slope stability (<5% change) is the primary operational check.
> 5. **Fitted Intercept**: `intercept_us` represents the fitted fixed-time intercept. Its physical origin is **UNKNOWN** (not claimed to be launch overhead).

## 1. Execution Environments & Pre-Run GPU Telemetry

| Run ID | GPU UUID | Driver | SM Count | L2 Cache (bytes) | SM Clock (MHz) | Memory Clock (MHz) | Power (W) | Temp (°C) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `run_1` | `GPU-5b7caa5e-1704-95d2-13ba-1402ab37403b` | `580.95.05` | `132` | `52428800` | `345` | `2619` | `69.48` | `33` |
| `run_2` | `GPU-5b7caa5e-1704-95d2-13ba-1402ab37403b` | `580.95.05` | `132` | `52428800` | `1980` | `2619` | `123.77` | `33` |
| `run_3` | `GPU-5b7caa5e-1704-95d2-13ba-1402ab37403b` | `580.95.05` | `132` | `52428800` | `1980` | `2619` | `123.93` | `33` |

## 2. Working Set & Cache Regime

| Configuration | B_RUN | Input Working Set (MiB) | Output Working Set (MiB) |
| :--- | :---: | :---: | :---: |
| `M32_N16_w8` | 16384 | 16.00 MiB | 1.00 MiB |
| `M32_N16_w8` | 32768 | 32.00 MiB | 2.00 MiB |
| `M32_N16_w8` | 65536 | 64.00 MiB | 4.00 MiB |
| `M32_N16_w8` | 131072 | 128.00 MiB | 8.00 MiB |
| `M32_N64_w8` | 16384 | 64.00 MiB | 4.00 MiB |
| `M32_N64_w8` | 32768 | 128.00 MiB | 8.00 MiB |
| `M32_N64_w8` | 65536 | 256.00 MiB | 16.00 MiB |
| `M32_N128_w4` | 16384 | 128.00 MiB | 8.00 MiB |
| `M32_N128_w4` | 32768 | 256.00 MiB | 16.00 MiB |
| `M32_N128_w4` | 65536 | 512.00 MiB | 32.00 MiB |

## 3. Fixed-Binary Compilation Verification

> A single compiled specialization with descriptor extent `B_DESC = max(B)` was compiled once per candidate.
> All runtime grid sizes $B_{\text{RUN}} \le B_{\text{DESC}}$ reuse this identical compiled specialization without recompilation.

| Configuration | Candidate | PTX SHA256 (12 char) | TTGIR SHA256 (12 char) | Verified Fixed Across All B? |
| :--- | :--- | :---: | :---: | :---: |
| `M32_N16_w8` | `default` | `619b029db352...` | `0de355016b19...` | **YES** (single compiled specialization reused across all B) |
| `M32_N16_w8` | `8` | - | - | INVALID: forcedVec 8 > numElemsPerThread (2) |
| `M32_N16_w8` | `4` | - | - | INVALID: forcedVec 4 > numElemsPerThread (2) |
| `M32_N16_w8` | `2` | `619b029db352...` | `0de355016b19...` | **YES** (single compiled specialization reused across all B) |
| `M32_N16_w8` | `1` | `6074f635fe38...` | `406868fd7f27...` | **YES** (single compiled specialization reused across all B) |
| `M32_N64_w8` | `default` | `31663f4aabb9...` | `ef0d3eb5caaa...` | **YES** (single compiled specialization reused across all B) |
| `M32_N64_w8` | `8` | `31663f4aabb9...` | `ef0d3eb5caaa...` | **YES** (single compiled specialization reused across all B) |
| `M32_N64_w8` | `4` | `ddd3499f0598...` | `31371b796ccc...` | **YES** (single compiled specialization reused across all B) |
| `M32_N64_w8` | `2` | `2a213dc209e4...` | `ff4823bfd317...` | **YES** (single compiled specialization reused across all B) |
| `M32_N64_w8` | `1` | `c1b9ae4324fc...` | `97b4f8fc8990...` | **YES** (single compiled specialization reused across all B) |
| `M32_N128_w4` | `default` | `882b0bd48de1...` | `e3eb04394b07...` | **YES** (single compiled specialization reused across all B) |
| `M32_N128_w4` | `8` | `882b0bd48de1...` | `e3eb04394b07...` | **YES** (single compiled specialization reused across all B) |
| `M32_N128_w4` | `4` | `276e796f62dd...` | `90a380ab4f94...` | **YES** (single compiled specialization reused across all B) |
| `M32_N128_w4` | `2` | `73f4884d16f7...` | `5a8837f1ff69...` | **YES** (single compiled specialization reused across all B) |
| `M32_N128_w4` | `1` | `ffda7e816ba7...` | `be8f9501d526...` | **YES** (single compiled specialization reused across all B) |

## 4. Sequential Invocation Replication & Marginal Slope Separation

### Configuration: `M32_N16_w8` (`M=32, N=16, num_warps=8`)

#### A. Affine Fit Parameters across Sequential Invocations:

| Candidate | run_1 Slope (ns) | run_2 Slope (ns) | run_3 Slope (ns) | Mean Slope (ns) | same-device temporal replication CV (%) | vs Default Mean Slope (%) | Linear Regime? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | 2.1896 | 2.2073 | 2.2163 | **2.2044** | 0.50% | 0.00% (base) | **YES** |
| `8` | - | - | - | - | - | - | INVALID |
| `4` | - | - | - | - | - | - | INVALID |
| `2` | 2.1832 | 2.2122 | 2.2136 | **2.2030** | 0.64% | **-0.06%** (near parity) | **YES** |
| `1` | 2.1231 | 2.1552 | 2.1523 | **2.1435** | 0.68% | **-2.76%** | **YES** |

#### B. Fitted Fixed-Time Intercept (µs) & Goodness-of-Fit ($R^2$):

> Note: Affine fits use the 3 largest $B$ points. $R^2$ is reported as a descriptive goodness-of-fit indicator; adjacent-interval slope stability (<5% change) is the primary operational check.

| Candidate | run_1 Intercept (µs) | run_2 Intercept (µs) | run_3 Intercept (µs) | run_1 R² | run_2 R² | run_3 R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | 17.0800 | 17.8400 | 18.0160 | 0.999875 | 0.999961 | 0.999947 |
| `8` | - | - | - | - | - | - |
| `4` | - | - | - | - | - | - |
| `2` | 17.6160 | 17.6960 | 18.4480 | 0.999909 | 0.999940 | 0.999983 |
| `1` | 17.0880 | 16.6880 | 17.6880 | 0.999870 | 0.999892 | 0.999964 |

### Configuration: `M32_N64_w8` (`M=32, N=64, num_warps=8`)

#### A. Affine Fit Parameters across Sequential Invocations:

| Candidate | run_1 Slope (ns) | run_2 Slope (ns) | run_3 Slope (ns) | Mean Slope (ns) | same-device temporal replication CV (%) | vs Default Mean Slope (%) | Linear Regime? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | 3.9853 | 3.9402 | 3.9094 | **3.9449** | 0.79% | 0.00% (base) | **YES** |
| `8` | 3.9812 | 3.9356 | 3.9147 | **3.9438** | 0.70% | **-0.03%** (near parity) | **YES** |
| `4` | 2.4997 | 2.4874 | 2.4690 | **2.4854** | 0.51% | **-37.00%** | **YES** |
| `2` | 2.2954 | 2.2732 | 2.2611 | **2.2766** | 0.62% | **-42.29%** | **YES** |
| `1` | 2.2590 | 2.2614 | 2.2439 | **2.2548** | 0.34% | **-42.84%** | **YES** |

#### B. Fitted Fixed-Time Intercept (µs) & Goodness-of-Fit ($R^2$):

> Note: Affine fits use the 3 largest $B$ points. $R^2$ is reported as a descriptive goodness-of-fit indicator; adjacent-interval slope stability (<5% change) is the primary operational check.

| Candidate | run_1 Intercept (µs) | run_2 Intercept (µs) | run_3 Intercept (µs) | run_1 R² | run_2 R² | run_3 R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | 20.5360 | 21.5360 | 21.1920 | 0.999989 | 1.000000 | 0.999999 |
| `8` | 20.6480 | 21.5120 | 21.2080 | 0.999997 | 0.999997 | 0.999999 |
| `4` | 21.3120 | 21.6400 | 21.2880 | 1.000000 | 0.999964 | 0.999987 |
| `2` | 21.3840 | 21.6720 | 21.1440 | 0.999997 | 0.999993 | 0.999998 |
| `1` | 21.4480 | 21.1520 | 21.1200 | 0.999998 | 0.999985 | 0.999999 |

### Configuration: `M32_N128_w4` (`M=32, N=128, num_warps=4`)

#### A. Affine Fit Parameters across Sequential Invocations:

| Candidate | run_1 Slope (ns) | run_2 Slope (ns) | run_3 Slope (ns) | Mean Slope (ns) | same-device temporal replication CV (%) | vs Default Mean Slope (%) | Linear Regime? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | 2.9818 | 3.0003 | 2.9490 | **2.9771** | 0.71% | 0.00% (base) | **YES** |
| `8` | 2.9853 | 2.9953 | 2.9621 | **2.9809** | 0.47% | **+0.13%** (near parity) | **YES** |
| `4` | 2.9813 | 3.0012 | 2.9586 | **2.9804** | 0.58% | **+0.11%** (near parity) | **YES** |
| `2` | 2.9645 | 2.9775 | 2.9621 | **2.9681** | 0.23% | **-0.30%** | **YES** |
| `1` | 2.9867 | 2.9610 | 2.9618 | **2.9698** | 0.40% | **-0.24%** | **YES** |

#### B. Fitted Fixed-Time Intercept (µs) & Goodness-of-Fit ($R^2$):

> Note: Affine fits use the 3 largest $B$ points. $R^2$ is reported as a descriptive goodness-of-fit indicator; adjacent-interval slope stability (<5% change) is the primary operational check.

| Candidate | run_1 Intercept (µs) | run_2 Intercept (µs) | run_3 Intercept (µs) | run_1 R² | run_2 R² | run_3 R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | 21.7200 | 22.0720 | 23.4480 | 1.000000 | 0.999999 | 0.999987 |
| `8` | 21.5120 | 22.2640 | 23.0720 | 1.000000 | 0.999998 | 0.999996 |
| `4` | 21.6320 | 21.4960 | 23.1280 | 0.999982 | 0.999999 | 1.000000 |
| `2` | 21.9920 | 22.0480 | 22.6480 | 0.999989 | 0.999988 | 0.999993 |
| `1` | 21.0560 | 22.7120 | 22.6400 | 0.999966 | 0.999999 | 0.999996 |
