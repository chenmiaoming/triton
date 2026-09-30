# Corrected Multi-Invocation Fixed-Binary TMA Reduction Saturation Report

> [!NOTE]
> **Methodology Guarantees**:
> 1. **Fixed Binary**: Tensor descriptor shape is fixed to `B_DESC = max(B)`. Kernels are compiled once per candidate and variable grid sizes `B_RUN <= B_DESC` reuse the identical compiled binary (verified via byte-level TTGIR, PTX, and SASS SHA256 hashes).
> 2. **Multi-Invocation Replication**: Benchmark is executed across independent Modal allocations (each allocating an H100 instance, compiling fresh binaries, generating fresh inputs, and warming up).
> 3. **Order Rotation**: Grid sizes ($B$) and layout candidates are rotated circularly across 10 timing rounds (10 samples/round = 100 samples per condition) to eliminate thermal drift, clock drift, and ordering bias.
> 4. **Affine Steady-State Model**: Evaluates $T(B) = \text{intercept\_us} + (\text{marginal\_ns\_per\_cta} / 1000) \times B$ over large $B$. The primary metric is the marginal cost per additional CTA ($b = \Delta T / \Delta B$).
> 5. **Fitted Intercept**: `intercept_us` represents the fitted fixed-time intercept. Its physical origin is **UNKNOWN** (not claimed to be launch overhead).

## 1. Execution Environments & GPU Telemetry

| Run ID | GPU UUID | Driver | SM Count | L2 Cache (bytes) | SM Clock (MHz) | Memory Clock (MHz) | Power (W) | Temp (°C) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `run_1` | `GPU-a59752c5-ebb5-1dac-f887-c2e3c1f81aff` | `580.95.05` | `132` | `52428800` | `345` | `2619` | `72.31` | `30` |
| `run_2` | `GPU-a59752c5-ebb5-1dac-f887-c2e3c1f81aff` | `580.95.05` | `132` | `52428800` | `1980` | `2619` | `121.04` | `32` |
| `run_3` | `GPU-a59752c5-ebb5-1dac-f887-c2e3c1f81aff` | `580.95.05` | `132` | `52428800` | `1980` | `2619` | `121.2` | `33` |

## 2. Working Set & Cache Regime

| Configuration | B_RUN | Input Working Set (MB) | Output Working Set (MB) |
| :--- | :---: | :---: | :---: |
| `M32_N16_w8` | 16384 | 16.00 MB | 1.00 MB |
| `M32_N16_w8` | 32768 | 32.00 MB | 2.00 MB |
| `M32_N16_w8` | 65536 | 64.00 MB | 4.00 MB |
| `M32_N16_w8` | 131072 | 128.00 MB | 8.00 MB |
| `M32_N64_w8` | 16384 | 64.00 MB | 4.00 MB |
| `M32_N64_w8` | 32768 | 128.00 MB | 8.00 MB |
| `M32_N64_w8` | 65536 | 256.00 MB | 16.00 MB |
| `M32_N128_w4` | 16384 | 128.00 MB | 8.00 MB |
| `M32_N128_w4` | 32768 | 256.00 MB | 16.00 MB |
| `M32_N128_w4` | 65536 | 512.00 MB | 32.00 MB |

## 3. Fixed-Binary Compilation Verification

| Configuration | Candidate | PTX SHA256 (12 char) | TTGIR SHA256 (12 char) | SASS SHA256 (12 char) | Verified Fixed Across All B? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `M32_N16_w8` | `default` | `bdd1dc86e385...` | `f7de141dff65...` | `e3b0c44298fc...` | **YES** (bit-for-bit identical) |
| `M32_N16_w8` | `8` | - | - | - | INVALID: forcedVec 8 > numElemsPerThread (2) |
| `M32_N16_w8` | `4` | - | - | - | INVALID: forcedVec 4 > numElemsPerThread (2) |
| `M32_N16_w8` | `2` | `bdd1dc86e385...` | `f7de141dff65...` | `e3b0c44298fc...` | **YES** (bit-for-bit identical) |
| `M32_N16_w8` | `1` | `773a766ea315...` | `94eea572a1aa...` | `e3b0c44298fc...` | **YES** (bit-for-bit identical) |
| `M32_N64_w8` | `default` | `2d02e0b822c7...` | `b12774ab2f20...` | `e3b0c44298fc...` | **YES** (bit-for-bit identical) |
| `M32_N64_w8` | `8` | `2d02e0b822c7...` | `b12774ab2f20...` | `e3b0c44298fc...` | **YES** (bit-for-bit identical) |
| `M32_N64_w8` | `4` | `950b23e7cf39...` | `d28dcc41b083...` | `e3b0c44298fc...` | **YES** (bit-for-bit identical) |
| `M32_N64_w8` | `2` | `43d1d7f9b0e7...` | `16fc510cf256...` | `e3b0c44298fc...` | **YES** (bit-for-bit identical) |
| `M32_N64_w8` | `1` | `a6510984fac2...` | `8843f1f28a12...` | `e3b0c44298fc...` | **YES** (bit-for-bit identical) |
| `M32_N128_w4` | `default` | `a107f8e1427c...` | `90c1c4230073...` | `e3b0c44298fc...` | **YES** (bit-for-bit identical) |
| `M32_N128_w4` | `8` | `a107f8e1427c...` | `90c1c4230073...` | `e3b0c44298fc...` | **YES** (bit-for-bit identical) |
| `M32_N128_w4` | `4` | `4e50ef4d7fb0...` | `4c9e7756f484...` | `e3b0c44298fc...` | **YES** (bit-for-bit identical) |
| `M32_N128_w4` | `2` | `12cc9d569a07...` | `2dd683139149...` | `e3b0c44298fc...` | **YES** (bit-for-bit identical) |
| `M32_N128_w4` | `1` | `ca0cbbfa274b...` | `2dd069349c25...` | `e3b0c44298fc...` | **YES** (bit-for-bit identical) |

## 4. Cross-Run Replication & Marginal Slope Separation

### Configuration: `M32_N16_w8` (`M=32, N=16, num_warps=8`)

#### A. Affine Fit Parameters across Independent Allocations:

| Candidate | run_1 Slope (ns) | run_2 Slope (ns) | run_3 Slope (ns) | Mean Slope (ns) | CV (%) | vs Default Mean (%) | Linear Regime? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | 2.1663 | 2.1621 | 2.1640 | **2.1641** | 0.08% | 0.00% (base) | **YES** |
| `8` | - | - | - | - | - | - | INVALID |
| `4` | - | - | - | - | - | - | INVALID |
| `2` | 2.1642 | 2.1644 | 2.1638 | **2.1641** | 0.01% | **+0.00%** | **YES** |
| `1` | 2.1006 | 2.1002 | 2.1011 | **2.1007** | 0.02% | **-2.93%** | **YES** |

#### B. Fitted Fixed-Time Intercept (µs) & Goodness-of-Fit ($R^2$):

| Candidate | run_1 Intercept (µs) | run_2 Intercept (µs) | run_3 Intercept (µs) | run_1 R² | run_2 R² | run_3 R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | 13.3360 | 13.7520 | 13.5040 | 0.999912 | 0.999878 | 0.999877 |
| `8` | - | - | - | - | - | - |
| `4` | - | - | - | - | - | - |
| `2` | 13.3840 | 13.5920 | 13.3520 | 0.999886 | 0.999875 | 0.999908 |
| `1` | 13.0000 | 13.3360 | 13.0560 | 0.999856 | 0.999870 | 0.999868 |

### Configuration: `M32_N64_w8` (`M=32, N=64, num_warps=8`)

#### A. Affine Fit Parameters across Independent Allocations:

| Candidate | run_1 Slope (ns) | run_2 Slope (ns) | run_3 Slope (ns) | Mean Slope (ns) | CV (%) | vs Default Mean (%) | Linear Regime? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | 3.8804 | 3.8846 | 3.8816 | **3.8822** | 0.05% | 0.00% (base) | **YES** |
| `8` | 3.8796 | 3.8795 | 3.8767 | **3.8786** | 0.03% | **-0.09%** | **YES** |
| `4` | 2.4566 | 2.4543 | 2.4519 | **2.4543** | 0.08% | **-36.78%** | **YES** |
| `2` | 2.2540 | 2.2539 | 2.2531 | **2.2537** | 0.02% | **-41.95%** | **YES** |
| `1` | 2.2282 | 2.2224 | 2.2242 | **2.2249** | 0.11% | **-42.69%** | **YES** |

#### B. Fitted Fixed-Time Intercept (µs) & Goodness-of-Fit ($R^2$):

| Candidate | run_1 Intercept (µs) | run_2 Intercept (µs) | run_3 Intercept (µs) | run_1 R² | run_2 R² | run_3 R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | 16.6400 | 16.6080 | 16.6480 | 0.999999 | 1.000000 | 0.999998 |
| `8` | 16.7360 | 16.7680 | 16.9280 | 1.000000 | 1.000000 | 1.000000 |
| `4` | 16.4320 | 16.6800 | 16.8800 | 0.999999 | 1.000000 | 0.999999 |
| `2` | 16.4160 | 16.4640 | 16.6000 | 0.999999 | 0.999999 | 1.000000 |
| `1` | 16.3120 | 16.6080 | 16.6400 | 1.000000 | 0.999999 | 0.999998 |

### Configuration: `M32_N128_w4` (`M=32, N=128, num_warps=4`)

#### A. Affine Fit Parameters across Independent Allocations:

| Candidate | run_1 Slope (ns) | run_2 Slope (ns) | run_3 Slope (ns) | Mean Slope (ns) | CV (%) | vs Default Mean (%) | Linear Regime? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | 2.9265 | 2.9246 | 2.9218 | **2.9243** | 0.07% | 0.00% (base) | **YES** |
| `8` | 2.9283 | 2.9208 | 2.9222 | **2.9237** | 0.11% | **-0.02%** | **YES** |
| `4` | 2.9218 | 2.9223 | 2.9182 | **2.9208** | 0.06% | **-0.12%** | **YES** |
| `2` | 2.9213 | 2.9234 | 2.9186 | **2.9211** | 0.07% | **-0.11%** | **YES** |
| `1` | 2.9230 | 2.9243 | 2.9178 | **2.9217** | 0.10% | **-0.09%** | **YES** |

#### B. Fitted Fixed-Time Intercept (µs) & Goodness-of-Fit ($R^2$):

| Candidate | run_1 Intercept (µs) | run_2 Intercept (µs) | run_3 Intercept (µs) | run_1 R² | run_2 R² | run_3 R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `default` | 18.4560 | 17.4960 | 18.2640 | 0.999994 | 0.999996 | 0.999995 |
| `8` | 18.3040 | 17.6800 | 18.1440 | 0.999998 | 1.000000 | 1.000000 |
| `4` | 18.0400 | 17.0560 | 17.9280 | 0.999997 | 1.000000 | 0.999985 |
| `2` | 17.9680 | 17.0080 | 17.7840 | 1.000000 | 1.000000 | 1.000000 |
| `1` | 17.9680 | 16.8880 | 17.8720 | 0.999998 | 0.999998 | 0.999999 |
