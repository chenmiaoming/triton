# Persistent Triton Build Caches on Modal Infrastructure

## 1. Overview & Architecture

To eliminate redundant compilation of Triton C/C++ translation units across source iterations and Modal image rebuilds, a persistent Modal Volume named `triton-build-cache` is mounted at `/cache` during both image build (`run_commands(..., volumes={"/cache": build_cache_volume})`) and container runtime execution (`@app.function(..., volumes={"/cache": build_cache_volume})`).

The cache architecture decouples mutable source trees from persistent build artifacts:

```text
Modal Host Environment
  └── Named Volume: "triton-build-cache"
        ├── /cache/ccache/         <- C/C++ object compilation cache (ccache 4.9.1)
        └── /cache/triton-home/    <- Prebuilt LLVM & NVIDIA toolchain dependency cache (.triton)
```

The Triton source tree continues to be uploaded directly from the local repository working tree into `/opt/triton-src` to guarantee absolute byte-level source fidelity.

---

## 2. Configuration & Environment Variables

The build environment is provisioned with the following explicit configuration:

| Variable | Value | Purpose |
| :--- | :--- | :--- |
| `CCACHE_DIR` | `/cache/ccache` | Target directory within persistent volume for object cache storage |
| `CCACHE_COMPILERCHECK` | `content` | Hash compiler binary contents rather than mtime to avoid spurious cache invalidations across ephemeral builder containers |
| `CCACHE_MAXSIZE` | `20G` | Maximum cache capacity allocated on the persistent volume |
| `TRITON_HOME` | `/cache/triton-home` | Target directory for Triton dependency downloader (`.triton/llvm`, `.triton/nvidia`) |
| `TRITON_BUILD_WITH_CCACHE` | `true` | Enables ccache detection and launcher wrapping in Triton's build system |
| `TRITON_BUILD_WITH_CLANG_LLD` | `true` | Directs compilation through Clang and LLD linker |
| `MAX_JOBS` | `4` | Parallel compilation job limit for builder stability |

### Rationale for `CCACHE_COMPILERCHECK=content`
In ephemeral Modal image builders, the filesystem mtime of `/usr/bin/clang` and `/usr/bin/clang++` can fluctuate across distinct builder VM instances even when the package versions are identical. Setting `CCACHE_COMPILERCHECK=content` forces ccache to hash the actual binary contents of the compiler executable, preventing unnecessary cache misses while preserving strict compiler integrity.

### Role of `TRITON_HOME` vs `CCACHE_DIR`
`TRITON_HOME` and `CCACHE_DIR` serve distinct, complementary caching roles:
- `TRITON_HOME`: Persists downloaded toolchains and third-party dependencies (e.g., prebuilt LLVM 358 MB distribution, `ptxas`, `nvcc`, `cupti`). By caching these assets in `/cache/triton-home/.triton`, CMake configuration time drops from 110.3s to 10.5s (a 10x speedup).
- `CCACHE_DIR`: Persists compiled C++ object files (`.o`) produced by Clang across the 547 compilation targets in Triton's Ninja build graph.

---

## 3. Build & Diagnostics Pipeline

The Triton image build step in `modal_runner.py` executes the following sequence with explicit diagnostics before and after compilation:

```bash
# 1. Initialize persistent directories and display pre-build cache status
mkdir -p /cache/ccache /cache/triton-home
ccache --version
ccache -s

# 2. Clean previous environment packages
python3 -m pip uninstall -y triton pytorch-triton || true
cd /opt/triton-src && python3 -m pip install -r python/requirements.txt

# 3. Build and install Triton from source
cd /opt/triton-src && \
  TRITON_BUILD_WITH_CLANG_LLD=true \
  TRITON_BUILD_WITH_CCACHE=true \
  python3 -m pip install -e . --no-build-isolation -v

# 4. Display post-build cache status
ccache -s

# 5. Verify installed Triton provenance
python3 -c 'import triton; print("Built Triton verification:", triton.__version__, triton.__file__)'
```

---

## 4. Empirical Validation: Cold Build vs. Warm Rebuild

Two complete end-to-end Modal builds and remote H100 execution runs were performed to validate cache persistence and speedup:

### Build 1 (Cold Build)
- **Condition**: New persistent volume with empty `/cache/ccache` and `/cache/triton-home`.
- **Image ID**: `im-4CgYQELty1KrZ26LeFEnk9`
- **Total Image Build Duration**: 1427.24s (~23.8 minutes)
- **CMake Configuration**: 110.3s (downloaded LLVM 358MB, NVIDIA CUDA toolchain packages)
- **Compilation**: 387 cacheable calls compiled from scratch.
- **Cache Result**: 387 misses, 0 hits (populated 150.5 MB cache).

### Build 2 (Warm Rebuild)
- **Condition**: Triggered by a Python-only source change in `experiments/tma_reduction_layout/modal_runner.py` that invalidated the source layer in Modal, forcing a re-execution of the Triton build step.
- **Image ID**: `im-ZLx09Qs8HAuc9znQ25gCvJ`
- **Total Image Build Duration**: 165.17s (~2.75 minutes) -> **8.6x end-to-end build speedup**
- **CMake Configuration**: 10.5s -> **10.5x configure speedup**
- **Proof of Re-Execution**: The entire Ninja build graph (547 targets) re-ran in container Step 4, generating the complete editable wheel and installing `triton-3.9.0`.
- **Cache Result**: 387 hits out of 387 compiled targets (100% hit rate for Build 2), 0 misses during build.

### Cumulative Cache Statistics Summary

| Metric | Build 1 (Cold) | Build 2 (Warm Rebuild) | Delta |
| :--- | :--- | :--- | :--- |
| **Total Cacheable Calls** | 387 | 774 | +387 |
| **Direct Hits** | 0 (0.0%) | 387 (50.0% cumulative, 100% in Build 2) | +387 |
| **Preprocessed Hits** | 0 | 0 | 0 |
| **Misses** | 387 (100.0%) | 387 (50.0% cumulative, 0 in Build 2) | 0 |
| **Cache Size** | 150.5 MB (0.2 GB) | 150.5 MB (0.2 GB) | 0 |
| **TRITON_HOME Size** | 2.5 GB | 2.5 GB | 0 |
| **Image Build Time** | 1427.24s | 165.17s | **-1262.07s (8.6x faster)** |

---

## 5. Source Fidelity & Hardware Smoke Test

The caching mechanism strictly accelerates build time without affecting source fidelity or execution integrity:

1. **Source Provenance Verification**:
   - Remote verification in `remote_verify_environment()` confirmed all 1,815 files in the local upload manifest matched remote `/opt/triton-src` byte-for-byte (`uploaded_source_fidelity_verified: true`, identical SHA-256: `5abf0a8118f77c4045560e38eefc6cd2d57ecfb22fc6dce7d4b3b275eb3a4235`).
   - Triton package loaded strictly from `/opt/triton-src/python/triton/__init__.py`.
2. **Hardware Smoke Test**:
   - Target GPU: NVIDIA H100 80GB HBM3 (`GPU-549ddf90-70ae-2829-5d0d-25cc0a3d5872`), Driver `580.95.05`, Compute Capability `(9, 0)`.
   - Vector addition smoke test (`vector_add_kernel` on 2048 float32 elements) executed successfully with `max_abs_diff: 0.0`.

---

## 6. Concurrency & Scope Boundaries

### Volume Concurrency Model
The persistent build cache volume assumes **serialized image builds**. Simultaneous image builds writing to the same named Modal volume concurrently could create race conditions in cache metadata. Modal builds must be run sequentially.

### Modal Workspace Boundaries
The named Modal Volume `triton-build-cache` persists across container and image rebuilds **within the specific Modal account and workspace where the volume resides**. A different Modal account or isolated workspace without access to that named volume will start with a fresh, cold cache. The cache is not globally shared across separate Modal accounts.

### Expected Cache Invalidation Triggers
In accordance with standard compiler caching semantics, the following changes will naturally and correctly trigger cache misses:
- Changes to compiler binaries or compiler versions (`clang`, `lld`).
- Changes to compiler or preprocessor flags (`-O3`, `-D...`).
- Changes to CUDA version or toolchain headers (`CUDA_HOME`).
- Content modifications to C/C++ translation units (`.cpp`, `.cc`, `.c`).
- Content modifications to relevant C/C++ header files (`.h`, `.inc`).
- Build configuration changes in CMake or Ninja targets.

These invalidations represent correct compiler behavior ensuring build correctness.
