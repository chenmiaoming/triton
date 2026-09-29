# TMA Reduction Layout Experiment

This directory hosts an isolated experimental setup to investigate layout trade-offs in:

```text
TMA descriptor load
    -> shared memory
    -> distributed/register layout
    -> non-innermost partial reduction
```

## Environment & Infrastructure Design

### Base Image & CUDA Toolchain
- **Base Image**: `nvidia/cuda:12.6.3-devel-ubuntu24.04` (with `add_python="3.12"`)
  - Rationale: A standard, modern CUDA devel image providing development headers, runtime, and critical toolchain binaries (`ptxas`, `cuobjdump`, `nvdisasm`) on Ubuntu 24.04.
  - Development tools: `build-essential`, `git`, `clang`, `lld`, `ccache`, `ninja-build`, `cmake`, `pkg-config`, `zlib1g-dev`.

### Source Guarantee & Provenance
- Local source code from `/home/chenmiaoming/triton-exp` is baked directly into the Modal image at `/opt/triton-src` using `modal.Image.add_local_dir(..., copy=True)`.
- `.git`, `.venv`, `build`, `__pycache__`, and experimental results directories are excluded from upload.
- Prior to building Triton:
  1. PyTorch and base packages are installed.
  2. Any bundled/packaged `triton` or `pytorch-triton` is explicitly removed.
  3. Triton is built and installed in editable mode from `/opt/triton-src` using the official upstream recommended approach:
     ```bash
     cd /opt/triton-src
     python3 -m pip install -r python/requirements.txt
     TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true python3 -m pip install -e . --no-build-isolation -v
     ```
- At runtime on the remote container, assertions strictly verify:
  - `triton.__file__` begins with `/opt/triton-src`
  - GPU is strictly NVIDIA H100 with Compute Capability (9, 0) via `gpu="H100!:1"`

### Repository Structure
```text
experiments/tma_reduction_layout/
    README.md
    modal_runner.py
    benchmark.py
    analyze_ir.py
    source_provenance.py
    results/
```
