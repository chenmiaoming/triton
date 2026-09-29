# TMA Reduction Layout Experiment

This directory hosts an isolated experimental setup to investigate layout trade-offs in:

```text
TMA descriptor load
    -> shared memory
    -> distributed/register layout
    -> non-innermost partial reduction
```

---

## Evidence Discipline

Every factual statement, metric, and conclusion recorded in this repository must adhere to the following evidence discipline:

- **Raw artifacts are authoritative**: Committed IR, PTX, SASS, tool outputs, and timing logs are the single source of truth.
- **Automated analysis must be reproducible from committed artifacts**: Scripts must re-derive all reported metrics directly from committed files.
- **Derived metrics must state their explicit formula**: Analytical values (e.g. partition ownership, repetition factors) must document their governing equations.
- **Mechanistic interpretations must cite exact artifact/line ranges**: Any claim regarding execution phases or instruction roles must cite specific instruction sequences.
- **Performance claims require a rigorous protocol and noise context**: No performance ranking may be claimed when differences fall within empirical measurement noise.
- **Unknown values remain UNKNOWN**: Do not extrapolate or guess missing measurements.
- **PTX virtual registers are NOT physical register allocation**: PTX `.reg .b32 %r<N>` declarations indicate virtual register index bounds, not physical registers per thread.
- **Whole-kernel opcode counts must NOT be labeled as reduction-specific**: Global PTX counts encompass TMA lifecycle, setup, LocalLoad, reduction arithmetic, layout conversion, and stores.
- **Semantic PTX phase attribution must not be inferred from absolute line number thresholds**: Phases must be human-audited and grounded in instruction sequences.
- **Phase annotations must be bound to artifact hashes**: When IR or PTX artifacts change, phase annotations must fail validation until re-audited.
- **cuobjdump SHARED and Triton launch-time shared-memory metadata are distinct quantities**: Static ELF shared memory (`cuobjdump -res-usage`) and launch-time dynamic shared memory (`compiled.metadata.shared`) must not be conflated.
- **Source fidelity verifies uploaded files, not the entire post-build directory**: A remote post-build source tree may contain generated files; source fidelity means all uploaded source-manifest files match byte-for-byte, not that the entire post-build directory tree is identical.

---

## Data Taxonomy

All data in this repository is categorized into five distinct tiers:

1. **`RAW`**: Raw timing samples, artifact content hashes, hardware queries, and tool outputs (`cuobjdump`).
2. **`OBSERVED`**: Direct extractions from compiled artifacts (e.g. BlockedEncoding attributes, opcode-family counts, LocalLoad lowering, physical registers).
3. **`DERIVED`**: Quantities calculated from layout definitions via explicit mathematical formulas (e.g. lane partitions, warp partitions, elements per partition).
4. **`INFERRED`**: Mechanistic explanations of instruction sequences justified by dataflow context and cited line ranges.
5. **`MEASURED`**: Statistical summary metrics (median, P10, P90, IQR, MAD) accompanied by experimental uncertainty.

---

## Environment & Infrastructure Design

### Base Image & CUDA Toolchain
- **Base Image**: `nvidia/cuda:12.6.3-devel-ubuntu24.04` (with `add_python="3.12"`)
  - Provides development headers, runtime, and critical toolchain binaries (`ptxas`, `cuobjdump`, `nvdisasm`) on Ubuntu 24.04.
  - Development tools: `build-essential`, `git`, `clang`, `lld`, `ccache`, `ninja-build`, `cmake`, `pkg-config`, `zlib1g-dev`.

### Source Guarantee & Provenance
- Local source code from `/home/chenmiaoming/triton-exp` is baked directly into the Modal image at `/opt/triton-src` using `modal.Image.add_local_dir(..., copy=True)`.
- Prior to image build:
  1. A deterministic SHA256 `source_manifest` is computed locally covering all source files.
  2. Git HEAD SHA, porcelain status, binary diff SHA256, and untracked file digests are recorded.
- Inside the Modal container:
  1. PyTorch and base packages are installed.
  2. Any bundled/packaged `triton` or `pytorch-triton` is explicitly removed.
  3. Triton is built and installed in editable mode from `/opt/triton-src` using the official upstream recommended approach:
     ```bash
     cd /opt/triton-src
     python3 -m pip install -r python/requirements.txt
     TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true python3 -m pip install -e . --no-build-isolation -v
     ```
  4. Remote runtime assertions strictly verify:
     - `triton.__file__` begins with `/opt/triton-src`
     - Remote source manifest matches the local source manifest byte-for-byte.
     - GPU is strictly NVIDIA H100 with Compute Capability (9, 0) via `gpu="H100!:1"`.

---

## Repository Structure

```text
experiments/tma_reduction_layout/
    README.md
    modal_runner.py
    benchmark.py
    analyze_ir.py
    source_provenance.py
    results/
        baseline_results.json
        baseline_summary_table.md
        baseline_evidence.md
        smoke_test.json
        artifacts/
            default.ttgir, default.ptx, default.sass, default.resource.txt
            8.ttgir, 8.ptx, 8.sass, 8.resource.txt
            4.ttgir, 4.ptx, 4.sass, 4.resource.txt
            2.ttgir, 2.ptx, 2.sass, 2.resource.txt
            1.ttgir, 1.ptx, 1.sass, 1.resource.txt
```
