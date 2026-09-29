"""
Modal runner for Triton TMA reduction layout experiments.

Provides:
- Modal image definition building Triton from current working tree (/opt/triton-src).
- Strict H100 hardware verification (gpu="H100!:1", compute capability == (9, 0)).
- Triton provenance verification (must import from /opt/triton-src).
- Infrastructure smoke test.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

for p in ["/opt/triton-src", str(Path(__file__).resolve().parent.parent.parent)]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import modal

from experiments.tma_reduction_layout.source_provenance import (
    MODAL_SOURCE_IGNORE_PATTERNS,
    generate_provenance,
    get_repo_root,
    save_provenance,
    verify_remote_source_manifest,
)

REPO_ROOT = get_repo_root()
APP_NAME = "triton-tma-reduction-layout"

app = modal.App(APP_NAME)

# ---------------------------------------------------------------------------
# Layered Modal Image definition
# ---------------------------------------------------------------------------
# 1. Base CUDA devel image with system build tools
base_cuda_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install(
        "build-essential",
        "git",
        "clang",
        "lld",
        "ccache",
        "ninja-build",
        "cmake",
        "pkg-config",
        "zlib1g-dev",
        "curl",
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
        "TRITON_BUILD_WITH_CLANG_LLD": "true",
        "TRITON_BUILD_WITH_CCACHE": "true",
        "MAX_JOBS": "8",
    })
)

# 2. PyTorch and base python packages layer (cached separately from Triton source)
# Notice: PyTorch is installed first, then packaged triton is explicitly removed.
torch_and_deps_image = (
    base_cuda_image
    .uv_pip_install(
        "torch",
        "tabulate",
        "numpy",
        "pandas",
    )
    .run_commands(
        "python3 -m pip uninstall -y triton pytorch-triton || true"
    )
    .uv_pip_install(
        "setuptools>=40.8.0",
        "wheel",
        "cmake>=3.20,<4.0",
        "ninja>=1.11.1",
        "nanobind==2.10.2",
        "lit",
    )
)

# 3. Add local working tree and build Triton from source (/opt/triton-src)
triton_image = (
    torch_and_deps_image
    .add_local_dir(
        REPO_ROOT,
        remote_path="/opt/triton-src",
        copy=True,
        ignore=MODAL_SOURCE_IGNORE_PATTERNS,
    )
    .run_commands(
        "python3 -m pip uninstall -y triton pytorch-triton || true",
        "cd /opt/triton-src && python3 -m pip install -r python/requirements.txt",
        "cd /opt/triton-src && TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true python3 -m pip install -e . --no-build-isolation -v",
        "python3 -c 'import triton; print(\"Built Triton verification:\", triton.__version__, triton.__file__)'",
    )
)


# ---------------------------------------------------------------------------
# Verification Helpers (Run inside container)
# ---------------------------------------------------------------------------
def remote_verify_environment(local_provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import subprocess
    import torch
    import triton
    from experiments.tma_reduction_layout.source_provenance import verify_remote_source_manifest

    # 1. Hardware verification
    smi_out = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=name,uuid,memory.total,driver_version",
        "--format=csv,noheader",
    ], text=True).strip()
    print(f"[Remote] nvidia-smi: {smi_out}")

    smi_parts = [p.strip() for p in smi_out.split(",")]
    gpu_name = smi_parts[0] if len(smi_parts) > 0 else "unknown"
    gpu_uuid = smi_parts[1] if len(smi_parts) > 1 else "unknown"
    gpu_mem = smi_parts[2] if len(smi_parts) > 2 else "unknown"
    driver_ver = smi_parts[3] if len(smi_parts) > 3 else "unknown"

    torch_dev_name = torch.cuda.get_device_name(0)
    dev_cap = torch.cuda.get_device_capability(0)
    torch_cuda = torch.version.cuda
    print(f"[Remote] Torch device: {torch_dev_name}, CC: {dev_cap}, CUDA: {torch_cuda}")

    # Strict H100 verification
    assert "H100" in torch_dev_name, (
        f"CRITICAL: Expected NVIDIA H100, but detected: {torch_dev_name}!"
    )
    assert dev_cap == (9, 0), (
        f"CRITICAL: Expected compute capability (9, 0) for H100, but got {dev_cap}!"
    )
    for forbidden in ["H200", "B200", "B300"]:
        assert forbidden not in torch_dev_name, (
            f"CRITICAL: Device {torch_dev_name} contains forbidden GPU tag '{forbidden}'!"
        )

    # 2. Triton provenance verification
    print(f"[Remote] triton.__version__ = {triton.__version__}")
    print(f"[Remote] triton.__file__ = {triton.__file__}")

    assert "/opt/triton-src" in triton.__file__, (
        f"CRITICAL: Triton was NOT imported from /opt/triton-src! Found: {triton.__file__}. "
        "Halting immediately."
    )

    # 3. Source Manifest fidelity verification
    manifest_ver = {}
    if local_provenance and "source_manifest" in local_provenance:
        print("[Remote] Verifying uploaded source-file fidelity in /opt/triton-src...")
        manifest_ver = verify_remote_source_manifest(local_provenance)
        print(
            f"[Remote] Uploaded source-file fidelity verified: {manifest_ver.get('files_verified')} files "
            f"match local manifest with identical bytes ({manifest_ver.get('remote_extra_file_count')} post-build extra files recorded)."
        )

    # 4. Toolchain inspection
    def run_tool(cmd):
        try:
            return subprocess.check_output(cmd, text=True).strip()
        except Exception as e:
            return f"Error: {e}"

    ptxas_ver = run_tool(["ptxas", "--version"])
    nvdisasm_ver = run_tool(["nvdisasm", "--version"])
    cuobjdump_ver = run_tool(["cuobjdump", "--version"])

    return {
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "gpu_memory": gpu_mem,
        "gpu_compute_capability": list(dev_cap),
        "driver_version": driver_ver,
        "torch_device_name": torch_dev_name,
        "torch_cuda_version": torch_cuda,
        "pytorch_version": torch.__version__,
        "python_version": sys.version,
        "triton_version": triton.__version__,
        "triton_file": triton.__file__,
        "ptxas_version": ptxas_ver,
        "nvdisasm_version": nvdisasm_ver,
        "cuobjdump_version": cuobjdump_ver,
        "manifest_verification": manifest_ver,
    }


# ---------------------------------------------------------------------------
# Smoke Test Kernel & Function
# ---------------------------------------------------------------------------
@app.function(
    image=triton_image,
    gpu="H100!:1",
    timeout=600,
)
def run_smoke_test_remote(local_provenance: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    import triton
    import triton.language as tl

    # 1. Environment & Hardware Verification
    env_info = remote_verify_environment(local_provenance)

    # 2. Define simple vector addition kernel
    @triton.jit
    def vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = x + y
        tl.store(out_ptr + offsets, out, mask=mask)

    # 3. Run kernel
    n = 2048
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=256)
    torch.cuda.synchronize()

    expected = x + y
    max_abs_diff = float(torch.max(torch.abs(out - expected)).item())
    assert torch.allclose(out, expected, atol=1e-5), f"Numerical mismatch! Max diff: {max_abs_diff}"
    print(f"[Remote] Smoke test kernel PASSED successfully! (max diff: {max_abs_diff})")

    result = {
        "status": "PASS",
        "smoke_kernel": {
            "name": "vector_add_kernel",
            "elements": n,
            "max_abs_diff": max_abs_diff,
        },
        "environment": env_info,
        "local_provenance": local_provenance,
    }
    return result


@app.local_entrypoint()
def smoke_test():
    """Local entrypoint for Modal CLI: modal run experiments/tma_reduction_layout/modal_runner.py::smoke_test"""
    execute_smoke_test()


def execute_smoke_test() -> Dict[str, Any]:
    print("==================================================")
    print("Starting Modal Infrastructure Smoke Test")
    print("==================================================")

    # 1. Check local proxy
    proxies = {k: v for k, v in os.environ.items() if "proxy" in k.lower()}
    if proxies:
        print("[Local] Active proxy environment variables detected:")
        for k, v in proxies.items():
            print(f"  {k} = {v}")
    else:
        print("[Local] Note: No proxy environment variables found.")

    # 2. Generate local provenance
    prov = generate_provenance(REPO_ROOT)
    print(f"[Local] Git HEAD: {prov['git_head_sha']} (branch: {prov['branch']})")
    print(f"[Local] Dirty status: {prov['is_dirty']} (diff sha256: {prov['git_diff_head_sha256']})")

    # 3. Run smoke test on Modal H100
    print("[Local] Dispatching smoke test to Modal (gpu='H100!:1')...")
    with modal.enable_output():
        with app.run():
            result = run_smoke_test_remote.remote(prov)

    # 4. Save results
    results_dir = REPO_ROOT / "experiments" / "tma_reduction_layout" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "smoke_test.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[Local] Smoke test results successfully saved to: {out_file}")

    print("==================================================")
    print("Smoke Test Verification Summary:")
    print(f"  Status: {result['status']}")
    print(f"  GPU Name: {result['environment']['gpu_name']}")
    print(f"  GPU Compute Capability: {result['environment']['gpu_compute_capability']}")
    print(f"  Torch Device: {result['environment']['torch_device_name']}")
    print(f"  Triton Version: {result['environment']['triton_version']}")
    print(f"  Triton Path: {result['environment']['triton_file']}")
    print(f"  PTXAS Version: {result['environment']['ptxas_version'].splitlines()[0] if result['environment']['ptxas_version'] else 'N/A'}")
    print("==================================================")
    return result


if __name__ == "__main__":
    execute_smoke_test()
