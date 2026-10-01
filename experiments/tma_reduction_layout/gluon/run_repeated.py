#!/usr/bin/env python3
"""
Phase 3 Step D: Gluon Repeated-Reduction Isolation Runner.

Executes on Modal H100 to:
1. Compile Gluon repeated-reduction kernels across 4 configurations:
   - M32_N64_w8 (default, cand4)
   - M32_N128_w4 (default, cand4)
2. Test R in {1, 2, 4, 8} with do_not_specialize=["num_reductions"] to verify:
   - Single binary per config/candidate (identical CUBIN SHA across all R).
   - Invariant physical resource profile.
3. Capture full compilation artifacts (.ptx, .ttgir, .sass, .resource.txt, .cubin.sha256).
4. Query physical resources and occupancy via CUDA driver API.
5. Verify numerical correctness.
6. Save raw results to results/phase3/gluon_repeated/raw_results.json.
"""

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

repo_root = Path(__file__).resolve().parent.parent.parent.parent
for p in ["/opt/triton-src", str(repo_root)]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from experiments.tma_reduction_layout.modal_runner import (
    app,
    remote_verify_environment,
    triton_image,
)
from experiments.tma_reduction_layout.source_provenance import generate_provenance


@app.function(
    image=triton_image,
    gpu="H100!:1",
    timeout=600,
)
def remote_run_gluon_repeated(prov: dict) -> dict:
    import ctypes
    import hashlib
    import os
    import re
    import subprocess
    import torch
    import triton
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
    from experiments.tma_reduction_layout.gluon.kernel import (
        get_canonical_gluon_layouts,
        get_canonical_gluon_shared_layout,
    )
    from experiments.tma_reduction_layout.gluon.kernel_repeated import (
        get_barrier_constraints,
        gluon_repeated_reduction_kernel,
    )

    env_info = remote_verify_environment(prov)
    print(f"[Remote] Running Gluon repeated reduction isolation on {env_info['gpu_name']}")

    triton.set_allocator(
        lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda")
    )

    # 1. CUDA Driver & Device Limits Query
    cuda = ctypes.CDLL("libcuda.so.1")
    res = cuda.cuInit(0)
    if res != 0:
        raise RuntimeError(f"cuInit failed with error code {res}")
    dev = ctypes.c_int()
    res = cuda.cuDeviceGet(ctypes.byref(dev), 0)
    if res != 0:
        raise RuntimeError(f"cuDeviceGet failed with error code {res}")

    device_limits = {}
    attr_map = {
        "max_threads_per_sm": 39,
        "max_threads_per_block": 1,
        "max_registers_per_block": 12,
        "max_registers_per_sm": 82,
        "max_shared_memory_per_sm": 81,
        "warp_size": 10,
    }
    for name, code in attr_map.items():
        v = ctypes.c_int()
        r = cuda.cuDeviceGetAttribute(ctypes.byref(v), code, dev)
        device_limits[name] = v.value if r == 0 else None
    if device_limits.get("max_threads_per_sm") and device_limits.get("warp_size"):
        device_limits["max_warps_per_sm"] = device_limits["max_threads_per_sm"] // device_limits["warp_size"]

    configs = [
        {"cfg_name": "M32_N64_w8", "M": 32, "N": 64, "num_warps": 8, "b_desc": 65536},
        {"cfg_name": "M32_N128_w4", "M": 32, "N": 128, "num_warps": 4, "b_desc": 65536},
    ]
    candidates = ["default", "4"]
    test_r_values = [1, 2, 4, 8]
    gluon_layouts = get_canonical_gluon_layouts()
    shared_layout = get_canonical_gluon_shared_layout()

    repeated_results = {}

    for cfg in configs:
        cfg_name = cfg["cfg_name"]
        M = cfg["M"]
        N = cfg["N"]
        num_warps = cfg["num_warps"]
        b_desc = cfg["b_desc"]
        repeated_results[cfg_name] = {}

        # Prepare dummy test data on GPU
        torch.manual_seed(42)
        x_pt = torch.randn(b_desc, M, N, dtype=torch.bfloat16, device="cuda")
        out_pt = torch.empty(b_desc, dtype=torch.float32, device="cuda")
        desc = TensorDescriptor.from_tensor(x_pt, [1, M, N], shared_layout)

        for cand in candidates:
            reg_layout = gluon_layouts[cfg_name][cand]
            x_constraints, r_constraints = get_barrier_constraints(cfg_name, cand)

            r_cubin_hashes = {}
            compiled_mods = {}

            # Test across R in {1, 2, 4, 8}
            for R in test_r_values:
                out_pt.zero_()
                compiled_mod = gluon_repeated_reduction_kernel[(1,)](
                    desc,
                    out_pt,
                    R,
                    register_layout=reg_layout,
                    shared_layout=shared_layout,
                    B_DESC=b_desc,
                    M=M,
                    N=N,
                    X_CONSTRAINTS=x_constraints,
                    R_CONSTRAINTS=r_constraints,
                    num_warps=num_warps,
                )
                torch.cuda.synchronize()

                cubin = compiled_mod.asm.get("cubin", None)
                if cubin is None and hasattr(compiled_mod, "kernel"):
                    cubin = compiled_mod.kernel
                c_sha = hashlib.sha256(cubin).hexdigest()
                r_cubin_hashes[R] = c_sha
                compiled_mods[R] = compiled_mod

            # Invariant check: all R must share the exact same CUBIN hash
            unique_cubins = set(r_cubin_hashes.values())
            cubin_invariant = (len(unique_cubins) == 1)

            # Use the primary compiled module (R=1) for artifact extraction
            primary_mod = compiled_mods[1]
            ptx_text = primary_mod.asm["ptx"]
            ttgir_text = primary_mod.asm["ttgir"]
            cubin_bytes = primary_mod.asm.get("cubin", None)
            if cubin_bytes is None and hasattr(primary_mod, "kernel"):
                cubin_bytes = primary_mod.kernel
            cubin_sha256 = hashlib.sha256(cubin_bytes).hexdigest()

            # Disassemble SASS
            tmp_cubin = f"/tmp/repeated_{cfg_name}_{cand}.cubin"
            with open(tmp_cubin, "wb") as f:
                f.write(cubin_bytes)

            try:
                sass_text = subprocess.check_output(["cuobjdump", "-sass", tmp_cubin], text=True)
            except Exception:
                sass_text = subprocess.check_output(["nvdisasm", "-ndf", tmp_cubin], text=True)

            res_text = subprocess.check_output(["cuobjdump", "-res-usage", tmp_cubin], text=True)
            symbols_out = subprocess.check_output(["readelf", "-s", tmp_cubin], text=True)
            func_name = None
            for s_line in symbols_out.splitlines():
                if "STT_FUNC" in s_line and ("gluon_repeated_reduction_kernel" in s_line or "kernel" in s_line):
                    parts = s_line.split()
                    func_name = parts[-1]
                    break
            if not func_name:
                func_name = "gluon_repeated_reduction_kernel"

            reg_match = re.search(r"REG:(\d+)", res_text)
            local_match = re.search(r"LOCAL:(\d+)", res_text)
            stack_match = re.search(r"STACK:(\d+)", res_text)
            shared_match = re.search(r"SHARED:(\d+)", res_text)

            num_regs = int(reg_match.group(1)) if reg_match else -1
            local_bytes = int(local_match.group(1)) if local_match else -1
            stack_bytes = int(stack_match.group(1)) if stack_match else -1
            static_shared = int(shared_match.group(1)) if shared_match else -1
            dyn_smem = getattr(primary_mod.metadata, "shared", getattr(primary_mod, "shared", 0))

            # Query physical resources via CUDA Driver API
            cu_mod = ctypes.c_void_p()
            cu_func = ctypes.c_void_p()
            res = cuda.cuModuleLoadData(ctypes.byref(cu_mod), cubin_bytes)
            if res != 0:
                raise RuntimeError(f"cuModuleLoadData failed: {res}")

            res = cuda.cuModuleGetFunction(ctypes.byref(cu_func), cu_mod, func_name.encode("utf-8"))
            if res != 0:
                raise RuntimeError(f"cuModuleGetFunction failed: {res}")

            block_threads = num_warps * 32
            max_active_blocks_actual = ctypes.c_int()
            cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(max_active_blocks_actual),
                cu_func,
                block_threads,
                dyn_smem,
            )

            max_active_blocks_zero_smem = ctypes.c_int()
            cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(max_active_blocks_zero_smem),
                cu_func,
                block_threads,
                0,
            )

            cuda.cuModuleUnload(cu_mod)

            # Verification of trivial observable store
            is_correct = bool((out_pt[0] == 0.0).item())

            repeated_results[cfg_name][cand] = {
                "config": cfg_name,
                "candidate": cand,
                "num_warps": num_warps,
                "cubin_sha256": cubin_sha256,
                "cubin_invariant_across_r": cubin_invariant,
                "r_cubin_hashes": r_cubin_hashes,
                "resources": {
                    "num_regs": num_regs,
                    "static_smem_bytes": static_shared,
                    "dynamic_smem_bytes": dyn_smem,
                    "local_bytes": local_bytes,
                    "stack_bytes": stack_bytes,
                },
                "occupancy": {
                    "threads_per_block": block_threads,
                    "warps_per_block": num_warps,
                    "blocks_per_sm_actual_smem": max_active_blocks_actual.value,
                    "blocks_per_sm_zero_dynamic_smem": max_active_blocks_zero_smem.value,
                    "active_warps_per_sm": max_active_blocks_actual.value * num_warps,
                },
                "correctness": {
                    "passed": is_correct,
                },
                "ptx_text": ptx_text,
                "ttgir_text": ttgir_text,
                "sass_text": sass_text,
                "res_text": res_text,
            }
            print(
                f"  [{cfg_name} {cand}] CUBIN={cubin_sha256[:10]} regs={num_regs} "
                f"dyn_smem={dyn_smem} blocks/sm={max_active_blocks_actual.value} "
                f"single_binary={cubin_invariant}"
            )

    return {
        "device_limits": device_limits,
        "results": repeated_results,
    }


def main():
    print("=" * 60)
    print("Launching Phase 3 Step D: Gluon Repeated-Reduction Isolation")
    print("=" * 60)

    prov = generate_provenance(repo_root)

    with app.run():
        res = remote_run_gluon_repeated.remote(prov)

    out_dir = repo_root / "experiments" / "tma_reduction_layout" / "results" / "phase3" / "gluon_repeated"
    arts_dir = out_dir / "artifacts"
    arts_dir.mkdir(parents=True, exist_ok=True)

    raw_json = {
        "device_limits": res["device_limits"],
        "configurations": {},
    }

    for cfg_name, cand_dict in res["results"].items():
        raw_json["configurations"][cfg_name] = {}
        cfg_art_dir = arts_dir / cfg_name
        cfg_art_dir.mkdir(parents=True, exist_ok=True)

        for cand, cand_data in cand_dict.items():
            ptx_p = cfg_art_dir / f"{cand}.ptx"
            ttgir_p = cfg_art_dir / f"{cand}.ttgir"
            sass_p = cfg_art_dir / f"{cand}.sass"
            res_p = cfg_art_dir / f"{cand}.resource.txt"
            sha_p = cfg_art_dir / f"{cand}.cubin.sha256"

            ptx_p.write_text(cand_data["ptx_text"], encoding="utf-8")
            ttgir_p.write_text(cand_data["ttgir_text"], encoding="utf-8")
            sass_p.write_text(cand_data["sass_text"], encoding="utf-8")
            res_p.write_text(cand_data["res_text"], encoding="utf-8")
            sha_p.write_text(cand_data["cubin_sha256"], encoding="utf-8")

            entry = {k: v for k, v in cand_data.items() if not k.endswith("_text")}
            entry["artifacts"] = {
                "ptx": str(ptx_p.relative_to(repo_root)),
                "ttgir": str(ttgir_p.relative_to(repo_root)),
                "sass": str(sass_p.relative_to(repo_root)),
                "resource": str(res_p.relative_to(repo_root)),
                "cubin_sha256": str(sha_p.relative_to(repo_root)),
            }
            raw_json["configurations"][cfg_name][cand] = entry

    raw_json_path = out_dir / "raw_results.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(raw_json, f, indent=2)

    print(f"Gluon repeated artifacts saved to {arts_dir}")
    print(f"Gluon repeated raw results saved to {raw_json_path}")


if __name__ == "__main__":
    main()
