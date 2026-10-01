#!/usr/bin/env python3
"""
Phase 3 Step C: Gluon Structural Reproduction Runner.

Executes on Modal H100 to:
1. Compile Gluon canonical reduction kernels across 4 configurations:
   - M32_N64_w8 (default, cand4)
   - M32_N128_w4 (default, cand4)
2. Capture full compilation artifacts (.ptx, .ttgir, .sass, .resource.txt, .cubin.sha256).
3. Query physical resources and occupancy via CUDA driver API.
4. Verify numerical correctness against PyTorch ref.
5. Extract Level 1 layout tensor/hardware views.
6. Save raw results to results/phase3/gluon_reproduction/raw_results.json.
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
def remote_run_gluon_reproduction(prov: dict) -> dict:
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
        gluon_canonical_reduction_kernel,
    )

    env_info = remote_verify_environment(prov)
    print(f"[Remote] Running Gluon structural reproduction on {env_info['gpu_name']}")

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
    gluon_layouts = get_canonical_gluon_layouts()
    shared_layout = get_canonical_gluon_shared_layout()

    reproduction_results = {}

    for cfg in configs:
        cfg_name = cfg["cfg_name"]
        m = cfg["M"]
        n = cfg["N"]
        num_warps = cfg["num_warps"]
        b_desc = cfg["b_desc"]
        reproduction_results[cfg_name] = {}

        input_tensor = torch.randn((b_desc, m, n), device="cuda", dtype=torch.bfloat16)
        out_tensor = torch.empty((b_desc, n), device="cuda", dtype=torch.float32)

        for cand in candidates:
            reg_layout = gluon_layouts[cfg_name][cand]
            in_desc = TensorDescriptor.from_tensor(input_tensor, [1, m, n], shared_layout)

            # Warmup compile
            print(f"[Remote] Compiling Gluon kernel for {cfg_name} {cand}...")
            compiled = gluon_canonical_reduction_kernel.warmup(
                in_desc, out_tensor,
                reg_layout, shared_layout,
                b_desc, m, n,
                grid=(1,), num_warps=num_warps
            )

            ptx_text = compiled.asm.get("ptx", "")
            ttgir_text = compiled.asm.get("ttgir", "")
            cubin_bytes = compiled.asm.get("cubin", None)
            if cubin_bytes is None and hasattr(compiled, "kernel"):
                cubin_bytes = compiled.kernel
            cubin_sha256 = hashlib.sha256(cubin_bytes).hexdigest()

            tmp_cubin = f"/tmp/gluon_{cfg_name}_{cand}.cubin"
            with open(tmp_cubin, "wb") as f:
                f.write(cubin_bytes)

            sass_text = subprocess.check_output(["cuobjdump", "-sass", tmp_cubin], text=True)
            res_text = subprocess.check_output(["cuobjdump", "-res-usage", tmp_cubin], text=True)
            symbols_out = subprocess.check_output(["cuobjdump", "-symbols", tmp_cubin], text=True)

            func_name = None
            for s_line in symbols_out.splitlines():
                if "STT_FUNC" in s_line and ("gluon_canonical_reduction_kernel" in s_line or "kernel" in s_line):
                    parts = s_line.split()
                    func_name = parts[-1]
                    break
            if not func_name:
                func_name = "gluon_canonical_reduction_kernel"

            reg_match = re.search(r"REG:(\d+)", res_text)
            local_match = re.search(r"LOCAL:(\d+)", res_text)
            stack_match = re.search(r"STACK:(\d+)", res_text)
            shared_match = re.search(r"SHARED:(\d+)", res_text)

            num_regs = int(reg_match.group(1)) if reg_match else -1
            local_bytes = int(local_match.group(1)) if local_match else -1
            stack_bytes = int(stack_match.group(1)) if stack_match else -1
            static_shared = int(shared_match.group(1)) if shared_match else -1
            dyn_smem = getattr(compiled.metadata, "shared", 0)

            # Occupancy calculation via CUDA driver API
            module = ctypes.c_void_p()
            cuda.cuModuleLoadData(ctypes.byref(module), cubin_bytes)
            func = ctypes.c_void_p()
            cuda.cuModuleGetFunction(ctypes.byref(func), module, func_name.encode("utf-8"))

            block_size = num_warps * 32
            max_active_blocks_actual = ctypes.c_int()
            cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(max_active_blocks_actual),
                func,
                ctypes.c_int(block_size),
                ctypes.c_size_t(dyn_smem)
            )
            max_active_blocks_zero_dyn = ctypes.c_int()
            cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(max_active_blocks_zero_dyn),
                func,
                ctypes.c_int(block_size),
                ctypes.c_size_t(0)
            )

            # Numerical correctness check
            sample_b = min(16, b_desc)
            out_tensor.zero_()
            gluon_canonical_reduction_kernel[(sample_b,)](
                in_desc, out_tensor,
                reg_layout, shared_layout,
                b_desc, m, n,
                num_warps=num_warps
            )
            torch.cuda.synchronize()

            ref_vals = input_tensor[:sample_b].float().max(dim=1).values
            act_vals = out_tensor[:sample_b].float()
            diff = (ref_vals - act_vals).abs()
            max_abs = diff.max().item()
            max_rel = (diff / (ref_vals.abs() + 1e-6)).max().item()
            is_correct = (max_abs <= 1e-4)

            # Level 1 Layout Format Views
            tensor_shape = [1, m, n]
            tensor_view = reg_layout.format_tensor_view(tensor_shape)
            hw_view = reg_layout.format_hardware_view(tensor_shape)

            reproduction_results[cfg_name][cand] = {
                "cubin_sha256": cubin_sha256,
                "func_name": func_name,
                "resources": {
                    "num_regs": num_regs,
                    "local_bytes": local_bytes,
                    "stack_bytes": stack_bytes,
                    "static_shared_bytes": static_shared,
                    "dynamic_smem_bytes": dyn_smem,
                },
                "occupancy": {
                    "block_size": block_size,
                    "blocks_per_sm_actual_smem": max_active_blocks_actual.value,
                    "blocks_per_sm_zero_dynamic_smem": max_active_blocks_zero_dyn.value,
                    "active_warps_per_sm": max_active_blocks_actual.value * num_warps,
                    "smem_limited": (max_active_blocks_actual.value != max_active_blocks_zero_dyn.value),
                },
                "correctness": {
                    "tolerance": 1e-4,
                    "max_abs_diff": max_abs,
                    "max_rel_diff": max_rel,
                    "passed": is_correct,
                },
                "layout_views": {
                    "tensor_view": tensor_view,
                    "hardware_view": hw_view,
                    "layout_str": str(reg_layout),
                },
                "ptx_text": ptx_text,
                "ttgir_text": ttgir_text,
                "sass_text": sass_text,
                "res_text": res_text,
            }
            print(f"  [{cfg_name} {cand}] CUBIN={cubin_sha256[:10]} regs={num_regs} dyn_smem={dyn_smem} blocks/sm={max_active_blocks_actual.value} correct={is_correct}")

    return {
        "device_limits": device_limits,
        "results": reproduction_results,
    }


def main():
    print("=" * 60)
    print("Launching Phase 3 Step C: Gluon Structural Reproduction")
    print("=" * 60)

    prov = generate_provenance(repo_root)

    with app.run():
        res = remote_run_gluon_reproduction.remote(prov)

    # Save artifacts & raw results
    out_dir = repo_root / "experiments" / "tma_reduction_layout" / "results" / "phase3" / "gluon_reproduction"
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

    print(f"Gluon artifacts saved to {arts_dir}")
    print(f"Gluon raw results saved to {raw_json_path}")


if __name__ == "__main__":
    main()
