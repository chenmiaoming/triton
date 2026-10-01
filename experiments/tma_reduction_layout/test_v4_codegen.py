#!/usr/bin/env python3
"""
Phase 3 Step B v4: Preloaded-Register Runtime-Loop Isolation Feasibility & Canonical Occupancy Baseline

Performs:
1. Dynamic CUDA Device Limit Query (H100 SM90 attributes via driver API).
2. Part B: Exact Canonical Fixed-Binary Specialization compilation & occupancy measurement.
   - cuOccupancyMaxActiveBlocksPerMultiprocessor with actual dynamic smem
   - cuOccupancyMaxActiveBlocksPerMultiprocessor with zero dynamic smem (limiter disambiguation)
3. Part C: Phase 3 Step B v4 Preloaded-Register Kernel compilation & evaluation across K in {1, 2, 4, 8}.
   - Pre-loop shared->register materialization via inline asm mov.b32
   - In-loop anti-CSE opaque barrier via inline asm mov.b32
   - Single CUBIN reuse verification across K
   - Numerical correctness verification across K
   - Occupancy measurement (actual vs zero dynamic smem)
4. Saves artifacts and raw JSON results.
"""

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

repo_root = Path(__file__).resolve().parent.parent.parent
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
    timeout=900,
)
def remote_test_canonical_and_v4(prov: dict) -> dict:
    import ctypes
    import hashlib
    import os
    import re
    import subprocess
    import torch
    import triton
    import triton.language as tl

    env_info = remote_verify_environment(prov)
    print(f"[Remote] Testing canonical occupancy baseline and v4 preloaded-register isolation")
    print(f"[Remote] GPU: {env_info['gpu_name']} UUID: {env_info['gpu_uuid']}")

    triton.set_allocator(
        lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda")
    )

    # 1. CUDA Driver & Device Query
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

    print(f"[Remote] Device Limits: {device_limits}")

    configs = [
        {"cfg_name": "M32_N64_w8", "M": 32, "N": 64, "num_warps": 8, "b_desc": 65536},
        {"cfg_name": "M32_N128_w4", "M": 32, "N": 128, "num_warps": 4, "b_desc": 65536},
    ]
    candidates = ["default", "4"]
    k_test_vals = [1, 2, 4, 8]

    canonical_results = {}
    v4_results = {}

    # =========================================================================
    # PART B: Canonical Kernels Baseline
    # =========================================================================
    print("\n--- Compiling & Evaluating Canonical Step A Kernels ---")
    for cfg in configs:
        cfg_name = cfg["cfg_name"]
        m = cfg["M"]
        n = cfg["N"]
        num_warps = cfg["num_warps"]
        b_desc = cfg["b_desc"]

        canonical_results[cfg_name] = {}

        input_tensor = torch.randn((b_desc, m, n), device="cuda", dtype=torch.bfloat16)
        out_tensor = torch.empty((b_desc, n), device="cuda", dtype=torch.float32)
        stride_b, stride_m, _ = input_tensor.stride()

        for cand in candidates:
            if cand == "default":
                os.environ.pop("TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT", None)
            else:
                os.environ["TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT"] = cand

            os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_canonical_{cfg_name}_{cand}"

            @triton.jit
            def canonical_kernel(
                a_ptr, out_ptr, stride_b, stride_m,
                B_DESC: tl.constexpr, M: tl.constexpr, N: tl.constexpr
            ):
                pid = tl.program_id(0)
                desc = tl.make_tensor_descriptor(
                    a_ptr, shape=[B_DESC, M, N], strides=[stride_b, stride_m, 1], block_shape=[1, M, N]
                )
                x = desc.load([pid, 0, 0])
                x_fp32 = x.to(tl.float32)
                y = tl.max(x_fp32, axis=1)
                offs_n = tl.arange(0, N)
                tl.store(out_ptr + pid * N + offs_n, tl.reshape(y, [N]))

            compiled = canonical_kernel.warmup(
                input_tensor, out_tensor, stride_b, stride_m,
                b_desc, m, n, grid=(1,), num_warps=num_warps
            )

            ptx_text = compiled.asm.get("ptx", "")
            ttgir_text = compiled.asm.get("ttgir", "")
            cubin_bytes = compiled.asm.get("cubin", None)
            if cubin_bytes is None and hasattr(compiled, "kernel"):
                cubin_bytes = compiled.kernel
            cubin_sha256 = hashlib.sha256(cubin_bytes).hexdigest()

            tmp_cubin = f"/tmp/canon_{cfg_name}_{cand}.cubin"
            with open(tmp_cubin, "wb") as f:
                f.write(cubin_bytes)
            sass_text = subprocess.check_output(["cuobjdump", "-sass", tmp_cubin], text=True)
            res_text = subprocess.check_output(["cuobjdump", "-res-usage", tmp_cubin], text=True)
            symbols_out = subprocess.check_output(["cuobjdump", "-symbols", tmp_cubin], text=True)

            func_name = None
            for s_line in symbols_out.splitlines():
                if "STT_FUNC" in s_line and ("canonical_kernel" in s_line or "kernel" in s_line):
                    parts = s_line.split()
                    func_name = parts[-1]
                    break
            if not func_name:
                func_name = "canonical_kernel"

            reg_match = re.search(r"REG:(\d+)", res_text)
            local_match = re.search(r"LOCAL:(\d+)", res_text)
            stack_match = re.search(r"STACK:(\d+)", res_text)
            shared_match = re.search(r"SHARED:(\d+)", res_text)

            num_regs = int(reg_match.group(1)) if reg_match else -1
            local_bytes = int(local_match.group(1)) if local_match else -1
            stack_bytes = int(stack_match.group(1)) if stack_match else -1
            static_shared = int(shared_match.group(1)) if shared_match else -1
            dyn_smem = getattr(compiled.metadata, "shared", 0)

            # Occupancy measurement
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

            canonical_results[cfg_name][cand] = {
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
                "ptx_text": ptx_text,
                "ttgir_text": ttgir_text,
                "sass_text": sass_text,
                "res_text": res_text,
            }
            print(f"  [Canonical {cfg_name} {cand}] regs={num_regs} dyn_smem={dyn_smem} blocks/sm={max_active_blocks_actual.value} (zero_dyn={max_active_blocks_zero_dyn.value}) warps/sm={max_active_blocks_actual.value * num_warps}")

    # =========================================================================
    # PART C: Phase 3 Step B v4 Preloaded-Register Isolation
    # =========================================================================
    print("\n--- Compiling & Evaluating Phase 3 Step B v4 Preloaded Kernels ---")
    for cfg in configs:
        cfg_name = cfg["cfg_name"]
        m = cfg["M"]
        n = cfg["N"]
        num_warps = cfg["num_warps"]
        b_desc = cfg["b_desc"]

        v4_results[cfg_name] = {}

        input_tensor = torch.randn((b_desc, m, n), device="cuda", dtype=torch.bfloat16)
        out_tensor = torch.empty((b_desc, n), device="cuda", dtype=torch.float32)
        stride_b, stride_m, _ = input_tensor.stride()

        for cand in candidates:
            if cand == "default":
                os.environ.pop("TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT", None)
            else:
                os.environ["TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT"] = cand

            os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_v4_{cfg_name}_{cand}"

            # Define v4 preloaded kernel
            @triton.jit(do_not_specialize=["k_iters"])
            def reduction_kernel_v4(
                a_ptr, out_ptr, stride_b, stride_m,
                k_iters: int,
                B_DESC: tl.constexpr, M: tl.constexpr, N: tl.constexpr
            ):
                pid = tl.program_id(0)
                desc = tl.make_tensor_descriptor(
                    a_ptr, shape=[B_DESC, M, N], strides=[stride_b, stride_m, 1], block_shape=[1, M, N]
                )
                x = desc.load([pid, 0, 0])
                # Materialize shared->register BEFORE loop
                x_preloaded = tl.inline_asm_elementwise(
                    "mov.b32 $0, $1;",
                    "=r,r",
                    [x],
                    dtype=tl.bfloat16,
                    is_pure=False,
                    pack=2,
                )
                acc = tl.zeros([1, N], dtype=tl.float32)
                for i in tl.range(0, k_iters, loop_unroll_factor=1, disable_licm=True):
                    # In-loop anti-CSE barrier
                    x_iter = tl.inline_asm_elementwise(
                        "mov.b32 $0, $1;",
                        "=r,r",
                        [x_preloaded],
                        dtype=tl.bfloat16,
                        is_pure=False,
                        pack=2,
                    )
                    r_i = tl.max(x_iter.to(tl.float32), axis=1)
                    acc += r_i
                offs_n = tl.arange(0, N)
                tl.store(out_ptr + pid * N + offs_n, tl.reshape(acc, [N]))

            # Compile once via warmup with k_iters=1
            compiled = reduction_kernel_v4.warmup(
                input_tensor, out_tensor, stride_b, stride_m, 1,
                b_desc, m, n, grid=(1,), num_warps=num_warps
            )

            ptx_text = compiled.asm.get("ptx", "")
            ttgir_text = compiled.asm.get("ttgir", "")
            cubin_bytes = compiled.asm.get("cubin", None)
            if cubin_bytes is None and hasattr(compiled, "kernel"):
                cubin_bytes = compiled.kernel
            cubin_sha256 = hashlib.sha256(cubin_bytes).hexdigest()

            tmp_cubin = f"/tmp/triton_v4_{cfg_name}_{cand}.cubin"
            with open(tmp_cubin, "wb") as f:
                f.write(cubin_bytes)

            sass_text = subprocess.check_output(["cuobjdump", "-sass", tmp_cubin], text=True)
            res_text = subprocess.check_output(["cuobjdump", "-res-usage", tmp_cubin], text=True)
            symbols_out = subprocess.check_output(["cuobjdump", "-symbols", tmp_cubin], text=True)

            func_name = None
            for s_line in symbols_out.splitlines():
                if "STT_FUNC" in s_line and ("reduction_kernel_v4" in s_line or "kernel" in s_line):
                    parts = s_line.split()
                    func_name = parts[-1]
                    break
            if not func_name:
                func_name = "reduction_kernel_v4"

            reg_match = re.search(r"REG:(\d+)", res_text)
            local_match = re.search(r"LOCAL:(\d+)", res_text)
            stack_match = re.search(r"STACK:(\d+)", res_text)
            shared_match = re.search(r"SHARED:(\d+)", res_text)

            num_regs = int(reg_match.group(1)) if reg_match else -1
            local_bytes = int(local_match.group(1)) if local_match else -1
            stack_bytes = int(stack_match.group(1)) if stack_match else -1
            static_shared = int(shared_match.group(1)) if shared_match else -1
            dyn_smem = getattr(compiled.metadata, "shared", 0)

            # Occupancy calculation
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

            # Check specialization constancy across K
            device_id = torch.cuda.current_device()
            kernel_cache = reduction_kernel_v4.device_caches[device_id][0]
            initial_cache_len = len(kernel_cache)

            for k_val in k_test_vals:
                reduction_kernel_v4[(1,)](
                    input_tensor, out_tensor, stride_b, stride_m, k_val,
                    b_desc, m, n, num_warps=num_warps
                )
            final_cache_len = len(kernel_cache)
            runtime_k_specialized = (final_cache_len != initial_cache_len)

            # Numerical correctness check
            correctness_results = {}
            tolerance = 1e-4
            all_correct = True
            sample_b = min(16, b_desc)

            for k_val in k_test_vals:
                out_tensor.zero_()
                reduction_kernel_v4[(sample_b,)](
                    input_tensor, out_tensor, stride_b, stride_m, k_val,
                    b_desc, m, n, num_warps=num_warps
                )
                torch.cuda.synchronize()

                act_sample = out_tensor[:sample_b].float()
                ref_tile_max = input_tensor[:sample_b].float().max(dim=1).values
                exp_sample = ref_tile_max * float(k_val)

                diff = (act_sample - exp_sample).abs()
                max_abs = diff.max().item()
                max_rel = (diff / (exp_sample.abs() + 1e-6)).max().item()
                passed = (max_abs <= tolerance)
                if not passed:
                    all_correct = False

                correctness_results[k_val] = {
                    "max_abs_diff": max_abs,
                    "max_rel_diff": max_rel,
                    "passed": passed,
                }

            # Mechanical loop detection
            scf_for_count = len(re.findall(r"\bscf\.for\b", ttgir_text))
            backward_branch_count = len(re.findall(r"(@%p\d+)?\s*bra(?:\.uni)?\s+(\$L__BB\d+_\d+);", ptx_text))
            ptx_has_loop_branch = bool(re.search(r"(\$L__BB\d+_\d+):.*?bra(?:\.uni)?\s+\1", ptx_text, re.DOTALL))
            sass_has_bra = ("BRA " in sass_text)
            runtime_loop_detected = (scf_for_count == 1) and ptx_has_loop_branch and sass_has_bra

            v4_results[cfg_name][cand] = {
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
                "specialization_check": {
                    "initial_cache_len": initial_cache_len,
                    "final_cache_len": final_cache_len,
                    "runtime_k_specialized": runtime_k_specialized,
                },
                "runtime_loop": {
                    "runtime_loop_detected": runtime_loop_detected,
                    "scf_for_count": scf_for_count,
                    "ptx_has_loop_branch": ptx_has_loop_branch,
                    "sass_has_bra": sass_has_bra,
                    "loop_body_copy_count": scf_for_count,
                },
                "correctness": {
                    "tolerance": tolerance,
                    "all_passed": all_correct,
                    "per_k": correctness_results,
                },
                "ptx_text": ptx_text,
                "ttgir_text": ttgir_text,
                "sass_text": sass_text,
                "res_text": res_text,
            }
            print(f"  [v4 {cfg_name} {cand}] CUBIN={cubin_sha256[:10]} regs={num_regs} dyn_smem={dyn_smem} blocks/sm={max_active_blocks_actual.value} (zero_dyn={max_active_blocks_zero_dyn.value}) loop={runtime_loop_detected} spec={runtime_k_specialized} correct={all_correct}")

    return {
        "device_limits": device_limits,
        "canonical_results": canonical_results,
        "v4_results": v4_results,
    }


def main():
    print("=" * 60)
    print("Launching Canonical Occupancy Baseline & Phase 3 Step B v4")
    print("=" * 60)

    prov = generate_provenance(repo_root)

    with app.run():
        res = remote_test_canonical_and_v4.remote(prov)

    # 1. Save Canonical Baseline
    canon_dir = repo_root / "experiments" / "tma_reduction_layout" / "results" / "phase3" / "canonical_occupancy"
    canon_dir.mkdir(parents=True, exist_ok=True)

    canon_json_path = canon_dir / "canonical_occupancy.json"
    with open(canon_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "device_limits": res["device_limits"],
            "canonical_results": {
                cfg: {
                    cand: {k: v for k, v in cand_data.items() if not k.endswith("_text")}
                    for cand, cand_data in cfg_data.items()
                }
                for cfg, cfg_data in res["canonical_results"].items()
            }
        }, f, indent=2)

    # 2. Save v4 Artifacts & Raw Results
    v4_dir = repo_root / "experiments" / "tma_reduction_layout" / "results" / "phase3" / "v4_preloaded_k"
    v4_arts_dir = v4_dir / "artifacts"
    v4_arts_dir.mkdir(parents=True, exist_ok=True)

    v4_raw_json = {}
    for cfg_name, cand_dict in res["v4_results"].items():
        v4_raw_json[cfg_name] = {}
        cfg_art_dir = v4_arts_dir / cfg_name
        cfg_art_dir.mkdir(parents=True, exist_ok=True)

        for cand, cand_data in cand_dict.items():
            # Write artifacts
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

            # Record in raw_results
            entry = {k: v for k, v in cand_data.items() if not k.endswith("_text")}
            entry["artifacts"] = {
                "ptx": str(ptx_p.relative_to(repo_root)),
                "ttgir": str(ttgir_p.relative_to(repo_root)),
                "sass": str(sass_p.relative_to(repo_root)),
                "resource": str(res_p.relative_to(repo_root)),
                "cubin_sha256": str(sha_p.relative_to(repo_root)),
            }
            v4_raw_json[cfg_name][cand] = entry

    raw_json_path = v4_dir / "raw_results.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(v4_raw_json, f, indent=2)

    print(f"Canonical occupancy saved to {canon_json_path}")
    print(f"v4 artifacts saved to {v4_arts_dir}")
    print(f"v4 raw results saved to {raw_json_path}")


if __name__ == "__main__":
    main()
