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
    timeout=600,
)
def remote_test_v3_runtime_k(prov: dict) -> dict:
    import ctypes
    import hashlib
    import os
    import re
    import subprocess
    import torch
    import triton
    import triton.language as tl

    env_info = remote_verify_environment(prov)
    print(f"[Remote] Testing v3 single-binary runtime-K isolation")
    print(f"[Remote] GPU: {env_info['gpu_name']} UUID: {env_info['gpu_uuid']}")

    triton.set_allocator(
        lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda")
    )

    configs = [
        {"cfg_name": "M32_N64_w8", "M": 32, "N": 64, "num_warps": 8, "b_desc": 65536},
        {"cfg_name": "M32_N128_w4", "M": 32, "N": 128, "num_warps": 4, "b_desc": 65536},
    ]
    candidates = ["default", "4"]
    k_test_vals = [1, 2, 4, 8]

    # CUDA driver setup for occupancy calculation
    cuda = ctypes.CDLL("libcuda.so.1")
    res = cuda.cuInit(0)
    if res != 0:
        raise RuntimeError(f"cuInit failed with error code {res}")
    dev = ctypes.c_int()
    cuda.cuDeviceGet(ctypes.byref(dev), 0)
    ctx = ctypes.c_void_p()
    cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
    cuda.cuCtxSetCurrent(ctx)

    results = {}

    for cfg in configs:
        cfg_name = cfg["cfg_name"]
        m = cfg["M"]
        n = cfg["N"]
        num_warps = cfg["num_warps"]
        b_desc = cfg["b_desc"]
        stride_b = m * n
        stride_m = n

        results[cfg_name] = {}

        input_tensor = torch.randn((b_desc, m, n), device="cuda", dtype=torch.bfloat16)
        out_tensor = torch.empty((b_desc, n), device="cuda", dtype=torch.float32)

        for cand in candidates:
            results[cfg_name][cand] = {}
            if cand == "default":
                os.environ.pop("TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT", None)
            else:
                os.environ["TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT"] = cand

            os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_v3_{cfg_name}_{cand}"

            # Define kernel with runtime scalar K_ITERS
            @triton.jit(do_not_specialize=["k_iters"])
            def reduction_kernel_v3(
                a_ptr, out_ptr, stride_b, stride_m,
                k_iters: int, # runtime scalar trip count!
                B_DESC: tl.constexpr, M: tl.constexpr, N: tl.constexpr
            ):
                pid = tl.program_id(0)
                desc = tl.make_tensor_descriptor(
                    a_ptr, shape=[B_DESC, M, N], strides=[stride_b, stride_m, 1], block_shape=[1, M, N]
                )
                x = desc.load([pid, 0, 0])
                acc = tl.zeros([1, N], dtype=tl.float32)
                for i in tl.range(0, k_iters, loop_unroll_factor=1, disable_licm=True):
                    x_opaque = tl.inline_asm_elementwise("mov.b32 $0, $1;", "=r,r", [x], dtype=tl.bfloat16, is_pure=False, pack=2)
                    r_i = tl.max(x_opaque.to(tl.float32), axis=1)
                    acc += r_i
                offs_n = tl.arange(0, N)
                tl.store(out_ptr + pid * N + offs_n, tl.reshape(acc, [N]))

            # 1. Compile once via warmup with k_iters=1
            compiled = reduction_kernel_v3.warmup(
                input_tensor, out_tensor, stride_b, stride_m, 1,
                b_desc, m, n, grid=(1,), num_warps=num_warps
            )

            ptx_text = compiled.asm.get("ptx", "")
            ttgir_text = compiled.asm.get("ttgir", "")
            cubin_bytes = compiled.asm.get("cubin", None)
            if cubin_bytes is None and hasattr(compiled, "kernel"):
                cubin_bytes = compiled.kernel
            cubin_sha256 = hashlib.sha256(cubin_bytes).hexdigest()

            # 2. Extract SASS and resource usage via cuobjdump
            tmp_cubin = f"/tmp/triton_v3_{cfg_name}_{cand}.cubin"
            with open(tmp_cubin, "wb") as f:
                f.write(cubin_bytes)

            sass_text = subprocess.check_output(["cuobjdump", "-sass", tmp_cubin], text=True)
            res_text = subprocess.check_output(["cuobjdump", "-res-usage", tmp_cubin], text=True)
            sym_text = subprocess.check_output(["cuobjdump", "-symbols", tmp_cubin], text=True)

            # Find function symbol in CUBIN
            func_name = None
            for line in sym_text.splitlines():
                parts = line.split()
                if len(parts) >= 4 and parts[1] == "g" and parts[2] == "F":
                    func_name = parts[-1]
                    break
            if func_name is None:
                for line in sass_text.splitlines():
                    if "Function :" in line:
                        func_name = line.split("Function :")[1].strip()
                        break

            # Parse resource usage
            num_regs = 0
            local_bytes = 0
            stack_bytes = 0
            static_shared_bytes = 0
            for line in res_text.splitlines():
                if "REG:" in line:
                    m_reg = re.search(r"REG:(\d+)", line)
                    m_loc = re.search(r"LOCAL:(\d+)", line)
                    m_stk = re.search(r"STACK:(\d+)", line)
                    m_shd = re.search(r"SHARED:(\d+)", line)
                    if m_reg: num_regs = int(m_reg.group(1))
                    if m_loc: local_bytes = int(m_loc.group(1))
                    if m_stk: stack_bytes = int(m_stk.group(1))
                    if m_shd: static_shared_bytes = int(m_shd.group(1))

            dyn_smem_bytes = compiled.metadata.shared

            # 3. Official CUDA Occupancy API calculation
            module = ctypes.c_void_p()
            r_mod = cuda.cuModuleLoadData(ctypes.byref(module), cubin_bytes)
            if r_mod != 0:
                raise RuntimeError(f"cuModuleLoadData failed: {r_mod}")

            c_func = ctypes.c_void_p()
            r_fn = cuda.cuModuleGetFunction(ctypes.byref(c_func), module, func_name.encode("utf-8"))
            if r_fn != 0:
                raise RuntimeError(f"cuModuleGetFunction for '{func_name}' failed: {r_fn}")

            block_size = num_warps * 32
            num_blocks = ctypes.c_int()
            r_occ = cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                ctypes.byref(num_blocks),
                c_func,
                ctypes.c_int(block_size),
                ctypes.c_size_t(dyn_smem_bytes)
            )
            if r_occ != 0:
                raise RuntimeError(f"cuOccupancyMaxActiveBlocksPerMultiprocessor failed: {r_occ}")

            max_active_blocks_per_sm = num_blocks.value
            active_warps_per_sm = max_active_blocks_per_sm * num_warps

            # 4. Verify runtime K does NOT trigger recompilation
            device_id = torch.cuda.current_device()
            kernel_cache = reduction_kernel_v3.device_caches[device_id][0]
            initial_cache_len = len(kernel_cache)
            for k_val in k_test_vals:
                reduction_kernel_v3[(1,)](
                    input_tensor, out_tensor, stride_b, stride_m, k_val,
                    b_desc, m, n, num_warps=num_warps
                )
            final_cache_len = len(kernel_cache)
            runtime_k_specialized = (final_cache_len != initial_cache_len)

            # 5. Numerical correctness check across K in {1, 2, 4, 8}
            # Reference: K * max(x, axis=1)
            # In input_tensor, shape is [B_DESC, M, N]. Axis 1 is M. Max along dim=1 yields [B_DESC, N].
            correctness_results = {}
            tolerance = 1e-4
            all_correct = True
            
            sample_b = min(16, b_desc)
            for k_val in k_test_vals:
                out_tensor.zero_()
                reduction_kernel_v3[(sample_b,)](
                    input_tensor, out_tensor, stride_b, stride_m, k_val,
                    b_desc, m, n, num_warps=num_warps
                )
                torch.cuda.synchronize()

                # Slice sample
                act_sample = out_tensor[:sample_b].float()
                # Compute reference along M dimension (dim 1)
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

            # 6. Runtime loop detection
            ttgir_has_scf_for = ("scf.for" in ttgir_text)
            ptx_has_loop_branch = bool(re.search(r"(\$L__BB\d+_\d+):.*?bra(?:\.uni)?\s+\1", ptx_text, re.DOTALL))
            sass_has_bra = ("BRA " in sass_text)
            runtime_loop_detected = ttgir_has_scf_for and ptx_has_loop_branch and sass_has_bra

            results[cfg_name][cand] = {
                "cubin_sha256": cubin_sha256,
                "func_name": func_name,
                "ptx_text": ptx_text,
                "ttgir_text": ttgir_text,
                "sass_text": sass_text,
                "resource_text": res_text,
                "resources": {
                    "num_regs": num_regs,
                    "local_bytes": local_bytes,
                    "stack_bytes": stack_bytes,
                    "static_shared_bytes": static_shared_bytes,
                    "dynamic_smem_bytes": dyn_smem_bytes,
                },
                "occupancy": {
                    "block_size": block_size,
                    "max_active_blocks_per_sm": max_active_blocks_per_sm,
                    "active_warps_per_sm": active_warps_per_sm,
                },
                "specialization_check": {
                    "initial_cache_len": initial_cache_len,
                    "final_cache_len": final_cache_len,
                    "runtime_k_specialized": runtime_k_specialized,
                },
                "runtime_loop": {
                    "runtime_loop_detected": runtime_loop_detected,
                    "ttgir_has_scf_for": ttgir_has_scf_for,
                    "ptx_has_loop_branch": ptx_has_loop_branch,
                    "sass_has_bra": sass_has_bra,
                    "loop_body_copy_count": 1,
                },
                "correctness": {
                    "tolerance": tolerance,
                    "all_passed": all_correct,
                    "per_k": correctness_results,
                }
            }

            print(f"  [{cfg_name} {cand}] CUBIN={cubin_sha256[:10]} regs={num_regs} local={local_bytes} stack={stack_bytes} dyn_smem={dyn_smem_bytes} occ_blocks={max_active_blocks_per_sm} occ_warps={active_warps_per_sm} loop={runtime_loop_detected} spec={runtime_k_specialized} correct={all_correct}")

    return results

def main():
    prov = generate_provenance()
    print("\n=======================================================")
    print("Launching Phase 3 Step B v3 Single-Binary Evaluation")
    print("=======================================================")
    
    with app.run():
        res = remote_test_v3_runtime_k.remote(prov)

    out_base = repo_root / "experiments" / "tma_reduction_layout" / "results" / "phase3" / "v3_runtime_k"
    arts_dir = out_base / "artifacts"
    arts_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts and summary JSON
    summary_results = {}
    for cfg_name, cfg_data in res.items():
        summary_results[cfg_name] = {}
        cfg_art_dir = arts_dir / cfg_name
        cfg_art_dir.mkdir(parents=True, exist_ok=True)

        for cand, cdata in cfg_data.items():
            # Save files
            (cfg_art_dir / f"{cand}.ptx").write_text(cdata["ptx_text"])
            (cfg_art_dir / f"{cand}.ttgir").write_text(cdata["ttgir_text"])
            (cfg_art_dir / f"{cand}.sass").write_text(cdata["sass_text"])
            (cfg_art_dir / f"{cand}.resource.txt").write_text(cdata["resource_text"])
            (cfg_art_dir / f"{cand}.cubin.sha256").write_text(cdata["cubin_sha256"] + "\n")

            summary_results[cfg_name][cand] = {
                "cubin_sha256": cdata["cubin_sha256"],
                "func_name": cdata["func_name"],
                "resources": cdata["resources"],
                "occupancy": cdata["occupancy"],
                "specialization_check": cdata["specialization_check"],
                "runtime_loop": cdata["runtime_loop"],
                "correctness": cdata["correctness"],
                "artifacts": {
                    "ptx": str((cfg_art_dir / f"{cand}.ptx").relative_to(repo_root)),
                    "ttgir": str((cfg_art_dir / f"{cand}.ttgir").relative_to(repo_root)),
                    "sass": str((cfg_art_dir / f"{cand}.sass").relative_to(repo_root)),
                    "resource": str((cfg_art_dir / f"{cand}.resource.txt").relative_to(repo_root)),
                    "cubin_sha256": str((cfg_art_dir / f"{cand}.cubin.sha256").relative_to(repo_root)),
                }
            }

    with open(out_base / "raw_results.json", "w") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nArtifacts saved to {arts_dir}")
    print(f"Raw results saved to {out_base / 'raw_results.json'}")

if __name__ == "__main__":
    main()
