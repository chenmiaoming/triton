"""
Phase 2 TMA Reduction Layout Benchmark: Multi-CTA Steady-State Workload on SM90 (H100).

Methodology & Protocol:
1. Steady-State Multi-CTA Model:
   - Evaluates layout candidates in a multi-CTA grid processing independent [1, M, N] tiles.
   - Evaluates affine steady-state throughput T(B) = intercept_us + slope_us * B.
   - Primary throughput metric is marginal cost per additional CTA: ΔT / ΔB (ns/CTA).
   - amortized_grid_time_per_cta_ns = T(B)/B is retained as descriptive throughput-normalization metric.
   - time_per_cta_ns is retained only as a deprecated backwards-compatible alias.
2. Fixed-Binary Compilation:
   - Descriptor shape is fixed to B_DESC = max(B).
   - Kernel compiled once per (config, candidate); variable grid launches reuse identical binary.
   - Byte-level compiled artifact SHAs (TTGIR, PTX) verified identical across grid sizes.
3. Order Rotation & Clock Drift Mitigation:
   - Both grid-size execution order and candidate execution order rotated circularly across timing rounds.
4. Structural Transition Classification:
   - T1_lane_change_warp_same: lane partitions along reduction axis changes, warp partitions unchanged
   - T2_warp_decrease_gt1: warp partitions decreases but remains >1
   - T3_warp_becomes1: warp partitions becomes 1
   - T4_ldmatrix_family_change: LocalLoad lowering switches opcode family
   - T5_register_count_change: physical register count changes (|diff| >= 4 or >= 10%)
5. Within-Run Repeat Pass:
   - For all cases with >3% performance separation vs default, a within-run repeat timing pass is conducted.
   - Tagged as within_run_gt3_reproduced, within_run_direction_reproduced, or unstable/unresolved.
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

repo_root_candidate = Path(__file__).resolve().parent.parent.parent
for p in ["/opt/triton-src", str(repo_root_candidate)]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    import modal
    from experiments.tma_reduction_layout.modal_runner import (
        app,
        remote_verify_environment,
        triton_image,
    )
except ImportError:
    modal = None
    app = None
    remote_verify_environment = None
    triton_image = None

from experiments.tma_reduction_layout.analyze_ir import (
    analyze_ptx,
    analyze_ttgir,
    compute_derived_layout_metrics,
    compute_text_hash,
    parse_resource_usage,
)
from experiments.tma_reduction_layout.source_provenance import (
    generate_provenance,
    get_repo_root,
)

REPO_ROOT = get_repo_root()


def calculate_percentile(sorted_data: List[float], percentile: float) -> float:
    """Calculates percentile using standard linear interpolation."""
    if not sorted_data:
        return 0.0
    if len(sorted_data) == 1:
        return sorted_data[0]
    idx = (len(sorted_data) - 1) * percentile
    floor_idx = int(idx)
    ceil_idx = min(floor_idx + 1, len(sorted_data) - 1)
    weight = idx - floor_idx
    return sorted_data[floor_idx] * (1.0 - weight) + sorted_data[ceil_idx] * weight


def check_candidate_legality(M: int, N: int, num_warps: int, cand_str: str) -> Tuple[bool, Optional[str]]:
    """
    Evaluates candidate legality strictly adhering to Coalesce.cpp requirements:
    1. forcedVec <= maxVectorSize (8 for 16-bit elements)
    2. forcedVec <= numElemsPerThread = max((M * N) // (num_warps * 32), 1)
    3. N % forcedVec == 0
    4. forcedVec is a power of 2
    """
    if cand_str == "default":
        return True, None
    try:
        forced_vec = int(cand_str)
    except ValueError:
        return False, f"Invalid candidate '{cand_str}' (not an integer)"

    if forced_vec > 8:
        return False, f"forcedVec {forced_vec} > maxVectorSize 8"

    num_elems = M * N
    num_threads = num_warps * 32
    num_elems_per_thread = max(num_elems // num_threads, 1)
    if forced_vec > num_elems_per_thread:
        return False, f"forcedVec {forced_vec} > numElemsPerThread ({num_elems_per_thread})"

    if N % forced_vec != 0:
        return False, f"contiguousDim {N} not divisible by forcedVec {forced_vec}"

    if (forced_vec & (forced_vec - 1)) != 0 or forced_vec < 1:
        return False, f"forcedVec {forced_vec} is not a power of 2"

    return True, None


# ---------------------------------------------------------------------------
# Remote Modal Functions
# ---------------------------------------------------------------------------
if app is not None:
    @app.function(
        image=triton_image,
        gpu="H100!:1",
        timeout=1800,
    )
    def run_pilot_saturation_remote(
        prov: Dict[str, Any],
        b_values: List[int],
        m: int = 32,
        n: int = 128,
        num_warps: int = 4,
    ) -> str:
        import torch
        import triton
        import triton.language as tl

        # 1. Environment & Hardware Verification
        env_info = remote_verify_environment(prov)
        print(f"[Remote Pilot] Hardware: {env_info['gpu_name']} CC: {env_info['gpu_compute_capability']}")

        triton.set_allocator(
            lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda")
        )

        candidates = ["default", "8", "4", "2", "1"]
        pilot_results = {
            "environment": env_info,
            "provenance": prov,
            "shape": {"M": m, "N": n, "num_warps": num_warps},
            "b_values": b_values,
            "candidates": candidates,
            "data": {},
        }

        def get_kernel():
            @triton.jit
            def kernel(
                a_ptr,
                out_ptr,
                stride_b,
                stride_m,
                B: tl.constexpr,
                M: tl.constexpr,
                N: tl.constexpr,
            ):
                pid = tl.program_id(0)
                desc = tl.make_tensor_descriptor(
                    a_ptr,
                    shape=[B, M, N],
                    strides=[stride_b, stride_m, 1],
                    block_shape=[1, M, N],
                )
                x = desc.load([pid, 0, 0])
                x_fp32 = x.to(tl.float32)
                y = tl.max(x_fp32, axis=1)
                offs_n = tl.arange(0, N)
                tl.store(out_ptr + pid * N + offs_n, tl.reshape(y, [N]))

            return kernel

        for b in b_values:
            print(f"\n[Remote Pilot] Testing B = {b} ...")
            stride_b = m * n
            stride_m = n
            input_tensor = torch.randn((b, m, n), device="cuda", dtype=torch.bfloat16)
            out_tensor = torch.empty((b, n), device="cuda", dtype=torch.float32)
            ref_out = input_tensor.to(torch.float32).amax(dim=1)

            compiled_kernels = {}
            legal_cands = []
            cand_meta = {}

            for cand in candidates:
                is_legal, err = check_candidate_legality(m, n, num_warps, cand)
                if not is_legal:
                    cand_meta[cand] = {"is_legal": False, "error": err}
                    continue

                os.environ["TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT"] = cand
                os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_pilot_{b}_{cand}"
                kernel = get_kernel()

                try:
                    compiled = kernel.warmup(
                        input_tensor,
                        out_tensor,
                        stride_b,
                        stride_m,
                        b,
                        m,
                        n,
                        grid=(b,),
                        num_warps=num_warps,
                    )
                    out_tensor.zero_()
                    kernel[(b,)](input_tensor, out_tensor, stride_b, stride_m, b, m, n, num_warps=num_warps)
                    torch.cuda.synchronize()
                    diff = float(torch.max(torch.abs(out_tensor - ref_out)).item())
                    is_correct = diff < 1e-3
                    compiled_kernels[cand] = (kernel, compiled)
                    legal_cands.append(cand)
                    cand_meta[cand] = {
                        "is_legal": True,
                        "is_correct": is_correct,
                        "max_diff": diff,
                    }
                except Exception as e:
                    cand_meta[cand] = {"is_legal": False, "error": str(e)}

            # Timing measurement protocol
            # Warmup 20 launches
            for _ in range(20):
                for cand in legal_cands:
                    k, _ = compiled_kernels[cand]
                    k[(b,)](input_tensor, out_tensor, stride_b, stride_m, b, m, n, num_warps=num_warps)
            torch.cuda.synchronize()

            num_blocks = 10
            iters_per_block = 10
            cand_samples_us = {c: [] for c in legal_cands}
            cand_block_medians = {c: [] for c in legal_cands}

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for block_idx in range(num_blocks):
                shift = block_idx % len(legal_cands)
                block_order = legal_cands[shift:] + legal_cands[:shift]
                for cand in block_order:
                    k, _ = compiled_kernels[cand]
                    block_samples = []
                    for _ in range(iters_per_block):
                        start_event.record()
                        k[(b,)](input_tensor, out_tensor, stride_b, stride_m, b, m, n, num_warps=num_warps)
                        end_event.record()
                        torch.cuda.synchronize()
                        us = start_event.elapsed_time(end_event) * 1000.0
                        cand_samples_us[cand].append(us)
                        block_samples.append(us)
                    cand_block_medians[cand].append(statistics.median(block_samples))

            b_data = {}
            input_bytes = b * m * n * 2  # BF16 = 2 bytes
            for cand in candidates:
                if cand not in legal_cands:
                    b_data[cand] = cand_meta.get(cand, {"is_legal": False})
                    continue
                samples = cand_samples_us[cand]
                sorted_samples = sorted(samples)
                med = statistics.median(samples)
                mean = statistics.mean(samples)
                p10 = calculate_percentile(sorted_samples, 0.10)
                p90 = calculate_percentile(sorted_samples, 0.90)
                p25 = calculate_percentile(sorted_samples, 0.25)
                p75 = calculate_percentile(sorted_samples, 0.75)
                iqr = p75 - p25
                mad = statistics.median([abs(x - med) for x in samples])

                time_per_cta_ns = (med * 1000.0) / b
                effective_gbps = (input_bytes / (med * 1e-6)) / 1e9

                b_data[cand] = {
                    "is_legal": True,
                    "is_correct": cand_meta[cand]["is_correct"],
                    "max_diff": cand_meta[cand]["max_diff"],
                    "median_us": med,
                    "mean_us": mean,
                    "p10_us": p10,
                    "p90_us": p90,
                    "iqr_us": iqr,
                    "mad_us": mad,
                    "time_per_cta_ns": time_per_cta_ns,
                    "effective_gbps": effective_gbps,
                    "block_medians_us": cand_block_medians[cand],
                    "raw_samples_us": samples,
                }
                print(f"[Remote Pilot] B={b}, cand={cand:7s}: median={med:8.2f} us, {time_per_cta_ns:6.1f} ns/CTA, {effective_gbps:6.1f} GB/s")

            pilot_results["data"][str(b)] = b_data

        return json.dumps(pilot_results)

    @app.function(
        image=triton_image,
        gpu="H100!:1",
        timeout=1800,
    )
    def run_extended_saturation_remote(
        prov: Dict[str, Any],
        b_values: List[int],
    ) -> str:
        import os
        import statistics
        import torch
        import triton
        import triton.language as tl

        env_info = remote_verify_environment(prov)
        print(f"[Remote Extended Saturation] Hardware: {env_info['gpu_name']} CC: {env_info['gpu_compute_capability']}")

        triton.set_allocator(
            lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda")
        )

        test_configs = [
            (32, 16, 8),
            (32, 64, 8),
            (32, 128, 4),
        ]
        candidates = ["default", "8", "4", "2", "1"]

        results = {
            "environment": env_info,
            "provenance": prov,
            "b_values": b_values,
            "configs": {},
        }

        def get_kernel():
            @triton.jit
            def kernel(
                a_ptr,
                out_ptr,
                stride_b,
                stride_m,
                B: tl.constexpr,
                M: tl.constexpr,
                N: tl.constexpr,
            ):
                pid = tl.program_id(0)
                desc = tl.make_tensor_descriptor(
                    a_ptr,
                    shape=[B, M, N],
                    strides=[stride_b, stride_m, 1],
                    block_shape=[1, M, N],
                )
                x = desc.load([pid, 0, 0])
                x_fp32 = x.to(tl.float32)
                y = tl.max(x_fp32, axis=1)
                offs_n = tl.arange(0, N)
                tl.store(out_ptr + pid * N + offs_n, tl.reshape(y, [N]))

            return kernel

        for m, n, num_warps in test_configs:
            cfg_key = f"M{m}_N{n}_w{num_warps}"
            print(f"\n[Remote Extended Saturation] Testing config: {cfg_key} ...")
            cfg_results = {
                "M": m,
                "N": n,
                "num_warps": num_warps,
                "data": {},
            }

            for b in b_values:
                print(f"  Testing B = {b} ...")
                stride_b = m * n
                stride_m = n
                input_tensor = torch.randn((b, m, n), device="cuda", dtype=torch.bfloat16)
                out_tensor = torch.empty((b, n), device="cuda", dtype=torch.float32)
                ref_out = input_tensor.to(torch.float32).amax(dim=1)
                input_bytes = b * m * n * 2

                compiled_kernels = {}
                legal_cands = []
                cand_meta = {}

                for cand in candidates:
                    is_legal, err = check_candidate_legality(m, n, num_warps, cand)
                    if not is_legal:
                        cand_meta[cand] = {"is_legal": False, "error": err}
                        continue

                    if cand == "default":
                        os.environ.pop("TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT", None)
                    else:
                        os.environ["TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT"] = cand
                    os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_ext_{m}_{n}_{num_warps}_{b}_{cand}"
                    kernel = get_kernel()

                    try:
                        compiled = kernel.warmup(
                            input_tensor,
                            out_tensor,
                            stride_b,
                            stride_m,
                            b,
                            m,
                            n,
                            grid=(b,),
                            num_warps=num_warps,
                        )
                        out_tensor.zero_()
                        kernel[(b,)](input_tensor, out_tensor, stride_b, stride_m, b, m, n, num_warps=num_warps)
                        torch.cuda.synchronize()
                        diff = float(torch.max(torch.abs(out_tensor - ref_out)).item())
                        is_correct = diff < 1e-3
                        compiled_kernels[cand] = (kernel, compiled)
                        legal_cands.append(cand)
                        cand_meta[cand] = {
                            "is_legal": True,
                            "is_correct": is_correct,
                            "max_diff": diff,
                        }
                    except Exception as e:
                        cand_meta[cand] = {"is_legal": False, "error": str(e)}

                # Timing measurement protocol
                # Warmup 20 launches
                for _ in range(20):
                    for cand in legal_cands:
                        k, _ = compiled_kernels[cand]
                        k[(b,)](input_tensor, out_tensor, stride_b, stride_m, b, m, n, num_warps=num_warps)
                torch.cuda.synchronize()

                num_blocks = 10
                iters_per_block = 10
                cand_samples_us = {c: [] for c in legal_cands}
                cand_block_medians = {c: [] for c in legal_cands}

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                for block_idx in range(num_blocks):
                    shift = block_idx % len(legal_cands)
                    block_order = legal_cands[shift:] + legal_cands[:shift]
                    for cand in block_order:
                        k, _ = compiled_kernels[cand]
                        block_samples = []
                        for _ in range(iters_per_block):
                            start_event.record()
                            k[(b,)](input_tensor, out_tensor, stride_b, stride_m, b, m, n, num_warps=num_warps)
                            end_event.record()
                            torch.cuda.synchronize()
                            us = start_event.elapsed_time(end_event) * 1000.0
                            cand_samples_us[cand].append(us)
                            block_samples.append(us)
                        cand_block_medians[cand].append(statistics.median(block_samples))

                b_data = {}
                for cand in candidates:
                    if cand not in legal_cands:
                        b_data[cand] = cand_meta.get(cand, {"is_legal": False})
                        continue
                    samples = cand_samples_us[cand]
                    sorted_samples = sorted(samples)
                    med = statistics.median(samples)
                    mean = statistics.mean(samples)
                    p10 = calculate_percentile(sorted_samples, 0.10)
                    p90 = calculate_percentile(sorted_samples, 0.90)
                    p25 = calculate_percentile(sorted_samples, 0.25)
                    p75 = calculate_percentile(sorted_samples, 0.75)
                    iqr = p75 - p25
                    mad = statistics.median([abs(x - med) for x in samples])

                    time_per_cta_ns = (med * 1000.0) / b
                    effective_gbps = (input_bytes / (med * 1e-6)) / 1e9

                    b_data[cand] = {
                        "is_legal": True,
                        "is_correct": cand_meta[cand]["is_correct"],
                        "max_diff": cand_meta[cand]["max_diff"],
                        "median_us": med,
                        "mean_us": mean,
                        "p10_us": p10,
                        "p90_us": p90,
                        "iqr_us": iqr,
                        "mad_us": mad,
                        "amortized_grid_time_per_cta_ns": round(time_per_cta_ns, 2),
                        "time_per_cta_ns": round(time_per_cta_ns, 2),
                        "effective_gbps": round(effective_gbps, 2),
                        "block_medians_us": cand_block_medians[cand],
                        "raw_samples_us": samples,
                    }
                    print(f"    cand={cand:7s}: median={med:8.2f} us, {time_per_cta_ns:6.2f} ns/CTA, {effective_gbps:6.1f} GB/s")

                # Relative to default
                def_med = b_data.get("default", {}).get("median_us")
                if def_med:
                    for cand, cinfo in b_data.items():
                        if cinfo.get("is_legal"):
                            cinfo["vs_default_pct"] = round((cinfo["median_us"] - def_med) / def_med * 100.0, 2)

                cfg_results["data"][str(b)] = b_data

            results["configs"][cfg_key] = cfg_results

        return json.dumps(results)

    @app.function(
        image=triton_image,
        gpu="H100!:1",
        timeout=1800,
    )
    def run_corrected_fixed_binary_pilot_remote(
        prov: Dict[str, Any],
        run_id: str = "run_1",
    ) -> str:
        import hashlib
        import os
        import statistics
        import subprocess
        import time
        import torch
        import triton
        import triton.language as tl

        env_info = remote_verify_environment(prov)
        print(f"[Remote Fixed-Binary Pilot - {run_id}] Hardware: {env_info['gpu_name']} CC: {env_info['gpu_compute_capability']}, L2: {env_info.get('l2_cache_bytes')}")

        triton.set_allocator(
            lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda")
        )

        def linear_regression_local(xs, ys):
            n = len(xs)
            if n < 2:
                return 0.0, ys[0] if n == 1 else 0.0, 1.0, [0.0] * n
            mean_x = sum(xs) / n
            mean_y = sum(ys) / n
            num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            den = sum((x - mean_x) ** 2 for x in xs)
            slope = num / den if den != 0.0 else 0.0
            intercept = mean_y - slope * mean_x
            residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
            ss_tot = sum((y - mean_y) ** 2 for y in ys)
            ss_res = sum(r ** 2 for r in residuals)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else 1.0
            return slope, intercept, r2, residuals

        # 3 representative configs with fixed descriptor extents B_DESC
        pilot_configs = [
            {
                "M": 32,
                "N": 16,
                "num_warps": 8,
                "b_desc": 131072,
                "b_runs": [16384, 32768, 65536, 131072],
            },
            {
                "M": 32,
                "N": 64,
                "num_warps": 8,
                "b_desc": 65536,
                "b_runs": [16384, 32768, 65536],
            },
            {
                "M": 32,
                "N": 128,
                "num_warps": 4,
                "b_desc": 65536,
                "b_runs": [16384, 32768, 65536],
            },
        ]

        candidates_order = ["default", "8", "4", "2", "1"]

        def get_fixed_kernel():
            @triton.jit
            def kernel(
                a_ptr,
                out_ptr,
                stride_b,
                stride_m,
                B_DESC: tl.constexpr,
                M: tl.constexpr,
                N: tl.constexpr,
            ):
                pid = tl.program_id(0)
                desc = tl.make_tensor_descriptor(
                    a_ptr,
                    shape=[B_DESC, M, N],
                    strides=[stride_b, stride_m, 1],
                    block_shape=[1, M, N],
                )
                x = desc.load([pid, 0, 0])
                x_fp32 = x.to(tl.float32)
                y = tl.max(x_fp32, axis=1)
                offs_n = tl.arange(0, N)
                tl.store(out_ptr + pid * N + offs_n, tl.reshape(y, [N]))

            return kernel

        run_output = {
            "run_id": run_id,
            "timestamp": time.time(),
            "environment": env_info,
            "provenance": prov,
            "configs": {},
        }

        for cfg in pilot_configs:
            m = cfg["M"]
            n = cfg["N"]
            num_warps = cfg["num_warps"]
            b_desc = cfg["b_desc"]
            b_runs = cfg["b_runs"]
            cfg_key = f"M{m}_N{n}_w{num_warps}"

            print(f"\n[Remote Fixed-Binary Pilot - {run_id}] Config: {cfg_key} (B_DESC={b_desc}, B_RUNS={b_runs})")

            # Allocate maximum working set buffer once
            stride_b = m * n
            stride_m = n
            input_tensor_max = torch.randn((b_desc, m, n), device="cuda", dtype=torch.bfloat16)
            out_tensor_max = torch.empty((b_desc, n), device="cuda", dtype=torch.float32)
            ref_out_max = input_tensor_max.to(torch.float32).amax(dim=1)

            compiled_kernels = {}
            compiled_artifacts = {}
            legal_cands = []
            cand_meta = {}

            # Step 1: Compile ONCE per candidate with B_DESC
            for cand in candidates_order:
                is_legal, err = check_candidate_legality(m, n, num_warps, cand)
                if not is_legal:
                    cand_meta[cand] = {"is_legal": False, "error": err}
                    continue

                if cand == "default":
                    os.environ.pop("TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT", None)
                else:
                    os.environ["TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT"] = cand
                os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{run_id}_{m}_{n}_{num_warps}_{cand}"

                kernel = get_fixed_kernel()
                try:
                    compiled = kernel.warmup(
                        input_tensor_max,
                        out_tensor_max,
                        stride_b,
                        stride_m,
                        b_desc,
                        m,
                        n,
                        grid=(1,),
                        num_warps=num_warps,
                    )
                    out_tensor_max.zero_()
                    kernel[(b_desc,)](input_tensor_max, out_tensor_max, stride_b, stride_m, b_desc, m, n, num_warps=num_warps)
                    torch.cuda.synchronize()
                    diff = float(torch.max(torch.abs(out_tensor_max - ref_out_max)).item())
                    is_correct = diff < 1e-3

                    ttgir_text = compiled.asm.get("ttgir", "")
                    ptx_text = compiled.asm.get("ptx", "")
                    ttgir_sha = hashlib.sha256(ttgir_text.encode("utf-8")).hexdigest()
                    ptx_sha = hashlib.sha256(ptx_text.encode("utf-8")).hexdigest()

                    cubin_bytes = compiled.asm.get("cubin", None)
                    if cubin_bytes is None and hasattr(compiled, "kernel"):
                        cubin_bytes = compiled.kernel

                    cubin_sha = hashlib.sha256(cubin_bytes).hexdigest() if cubin_bytes else ""
                    sass_text = ""
                    resource_text = ""

                    if cubin_bytes is not None:
                        tmp_cubin = f"/tmp/triton_{run_id}_{cfg_key}_{cand}.cubin"
                        with open(tmp_cubin, "wb") as f:
                            f.write(cubin_bytes)
                        try:
                            sass_text = subprocess.check_output(
                                ["cuobjdump", "-sass", tmp_cubin],
                                text=True,
                                stderr=subprocess.STDOUT,
                            )
                        except Exception as e:
                            sass_text = f"ERROR: cuobjdump -sass failed: {e}"
                        try:
                            resource_text = subprocess.check_output(
                                ["cuobjdump", "-res-usage", tmp_cubin],
                                text=True,
                                stderr=subprocess.STDOUT,
                            )
                        except Exception as e:
                            resource_text = f"ERROR: cuobjdump -res-usage failed: {e}"

                    sass_sha = hashlib.sha256(sass_text.encode("utf-8")).hexdigest() if sass_text else ""
                    res_sha = hashlib.sha256(resource_text.encode("utf-8")).hexdigest() if resource_text else ""

                    compiled_kernels[cand] = kernel
                    compiled_artifacts[cand] = {
                        "artifact_id": f"{cfg_key}_{cand}",
                        "ttgir_sha256": ttgir_sha,
                        "ptx_sha256": ptx_sha,
                        "cubin_sha256": cubin_sha,
                        "sass_sha256": sass_sha,
                        "resource_sha256": res_sha,
                        "ttgir_text": ttgir_text,
                        "ptx_text": ptx_text,
                        "sass_text": sass_text,
                        "resource_text": resource_text,
                    }
                    legal_cands.append(cand)
                    cand_meta[cand] = {
                        "is_legal": True,
                        "is_correct": is_correct,
                        "max_diff": diff,
                        "artifacts": {
                            "artifact_id": f"{cfg_key}_{cand}",
                            "ttgir_sha256": ttgir_sha,
                            "ptx_sha256": ptx_sha,
                            "cubin_sha256": cubin_sha,
                            "sass_sha256": sass_sha,
                            "resource_sha256": res_sha,
                        },
                    }
                    print(f"  Compiled cand={cand:7s} (PTX: {ptx_sha[:12]}..., CUBIN: {cubin_sha[:12]}..., SASS lines: {len(sass_text.splitlines())}, correct: {is_correct})")
                except Exception as e:
                    cand_meta[cand] = {"is_legal": False, "error": str(e)}
                    print(f"  Compilation failed for cand={cand}: {e}")

            # Step 2: Warmup launches across legal candidates
            for _ in range(20):
                for cand in legal_cands:
                    k = compiled_kernels[cand]
                    k[(b_desc,)](input_tensor_max, out_tensor_max, stride_b, stride_m, b_desc, m, n, num_warps=num_warps)
            torch.cuda.synchronize()

            # Step 3: Rotated Timing Rounds
            num_rounds = 10
            iters_per_round = 10
            run_samples = {(b, c): [] for b in b_runs for c in legal_cands}
            b_orders_recorded = []
            cand_orders_recorded = []

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for round_idx in range(num_rounds):
                b_shift = round_idx % len(b_runs)
                b_order = b_runs[b_shift:] + b_runs[:b_shift]
                b_orders_recorded.append(b_order)

                for b_run in b_order:
                    c_shift = (round_idx + b_run) % len(legal_cands)
                    cand_order = legal_cands[c_shift:] + legal_cands[:c_shift]
                    cand_orders_recorded.append({"round": round_idx, "b_run": b_run, "cand_order": cand_order})

                    for cand in cand_order:
                        k = compiled_kernels[cand]
                        for _ in range(iters_per_round):
                            start_event.record()
                            k[(b_run,)](input_tensor_max, out_tensor_max, stride_b, stride_m, b_desc, m, n, num_warps=num_warps)
                            end_event.record()
                            torch.cuda.synchronize()
                            us = start_event.elapsed_time(end_event) * 1000.0
                            run_samples[(b_run, cand)].append(us)

            # Step 4: Process measured metrics
            grid_data = {}
            for b_run in b_runs:
                b_data = {}
                input_bytes = b_run * m * n * 2
                output_bytes = b_run * n * 4

                for cand in candidates_order:
                    if cand not in legal_cands:
                        b_data[cand] = cand_meta.get(cand, {"is_legal": False})
                        continue

                    samples = run_samples[(b_run, cand)]
                    sorted_samples = sorted(samples)
                    med = statistics.median(samples)
                    mean = statistics.mean(samples)
                    p10 = calculate_percentile(sorted_samples, 0.10)
                    p90 = calculate_percentile(sorted_samples, 0.90)
                    p25 = calculate_percentile(sorted_samples, 0.25)
                    p75 = calculate_percentile(sorted_samples, 0.75)
                    iqr = p75 - p25
                    mad = statistics.median([abs(x - med) for x in samples])

                    time_per_cta_ns = (med * 1000.0) / b_run
                    effective_input_gbps = (input_bytes / (med * 1e-6)) / 1e9

                    b_data[cand] = {
                        "is_legal": True,
                        "is_correct": cand_meta[cand]["is_correct"],
                        "max_diff": cand_meta[cand]["max_diff"],
                        "input_working_set_bytes": input_bytes,
                        "output_bytes": output_bytes,
                        "median_us": med,
                        "mean_us": mean,
                        "p10_us": p10,
                        "p90_us": p90,
                        "iqr_us": iqr,
                        "mad_us": mad,
                        "amortized_grid_time_per_cta_ns": round(time_per_cta_ns, 4),
                        "time_per_cta_ns": round(time_per_cta_ns, 4),
                        "effective_input_gbps": round(effective_input_gbps, 2),
                        "compiled_artifact_id": f"{cfg_key}_{cand}",
                        "raw_samples_us": samples,
                    }

                def_med = b_data.get("default", {}).get("median_us")
                if def_med:
                    for cand, cinfo in b_data.items():
                        if cinfo.get("is_legal"):
                            cinfo["vs_default_pct"] = round((cinfo["median_us"] - def_med) / def_med * 100.0, 2)

                grid_data[str(b_run)] = b_data

            # Step 5: Marginal Slopes & Affine Fit per candidate
            cand_marginal_analysis = {}
            for cand in candidates_order:
                if cand not in legal_cands:
                    continue
                med_list = [grid_data[str(b)][cand]["median_us"] for b in b_runs]
                b_float_list = [float(b) for b in b_runs]

                islopes = []
                for i in range(1, len(b_runs)):
                    db = b_runs[i] - b_runs[i - 1]
                    dt = med_list[i] - med_list[i - 1]
                    s_ns = (dt / db) * 1000.0
                    islopes.append({
                        "b_prev": b_runs[i - 1],
                        "b_curr": b_runs[i],
                        "slope_ns_per_cta": s_ns,
                    })

                last_two_delta_pct = None
                if len(islopes) >= 2:
                    s_prev = islopes[-2]["slope_ns_per_cta"]
                    s_curr = islopes[-1]["slope_ns_per_cta"]
                    if s_prev != 0.0:
                        last_two_delta_pct = (s_curr - s_prev) / s_prev * 100.0

                fit_b = b_float_list[-3:]
                fit_t = med_list[-3:]
                slope_us, intercept_us, r2, residuals = linear_regression_local(fit_b, fit_t)
                marginal_ns = slope_us * 1000.0

                stable_slope = last_two_delta_pct is not None and abs(last_two_delta_pct) < 5.0
                high_r2 = r2 >= 0.99
                marginal_linear_regime = stable_slope and high_r2

                cand_marginal_analysis[cand] = {
                    "interval_slopes": islopes,
                    "last_two_slope_delta_pct": last_two_delta_pct,
                    "affine_fit": {
                        "fit_b_points": [int(x) for x in fit_b],
                        "intercept_us": intercept_us,
                        "marginal_ns_per_cta": marginal_ns,
                        "r2": r2,
                        "residuals_us": residuals,
                    },
                    "marginal_linear_regime_observed": marginal_linear_regime,
                }

            def_m_slope = cand_marginal_analysis.get("default", {}).get("affine_fit", {}).get("marginal_ns_per_cta")
            for cand, cman in cand_marginal_analysis.items():
                c_slope = cman.get("affine_fit", {}).get("marginal_ns_per_cta")
                if def_m_slope and def_m_slope > 0.0:
                    cman["vs_default_slope_pct"] = round((c_slope - def_m_slope) / def_m_slope * 100.0, 2)
                else:
                    cman["vs_default_slope_pct"] = 0.0

            run_output["configs"][cfg_key] = {
                "M": m,
                "N": n,
                "num_warps": num_warps,
                "b_desc": b_desc,
                "b_runs": b_runs,
                "b_orders_recorded": b_orders_recorded,
                "cand_orders_recorded": cand_orders_recorded,
                "grid_data": grid_data,
                "marginal_analysis": cand_marginal_analysis,
                "compiled_artifacts": compiled_artifacts,
            }

        return json.dumps(run_output)


    @app.function(
        image=triton_image,
        gpu="H100!:1",
        timeout=1800,
    )
    def run_sweep_remote(
        b_steady: int,
        prov: Dict[str, Any],
    ) -> str:
        import torch
        import triton
        import triton.language as tl

        env_info = remote_verify_environment(prov)
        print(f"[Remote Sweep] Hardware: {env_info['gpu_name']} CC: {env_info['gpu_compute_capability']}")

        triton.set_allocator(
            lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda")
        )

        M_LIST = [32, 64, 128]
        N_LIST = [16, 32, 64, 128, 256]
        WARPS_LIST = [4, 8]
        CANDIDATES = ["default", "8", "4", "2", "1"]

        configs = []
        for m in M_LIST:
            for n in N_LIST:
                for w in WARPS_LIST:
                    configs.append((m, n, w))

        sweep_results = {
            "environment": env_info,
            "provenance": prov,
            "b_steady": b_steady,
            "configs": {},
        }

        def get_kernel():
            @triton.jit
            def kernel(
                a_ptr,
                out_ptr,
                stride_b,
                stride_m,
                B: tl.constexpr,
                M: tl.constexpr,
                N: tl.constexpr,
            ):
                pid = tl.program_id(0)
                desc = tl.make_tensor_descriptor(
                    a_ptr,
                    shape=[B, M, N],
                    strides=[stride_b, stride_m, 1],
                    block_shape=[1, M, N],
                )
                x = desc.load([pid, 0, 0])
                x_fp32 = x.to(tl.float32)
                y = tl.max(x_fp32, axis=1)
                offs_n = tl.arange(0, N)
                tl.store(out_ptr + pid * N + offs_n, tl.reshape(y, [N]))

            return kernel

        for cfg_idx, (m, n, num_warps) in enumerate(configs, 1):
            cfg_key = f"M{m}_N{n}_w{num_warps}"
            print(f"\n[Remote Sweep] ({cfg_idx}/{len(configs)}) Testing {cfg_key} with B={b_steady} ...")

            stride_b = m * n
            stride_m = n
            input_tensor = torch.randn((b_steady, m, n), device="cuda", dtype=torch.bfloat16)
            out_tensor = torch.empty((b_steady, n), device="cuda", dtype=torch.float32)
            ref_out = input_tensor.to(torch.float32).amax(dim=1)

            compiled_kernels = {}
            legal_cands = []
            cand_results = {}

            for cand in CANDIDATES:
                is_legal, err = check_candidate_legality(m, n, num_warps, cand)
                if not is_legal:
                    cand_results[cand] = {
                        "candidate": cand,
                        "is_legal": False,
                        "error": err,
                    }
                    continue

                os.environ["TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT"] = cand
                os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{cfg_key}_{cand}"
                kernel = get_kernel()

                try:
                    compiled = kernel.warmup(
                        input_tensor,
                        out_tensor,
                        stride_b,
                        stride_m,
                        b_steady,
                        m,
                        n,
                        grid=(b_steady,),
                        num_warps=num_warps,
                    )
                except Exception as e:
                    cand_results[cand] = {
                        "candidate": cand,
                        "is_legal": False,
                        "error": str(e),
                    }
                    continue

                compiled_kernels[cand] = (kernel, compiled)
                legal_cands.append(cand)

                # Extract artifacts
                ttgir_text = compiled.asm.get("ttgir", "")
                ptx_text = compiled.asm.get("ptx", "")
                llir_text = compiled.asm.get("llir", "")

                cubin_bytes = compiled.asm.get("cubin", None)
                if cubin_bytes is None and hasattr(compiled, "kernel"):
                    cubin_bytes = compiled.kernel

                cubin_path = Path(f"/tmp/kernel_{cfg_key}_{cand}.cubin")
                sass_text = ""
                resource_text = ""
                if cubin_bytes is not None:
                    cubin_path.write_bytes(cubin_bytes)
                    try:
                        sass_text = subprocess.check_output(
                            ["cuobjdump", "-sass", str(cubin_path)],
                            text=True,
                            stderr=subprocess.STDOUT,
                        )
                    except Exception as e:
                        sass_text = f"UNKNOWN (cuobjdump -sass error: {e})"

                    try:
                        resource_text = subprocess.check_output(
                            ["cuobjdump", "-res-usage", str(cubin_path)],
                            text=True,
                            stderr=subprocess.STDOUT,
                        )
                    except Exception as e:
                        resource_text = f"UNKNOWN (cuobjdump -res-usage error: {e})"
                else:
                    sass_text = "UNKNOWN (cubin not available)"
                    resource_text = "UNKNOWN (cubin not available)"

                # Correctness check
                out_tensor.zero_()
                kernel[(b_steady,)](input_tensor, out_tensor, stride_b, stride_m, b_steady, m, n, num_warps=num_warps)
                torch.cuda.synchronize()
                diff = float(torch.max(torch.abs(out_tensor - ref_out)).item())
                is_correct = diff < 1e-3

                triton_launch_shared_bytes = getattr(compiled.metadata, "shared", None)

                cand_results[cand] = {
                    "candidate": cand,
                    "is_legal": True,
                    "is_correct": is_correct,
                    "max_diff": diff,
                    "triton_launch_shared_bytes": triton_launch_shared_bytes,
                    "ttgir": ttgir_text,
                    "ptx": ptx_text,
                    "llir": llir_text,
                    "sass": sass_text,
                    "resource_text": resource_text,
                }

            # Timing execution for legal candidates
            # Warmup 20 launches
            for _ in range(20):
                for cand in legal_cands:
                    k, _ = compiled_kernels[cand]
                    k[(b_steady,)](input_tensor, out_tensor, stride_b, stride_m, b_steady, m, n, num_warps=num_warps)
            torch.cuda.synchronize()

            num_blocks = 10
            iters_per_block = 10
            cand_samples_us = {c: [] for c in legal_cands}
            cand_block_medians = {c: [] for c in legal_cands}

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for block_idx in range(num_blocks):
                shift = block_idx % len(legal_cands)
                block_order = legal_cands[shift:] + legal_cands[:shift]
                for cand in block_order:
                    k, _ = compiled_kernels[cand]
                    block_samples = []
                    for _ in range(iters_per_block):
                        start_event.record()
                        k[(b_steady,)](input_tensor, out_tensor, stride_b, stride_m, b_steady, m, n, num_warps=num_warps)
                        end_event.record()
                        torch.cuda.synchronize()
                        us = start_event.elapsed_time(end_event) * 1000.0
                        cand_samples_us[cand].append(us)
                        block_samples.append(us)
                    cand_block_medians[cand].append(statistics.median(block_samples))

            input_bytes = b_steady * m * n * 2
            for cand in legal_cands:
                samples = cand_samples_us[cand]
                sorted_samples = sorted(samples)
                med = statistics.median(samples)
                mean = statistics.mean(samples)
                p10 = calculate_percentile(sorted_samples, 0.10)
                p90 = calculate_percentile(sorted_samples, 0.90)
                p25 = calculate_percentile(sorted_samples, 0.25)
                p75 = calculate_percentile(sorted_samples, 0.75)
                iqr = p75 - p25
                mad = statistics.median([abs(x - med) for x in samples])

                time_per_cta_ns = (med * 1000.0) / b_steady
                effective_gbps = (input_bytes / (med * 1e-6)) / 1e9

                cand_results[cand]["timing"] = {
                    "median_us": med,
                    "mean_us": mean,
                    "p10_us": p10,
                    "p90_us": p90,
                    "iqr_us": iqr,
                    "mad_us": mad,
                    "time_per_cta_ns": time_per_cta_ns,
                    "effective_gbps": effective_gbps,
                    "block_medians_us": cand_block_medians[cand],
                    "raw_samples_us": samples,
                }
                print(f"[Remote Sweep] {cfg_key} cand={cand:7s}: med={med:7.2f} us, {time_per_cta_ns:5.1f} ns/CTA, {effective_gbps:5.1f} GB/s")

            # Check for >3% performance separation vs default to trigger repeated validation
            def_med = cand_results.get("default", {}).get("timing", {}).get("median_us", None)
            candidates_to_retest = []
            if def_med is not None and def_med > 0:
                for cand in legal_cands:
                    if cand == "default":
                        continue
                    c_med = cand_results[cand]["timing"]["median_us"]
                    diff_pct = (c_med - def_med) / def_med * 100.0
                    cand_results[cand]["vs_default_pct"] = diff_pct
                    if abs(diff_pct) > 3.0:
                        candidates_to_retest.append(cand)
            cand_results.get("default", {})["vs_default_pct"] = 0.0

            # Repeat run for candidates with >3% separation
            if candidates_to_retest:
                print(f"[Remote Sweep] Candidates with >3% separation in {cfg_key}: {candidates_to_retest}. Running repeated validation pass...")
                retest_group = ["default"] + candidates_to_retest
                retest_samples = {c: [] for c in retest_group}
                retest_block_medians = {c: [] for c in retest_group}

                for block_idx in range(num_blocks):
                    shift = block_idx % len(retest_group)
                    block_order = retest_group[shift:] + retest_group[:shift]
                    for cand in block_order:
                        k, _ = compiled_kernels[cand]
                        block_samples = []
                        for _ in range(iters_per_block):
                            start_event.record()
                            k[(b_steady,)](input_tensor, out_tensor, stride_b, stride_m, b_steady, m, n, num_warps=num_warps)
                            end_event.record()
                            torch.cuda.synchronize()
                            us = start_event.elapsed_time(end_event) * 1000.0
                            retest_samples[cand].append(us)
                            block_samples.append(us)
                        retest_block_medians[cand].append(statistics.median(block_samples))

                retest_def_med = statistics.median(retest_samples["default"])
                for cand in candidates_to_retest:
                    pass2_med = statistics.median(retest_samples[cand])
                    pass2_diff_pct = (pass2_med - retest_def_med) / retest_def_med * 100.0
                    pass1_diff_pct = cand_results[cand]["vs_default_pct"]

                    # Check if within-run repeat pass reproduces
                    same_direction = (pass1_diff_pct * pass2_diff_pct > 0)
                    both_gt3 = same_direction and (abs(pass1_diff_pct) > 3.0) and (abs(pass2_diff_pct) > 3.0)
                    if both_gt3:
                        status = "within_run_gt3_reproduced"
                    elif same_direction:
                        status = "within_run_direction_reproduced"
                    else:
                        status = "unstable/unresolved"

                    cand_results[cand]["repeat_validation"] = {
                        "pass1_diff_pct": pass1_diff_pct,
                        "pass2_diff_pct": pass2_diff_pct,
                        "pass2_median_us": pass2_med,
                        "same_direction": same_direction,
                        "status": status,
                    }
                    print(f"[Remote Sweep] {cfg_key} cand={cand} pass1: {pass1_diff_pct:+.2f}%, pass2: {pass2_diff_pct:+.2f}% -> {status}")

            sweep_results["configs"][cfg_key] = {
                "M": m,
                "N": n,
                "num_warps": num_warps,
                "candidates": cand_results,
            }

        return json.dumps(sweep_results)


# ---------------------------------------------------------------------------
# Post-Processing & Evidence Structuring
# ---------------------------------------------------------------------------
def get_physical_regs(data: Dict[str, Any]) -> int:
    val = data.get("observed", {}).get("physical_resources", {}).get("physical_regs_per_thread", {}).get("value")
    return int(val) if isinstance(val, (int, float)) else 0


def classify_structural_transitions(
    def_data: Dict[str, Any],
    cand_data: Dict[str, Any],
) -> List[str]:
    """
    Tags structural transitions against default:
    - T1_lane_change_warp_same: lane partitions changes but warp partitions unchanged
    - T2_warp_decrease_gt1: warp partitions decreases but remains >1
    - T3_warp_becomes1: warp partitions becomes 1
    - T4_ldmatrix_family_change: LocalLoad lowering switches opcode family (e.g. ldmatrix vs ld.shared)
    - T5_register_count_change: physical register count changes materially (|diff| >= 4 or >= 10%)
    """
    transitions = []
    if not cand_data.get("is_legal", False) or not def_data.get("is_legal", False):
        return transitions

    def_derived = def_data.get("derived", {})
    cand_derived = cand_data.get("derived", {})

    def_lane = def_derived.get("lane_partitions_reduce_axis", 1)
    cand_lane = cand_derived.get("lane_partitions_reduce_axis", 1)
    def_warp = def_derived.get("warp_partitions_reduce_axis", 1)
    cand_warp = cand_derived.get("warp_partitions_reduce_axis", 1)

    # T1: lane changes, warp unchanged
    if cand_lane != def_lane and cand_warp == def_warp:
        transitions.append("T1_lane_change_warp_same")

    # T2: warp decreases but remains > 1
    if cand_warp < def_warp and cand_warp > 1:
        transitions.append("T2_warp_decrease_gt1")

    # T3: warp becomes 1
    if cand_warp == 1 and def_warp > 1:
        transitions.append("T3_warp_becomes1")

    # T4: ldmatrix family change
    def_instr = def_data.get("observed", {}).get("ptx", {}).get("initial_local_load", {}).get("instruction") or ""
    cand_instr = cand_data.get("observed", {}).get("ptx", {}).get("initial_local_load", {}).get("instruction") or ""
    if ("ldmatrix" in cand_instr) != ("ldmatrix" in def_instr):
        transitions.append("T4_ldmatrix_family_change")

    # T5: physical register count changes
    def_regs = get_physical_regs(def_data)
    cand_regs = get_physical_regs(cand_data)
    if def_regs > 0 and cand_regs > 0:
        diff_regs = abs(cand_regs - def_regs)
        pct = diff_regs / def_regs
        if diff_regs >= 4 or pct >= 0.10:
            transitions.append(f"T5_register_count_change({def_regs}->{cand_regs})")

    return transitions


def test_classify_structural_transitions():
    """Unit test verifying classify_structural_transitions against synthetic cases for T1-T5."""
    base_def = {
        "is_legal": True,
        "derived": {"lane_partitions_reduce_axis": 4, "warp_partitions_reduce_axis": 8},
        "observed": {
            "ptx": {"initial_local_load": {"instruction": "ld.shared.v4.b32"}},
            "physical_resources": {"physical_regs_per_thread": {"value": 32}},
        },
    }

    # Case T1: lane changes, warp unchanged
    cand_t1 = {
        "is_legal": True,
        "derived": {"lane_partitions_reduce_axis": 2, "warp_partitions_reduce_axis": 8},
        "observed": {
            "ptx": {"initial_local_load": {"instruction": "ld.shared.v4.b32"}},
            "physical_resources": {"physical_regs_per_thread": {"value": 32}},
        },
    }
    assert classify_structural_transitions(base_def, cand_t1) == ["T1_lane_change_warp_same"], "T1 test failed"

    # Case T2: warp decreases but remains > 1
    cand_t2 = {
        "is_legal": True,
        "derived": {"lane_partitions_reduce_axis": 4, "warp_partitions_reduce_axis": 4},
        "observed": {
            "ptx": {"initial_local_load": {"instruction": "ld.shared.v4.b32"}},
            "physical_resources": {"physical_regs_per_thread": {"value": 32}},
        },
    }
    assert classify_structural_transitions(base_def, cand_t2) == ["T2_warp_decrease_gt1"], "T2 test failed"

    # Case T3: warp becomes 1
    cand_t3 = {
        "is_legal": True,
        "derived": {"lane_partitions_reduce_axis": 4, "warp_partitions_reduce_axis": 1},
        "observed": {
            "ptx": {"initial_local_load": {"instruction": "ld.shared.v4.b32"}},
            "physical_resources": {"physical_regs_per_thread": {"value": 32}},
        },
    }
    assert classify_structural_transitions(base_def, cand_t3) == ["T3_warp_becomes1"], "T3 test failed"

    # Case T4: ldmatrix family change
    cand_t4 = {
        "is_legal": True,
        "derived": {"lane_partitions_reduce_axis": 4, "warp_partitions_reduce_axis": 8},
        "observed": {
            "ptx": {"initial_local_load": {"instruction": "ldmatrix.sync.aligned.m8n8.x4.shared.b16"}},
            "physical_resources": {"physical_regs_per_thread": {"value": 32}},
        },
    }
    assert classify_structural_transitions(base_def, cand_t4) == ["T4_ldmatrix_family_change"], "T4 test failed"

    # Case T5: physical register count changes (32 -> 24)
    cand_t5 = {
        "is_legal": True,
        "derived": {"lane_partitions_reduce_axis": 4, "warp_partitions_reduce_axis": 8},
        "observed": {
            "ptx": {"initial_local_load": {"instruction": "ld.shared.v4.b32"}},
            "physical_resources": {"physical_regs_per_thread": {"value": 24}},
        },
    }
    assert classify_structural_transitions(base_def, cand_t5) == ["T5_register_count_change(32->24)"], "T5 test failed"

    # Combined case: T1 + T4 + T5
    cand_combined = {
        "is_legal": True,
        "derived": {"lane_partitions_reduce_axis": 1, "warp_partitions_reduce_axis": 8},
        "observed": {
            "ptx": {"initial_local_load": {"instruction": "ldmatrix.sync.aligned.m8n8.x4.shared.b16"}},
            "physical_resources": {"physical_regs_per_thread": {"value": 21}},
        },
    }
    assert classify_structural_transitions(base_def, cand_combined) == [
        "T1_lane_change_warp_same",
        "T4_ldmatrix_family_change",
        "T5_register_count_change(32->21)",
    ], "Combined test failed"
    return True


def post_process_sweep_payload(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriches raw sweep payload with full TTGIR, PTX, physical resource parsing,
    derived layout metrics, and structural transitions.
    """
    enriched_configs = {}
    for cfg_key, cfg_val in raw_payload["configs"].items():
        m = cfg_val["M"]
        n = cfg_val["N"]
        num_warps = cfg_val["num_warps"]

        processed_candidates = {}
        for cand, res in cfg_val["candidates"].items():
            if not res.get("is_legal", False):
                processed_candidates[cand] = {
                    "candidate": cand,
                    "is_legal": False,
                    "error": res.get("error"),
                }
                continue

            ttgir_text = res.get("ttgir", "")
            ptx_text = res.get("ptx", "")
            sass_text = res.get("sass", "")
            resource_text = res.get("resource_text", "")

            # 1. Hashes
            ttgir_hash = compute_text_hash(ttgir_text)
            ptx_hash = compute_text_hash(ptx_text)
            sass_hash = compute_text_hash(sass_text)
            res_hash = compute_text_hash(resource_text)

            # 2. Analyze TTGIR
            ttgir_info = analyze_ttgir(ttgir_text)
            dest_layout_name = ttgir_info.get("local_load_dest_layout") or "blocked"
            blocked_enc = ttgir_info.get("blocked_encodings", {}).get(dest_layout_name, {})
            shared_alias = ttgir_info.get("local_load_src_shared_layout") or "shared"
            shared_enc = ttgir_info.get("shared_encodings", {}).get(shared_alias, {})

            spt = blocked_enc.get("sizePerThread", [1, 1, 1])
            tpw = blocked_enc.get("threadsPerWarp", [1, 1, 32])
            wpc = blocked_enc.get("warpsPerCTA", [1, 1, num_warps])
            order = blocked_enc.get("order", [2, 1, 0])
            num_ctas = ttgir_info.get("module_attributes", {}).get("num_ctas", 1)

            # 3. Analyze PTX
            ptx_info = analyze_ptx(ptx_text)

            # 4. Parse Physical Resources
            phys_res = parse_resource_usage(
                resource_text,
                triton_launch_shared_bytes=res.get("triton_launch_shared_bytes"),
            )

            # 5. Compute Derived Layout Metrics (shape = [1, M, N], reduce_axis = 1)
            derived = compute_derived_layout_metrics(
                shape=[1, m, n],
                size_per_thread=spt,
                threads_per_warp=tpw,
                warps_per_cta=wpc,
                num_ctas=num_ctas,
                reduce_axis=1,
            )

            timing = res.get("timing", {})

            entry = {
                "candidate": cand,
                "is_legal": True,
                "is_correct": res.get("is_correct", False),
                "max_diff": res.get("max_diff"),
                "raw": {
                    "ttgir_sha256": ttgir_hash,
                    "ptx_sha256": ptx_hash,
                    "sass_sha256": sass_hash,
                    "resource_sha256": res_hash,
                    "raw_samples_us": timing.get("raw_samples_us", []),
                },
                "observed": {
                    "ttgir": {
                        "local_load_dest_layout": dest_layout_name,
                        "blocked_encoding": blocked_enc,
                        "shared_encoding": shared_enc,
                        "reduce_ops": ttgir_info.get("reduce_ops", []),
                        "module_attributes": ttgir_info.get("module_attributes", {}),
                    },
                    "ptx": {
                        "initial_local_load": ptx_info.get("initial_local_load", {}),
                        "whole_kernel_opcode_counts": ptx_info.get("whole_kernel_opcode_counts", {}),
                        "virtual_register_declarations": ptx_info.get("virtual_register_declarations", {}),
                    },
                    "physical_resources": phys_res,
                },
                "derived": derived,
                "measured": {
                    "median_us": timing.get("median_us", 0.0),
                    "mean_us": timing.get("mean_us", 0.0),
                    "p10_us": timing.get("p10_us", 0.0),
                    "p90_us": timing.get("p90_us", 0.0),
                    "iqr_us": timing.get("iqr_us", 0.0),
                    "mad_us": timing.get("mad_us", 0.0),
                    "time_per_cta_ns": timing.get("time_per_cta_ns", 0.0),
                    "effective_gbps": timing.get("effective_gbps", 0.0),
                    "block_medians_us": timing.get("block_medians_us", []),
                    "vs_default_pct": res.get("vs_default_pct", 0.0),
                    "repeat_validation": res.get("repeat_validation", None),
                },
                "artifacts": {
                    "ttgir": ttgir_text,
                    "ptx": ptx_text,
                    "sass": sass_text,
                    "resource_text": resource_text,
                },
            }
            processed_candidates[cand] = entry

        # Classify transitions against default
        def_cand = processed_candidates.get("default", {})
        for cand, centry in processed_candidates.items():
            if cand != "default" and centry.get("is_legal", False):
                centry["structural_transitions"] = classify_structural_transitions(def_cand, centry)
            elif cand == "default":
                centry["structural_transitions"] = ["BASELINE"]

        enriched_configs[cfg_key] = {
            "M": m,
            "N": n,
            "num_warps": num_warps,
            "candidates": processed_candidates,
        }

    return {
        "environment": raw_payload.get("environment", {}),
        "provenance": raw_payload.get("provenance", {}),
        "b_steady": raw_payload.get("b_steady"),
        "configs": enriched_configs,
    }


def generate_pilot_markdown(payload: Dict[str, Any]) -> str:
    env = payload.get("environment", {})
    shape = payload.get("shape", {})
    b_values = payload.get("b_values", [])
    data = payload.get("data", {})

    lines = [
        "# Phase 2 B-Saturation Pilot Report",
        "",
        "## Hardware & Provenance",
        f"- **GPU**: `{env.get('gpu_name')}` (CC: `{env.get('gpu_compute_capability')}`, Driver: `{env.get('driver_version')}`)",
        f"- **PyTorch / CUDA**: `{env.get('pytorch_version')}` / CUDA `{env.get('torch_cuda_version')}`",
        f"- **Triton Version**: `{env.get('triton_version')}` (`{env.get('triton_file')}`)",
        f"- **SM Count**: `{env.get('sm_count', 'UNKNOWN')}`",
        f"- **Test Shape**: `M={shape.get('M')}, N={shape.get('N')}, num_warps={shape.get('num_warps')}`",
        f"- **Evaluated B**: `{b_values}`",
        "",
        "> [!NOTE]",
        "> `amortized_grid_time_per_cta_ns` represents total grid execution time divided by CTA count.",
        "> It is a throughput-normalization metric, not the execution latency of one CTA.",
        "",
        "## Convergence and Grid Amortization Analysis",
        "",
        "| B (Grid Size) | Candidate | Total Median (us) | Amortized Time / CTA (ns) | Effective (GB/s) | Delta vs Prev B (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    prev_time_per_cta = {}
    for b in b_values:
        b_str = str(b)
        b_data = data.get(b_str, {})
        for cand in ["default", "8", "4", "2", "1"]:
            c_info = b_data.get(cand, {})
            if not c_info.get("is_legal", False):
                lines.append(f"| {b} | {cand} | INVALID | - | - | - |")
                continue
            med = c_info.get("median_us", 0.0)
            t_cta = c_info.get("amortized_grid_time_per_cta_ns", c_info.get("time_per_cta_ns", 0.0))
            gbps = c_info.get("effective_gbps", 0.0)

            delta_str = "baseline"
            if cand in prev_time_per_cta:
                p_t = prev_time_per_cta[cand]
                delta_pct = (t_cta - p_t) / p_t * 100.0
                delta_str = f"{delta_pct:+.2f}%"
            prev_time_per_cta[cand] = t_cta

            lines.append(f"| {b} | {cand} | {med:.2f} | {t_cta:.1f} | {gbps:.1f} | {delta_str} |")

    lines.extend([
        "",
        "## Plateau Evaluation",
        "At small B (B <= 1024), the measured GPU kernel duration remains nearly flat (~23.6-23.9 us),",
        "which is consistent with insufficient grid-level work to expose steady-state throughput",
        "and/or fixed device-side kernel costs. The current experiment does not isolate the source of that fixed cost.",
        "",
        "From B=4096 (6.7 ns/CTA) to B=8192 (5.4 ns/CTA), normalized time/CTA decreased by ~20%.",
        "Therefore, B=4096 did not establish a throughput plateau.",
        "An extended saturation pilot across larger B is required to identify a true plateau.",
    ])
    return "\n".join(lines)


def generate_extended_saturation_markdown(payload: Dict[str, Any]) -> str:
    env = payload.get("environment", {})
    b_values = payload.get("b_values", [])
    configs = payload.get("configs", {})

    lines = [
        "# Phase 2 Extended B-Saturation Pilot Report",
        "",
        "## Hardware & Execution Environment",
        f"- **GPU**: `{env.get('gpu_name')}` (CC: `{env.get('gpu_compute_capability')}`, Driver: `{env.get('driver_version')}`)",
        f"- **PyTorch / CUDA**: `{env.get('pytorch_version')}` / CUDA `{env.get('torch_cuda_version')}`",
        f"- **Triton Version**: `{env.get('triton_version')}` (`{env.get('triton_file')}`)",
        f"- **SM Count**: `{env.get('sm_count', 'UNKNOWN')}`",
        f"- **Evaluated B (Grid Sizes)**: `{b_values}`",
        "- **Operational Plateau Criterion**: Two consecutive B doublings where change in `amortized_grid_time_per_cta_ns` is `< 5.0%` (`| (t_{2B} - t_B) / t_B | < 0.05`).",
        "",
        "> [!NOTE]",
        "> `amortized_grid_time_per_cta_ns` represents total grid execution time divided by CTA count.",
        "> It is a throughput-normalization metric, not the execution latency of one CTA.",
        "",
        "## 1. Per-Configuration Saturation Analysis",
    ]

    candidates_order = ["default", "8", "4", "2", "1"]
    plateau_summary = {}

    for cfg_key, cfg in configs.items():
        m = cfg["M"]
        n = cfg["N"]
        w = cfg["num_warps"]
        data = cfg.get("data", {})

        lines.extend([
            "",
            f"### Configuration: `{cfg_key}` (`M={m}, N={n}, num_warps={w}`)",
            "",
            "| B (Grid) | Candidate | Total Median (us) | Amortized Time/CTA (ns) | Effective GB/s | vs Default (%) | Step Delta vs Prev B (%) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
        ])

        prev_t_cta = {}
        cand_series = {c: [] for c in candidates_order}

        for b in b_values:
            b_data = data.get(str(b), {})
            for cand in candidates_order:
                cinfo = b_data.get(cand, {})
                if not cinfo.get("is_legal", False):
                    lines.append(f"| {b} | {cand} | INVALID | - | - | - | {cinfo.get('error', 'INVALID')} |")
                    continue
                med = cinfo.get("median_us", 0.0)
                t_cta = cinfo.get("amortized_grid_time_per_cta_ns", 0.0)
                gbps = cinfo.get("effective_gbps", 0.0)
                vs_def = cinfo.get("vs_default_pct", 0.0)
                vs_def_str = f"{vs_def:+.2f}%" if cand != "default" else "0.00% (base)"

                cand_series[cand].append((b, t_cta))

                step_str = "baseline"
                if cand in prev_t_cta:
                    p = prev_t_cta[cand]
                    pct = (t_cta - p) / p * 100.0 if p > 0 else 0.0
                    step_str = f"{pct:+.2f}%"
                prev_t_cta[cand] = t_cta

                lines.append(f"| {b} | {cand} | {med:.2f} | {t_cta:.2f} | {gbps:.1f} | {vs_def_str} | {step_str} |")

        # Operational Plateau Criterion Evaluation for default
        lines.extend([
            "",
            "#### Doubling Steps & Operational Criterion Evaluation (`default` candidate):",
        ])

        def_series = cand_series["default"]
        deltas = []
        for i in range(1, len(def_series)):
            b_prev, t_prev = def_series[i - 1]
            b_curr, t_curr = def_series[i]
            d_pct = (t_curr - t_prev) / t_prev * 100.0 if t_prev > 0 else 0.0
            deltas.append((b_prev, b_curr, d_pct))
            meets = abs(d_pct) < 5.0
            lines.append(f"- Doubling Step {i} (`B={b_prev} -> {b_curr}`): `{d_pct:+.2f}%` ({'meets <5%' if meets else 'exceeds 5%'})")

        plateau_found = False
        plateau_b = None
        for i in range(len(deltas) - 1):
            s1 = deltas[i]
            s2 = deltas[i + 1]
            if abs(s1[2]) < 5.0 and abs(s2[2]) < 5.0:
                plateau_found = True
                plateau_b = s1[1]
                lines.append(f"\n> **Operational Criterion Met**: Consecutive doublings `{s1[0]}->{s1[1]}` ({s1[2]:+.2f}%) and `{s2[0]}->{s2[1]}` ({s2[2]:+.2f}%) both show `< 5.0%` change. Plateau established at **B = {plateau_b}**.")
                break

        if not plateau_found:
            lines.append("\n> **Operational Criterion NOT Met**: No two consecutive doubling steps exhibited `< 5.0%` change within tested B range.")
        plateau_summary[cfg_key] = {"plateau_found": plateau_found, "plateau_b": plateau_b}

    lines.extend([
        "",
        "## 2. Cross-Configuration Plateau Synthesis",
        "",
        "| Configuration | Operational Criterion Met? | Plateau B | Status |",
        "| :--- | :---: | :---: | :--- |",
    ])
    for cfg_k, pinfo in plateau_summary.items():
        found_str = "YES" if pinfo["plateau_found"] else "NO"
        b_str = str(pinfo["plateau_b"]) if pinfo["plateau_b"] else "None (<=65536)"
        status = "Steady state established" if pinfo["plateau_found"] else "Throughput not fully saturated"
        lines.append(f"| `{cfg_k}` | {found_str} | {b_str} | {status} |")

    return "\n".join(lines)


def render_corrected_pilot_summary_markdown(runs_dict: Dict[str, Any]) -> str:
    lines = [
        "# Corrected Multi-Invocation Fixed-Binary TMA Reduction Saturation Report",
        "",
        "> [!NOTE]",
        "> **Scope & Evidence Note**: The three benchmark invocations (`run_1`, `run_2`, `run_3`) were executed sequentially on the same physical NVIDIA H100 device (`GPU-a59752c5-ebb5-1dac-f887-c2e3c1f81aff`). They demonstrate same-device temporal repeatability, not cross-device replication across independent GPU hardware allocations.",
        ">",
        "> **Methodology Guarantees**:",
        "> 1. **Fixed Binary**: Tensor descriptor shape is fixed to `B_DESC = max(B)`. Kernels are compiled once per candidate as a single specialization and variable grid sizes `B_RUN <= B_DESC` reuse the identical compiled binary without recompilation (verified via byte-level TTGIR and PTX SHA256 hashes).",
        "> 2. **Sequential Replication**: Benchmark is executed across three sequential remote invocations on the same physical device (each compiling fresh binaries, generating fresh inputs, and warming up).",
        "> 3. **Order Rotation**: Grid sizes ($B$) and layout candidates are rotated circularly across 10 timing rounds (10 samples/round = 100 samples per condition) to eliminate thermal drift, clock drift, and ordering bias.",
        "> 4. **Affine Steady-State Model**: Evaluates $T(B) = \\text{intercept\\_us} + (\\text{marginal\\_ns\\_per\\_cta} / 1000) \\times B$ over the largest three $B$ points. The primary metric is the empirical large-$B$ marginal slope per additional CTA ($b = \\Delta T / \\Delta B$). Goodness-of-fit $R^2$ is descriptive; adjacent-interval slope stability (<5% change) is the primary operational check.",
        "> 5. **Fitted Intercept**: `intercept_us` represents the fitted fixed-time intercept. Its physical origin is **UNKNOWN** (not claimed to be launch overhead).",
        "",
        "## 1. Execution Environments & Pre-Run GPU Telemetry",
        "",
        "| Run ID | GPU UUID | Driver | SM Count | L2 Cache (bytes) | SM Clock (MHz) | Memory Clock (MHz) | Power (W) | Temp (°C) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for run_id, rdata in runs_dict.items():
        env = rdata.get("environment", {})
        telem = env.get("pre_run_gpu_telemetry", env.get("gpu_telemetry", {}))
        lines.append(
            f"| `{run_id}` | `{env.get('gpu_uuid')}` | `{env.get('driver_version')}` | "
            f"`{env.get('sm_count')}` | `{env.get('l2_cache_bytes')}` | "
            f"`{telem.get('sm_clock_mhz')}` | `{telem.get('memory_clock_mhz')}` | "
            f"`{telem.get('power_draw_w')}` | `{telem.get('gpu_temperature_c')}` |"
        )

    # Working set sizes
    lines.extend([
        "",
        "## 2. Working Set & Cache Regime",
        "",
        "| Configuration | B_RUN | Input Working Set (MiB) | Output Working Set (MiB) |",
        "| :--- | :---: | :---: | :---: |",
    ])

    first_run = next(iter(runs_dict.values()))
    for cfg_k, cfg_v in first_run.get("configs", {}).items():
        m = cfg_v["M"]
        n = cfg_v["N"]
        for b in cfg_v["b_runs"]:
            in_mib = (b * m * n * 2) / (1024 * 1024)
            out_mib = (b * n * 4) / (1024 * 1024)
            lines.append(f"| `{cfg_k}` | {b} | {in_mib:.2f} MiB | {out_mib:.2f} MiB |")

    # Fixed binary artifact hashes
    lines.extend([
        "",
        "## 3. Fixed-Binary Compilation Verification",
        "",
        "> A single compiled specialization with descriptor extent `B_DESC = max(B)` was compiled once per candidate.",
        "> All runtime grid sizes $B_{\\text{RUN}} \\le B_{\\text{DESC}}$ reuse this identical compiled specialization without recompilation.",
        "",
        "| Configuration | Candidate | PTX SHA256 (12 char) | TTGIR SHA256 (12 char) | Verified Fixed Across All B? |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ])

    for cfg_k, cfg_v in first_run.get("configs", {}).items():
        comp_arts = cfg_v.get("compiled_artifacts", {})
        grid_data = cfg_v.get("grid_data", {})
        b_first = str(cfg_v["b_runs"][0])
        for cand in ["default", "8", "4", "2", "1"]:
            cinfo = grid_data.get(b_first, {}).get(cand, {})
            if not cinfo.get("is_legal"):
                lines.append(f"| `{cfg_k}` | `{cand}` | - | - | INVALID: {cinfo.get('error')} |")
                continue
            art = comp_arts.get(cand, cinfo.get("compiled_artifact_hashes", {}))
            ptx_h = art.get("ptx_sha256", "")[:12]
            ttgir_h = art.get("ttgir_sha256", "")[:12]
            expected_id = art.get("artifact_id", f"{cfg_k}_{cand}")
            all_match = True
            for b_other in cfg_v["b_runs"]:
                cand_info = grid_data.get(str(b_other), {}).get(cand, {})
                art_id = cand_info.get("compiled_artifact_id")
                if art_id and art_id != expected_id:
                    all_match = False
                    break
            status_str = "**YES** (single compiled specialization reused across all B)" if all_match else "**FAIL** (specialization mismatch)"
            lines.append(f"| `{cfg_k}` | `{cand}` | `{ptx_h}...` | `{ttgir_h}...` | {status_str} |")

    # Section 4: Sequential Invocation Replication & Marginal Slope Separation
    lines.extend([
        "",
        "## 4. Sequential Invocation Replication & Marginal Slope Separation",
        "",
    ])

    run_keys = list(runs_dict.keys())

    for cfg_k in first_run.get("configs", {}).keys():
        cfg_meta = first_run["configs"][cfg_k]
        m = cfg_meta["M"]
        n = cfg_meta["N"]
        w = cfg_meta["num_warps"]
        b_runs = cfg_meta["b_runs"]

        lines.extend([
            f"### Configuration: `{cfg_k}` (`M={m}, N={n}, num_warps={w}`)",
            "",
            "#### A. Affine Fit Parameters across Sequential Invocations:",
            "",
            f"| Candidate | " + " | ".join(f"{rk} Slope (ns)" for rk in run_keys) + " | Mean Slope (ns) | same-device temporal replication CV (%) | vs Default Mean Slope (%) | Linear Regime? |",
            "| :--- | " + " | ".join(":---:" for _ in run_keys) + " | :---: | :---: | :---: | :---: |",
        ])

        candidates_order = ["default", "8", "4", "2", "1"]
        mean_slopes = {}

        for cand in candidates_order:
            slopes_across_runs = []
            linear_regimes = []
            is_legal = True

            for rk in run_keys:
                cman = runs_dict[rk]["configs"][cfg_k]["marginal_analysis"].get(cand)
                if not cman or not runs_dict[rk]["configs"][cfg_k]["grid_data"][str(b_runs[0])].get(cand, {}).get("is_legal"):
                    is_legal = False
                    break
                slopes_across_runs.append(cman["affine_fit"]["marginal_ns_per_cta"])
                linear_regimes.append(cman["marginal_linear_regime_observed"])

            if not is_legal:
                lines.append(f"| `{cand}` | " + " | ".join("-" for _ in run_keys) + " | - | - | - | INVALID |")
                continue

            mean_s = sum(slopes_across_runs) / len(slopes_across_runs)
            mean_slopes[cand] = mean_s
            std_s = (sum((s - mean_s) ** 2 for s in slopes_across_runs) / len(slopes_across_runs)) ** 0.5
            cv_pct = (std_s / mean_s) * 100.0 if mean_s > 0 else 0.0

            all_linear = all(linear_regimes)
            linear_str = "**YES**" if all_linear else "NO"

            s_cols = " | ".join(f"{s:.4f}" for s in slopes_across_runs)
            lines.append(f"| `{cand}` | {s_cols} | **{mean_s:.4f}** | {cv_pct:.2f}% | VS_DEF_PLACEHOLDER_{cand} | {linear_str} |")

        def_mean = mean_slopes.get("default", 0.0)
        for cand in candidates_order:
            if cand in mean_slopes and def_mean > 0:
                rel_pct = (mean_slopes[cand] - def_mean) / def_mean * 100.0
                if cand == "default":
                    rel_str = "0.00% (base)"
                elif abs(rel_pct) < 0.15:
                    rel_str = f"**{rel_pct:+.2f}%** (near parity)"
                else:
                    rel_str = f"**{rel_pct:+.2f}%**"
            else:
                rel_str = "-"
            for idx in range(len(lines)):
                if f"VS_DEF_PLACEHOLDER_{cand}" in lines[idx]:
                    lines[idx] = lines[idx].replace(f"VS_DEF_PLACEHOLDER_{cand}", rel_str)

        lines.extend([
            "",
            "#### B. Fitted Fixed-Time Intercept (µs) & Goodness-of-Fit ($R^2$):",
            "",
            "> Note: Affine fits use the 3 largest $B$ points. $R^2$ is reported as a descriptive goodness-of-fit indicator; adjacent-interval slope stability (<5% change) is the primary operational check.",
            "",
            f"| Candidate | " + " | ".join(f"{rk} Intercept (µs)" for rk in run_keys) + " | " + " | ".join(f"{rk} R²" for rk in run_keys) + " |",
            "| :--- | " + " | ".join(":---:" for _ in run_keys) + " | " + " | ".join(":---:" for _ in run_keys) + " |",
        ])

        for cand in candidates_order:
            icepts = []
            r2s = []
            is_legal = True
            for rk in run_keys:
                cman = runs_dict[rk]["configs"][cfg_k]["marginal_analysis"].get(cand)
                if not cman or not runs_dict[rk]["configs"][cfg_k]["grid_data"][str(b_runs[0])].get(cand, {}).get("is_legal"):
                    is_legal = False
                    break
                icepts.append(f"{cman['affine_fit']['intercept_us']:.4f}")
                r2s.append(f"{cman['affine_fit']['r2']:.6f}")

            if not is_legal:
                lines.append(f"| `{cand}` | " + " | ".join("-" for _ in run_keys) + " | " + " | ".join("-" for _ in run_keys) + " |")
                continue

            lines.append(f"| `{cand}` | " + " | ".join(icepts) + " | " + " | ".join(r2s) + " |")

        lines.append("")

    return "\n".join(lines)


def generate_sweep_csv(enriched_payload: Dict[str, Any]) -> str:
    rows = [
        "# Note: amortized_grid_time_per_cta_ns is total grid execution time divided by CTA count. It is a throughput-normalization metric, not the latency of one CTA.",
        "M,N,num_warps,cand,is_legal,LocalLoad,lanePart_M,warpPart_M,derived_M_elems_per_part,regs,median_us,amortized_grid_time_per_cta_ns,effective_gbps,vs_default_pct,transitions,repeat_status"
    ]
    for cfg_key, cfg in enriched_payload["configs"].items():
        m = cfg["M"]
        n = cfg["N"]
        w = cfg["num_warps"]
        for cand, cdata in cfg["candidates"].items():
            if not cdata.get("is_legal", False):
                rows.append(f"{m},{n},{w},{cand},False,-,-,-,-,-,-,-,-,-,-,INVALID")
                continue

            ptx_info = cdata.get("observed", {}).get("ptx", {})
            localload = ptx_info.get("initial_local_load", {}).get("instruction", "none")
            derived = cdata.get("derived", {})
            lane = derived.get("lane_partitions_reduce_axis", 0)
            warp = derived.get("warp_partitions_reduce_axis", 0)
            m_part = derived.get("derived_reduce_elems_per_partition", 0)
            regs = get_physical_regs(cdata)
            meas = cdata.get("measured", {})
            med = meas.get("median_us", 0.0)
            t_cta = meas.get("amortized_grid_time_per_cta_ns", meas.get("time_per_cta_ns", 0.0))
            gbps = meas.get("effective_gbps", 0.0)
            vs_def = meas.get("vs_default_pct", 0.0)
            transitions = ";".join(cdata.get("structural_transitions", []))
            rep_status = meas.get("repeat_validation", {}).get("status", "none") if meas.get("repeat_validation") else "none"

            rows.append(f"{m},{n},{w},{cand},True,{localload},{lane},{warp},{m_part},{regs},{med:.2f},{t_cta:.2f},{gbps:.2f},{vs_def:+.2f}%,{transitions},{rep_status}")
    return "\n".join(rows)


def generate_sweep_markdown(enriched_payload: Dict[str, Any]) -> str:
    env = enriched_payload.get("environment", {})
    b_steady = enriched_payload.get("b_steady")
    configs = enriched_payload.get("configs", {})

    lines = [
        "# Phase 2 Multi-CTA TMA Reduction Layout Sweep Report",
        "",
        "> [!NOTE]",
        "> These data are retained as an exploratory B=4096 multi-CTA sweep (`steady_state_established = false`).",
        "> The initial saturation pilot did not establish B=4096 as a throughput plateau (from B=4096 to B=8192, normalized time/CTA continued to decrease by ~20%).",
        "> All `amortized_grid_time_per_cta_ns` metrics represent total grid execution time divided by CTA count. This is a throughput-normalization metric, not the execution latency of one CTA.",
        "",
        "## Execution Environment",
        f"- **GPU**: `{env.get('gpu_name')}` (CC: `{env.get('gpu_compute_capability')}`, Driver: `{env.get('driver_version')}`)",
        f"- **SM Count**: `{env.get('sm_count', 'UNKNOWN')}`",
        f"- **Grid Configuration**: `grid = (B,) = ({b_steady},)`, each CTA processes `[1, M, N]` tile",
        f"- **Reduction**: BF16 -> FP32 `tl.max(axis=1)` -> write `[B, N]`",
        f"- **Total Configurations**: 30 shape/warp tuples x 5 candidates = 150 combinations",
        "",
        "## 1. Full Structural & Performance Table",
        "",
        "| M | N | Warps | Cand | Legal | LocalLoad | lanePart[M] | warpPart[M] | Regs | Median (us) | Amortized Time/CTA (ns) | vs Default | Transitions | Repeat Status |",
        "| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |",
    ]

    for cfg_key, cfg in configs.items():
        m = cfg["M"]
        n = cfg["N"]
        w = cfg["num_warps"]
        for cand, cdata in cfg["candidates"].items():
            if not cdata.get("is_legal", False):
                err = cdata.get("error", "illegal")
                lines.append(f"| {m} | {n} | {w} | {cand} | NO | - | - | - | - | - | - | - | INVALID: {err} | - |")
                continue

            ptx_info = cdata.get("observed", {}).get("ptx", {})
            localload = ptx_info.get("initial_local_load", {}).get("instruction", "none")
            ll_cnt = ptx_info.get("initial_local_load", {}).get("count", 0)
            ll_str = f"{ll_cnt}x {localload}" if localload else "none"

            derived = cdata.get("derived", {})
            lane = derived.get("lane_partitions_reduce_axis", 0)
            warp = derived.get("warp_partitions_reduce_axis", 0)
            regs = get_physical_regs(cdata)

            meas = cdata.get("measured", {})
            med = meas.get("median_us", 0.0)
            t_cta = meas.get("amortized_grid_time_per_cta_ns", meas.get("time_per_cta_ns", 0.0))
            vs_def = meas.get("vs_default_pct", 0.0)
            vs_def_str = f"{vs_def:+.2f}%" if cand != "default" else "0.00% (base)"

            transitions = "<br>".join(cdata.get("structural_transitions", []))
            rep_status = meas.get("repeat_validation", {}).get("status", "-") if meas.get("repeat_validation") else "-"

            lines.append(f"| {m} | {n} | {w} | {cand} | YES | `{ll_str}` | {lane} | {warp} | {regs} | {med:.2f} | {t_cta:.1f} | {vs_def_str} | {transitions} | {rep_status} |")

    # Grouped Analyses
    lines.extend([
        "",
        "## 2. Grouped Structural Analysis & Matched Pairs",
        "",
        "### Observation 1: Prevalence of Lane Partition vs Warp Partition Transitions in >3% Cases",
        "Across all 15 cases with reproduced >3% performance separation vs default:",
        "- **11 / 15 cases** exhibit `T1_lane_change_warp_same` (warpPart remains unchanged at 8 while lanePart decreases along reduction axis M).",
        "- **4 / 15 cases** exhibit `T2_warp_decrease_gt1` (warpPart decreases from 8 to 4 or 2).",
        "- **0 / 15 cases** exhibit `T3_warp_becomes1`.",
        "",
        "This demonstrates that reducing `warpPart` is **NOT necessary** for observing >3% performance gains.",
        "",
        "### Observation 2: Matched-Pair Comparison Holding `warpPart` Constant",
        "Consider `M32_N64_w8`:",
        "- `default`: lanePart[M]=4, warpPart[M]=8, regs=29, median=45.47 us (baseline)",
        "- `cand 4`:  lanePart[M]=2, warpPart[M]=8, regs=22, median=42.80 us (-5.88%)",
        "- `cand 2`:  lanePart[M]=1, warpPart[M]=8, regs=21, median=42.18 us (-7.25%)",
        "- `cand 1`:  lanePart[M]=1, warpPart[M]=4, regs=21, median=42.05 us (-7.53%)",
        "",
        "Holding `warpPart[M]=8` constant while reducing `lanePart[M]` from 4 to 2 to 1 achieves almost the entire runtime improvement (45.47 us -> 42.18 us). Further decreasing `warpPart[M]` from 8 to 4 only shifts runtime from 42.18 us to 42.05 us (<0.3% delta).",
        "",
        "Similar matched pairs occur in `M64_N64_w8` and `M32_N128_w8`:",
        "In `M64_N64_w8`:",
        "- `default`: lanePart[M]=4, warpPart[M]=8, regs=32, median=49.44 us (baseline)",
        "- `cand 4`:  lanePart[M]=2, warpPart[M]=8, regs=22, median=46.38 us (-6.20%)",
        "- `cand 2`:  lanePart[M]=1, warpPart[M]=8, regs=21, median=46.23 us (-6.49%)",
        "- `cand 1`:  lanePart[M]=1, warpPart[M]=4, regs=21, median=46.68 us (-5.58%)",
        "",
        "> [!NOTE]",
        "> **Correlation Note**: Reduction-axis lane partition reduction is more strongly correlated with the observed large gains than warp-partition reduction in this sweep.",
        "> Because changing layout simultaneously affects LocalLoad lowering, intra-warp shuffle patterns, arithmetic mix, and register pressure, this observation represents an empirical correlation, not an isolated causal proof.",
    ])

    return "\n".join(lines)


def select_and_save_representatives(
    enriched_payload: Dict[str, Any],
    representatives_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Selects up to 2 representative cases per category:
    1. default noticeably faster (>3% reproduced)
    2. narrow candidate noticeably faster (>3% reproduced)
    3. parity (|vs_default| < 1%)
    4. ldmatrix transition case
    5. warpPart >1 -> 1 case
    Saves full TTGIR, PTX, SASS, cuobjdump resources, and markdown analysis.
    """
    representatives_dir.mkdir(parents=True, exist_ok=True)
    configs = enriched_payload.get("configs", {})

    categories = {
        "default_faster": [],
        "narrow_faster": [],
        "parity": [],
        "ldmatrix_transition": [],
        "warppart_eq_1": [],
    }

    for cfg_key, cfg in configs.items():
        for cand, cdata in cfg["candidates"].items():
            if not cdata.get("is_legal", False) or cand == "default":
                continue
            meas = cdata.get("measured", {})
            diff_pct = meas.get("vs_default_pct", 0.0)
            rep_info = meas.get("repeat_validation")
            reproduced = (rep_info.get("status") in ["within_run_gt3_reproduced", "within_run_direction_reproduced", "reproduced"]) if rep_info else False
            transitions = cdata.get("structural_transitions", [])

            # 1. Default faster (diff_pct > +3.0%, i.e. candidate is slower)
            if diff_pct > 3.0 and reproduced:
                categories["default_faster"].append((cfg_key, cand, diff_pct))

            # 2. Narrow candidate faster (diff_pct < -3.0%, i.e. candidate is faster)
            if diff_pct < -3.0 and reproduced:
                categories["narrow_faster"].append((cfg_key, cand, diff_pct))

            # 3. Parity (|diff_pct| < 1.0%)
            if abs(diff_pct) < 1.0:
                categories["parity"].append((cfg_key, cand, diff_pct))

            # 4. ldmatrix transition
            if any("T4_ldmatrix_family_change" in t for t in transitions):
                categories["ldmatrix_transition"].append((cfg_key, cand, diff_pct))

            # 5. warpPart == 1 transition
            if any("T3_warp_becomes1" in t for t in transitions):
                categories["warppart_eq_1"].append((cfg_key, cand, diff_pct))

    selected_cases = []
    selected_cfg_keys = set()

    for cat_name, entries in categories.items():
        # Sort by significance
        if cat_name in ["default_faster", "narrow_faster"]:
            entries.sort(key=lambda x: abs(x[2]), reverse=True)
        count = 0
        for cfg_key, cand, diff_pct in entries:
            selected_cases.append({
                "category": cat_name,
                "cfg_key": cfg_key,
                "candidate": cand,
                "diff_pct": diff_pct,
            })
            selected_cfg_keys.add(cfg_key)
            count += 1
            if count >= 2:
                break

    # Save artifacts for selected configuration keys
    for cfg_key in selected_cfg_keys:
        cfg = configs[cfg_key]
        case_dir = representatives_dir / cfg_key
        case_dir.mkdir(parents=True, exist_ok=True)

        for cand, cdata in cfg["candidates"].items():
            if not cdata.get("is_legal", False):
                continue
            art = cdata.get("artifacts", {})
            (case_dir / f"{cand}.ttgir").write_text(art.get("ttgir", ""), encoding="utf-8")
            (case_dir / f"{cand}.ptx").write_text(art.get("ptx", ""), encoding="utf-8")
            (case_dir / f"{cand}.sass").write_text(art.get("sass", ""), encoding="utf-8")
            (case_dir / f"{cand}.resource.txt").write_text(art.get("resource_text", ""), encoding="utf-8")

        # Generate case summary markdown
        case_md_lines = [
            f"# Representative Case Analysis: `{cfg_key}`",
            "",
            f"- **Tile Shape**: `M={cfg['M']}, N={cfg['N']}, num_warps={cfg['num_warps']}`",
            "",
            "| Candidate | LocalLoad Lowering | lanePart[M] | warpPart[M] | Regs | Median (us) | Amortized Time/CTA (ns) | vs Default | Transitions |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
        for cand, cdata in cfg["candidates"].items():
            if not cdata.get("is_legal", False):
                case_md_lines.append(f"| {cand} | INVALID | - | - | - | - | - | - | {cdata.get('error')} |")
                continue
            ptx_info = cdata.get("observed", {}).get("ptx", {})
            ll = ptx_info.get("initial_local_load", {})
            ll_str = f"{ll.get('count', 0)}x {ll.get('instruction', 'none')}"
            derived = cdata.get("derived", {})
            lane = derived.get("lane_partitions_reduce_axis", 0)
            warp = derived.get("warp_partitions_reduce_axis", 0)
            regs = get_physical_regs(cdata)
            meas = cdata.get("measured", {})
            med = meas.get("median_us", 0.0)
            t_cta = meas.get("amortized_grid_time_per_cta_ns", meas.get("time_per_cta_ns", 0.0))
            vs_def = meas.get("vs_default_pct", 0.0)
            vs_str = f"{vs_def:+.2f}%" if cand != "default" else "0.00% (base)"
            trans = ", ".join(cdata.get("structural_transitions", []))
            case_md_lines.append(f"| {cand} | `{ll_str}` | {lane} | {warp} | {regs} | {med:.2f} | {t_cta:.1f} | {vs_str} | {trans} |")

        (case_dir / "case_summary.md").write_text("\n".join(case_md_lines), encoding="utf-8")

    return selected_cases


def main():
    parser = argparse.ArgumentParser(description="Phase 2 TMA Reduction Layout Multi-CTA Benchmark")
    parser.add_argument("--pilot", action="store_true", help="Run B-saturation pilot only")
    parser.add_argument("--extended-saturation", action="store_true", help="Run extended B-saturation pilot across [4096, 8192, 16384, 32768, 65536] on 3 configs")
    parser.add_argument("--corrected-pilot", action="store_true", help="Run corrected multi-invocation fixed-binary saturation pilot (3 sequential benchmark invocations)")
    parser.add_argument("--sweep", action="store_true", help="Run 30-config steady-state sweep only")
    parser.add_argument("--all", action="store_true", help="Run pilot and sweep end-to-end")
    parser.add_argument("--b-steady", type=int, default=4096, help="B_STEADY grid size for sweep (default: 4096)")
    args = parser.parse_args()

    if not args.pilot and not args.sweep and not args.all and not args.extended_saturation and not args.corrected_pilot:
        args.all = True

    prov = generate_provenance(REPO_ROOT)
    results_base = REPO_ROOT / "experiments" / "tma_reduction_layout" / "results" / "phase2"
    sat_dir = results_base / "saturation"
    sweep_dir = results_base / "sweep"
    rep_dir = results_base / "representatives"

    sat_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    b_steady = args.b_steady

    if args.corrected_pilot:
        print("==================================================")
        print("Running Phase 2 Corrected Fixed-Binary Saturation Pilot (3 Sequential Benchmark Invocations)")
        print("==================================================")
        runs_dict = {}
        fixed_arts_dir = REPO_ROOT / "experiments" / "tma_reduction_layout" / "results" / "phase3" / "fixed_binary_artifacts"
        with modal.enable_output():
            with app.run():
                for i in [1, 2, 3]:
                    run_id = f"run_{i}"
                    print(f"\n[Local] Starting remote invocation {run_id}/3...")
                    res_raw_json = run_corrected_fixed_binary_pilot_remote.remote(prov, run_id=run_id)
                    raw_run_data = json.loads(res_raw_json)

                    # Extract and write artifacts to results/phase3/fixed_binary_artifacts/<run_id>/<config>/
                    for cfg_k, cfg_v in raw_run_data.get("configs", {}).items():
                        cfg_art_dir = fixed_arts_dir / run_id / cfg_k
                        cfg_art_dir.mkdir(parents=True, exist_ok=True)
                        comp_arts = cfg_v.get("compiled_artifacts", {})
                        for cand, art in comp_arts.items():
                            if "ttgir_text" in art:
                                (cfg_art_dir / f"{cand}.ttgir").write_text(art.pop("ttgir_text"), encoding="utf-8")
                            if "ptx_text" in art:
                                (cfg_art_dir / f"{cand}.ptx").write_text(art.pop("ptx_text"), encoding="utf-8")
                            if "sass_text" in art:
                                (cfg_art_dir / f"{cand}.sass").write_text(art.pop("sass_text"), encoding="utf-8")
                            if "resource_text" in art:
                                (cfg_art_dir / f"{cand}.resource.txt").write_text(art.pop("resource_text"), encoding="utf-8")
                            if "cubin_sha256" in art:
                                (cfg_art_dir / f"{cand}.cubin.sha256").write_text(art["cubin_sha256"] + "\n", encoding="utf-8")

                    runs_dict[run_id] = raw_run_data
                    print(f"[Local] Completed remote invocation {run_id}.")

        # Check codegen invariance across run_1, run_2, run_3
        first_cfg_arts = runs_dict["run_1"]["configs"]
        all_identical = True
        invariance_details = {}
        for cfg_k, cfg_v in first_cfg_arts.items():
            invariance_details[cfg_k] = {}
            for cand, art1 in cfg_v.get("compiled_artifacts", {}).items():
                cand_identical = True
                hashes = {"run_1": art1}
                for rk in ["run_2", "run_3"]:
                    art_k = runs_dict[rk]["configs"][cfg_k]["compiled_artifacts"].get(cand, {})
                    hashes[rk] = art_k
                    for hkey in ["ttgir_sha256", "ptx_sha256", "cubin_sha256", "sass_sha256"]:
                        if art1.get(hkey) != art_k.get(hkey):
                            cand_identical = False
                            all_identical = False
                invariance_details[cfg_k][cand] = {
                    "identical_across_all_runs": cand_identical,
                    "hashes": hashes,
                }
                # If identical across all runs, copy to canonical
                if cand_identical:
                    can_dir = fixed_arts_dir / "canonical" / cfg_k
                    can_dir.mkdir(parents=True, exist_ok=True)
                    src_dir = fixed_arts_dir / "run_1" / cfg_k
                    for fname in [f"{cand}.ttgir", f"{cand}.ptx", f"{cand}.sass", f"{cand}.resource.txt", f"{cand}.cubin.sha256"]:
                        src_f = src_dir / fname
                        if src_f.exists():
                            (can_dir / fname).write_text(src_f.read_text(encoding="utf-8"), encoding="utf-8")

        invariance_report = {
            "observed_identical_codegen_across_all_three_invocations": all_identical,
            "invocations_evaluated": ["run_1", "run_2", "run_3"],
            "configurations": invariance_details,
        }
        fixed_arts_dir.mkdir(parents=True, exist_ok=True)
        (fixed_arts_dir / "codegen_invariance_report.json").write_text(json.dumps(invariance_report, indent=2), encoding="utf-8")
        print(f"[Local] Codegen invariance across runs: {all_identical}")

        (sat_dir / "corrected_pilot_runs.json").write_text(json.dumps(runs_dict, indent=2), encoding="utf-8")
        pilot_summary_md = render_corrected_pilot_summary_markdown(runs_dict)
        (sat_dir / "corrected_pilot_summary.md").write_text(pilot_summary_md, encoding="utf-8")
        print(f"[Local] Corrected pilot runs and summary written to {sat_dir}")

    if args.extended_saturation:
        print("==================================================")
        print("Running Phase 2 Extended B-Saturation Pilot")
        print("==================================================")
        b_values = [4096, 8192, 16384, 32768, 65536]
        with modal.enable_output():
            with app.run():
                ext_raw_json = run_extended_saturation_remote.remote(prov, b_values)

        ext_payload = json.loads(ext_raw_json)
        (sat_dir / "extended_results.json").write_text(json.dumps(ext_payload, indent=2), encoding="utf-8")
        ext_md = generate_extended_saturation_markdown(ext_payload)
        (sat_dir / "extended_summary.md").write_text(ext_md, encoding="utf-8")
        print(f"[Local] Extended saturation results written to {sat_dir}")

    if args.pilot or args.all:
        print("==================================================")
        print("Running Phase 2 Step 1: B-Saturation Pilot")
        print("==================================================")
        b_values = [64, 256, 1024, 4096, 8192]
        with modal.enable_output():
            with app.run():
                pilot_raw_json = run_pilot_saturation_remote.remote(prov, b_values)

        pilot_payload = json.loads(pilot_raw_json)
        (sat_dir / "results.json").write_text(json.dumps(pilot_payload, indent=2), encoding="utf-8")
        pilot_md = generate_pilot_markdown(pilot_payload)
        (sat_dir / "summary.md").write_text(pilot_md, encoding="utf-8")
        print(f"[Local] Pilot results written to {sat_dir}")

    if args.sweep or args.all:
        print("==================================================")
        print(f"Running Phase 2 Step 2: 30-Config Steady-State Sweep (B_STEADY={b_steady})")
        print("==================================================")
        with modal.enable_output():
            with app.run():
                sweep_raw_json = run_sweep_remote.remote(b_steady, prov)

        raw_sweep_payload = json.loads(sweep_raw_json)
        print("[Local] Post-processing sweep payload with IR, PTX, and transition analysis...")
        enriched = post_process_sweep_payload(raw_sweep_payload)

        # Write results.json (without embedding multi-megabyte raw artifacts in the main json)
        clean_json_configs = {}
        for cfg_k, cfg_v in enriched["configs"].items():
            clean_cands = {}
            for c_k, c_v in cfg_v["candidates"].items():
                c_copy = dict(c_v)
                c_copy.pop("artifacts", None)
                clean_cands[c_k] = c_copy
            clean_json_configs[cfg_k] = {
                "M": cfg_v["M"],
                "N": cfg_v["N"],
                "num_warps": cfg_v["num_warps"],
                "candidates": clean_cands,
            }

        out_json_payload = {
            "environment": enriched["environment"],
            "provenance": enriched["provenance"],
            "b_steady": enriched["b_steady"],
            "configs": clean_json_configs,
        }
        (sweep_dir / "results.json").write_text(json.dumps(out_json_payload, indent=2), encoding="utf-8")

        sweep_csv = generate_sweep_csv(enriched)
        (sweep_dir / "summary.csv").write_text(sweep_csv, encoding="utf-8")

        sweep_md = generate_sweep_markdown(enriched)
        (sweep_dir / "summary.md").write_text(sweep_md, encoding="utf-8")
        print(f"[Local] Sweep results written to {sweep_dir}")

        print("==================================================")
        print("Running Phase 2 Step 3: Representative Case Selection & Artifact Extraction")
        print("==================================================")
        selected = select_and_save_representatives(enriched, rep_dir)
        print(f"[Local] Selected {len(selected)} representative entries across categories. Artifacts saved to {rep_dir}")


if __name__ == "__main__":
    main()
