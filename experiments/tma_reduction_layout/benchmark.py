"""
TMA Reduction Layout Benchmark for H100 (SM90).

Study Case:
  shape = [1, 32, 128]
  input dtype = bf16
  acc / conversion = fp32
  reduce op = max
  reduce axis = 1
  num_warps = 4
  GPU = strict H100 (SM90)

Candidates:
  default, 8, 4, 2, 1
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

repo_root_candidate = Path(__file__).resolve().parent.parent.parent
for p in ["/opt/triton-src", str(repo_root_candidate)]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import modal
from experiments.tma_reduction_layout.analyze_ir import (
    analyze_ptx,
    analyze_ttgir,
    compute_derived_layout_metrics,
)
from experiments.tma_reduction_layout.modal_runner import (
    app,
    remote_verify_environment,
    triton_image,
)
from experiments.tma_reduction_layout.source_provenance import (
    generate_provenance,
    get_repo_root,
)

REPO_ROOT = get_repo_root()


# ---------------------------------------------------------------------------
# Remote GPU Benchmark Function (Runs on strict H100)
# ---------------------------------------------------------------------------
@app.function(
    image=triton_image,
    gpu="H100!:1",
    timeout=1200,
)
def run_tma_benchmark_remote(
    candidates: List[str],
    local_provenance: Dict[str, Any],
) -> str:
    import torch
    import triton
    import triton.language as tl

    # 1. Hardware & Triton provenance verification
    env_info = remote_verify_environment()

    # Set scratch allocator required for TMA descriptors
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))

    # 2. Benchmark parameters
    B, M, N = 1, 32, 128
    num_warps = 4
    reduce_axis = 1

    torch.manual_seed(42)
    # Fixed input tensor for all candidates
    input_tensor = torch.randn((B, M, N), device="cuda", dtype=torch.bfloat16)
    out_tensor = torch.empty((N,), device="cuda", dtype=torch.float32)

    # Reference PyTorch computation
    ref_out = input_tensor.to(torch.float32).amax(dim=1).squeeze(0)

    stride_b = M * N
    stride_m = N

    results_by_candidate: Dict[str, Any] = {}

    # Define the kernel generator
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
            desc = tl.make_tensor_descriptor(
                a_ptr,
                shape=[B, M, N],
                strides=[stride_b, stride_m, 1],
                block_shape=[B, M, N],
            )
            x = desc.load([0, 0, 0])
            x_fp32 = x.to(tl.float32)
            y = tl.max(x_fp32, axis=1)  # shape: [1, 128]
            offs_n = tl.arange(0, N)
            tl.store(out_ptr + offs_n, tl.reshape(y, [N]))

        return kernel

    # Compile and evaluate each candidate
    compiled_kernels = {}
    for cand in candidates:
        print(f"\n[Remote] === Preparing candidate: {cand} ===")
        # Set experiment environment variable
        os.environ["TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT"] = cand
        os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{cand}"

        kernel = get_kernel()

        # Warmup / compilation
        is_legal = True
        compile_err = None
        compiled = None
        try:
            compiled = kernel.warmup(
                input_tensor,
                out_tensor,
                stride_b,
                stride_m,
                B,
                M,
                N,
                grid=(1,),
                num_warps=num_warps,
            )
        except Exception as e:
            is_legal = False
            compile_err = str(e)
            print(f"[Remote] Candidate {cand} failed compilation/validity: {e}")

        if not is_legal:
            results_by_candidate[cand] = {
                "candidate": cand,
                "is_legal": False,
                "error": compile_err,
            }
            continue

        compiled_kernels[cand] = (kernel, compiled)

        # Extract IR and ASM
        ttgir_text = compiled.asm.get("ttgir", "")
        ptx_text = compiled.asm.get("ptx", "")
        llir_text = compiled.asm.get("llir", "")
        sass_text = ""
        try:
            sass_text = compiled.asm.get("sass", "")
        except Exception as e:
            sass_text = f"SASS error: {e}"

        n_regs = getattr(compiled.metadata, "n_regs", None)
        shared_bytes = getattr(compiled.metadata, "shared", None)

        # Correctness check
        out_tensor.zero_()
        kernel[(1,)](
            input_tensor,
            out_tensor,
            stride_b,
            stride_m,
            B,
            M,
            N,
            num_warps=num_warps,
        )
        torch.cuda.synchronize()

        max_abs_diff = float(torch.max(torch.abs(out_tensor - ref_out)).item())
        is_correct = max_abs_diff < 1e-3
        print(f"[Remote] Candidate {cand}: correctness = {is_correct} (max abs diff: {max_abs_diff})")

        results_by_candidate[cand] = {
            "candidate": cand,
            "is_legal": True,
            "is_correct": is_correct,
            "max_abs_diff": max_abs_diff,
            "n_regs": n_regs,
            "shared_bytes": shared_bytes,
            "ttgir": ttgir_text,
            "ptx": ptx_text,
            "llir": llir_text,
            "sass": sass_text,
        }

    # Paired timing measurements to avoid thermal/warm-cache bias
    legal_cands = [c for c in candidates if results_by_candidate[c].get("is_legal", False)]
    print(f"\n[Remote] === Running paired timing measurements for legal candidates: {legal_cands} ===")

    # Warmup all legal kernels
    for _ in range(50):
        for cand in legal_cands:
            k, _ = compiled_kernels[cand]
            k[(1,)](input_tensor, out_tensor, stride_b, stride_m, B, M, N, num_warps=num_warps)
    torch.cuda.synchronize()

    # Timing: Run multiple interleaved blocks
    # 10 blocks of 20 iterations each = 200 raw samples per candidate
    cand_samples_us: Dict[str, List[float]] = {c: [] for c in legal_cands}
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    num_blocks = 10
    iters_per_block = 20

    for b in range(num_blocks):
        for cand in legal_cands:
            k, _ = compiled_kernels[cand]
            for _ in range(iters_per_block):
                start_event.record()
                k[(1,)](input_tensor, out_tensor, stride_b, stride_m, B, M, N, num_warps=num_warps)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_us = start_event.elapsed_time(end_event) * 1000.0
                cand_samples_us[cand].append(elapsed_us)

    # Compute timing statistics
    for cand in legal_cands:
        samples = cand_samples_us[cand]
        med = statistics.median(samples)
        mean = statistics.mean(samples)
        stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
        results_by_candidate[cand]["raw_samples_us"] = samples
        results_by_candidate[cand]["median_us"] = med
        results_by_candidate[cand]["mean_us"] = mean
        results_by_candidate[cand]["stdev_us"] = stdev
        results_by_candidate[cand]["min_us"] = min(samples)
        results_by_candidate[cand]["max_us"] = max(samples)
        print(f"[Remote] Candidate {cand}: median = {med:.2f} us, mean = {mean:.2f} us, min = {min(samples):.2f} us")

    final_payload = {
        "status": "PASS",
        "study_case": {
            "shape": [B, M, N],
            "input_dtype": "bf16",
            "acc_dtype": "fp32",
            "op": "max",
            "reduce_axis": reduce_axis,
            "num_warps": num_warps,
        },
        "environment": env_info,
        "local_provenance": local_provenance,
        "results": results_by_candidate,
    }

    return json.dumps(final_payload)


# ---------------------------------------------------------------------------
# Local Execution & Full Result Table Generation
# ---------------------------------------------------------------------------
def execute_benchmark() -> Dict[str, Any]:
    print("==================================================")
    print("TMA Reduction Layout Benchmark on H100 (SM90)")
    print("==================================================")

    # 1. Local provenance
    prov = generate_provenance(REPO_ROOT)
    print(f"[Local] Git HEAD: {prov['git_head_sha']} (branch: {prov['branch']})")
    print(f"[Local] Dirty status: {prov['is_dirty']} (diff sha256: {prov['git_diff_head_sha256']})")

    candidates = ["default", "8", "4", "2", "1"]
    print(f"[Local] Candidates to evaluate: {candidates}")

    # 2. Dispatch to Modal H100
    print("[Local] Dispatching benchmark to Modal H100 (gpu='H100!:1')...")
    with modal.enable_output():
        with app.run():
            raw_json = run_tma_benchmark_remote.remote(candidates, prov)

    payload = json.loads(raw_json)

    # 3. Post-process with analyze_ir
    results_dir = REPO_ROOT / "experiments" / "tma_reduction_layout" / "results"
    artifacts_dir = results_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for cand, res in payload["results"].items():
        if not res.get("is_legal", False):
            continue

        # Save artifacts
        ttgir_path = artifacts_dir / f"{cand}.ttgir"
        ptx_path = artifacts_dir / f"{cand}.ptx"
        ttgir_path.write_text(res.get("ttgir", ""))
        ptx_path.write_text(res.get("ptx", ""))

        if res.get("sass"):
            (artifacts_dir / f"{cand}.sass").write_text(res["sass"])

        # Analyze TTGIR & PTX
        ttgir_info = analyze_ttgir(res.get("ttgir", ""))
        ptx_info = analyze_ptx(res.get("ptx", ""))

        res["ttgir_analysis"] = ttgir_info
        res["ptx_analysis"] = ptx_info

        # Extract Blocked layout attributes
        blocked_encs = ttgir_info.get("blocked_encodings", [])
        # The main input layout is usually the first blocked encoding
        main_enc = blocked_encs[0] if blocked_encs else {}

        spt = main_enc.get("sizePerThread", [1, 1, 1])
        tpw = main_enc.get("threadsPerWarp", [1, 1, 32])
        wpc = main_enc.get("warpsPerCTA", [1, 1, 4])
        order = main_enc.get("order", [2, 1, 0])
        cga = main_enc.get("CTAsPerCGA", [1, 1, 1])

        derived = compute_derived_layout_metrics(
            shape=[1, 32, 128],
            size_per_thread=spt,
            threads_per_warp=tpw,
            warps_per_cta=wpc,
            ctas_per_cga=cga,
            reduce_axis=1,
        )
        res["derived_metrics"] = derived

        # Determine shared memory swizzle / layout
        shared_encs = ttgir_info.get("shared_encodings", [])
        shared_desc = str(shared_encs[0]["attrs"]) if shared_encs else "none"

        summary_rows.append({
            "candidate": cand,
            "sizePerThread": str(spt),
            "threadsPerWarp": str(tpw),
            "warpsPerCTA": str(wpc),
            "CTAsPerCGA": str(cga),
            "elements_per_thread": derived["elements_per_thread"],
            "ownership_M": derived["thread_ownership_reduce_axis"],
            "ownership_N": derived["thread_ownership_contig_axis"],
            "repetitions_M": derived["repetitions_reduce_axis"],
            "repetitions_N": derived["repetitions_contig_axis"],
            "lane_partitions": derived["lane_partitions_reduce_axis"],
            "warp_partitions": derived["warp_partitions_reduce_axis"],
            "cta_partitions": derived["cta_partitions_reduce_axis"],
            "shared_encoding": shared_desc,
            "ld_shared_total": ptx_info["ld_shared_total"],
            "ld_shared_details": ptx_info["ld_shared_details"],
            "st_shared_total": ptx_info["st_shared_total"],
            "shfl_total": ptx_info["shfl_total"],
            "bar_sync_total": ptx_info["bar_sync_total"],
            "n_regs": res.get("n_regs"),
            "correctness": "PASS" if res.get("is_correct") else "FAIL",
            "median_us": f"{res.get('median_us', 0.0):.2f}",
            "mean_us": f"{res.get('mean_us', 0.0):.2f}",
        })

    # Save complete JSON
    out_json = results_dir / "baseline_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[Local] Complete benchmark results saved to: {out_json}")

    # Generate Markdown Table with annotations [A. static layout-derived], [B. observed from TTGIR/PTX], [C. measured on H100]
    md_lines = [
        "# TMA Reduction Layout Experiment Results: [1, 32, 128] BF16 -> FP32 Max Axis=1",
        "",
        f"- **Hardware**: {payload['environment']['gpu_name']} ({payload['environment']['driver_version']}, CC {payload['environment']['gpu_compute_capability']})",
        f"- **Git HEAD**: `{prov['git_head_sha'][:10]}` (branch: `{prov['branch']}`, dirty: `{prov['is_dirty']}`)",
        f"- **Triton**: `{payload['environment']['triton_version']}` (`{payload['environment']['triton_file']}`)",
        "",
        "### Characterization Table",
        "",
        "| Candidate | sizePerThread [A] | threadsPerWarp [B] | warpsPerCTA [B] | Lane Parts (M) [A] | Warp Parts (M) [A] | M Ownership [A] | ld.shared [B] | shfl.sync [B] | st.shared [B] | bar.sync [B] | Regs [B] | Correct [C] | Median (us) [C] |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for row in summary_rows:
        md_lines.append(
            f"| `{row['candidate']}` | `{row['sizePerThread']}` | `{row['threadsPerWarp']}` | `{row['warpsPerCTA']}` | "
            f"{row['lane_partitions']} | {row['warp_partitions']} | {row['ownership_M']} | "
            f"{row['ld_shared_total']} | {row['shfl_total']} | {row['st_shared_total']} | {row['bar_sync_total']} | "
            f"{row['n_regs']} | {row['correctness']} | **{row['median_us']}** |"
        )

    md_lines.extend([
        "",
        "**Legend**:",
        "- **[A. static layout-derived]**: Theoretically derived from tensor dimensions, vector size, and BlockedEncoding rules.",
        "- **[B. observed from TTGIR/PTX/SASS]**: Extracted directly from compiled IR / PTX assembly.",
        "- **[C. measured on H100]**: Empirically measured via CUDA events on strict NVIDIA H100.",
    ])

    table_md = "\n".join(md_lines)
    out_table = results_dir / "baseline_summary_table.md"
    out_table.write_text(table_md)
    print(f"[Local] Summary Markdown table saved to: {out_table}\n")
    print(table_md)

    return payload


if __name__ == "__main__":
    execute_benchmark()
