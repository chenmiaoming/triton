#!/usr/bin/env python3
"""
Phase 3 Step E: Controlled Gluon Repeated-Reduction Timing Runner.

Executes on Modal H100 to:
1. Conduct controlled timing across:
   - PRIMARY: M32_N64_w8 (default, 4)
   - SECONDARY CONTROL: M32_N128_w4 (default, 4)
2. Across repetition counts R in {0, 1, 2, 4, 8} and grid sizes B_RUN in {16384, 32768, 65536}.
3. Single binary per config x candidate (B_DESC=65536 fixed, runtime grid=(B_RUN,)).
4. 10 rounds x 10 timed samples = 100 samples per condition.
5. Deterministic 3D circular order rotation across rounds:
   - candidate order
   - R order
   - B_RUN order
6. 3 separate benchmark invocations saving raw_run_1.json, raw_run_2.json, raw_run_3.json.
7. Captures GPU UUID, driver, CUDA, pre- and post-run telemetry.
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
    timeout=900,
)
def remote_benchmark_timing(prov: dict, run_id: int) -> dict:
    import ctypes
    import hashlib
    import os
    import re
    import subprocess
    import time
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
    print(f"[Remote Run {run_id}] Benchmark starting on {env_info['gpu_name']} (UUID: {env_info.get('gpu_uuid')})")

    triton.set_allocator(
        lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda")
    )

    # Pre-run telemetry
    def get_telemetry():
        try:
            smi_out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            parts = [p.strip() for p in smi_out.split(",")]
            return {
                "pstate": parts[0] if len(parts) > 0 else "UNKNOWN",
                "sm_clock_mhz": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else parts[1] if len(parts) > 1 else "UNKNOWN",
                "memory_clock_mhz": int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else parts[2] if len(parts) > 2 else "UNKNOWN",
                "power_draw_w": float(parts[3]) if len(parts) > 3 else "UNKNOWN",
                "gpu_temperature_c": int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else parts[4] if len(parts) > 4 else "UNKNOWN",
            }
        except Exception as e:
            return {"error": str(e)}

    pre_run_telemetry = get_telemetry()
    print(f"[Remote Run {run_id}] Pre-run telemetry: {pre_run_telemetry}")

    configs = [
        {"cfg_name": "M32_N64_w8", "M": 32, "N": 64, "num_warps": 8, "b_desc": 65536, "role": "PRIMARY"},
        {"cfg_name": "M32_N128_w4", "M": 32, "N": 128, "num_warps": 4, "b_desc": 65536, "role": "SECONDARY_CONTROL"},
    ]
    candidates = ["default", "4"]
    r_values = [0, 1, 2, 4, 8]
    b_runs = [16384, 32768, 65536]

    gluon_layouts = get_canonical_gluon_layouts()
    shared_layout = get_canonical_gluon_shared_layout()

    compiled_kernels = {}
    compiled_cubin_hashes = {}
    compiled_artifacts = {}

    run_results = {
        "run_id": run_id,
        "env_info": env_info,
        "pre_run_telemetry": pre_run_telemetry,
        "configurations": {},
    }

    cfg_buffers = {}

    # Step 1: Compile each specialization ONCE (Fixed-binary protocol)
    for cfg in configs:
        cfg_name = cfg["cfg_name"]
        M = cfg["M"]
        N = cfg["N"]
        num_warps = cfg["num_warps"]
        b_desc = cfg["b_desc"]
        run_results["configurations"][cfg_name] = {
            "role": cfg["role"],
            "candidates": {},
        }

        # Fixed GPU memory buffers
        torch.manual_seed(42 + run_id)
        x_pt = torch.randn(b_desc, M, N, dtype=torch.bfloat16, device="cuda")
        out_pt = torch.empty(b_desc, dtype=torch.float32, device="cuda")
        desc = TensorDescriptor.from_tensor(x_pt, [1, M, N], shared_layout)
        cfg_buffers[cfg_name] = {
            "x_pt": x_pt,
            "out_pt": out_pt,
            "desc": desc,
        }

        for cand in candidates:
            reg_layout = gluon_layouts[cfg_name][cand]
            x_constraints, r_constraints = get_barrier_constraints(cfg_name, cand)

            # Compile once with R=1
            compiled_mod = gluon_repeated_reduction_kernel[(1,)](
                desc,
                out_pt,
                1,
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

            cubin_bytes = compiled_mod.asm.get("cubin", None)
            if cubin_bytes is None and hasattr(compiled_mod, "kernel"):
                cubin_bytes = compiled_mod.kernel
            cubin_sha256 = hashlib.sha256(cubin_bytes).hexdigest()

            # Verify CUBIN invariance with R=0 launch
            out_pt.zero_()
            mod_r0 = gluon_repeated_reduction_kernel[(1,)](
                desc,
                out_pt,
                0,
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
            cubin_r0 = mod_r0.asm.get("cubin", None)
            if cubin_r0 is None and hasattr(mod_r0, "kernel"):
                cubin_r0 = mod_r0.kernel
            assert hashlib.sha256(cubin_r0).hexdigest() == cubin_sha256, (
                f"CUBIN mismatch between R=1 and R=0 for {cfg_name} {cand}!"
            )

            # Capture disassembly & resources
            ptx_text = compiled_mod.asm["ptx"]
            ttgir_text = compiled_mod.asm["ttgir"]
            tmp_cubin = f"/tmp/timing_{cfg_name}_{cand}_run{run_id}.cubin"
            with open(tmp_cubin, "wb") as f:
                f.write(cubin_bytes)
            try:
                sass_text = subprocess.check_output(["cuobjdump", "-sass", tmp_cubin], text=True)
            except Exception:
                sass_text = subprocess.check_output(["nvdisasm", "-ndf", tmp_cubin], text=True)
            res_text = subprocess.check_output(["cuobjdump", "-res-usage", tmp_cubin], text=True)

            key = (cfg_name, cand)
            compiled_kernels[key] = (compiled_mod, reg_layout, shared_layout, x_constraints, r_constraints, num_warps, b_desc, M, N)
            compiled_cubin_hashes[key] = cubin_sha256
            compiled_artifacts[key] = {
                "ptx_text": ptx_text,
                "ttgir_text": ttgir_text,
                "sass_text": sass_text,
                "res_text": res_text,
                "cubin_sha256": cubin_sha256,
            }
            print(f"[Remote Run {run_id}] Compiled {cfg_name} {cand}: CUBIN {cubin_sha256[:12]}...")

    # Step 2: Warmup across all conditions
    print(f"[Remote Run {run_id}] Performing warmup...")
    for cfg in configs:
        cfg_name = cfg["cfg_name"]
        desc = cfg_buffers[cfg_name]["desc"]
        out_pt = cfg_buffers[cfg_name]["out_pt"]
        for cand in candidates:
            k_tuple = compiled_kernels[(cfg_name, cand)]
            mod, reg_l, sh_l, xc, rc, nw, bd, M, N = k_tuple
            for R in [0, 4]:
                for b_run in [16384, 65536]:
                    for _ in range(3):
                        gluon_repeated_reduction_kernel[(b_run,)](
                            desc, out_pt, R,
                            register_layout=reg_l, shared_layout=sh_l,
                            B_DESC=bd, M=M, N=N,
                            X_CONSTRAINTS=xc, R_CONSTRAINTS=rc,
                            num_warps=nw,
                        )
    torch.cuda.synchronize()

    # Step 3: Rotated Timing Rounds (10 rounds x 10 samples = 100 samples per condition)
    print(f"[Remote Run {run_id}] Starting 10-round rotated timing...")
    num_rounds = 10
    iters_per_round = 10

    # raw_samples[(cfg_name, cand, R, b_run)] = list of 100 us values
    raw_samples = {}
    for cfg in configs:
        cfg_name = cfg["cfg_name"]
        for cand in candidates:
            for R in r_values:
                for b_run in b_runs:
                    raw_samples[(cfg_name, cand, R, b_run)] = []

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    rotation_log = []

    for round_idx in range(num_rounds):
        for cfg in configs:
            cfg_name = cfg["cfg_name"]
            desc = cfg_buffers[cfg_name]["desc"]
            out_pt = cfg_buffers[cfg_name]["out_pt"]

            # Deterministic candidate rotation
            c_shift = round_idx % len(candidates)
            c_order = candidates[c_shift:] + candidates[:c_shift]

            for c_idx, cand in enumerate(c_order):
                k_tuple = compiled_kernels[(cfg_name, cand)]
                mod, reg_l, sh_l, xc, rc, nw, bd, M, N = k_tuple

                # Deterministic R rotation
                r_shift = (round_idx + c_idx) % len(r_values)
                r_order = r_values[r_shift:] + r_values[:r_shift]

                for r_idx_inner, R in enumerate(r_order):
                    # Deterministic B_RUN rotation
                    b_shift = (round_idx + c_idx + r_idx_inner) % len(b_runs)
                    b_order = b_runs[b_shift:] + b_runs[:b_shift]

                    for b_run in b_order:
                        rotation_log.append({
                            "round": round_idx,
                            "cfg": cfg_name,
                            "cand": cand,
                            "R": R,
                            "B": b_run,
                        })

                        # Timed iterations
                        for _ in range(iters_per_round):
                            start_event.record()
                            gluon_repeated_reduction_kernel[(b_run,)](
                                desc, out_pt, R,
                                register_layout=reg_l, shared_layout=sh_l,
                                B_DESC=bd, M=M, N=N,
                                X_CONSTRAINTS=xc, R_CONSTRAINTS=rc,
                                num_warps=nw,
                            )
                            end_event.record()
                            torch.cuda.synchronize()
                            us = start_event.elapsed_time(end_event) * 1000.0
                            raw_samples[(cfg_name, cand, R, b_run)].append(us)

        print(f"[Remote Run {run_id}] Finished round {round_idx + 1}/{num_rounds}")

    post_run_telemetry = get_telemetry()
    print(f"[Remote Run {run_id}] Post-run telemetry: {post_run_telemetry}")

    # Step 4: Package results
    for cfg in configs:
        cfg_name = cfg["cfg_name"]
        for cand in candidates:
            cand_entry = {
                "cubin_sha256": compiled_cubin_hashes[(cfg_name, cand)],
                "r_timing": {},
                "artifacts": compiled_artifacts[(cfg_name, cand)],
            }
            for R in r_values:
                cand_entry["r_timing"][str(R)] = {}
                for b_run in b_runs:
                    samples = raw_samples[(cfg_name, cand, R, b_run)]
                    assert len(samples) == 100, f"Expected 100 samples, got {len(samples)}"
                    cand_entry["r_timing"][str(R)][str(b_run)] = {
                        "samples_us": samples,
                    }
            run_results["configurations"][cfg_name]["candidates"][cand] = cand_entry

    run_results["post_run_telemetry"] = post_run_telemetry
    run_results["rotation_log_length"] = len(rotation_log)
    return run_results


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description="Phase 3 Step E Gluon Repeated-Reduction Timing Runner")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark invocations (default: 3)")
    args = parser.parse_args()

    print("=" * 60)
    print("Launching Phase 3 Step E: Controlled Gluon Repeated-Reduction Timing")
    print(f"Target: {args.runs} separate benchmark invocations on Modal H100")
    print("=" * 60)

    prov = generate_provenance(repo_root)

    timing_dir = repo_root / "experiments" / "tma_reduction_layout" / "results" / "phase3" / "gluon_timing"
    timing_dir.mkdir(parents=True, exist_ok=True)
    arts_dir = timing_dir / "artifacts"
    arts_dir.mkdir(parents=True, exist_ok=True)

    with app.run():
        for run_id in range(1, args.runs + 1):
            print(f"\n>>> Executing Benchmark Invocation {run_id}/{args.runs} on Modal H100...")
            t0 = time.time()
            res = remote_benchmark_timing.remote(prov, run_id)
            elapsed = time.time() - t0
            print(f">>> Invocation {run_id} completed in {elapsed:.1f}s")

            # Save artifacts on run 1
            if run_id == 1:
                for cfg_name, c_data in res["configurations"].items():
                    c_art_dir = arts_dir / cfg_name
                    c_art_dir.mkdir(parents=True, exist_ok=True)
                    for cand, cand_entry in c_data["candidates"].items():
                        arts = cand_entry["artifacts"]
                        (c_art_dir / f"{cand}.ptx").write_text(arts["ptx_text"], encoding="utf-8")
                        (c_art_dir / f"{cand}.ttgir").write_text(arts["ttgir_text"], encoding="utf-8")
                        (c_art_dir / f"{cand}.sass").write_text(arts["sass_text"], encoding="utf-8")
                        (c_art_dir / f"{cand}.resource.txt").write_text(arts["res_text"], encoding="utf-8")
                        (c_art_dir / f"{cand}.cubin.sha256").write_text(arts["cubin_sha256"], encoding="utf-8")

            # Remove bulky texts from raw_run JSON
            for cfg_name, c_data in res["configurations"].items():
                for cand, cand_entry in c_data["candidates"].items():
                    if "artifacts" in cand_entry:
                        cand_entry["artifacts"] = {
                            "cubin_sha256": cand_entry["artifacts"]["cubin_sha256"],
                            "ptx": str((arts_dir / cfg_name / f"{cand}.ptx").relative_to(repo_root)),
                            "ttgir": str((arts_dir / cfg_name / f"{cand}.ttgir").relative_to(repo_root)),
                            "sass": str((arts_dir / cfg_name / f"{cand}.sass").relative_to(repo_root)),
                            "resource": str((arts_dir / cfg_name / f"{cand}.resource.txt").relative_to(repo_root)),
                        }

            raw_file = timing_dir / f"raw_run_{run_id}.json"
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Saved: {raw_file}")

    print("\nAll benchmark invocations finished successfully!")


if __name__ == "__main__":
    main()
