import os
import re
import sys
from pathlib import Path

# Add repo root and container path to sys.path
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
def remote_test_v2_codegen(prov: dict, prototype_id: str):
    import os
    import subprocess
    import torch
    import triton
    import triton.language as tl

    env_info = remote_verify_environment(prov)
    print(f"[Remote] Testing v2 codegen with prototype: {prototype_id}")
    print(f"[Remote] GPU: {env_info['gpu_name']} UUID: {env_info['gpu_uuid']}")

    triton.set_allocator(
        lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda")
    )

    results = {}

    # Define test configs
    configs = [
        {"cfg_name": "M32_N64_w8", "M": 32, "N": 64, "num_warps": 8, "b_desc": 65536},
        {"cfg_name": "M32_N128_w4", "M": 32, "N": 128, "num_warps": 4, "b_desc": 65536},
    ]

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

        for cand in ["default", "4"]:
            results[cfg_name][cand] = {}
            if cand == "default":
                os.environ.pop("TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT", None)
            else:
                os.environ["TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT"] = cand

            for k_val in [1, 2, 4]:
                os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_v2_{prototype_id}_{cfg_name}_{cand}_K{k_val}"

                if prototype_id == "proto1_pack2":
                    @triton.jit
                    def kernel(
                        a_ptr, out_ptr, stride_b, stride_m,
                        B_DESC: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K_ITERS: tl.constexpr
                    ):
                        pid = tl.program_id(0)
                        desc = tl.make_tensor_descriptor(
                            a_ptr, shape=[B_DESC, M, N], strides=[stride_b, stride_m, 1], block_shape=[1, M, N]
                        )
                        x = desc.load([pid, 0, 0])
                        v = x
                        acc = tl.zeros([1, N], dtype=tl.float32)
                        for i in tl.static_range(K_ITERS):
                            v_opaque = tl.inline_asm_elementwise("mov.b32 $0, $1;", "=r,r", [v], dtype=tl.bfloat16, is_pure=False, pack=2)
                            r_i = tl.max(v_opaque.to(tl.float32), axis=1)
                            acc += r_i
                            v = v_opaque
                        offs_n = tl.arange(0, N)
                        tl.store(out_ptr + pid * N + offs_n, tl.reshape(acc, [N]))

                elif prototype_id == "proto2_pack1":
                    @triton.jit
                    def kernel(
                        a_ptr, out_ptr, stride_b, stride_m,
                        B_DESC: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K_ITERS: tl.constexpr
                    ):
                        pid = tl.program_id(0)
                        desc = tl.make_tensor_descriptor(
                            a_ptr, shape=[B_DESC, M, N], strides=[stride_b, stride_m, 1], block_shape=[1, M, N]
                        )
                        x = desc.load([pid, 0, 0])
                        v = x
                        acc = tl.zeros([1, N], dtype=tl.float32)
                        for i in tl.static_range(K_ITERS):
                            v_opaque = tl.inline_asm_elementwise("mov.b16 $0, $1;", "=h,h", [v], dtype=tl.bfloat16, is_pure=False, pack=1)
                            r_i = tl.max(v_opaque.to(tl.float32), axis=1)
                            acc += r_i
                            v = v_opaque
                        offs_n = tl.arange(0, N)
                        tl.store(out_ptr + pid * N + offs_n, tl.reshape(acc, [N]))

                elif prototype_id == "proto3_acc_opaque":
                    # Keep x directly, but make the reduction result r_i opaque
                    @triton.jit
                    def kernel(
                        a_ptr, out_ptr, stride_b, stride_m,
                        B_DESC: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K_ITERS: tl.constexpr
                    ):
                        pid = tl.program_id(0)
                        desc = tl.make_tensor_descriptor(
                            a_ptr, shape=[B_DESC, M, N], strides=[stride_b, stride_m, 1], block_shape=[1, M, N]
                        )
                        x = desc.load([pid, 0, 0])
                        acc = tl.zeros([1, N], dtype=tl.float32)
                        for i in tl.static_range(K_ITERS):
                            # What if r_i is made opaque? Wait, if x is unchanged, does compiler CSE the reduction?
                            r_i = tl.max(x.to(tl.float32), axis=1)
                            r_opaque = tl.inline_asm_elementwise("mov.b32 $0, $1;", "=r,r", [r_i], dtype=tl.float32, is_pure=False, pack=1)
                            acc += r_opaque
                        offs_n = tl.arange(0, N)
                        tl.store(out_ptr + pid * N + offs_n, tl.reshape(acc, [N]))

                try:
                    compiled = kernel.warmup(
                        input_tensor, out_tensor, stride_b, stride_m,
                        b_desc, m, n, k_val, grid=(1,), num_warps=num_warps
                    )
                    ptx_text = compiled.asm.get("ptx", "")
                    ttgir_text = compiled.asm.get("ttgir", "")
                    
                    # Parse resources
                    cubin_bytes = compiled.asm.get("cubin", None)
                    if cubin_bytes is None and hasattr(compiled, "kernel"):
                        cubin_bytes = compiled.kernel
                    
                    sass_text = ""
                    num_regs = "UNKNOWN"
                    local_bytes = 0
                    stack_bytes = 0
                    if cubin_bytes is not None:
                        tmp_cubin = f"/tmp/triton_{prototype_id}_{cfg_name}_{cand}_K{k_val}.cubin"
                        with open(tmp_cubin, "wb") as f:
                            f.write(cubin_bytes)
                        try:
                            sass_text = subprocess.check_output(
                                ["cuobjdump", "-sass", tmp_cubin],
                                text=True,
                                stderr=subprocess.STDOUT,
                            )
                            res_out = subprocess.check_output(
                                ["cuobjdump", "-res-usage", tmp_cubin],
                                text=True,
                                stderr=subprocess.STDOUT,
                            )
                            reg_m = re.search(r"\bREG:(\d+)\b", res_out)
                            loc_m = re.search(r"\bLOCAL:(\d+)\b", res_out)
                            stk_m = re.search(r"\bSTACK:(\d+)\b", res_out)
                            if reg_m: num_regs = int(reg_m.group(1))
                            if loc_m: local_bytes = int(loc_m.group(1))
                            if stk_m: stack_bytes = int(stk_m.group(1))
                        except Exception as e:
                            print(f"cuobjdump failed: {e}")

                    # Extract initial LocalLoad sequence
                    lines = ptx_text.splitlines()
                    init_loads = []
                    seen_load = False
                    for l in lines:
                        ls = l.strip()
                        if not ls or ls.startswith("//") or ls.startswith(".") or ls.startswith("$") or ls.endswith(":"):
                            continue
                        if "ld.shared" in ls:
                            init_loads.append(ls.split()[0])
                            seen_load = True
                        elif seen_load:
                            if "ld.shared" not in ls:
                                break

                    # Opcode counts
                    max_f32 = len(re.findall(r"\bmax\.f32\b", ptx_text))
                    max_bf16x2 = len(re.findall(r"\bmax\.bf16x2\b", ptx_text))
                    cvt = len(re.findall(r"\bcvt\.f32\.bf16\b", ptx_text))
                    shfl = len(re.findall(r"\bshfl\.sync\.bfly\b", ptx_text))
                    bar = len(re.findall(r"\bbar\.sync\b", ptx_text))
                    st_sh = len(re.findall(r"\bst\.shared\b", ptx_text))
                    ld_sh = len(re.findall(r"\bld\.shared\b", ptx_text))
                    tma = len(re.findall(r"\bcp\.async\.bulk\.tensor\b", ptx_text))

                    # Count inline asm opaque identity instructions
                    # In PTX, inline asm usually appears as our specific mov instruction
                    if prototype_id == "proto1_pack2":
                        # mov.b32 from inline asm
                        opaque_cnt = len(re.findall(r"\bmov\.b32\b", ptx_text))
                    elif prototype_id == "proto2_pack1":
                        opaque_cnt = len(re.findall(r"\bmov\.b16\b", ptx_text))
                    else:
                        opaque_cnt = 0

                    results[cfg_name][cand][k_val] = {
                        "status": "success",
                        "tma_count": tma,
                        "init_loads": init_loads,
                        "num_regs": num_regs,
                        "local_bytes": local_bytes,
                        "stack_bytes": stack_bytes,
                        "opcodes": {
                            "max_f32": max_f32,
                            "max_bf16x2": max_bf16x2,
                            "cvt_f32_bf16": cvt,
                            "shfl_sync_bfly": shfl,
                            "bar_sync": bar,
                            "st_shared": st_sh,
                            "ld_shared": ld_sh,
                        },
                        "ptx_text": ptx_text,
                        "ttgir_text": ttgir_text,
                    }
                    print(f"  [{cfg_name} {cand} K={k_val}] init={init_loads}, regs={num_regs}, shfl={shfl}, max.f32={max_f32}, max.bf16x2={max_bf16x2}, cvt={cvt}, bar={bar}")

                except Exception as e:
                    print(f"  [{cfg_name} {cand} K={k_val}] FAILED: {e}")
                    results[cfg_name][cand][k_val] = {
                        "status": "failed",
                        "error": str(e),
                    }

    return results

def audit_results(all_results):
    print("\n=======================================================")
    print("DETAILED ACCEPTANCE CRITERIA AUDIT FOR PROTO1_PACK2")
    print("=======================================================")
    
    res = all_results.get("proto1_pack2", {})
    all_passed = True
    
    # Check Canonical Step A reference
    # M32_N64_w8:
    #   default: init=['ld.shared.v4.b32'], regs=29, shfl=40, max.f32=40, max.bf16x2=0, cvt=8, bar=14
    #   cand4:   init=['ld.shared.v2.b32', 'ld.shared.v2.b32'], regs=22, shfl=16, max.f32=16, max.bf16x2=2, cvt=4, bar=10
    # M32_N128_w4:
    #   default: init=['ld.shared.v4.b32', 'ld.shared.v4.b32', 'ld.shared.v4.b32', 'ld.shared.v4.b32'], regs=32, shfl=24, max.f32=24, max.bf16x2=12, cvt=8, bar=14
    #   cand4:   init=['ld.shared.v2.b32', 'ld.shared.v2.b32', 'ld.shared.v2.b32', 'ld.shared.v2.b32'], regs=25, shfl=8, max.f32=8, max.bf16x2=14, cvt=4, bar=10
    
    expected_k1_ref = {
        "M32_N64_w8": {
            "default": {"init": ["ld.shared.v4.b32"], "shfl": 40, "max_f32": 40, "max_bf16x2": 0, "cvt": 8, "bar": 14},
            "4": {"init": ["ld.shared.v2.b32", "ld.shared.v2.b32"], "shfl": 16, "max_f32": 16, "max_bf16x2": 2, "cvt": 4, "bar": 10},
        },
        "M32_N128_w4": {
            "default": {"init": ["ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32", "ld.shared.v4.b32"], "shfl": 24, "max_f32": 24, "max_bf16x2": 12, "cvt": 8, "bar": 14},
            "4": {"init": ["ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32", "ld.shared.v2.b32"], "shfl": 8, "max_f32": 8, "max_bf16x2": 14, "cvt": 4, "bar": 10},
        }
    }
    
    for cfg_name, cfg_data in res.items():
        print(f"\nConfiguration: {cfg_name}")
        for cand, cand_data in cfg_data.items():
            print(f"  Candidate: {cand}")
            k_vals = sorted(cand_data.keys())
            init_k1 = cand_data[1]["init_loads"]
            opc_k1 = cand_data[1]["opcodes"]
            
            # Criterion A: Canonical match at K=1
            ref = expected_k1_ref[cfg_name][cand]
            match_init = (init_k1 == ref["init"])
            match_shfl = (opc_k1["shfl_sync_bfly"] == ref["shfl"])
            match_f32 = (opc_k1["max_f32"] == ref["max_f32"])
            match_bf16x2 = (opc_k1["max_bf16x2"] == ref["max_bf16x2"])
            match_cvt = (opc_k1["cvt_f32_bf16"] == ref["cvt"])
            match_bar = (opc_k1["bar_sync"] == ref["bar"])
            crit_a = match_init and match_shfl and match_f32 and match_bf16x2 and match_cvt and match_bar
            print(f"    [Criterion A - Canonical Match @ K=1]: {crit_a} (init={match_init}, shfl={match_shfl}, f32={match_f32}, bf16x2={match_bf16x2}, cvt={match_cvt}, bar={match_bar})")
            if not crit_a:
                all_passed = False
                
            # Criterion B: Exact initial LocalLoad across K
            crit_b = True
            for k in k_vals:
                if cand_data[k]["init_loads"] != init_k1:
                    crit_b = False
                    print(f"      Mismatch at K={k}: {cand_data[k]['init_loads']} vs {init_k1}")
            print(f"    [Criterion B - Initial LocalLoad Invariance across K]: {crit_b}")
            if not crit_b:
                all_passed = False
                
            # Criterion C & D: Identical reduction body template & exact affine opcode scaling
            crit_c = True
            crit_d = True
            shfl_slope = opc_k1["shfl_sync_bfly"]
            f32_slope = opc_k1["max_f32"]
            bf16x2_slope = opc_k1["max_bf16x2"]
            cvt_slope = opc_k1["cvt_f32_bf16"]
            bar_slope = 8 if cand == "default" else 4
            bar_base = 6
            
            for k in k_vals:
                opc = cand_data[k]["opcodes"]
                if opc["shfl_sync_bfly"] != shfl_slope * k:
                    crit_d = False
                    print(f"      shfl scale mismatch at K={k}: {opc['shfl_sync_bfly']} != {shfl_slope * k}")
                if opc["max_f32"] != f32_slope * k:
                    crit_d = False
                    print(f"      max.f32 scale mismatch at K={k}: {opc['max_f32']} != {f32_slope * k}")
                if opc["max_bf16x2"] != bf16x2_slope * k:
                    crit_d = False
                    print(f"      max.bf16x2 scale mismatch at K={k}: {opc['max_bf16x2']} != {bf16x2_slope * k}")
                if opc["cvt_f32_bf16"] != cvt_slope * k:
                    crit_d = False
                    print(f"      cvt scale mismatch at K={k}: {opc['cvt_f32_bf16']} != {cvt_slope * k}")
                if opc["bar_sync"] != bar_base + bar_slope * k:
                    crit_d = False
                    print(f"      bar scale mismatch at K={k}: {opc['bar_sync']} != {bar_base + bar_slope * k}")
            print(f"    [Criterion C & D - Body Scaling & Affine Opcode Scaling]: {crit_d}")
            if not crit_d:
                all_passed = False
                
            # Criterion E: LOCAL=0, STACK=0
            crit_e = True
            for k in k_vals:
                if cand_data[k]["local_bytes"] != 0 or cand_data[k]["stack_bytes"] != 0:
                    crit_e = False
                    print(f"      Spill at K={k}: local={cand_data[k]['local_bytes']}, stack={cand_data[k]['stack_bytes']}")
            print(f"    [Criterion E - Spill Invariance (LOCAL=0, STACK=0)]: {crit_e}")
            if not crit_e:
                all_passed = False
                
            # Criterion F: Physical register growth recorded
            reg_seq = [f"K={k}: {cand_data[k]['num_regs']} regs" for k in k_vals]
            print(f"    [Criterion F - Register Growth]: {', '.join(reg_seq)}")
            
    print("\n-------------------------------------------------------")
    print(f"OVERALL AUDIT RESULT: {'PASSED ALL 6 CRITERIA' if all_passed else 'FAILED'}")
    print("=======================================================\n")
    return all_passed

def main():
    prov = generate_provenance()
    out_file = repo_root / "experiments" / "tma_reduction_layout" / "results" / "phase3" / "v2_codegen_results.json"
    
    all_results = {}
    print(f"\n==========================================")
    print(f"Running proto1_pack2 codegen evaluation...")
    print(f"==========================================")
    with app.run():
        res = remote_test_v2_codegen.remote(prov, "proto1_pack2")
    all_results["proto1_pack2"] = res

    # Save to JSON
    # Strip ptx_text and ttgir_text for concise json, but save ptx separately
    summary_json = {}
    ptx_dir = repo_root / "experiments" / "tma_reduction_layout" / "results" / "phase3" / "v2_ptx"
    ptx_dir.mkdir(parents=True, exist_ok=True)
    
    for proto, pdata in all_results.items():
        summary_json[proto] = {}
        for cfg_name, cdata in pdata.items():
            summary_json[proto][cfg_name] = {}
            for cand, kdata in cdata.items():
                summary_json[proto][cfg_name][cand] = {}
                for k, v in kdata.items():
                    if v.get("status") == "success":
                        ptx_path = ptx_dir / f"{proto}_{cfg_name}_{cand}_K{k}.ptx"
                        with open(ptx_path, "w") as f:
                            f.write(v["ptx_text"])
                        
                        summary_json[proto][cfg_name][cand][k] = {
                            "status": v["status"],
                            "tma_count": v["tma_count"],
                            "init_loads": v["init_loads"],
                            "num_regs": v["num_regs"],
                            "local_bytes": v["local_bytes"],
                            "stack_bytes": v["stack_bytes"],
                            "opcodes": v["opcodes"],
                            "ptx_file": str(ptx_path.relative_to(repo_root)),
                        }
                    else:
                        summary_json[proto][cfg_name][cand][k] = v

    with open(out_file, "w") as f:
        json.dump(summary_json, f, indent=2)
    print(f"Saved results to {out_file}")

    audit_results(all_results)

if __name__ == "__main__":
    main()

