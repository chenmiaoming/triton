"""Three strict architecture artifact gates. No GPU event timing or native edits."""
import io
import os
from pathlib import Path
import uuid
import zipfile
import modal
from experiments.tma_reduction_layout.phase9 import common as c
from experiments.tma_reduction_layout.source_provenance import (
    MODAL_SOURCE_IGNORE_PATTERNS, generate_provenance, verify_remote_source_manifest)
from experiments.tma_reduction_layout.modal_runner import build_cache_volume
from experiments.tma_reduction_layout.phase6.run_fresh_gate import core_image

LOCAL_PROVENANCE = generate_provenance() if modal.is_local() else None
app = modal.App("triton-phase9-cross-architecture-artifacts")
evidence_volume = modal.Volume.from_name("triton-phase9-evidence", create_if_missing=True)
tools_image = (core_image.add_local_file(Path(__file__).with_name("install_tools.py"),
    "/opt/phase9-install-tools.py", copy=True).run_commands("python3 /opt/phase9-install-tools.py"))
image = tools_image.env({"PYTHONPATH": "/opt/triton-src",
    "PATH": "/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"}).add_local_dir(
    c.ROOT, remote_path="/opt/triton-src", copy=False,
    ignore=MODAL_SOURCE_IGNORE_PATTERNS + ["experiments/tma_reduction_layout/phase4/results/**"])


def collect_impl(target, provenance, frozen, expected_native):
    import ast
    import json
    import re
    import subprocess
    import traceback
    import torch
    import triton
    import triton._C.libtriton as native
    from triton.backends.nvidia.compiler import get_ptxas
    from experiments.tma_reduction_layout.phase9.kernels import make_kernel
    from experiments.tma_reduction_layout.phase4.exact_cuda import ExactCUDA
    from experiments.tma_reduction_layout.phase4.artifact_gate import ttgir_contract
    from experiments.tma_reduction_layout.gluon import artifact_checks as ac

    stage = Path("/evidence/stage_b") / target / modal.current_function_call_id()
    stage.mkdir(parents=True, exist_ok=True)
    root = Path("/opt/triton-src")
    records = []
    env = {"target": target, "profile": provenance["modal_profile"],
           "dispatch_id": modal.current_function_call_id(), "no_performance_observation": True}
    def command(*args):
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)
    def save(name, data):
        c.write(stage / name, data)
    try:
        env["source_verification"] = verify_remote_source_manifest(provenance, root)
        torch.cuda.init()
        name, cc = torch.cuda.get_device_name(0), list(torch.cuda.get_device_capability(0))
        spec = c.TARGETS[target]
        c.require(spec["name_contains"] in name and cc == spec["cc"], f"Strict target mismatch: {target}: {name}, {cc}")
        cuda = ExactCUDA()
        env.update(cuda.environment())
        env.update({"gpu_name": name, "compute_capability": cc,
            "SM_count": torch.cuda.get_device_properties(0).multi_processor_count,
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "triton": triton.__version__, "triton_import": triton.__file__,
            "modal_image_id": os.environ.get("MODAL_IMAGE_ID", "UNAVAILABLE"),
            "modal_task_id": os.environ.get("MODAL_TASK_ID", "UNAVAILABLE"),
            "persistent_cache": {k: os.environ.get(k, "UNAVAILABLE") for k in ("CCACHE_DIR", "TRITON_HOME")}})
        c.require(str(triton.__file__).startswith("/opt/triton-src/"), "Wrong Triton source import")
        native_sha = c.sha(Path(native.__file__).read_bytes())
        c.require(native_sha == expected_native, "Exact Phase6/Phase8 native SHA changed")
        ptxas = Path(get_ptxas(cc[0]*10+cc[1]).path)
        env["native_extension"] = {"path": native.__file__, "SHA256": native_sha}
        env["actual_ptxas"] = {"path": str(ptxas), "SHA256": c.sha(ptxas.read_bytes()), "version": command(str(ptxas), "--version")}
        env["tools"] = {name: {"version": command(name, "--version")} for name in ("cuobjdump", "nvdisasm", "ccache")}
        env["tool_installation"] = c.read("/opt/phase9-cuda-tools/installation.json")
        env["persistent_cache_stats_start"] = command("ccache", "--print-stats")
        env["core_build_evidence"] = {p.name: p.read_text() for p in Path("/opt/phase6-build-evidence").iterdir() if p.is_file()}
        save("environment.json", env)
        save("local_source_provenance.json", provenance)
        (stage / "frozen_protocol.json").write_text(frozen)
        (stage / "official_tool_redistrib.json").write_bytes(Path("/opt/phase9-cuda-tools/redistrib_13.0.2.json").read_bytes())
        with zipfile.ZipFile(stage / "uploaded_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(provenance["source_manifest"]):
                z.writestr(path, (root / path).read_bytes())
        save("source_bindings.json", {"protocol_SHA256": c.sha(frozen.encode()),
            "uploaded_source_archive_SHA256": c.sha((stage / "uploaded_source.zip").read_bytes()),
            "source_manifest_SHA256": provenance["source_manifest_sha256"],
            "source_HEAD": provenance["git_head_sha"], "kernel_source_SHA256": c.sha((root / "experiments/tma_reduction_layout/phase9/kernels.py").read_bytes())})
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
        for case in json.loads(frozen)["cases"]:
            cfg, m, n, w = case["case_id"], case["M"], case["N"], case["num_warps"]
            x = torch.empty((c.B_DESC, m, n), dtype=torch.bfloat16, device="cuda")
            values = (((torch.arange(4*m*n, dtype=torch.int32).reshape(4,m,n) % 257)-128).float()/8).to(torch.bfloat16)
            x[:4].copy_(values)
            reference = values.float().amax(dim=1).cuda()
            for harness in case["harnesses"]:
                out = torch.empty((c.B_DESC,n) if harness == "reduction" else (c.B_DESC,m,n),
                    dtype=torch.float32 if harness == "reduction" else torch.bfloat16, device="cuda")
                for candidate in c.CANDIDATES:
                    directory = stage / cfg / harness / candidate
                    directory.mkdir(parents=True, exist_ok=True)
                    record = {"case_id": cfg, "harness": harness, "candidate": candidate,
                              "target": target, "attempted": True, "compile_calls": 0}
                    failure_stage = "COMPILE"
                    compiled = args = None
                    try:
                        os.environ["TRITON_CACHE_DIR"] = f"/tmp/phase9_cache/{target}/{cfg}/{harness}/{candidate}"
                        os.environ.pop("TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT", None)
                        if candidate != "default": os.environ["TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT"] = candidate
                        kernel = make_kernel(harness)
                        args = (x,out,m*n,n,c.B_DESC,m,n)
                        record["compile_calls"] = 1
                        compiled = kernel.warmup(*args, grid=(4,), num_warps=w)
                        failure_stage = "EXPORT"
                        for ext in ("ttir", "ttgir", "llir", "ptx", "cubin"):
                            data = compiled.asm[ext]
                            (directory / ("kernel."+ext)).write_bytes(data if isinstance(data,bytes) else data.encode())
                        digest = c.sha((directory / "kernel.cubin").read_bytes())
                        for ext, option in (("sass", "-sass"), ("resource.txt", "-res-usage")):
                            (directory / ("kernel."+ext)).write_text(command("cuobjdump",option,str(directory / "kernel.cubin")))
                        resource = ac.parse_resource((directory / "kernel.resource.txt").read_text())
                        source_rel = "experiments/tma_reduction_layout/phase4/kernels_stage_b.py" if harness == "reduction" else "experiments/tma_reduction_layout/phase9/kernels.py"
                        source = (root / source_rel).read_text()
                        fn = next(node for node in ast.walk(ast.parse(source)) if isinstance(node,ast.FunctionDef) and node.name == compiled.name)
                        phase_lines = sorted(node.lineno for node in ast.walk(fn) if isinstance(node,ast.Call)
                            and isinstance(node.func,ast.Attribute) and node.func.attr in ("to","max")) if harness == "reduction" else []
                        metadata = {"case_id": cfg, "experiment_harness": harness, "harness": "canonical",
                            "candidate": candidate, "target": target, "CC": cc, "num_warps": w,
                            "B_DESC": c.B_DESC, "logical_shape": [1,m,n], "function_name": compiled.name,
                            "dynamic_smem_bytes": compiled.metadata.shared, "resources": resource,
                            "cubin_SHA256": digest, "compile_calls": 1, "compiled_object_hash": compiled.hash,
                            "source_stage": {"source_path": source_rel,"source_SHA256": c.sha(source.encode()),"reduction_source_lines": phase_lines}}
                        c.write(directory / "metadata.json", metadata)
                        ptx = compiled.asm["ptx"]
                        observed = {"TTGIR_local_operations": [line for line in compiled.asm["ttgir"].splitlines() if "ttg.local_" in line],
                                    "PTX_target": re.search(r"\.target\s+([^\n]+)",ptx).group(1)}
                        for key, query in (("layout",lambda: ttgir_contract(compiled.asm["ttgir"],[1,m,n],w)),
                                           ("initial_localload",lambda: ac.initial_load_signature(ptx))):
                            try: observed[key] = query()
                            except (ValueError,AssertionError) as exc: observed[key] = {"unavailable": str(exc)}
                        c.write(directory / "observed_structure.json", observed)
                        failure_stage = "OCCUPANCY"
                        c.require(compiled.metadata.shared <= env["device_limits"]["max_shared_memory_per_block_optin"], "RESOURCE_UNSUPPORTED: dynamic shared memory exceeds actual target limit")
                        occupancy = cuda.occupancy(directory / "kernel.cubin",compiled.name,compiled.metadata.shared,w,
                            c.sha((directory / "kernel.resource.txt").read_bytes()))
                        occupancy["gpu_uuid"] = env["gpu_uuid"]
                        c.write(directory / "occupancy.json",occupancy)
                        c.require(occupancy["blocks_per_sm_actual_dynamic_smem"] > 0, "RESOURCE_UNSUPPORTED: zero resident blocks")
                        failure_stage = "SMOKE"
                        compiled._init_handles()
                        c.require(c.sha(compiled.kernel) == digest, "Compiled launch SHA mismatch")
                        out[:4].fill_(float("nan"))
                        compiled[(4,1,1)](*args)
                        torch.cuda.synchronize()
                        expected = reference if harness == "reduction" else x[:4]
                        error = (out[:4].float()-expected.float()).abs().max().item()
                        smoke = {"B_RUN": 4, "B_DESC": c.B_DESC, "timed": False,"prefix_initialized": 4,
                            "full_descriptor_allocation": True,"input_shape": list(x.shape),"input_bytes": x.numel()*x.element_size(),
                            "max_abs_diff": error,"passed": error == 0,"launched_cubin_SHA256": digest,
                            "after_cubin_SHA256": c.sha(compiled.kernel),"compile_calls": 1}
                        c.write(directory / "smoke.json",smoke)
                        c.require(error == 0 and smoke["after_cubin_SHA256"] == digest, "Correctness/SHA smoke failed")
                        record.update({"status": "EXPORTED", "cubin_SHA256": digest})
                    except Exception as exc:
                        resource_fail = "resource" in str(exc).lower() or type(exc).__name__ == "OutOfResources"
                        record.update({"status": "FAILED", "failure_stage": failure_stage,
                            "failure_kind": "RESOURCE_UNSUPPORTED" if resource_fail else "COMPILE_OR_CORRECTNESS_FAILURE" if failure_stage in ("COMPILE","SMOKE") else "TOOL_OR_ABI_UNAVAILABLE",
                            "error": traceback.format_exc()})
                        (directory / "error.txt").write_text(record["error"])
                    finally:
                        os.environ.pop("TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT",None)
                        record["artifact_SHA256"] = {p.name:c.sha(p.read_bytes()) for p in sorted(directory.iterdir()) if p.is_file()}
                        c.write(directory / "attempt.json",record)
                        records.append(record)
                        save("attempts.json",records)
                        evidence_volume.commit()
                        print(f'{target} {cfg} {harness} {candidate}: {record["status"]}',flush=True)
                    compiled = args = kernel = None
                del out
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            del x, values, reference
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        env["persistent_cache_stats_end"] = command("ccache","--print-stats")
        env["status"] = "COMPLETE_ARTIFACT_GATE"
    except Exception:
        env["status"] = "INCOMPLETE_ARTIFACT_GATE"
        env["failure_reason"] = traceback.format_exc()
        print(env["failure_reason"],flush=True)
    save("environment.json",env)
    save("attempts.json",records)
    evidence_volume.commit()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream,"w",zipfile.ZIP_DEFLATED) as z:
        for path in sorted(stage.rglob("*")):
            if path.is_file(): z.write(path,path.relative_to(stage))
    return stream.getvalue()


@app.function(image=image,gpu="H100!:1",timeout=3600,single_use_containers=True,volumes={"/cache":build_cache_volume,"/evidence":evidence_volume})
def collect_sm90(provenance,frozen,expected_native):
    return collect_impl("sm90",provenance,frozen,expected_native)


@app.function(image=image,gpu="B200:1",timeout=3600,single_use_containers=True,volumes={"/cache":build_cache_volume,"/evidence":evidence_volume})
def collect_sm100(provenance,frozen,expected_native):
    return collect_impl("sm100",provenance,frozen,expected_native)


@app.function(image=image,gpu="RTX-PRO-6000:1",timeout=3600,single_use_containers=True,volumes={"/cache":build_cache_volume,"/evidence":evidence_volume})
def collect_sm120(provenance,frozen,expected_native):
    return collect_impl("sm120",provenance,frozen,expected_native)


@app.local_entrypoint()
def main(target: str = "all"):
    c.protect()
    targets = list(c.TARGETS) if target == "all" else [target]
    frozen = (c.OUT / "stage_a/protocol.json").read_text()
    expected = c.read(c.BASE / "results/phase8/stage_b/environment.json")["toolchain"]["built_triton_native_extension"]["sha256"]
    provenance = LOCAL_PROVENANCE
    provenance["modal_profile"] = os.environ.get("MODAL_PROFILE","UNSPECIFIED")
    functions = {"sm90":collect_sm90,"sm100":collect_sm100,"sm120":collect_sm120}
    for arch in targets:
        c.require(arch in functions,"Unknown target")
        root = c.OUT / "stage_b" / arch
        root.mkdir(parents=True,exist_ok=True)
        c.require(not (root / "raw_export.zip").exists(),"Never overwrite original compiler export")
        directory = root / "dispatch_attempts" / uuid.uuid4().hex
        directory.mkdir(parents=True,exist_ok=False)
        c.write(directory / "dispatch.json",{"target":arch,"profile":provenance["modal_profile"],"source_manifest_SHA256":provenance["source_manifest_sha256"]})
        try:
            call = functions[arch].spawn(provenance,frozen,expected)
            c.write(directory / "dispatch.json",{"target":arch,"profile":provenance["modal_profile"],"call_id":call.object_id,"image_id":image.object_id,"core_image_id":core_image.object_id,"source_manifest_SHA256":provenance["source_manifest_sha256"]})
            data = call.get()
            with (root / "raw_export.zip").open("xb") as f: f.write(data)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for name in z.namelist():
                    c.require(not Path(name).is_absolute() and ".." not in Path(name).parts,"Unsafe export path")
                z.extractall(root / "export")
            c.write(root / "dispatch.json",{"target":arch,"profile":provenance["modal_profile"],"call_id":call.object_id,"image_id":image.object_id,"core_image_id":core_image.object_id,"original_export_SHA256":c.sha(data)})
            env = c.read(root / "export/environment.json")
            c.require(env["status"] == "COMPLETE_ARTIFACT_GATE", "Incomplete target gate; original return preserved: " + env.get("failure_reason",""))
        except Exception as exc:
            c.write(directory / "local_failure.json",{"error":str(exc),"original_return_preserved":(root / "raw_export.zip").exists(),"profile":provenance["modal_profile"]})
            raise
        print(f"Original {arch} compiler export retained: {len(data)} bytes",flush=True)
