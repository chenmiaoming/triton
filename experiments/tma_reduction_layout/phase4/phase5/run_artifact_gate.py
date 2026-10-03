"""Phase 5 strict H100 session: 114 attempted bundles, no benchmark or timing APIs.

Run with .venv/bin/modal run -m experiments.tma_reduction_layout.phase4.phase5.run_artifact_gate
Raw exports are immutable. Derivation is performed separately by stage_b_audit.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile
import modal
from experiments.tma_reduction_layout.modal_runner import torch_and_deps_image, build_cache_volume
from experiments.tma_reduction_layout.phase4.phase5 import gate_contract as contract
from experiments.tma_reduction_layout.source_provenance import (
    MODAL_SOURCE_IGNORE_PATTERNS, generate_provenance, get_repo_root)

ROOT = get_repo_root()
BASE = ROOT / "experiments/tma_reduction_layout"
OUT = contract.OUT
# SDK builds the image before calling the local entrypoint. Freeze the source
# manifest before that build so later local files cannot enter the dispatch
# manifest without entering the uploaded image snapshot.
LOCAL_PROVENANCE = generate_provenance() if modal.is_local() else None
app = modal.App("triton-phase5-stage-b-artifact-gate")
image = (torch_and_deps_image.add_local_dir(
    ROOT, remote_path="/opt/triton-src", copy=True,
    ignore=MODAL_SOURCE_IGNORE_PATTERNS + ["experiments/tma_reduction_layout/phase4/results/**"])
    .run_commands(
        "mkdir -p /cache/ccache /cache/triton-home",
        "python3 -m pip uninstall -y triton pytorch-triton || true",
        "cd /opt/triton-src && python3 -m pip install -r python/requirements.txt",
        "cd /opt/triton-src && TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=true python3 -m pip install -e . --no-build-isolation -v",
        "ccache -s",
        "python3 -c 'import triton; print(triton.__version__, triton.__file__)'",
        volumes={"/cache": build_cache_volume}))


@app.function(image=image, single_use_containers=True, gpu="H100!:1", timeout=3600, volumes={"/cache": build_cache_volume})
def collect(provenance, frozen, case_ids=None):
    import ast
    import math
    import re
    import subprocess
    import traceback
    import torch
    import triton
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
    from experiments.tma_reduction_layout.source_provenance import verify_remote_source_manifest
    from experiments.tma_reduction_layout.phase4.exact_cuda import ExactCUDA, sha
    from experiments.tma_reduction_layout.phase4.kernels_stage_b import (
        make_canonical, single_kernel, repeated_kernel)
    from experiments.tma_reduction_layout.phase4.artifact_gate import ttgir_contract
    from experiments.tma_reduction_layout.gluon.artifact_checks import parse_resource

    remote_root = Path("/opt/triton-src")
    verification = verify_remote_source_manifest(provenance, remote_root)
    torch.cuda.init()
    name, cc = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
    if "H100" not in name or cc != (9, 0):
        raise RuntimeError(f"Strict H100 requirement: {name}, CC={cc}")
    if not str(triton.__file__).startswith("/opt/triton-src/"):
        raise RuntimeError("Triton import provenance mismatch")
    cuda = ExactCUDA()
    environment = cuda.environment()
    def command(*args):
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)
    versions = {}
    for tool in ("ptxas", "cuobjdump", "nvdisasm", "nvcc", "clang", "ld.lld", "ccache"):
        try:
            versions[tool] = command(tool, "--version").strip()
        except Exception as exc:
            versions[tool] = f"UNAVAILABLE: {exc}"
    # Triton uses its bundled ptxas, which may differ from PATH's CUDA toolkit.
    from triton.backends.nvidia.compiler import get_ptxas
    actual_ptxas = Path(get_ptxas(90).path)
    versions["actual_compiler_ptxas"] = {"path": str(actual_ptxas),
        "version": command(str(actual_ptxas), "--version").strip(), "sha256": sha(actual_ptxas.read_bytes())}
    import triton._C.libtriton as native
    native_path = Path(native.__file__)
    versions["built_triton_native_extension"] = {"path": str(native_path),
        "sha256": sha(native_path.read_bytes())}
    import ctypes as C
    rt = C.CDLL("/usr/local/cuda/lib64/libcudart.so")
    rt.cudaRuntimeGetVersion.argtypes = [C.POINTER(C.c_int)]
    rt.cudaRuntimeGetVersion.restype = C.c_int
    runtime_version = C.c_int()
    code = rt.cudaRuntimeGetVersion(C.byref(runtime_version))
    if code:
        raise RuntimeError(f"cudaRuntimeGetVersion: {code}")
    try:
        driver_name = command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader").strip()
    except Exception:
        driver_name = "UNAVAILABLE (driver API version recorded)"
    environment.update({"gpu_name": name, "compute_capability": list(cc),
        "driver": driver_name, "CUDA_runtime_version": runtime_version.value,
        "cudaRuntimeGetVersion_return_code": code, "toolchain": versions,
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda,
        "triton": triton.__version__, "triton_import": triton.__file__,
        "modal_image": os.environ.get("MODAL_IMAGE_ID", "UNAVAILABLE"),
        "modal_task_id": os.environ.get("MODAL_TASK_ID", "UNAVAILABLE"),
        "modal_function_id": os.environ.get("MODAL_FUNCTION_ID", "UNAVAILABLE"),
        "persistent_cache": {k: os.environ.get(k, "UNAVAILABLE") for k in
            ("CCACHE_DIR", "TRITON_HOME", "TRITON_BUILD_WITH_CCACHE", "TRITON_BUILD_WITH_CLANG_LLD")},
        "volume": "triton-build-cache", "source_verification": verification,
        "no_performance_observation": True})
    stage = Path("/tmp/phase5_stage_b_export")
    stage.mkdir(exist_ok=True)
    def write_json(path, obj):
        path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    write_json(stage / "environment.json", environment)
    write_json(stage / "local_source_provenance.json", provenance)
    for filename, text in frozen.items():
        (stage / ("frozen_" + filename)).write_text(text)
    # Complete bytes of the uploaded source subset, with per-file hashes.
    with zipfile.ZipFile(stage / "uploaded_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(provenance["source_manifest"]):
            z.write(remote_root / path, path)
    write_json(stage / "source_bindings.json", {
        "local_source_manifest_sha256": provenance["source_manifest_sha256"],
        "uploaded_source_subset_sha256": verification["remote_source_subset_sha256"],
        "uploaded_source_archive_sha256": sha((stage / "uploaded_source.zip").read_bytes()),
        "frozen_SHA256": {k: sha(v.encode()) for k, v in frozen.items()},
        "kernel_source_sha256": sha((remote_root / "experiments/tma_reduction_layout/phase4/kernels_stage_b.py").read_bytes()),
        "git_head": provenance["git_head_sha"], "build_identity": provenance["composite_digest_sha256"]})
    cases = contract.population(json.loads(frozen["structural_pool.json"]))
    if case_ids is not None:
        cases = [c for c in cases if c["config_id"] in case_ids]
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    attempts = []
    source_rel = "experiments/tma_reduction_layout/phase4/kernels_stage_b.py"
    source_text = (remote_root / source_rel).read_text()
    source_ast = ast.parse(source_text)
    def source_stage(ptx, fn_name):
        fn = next(n for n in ast.walk(source_ast) if isinstance(n, ast.FunctionDef) and n.name == fn_name)
        lines = sorted(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr in ("to", "max"))
        ptx_paths = re.findall(r'\.file\s+(\d+)\s+"([^"]+)"', ptx)
        matches = [(int(idx), path) for idx, path in ptx_paths if path.endswith(source_rel)]
        if len(matches) != 1 or len(lines) != 2:
            raise RuntimeError(f"Source-stage binding unavailable: {fn_name} {matches} {lines}")
        idx, path = matches[0]
        return {"source_path": source_rel, "source_sha256": sha(source_text.encode()),
                "ptx_source_path": path, "file_id": idx, "source_lines": lines,
                "kernel_function": fn_name}

    for case in cases:
        cfg, m, n, w = case["config_id"], case["M"], case["N"], case["num_warps"]
        # The descriptor extent matches the frozen future harness. Only the four
        # launched prefix tiles are initialized/read in this non-timed smoke.
        b_desc, smoke_b = 65536, 4
        x = out = reference = values = None
        try:
            x = torch.empty((b_desc, m, n), dtype=torch.bfloat16, device="cuda")
            values = ((torch.arange(smoke_b * m * n, dtype=torch.int32).reshape(smoke_b, m, n) % 257) - 128).float() / 8
            x[:smoke_b].copy_(values.to(torch.bfloat16))
            reference = values.to(torch.bfloat16).float().amax(dim=1).cuda()
            out = torch.empty((b_desc, n), dtype=torch.float32, device="cuda")
            torch.cuda.synchronize()
        except Exception as exc:
            error = traceback.format_exc()
            free_bytes, capacity = torch.cuda.mem_get_info()
            needed = b_desc * m * n * 2 + b_desc * n * 4
            infrastructure = isinstance(exc, torch.cuda.OutOfMemoryError) and needed < capacity
            for candidate in ("default", "4"):
                for harness in ("canonical", "single", "repeated"):
                    directory = stage / harness / cfg / candidate
                    directory.mkdir(parents=True, exist_ok=True)
                    record = {"case_id": cfg, "candidate": candidate, "harness": harness,
                        "attempted": True, "status": "PENDING" if infrastructure else "FAILED_DETERMINISTIC",
                        "failure_stage": "ALLOCATION", "failure_kind": "INFRASTRUCTURE_FRAGMENTATION" if infrastructure else "RESOURCE_UNSUPPORTED",
                        "error": error, "allocation_free_bytes": free_bytes, "allocation_capacity_bytes": capacity,
                        "required_allocation_bytes": needed, "full_descriptor_allocation": False}
                    (directory / "error.txt").write_text(error)
                    record["artifact_SHA256"] = {"error.txt": sha(error.encode())}
                    write_json(directory / "attempt.json", record)
                    attempts.append(record)
            del x, out, reference, values
            torch.cuda.empty_cache()
            continue
        for candidate in ("default", "4"):
            expected = case["default" if candidate == "default" else "cand4"]["layout"]
            layout = gl.BlockedLayout(size_per_thread=expected["sizePerThread"],
                threads_per_warp=expected["threadsPerWarp"], warps_per_cta=expected["warpsPerCTA"], order=expected["order"])
            observed_shared = None
            for harness in ("canonical", "single", "repeated"):
                directory = stage / harness / cfg / candidate
                directory.mkdir(parents=True, exist_ok=True)
                record = {"case_id": cfg, "candidate": candidate, "harness": harness, "attempted": True,
                    "B_DESC": b_desc, "full_descriptor_allocation": True,
                    "input_allocation_shape": list(x.shape), "input_allocation_bytes": x.numel() * x.element_size()}
                failure_stage = "COMPILE"
                try:
                    os.environ["TRITON_CACHE_DIR"] = f"/tmp/phase5_stage_b_cache_{cfg}_{candidate}_{harness}"
                    os.environ.pop("TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT", None)
                    if harness == "canonical":
                        if candidate != "default":
                            os.environ["TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT"] = candidate
                        kernel, fn_name = make_canonical(), "canonical_kernel"
                        args = (x, out, m * n, n, b_desc, m, n)
                    else:
                        attrs = observed_shared or {"swizzlingByteWidth": 128, "elementBitWidth": 16, "rank": 3, "transposed": False}
                        shared = gl.NVMMASharedLayout(swizzle_byte_width=attrs["swizzlingByteWidth"],
                            element_bitwidth=attrs["elementBitWidth"], rank=attrs["rank"], transposed=attrs["transposed"])
                        desc = TensorDescriptor.from_tensor(x, [1, m, n], shared)
                        if harness == "single":
                            kernel, fn_name = single_kernel, "single_kernel"
                            args = (desc, out, layout, shared, b_desc, m, n)
                        else:
                            extent = math.prod(s * math.ceil(dim / (s * t * p)) for dim, s, t, p in
                                zip([1, m, n], expected["sizePerThread"], expected["threadsPerWarp"], expected["warpsPerCTA"]))
                            constraints = ",".join(["=h"] * extent + [str(i) for i in range(extent)])
                            kernel, fn_name = repeated_kernel, "repeated_kernel"
                            args = (desc, out, 1, layout, shared, b_desc, m, n, constraints, ("f",))
                    compiled = kernel.warmup(*args, grid=(smoke_b,), num_warps=w)
                    record["binary_generated"] = True
                    failure_stage = "EXPORT"
                    for key in ("ttgir", "ptx", "cubin"):
                        data = compiled.asm[key]
                        (directory / ("kernel." + key)).write_bytes(data if isinstance(data, bytes) else data.encode())
                    digest = sha((directory / "kernel.cubin").read_bytes())
                    (directory / "kernel.cubin.sha256").write_text(digest + "\n")
                    for suffix, option in (("sass", "-sass"), ("resource.txt", "-res-usage")):
                        (directory / ("kernel." + suffix)).write_text(command("cuobjdump", option, str(directory / "kernel.cubin")))
                    resources = parse_resource((directory / "kernel.resource.txt").read_text())
                    ptx = (directory / "kernel.ptx").read_text()
                    reduction_stage = source_stage(ptx, fn_name)
                    if harness == "canonical":
                        try:
                            observed_shared = ttgir_contract(compiled.asm["ttgir"], [1, m, n], w)["shared"]
                        except ValueError:
                            observed_shared = None
                    metadata = {"case_id": cfg, "candidate": candidate, "harness": harness,
                        "logical_shape": [1, m, n], "reduction_axis": 1, "num_warps": w,
                        "B_DESC": b_desc, "function_name": compiled.name, "dynamic_smem_bytes": compiled.metadata.shared,
                        "resources": resources, "cubin_sha256": digest, "reduction_stage": reduction_stage,
                        "compiled_object_hash": compiled.hash, "compile_calls": 1,
                        "R0_limitation": "R=0 subtraction controls fixed one-time harness differential. It does not prove absence of interactions via register allocation, live ranges, instruction scheduling, or compiler decisions."}
                    write_json(directory / "metadata.json", metadata)
                    # Observations are re-extracted offline; never copied from the pool.
                    from experiments.tma_reduction_layout.gluon import artifact_checks as ac
                    from experiments.tma_reduction_layout.phase4.artifact_gate import body_fingerprint
                    try:
                        observed = {"layout": ttgir_contract(compiled.asm["ttgir"], [1, m, n], w),
                            "initial_localload": ac.initial_load_signature(ptx),
                            "all_ptx_backedges": ac.ptx_backedges(ptx)}
                        if harness == "repeated":
                            loops = [e for e in observed["all_ptx_backedges"] if e["kind"] == "compiler_loop"]
                            if len(loops) != 1:
                                raise ValueError("Unique runtime loop missing")
                            observed["reduction_fingerprint"] = ac.fingerprint([i for i in ac.ptx_instructions(ptx)
                                if loops[0]["start"] <= i["line"] <= loops[0]["end"]])
                        else:
                            observed["reduction_fingerprint"] = body_fingerprint(ptx, reduction_stage, minimal_output=harness == "single")
                    except ValueError as exc:
                        observed = {"structural_parse_error": str(exc)}
                    write_json(directory / "observed_structure.json", observed)
                    failure_stage = "OCCUPANCY"
                    occupancy = cuda.occupancy(directory / "kernel.cubin", compiled.name,
                        compiled.metadata.shared, w, sha((directory / "kernel.resource.txt").read_bytes()))
                    occupancy.update({"gpu_uuid": environment["gpu_uuid"],
                        "build_identity": provenance["composite_digest_sha256"]})
                    write_json(directory / "occupancy.json", occupancy)
                    failure_stage = "SMOKE"
                    compiled._init_handles()
                    if sha(compiled.kernel) != digest:
                        raise RuntimeError("Launch object's loaded bytes differ from archived CUBIN")
                    smoke = {"timed": False, "B_RUN": smoke_b, "input_dtype": "bfloat16",
                        "output_dtype": "float32", "input_generation": "CPU integer arange modulo 257, subtract 128, divide by 8; cast BF16; copy four initialized prefix tiles",
                        "initialized_prefix_tiles": smoke_b, "B_DESC": b_desc,
                        "full_descriptor_allocation": True, "input_allocation_shape": list(x.shape),
                        "input_allocation_bytes": x.numel() * x.element_size(),
                        "uninitialized_regions_never_launched_or_read": True, "launch_api": "CompiledKernel.__getitem__; no additional JIT calls",
                        "compile_calls": 1, "launched_cubin_sha256": digest}
                    if harness == "repeated":
                        smoke["purpose"] = "launch safety / structural execution; not reduction correctness"
                        bindings = []
                        for r in (0, 1):
                            out[:smoke_b].fill_(float("nan"))
                            run_args = args[:2] + (r,) + args[3:]
                            before_sha = sha(compiled.kernel)
                            compiled[(smoke_b, 1, 1)](*run_args)
                            torch.cuda.synchronize()
                            after_sha = sha(compiled.kernel)
                            bindings.append({"R": r, "before_cubin_sha256": before_sha,
                                "after_cubin_sha256": after_sha, "archived_cubin_sha256": sha((directory / "kernel.cubin").read_bytes()),
                                "trivial_output_zero": bool((out.reshape(-1)[:smoke_b] == 0).all().item())})
                        smoke["runtime_bindings"] = bindings
                        smoke["passed"] = all(b["trivial_output_zero"] for b in bindings)
                    else:
                        out[:smoke_b].fill_(float("nan"))
                        compiled[(smoke_b, 1, 1)](*args)
                        torch.cuda.synchronize()
                        error = (out[:smoke_b] - reference).abs().max().item()
                        smoke.update({"purpose": "reduction correctness", "max_abs_diff": error,
                            "passed": error == 0.0, "after_cubin_sha256": sha(compiled.kernel)})
                    write_json(directory / "smoke.json", smoke)
                    record.update({"status": "EXPORTED", "cubin_sha256": digest})
                except Exception as exc:
                    resource = type(exc).__name__ == "OutOfResources" or "out of resource" in str(exc).lower()
                    deterministic = failure_stage in ("COMPILE", "SMOKE") or resource
                    record.update({"status": "FAILED_DETERMINISTIC" if deterministic else "PENDING",
                        "failure_stage": failure_stage,
                        "failure_kind": "RESOURCE_UNSUPPORTED" if resource else "COMPILE_FAILURE_DETERMINISTIC" if failure_stage == "COMPILE" else "CORRECTNESS_FAILURE" if failure_stage == "SMOKE" else "UNRESOLVED_TOOL_FAILURE",
                        "error": traceback.format_exc()})
                    (directory / "error.txt").write_text(record["error"])
                finally:
                    os.environ.pop("TRITON_TMA_REDUCTION_LAYOUT_EXPERIMENT", None)
                    record["artifact_SHA256"] = {p.name: sha(p.read_bytes()) for p in sorted(directory.iterdir()) if p.is_file()}
                    write_json(directory / "attempt.json", record)
                    attempts.append(record)
                    print(f"{cfg} {candidate} {harness}: {record['status']}", flush=True)
        args = run_args = desc = compiled = None
        del x, out, reference, values
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    write_json(stage / "attempts.json", attempts)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(stage))
    return stream.getvalue()


@app.local_entrypoint()
def main():
    contract.protected_inventory()
    if OUT.exists():
        raise RuntimeError("Preserve existing raw Phase 5 exports; no implicit rerun")
    frozen = {name: (contract.PREREG / name).read_text() for name in contract.FROZEN}
    provenance = LOCAL_PROVENANCE
    if provenance["git_head_sha"] != contract.BASELINE:
        raise RuntimeError("Wrong Phase 5 baseline")
    archive = collect.remote(provenance, frozen)
    OUT.mkdir(parents=True)
    def extract(raw, target):
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for info in z.infolist():
                if Path(info.filename).is_absolute() or ".." in Path(info.filename).parts:
                    raise RuntimeError("Unsafe export path")
            z.extractall(target)
    extract(archive, OUT)
    (OUT / "raw_export.zip").write_bytes(archive)
    dispatch = {"resolved_image_id": image.object_id, "function_id": collect.object_id,
        "raw_export_zip_sha256": hashlib.sha256(archive).hexdigest(), "profile": "miaomingc", "strict_gpu": "H100!:1"}
    (OUT / "modal_dispatch.json").write_bytes(contract.encode(dispatch))
    affected = sorted({a["case_id"] for a in contract.read(OUT / "attempts.json")
                       if a.get("failure_kind") == "INFRASTRUCTURE_FRAGMENTATION"})
    if affected:
        # Never retry only a candidate or an unfavorable artifact. Each entire
        # affected case gets at most one fresh H100 container; keep first exports.
        for case_id in affected:
            retry = collect.remote(provenance, frozen, [case_id])
            destination = OUT / "infrastructure_retries" / case_id
            destination.mkdir(parents=True)
            extract(retry, destination)
            (destination / "raw_export.zip").write_bytes(retry)
            (destination / "modal_dispatch.json").write_bytes(contract.encode({**dispatch,
                "raw_export_zip_sha256": hashlib.sha256(retry).hexdigest(), "retry_reason": "INFRASTRUCTURE_FRAGMENTATION",
                "entire_case": True, "fresh_container": True, "max_case_retries": 1}))
    print(f"Archived {len(contract.read(OUT / 'attempts.json'))} original attempts at {OUT}; no timing")
