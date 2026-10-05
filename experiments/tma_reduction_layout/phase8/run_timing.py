"""Phase 8: archived ELF-only timing; durable remote and local attempt returns."""
import io
import json
import os
from pathlib import Path
import zipfile
import modal
from experiments.tma_reduction_layout.modal_runner import torch_and_deps_image, build_cache_volume
from experiments.tma_reduction_layout.source_provenance import (
    MODAL_SOURCE_IGNORE_PATTERNS, generate_provenance, verify_remote_source_manifest)
from experiments.tma_reduction_layout.phase8 import timing_contract as tc

LOCAL_PROVENANCE = generate_provenance() if modal.is_local() else None
app = modal.App("triton-phase8-exact-timing")
evidence_volume = modal.Volume.from_name("triton-phase8-evidence", create_if_missing=True)
image = torch_and_deps_image.env({"PYTHONPATH": "/opt/triton-src"}).add_local_dir(tc.ROOT, remote_path="/opt/triton-src", copy=True,
    ignore=MODAL_SOURCE_IGNORE_PATTERNS + ["experiments/tma_reduction_layout/phase4/results/**"])


@app.function(image=image, gpu="H100!:1", timeout=3600,
              single_use_containers=True, volumes={"/cache": build_cache_volume, "/evidence": evidence_volume})
def benchmark(prov, cases, planned, binaries, archived_zip):
    import ctypes as C
    import datetime
    import math
    import socket
    import subprocess
    import traceback
    import torch
    from experiments.tma_reduction_layout.phase8.launch import Driver, Loaded
    import sys

    raw = {"invocation": planned["invocation"], "status": "INVALID_PROTOCOL_RUN",
           "dispatch_id": modal.current_function_call_id(),
           "input_id": modal.current_input_id(),
           "process_identity": f'{socket.gethostname()}:{os.getpid()}:{os.environ.get("MODAL_TASK_ID", "UNAVAILABLE")}',
           "recompilation": False, "timing_primitive": "CUDA_DRIVER_EVENTS_ONE_KERNEL_PER_SAMPLE",
           "warmup_policy": {"count_per_condition": tc.WARMUP, "condition_order": "first frozen round", "timed": False},
           "warmups": [], "visits": [], "loaded_binaries": {}, "allocations": {}, "tensor_maps": {},
           "compilation_audit": {"compiler_subprocess_calls": 0, "triton_imported": False,
               "jit_compile_calls": 0, "load_input": "ELF_CUBIN_ONLY"}}
    subprocess_commands = []
    def audit(event, args):
        if event == "subprocess.Popen":
            command = args[1]
            subprocess_commands.append(list(command) if isinstance(command, (list, tuple)) else command)
            executable = Path(args[0]).name
            if executable != "nvidia-smi":
                raw["compilation_audit"]["compiler_subprocess_calls"] += 1
                raise RuntimeError("Disallowed subprocess in timing worker: " + executable)
    sys.addaudithook(audit)
    driver, kernels = None, {}
    def telemetry():
        fields = "uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,driver_version"
        result = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            values = subprocess.check_output(["nvidia-smi", "--query-gpu=" + fields,
                "--format=csv,noheader,nounits"], text=True).strip().split(",")
            result.update(dict(zip(fields.split(","), [v.strip() for v in values])))
        except Exception as exc:
            result["unavailable"] = str(exc)
        return result
    try:
        verification = verify_remote_source_manifest(prov)
        archives = Path("/tmp/stage_c_archives")
        with zipfile.ZipFile(io.BytesIO(archived_zip)) as z:
            for item in z.namelist():
                if Path(item).is_absolute() or ".." in Path(item).parts:
                    raise RuntimeError("Unsafe binary archive path")
            z.extractall(archives)
        torch.cuda.init()
        name, cc = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
        if "H100" not in name or cc != (9, 0):
            raise RuntimeError(f"Strict H100 verification failed: {name}, {cc}")
        driver = Driver()
        env = driver.environment()
        rt = C.CDLL("/usr/local/cuda/lib64/libcudart.so")
        rt.cudaRuntimeGetVersion.argtypes, rt.cudaRuntimeGetVersion.restype = [C.POINTER(C.c_int)], C.c_int
        runtime = C.c_int()
        code = rt.cudaRuntimeGetVersion(C.byref(runtime))
        if code: raise RuntimeError(f"cudaRuntimeGetVersion: {code}")
        env.update({"gpu_name": name, "compute_capability": list(cc), "CUDA_runtime_version": runtime.value,
            "cudaRuntimeGetVersion_return_code": code, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "modal_image_id": os.environ.get("MODAL_IMAGE_ID", "UNAVAILABLE"),
            "modal_task_id": os.environ.get("MODAL_TASK_ID", "UNAVAILABLE"),
            "source_verification": verification, "archived_payload_sha256": tc.sha(archived_zip),
            "input_generation": "torch.manual_seed(42); sorted-case first occurrence of unique(M,N) contiguous BF16 torch.randn shared across warps [65536,M,N]; same allocation for both candidates/harnesses/all R,B",
            "source_HEAD": prov["git_head_sha"], "source_manifest_sha256": prov["source_manifest_sha256"]})
        env["modal_profile"] = prov["modal_profile"]
        raw["environment"] = env
        raw["telemetry_start"] = telemetry()
        torch.manual_seed(42)
        tensors = {}
        allocations_by_shape = {}
        for cfg in sorted(cases):
            m, n = cases[cfg]["M"], cases[cfg]["N"]
            if (m, n) not in allocations_by_shape:
                x = torch.randn((65536, m, n), dtype=torch.bfloat16, device="cuda")
                out = torch.empty((65536, n), dtype=torch.float32, device="cuda")
                scratch = torch.empty(65536 * 128, dtype=torch.uint8, device="cuda")
                allocations_by_shape[m, n] = (x, out, scratch)
            x, out, scratch = allocations_by_shape[m, n]
            tensors[cfg] = (x, out, scratch)
            raw["allocations"][cfg] = {"input_shape": list(x.shape), "input_dtype": str(x.dtype),
                "input_bytes": x.numel() * x.element_size(), "input_pointer": x.data_ptr(),
                "output_shape": list(out.shape), "output_dtype": str(out.dtype),
                "output_bytes": out.numel() * out.element_size(), "output_pointer": out.data_ptr(),
                "scratch_bytes": scratch.numel(), "scratch_pointer": scratch.data_ptr(),
                "all_descriptor_elements_initialized": True}
        torch.cuda.synchronize()
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        descriptors = {}
        for key, binding in binaries.items():
            path = archives / binding["archive_path"]
            ptx = path.with_suffix(".ptx").read_bytes()
            if tc.sha(ptx) != binding["ptx_sha256"]:
                raise RuntimeError("Frozen PTX ABI SHA mismatch")
            kernels[key] = Loaded(driver, archives, binding, ptx.decode())
            cfg, harness, _ = key.split(":")
            if harness.startswith("host_") or harness == "switch":
                m, n = cases[cfg]["M"], cases[cfg]["N"]
                descriptors[key] = driver.tensor_map(tensors[cfg][0].data_ptr(), m, n,
                    binding["shared_layout"]["swizzlingByteWidth"])
                sw = binding["shared_layout"]["swizzlingByteWidth"]
                raw["tensor_maps"][key] = {"global_dims_n_m_b": [n, m, 65536],
                    "global_strides_bytes": [n * 2, m * n * 2],
                    "message_box_n_m_b": [min(n, sw // 2), min(m, 256), 1],
                    "element_strides": [1, 1, 1], "swizzling_bytes": sw,
                    "source_rule": "frozen getTMABlockShapeTiled: each message box dimension <=256; logical descriptor shape unchanged",
                    "descriptor_SHA256": tc.sha(C.string_at(descriptors[key][1], 128))}
        # All runtime conditions warmed identically, in the first frozen round.
        for tag in planned["rounds"][0]["order"]:
            p = tc.condition(tag)
            cfg, h, c = p["config_id"], p["harness"], p["candidate"]
            key = tc.binary_key(cfg,h,c)
            k = kernels[key]
            values, params = k.parameters(tensors[cfg], p["R"], descriptors.get(key))
            for _ in range(tc.WARMUP):
                digest = k.guard()
                k.launch(p["B_RUN"], params, stream)
            driver.call("cuStreamSynchronize", [C.c_void_p], stream)
            record = {"condition_tag": tag, "count": tc.WARMUP, "binary_sha256": digest,
                      "runtime_mode_argument": values[10].value if k.harness == "switch" else None}
            if h != "repeated":
                expected = tensors[cfg][0][:4].float().amax(dim=1)
                error = (tensors[cfg][1][:4] - expected).abs().max().item()
                record["loaded_ABI_correctness_max_abs_diff"] = error
                if error != 0.0: raise RuntimeError("Archived full-output ABI correctness failure")
            raw["warmups"].append(record)
        first, last = driver.event(), driver.event()
        for round_ in planned["rounds"]:
            for tag in round_["order"]:
                p = tc.condition(tag)
                cfg, h, c = p["config_id"], p["harness"], p["candidate"]
                case, key = cases[cfg], tc.binary_key(cfg,h,c)
                k = kernels[key]
                values, params = k.parameters(tensors[cfg], p["R"], descriptors.get(key))
                samples, guards = [], []
                visit = {**p, "condition_tag": tag, "origin": case["origin"],
                    "stage_b_class": case["final_class"], "invocation": planned["invocation"],
                    "round": round_["round"], "samples_us": samples,
                    "archived_cubin_sha256": k.binding["archive_sha256"],
                    "runtime_loaded_cubin_sha256": k.digest, "sample_launch_sha256": guards,
                    "binary_sha_match": True, "gpu_uuid": env["gpu_uuid"],
                    "runtime_mode_argument": values[10].value if k.harness == "switch" else None,
                    "runtime_module_identity": hex(k.module.value), "runtime_function_identity": hex(k.function.value)}
                # Retain already measured samples if a later sample in this visit fails.
                raw["visits"].append(visit)
                for _ in range(10):
                    digest = k.guard()  # SHA I/O entirely outside the event interval.
                    driver.record(first, stream)
                    k.launch(p["B_RUN"], params, stream)
                    driver.record(last, stream)
                    driver.synchronize_event(last)
                    value = driver.elapsed_us(first, last)
                    if not math.isfinite(value) or value <= 0:
                        raise RuntimeError("Non-positive/non-finite CUDA event sample: invalid entire invocation")
                    samples.append(value)
                    guards.append(digest)
            print(f'Invocation {planned["invocation"]}: completed frozen round {round_["round"]}/10', flush=True)
        driver.call("cuEventDestroy_v2", [C.c_void_p], first)
        driver.call("cuEventDestroy_v2", [C.c_void_p], last)
        if any(n == "triton" or n.startswith("triton.") for n in sys.modules):
            raw["compilation_audit"]["triton_imported"] = True
            raise RuntimeError("Triton imported during exact-binary worker")
        raw["status"] = "VALID_PROTOCOL_RUN"
    except Exception:
        raw["failure_reason"] = traceback.format_exc()
        print(raw["failure_reason"], flush=True)
    finally:
        raw["telemetry_end"] = telemetry()
        raw["subprocess_commands"] = subprocess_commands
        if driver is not None:
            raw["loaded_binaries"] = {k: v.record_binding() for k, v in kernels.items()}
            for k in kernels.values():
                try:
                    driver.call("cuModuleUnload", [C.c_void_p], k.module)
                except Exception:
                    raw["status"] = "INVALID_PROTOCOL_RUN"
                    raw["failure_reason"] = traceback.format_exc()
            raw["cuda_call_counts"] = dict(driver.counts)
            raw["cuda_return_codes"] = sorted(driver.codes)
    evidence_path = Path("/evidence") / prov["phase8_stage"] / (raw["dispatch_id"] + ".json")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    raw["remote_evidence"] = {"volume": "triton-phase8-evidence", "profile": prov["modal_profile"], "path": str(evidence_path)}
    evidence_path.write_bytes(tc.encode(raw))
    evidence_volume.commit()
    return raw


@app.local_entrypoint()
def main(stage: str = "stage_c"):
    import uuid
    from experiments.tma_reduction_layout.phase8 import common as c
    cases, binaries, plan = tc.inputs(stage)
    root = c.OUT / stage
    root.mkdir(parents=True, exist_ok=True)
    if (root / "raw_manifest.json").exists():
        raise RuntimeError("Raw stage already frozen; do not rerun scientific outcomes")
    c.protect()
    provenance = LOCAL_PROVENANCE
    provenance["modal_profile"] = os.environ.get("MODAL_PROFILE", "UNSPECIFIED")
    provenance["phase8_stage"] = stage
    source = root / "source_bindings.json"
    if source.exists() and list(root.glob("raw_invocation_*.json")):
        previous = c.read(source)["provenance"]
        c.require(previous["source_manifest_sha256"] == provenance["source_manifest_sha256"],
                  "Source changed after valid invocation; stop")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        for binding in binaries.values():
            path = c.ROOT / binding["archive_path"]
            z.write(path, binding["archive_path"])
            z.write(path.with_suffix(".ptx"), str(Path(binding["archive_path"]).with_suffix(".ptx")))
    payload = stream.getvalue()
    c.write(root / "executed_schedule.json", {**plan, "executed": True})
    c.write(root / "binary_bindings.json", binaries)
    (root / "archived_timing_payload.zip").write_bytes(payload)
    if not source.exists():
        with zipfile.ZipFile(root / "uploaded_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for name in provenance["source_manifest"]:
                z.write(c.ROOT / name, name)
        c.write(source, {"provenance": provenance, "payload_SHA256": c.sha(payload),
                         "protocol_SHA256": c.sha((c.OUT / "stage_a/protocol.json").read_bytes()),
                         "timing_kernel_recompilation": False})
    else:
        c.require(c.read(source)["provenance"]["source_manifest_sha256"] == provenance["source_manifest_sha256"],
                  "Existing source snapshot must match every resumed dispatch")
    for planned in plan["invocations"]:
        number = planned["invocation"]
        target = root / f"raw_invocation_{number}.json"
        if target.exists():
            tc.validate_invocation(c.read(target), planned, cases, binaries)
            continue
        for attempt in range(1, 4):
            label = f"invocation_{number}_{uuid.uuid4().hex}"
            directory = root / "attempts" / label
            directory.mkdir(parents=True, exist_ok=False)
            c.write(directory / "dispatch.json", {"invocation": number, "attempt_in_session": attempt,
                "profile": provenance["modal_profile"], "source_manifest_SHA256": provenance["source_manifest_sha256"]})
            try:
                call = benchmark.spawn(provenance, cases, planned, binaries, payload)
                c.write(directory / "dispatch.json", {"invocation": number, "attempt_in_session": attempt,
                    "profile": provenance["modal_profile"], "call_id": call.object_id,
                    "image_id": image.object_id, "source_manifest_SHA256": provenance["source_manifest_sha256"]})
                returned = call.get()
                # Preserve the complete original return before any validation or normalization.
                with (directory / "original_return.json").open("xb") as f:
                    f.write(c.encode(returned))
                tc.validate_invocation(returned, planned, cases, binaries)
            except Exception as exc:
                c.write(directory / "local_failure.json", {"error": str(exc), "invocation": number,
                    "attempt_in_session": attempt, "original_return_preserved": (directory / "original_return.json").exists()})
                cause = str(exc)
                if (directory / "original_return.json").exists():
                    cause += c.read(directory / "original_return.json").get("failure_reason", "")
                allowed = ("CUDA return code", "out of memory", "timeout", "interrupted", "Connection", "RPC", "Unavailable")
                if not any(word.lower() in cause.lower() for word in allowed) or attempt == 3:
                    raise
                print(f"Whole-invocation infrastructure repair: {number}/{attempt}", flush=True)
                continue
            with target.open("xb") as f:
                f.write(c.encode(returned))
            print(f"Validated invocation {number}; {len(planned['rounds'][0]['order'])*100} scalar samples", flush=True)
            break
    c.write(root / "environment.json", {"invocations": {
        str(i): c.read(root / f"raw_invocation_{i}.json")["environment"] for i in (1, 2, 3)}})
    print("Three complete raw invocations retained. Freeze manifest after CLI logs close; no hypothesis analysis here.", flush=True)

