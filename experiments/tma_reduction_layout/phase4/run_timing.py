"""Stage C: three separately dispatched single-use H100 containers, frozen CUBINs."""
import io
import json
import os
from pathlib import Path
import zipfile
import modal
from experiments.tma_reduction_layout.modal_runner import torch_and_deps_image, build_cache_volume
from experiments.tma_reduction_layout.source_provenance import (
    MODAL_SOURCE_IGNORE_PATTERNS, generate_provenance, verify_remote_source_manifest)
from experiments.tma_reduction_layout.phase4 import timing_contract as tc

app = modal.App("triton-phase4-stage-c-frozen-timing")
image = torch_and_deps_image.add_local_dir(tc.ROOT, remote_path="/opt/triton-src", copy=True,
    ignore=MODAL_SOURCE_IGNORE_PATTERNS + ["experiments/tma_reduction_layout/phase4/results/**"])


@app.function(image=image, gpu="H100!:1", timeout=3600,
              single_use_containers=True, volumes={"/cache": build_cache_volume})
def benchmark(prov, cases, planned, binaries, archived_zip):
    import ctypes as C
    import datetime
    import math
    import socket
    import subprocess
    import traceback
    import torch
    from experiments.tma_reduction_layout.phase4.archived_launch import Driver, Loaded

    raw = {"invocation": planned["invocation"], "status": "INVALID_PROTOCOL_RUN",
           "dispatch_id": modal.current_function_call_id(),
           "input_id": modal.current_input_id(),
           "process_identity": f'{socket.gethostname()}:{os.getpid()}:{os.environ.get("MODAL_TASK_ID", "UNAVAILABLE")}',
           "recompilation": False, "timing_primitive": "CUDA_DRIVER_EVENTS_ONE_KERNEL_PER_SAMPLE",
           "warmup_policy": {"count_per_condition": tc.WARMUP, "condition_order": "first frozen round", "timed": False},
           "warmups": [], "visits": [], "loaded_binaries": {}}
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
            "input_generation": "torch.manual_seed(42); sorted-case contiguous BF16 torch.randn [65536,M,N]; same allocation for both candidates/harnesses/all R,B",
            "source_HEAD": prov["git_head_sha"], "source_manifest_sha256": prov["source_manifest_sha256"]})
        env["modal_profile"] = prov["modal_profile"]
        raw["environment"] = env
        raw["telemetry_start"] = telemetry()
        torch.manual_seed(42)
        tensors = {}
        for cfg in sorted(cases):
            m, n = cases[cfg]["M"], cases[cfg]["N"]
            x = torch.randn((65536, m, n), dtype=torch.bfloat16, device="cuda")
            out = torch.empty((65536, n), dtype=torch.float32, device="cuda")
            scratch = torch.empty(65536 * 128, dtype=torch.uint8, device="cuda")
            tensors[cfg] = (x, out, scratch)
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
            if harness == "repeated":
                m, n = cases[cfg]["M"], cases[cfg]["N"]
                descriptors[key] = driver.tensor_map(tensors[cfg][0].data_ptr(), m, n,
                    binding["shared_layout"]["swizzlingByteWidth"])
        # All runtime conditions warmed identically, in the first frozen round.
        for tag in planned["rounds"][0]["order"]:
            p = tc.condition(tag)
            cfg, h, c = p["config_id"], p["harness"], p["candidate"]
            key = f"{cfg}:{h}:{c}"
            k = kernels[key]
            values, params = k.parameters(tensors[cfg], p["R"], descriptors.get(key))
            for _ in range(tc.WARMUP):
                digest = k.guard()
                k.launch(p["B_RUN"], params, stream)
            driver.call("cuStreamSynchronize", [C.c_void_p], stream)
            record = {"condition_tag": tag, "count": tc.WARMUP, "binary_sha256": digest}
            if h == "canonical":
                expected = tensors[cfg][0][:4].float().amax(dim=1)
                error = (tensors[cfg][1][:4] - expected).abs().max().item()
                record["loaded_ABI_correctness_max_abs_diff"] = error
                if error != 0.0: raise RuntimeError("Archived canonical ABI correctness failure")
            raw["warmups"].append(record)
        first, last = driver.event(), driver.event()
        for round_ in planned["rounds"]:
            for tag in round_["order"]:
                p = tc.condition(tag)
                cfg, h, c = p["config_id"], p["harness"], p["candidate"]
                case, key = cases[cfg], f"{cfg}:{h}:{c}"
                k = kernels[key]
                values, params = k.parameters(tensors[cfg], p["R"], descriptors.get(key))
                samples, guards = [], []
                visit = {**p, "condition_tag": tag, "origin": case["origin"],
                    "stage_b_class": case["final_class"], "invocation": planned["invocation"],
                    "round": round_["round"], "samples_us": samples,
                    "archived_cubin_sha256": k.binding["archive_sha256"],
                    "runtime_loaded_cubin_sha256": k.digest, "sample_launch_sha256": guards,
                    "binary_sha_match": True, "gpu_uuid": env["gpu_uuid"]}
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
        raw["status"] = "VALID_PROTOCOL_RUN"
    except Exception:
        raw["failure_reason"] = traceback.format_exc()
    finally:
        raw["telemetry_end"] = telemetry()
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
    return raw


@app.local_entrypoint()
def main():
    cases, plan, preview = tc.inputs()
    binaries = tc.binary_map(cases)
    tc.OUT.mkdir(parents=True, exist_ok=True)
    if (tc.OUT / "raw_manifest.json").exists():
        raise RuntimeError("Raw data already frozen; never rerun based on scientific outcomes")
    (tc.OUT / "executed_schedule.json").write_bytes(tc.encode(plan))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        for binding in binaries.values():
            path = tc.ROOT / binding["archive_path"]
            z.write(path, binding["archive_path"])
            z.write(path.with_suffix(".ptx"), str(Path(binding["archive_path"]).with_suffix(".ptx")))
    payload = stream.getvalue()
    provenance = generate_provenance()
    provenance["modal_profile"] = os.environ.get("MODAL_PROFILE", "UNSPECIFIED")
    source_file = tc.OUT / "source_bindings.json"
    source_bindings = {"provenance": provenance,
        "archived_payload_sha256": tc.sha(payload), "protected_SHA256": tc.protected_bindings(),
        "warmup_count_per_condition": tc.WARMUP, "timing_kernel_recompilation": False}
    if source_file.exists() and any(tc.OUT.glob("raw_invocation_*.json")):
        previous = json.loads(source_file.read_text())
        tc.require(previous["provenance"]["source_manifest_sha256"] == provenance["source_manifest_sha256"],
                   "Source changed after a valid invocation; STOP before further dispatch")
        tc.require(previous["archived_payload_sha256"] == tc.sha(payload), "Archived payload changed during resume")
        provenance = previous["provenance"]
    else:
        source_file.write_bytes(tc.encode(source_bindings))
    environments, launches = {}, {}
    for planned in plan["invocations"]:
        number = planned["invocation"]
        file = tc.OUT / f"raw_invocation_{number}.json"
        if file.exists():
            raw = json.loads(file.read_text())
            tc.validate_invocation(raw, planned, cases, binaries)
        else:
            for attempt in range(1, 4):
                raw = {}
                try:
                    raw = benchmark.remote(provenance, cases, planned, binaries, payload)
                    tc.validate_invocation(raw, planned, cases, binaries)
                except Exception as exc:
                    raw = {**raw, "status": "INVALID_PROTOCOL_RUN", "invocation": number,
                           "local_protocol_failure": str(exc), "attempt": attempt}
                    invalid = tc.OUT / "invalid_invocations"
                    invalid.mkdir(exist_ok=True)
                    (invalid / f"invocation_{number}_attempt_{attempt}.json").write_bytes(tc.encode(raw))
                    print(f"Invalid entire invocation {number}/{attempt}: {exc}")
                    if attempt == 3: raise RuntimeError("Repeated protocol/infrastructure failure; STOP") from exc
                    continue
                file.write_bytes(tc.encode(raw))
                break
        environments[str(number)] = {"environment": raw["environment"], "start": raw["telemetry_start"], "end": raw["telemetry_end"],
            "dispatch_id": raw["dispatch_id"], "process_identity": raw["process_identity"]}
        launches[str(number)] = raw["loaded_binaries"]
    (tc.OUT / "environment.json").write_bytes(tc.encode({"invocations": environments, "resolved_image_id": image.object_id}))
    (tc.OUT / "launch_bindings.json").write_bytes(tc.encode(launches))
    manifest = {"starting_HEAD": tc.BASELINE, "raw_SHA256": {n: tc.sha((tc.OUT / n).read_bytes()) for n in tc.raw_names(tc.OUT)},
        "protected_SHA256": tc.protected_bindings(), "Stage_B_binary_SHA256": {k: v["archive_sha256"] for k, v in binaries.items()},
        "protocol_SHA256": tc.sha((tc.stage_b.PREREG / "protocol.json").read_bytes()), "expected_counts": tc.EXPECTED_COUNTS}
    (tc.OUT / "raw_manifest.json").write_bytes(tc.encode(manifest))
    print(json.dumps(tc.validate_all(), indent=2))
