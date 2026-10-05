"""Three per-target processes load archived ELF bytes; no compiler/JIT imports."""
import io
import os
from pathlib import Path
import uuid
import zipfile
import modal
from experiments.tma_reduction_layout.phase9 import common as c, timing_contract as tc
from experiments.tma_reduction_layout.modal_runner import torch_and_deps_image, build_cache_volume
from experiments.tma_reduction_layout.source_provenance import (
    MODAL_SOURCE_IGNORE_PATTERNS, generate_provenance, verify_remote_source_manifest)

LOCAL_PROVENANCE = generate_provenance() if modal.is_local() else None
app = modal.App("triton-phase9-cross-architecture-exact-timing")
evidence_volume = modal.Volume.from_name("triton-phase9-evidence",create_if_missing=True)
image = torch_and_deps_image.env({"PYTHONPATH":"/opt/triton-src"}).add_local_dir(
    c.ROOT,remote_path="/opt/triton-src",copy=True,
    ignore=MODAL_SOURCE_IGNORE_PATTERNS + ["experiments/tma_reduction_layout/phase4/results/**"])


def benchmark_impl(target,provenance,cases,planned,binaries,payload):
    import ctypes as C
    import datetime
    import math
    import socket
    import subprocess
    import sys
    import traceback
    import torch
    from experiments.tma_reduction_layout.phase8.launch import Driver,Loaded
    raw = {"target":target,"invocation":planned["invocation"],"status":"INVALID_PROTOCOL_RUN",
        "dispatch_id":modal.current_function_call_id(),"process_identity":f'{socket.gethostname()}:{os.getpid()}:{os.environ.get("MODAL_TASK_ID","UNAVAILABLE")}',
        "recompilation":False,"timing_primitive":"CUDA_DRIVER_EVENTS_ONE_KERNEL_PER_SAMPLE",
        "warmups":[],"visits":[],"loaded_binaries":{},"allocations":{},
        "compilation_audit":{"compiler_subprocess_calls":0,"triton_imported":False,"jit_compile_calls":0,"load_input":"ELF_CUBIN_ONLY"}}
    commands = []
    def audit(event,args):
        if event == "subprocess.Popen":
            commands.append(list(args[1]) if isinstance(args[1],(list,tuple)) else args[1])
            if Path(args[0]).name != "nvidia-smi":
                raw["compilation_audit"]["compiler_subprocess_calls"] += 1
                raise RuntimeError("Disallowed subprocess in timing worker")
    sys.addaudithook(audit)
    def telemetry():
        fields = "uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,driver_version"
        result = {"timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            values = subprocess.check_output(["nvidia-smi","--query-gpu="+fields,"--format=csv,noheader,nounits"],text=True).strip().split(",")
            result.update(dict(zip(fields.split(","),[v.strip() for v in values])))
        except Exception as exc: result["unavailable"] = str(exc)
        return result
    driver,kernels = None,{}
    try:
        verification = verify_remote_source_manifest(provenance)
        archive = Path("/tmp/phase9_archived_binaries")
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            for name in z.namelist(): c.require(not Path(name).is_absolute() and ".." not in Path(name).parts,"Unsafe payload path")
            z.extractall(archive)
        torch.cuda.init()
        name,cc = torch.cuda.get_device_name(0),list(torch.cuda.get_device_capability(0))
        c.require(c.TARGETS[target]["name_contains"] in name and cc == c.TARGETS[target]["cc"],f"Strict target mismatch: {name}, {cc}")
        driver = Driver()
        env = driver.environment()
        env.update({"gpu_name":name,"compute_capability":cc,"torch":torch.__version__,"torch_cuda":torch.version.cuda,
            "SM_count":torch.cuda.get_device_properties(0).multi_processor_count,
            "modal_image_id":os.environ.get("MODAL_IMAGE_ID","UNAVAILABLE"),"modal_profile":provenance["modal_profile"],
            "source_verification":verification,"source_HEAD":provenance["git_head_sha"],"source_manifest_sha256":provenance["source_manifest_sha256"],
            "archived_payload_sha256":c.sha(payload)})
        raw["environment"] = env
        raw["telemetry_start"] = telemetry()
        torch.manual_seed(42)
        tensors,unique = {},{}
        needed_cfg = {key.split(":")[0] for key in binaries}
        for cfg in sorted(needed_cfg):
            m,n = cases[cfg]["M"],cases[cfg]["N"]
            copy_needed = any(key.startswith(cfg+":") and key.split(":")[1] != "reduction" for key in binaries)
            if (m,n) not in unique:
                x = torch.randn((c.B_DESC,m,n),dtype=torch.bfloat16,device="cuda")
                red_out = torch.empty((c.B_DESC,n),dtype=torch.float32,device="cuda")
                scratch = torch.empty(c.B_DESC*128,dtype=torch.uint8,device="cuda")
                unique[m,n] = [x,red_out,None,scratch]
            if copy_needed and unique[m,n][2] is None:
                unique[m,n][2] = torch.empty((c.B_DESC,m,n),dtype=torch.bfloat16,device="cuda")
            x,red_out,copy_out,scratch = unique[m,n]
            tensors[cfg] = unique[m,n]
            raw["allocations"][cfg] = {"input_shape":list(x.shape),"input_bytes":x.numel()*2,"input_pointer":x.data_ptr(),
                "reduction_output_shape":list(red_out.shape),"reduction_output_bytes":red_out.numel()*4,"reduction_output_pointer":red_out.data_ptr(),
                "copy_output_shape":list(copy_out.shape) if copy_out is not None else None,
                "copy_output_bytes":copy_out.numel()*2 if copy_out is not None else None,
                "copy_output_pointer":copy_out.data_ptr() if copy_out is not None else None,
                "scratch_bytes":scratch.numel(),"scratch_pointer":scratch.data_ptr(),"all_descriptor_elements_initialized":True}
        torch.cuda.synchronize()
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        for key,binding in binaries.items():
            c.require(binding["metadata"]["target"] == target,"Wrong architecture payload")
            path = archive / binding["archive_path"]
            ptx = path.with_suffix(".ptx").read_bytes()
            c.require(c.sha(ptx) == binding["ptx_sha256"],"PTX ABI SHA mismatch")
            kernels[key] = Loaded(driver,archive,binding,ptx.decode())
        def arguments(tag):
            p = c.condition(tag)
            key = ":".join(tag.split(":")[:3])
            x,red_out,copy_out,scratch = tensors[p["case_id"]]
            out = red_out if p["harness"] == "reduction" else copy_out
            values,params = kernels[key].parameters((x,out,scratch),None,None)
            return p,key,x,out,values,params
        for tag in planned["rounds"][0]["order"]:
            p,key,x,out,values,params = arguments(tag)
            k = kernels[key]
            out[:4].fill_(float("nan"))
            for _ in range(3):
                digest = k.guard()
                k.launch(p["B_RUN"],params,stream)
            driver.call("cuStreamSynchronize",[C.c_void_p],stream)
            expected = x[:4].float().amax(dim=1) if p["harness"] == "reduction" else x[:4].float()
            error = (out[:4].float()-expected).abs().max().item()
            record = {"condition_tag":tag,"count":3,"binary_sha256":digest,"loaded_ABI_correctness_max_abs_diff":error,
                "runtime_module_identity":hex(k.module.value),"runtime_function_identity":hex(k.function.value),
                "gpu_uuid":env["gpu_uuid"],"actual_parameter_values":[v.value for v in values]}
            raw["warmups"].append(record)
            c.require(error == 0,"Archived-ABI correctness failed")
        first,last = driver.event(),driver.event()
        for round_ in planned["rounds"]:
            for tag in round_["order"]:
                p,key,x,out,values,params = arguments(tag)
                k = kernels[key]
                visit = {**p,"condition_tag":tag,"round":round_["round"],"samples_us":[],"sample_launch_sha256":[],
                    "binary_sha256":k.digest,"gpu_uuid":env["gpu_uuid"],
                    "runtime_module_identity":hex(k.module.value),"runtime_function_identity":hex(k.function.value),
                    "actual_parameter_values":[v.value for v in values]}
                raw["visits"].append(visit)
                for _ in range(10):
                    digest = k.guard()
                    driver.record(first,stream)
                    k.launch(p["B_RUN"],params,stream)
                    driver.record(last,stream)
                    driver.synchronize_event(last)
                    sample = driver.elapsed_us(first,last)
                    c.require(math.isfinite(sample) and sample > 0,"Non-positive/non-finite event sample")
                    visit["samples_us"].append(sample)
                    visit["sample_launch_sha256"].append(digest)
            print(f'{target} invocation{planned["invocation"]}: frozen round{round_["round"]}/10',flush=True)
        driver.call("cuEventDestroy_v2",[C.c_void_p],first)
        driver.call("cuEventDestroy_v2",[C.c_void_p],last)
        if any(n == "triton" or n.startswith("triton.") for n in sys.modules):
            raw["compilation_audit"]["triton_imported"] = True
            raise RuntimeError("Triton imported in archived-binary timing worker")
        raw["status"] = "VALID_PROTOCOL_RUN"
    except Exception:
        raw["failure_reason"] = traceback.format_exc()
        print(raw["failure_reason"],flush=True)
    finally:
        raw["telemetry_end"] = telemetry()
        raw["subprocess_commands"] = commands
        if driver is not None:
            raw["loaded_binaries"] = {key:k.record_binding() for key,k in kernels.items()}
            for k in kernels.values():
                try: driver.call("cuModuleUnload",[C.c_void_p],k.module)
                except Exception:
                    raw["status"] = "INVALID_PROTOCOL_RUN"
                    raw["failure_reason"] = traceback.format_exc()
            raw["cuda_call_counts"] = dict(driver.counts)
            raw["cuda_return_codes"] = sorted(driver.codes)
    path = Path("/evidence/stage_c") / target / (raw["dispatch_id"]+".json")
    raw["remote_evidence"] = {"volume":"triton-phase9-evidence","path":str(path),"profile":provenance["modal_profile"]}
    c.write(path,raw)
    evidence_volume.commit()
    return raw


@app.function(image=image,gpu="H100!:1",timeout=3600,single_use_containers=True,volumes={"/cache":build_cache_volume,"/evidence":evidence_volume})
def benchmark_sm90(prov,cases,planned,binaries,payload):
    return benchmark_impl("sm90",prov,cases,planned,binaries,payload)


@app.function(image=image,gpu="B200:1",timeout=3600,single_use_containers=True,volumes={"/cache":build_cache_volume,"/evidence":evidence_volume})
def benchmark_sm100(prov,cases,planned,binaries,payload):
    return benchmark_impl("sm100",prov,cases,planned,binaries,payload)


@app.function(image=image,gpu="RTX-PRO-6000:1",timeout=3600,single_use_containers=True,volumes={"/cache":build_cache_volume,"/evidence":evidence_volume})
def benchmark_sm120(prov,cases,planned,binaries,payload):
    return benchmark_impl("sm120",prov,cases,planned,binaries,payload)


@app.local_entrypoint()
def main(target: str = "all"):
    c.protect()
    functions = {"sm90":benchmark_sm90,"sm100":benchmark_sm100,"sm120":benchmark_sm120}
    prov = LOCAL_PROVENANCE
    prov["modal_profile"] = os.environ.get("MODAL_PROFILE","UNSPECIFIED")
    for arch in c.TARGETS if target == "all" else [target]:
        cases,binaries,plan = tc.inputs(arch)
        c.require(bool(binaries),"No eligible architecture binaries")
        root = c.OUT / "stage_c" / arch
        root.mkdir(parents=True,exist_ok=True)
        c.require(not (root / "raw_manifest.json").exists(),"Timing stage already frozen; no outcome reruns")
        stream = io.BytesIO()
        with zipfile.ZipFile(stream,"w",zipfile.ZIP_DEFLATED) as z:
            for binding in binaries.values():
                path = c.ROOT / binding["archive_path"]
                z.write(path,binding["archive_path"])
                z.write(path.with_suffix(".ptx"),str(Path(binding["archive_path"]).with_suffix(".ptx")))
        payload = stream.getvalue()
        if not (root / "source_bindings.json").exists():
            (root / "archived_timing_payload.zip").write_bytes(payload)
            c.write(root / "source_bindings.json",{"provenance":prov,"payload_SHA256":c.sha(payload),
                "protocol_SHA256":c.sha((c.OUT / "stage_a/protocol.json").read_bytes()),"timing_kernel_recompilation":False})
            c.write(root / "binary_bindings.json",binaries)
            c.write(root / "executed_schedule.json",{**plan,"executed":True})
            with zipfile.ZipFile(root / "uploaded_source.zip","w",zipfile.ZIP_DEFLATED) as z:
                for name in prov["source_manifest"]: z.write(c.ROOT / name,name)
        else:
            c.require(c.read(root / "source_bindings.json")["provenance"]["source_manifest_sha256"] == prov["source_manifest_sha256"],"Source changed after timing dispatch")
        for planned in plan["invocations"]:
            destination = root / f'raw_invocation_{planned["invocation"]}.json'
            if destination.exists():
                tc.validate_invocation(c.read(destination),planned,cases,binaries,arch)
                continue
            for attempt in range(1,4):
                directory = root / "attempts" / uuid.uuid4().hex
                directory.mkdir(parents=True,exist_ok=False)
                dispatch = {"target":arch,"invocation":planned["invocation"],"attempt_in_session":attempt,"profile":prov["modal_profile"],"source_manifest_SHA256":prov["source_manifest_sha256"]}
                c.write(directory / "dispatch.json",dispatch)
                try:
                    call = functions[arch].spawn(prov,cases,planned,binaries,payload)
                    c.write(directory / "dispatch.json",{**dispatch,"call_id":call.object_id,"image_id":image.object_id})
                    raw = call.get()
                    with (directory / "original_return.json").open("xb") as f: f.write(c.encode(raw))
                    tc.validate_invocation(raw,planned,cases,binaries,arch)
                except Exception as exc:
                    c.write(directory / "local_failure.json",{"error":str(exc),"original_return_preserved":(directory / "original_return.json").exists(),"profile":prov["modal_profile"]})
                    cause = str(exc)
                    if (directory / "original_return.json").exists(): cause += c.read(directory / "original_return.json").get("failure_reason","")
                    if "quota" in cause.lower() or "credit" in cause.lower() or "billing" in cause.lower(): raise
                    allowed = ("out of memory","timeout","interrupted","Connection","RPC","Unavailable")
                    if not any(x.lower() in cause.lower() for x in allowed) or attempt == 3: raise
                    print(f"Whole-invocation infrastructure repair {arch}/{planned['invocation']}/{attempt}",flush=True)
                    continue
                with destination.open("xb") as f: f.write(c.encode(raw))
                print(f"Accepted {arch} invocation{planned['invocation']}; complete original retained",flush=True)
                break
    print("Raw timing complete. Close CLI logs before freezing; no outcome analysis in worker.")
