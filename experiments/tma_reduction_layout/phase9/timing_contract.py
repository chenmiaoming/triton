"""Validate exact architecture binaries and every predeclared event sample."""
import argparse
import math
from experiments.tma_reduction_layout.phase9 import common as c


def inputs(target):
    gate = c.read(c.OUT / "stage_b" / target / "gate.json")
    cases = {x["case_id"]:x for x in c.protocol()["cases"]}
    return cases,gate["binaries"],gate["schedule"]


def validate_invocation(raw,planned,cases,binaries,target):
    c.require(raw["status"] == "VALID_PROTOCOL_RUN", raw.get("failure_reason","Invalid protocol run"))
    c.require(raw["invocation"] == planned["invocation"] and raw["target"] == target,"Actual invocation/architecture")
    env = raw["environment"]
    spec = c.TARGETS[target]
    c.require(env["compute_capability"] == spec["cc"] and spec["name_contains"] in env["gpu_name"] and env["gpu_uuid"].startswith("GPU-"),"Actual worker name/CC/UUID")
    c.require(raw["timing_primitive"] == "CUDA_DRIVER_EVENTS_ONE_KERNEL_PER_SAMPLE" and not raw["recompilation"],"One archived kernel event sample")
    audit = raw["compilation_audit"]
    c.require(not audit["triton_imported"] and audit["compiler_subprocess_calls"] == audit["jit_compile_calls"] == 0 and audit["load_input"] == "ELF_CUBIN_ONLY","Compiler-free timing")
    c.require(raw["cuda_return_codes"] == [0],"Every CUDA return checked")
    c.require(set(raw["loaded_binaries"]) == set(binaries),"Every eligible binary loaded, no replacement")
    first_order = planned["rounds"][0]["order"]
    c.require([x["condition_tag"] for x in raw["warmups"]] == first_order,"Frozen warmup order")
    expected_order = [(r["round"],tag) for r in planned["rounds"] for tag in r["order"]]
    c.require([(v["round"],v["condition_tag"]) for v in raw["visits"]] == expected_order,"Frozen visit order")
    for cfg in {c.condition(tag)["case_id"] for tag in first_order}:
        a = raw["allocations"][cfg]
        m,n = cases[cfg]["M"],cases[cfg]["N"]
        c.require(a["input_shape"] == [c.B_DESC,m,n] and a["input_bytes"] == c.B_DESC*m*n*2 and a["all_descriptor_elements_initialized"],"Full initialized descriptor bounds")
        c.require(a["reduction_output_shape"] == [c.B_DESC,n] and a["reduction_output_bytes"] == c.B_DESC*n*4,"FP32 full reduction output")
        if a["copy_output_shape"] is not None:
            c.require(a["copy_output_shape"] == [c.B_DESC,m,n] and a["copy_output_bytes"] == c.B_DESC*m*n*2,"BF16 copy-control output")
        c.require(a["scratch_bytes"] >= c.B_DESC*128,"Descriptor scratch extent")
    for record in raw["warmups"] + raw["visits"]:
        p = c.condition(record["condition_tag"])
        key = ":".join(record["condition_tag"].split(":")[:3])
        b = binaries[key]
        loaded = raw["loaded_binaries"][key]
        c.require(record["binary_sha256"] == b["archive_sha256"] == loaded["runtime_loaded_cubin_sha256"],"Actual loaded/archive SHA")
        c.require(record["runtime_module_identity"] == loaded["module_identity"] and record["runtime_function_identity"] == loaded["function_identity"],"Actual loaded module/function identity")
        c.require(record["gpu_uuid"] == env["gpu_uuid"],"Each visit UUID binding")
        a = raw["allocations"][p["case_id"]]
        out = a["reduction_output_pointer"] if p["harness"] == "reduction" else a["copy_output_pointer"]
        case = cases[p["case_id"]]
        c.require(record["actual_parameter_values"] == [a["input_pointer"],out,case["M"]*case["N"],case["N"],a["scratch_pointer"],0],"Actual marshalled kernel arguments")
        if "samples_us" in record:
            c.require(all(record[k] == v for k,v in p.items()),"Actual visit condition")
            samples = record["samples_us"]
            c.require(len(samples) == 10 and all(math.isfinite(x) and x > 0 for x in samples),"Ten finite positive event samples")
            c.require(record["sample_launch_sha256"] == [b["archive_sha256"]]*10,"Every timed launch SHA guard")
        else:
            c.require(record["count"] == 3 and record["loaded_ABI_correctness_max_abs_diff"] == 0,"Untimed archived-ABI correctness")
    for key,b in binaries.items():
        loaded = raw["loaded_binaries"][key]
        c.require(loaded["ABI"] == b["ABI"] and loaded["load_api"] == "cuModuleLoadData(archived_file_bytes)","Exact ABI/module loading")
        c.require(loaded["launch_count"] == loaded["sha_guard_count"] == 309,"100 timed+3 warmup launches for each of three grids")
        c.require(loaded["dynamic_smem_bytes"] == b["metadata"]["dynamic_smem_bytes"],"Archived dynamic shared memory")
    count = len(first_order)*100
    calls = raw["cuda_call_counts"]
    c.require(calls["cuLaunchKernel"] == len(binaries)*309 and calls["cuEventRecord"] == count*2 and calls["cuEventElapsedTime"] == calls["cuEventSynchronize"] == count,"Exact event/kernel call totals")


def validate_target(target):
    cases,binaries,plan = inputs(target)
    root = c.OUT / "stage_c" / target
    c.require(c.read(root / "executed_schedule.json") == {**plan,"executed":True},"Timing schedule rederived")
    c.require(c.read(root / "binary_bindings.json") == binaries,"Timing binary bindings")
    source = c.read(root / "source_bindings.json")
    c.require(source["protocol_SHA256"] == c.sha((c.OUT / "stage_a/protocol.json").read_bytes()),"Timing protocol SHA")
    c.require(source["payload_SHA256"] == c.sha((root / "archived_timing_payload.zip").read_bytes()),"Timing payload SHA")
    import zipfile
    with zipfile.ZipFile(root / "uploaded_source.zip") as z:
        c.require(set(z.namelist()) == set(source["provenance"]["source_manifest"]),"Timing source archive inventory")
        for name,digest in source["provenance"]["source_manifest"].items():
            c.require(c.sha(z.read(name)) == digest,"Timing source archive byte binding")
    with zipfile.ZipFile(root / "archived_timing_payload.zip") as z:
        for binding in binaries.values():
            c.require(c.sha(z.read(binding["archive_path"])) == binding["archive_sha256"],"Timing ZIP actual CUBIN SHA")
            c.require(c.sha(z.read(str(__import__('pathlib').Path(binding["archive_path"]).with_suffix('.ptx')))) == binding["ptx_sha256"],"Timing ZIP actual PTX SHA")
    uuids,processes = set(),set()
    for planned in plan["invocations"]:
        raw = c.read(root / f'raw_invocation_{planned["invocation"]}.json')
        validate_invocation(raw,planned,cases,binaries,target)
        matches = [p for p in (root / "attempts").glob("*/original_return.json") if p.read_bytes() == c.encode(raw)]
        c.require(len(matches) == 1,"Every accepted raw invocation has one complete original return")
        c.require(raw["environment"]["source_manifest_sha256"] == source["provenance"]["source_manifest_sha256"],"Frozen timing worker source")
        uuids.add(raw["environment"]["gpu_uuid"])
        processes.add(raw["process_identity"])
    c.require(len(processes) == 3,"Three separate timing processes")
    return {"target":target,"processes":3,"physical_GPU_UUIDs":sorted(uuids),"samples":plan["samples"],"eligible_pairs":len(binaries)//2,"status":"PASS"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target",default="all")
    args = p.parse_args()
    c.protect()
    for target in c.TARGETS if args.target == "all" else [args.target]:
        print(c.encode(validate_target(target)).decode(),end="")


if __name__ == "__main__":
    main()
