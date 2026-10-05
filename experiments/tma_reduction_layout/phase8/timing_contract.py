"""Offline exact-binary, actual-mode, and complete-schedule timing closure."""
from collections import Counter
import io
import math
from pathlib import Path
import zipfile
from experiments.tma_reduction_layout.phase8 import common as c

ROOT, WARMUP = c.ROOT, 3
sha, encode = c.sha, c.encode


def binary_key(cfg, path, candidate):
    return f'{cfg}:{"switch" if path.startswith("switch_") else path}:{candidate}'


def condition(tag):
    p = c.condition(tag)
    return {"config_id": p["case_id"], "harness": p["path"], "candidate": p["candidate"],
            "R": p["mode"], "B_RUN": p["B_RUN"]}


def inputs(stage="stage_c"):
    c.require(stage in ("stage_c","stage_d"), "Known stage")
    frozen = c.read(c.OUT / "stage_b" / ("launch_"+stage+".json"))
    cases,binaries,plan = frozen["cases"],frozen["binaries"],frozen["schedule"]
    c.require(cases and plan == c.schedule(cases), "Complete frozen schedule")
    for binding in binaries.values():
        p = ROOT / binding["archive_path"]
        c.require(sha(p.read_bytes()) == binding["archive_sha256"] and
                  sha(p.with_suffix(".ptx").read_bytes()) == binding["ptx_sha256"], "Frozen CUBIN/PTX")
    return cases,binaries,plan


def validate_invocation(raw,planned,cases,binaries):
    c.require(raw["status"] == "VALID_PROTOCOL_RUN" and raw["invocation"] == planned["invocation"], "Valid invocation")
    c.require(raw["recompilation"] is False and raw["timing_primitive"] == "CUDA_DRIVER_EVENTS_ONE_KERNEL_PER_SAMPLE"
              and raw["compilation_audit"] == {"compiler_subprocess_calls":0,"triton_imported":False,
              "jit_compile_calls":0,"load_input":"ELF_CUBIN_ONLY"}, "Archived ELF only")
    env = raw["environment"]
    c.require("H100" in env["gpu_name"] and env["compute_capability"] == [9,0] and env["gpu_uuid"].startswith("GPU-"), "Strict H100")
    c.require(env["cudaRuntimeGetVersion_return_code"] == 0 and env["source_verification"]["status"] == "PASS"
              and env["source_verification"]["remote_source_subset_sha256"] == env["source_manifest_sha256"], "Source/runtime closure")
    c.require(raw["warmup_policy"] == {"count_per_condition":3,"condition_order":"first frozen round","timed":False}, "Frozen warmups")
    tags = planned["rounds"][0]["order"]
    c.require(set(raw["allocations"]) == set(cases), "Full allocation population")
    for cfg, case in cases.items():
        a = raw["allocations"][cfg]; m,n = case["M"],case["N"]
        c.require(a["input_shape"] == [65536,m,n] and a["input_dtype"] == "torch.bfloat16" and
                  a["input_bytes"] == 65536*m*n*2 and a["output_shape"] == [65536,n] and
                  a["output_dtype"] == "torch.float32" and a["output_bytes"] == 65536*n*4 and
                  a["scratch_bytes"] == 65536*128 and a["all_descriptor_elements_initialized"] is True, "Full initialized descriptor/output/scratch")
    c.require(set(raw["loaded_binaries"]) == set(binaries), "All unique physical binaries")
    for key,b in binaries.items():
        record = raw["loaded_binaries"][key]
        launches = 618 if b["metadata"]["harness"] == "switch" else 309
        c.require(record["archive_sha256"] == record["runtime_loaded_cubin_sha256"] == b["archive_sha256"]
                  and record["ABI"] == b["ABI"] and record["load_api"] == "cuModuleLoadData(archived_file_bytes)"
                  and record["function_name"] == b["metadata"]["function_name"] and
                  record["sha_guard_count"] == record["launch_count"] == launches and
                  record["dynamic_smem_bytes"] == b["metadata"]["dynamic_smem_bytes"], "Unique ELF load and all guards")
    descriptor_keys = {key for key,b in binaries.items() if b["metadata"]["harness"] in ("host_canonical","switch")}
    c.require(set(raw["tensor_maps"]) == descriptor_keys, "Exact host-descriptor population")
    for key,desc in raw["tensor_maps"].items():
        case = cases[key.split(":")[0]];m,n = case["M"],case["N"]
        sw = binaries[key]["shared_layout"]["swizzlingByteWidth"]
        c.require(desc["global_dims_n_m_b"] == [n,m,65536] and desc["global_strides_bytes"] == [n*2,m*n*2]
                  and desc["message_box_n_m_b"] == [min(n,sw//2),min(m,256),1]
                  and desc["element_strides"] == [1,1,1] and desc["swizzling_bytes"] == sw, "Unshrunk native tensor map")
    c.require([w["condition_tag"] for w in raw["warmups"]] == tags, "Actual warmup order")
    for warm in raw["warmups"]:
        p=condition(warm["condition_tag"]);key=binary_key(p["config_id"],p["harness"],p["candidate"])
        c.require(warm["count"] == 3 and warm["binary_sha256"] == binaries[key]["archive_sha256"]
                  and warm["runtime_mode_argument"] == p["R"] and warm["loaded_ABI_correctness_max_abs_diff"] == 0,
                  "Every actual-mode warmup correctness")
    expected = [(r["round"],tag) for r in planned["rounds"] for tag in r["order"]]
    c.require([(v["round"],v["condition_tag"]) for v in raw["visits"]] == expected, "All complete ordered visits")
    counts=Counter()
    for visit in raw["visits"]:
        p=condition(visit["condition_tag"]);key=binary_key(p["config_id"],p["harness"],p["candidate"])
        b=binaries[key];record=raw["loaded_binaries"][key];digest=b["archive_sha256"]
        c.require(all(visit[name] == value for name,value in p.items()) and visit["invocation"] == planned["invocation"]
                  and visit["gpu_uuid"] == env["gpu_uuid"] and visit["stage_b_class"] == cases[p["config_id"]]["final_class"], "Actual condition identity/class")
        c.require(visit["archived_cubin_sha256"] == visit["runtime_loaded_cubin_sha256"] == digest
                  and visit["sample_launch_sha256"] == [digest]*10 and visit["binary_sha_match"] is True,
                  "Every sample archive/load SHA")
        c.require(visit["runtime_mode_argument"] == p["R"] and visit["runtime_module_identity"] == record["module_identity"]
                  and visit["runtime_function_identity"] == record["function_identity"], "Both runtime modes share physical module/function")
        c.require(len(visit["samples_us"]) == 10 and all(math.isfinite(v) and v>0 for v in visit["samples_us"]), "Positive finite scalar samples")
        counts[visit["condition_tag"]] += 10
    c.require(set(counts) == set(tags) and set(counts.values()) == {100}, "Median100 complete domain")
    calls=raw["cuda_call_counts"]
    c.require(calls["cuLaunchKernel"] == len(tags)*103 and calls["cuEventElapsedTime"] == len(tags)*100
              and calls["cuEventRecord"] == len(tags)*200 and calls["cuEventSynchronize"] == len(tags)*100
              and calls["cuModuleLoadData"] == calls["cuModuleUnload"] == len(binaries)
              and raw["cuda_return_codes"] == [0], "Exact checked CUDA calls")
    c.require(all(Path(command[0]).name == "nvidia-smi" for command in raw["subprocess_commands"]), "Telemetry-only subprocesses")
    c.require(raw["remote_evidence"]["path"].endswith(raw["dispatch_id"]+".json"), "Durable original return")
    return {"invocation":planned["invocation"],"samples":sum(counts.values()),"gpu_uuid":env["gpu_uuid"],
            "physical_binaries":len(binaries),"same_module_and_function_across_modes":True}


def validate_raw(stage):
    root=c.OUT/stage;c.validate_freeze(root)
    cases,binaries,plan=inputs(stage)
    c.require(c.read(root/"binary_bindings.json") == binaries and c.read(root/"executed_schedule.json") ==
              {**plan,"executed":True}, "Frozen actual bindings/schedule")
    source=c.read(root/"source_bindings.json");payload=(root/"archived_timing_payload.zip").read_bytes()
    c.require(source["payload_SHA256"] == sha(payload) and source["protocol_SHA256"] ==
              sha((c.OUT/"stage_a/protocol.json").read_bytes()), "Protocol/payload closure")
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names={name for b in binaries.values() for name in (b["archive_path"],str(Path(b["archive_path"]).with_suffix(".ptx")))}
        c.require(set(z.namelist()) == names and len(z.namelist()) == len(names), "Unique archived payload")
        for name in names:c.require(z.read(name) == (ROOT/name).read_bytes(), "Actual payload/archive bytes")
    with zipfile.ZipFile(root/"uploaded_source.zip") as z:
        c.require(set(z.namelist()) == set(source["provenance"]["source_manifest"]), "Complete timing source snapshot")
        for name,digest in source["provenance"]["source_manifest"].items():c.require(sha(z.read(name)) == digest, "Original uploaded source bytes")
    records=[];raws=[]
    for planned in plan["invocations"]:
        raw=c.read(root/f'raw_invocation_{planned["invocation"]}.json');raws.append(raw)
        original=[p for p in (root/"attempts").glob("*/original_return.json") if c.read(p)["dispatch_id"] == raw["dispatch_id"]]
        c.require(len(original) == 1 and original[0].read_bytes() == (root/f'raw_invocation_{planned["invocation"]}.json').read_bytes(), "Unmodified original return")
        c.require(raw["environment"]["archived_payload_sha256"] == sha(payload) and
                  raw["environment"]["source_manifest_sha256"] == source["provenance"]["source_manifest_sha256"], "Actual worker payload/source")
        records.append(validate_invocation(raw,planned,cases,binaries))
    c.require(len({r["dispatch_id"] for r in raws}) == len({r["process_identity"] for r in raws}) == 3
              and sum(r["samples"] for r in records) == plan["samples"], "Three separate processes and sample totals")
    result={"status":"PASS","stage":stage,"records":records,"total_samples":plan["samples"],
            "physical_UUIDs":len({r["gpu_uuid"] for r in records}),"prior_files_unchanged":len(c.protect())}
    c.write(root/"validation.json",result)
    return result


if __name__ == "__main__":
    import sys
    print(validate_raw(sys.argv[1]))
