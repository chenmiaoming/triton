"""Generic raw closure for supplementary and fresh Phase 6 archived binaries."""
from collections import Counter
import copy
import io
import json
import math
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.tma_reduction_layout.phase6 import common as c
from experiments.tma_reduction_layout.phase6 import contracts

ROOT = c.ROOT
OUT = c.OUT / "stage_c"
WARMUP = 3
sha, encode, require = c.sha, c.encode, c.require


def condition(tag):
    cfg, harness, candidate, r, b = tag.split(":")
    return {"config_id": cfg, "harness": harness, "candidate": candidate,
            "R": None if r == "RNone" else int(r[1:]), "B_RUN": int(b[1:])}


def inputs(stage="stage_c"):
    if stage == "stage_c":
        frozen = c.read(contracts.DEST / "contracts.json")
        cases, binaries, plan = frozen["cases"], frozen["binaries"], frozen["schedule"]
    else:
        require(stage == "stage_d", "Known frozen stage")
        frozen = c.read(c.OUT / "stage_d_gate/launch_contract.json")
        cases, binaries, plan = frozen["cases"], frozen["binaries"], frozen["schedule"]
    if stage == "stage_c":
        expected_schedule = contracts.schedule(cases)
    else:
        from experiments.tma_reduction_layout.phase6 import fresh_contract
        expected_schedule = fresh_contract.schedule(cases)
    require(plan == expected_schedule, "Frozen deterministic complete schedule")
    for key, binding in binaries.items():
        path = ROOT / binding["archive_path"]
        require(sha(path.read_bytes()) == binding["archive_sha256"] and
                sha(path.with_suffix(".ptx").read_bytes()) == binding["ptx_sha256"], "Archived binary/PTX SHA")
    return cases, binaries, plan


def validate_invocation(raw, planned, cases, binaries):
    number = planned["invocation"]
    require(raw["status"] == "VALID_PROTOCOL_RUN" and raw["invocation"] == number, "Valid invocation identity")
    require(raw["recompilation"] is False and raw["timing_primitive"] == "CUDA_DRIVER_EVENTS_ONE_KERNEL_PER_SAMPLE",
            "Exact ELF-only scalar timing")
    require(raw["compilation_audit"] == {"compiler_subprocess_calls": 0, "triton_imported": False,
            "jit_compile_calls": 0, "load_input": "ELF_CUBIN_ONLY"}, "No compiler/JIT activity")
    env = raw["environment"]
    require("H100" in env["gpu_name"] and env["compute_capability"] == [9, 0] and env["gpu_uuid"].startswith("GPU-"), "Strict H100 identity")
    require(env["CUDA_runtime_version"] > 0 and env["cudaRuntimeGetVersion_return_code"] == 0
            and env["source_verification"]["remote_source_subset_sha256"] == env["source_manifest_sha256"], "Runtime/source closure")
    require(raw["warmup_policy"] == {"count_per_condition": 3, "condition_order": "first frozen round", "timed": False}, "Frozen untimed warmups")
    tags = planned["rounds"][0]["order"]
    require([w["condition_tag"] for w in raw["warmups"]] == tags, "Exact warmup order")
    require(set(raw["allocations"]) == set(cases), "All eligible full allocations")
    for cfg, case in cases.items():
        allocation = raw["allocations"][cfg]
        m, n = case["M"], case["N"]
        require(allocation["input_shape"] == [65536, m, n] and allocation["input_dtype"] == "torch.bfloat16"
                and allocation["input_bytes"] == 65536*m*n*2 and allocation["all_descriptor_elements_initialized"] is True
                and allocation["output_shape"] == [65536, n] and allocation["output_dtype"] == "torch.float32"
                and allocation["output_bytes"] == 65536*n*4 and allocation["scratch_bytes"] == 65536*128,
                "Full descriptor allocation: " + cfg)
    descriptor_keys = {key for key in binaries if key.split(":")[1] != "canonical"}
    require(set(raw["tensor_maps"]) == descriptor_keys, "Exact single/repeated descriptor domain")
    for key, descriptor in raw["tensor_maps"].items():
        cfg = key.split(":")[0]; m, n = cases[cfg]["M"], cases[cfg]["N"]
        sw = binaries[key]["shared_layout"]["swizzlingByteWidth"]
        require(descriptor["global_dims_n_m_b"] == [n, m, 65536] and
                descriptor["global_strides_bytes"] == [n*2, m*n*2] and
                descriptor["message_box_n_m_b"] == [min(n, sw//2), min(m, 256), 1] and
                descriptor["swizzling_bytes"] == sw, "Native host TMA message box without descriptor shrink")
    require(set(raw["loaded_binaries"]) == set(binaries), "All admitted binary modules")
    for key, record in raw["loaded_binaries"].items():
        b = binaries[key]; h = key.split(":")[1]
        require(record["archive_sha256"] == record["runtime_loaded_cubin_sha256"] == b["archive_sha256"]
                and record["ABI"] == b["ABI"] and record["function_name"] == b["metadata"]["function_name"]
                and record["load_api"] == "cuModuleLoadData(archived_file_bytes)", "Exact CUBIN/ABI load closure")
        launches = 3*(5 if h == "repeated" else 1)*103
        require(record["sha_guard_count"] == record["launch_count"] == launches and
                record["dynamic_smem_bytes"] == b["metadata"]["dynamic_smem_bytes"] and
                record["dynamic_smem_optin"] == (b["metadata"]["dynamic_smem_bytes"] > 49152), "Every launch SHA guarded/SMEM opt-in")
    for warmup in raw["warmups"]:
        p = condition(warmup["condition_tag"])
        key = f'{p["config_id"]}:{p["harness"]}:{p["candidate"]}'
        require(warmup["count"] == 3 and warmup["binary_sha256"] == binaries[key]["archive_sha256"], "Warmup count/binary")
        if p["harness"] != "repeated":
            require(warmup["loaded_ABI_correctness_max_abs_diff"] == 0, "Loaded canonical/single reduction correctness")
    expected = [(r["round"], tag) for r in planned["rounds"] for tag in r["order"]]
    visits = raw["visits"]
    require([(v["round"], v["condition_tag"]) for v in visits] == expected, "Complete frozen visit order")
    counts = Counter()
    for visit in visits:
        p = condition(visit["condition_tag"])
        require(all(visit[k] == v for k, v in p.items()) and visit["invocation"] == number
                and visit["gpu_uuid"] == env["gpu_uuid"], "Exact case/R/B/invocation identity")
        key = f'{p["config_id"]}:{p["harness"]}:{p["candidate"]}'
        digest = binaries[key]["archive_sha256"]
        require(visit["archived_cubin_sha256"] == visit["runtime_loaded_cubin_sha256"] == digest
                and visit["sample_launch_sha256"] == [digest]*10 and visit["binary_sha_match"] is True, "All sample binary guards")
        require(len(visit["samples_us"]) == 10 and all(math.isfinite(x) and x > 0 for x in visit["samples_us"]), "Ten positive finite one-kernel samples")
        counts[visit["condition_tag"]] += 10
    require(set(counts) == set(tags) and set(counts.values()) == {100}, "100 samples / every frozen condition")
    calls = raw["cuda_call_counts"]
    require(calls["cuLaunchKernel"] == len(tags)*103 and calls["cuEventElapsedTime"] == len(tags)*100
            and calls["cuEventRecord"] == len(tags)*200 and calls["cuEventSynchronize"] == len(tags)*100
            and calls["cuModuleLoadData"] == calls["cuModuleUnload"] == len(binaries)
            and all(code == 0 for code in raw["cuda_return_codes"]), "Exact one-kernel event counts / checked CUDA codes")
    require(raw["subprocess_commands"] and all(Path(cmd[0]).name == "nvidia-smi" for cmd in raw["subprocess_commands"]), "Only telemetry subprocesses")
    require(raw["remote_evidence"]["path"].endswith(raw["dispatch_id"] + ".json"), "Durable remote return before retrieval")
    return {"conditions": len(tags), "samples": sum(counts.values()), "gpu_uuid": env["gpu_uuid"]}


DERIVED = {"raw_validation.json", "results.json", "summary.md", "analysis_validation.json", "validator_suite.json"}


def raw_inventory(root):
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name not in DERIVED and p.name != "raw_manifest.json"}


def validate_raw(stage="stage_c"):
    root = c.OUT / stage
    cases, binaries, plan = inputs(stage)
    require(c.read(root / "executed_schedule.json") == {**plan, "executed": True}, "Frozen executed schedule")
    manifest = c.read(root / "raw_manifest.json")
    require(manifest["files"] == raw_inventory(root), "All raw bytes frozen")
    require(manifest["prior_inventory_SHA256"] == sha(encode(c.inventory())), "All baseline files unchanged")
    require(c.read(root / "binary_bindings.json") == binaries, "Actual full launch bindings")
    source = c.read(root / "source_bindings.json")
    with zipfile.ZipFile(root / "uploaded_source.zip") as z:
        require(set(z.namelist()) == set(source["provenance"]["source_manifest"]), "Source snapshot inventory")
        for name, digest in source["provenance"]["source_manifest"].items():
            require(sha(z.read(name)) == digest, "Snapshot source bytes")
    raws, records = [], []
    for planned in plan["invocations"]:
        raw = c.read(root / f'raw_invocation_{planned["invocation"]}.json')
        require(raw["environment"]["source_manifest_sha256"] == source["provenance"]["source_manifest_sha256"], "Every invocation source snapshot")
        records.append(validate_invocation(raw, planned, cases, binaries)); raws.append(raw)
    require(len({r["dispatch_id"] for r in raws}) == 3 and len({r["process_identity"] for r in raws}) == 3,
            "Three separately dispatched processes")
    require(sum(r["samples"] for r in records) == plan["scalar_samples"], "Dimensions mechanically reconcile")
    probes = []
    for name, mutate in (("missing_visit", lambda r: r["visits"].pop()),
                         ("binary", lambda r: r["visits"][0]["sample_launch_sha256"].__setitem__(0, "0"*64)),
                         ("R", lambda r: r["visits"][0].update(R=99)),
                         ("sample", lambda r: r["visits"][0]["samples_us"].__setitem__(0, -1)),
                         ("recompile", lambda r: r.update(recompilation=True)),
                         ("correctness", lambda r: next(w for w in r["warmups"] if condition(w["condition_tag"])["harness"] == "single").update(loaded_ABI_correctness_max_abs_diff=1))):
        bad = copy.deepcopy(raws[0]); mutate(bad)
        try:
            validate_invocation(bad, plan["invocations"][0], cases, binaries)
        except ValueError:
            probes.append(name)
        else:
            raise RuntimeError("Corruption accepted: " + name)
    result = {"status": "PASS", "stage": stage, "records": records, "total_samples": plan["scalar_samples"],
              "corruption_probes": probes, "no_recompilation": True, "prior_files_unchanged": len(c.inventory())}
    c.write(root / "raw_validation.json", result)
    return result


if __name__ == "__main__":
    print(validate_raw(sys.argv[1] if len(sys.argv) > 1 else "stage_c"))
