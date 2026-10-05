"""Re-derive architecture eligibility/ABI/resources from original compiler exports."""
import argparse
import io
from pathlib import Path
import re
import zipfile
from experiments.tma_reduction_layout.phase9 import common as c
from experiments.tma_reduction_layout.gluon.artifact_checks import parse_resource

ABI = ["u64", "u64", "u32", "u32", "u64", "u64"]


def ptx_abi(ptx):
    entry = re.search(r"\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid", ptx, re.S)
    c.require(entry is not None, "PTX entry ABI missing")
    params = [line for line in entry[1].splitlines() if ".param" in line]
    return ["u32" if ".u32" in x else "u64" if ".u64" in x else "UNKNOWN" for x in params]


def derive(target):
    root = c.OUT / "stage_b" / target
    export = root / "export"
    env = c.read(export / "environment.json")
    c.require(env["status"] == "COMPLETE_ARTIFACT_GATE", "Incomplete architecture artifact collection")
    c.require(c.TARGETS[target]["cc"] == env["compute_capability"] and c.TARGETS[target]["name_contains"] in env["gpu_name"], "Actual architecture mismatch")
    expected_native = c.read(c.BASE / "results/phase8/stage_b/environment.json")["toolchain"]["built_triton_native_extension"]["sha256"]
    c.require(env["native_extension"]["SHA256"] == expected_native, "Native extension changed")
    dispatch = c.read(root / "dispatch.json")
    data = (root / "raw_export.zip").read_bytes()
    c.require(c.sha(data) == dispatch["original_export_SHA256"], "Original ZIP SHA changed")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = set(z.namelist())
        actual = {p.relative_to(export).as_posix() for p in export.rglob("*") if p.is_file()}
        c.require(actual == names, "Original export extraction inventory mismatch")
        for name in names: c.require((export / name).read_bytes() == z.read(name), "Original ZIP/extraction byte mismatch: " + name)
    protocol_bytes = (c.OUT / "stage_a/protocol.json").read_bytes()
    c.require((export / "frozen_protocol.json").read_bytes() == protocol_bytes, "Compiler dispatched different preregistration")
    source = c.read(export / "source_bindings.json")
    c.require(source["protocol_SHA256"] == c.sha(protocol_bytes), "Protocol source hash")
    c.require(source["uploaded_source_archive_SHA256"] == c.sha((export / "uploaded_source.zip").read_bytes()), "Uploaded source archive hash")
    provenance = c.read(export / "local_source_provenance.json")
    c.require(env["source_verification"]["remote_source_subset_sha256"] == provenance["source_manifest_sha256"], "Remote source verification")
    with zipfile.ZipFile(export / "uploaded_source.zip") as z:
        for name, expected in provenance["source_manifest"].items():
            c.require(c.sha(z.read(name)) == expected, "Uploaded source file bytes changed")
    attempts = c.read(export / "attempts.json")
    expected_keys = {f'{case["case_id"]}:{h}:{candidate}' for case in c.protocol()["cases"] for h in case["harnesses"] for candidate in c.CANDIDATES}
    keys = [f'{a["case_id"]}:{a["harness"]}:{a["candidate"]}' for a in attempts]
    c.require(len(keys) == len(set(keys)) and set(keys) == expected_keys, "Every frozen compiler attempt retained")
    records, bindings, outcomes = {}, {}, []
    for attempt, key in zip(attempts, keys):
        cfg, harness, candidate = key.split(":")
        directory = export / cfg / harness / candidate
        c.require(c.read(directory / "attempt.json") == attempt, "Attempt ledger mismatch")
        files = {p.name:c.sha(p.read_bytes()) for p in directory.iterdir() if p.is_file() and p.name != "attempt.json"}
        c.require(files == attempt["artifact_SHA256"], "Compiler attempt file hash mismatch")
        records[key] = attempt
        if attempt["status"] != "EXPORTED": continue
        metadata = c.read(directory / "metadata.json")
        occupancy = c.read(directory / "occupancy.json")
        smoke = c.read(directory / "smoke.json")
        ptx = (directory / "kernel.ptx").read_text()
        digest = c.sha((directory / "kernel.cubin").read_bytes())
        c.require((directory / "kernel.cubin").read_bytes().startswith(b"\x7fELF"), "Archived binary not ELF")
        c.require(digest == metadata["cubin_SHA256"] == attempt["cubin_SHA256"] == occupancy["queried_cubin_sha256"] == smoke["launched_cubin_SHA256"] == smoke["after_cubin_SHA256"], "Artifact/smoke/query SHA binding")
        c.require(smoke["passed"] and smoke["max_abs_diff"] == 0 and not smoke["timed"] and smoke["B_RUN"] == 4 and smoke["B_DESC"] == c.B_DESC, "Untimed correctness smoke")
        c.require(metadata["compile_calls"] == smoke["compile_calls"] == attempt["compile_calls"] == 1, "One compilation per physical binary")
        resource = parse_resource((directory / "kernel.resource.txt").read_text())
        c.require(resource == metadata["resources"], "Resource extraction rederivation")
        c.require(metadata["target"] == target and metadata["CC"] == env["compute_capability"], "Compiler metadata target")
        c.require(re.search(r"\.target\s+sm_" + str(env["compute_capability"][0]*10+env["compute_capability"][1]) + r"(?:a|f)?(?:\s|,|$)", ptx), "Actual PTX architecture target")
        c.require(occupancy["gpu_uuid"] == env["gpu_uuid"] and occupancy["blocks_per_sm_actual_dynamic_smem"] > 0, "Checked actual-target occupancy")
        c.require(resource["num_regs"] == occupancy["num_regs"], "CUDA driver and cuobjdump register agreement")
        abi = ptx_abi(ptx)
        # ABI failure is explicit pre-timing unavailability, never a repaired launch guess.
        if abi != ABI:
            records[key] = {**attempt,"status":"ABI_UNAVAILABLE","observed_ABI":abi}
            continue
        bindings[key] = {"archive_path": str((directory / "kernel.cubin").relative_to(c.ROOT)),
            "archive_sha256": digest,"ptx_sha256": c.sha(ptx.encode()),"metadata": metadata,"ABI": abi,
            "occupancy": occupancy,"resources": resource}
    for case in c.protocol()["cases"]:
        for h in case["harnesses"]:
            pair = f'{case["case_id"]}:{h}'
            ks = [pair+":"+candidate for candidate in c.CANDIDATES]
            valid = all(k in bindings for k in ks)
            row = {"pair":pair,"eligible":valid,"target":target,"origin":case["selection_reason"]}
            if valid:
                b0,b1 = (bindings[k] for k in ks)
                zero = all(b["resources"]["local_bytes"] == 0 and b["resources"]["stack_bytes"] == 0 and b["occupancy"]["local_bytes"] == 0 for b in (b0,b1))
                residency = b0["occupancy"]["blocks_per_sm_actual_dynamic_smem"] == b1["occupancy"]["blocks_per_sm_actual_dynamic_smem"]
                row.update({"resource_stratum": "MATCHED_ZERO_LOCAL_RESIDENCY" if zero and residency else "LOCAL_OR_STACK" if not zero else "CHANGED_RESIDENCY",
                    "registers": {k.split(":")[-1]:bindings[k]["resources"]["num_regs"] for k in ks},
                    "resident_blocks": {k.split(":")[-1]:bindings[k]["occupancy"]["blocks_per_sm_actual_dynamic_smem"] for k in ks},
                    "same_CUBIN": b0["archive_sha256"] == b1["archive_sha256"]})
            else:
                row["exclusions"] = {k:records[k] for k in ks if k not in bindings}
            outcomes.append(row)
    eligible = [row["pair"] for row in outcomes if row["eligible"]]
    return {"target":target,"environment":env,"attempt_count":len(attempts),"pairs":outcomes,
            "eligible_pairs":eligible,"binaries":{k:v for k,v in bindings.items() if ":".join(k.split(":")[:2]) in eligible},
            "schedule":c.schedule(eligible),"no_performance_observation":True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target",default="all")
    p.add_argument("--validate",action="store_true")
    args = p.parse_args()
    c.protect()
    targets = c.TARGETS if args.target == "all" else [args.target]
    for target in targets:
        result = derive(target)
        root = c.OUT / "stage_b" / target
        summary = f'# Phase9 StageB — {target} original artifact gate\n\n{result["attempt_count"]} attempts retained; {len(result["eligible_pairs"])}/18 eligible pairs. No performance observations. All valid resource strata proceed to timing.\n\n'
        summary += c.table(["Pair","Eligible","Resource stratum / exclusion","Registers","Residency"],
            [[r["pair"],r["eligible"],r["resource_stratum"] if r["eligible"] else ", ".join(v.get("failure_kind",v["status"]) for v in r["exclusions"].values()),r.get("registers","—"),r.get("resident_blocks","—")] for r in result["pairs"]]) + "\n"
        for name, blob in (("gate.json",c.encode(result)),("summary.md",summary.encode())):
            if args.validate: c.require((root / name).read_bytes() == blob,"Derived artifact gate stale: "+name)
            else: (root / name).write_bytes(blob)
        print(f'{target} artifact gate PASS: {len(result["eligible_pairs"])}/18 pairs')


if __name__ == "__main__":
    main()
