"""Offline closure of the one-worker NCU capability probe; never launches a GPU."""
import csv
import io
import json
from pathlib import Path
import zipfile

from experiments.tma_reduction_layout.phase6 import common as c
from experiments.tma_reduction_layout.phase7 import timing_contract as tc
from experiments.tma_reduction_layout.source_provenance import compute_manifest_digest

OUT = c.BASE / "results/diagnostics/ncu_bank_capability"
BASELINE = "146e307cf34cbb55a48e531b22fa99f7f1456a7f"
KEY = "M128_N32_w16:canonical:4"
METRICS = [
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_shared_op_st.sum",
]


def profile_row(output):
    lines = output.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith('"ID","Process ID"')]
    c.require(len(starts) == 1, "One raw CSV table")
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[starts[0]:]))))
    records = [row for row in rows if row["ID"]]
    c.require(len(records) == 1, "One profiled kernel")
    return records[0]


def validate():
    manifest = c.read(OUT / "raw_manifest.json")
    c.require(manifest["protected_HEAD"] == BASELINE, "Fixed pre-probe baseline")
    export = (OUT / "original_export.zip").read_bytes()
    with zipfile.ZipFile(io.BytesIO(export)) as z:
        c.require(len(z.namelist()) == len(set(z.namelist())), "Unique original ZIP members")
        original = {name: z.read(name) for name in z.namelist()}
    c.require({name: c.sha(data) for name, data in original.items()} ==
              manifest["original_export_members"], "Complete original remote return")
    c.require(set(original) == {"version.json", "query_metrics.json", "collection.json",
                               "result.json", "bank_report.ncu-rep"}, "Exact return inventory")
    for name, digest in manifest["files"].items():
        path = OUT / name
        data = original[name] if name in original else path.read_bytes()
        c.require(c.sha(data) == digest, "Frozen bytes: " + name)
        if path.exists():
            c.require(path.read_bytes() == data, "Extracted bytes equal original: " + name)
    c.require(c.read(OUT / "export_binding.json")["original_export_SHA256"] == c.sha(export),
              "Unmodified original ZIP")
    c.require(original["bank_report.ncu-rep"].startswith(b"NVR"), "Real NCU report")

    request = c.read(OUT / "request.json")
    dispatch = c.read(OUT / "dispatch.json")
    result = json.loads(original["result.json"])
    c.require(request["protected_HEAD"] == BASELINE and request["protected_files"] ==
              len(c.inventory(BASELINE)) == 8158, "Every earlier experiment file unchanged")
    c.require(result["role"] == "TOOL_CAPABILITY_ONLY" and result["formal_timing"] is False
              and result["new_kernel_compilation"] is False, "Capability scope")
    c.require(request["key"] == result["key"] == KEY and request["metrics"] ==
              result["requested_metrics"] == METRICS, "Exact requested binary/metrics")
    c.require(result["dispatch_id"] == dispatch["call_id"] and
              result["remote_evidence"].endswith("/" + dispatch["call_id"]), "Durable dispatch binding")
    c.require(request["payload_SHA256"] == result["payload_SHA256"], "Transferred payload identity")
    c.require(dispatch["profile"] == request["provenance"]["modal_profile"] == "chenmiaoming"
              and dispatch["core_image_id"] == "im-joNh6Ry3lq2oFqOues9VFh", "Reused native image/profile")
    c.require("H100" in result["GPU_name"] and result["compute_capability"] == [9, 0]
              and result["driver_environment"]["gpu_uuid"].startswith("GPU-"), "Actual H100 worker")
    c.require(all(call["return_code"] == 0 for call in
                  result["driver_environment"]["checked_cuda_calls"]), "Checked driver identity queries")

    prov = request["provenance"]
    verification = result["source_verification"]
    c.require(prov == result["source_provenance"] and prov["git_head_sha"] == BASELINE,
              "Original local/remote provenance")
    digest = compute_manifest_digest(prov["source_manifest"])
    c.require(digest == prov["source_manifest_sha256"] == verification["local_manifest_sha256"]
              == verification["remote_source_subset_sha256"] and verification["status"] == "PASS"
              and verification["uploaded_source_fidelity_verified"] is True and
              verification["missing_count"] == verification["mismatched_count"] == 0 and
              verification["files_verified"] == len(prov["source_manifest"]), "Verified uploaded source")
    for name, expected in prov["source_manifest"].items():
        c.require(c.sha((c.ROOT / name).read_bytes()) == expected, "Uploaded source unchanged: " + name)
    source = c.BASE / "diagnostics/probe_ncu_bank.py"
    c.require(source.read_bytes() == (OUT / "probe_source.py").read_bytes(), "Executed probe snapshot")
    _, binaries, _ = tc.inputs("stage_c")
    c.require(request["bindings"] == {KEY: binaries[KEY]}, "Previously frozen launch binding")

    commands = {record["label"]: record for record in result["commands"]}
    c.require(list(commands) == ["version", "query_metrics", "collection"], "One bounded attempt")
    for label, record in commands.items():
        c.require(record == json.loads(original[label + ".json"]) and record["return_code"] == 0,
                  "Original successful command: " + label)
    command = commands["collection"]["command"]
    expected = ["ncu", "--replay-mode", "kernel", "--cache-control", "all", "--clock-control", "none",
                "--profile-from-start", "off", "--metrics", ",".join(METRICS), "--csv", "--page", "raw",
                "--export", result["remote_evidence"] + "/bank_report", "/usr/local/bin/python",
                "-m", "experiments.tma_reduction_layout.phase7.profile_one", "--bundle-root",
                "/tmp/ncu_bank_archives", "--binding-file", "/tmp/ncu_bank_bindings.json", "--key", KEY]
    c.require(command == expected, "Actual collection flags/archived launcher")
    output = commands["collection"]["output"]
    bindings = [json.loads(line.removeprefix("PROFILE_ARCHIVE_BINDING ")) for line in output.splitlines()
                if line.startswith("PROFILE_ARCHIVE_BINDING ")]
    c.require(bindings == [{"key": KEY, "archive_sha256": binaries[KEY]["archive_sha256"],
                           "loaded_cubin_sha256": binaries[KEY]["archive_sha256"], "launch_count": 2,
                           "sha_guard_count": 2, "B_DESC": 65536, "B_RUN": 16384,
                           "correctness_prefix4_max_abs_diff": 0.0, "triton_imported": False,
                           "compile_calls": 0}], "Exact CUBIN, both guards, correctness/no compile")
    row = profile_row(output)
    c.require(row["Kernel Name"] == "canonical_kernel" and row["Grid Size"] == "(16384, 1, 1)"
              and row["Block Size"] == "(512, 1, 1)" and row["CC"] == "9.0"
              and float(row["profiler__replayer_passes"]) == 1, "Actual single kernel replay")
    imported = c.read(OUT / "local_report_import.json")
    c.require(imported["role"] == "OFFLINE_REPORT_READ_ONLY" and imported["GPU_collection"] is False
              and imported["return_code"] == 0 and imported["report_SHA256"] ==
              c.sha(original["bank_report.ncu-rep"]), "Offline original report import")
    reimport = profile_row(imported["output"])
    metric_records = []
    for index, name in enumerate(METRICS):
        queried = name in commands["query_metrics"]["output"]
        c.require(queried == result["queried_metric_name_present"][name], "Metric query presence: " + name)
        if index < 4:
            c.require(name in row and name in reimport and queried, "Returned supported metric: " + name)
            value = int(row[name])
            c.require(value > 0 and float(reimport[name]) == value, "Positive counter/report roundtrip: " + name)
            metric_records.append({"metric": name, "status": "COLLECTED", "count": value})
        else:
            c.require(name not in row and name not in reimport and not queried, "Unavailable request metric")
            metric_records.append({"metric": name, "status": "NOT_RETURNED", "count": None})
    validation = {"status": "PASS", "scope": "ONE_WORKER_CAPABILITY_ONLY", "protected_HEAD": BASELINE,
                  "protected_files_unchanged": 8158, "original_export_SHA256": c.sha(export),
                  "original_report_SHA256": c.sha(original["bank_report.ncu-rep"]),
                  "source_files_verified": len(prov["source_manifest"]), "dispatch": dispatch,
                  "GPU_name": result["GPU_name"], "gpu_uuid": result["driver_environment"]["gpu_uuid"],
                  "remote_ncu_version": commands["version"]["output"], "uid": result["uid"],
                  "process_capabilities": result["process_capabilities"],
                  "driver_counter_policy": result["driver_counter_policy"], "metrics": metric_records,
                  "new_kernel_compilation": False, "formal_timing": False,
                  "offline_report_roundtrip": "PASS", "no_historical_failure_explanation": True,
                  "no_layout_causality_claim": True}
    c.write(OUT / "validation.json", validation)
    return validation


if __name__ == "__main__":
    validated = validate()
    print("PASS: bank-conflict/wavefront counters readable; request counters NOT_RETURNED; "
          f'{validated["protected_files_unchanged"]} prior files unchanged; offline report roundtrip PASS')
