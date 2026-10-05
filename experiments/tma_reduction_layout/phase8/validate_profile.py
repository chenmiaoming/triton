"""Original report/command closure and per-field counter availability."""
import ast
import csv
import io
import json
import math
import sys
import zipfile
from pathlib import Path
from experiments.tma_reduction_layout.phase8 import common as c, timing_contract as tc
from experiments.tma_reduction_layout.source_provenance import compute_manifest_digest


def csv_row(output):
    lines=output.splitlines();starts=[i for i,line in enumerate(lines) if line.startswith('"ID","Process ID"')]
    if len(starts)!=1:return None
    rows=[row for row in csv.DictReader(io.StringIO("\n".join(lines[starts[0]:]))) if row["ID"]]
    c.require(len(rows)==1,"One profiled kernel/table")
    return rows[0]


def derive():
    root=c.OUT/"stage_c/profiling";request=c.read(root/"request.json");status=c.read(root/"status.json")
    tree=ast.parse((c.BASE/"phase8/run_profile.py").read_text())
    settings=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
                  and any(isinstance(target,ast.Name) and target.id=="SETTINGS" for target in node.targets))
    protocol=c.read(c.OUT/"stage_a/protocol.json")["profiling"]
    c.require(settings==request["settings"]==status["settings"] and settings["cases"]==protocol["cases"]
              and settings["metrics"]==protocol["metrics"],"Preregistered actual settings/metrics")
    export=(root/"original_export.zip").read_bytes()
    c.require(c.sha(export)==c.read(root/"export_binding.json")["original_export_SHA256"],"Original export SHA")
    with zipfile.ZipFile(io.BytesIO(export)) as z:
        c.require(len(z.namelist())==len(set(z.namelist())),"Unique original members")
        for name in z.namelist():c.require(z.read(name)==(root/name).read_bytes(),"Unmodified original profiler member")
    cases,binaries,_=tc.inputs("stage_c")
    expected={key:b for key,b in binaries.items() if key.split(":")[0] in settings["cases"] and key.split(":")[1]=="switch"}
    c.require(request["bindings"]==expected and len(expected)==4,"Fixed four physical CUBINs")
    payload=(root/"archived_profile_payload.zip").read_bytes()
    c.require(c.sha(payload)==request["payload_SHA256"]==status["payload_SHA256"] and
              status["counters_enter_formal_timing"] is False,"Separate actual payload")
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        names={name for b in expected.values() for name in (b["archive_path"],str(Path(b["archive_path"]).with_suffix(".ptx")))}
        c.require(set(z.namelist())==names and len(z.namelist())==len(names),"Unique complete profile payload")
        for name in names:c.require(z.read(name)==(c.ROOT/name).read_bytes(),"Actual profile CUBIN/PTX archive")
    prov=request["provenance"]
    c.require(prov==status["source_provenance"] and compute_manifest_digest(prov["source_manifest"])==prov["source_manifest_sha256"]
              ==status["source_verification"]["remote_source_subset_sha256"] and status["source_verification"]["status"]=="PASS","Verified profile source")
    with zipfile.ZipFile(root/"uploaded_source.zip") as z:
        c.require(set(z.namelist())==set(prov["source_manifest"]),"Full original profile source snapshot")
        for name,digest in prov["source_manifest"].items():c.require(c.sha(z.read(name))==digest,"Snapshot source bytes")
    c.require("H100" in status["GPU_name"] and status["compute_capability"]==[9,0],"Actual H100 profiler")
    wanted={(key,mode) for key in expected for mode in (0,1)}
    c.require(len(status["commands"])==8 and {(r["key"],r["mode"]) for r in status["commands"]}==wanted,"All eight fixed attempts")
    rows={};completed=0;complete_fields=0
    for r in status["commands"]:
        key,mode=r["key"],r["mode"];tag=key+":mode"+str(mode);directory=root/tag.replace(":","_")
        c.require(r["tag"]==tag and c.read(directory/"command.json")==r and r["archive_sha256"]==expected[key]["archive_sha256"],"Original command/CUBIN/mode")
        cmd=r["command"]
        for flag,value in (("--replay-mode","kernel"),("--cache-control","all"),("--clock-control","none"),
                           ("--profile-from-start","off"),("--metrics",",".join(settings["metrics"])),("--mode",str(mode)),("--key",key)):
            c.require(cmd[cmd.index(flag)+1]==value,"Actual profile setting "+flag)
        row=csv_row(r["output"]) if r["return_code"]==0 else None
        if row is not None:
            printed=[json.loads(line.removeprefix("PROFILE_ARCHIVE_BINDING ")) for line in r["output"].splitlines()
                     if line.startswith("PROFILE_ARCHIVE_BINDING ")]
            c.require(len(printed)==1,"Actual profile launch binding")
            p=printed[0];digest=expected[key]["archive_sha256"]
            c.require(p["key"]==key and p["mode"]==mode and p["archive_sha256"]==p["loaded_cubin_sha256"]==digest
                      and p["compile_calls"]==0 and p["triton_imported"] is False and p["correctness_prefix4_max_abs_diff"]==0
                      and p["sha_guard_count"]==p["launch_count"]==2 and p["B_DESC"]==65536 and p["B_RUN"]==16384,
                      "Exact archived actual-mode launch and both guards")
            report=(directory/"report.ncu-rep").read_bytes();c.require(report.startswith(b"NVR"),"Real binary NCU report")
            c.require(row["Kernel Name"]==expected[key]["metadata"]["function_name"] and row["Grid Size"]=="(16384, 1, 1)","Actual one profiled launch")
            completed+=1
        values={}
        for metric in settings["metrics"]:
            text=row.get(metric) if row is not None else None
            try:value=float(text)
            except (TypeError,ValueError):value=None
            available=value is not None and math.isfinite(value) and value>=0
            values[metric]={"status":"COLLECTED" if available else "UNAVAILABLE","value":value if available else None}
        complete_fields+=int(all(v["status"]=="COLLECTED" for v in values.values()))
        rows[tag]={"key":key,"mode":mode,"CUBIN_SHA256":expected[key]["archive_sha256"],"return_code":r["return_code"],
                   "metrics":values,"report_SHA256":c.sha((directory/"report.ncu-rep").read_bytes()) if (directory/"report.ncu-rep").exists() else None,
                   "profiler_passes":row.get("profiler__replayer_passes") if row else None}
    expected_status="COUNTERS_COLLECTED" if completed==8 else "COUNTERS_UNAVAILABLE_OR_PARTIAL"
    c.require(status["status"]==expected_status,"Return-code status; individual fields classified independently")
    offline=c.read(root/"offline_imports.json")
    c.require(offline["role"]=="OFFLINE_REPORT_READ_ONLY" and offline["GPU_collection"] is False
              and len(offline["imports"])==completed,"Separate saved-report imports")
    imports={record["tag"]:record for record in offline["imports"]}
    base=c.read(root/"offline_base/validation.json")
    c.require(base["role"]=="DERIVED_OFFLINE_REPORT_VIEW" and base["GPU_collection"] is False
              and len(base["imports"])==completed,"Explicit base-unit derived views")
    base_imports={record["tag"]:record for record in base["imports"]}
    for tag,row in rows.items():
        if row["report_SHA256"] is None:continue
        record=imports[tag.replace(":","_")]
        c.require(record["return_code"]==0 and record["report_SHA256"]==row["report_SHA256"]
                  and "--import" in record["command"],"Offline import of exact original report")
        original_imported=csv_row(record["output"])
        view=base_imports[tag.replace(":","_")]
        c.require(view["return_code"]==0 and view["report_SHA256"]==row["report_SHA256"]
                  and view["command"][view["command"].index("--print-units")+1]=="base", "Explicit base-unit original report read")
        imported=csv_row(view["output"])
        c.require(imported is not None,"Offline report contains one kernel")
        for metric,value in row["metrics"].items():
            if value["status"]=="COLLECTED":
                c.require(metric in imported and float(imported[metric])==value["value"],"Independent saved-report counter roundtrip")
            else:
                c.require(metric not in imported or imported[metric] in ("","N/A"),"Unavailable field remains unavailable")
        for metric in settings["metrics"][:4]:
            value=row["metrics"][metric]
            if value["status"]=="COLLECTED":
                c.require(float(original_imported[metric])==value["value"],"Bank/wavefront counts also exact in initial import")
    contrasts={}
    for key in expected:
        pair=[rows[key+":mode"+str(mode)] for mode in (0,1)]
        contrasts[key]={metric:pair[1]["metrics"][metric]["value"]-pair[0]["metrics"][metric]["value"]
                        if all(row["metrics"][metric]["status"]=="COLLECTED" for row in pair) else None for metric in settings["metrics"]}
    return {"status":"PASS","attempts":8,"completed_reports":completed,"all_requested_fields_present_reports":complete_fields,
            "availability":expected_status,"rows":rows,"mode1_minus_mode0_counts":contrasts,
            "formal_timing_use":False,"offline_report_roundtrip":"PASS",
            "profiler_worker_UUID":None,"metadata_deviations":["PROFILER_WORKER_UUID_NOT_CAPTURED_IN_ORIGINAL_EXPORT"],
            "interpretation":"Single report per mode/binary; aggregate diagnostics, no variance or pure bank-conflict/descriptor latency claim."}


def report(result):
    metrics=c.read(c.OUT/"stage_a/protocol.json")["profiling"]["metrics"]
    return "\n".join(["# Phase8 StageC — Separate same-CUBIN NCU diagnostics","",result["interpretation"],"",
        c.table(["Physical binary/mode",*[m for m in metrics]],[[tag,*[row["metrics"][m]["value"] if row["metrics"][m]["status"]=="COLLECTED" else "UNAVAILABLE" for m in metrics]]
                 for tag,row in sorted(result["rows"].items())]),"",
        "All original stdout/return codes/commands/NVR reports/source and payload snapshots are retained. A missing field is unavailable evidence, never a zero. Profiler duration is excluded from formal timing.","",
        "Metadata deviation: the original profiler export recorded H100 name/CC and dispatch identity but did not capture its worker UUID. UUID is unavailable and is not substituted from another run. All three formal timing worker UUIDs are recorded separately. No GPU recollection was performed to repair this metadata omission.",""]) + "\n"


if __name__=="__main__":
    root=c.OUT/"stage_c/profiling";value=derive()
    if "--validate" in sys.argv:
        c.require(c.read(root/"results.json")==value and (root/"summary.md").read_text()==report(value),"Counter result/report closure")
    else:
        c.write(root/"results.json",value);(root/"summary.md").write_text(report(value))
    print("PASS",value["completed_reports"],"reports;",value["all_requested_fields_present_reports"],"with all requested fields")
