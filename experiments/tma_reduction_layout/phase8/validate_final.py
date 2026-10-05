"""Phase8 final closure, trusted prior replay, and isolated integrity probes."""
import copy
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch
from experiments.tma_reduction_layout.phase8 import common as c, timing_contract as tc

DEST=c.OUT/"final_validation"
COMMANDS=(("prereg","preregister","--validate"),("artifacts","audit_artifacts","--validate"),
          ("stage_c_raw","timing_contract","stage_c"),("stage_c_analysis","analyze","stage_c","--validate"),
          ("stage_c_counters","validate_profile","--validate"),("stage_d_raw","timing_contract","stage_d"),
          ("stage_d_analysis","analyze","stage_d","--validate"))


def first_freeze(stage):
    root=c.OUT/stage;rel=root.relative_to(c.ROOT).as_posix()
    commits=subprocess.check_output(["git","log","--diff-filter=A","--format=%H","--",rel+"/raw_manifest.json"],cwd=c.ROOT,text=True).splitlines()
    c.require(len(commits)==1,"Unique first raw freeze "+stage)
    tree=subprocess.check_output(["git","ls-tree","-rz",commits[0],"--",rel],cwd=c.ROOT)
    blobs={}
    for entry in tree.split(b"\0"):
        if entry:
            meta,name=entry.split(b"\t",1);blobs[name.decode()]=meta.decode().split()[2]
    files=set(c.read(root/"raw_manifest.json")["files"])|{"raw_manifest.json"}
    for name in files:
        data=(root/name).read_bytes();digest=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
        c.require(blobs.get(rel+"/"+name)==digest,"First-commit original bytes "+stage+"/"+name)
    return {"first_commit":commits[0],"protected_files":len(files)}


def probes():
    cases,binaries,plan=tc.inputs("stage_d");raw=c.read(c.OUT/"stage_d/raw_invocation_1.json")
    changed=copy.deepcopy(raw)
    visit=next(v for v in changed["visits"] if v["harness"]=="switch_host")
    visit["runtime_mode_argument"]=1
    try:tc.validate_invocation(changed,plan["invocations"][0],cases,binaries)
    except ValueError as exc:c.require("runtime modes" in str(exc),"Actual mode tamper reason")
    else:raise RuntimeError("Actual runtime mode tamper accepted")
    changed=copy.deepcopy(raw);changed["visits"][0]["sample_launch_sha256"][0]="0"*64
    try:tc.validate_invocation(changed,plan["invocations"][0],cases,binaries)
    except ValueError as exc:c.require("archive/load SHA" in str(exc),"Launch SHA tamper reason")
    else:raise RuntimeError("Launch SHA tamper accepted")
    changed=copy.deepcopy(raw);v=changed["visits"][0]["samples_us"][0]
    changed["visits"][0]["samples_us"][0]=math.nextafter(v,math.inf)
    tc.validate_invocation(changed,plan["invocations"][0],cases,binaries)
    with tempfile.TemporaryDirectory(prefix="tma-phase8-integrity-") as name:
        isolated=Path(name);shutil.copytree(c.OUT/"stage_d",isolated/"stage_d")
        c.write(isolated/"stage_d/raw_invocation_1.json",changed)
        try:c.validate_freeze(isolated/"stage_d")
        except ValueError as exc:c.require("Frozen original bytes" in str(exc),"Positive finite byte tamper reason")
        else:raise RuntimeError("Positive finite sample tamper accepted")
        (isolated/"stage_b").mkdir();(isolated/"stage_a").mkdir()
        shutil.copyfile(c.OUT/"stage_a/pool.json",isolated/"stage_a/pool.json")
        frozen=c.read(c.OUT/"stage_b/launch_stage_d.json")
        order=frozen["schedule"]["invocations"][0]["rounds"][0]["order"]
        order[0],order[1]=order[1],order[0]
        c.write(isolated/"stage_b/launch_stage_d.json",frozen)
        with patch.object(c,"OUT",isolated):
            try:tc.inputs("stage_d")
            except ValueError as exc:c.require("Complete frozen schedule" in str(exc),"Frozen order tamper reason")
            else:raise RuntimeError("Frozen order tamper accepted")
    return ["ACTUAL_MODE_ARGUMENT","PER_LAUNCH_SHA","POSITIVE_FINITE_SAMPLE_BYTES","FROZEN_MASTER_ORDER"]


def report(result):
    d=c.read(c.OUT/"stage_d/results.json");p=c.read(c.OUT/"stage_c/profiling/results.json")
    lines=["# Phase8 completion and final evidence closure","",
        "All four stages completed with separate stage commits. No compiler/production changes or later phase. Every one of the8,174 pre-Phase8 experiment files remains byte-identical.","",
        "StageA retained the Phase7 influence and leave-one-out audit, and froze four old diagnostic cases plus16 unseen identities. Seven prospective identities failed the unchanged structural rules before compilation; none were replaced. StageB retained all104 actual compiler attempts. Four diagnostic andnine unseen cases passed PRIMARY gates, zero spills and candidate residency matching.","",
        "The Gluon switch uses a block-uniform, unspecialized mode scalar. Both modes share one archived CUBIN/module/function and resource allocation, with two TMA arms and one common LocalLoad/reduction/store. Both-mode smoke and all launch SHA guards passed. Full operand/PTX/SASS inventories are retained. This establishes a valid same-binary runtime descriptor-path intervention, not isolated intrinsic tensormap creation latency.","",
        c.table(["Stage","Eligible","PRIMARY","Samples","Independent processes","GPU UUIDs"],
                [[s,len(c.read(c.OUT/s/"results.json")["cases"]),sum(r["final_class"]=="PRIMARY" for r in c.read(c.OUT/s/"results.json")["cases"].values()),
                  result["timing"][s]["total_samples"],3,result["timing"][s]["physical_UUIDs"]] for s in ("stage_c","stage_d")]),"",
        c.table(["Scope","Hypothesis","n","Reference MAE / RMSE","Alternative MAE / RMSE","Decision","Practical threshold"],
                [[scope,name,h["n"],h["reference_errors"],h["alternative_errors"],h["decision"],h["practical_threshold_met"]]
                 for scope,hs in d["comparisons"].items() for name,h in hs.items()]),"",
        "Fresh decisions use the fixed coefficient1/intercept0 predictors, strict MAE/RMSE rule, and separate preregistered practical threshold (MAE decrease≥0.01 ns/CTA and≥10%). No model fit, case removal, threshold tuning or adverse-result rerun. All individual errors, counterexamples, sign bands, leave-one-out comparisons and fit residuals remain in the detailed results.","",
        f'NCU: {p["completed_reports"]}/8 original NVR reports and {p["all_requested_fields_present_reports"]}/8 complete requested metric sets. Original and explicit-base-unit offline views agree. Initial NCU2026 import auto-scaled DRAM values; precise base-unit derived views are retained. Counter aggregates are descriptive and profiler duration is excluded from formal timing.',"",
        "Metadata deviation: the original profiling worker export omitted its UUID (recorded as unavailable, with H100 name/CC and dispatch identity retained). Formal timing worker UUIDs are complete. No GPU profile was repeated to repair that metadata omission.","",
        "The exact Phase6 native image and persistent ccache/triton-home were reused with unchanged native/compiler SHA identities. Formal timing workers only load archived ELF bytes and never import Triton/JIT/compiler.","",
        f'Final validator PASS: seven Phase8 checks, trusted Phase7 closure including its Phase6/historical replay, prior bank-probe replay, first-commit B/C original-byte checks, frozen D original-byte closure and four isolated corruption probes. Prior replays run in a separate checkout at {c.BASELINE} without altering current prior results.',"",
        "The prospective scope has nine PRIMARY cases extending warp count to32; it does not establish arbitrary shape/warp generalization. Runtime mode includes its branch and descriptor issue path. H2b/H2c remain UNVERIFIED. No pure component-cost, causal-share, or compiler heuristic claim is made.","",
        "[Preregistration](../stage_a/summary.md) · [Artifacts](../stage_b/summary.md) · [Diagnostic timing](../stage_c/summary.md) · [NCU](../stage_c/profiling/summary.md) · [Held-out timing](../stage_d/summary.md) · [Validator](suite.json)","",
        "Work stops after Phase8 StageD. No Phase9 is started."]
    robustness=[]
    for name,h in d["comparisons"]["PRIMARY"].items():
        improved=sum(abs(v["alternative"])<abs(v["reference"]) for v in h["case_errors"].values())
        stable=all(v["alternative"]["MAE"]<v["reference"]["MAE"] and v["alternative"]["RMSE"]<v["reference"]["RMSE"]
                   for v in h["leave_one_out"].values())
        robustness.append(f'{name}: {improved}/{h["n"]} individual absolute errors improve; every leave-one-out improves both MAE/RMSE={stable}; relative MAE decrease={h["relative_MAE_decrease"]:.6%}.')
    insertion=next(i for i,line in enumerate(lines) if line.startswith("NCU:"))
    lines[insertion:insertion]=robustness+[""]
    return "\n".join(lines)+"\n"


def main():
    if "--committed" in sys.argv:
        result=c.read(DEST/"suite.json");c.require(result["status"]=="PASS","Stored final suite")
        for stage in ("stage_b","stage_c","stage_d"):print(stage,first_freeze(stage))
        c.validate_freeze(c.OUT/"stage_c");c.validate_freeze(c.OUT/"stage_d")
        c.require((DEST/"summary.md").read_text()==report(result),"Final report closure")
        print("PASS committed Phase8 final closure;",len(c.protect()),"prior files unchanged; no writes")
        return
    DEST.mkdir(parents=True,exist_ok=True)
    prior=c.read(DEST/"prior_replay/suite.json")
    c.require(prior["status"]=="PASS" and prior["trusted_commit"]==c.BASELINE and prior["GPU_work"] is False,"Trusted offline prior replay")
    for r in prior["records"]:
        c.require(r["return_code"]==0 and c.sha((DEST/"prior_replay"/(r["label"]+".log")).read_bytes())==r["log_SHA256"],"Original prior replay logs")
    validators=[]
    for label,module,*args in COMMANDS:
        cmd=[sys.executable,"-m","experiments.tma_reduction_layout.phase8."+module,*args]
        run=subprocess.run(cmd,cwd=c.ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (DEST/(label+".log")).write_text(run.stdout)
        validators.append({"label":label,"command":cmd,"return_code":run.returncode,"log_SHA256":c.sha(run.stdout.encode())})
        c.require(run.returncode==0,"Final validator "+label)
    result={"status":"PASS","prior_files_unchanged":len(c.protect()),"prior_replay":prior,"validators":validators,
            "first_commit_freezes":{s:first_freeze(s) for s in ("stage_b","stage_c")},
            "stage_d_raw_manifest_SHA256":c.sha((c.OUT/"stage_d/raw_manifest.json").read_bytes()),
            "timing":{s:c.read(c.OUT/s/"validation.json") for s in ("stage_c","stage_d")},"isolated_corruption_probes":probes(),
            "profiler_metadata_deviations":c.read(c.OUT/"stage_c/profiling/results.json")["metadata_deviations"],"no_next_phase":True}
    c.write(DEST/"suite.json",result);(DEST/"summary.md").write_text(report(result))
    print("PASS Phase8 final suite;",len(validators),"checks; nine fresh PRIMARY cases;",result["isolated_corruption_probes"])


if __name__=="__main__":main()
