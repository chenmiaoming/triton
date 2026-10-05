"""Offline closure, independent statistical checks and isolated tamper probes."""
import argparse
import copy
from fractions import Fraction
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from experiments.tma_reduction_layout.phase9 import common as c, timing_contract as tc


def committed_bytes(root):
    manifest_path = (root / "raw_manifest.json").relative_to(c.ROOT).as_posix()
    commits = subprocess.check_output(["git","log","--reverse","--format=%H","--",manifest_path],cwd=c.ROOT,text=True).splitlines()
    c.require(bool(commits),"Raw freeze not committed: "+str(root))
    first = commits[0]
    record = c.read(root / "raw_manifest.json")
    files = list(record["files"])+["raw_manifest.json"]
    for relative in files:
        name = (root / relative).relative_to(c.ROOT).as_posix()
        expected = subprocess.check_output(["git","show",first+":"+name],cwd=c.ROOT)
        c.require((root / relative).read_bytes() == expected,"First-commit original bytes changed: "+name)
    return {"stage":root.name,"first_commit":first,"original_files":len(files),"status":"PASS"}


def independent_statistics():
    count = 0
    for target in c.TARGETS:
        result = c.read(c.OUT / "stage_c" / target / "results.json")
        for invocation in (1,2,3):
            raw = c.read(c.OUT / "stage_c" / target / f"raw_invocation_{invocation}.json")
            groups = {}
            for visit in raw["visits"]: groups.setdefault(visit["condition_tag"],[]).extend(visit["samples_us"])
            for tag,values in groups.items():
                values = sorted(values)
                c.require(len(values) == 100,"Independent100-sample count")
                median = (values[49]+values[50])/2
                c.require(result["statistics"][str(invocation)][tag]["median_us"] == median,"Independent order-statistic median")
                count += 1
            for fit in result["grid_time_fits"][str(invocation)].values():
                points = fit["points"]
                xs,ys = [Fraction(p[0]) for p in points],[Fraction(p[1]) for p in points]
                n = len(xs)
                sx,sy = sum(xs),sum(ys)
                b = (n*sum(x*y for x,y in zip(xs,ys))-sx*sy)/(n*sum(x*x for x in xs)-sx*sx)
                a = (sy-b*sx)/n
                residual = [y-a-b*x for x,y in zip(xs,ys)]
                total = sum(y*y for y in ys)-sy*sy/n
                rsquared = 1-sum(r*r for r in residual)/total if total else None
                c.require(fit["intercept_us"] == float(a) and fit["slope_ns_per_additional_CTA"] == float(1000*b),"Independent rational OLS slope/intercept")
                c.require(fit["residuals_us"] == list(map(float,residual)) and fit["R_squared"] == (float(rsquared) if rsquared is not None else None),"Independent rational OLS residual/R²")
    return {"status":"PASS","independently_rederived_condition_medians":count}


def probes():
    target = "sm90"
    cases,binaries,plan = tc.inputs(target)
    raw = c.read(c.OUT / "stage_c" / target / "raw_invocation_1.json")
    results = []
    for name in ("actual_argument_flip","launch_SHA_flip","frozen_order_swap"):
        tampered = copy.deepcopy(raw)
        if name == "actual_argument_flip": tampered["visits"][0]["actual_parameter_values"][3] += 1
        elif name == "launch_SHA_flip": tampered["visits"][0]["sample_launch_sha256"][0] = "0"*64
        else: tampered["visits"][0],tampered["visits"][1] = tampered["visits"][1],tampered["visits"][0]
        try: tc.validate_invocation(tampered,plan["invocations"][0],cases,binaries,target)
        except ValueError as exc: results.append({"probe":name,"status":"REJECTED_AS_REQUIRED","reason":str(exc)})
        else: raise ValueError("Tamper accepted: "+name)
    with tempfile.TemporaryDirectory(prefix="tma-phase9-tamper-") as directory:
        root = Path(directory)
        c.write(root / "sample.json",raw)
        c.freeze(root)
        tampered = copy.deepcopy(raw)
        value = tampered["visits"][0]["samples_us"][0]
        tampered["visits"][0]["samples_us"][0] = math.nextafter(value,math.inf)
        tc.validate_invocation(tampered,plan["invocations"][0],cases,binaries,target)
        c.write(root / "sample.json",tampered)
        try: c.validate_freeze(root)
        except ValueError as exc: results.append({"probe":"positive_finite_sample_byte_mutation","status":"REJECTED_AS_REQUIRED","reason":str(exc),"semantic_protocol_remains_valid":True})
        else: raise ValueError("Positive sample byte mutation accepted")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--committed",action="store_true")
    args = p.parse_args()
    protected = c.protect()
    c.validate_freeze(c.OUT / "stage_b")
    c.validate_freeze(c.OUT / "stage_c")
    closure = [committed_bytes(c.OUT / "stage_b"),committed_bytes(c.OUT / "stage_c")]
    root = c.OUT / "final_validation"
    if args.committed:
        c.require(c.read(root / "suite.json")["status"] == "PASS","Final suite not PASS")
        from experiments.tma_reduction_layout.phase9.scope import derive,summary
        scoped = derive()
        c.require(c.read(c.OUT / "stage_d/results.json") == scoped,"Committed scope report stale")
        c.require((c.OUT / "stage_d/summary.md").read_text() == summary(scoped),"Committed scope summary stale")
        print(c.encode({"status":"PASS","read_only":True,"protected_prior_files":len(protected),"committed_originals":closure}).decode(),end="")
        return
    root.mkdir(parents=True,exist_ok=True)
    commands = [
        ["-m","experiments.tma_reduction_layout.phase9.preregister","--validate"],
        ["-m","experiments.tma_reduction_layout.phase9.audit_artifacts","--validate"],
        ["-m","experiments.tma_reduction_layout.phase9.timing_contract"],
        ["-m","experiments.tma_reduction_layout.phase9.analyze","--validate"],
        ["-m","experiments.tma_reduction_layout.phase9.scope","--validate"],
        ["-m","experiments.tma_reduction_layout.phase8.validate_final","--committed"],
    ]
    checks = []
    for command in commands:
        run = subprocess.run([sys.executable,*command],cwd=c.ROOT,capture_output=True,text=True)
        checks.append({"command":[sys.executable,*command],"return_code":run.returncode,"stdout":run.stdout,"stderr":run.stderr})
        print(command[1],"PASS" if run.returncode == 0 else "FAIL",flush=True)
    stats = independent_statistics()
    integrity = probes()
    manifests = [c.read(c.OUT / "stage_b" / t / "export/source_bindings.json")["source_manifest_SHA256"] for t in c.TARGETS]
    c.require(len(set(manifests)) == 1,"Compiler source differed across targets")
    timing_manifests = [c.read(c.OUT / "stage_c" / t / "source_bindings.json")["provenance"]["source_manifest_sha256"] for t in c.TARGETS]
    c.require(len(set(timing_manifests)) == 1,"Timing source differed across targets")
    c.protect()
    suite = {"status":"PASS" if all(x["return_code"] == 0 for x in checks) else "FAIL",
        "checks":checks,"independent_statistics":stats,"integrity_probes":integrity,
        "protected_prior_files":len(protected),"committed_originals":closure,
        "same_compiler_source_across_targets":True,"same_timing_source_across_targets":True,
        "no_compiler_edit":True,"no_PR_created":True,"stop_after_phase9":True}
    c.write(root / "suite.json",suite)
    c.require(suite["status"] == "PASS","Final validator failure; original outputs preserved")
    from experiments.tma_reduction_layout.phase9.scope import derive
    scope = derive()
    report = "# Phase9 completion and final evidence closure\n\nStages A-D completed with separate stage commits. No compiler/production changes, PR creation or later phase. All9,489 pre-Phase9 experiment files remain byte-identical.\n\n"
    report += "The strict targets H100/SM90, B200/SM100 and RTX PRO6000 Blackwell Server Edition/SM120 were all allocated. Twelve retrospective reduction identities and six non-reduction TMA load/store controls were frozen before dispatch. All108 compiler attempts and original source/IR/PTX/CUBIN/SASS/resource/occupancy/smoke/failure artifacts remain retained. The exact Phase6 native core and unchanged native extension SHA were reused; ccache counters did not advance. Read-only CUDA13 disassembly tools were layered without a native rebuild. Actual architecture-selected compiler ptxas identities are recorded separately.\n\n"
    rows = []
    for target,r in scope["targets"].items():
        v = r["validation"]
        original = r["original_case"]["primary"]
        rows.append([target,v["eligible_pairs"],v["samples"],v["processes"],len(v["physical_GPU_UUIDs"]),original["status"],f'{original["mean_percent"]:.6f}'])
    report += c.table(["Target","Eligible pairs","Samples","Processes","Physical UUIDs","Original case decision","Mean improvement %"],rows)+"\n\n"
    report += f'Total{sum(r["validation"]["samples"] for r in scope["targets"].values())} exact-binary event samples. Actual ABI parameters, module/function identities, per-launch SHA guards, checked CUDA call totals and worker UUIDs pass. Two SM120 large tiles exceed shared-memory limits and remain unavailable, with no replacement. All valid pairs and resource strata are retained.\n\n'
    report += f'The fixed3% practical bands classify{len(scope["regressions"])} regression pairs and{len(scope["unresolved"])} unresolved pairs. A global vector4 policy is not justified. Full timings at all three grids,100-sample medians/statistics, OLS intercepts/slopes/R²/residuals and individual invocation effects remain in results.json. Bands are descriptive three-process summaries, not familywise significance or proof of universal safety.\n\n'
    report += f'Secondary grids retain{len(scope["secondary_unresolved"])} unresolved practical effects and{len(scope["secondary_regressions"])} classified regressions. Some small-grid bands include decreases beyond3%; primary-grid non-regression cannot be transferred to those conditions. No GPU timing was repeated after these outcomes.\n\n'
    report += "Final validator PASS: six checks including read-only Phase8 committed-byte closure, independent order-statistic medians and rational OLS, four isolated tamper probes, same-source checks across targets and first-commit B/C original-byte verification. No raw samples or original compiler artifacts were modified.\n\n"
    report += "This phase establishes cross-architecture candidate contrasts for historically selected BF16/max shapes. It does not identify a historical good/bad compiler commit, isolate intrinsic descriptor/bank/lane cost, or validate a production patch on held-out cases. The next fix requires explicit target/IR applicability and a separate frozen-patch validation. User approval is mandatory before creating any PR, including draft. Work stops after Phase9 StageD.\n\n"
    report += "[Scope decision](../stage_d/summary.md) · [Full suite](suite.json) · [Preregistration](../stage_a/summary.md) · [Artifact gate](../stage_b/summary.md)\n"
    (root / "summary.md").write_text(report)
    print("Final Phase9 validator PASS",flush=True)


if __name__ == "__main__":
    main()
