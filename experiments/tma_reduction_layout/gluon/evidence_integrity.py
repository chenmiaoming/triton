"""Outcome-neutral offline fidelity checks, independently computed with statistics.

The hypothesis decision is retrospective and reported separately from integrity.
No compiler, GPU, Modal, or timing entry points are imported.
"""
import copy
import hashlib
import json
import math
import statistics as st
import subprocess
from pathlib import Path
try:
    from . import artifact_checks as ac
except ImportError:
    import artifact_checks as ac

RS = [0, 1, 2, 4, 8]
BS = [16384, 32768, 65536]
DECISION_RULE = {
    "kind": "retrospective_descriptive_rule_not_integrity_invariant",
    "all_run_beta_strictly_positive": True,
    "delta_g_1_mean_threshold_ns_per_cta": 0.1,
    "mean_run_fit_r2_minimum": 0.90,
    "mean_and_each_run_g_nondecreasing": True,
    "scope": "primary matched-residency composite-body harness; no component dominance claim",
}


def finite_tree(obj, path="root"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            finite_tree(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            finite_tree(v, f"{path}[{i}]")
    elif type(obj) in (int, float) and not math.isfinite(obj):
        raise ValueError(f"Nonfinite number at {path}")


def read_json(path):
    obj = json.loads(ac.nonempty_text(path))
    finite_tree(obj, str(path))
    return obj


def compare(actual, expected, path="root"):
    """Strict recursive keys/types/lengths and finite numeric comparison."""
    finite_tree(actual, path)
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Schema keys mismatch at {path}")
        for k in expected:
            compare(actual[k], expected[k], f"{path}.{k}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"Array length/type mismatch at {path}")
        for i, v in enumerate(expected):
            compare(actual[i], v, f"{path}[{i}]")
    elif type(expected) in (int, float):
        if type(actual) not in (int, float) or (type(expected) is int and type(actual) is not int):
            raise ValueError(f"Numeric type mismatch at {path}")
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-11):
            raise ValueError(f"Numeric mismatch at {path}: {actual} != {expected}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Value/type mismatch at {path}: {actual!r} != {expected!r}")


def samples_stats(samples, count=100):
    if not isinstance(samples, list) or len(samples) != count or any(
            type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError(f"Expected {count} positive finite numeric samples")
    q = st.quantiles(samples, n=100, method="inclusive")
    mean, sd = st.mean(samples), st.stdev(samples)
    return dict(count=len(samples), median=st.median(samples), mean=mean, std=sd,
                iqr=q[74]-q[24], cv_percent=100*sd/abs(mean), min=min(samples),
                max=max(samples), p10=q[9], p90=q[89])


def fit(xs, ys):
    xm, ym = st.mean(xs), st.mean(ys)
    b = sum((x-xm)*(y-ym) for x, y in zip(xs, ys))/sum((x-xm)**2 for x in xs)
    a = ym-b*xm
    residuals = [y-(a+b*x) for x, y in zip(xs, ys)]
    total = sum((y-ym)**2 for y in ys)
    return b, a, 1-sum(e*e for e in residuals)/total if total else 1.0, residuals


def canonical_baseline(base):
    path = base / "results/phase2/saturation/corrected_pilot_runs.json"
    raw = read_json(path)
    cfg = "M32_N64_w8"
    per_run = {}
    for run in ("run_1", "run_2", "run_3"):
        grid = raw[run]["configs"][cfg]["grid_data"]
        per_run[run] = {}
        for cand in ac.CANDIDATES:
            medians = [samples_stats(grid[str(b)][cand]["raw_samples_us"])["median"] for b in BS]
            per_run[run][cand] = fit(BS, medians)[0]*1000
    means = {c: st.mean(row[c] for row in per_run.values()) for c in ac.CANDIDATES}
    gap = means["default"]-means["4"]
    if gap == 0:
        raise ValueError("Undefined descriptive ratio: canonical gap is zero")
    root = base.parents[1]
    rel = str(path.relative_to(root))
    commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", rel], cwd=root, text=True).strip()
    return {"raw_path": rel, "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "raw_last_change_commit": commit, "config": cfg, "candidates": list(ac.CANDIDATES),
            "B": BS, "estimator": "per-condition sample median -> OLS T(B) on B -> arithmetic mean of three run slopes; ns/additional CTA",
            "per_run_slopes": per_run, "mean_slopes": means, "gap_ns_per_cta": gap}


def aggregate(values):
    mean, sd = st.mean(values), st.stdev(values) if len(values)>1 else 0.0
    unstable = abs(mean) <= 3*sd
    return dict(mean=mean, std=sd, cv="N/A" if unstable else 100*sd/abs(mean),
                stability_class="near_zero_or_sign_unstable" if unstable else "stable", values=values)


def residual_table(xs, ys, b, a):
    return [dict(R=x, observed=y, fitted=a+b*x, residual=y-(a+b*x)) for x, y in zip(xs, ys)]


def recompute_runs(raws, baseline):
    runs = []
    for raw in raws:
        if set(raw["configurations"]) != set(ac.CONFIGS):
            raise ValueError("Timing configuration domain mismatch")
        run = {"run_id": raw["run_id"], "device_info": raw["env_info"],
               "pre_run_telemetry": raw["pre_run_telemetry"], "post_run_telemetry": raw["post_run_telemetry"], "configurations": {}}
        for cfg in ac.CONFIGS:
            candidates = raw["configurations"][cfg]["candidates"]
            if set(candidates) != set(ac.CANDIDATES):
                raise ValueError("Candidate domain mismatch")
            out = dict(role="PRIMARY" if cfg==ac.CONFIGS[0] else "SECONDARY_CONTROL", slopes={}, intercepts={}, grid_fit_r2={}, grid_fit_residuals_us={}, per_b_stats={})
            for c in ac.CANDIDATES:
                rt = candidates[c]["r_timing"]
                if set(rt) != set(map(str,RS)):
                    raise ValueError("R domain mismatch")
                for key in ("slopes", "intercepts", "grid_fit_r2", "grid_fit_residuals_us", "per_b_stats"):
                    out[key][c] = {}
                for r in map(str,RS):
                    if set(rt[r]) != set(map(str,BS)):
                        raise ValueError("B domain mismatch")
                    stats = {str(b): samples_stats(rt[r][str(b)]["samples_us"]) for b in BS}
                    out["per_b_stats"][c][r] = stats
                    b,a,r2,residuals = fit(BS,[stats[str(x)]["median"] for x in BS])
                    out["grid_fit_residuals_us"][c][r] = residuals
                    out["slopes"][c][r],out["intercepts"][c][r],out["grid_fit_r2"][c][r] = b*1000,a,r2
            g = {r:out["slopes"]["default"][r]-out["slopes"]["4"][r] for r in map(str,RS)}
            dg = {r:v-g["0"] for r,v in g.items()}
            out["differentials"] = dict(g_0=g["0"], delta_g_1=dg["1"], g_r=g, delta_g_r=dg)
            out["incremental_deltas"] = dict(d_01=dg["1"],d_12=g["2"]-g["1"],d_24=(g["4"]-g["2"])/2,d_48=(g["8"]-g["4"])/4)
            out["linear_fits"] = {}
            for key, vals in (("g_r_fit",g),("b_default_fit",out["slopes"]["default"]),("b_cand4_fit",out["slopes"]["4"])):
                b,a,r2,res = fit(RS,[vals[str(r)] for r in RS])
                f = dict(alpha=a,beta=b,r2=r2)
                if key=="g_r_fit":
                    f.update(residuals=res,residual_table=residual_table(RS,[g[str(r)] for r in RS],b,a))
                out["linear_fits"][key]=f
            out["linear_fits"]["canonical_attribution_ratio"] = dg["1"]/baseline["gap_ns_per_cta"] if cfg==ac.CONFIGS[0] else None
            run["configurations"][cfg] = out
        runs.append(run)
    return runs


def recompute_cross(runs):
    cross = {}
    for cfg in ac.CONFIGS:
        rows = [r["configurations"][cfg] for r in runs]
        cr = {key:aggregate([r["linear_fits"]["g_r_fit"][field] for r in rows]) for key,field in
              (("beta","beta"),("alpha","alpha"),("mean_run_fit_r2","r2"))}
        for key in ("delta_g_1","g_0"):
            cr[key] = aggregate([r["differentials"][key] for r in rows])
        ratios = [r["linear_fits"]["canonical_attribution_ratio"] for r in rows if r["linear_fits"]["canonical_attribution_ratio"] is not None]
        cr["attribution_ratio"] = aggregate(ratios) if ratios else dict(mean=0.0,std=0.0,cv="N/A",stability_class="near_zero_or_sign_unstable",values=[])
        cr["incremental_deltas"] = {k:aggregate([r["incremental_deltas"][k] for r in rows]) for k in ("d_01","d_12","d_24","d_48")}
        cr["r_breakdown"] = {}
        for r in map(str,RS):
            row = {}
            for prefix,values in (("b_default",[x["slopes"]["default"][r] for x in rows]),("b_cand4",[x["slopes"]["4"][r] for x in rows]),
                                  ("g_r",[x["differentials"]["g_r"][r] for x in rows]),("delta_g_r",[x["differentials"]["delta_g_r"][r] for x in rows])):
                ag = aggregate(values)
                row.update({prefix+"_"+k:ag[k] for k in ("mean","std","cv","stability_class")})
            cr["r_breakdown"][r]=row
        ys = [cr["r_breakdown"][str(r)]["g_r_mean"] for r in RS]
        b,a,r2,res = fit(RS,ys)
        cr["cross_mean_fit"] = dict(alpha=a,beta=b,r2=r2,residuals=res)
        cr["residual_table"] = residual_table(RS,ys,b,a)
        cross[cfg] = cr
    return cross


def decision(cross, runs):
    prim = cross[ac.CONFIGS[0]]
    monotonic = lambda values: all(a<=b for a,b in zip(values,values[1:]))
    flags = dict(all_betas_positive=all(b>0 for b in prim["beta"]["values"]),
                 delta_g_1_materially_positive=prim["delta_g_1"]["mean"]>0.1,
                 good_linear_fit=prim["mean_run_fit_r2"]["mean"]>=0.90,
                 is_monotonic=monotonic([prim["r_breakdown"][str(r)]["g_r_mean"] for r in RS]),
                 per_run_monotonic=all(monotonic([run["configurations"][ac.CONFIGS[0]]["differentials"]["g_r"][str(r)] for r in RS]) for run in runs))
    return ("SUPPORTED_AT_REDUCTION_BODY_LEVEL" if all(flags.values()) else "NOT_SUPPORTED"), flags


def timing_bindings(base, raws):
    d = base/"results/phase3/gluon_repeated"
    e = base/"results/phase3/gluon_timing"
    dr = read_json(d/"raw_results.json")
    out = {"raw_hashes": {f"raw_run_{i}.json":hashlib.sha256((e/f"raw_run_{i}.json").read_bytes()).hexdigest() for i in (1,2,3)}, "configurations":{}}
    for cfg in ac.CONFIGS:
        out["configurations"][cfg] = {}
        for c in ac.CANDIDATES:
            # E has no archived R-by-R SHA map. Bind only its recorded candidate
            # SHA and use byte-equal D resource evidence; invent no hash records.
            resource_binding = dict(resources=dr["configurations"][cfg][c]["resources"],
                                    cubin_sha256=raws[0]["configurations"][cfg]["candidates"][c]["cubin_sha256"])
            audit = ac.inspect_repeated(base,e/"artifacts"/cfg,cfg,c,resource_binding,require_r_hashes=False)
            if not audit["all_checks_passed"]:
                raise ValueError(f"E artifact gate failed: {cfg}/{c}: {audit['checks']}")
            for ext in ("ptx","ttgir","sass","resource.txt"):
                if (e/"artifacts"/cfg/f"{c}.{ext}").read_bytes() != (d/"artifacts"/cfg/f"{c}.{ext}").read_bytes():
                    raise ValueError(f"D/E {ext} mismatch: {cfg}/{c}")
            sha = ac.nonempty_text(e/"artifacts"/cfg/f"{c}.cubin.sha256").strip()
            if not all(r["configurations"][cfg]["candidates"][c]["cubin_sha256"]==sha for r in raws):
                raise ValueError("E raw/artifact CUBIN SHA mismatch")
            out["configurations"][cfg][c] = {"artifact_audit":audit,"cubin_sha256":sha,
                "d_e_text_resource_equivalent":True,"binary_archive_limit":"SHA record only; no archived CUBIN or R-by-R E hash map"}
    return out


def validate_timing(base):
    directory = base/"results/phase3/gluon_timing"
    actual = read_json(directory/"results.json")
    raws = [read_json(directory/f"raw_run_{i}.json") for i in (1,2,3)]
    baseline = canonical_baseline(base)
    compare(actual["canonical_baseline"], baseline, "canonical_baseline")
    compare(actual["evidence_bindings"], timing_bindings(base, raws), "E bindings")
    runs = recompute_runs(raws,baseline)
    cross = recompute_cross(runs)
    compare(actual["runs"], runs, "runs")
    expected_cross = {"replication_mode":"same-device temporal replication" if len({r["env_info"]["gpu_uuid"] for r in raws})==1 else "multi-device replication",
                      "cubin_invariance_passed":True,"configurations":cross}
    compare(actual["cross_invocation_summary"],expected_cross,"cross")
    for field,cfg in (("primary_analysis",ac.CONFIGS[0]),("secondary_analysis",ac.CONFIGS[1])):
        compare(actual[field],dict(config=cfg,role="PRIMARY (EXACT_SEQUENCE_EQUIVALENT)" if field=="primary_analysis" else "SECONDARY_CONTROL (PIPELINED_OPCODE_EQUIVALENT)",cross_run=cross[cfg]),field)
    status,flags = decision(cross,runs)
    h = actual["h2_evaluation"]
    compare(h["decision_rule"],DECISION_RULE,"decision rule")
    compare(actual["h2_status"],status,"H2 status")
    compare(h["status"],status,"H2 evaluation status")
    for k,v in flags.items(): compare(h[k],v,k)
    compare(h["sub_hypotheses"]["H2a"]["status"],status,"H2a status")
    compare(h["sub_hypotheses"]["H2a"]["attribution_scope"],status,"H2a scope")
    compare(h["sub_hypotheses"]["H2a"]["isolated_delta_g_1_ns"],cross[ac.CONFIGS[0]]["delta_g_1"]["mean"])
    compare(h["sub_hypotheses"]["H2a"]["isolated_delta_g_1_std_ns"],cross[ac.CONFIGS[0]]["delta_g_1"]["std"])
    compare(h["sub_hypotheses"]["H2a"]["canonical_attribution_ratio"],cross[ac.CONFIGS[0]]["attribution_ratio"]["mean"])
    for k in ("H2b","H2c"): compare(h["sub_hypotheses"][k]["status"],"UNVERIFIED",k)
    validation = read_json(directory/"validation.json")
    compare(validation["evaluations"],actual,"E validation mirror")
    compare(validation["h2_status"],status)
    compare(validation["replication_mode"],expected_cross["replication_mode"])
    compare(validation["cubin_invariance_passed"],True)
    # Renderer equality checks presentation fidelity, never scientific support.
    from .audit_gluon_timing import render_timing_summary
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/"summary.md"
        render_timing_summary(actual,path)
        compare(ac.nonempty_text(directory/"summary.md"),path.read_text(),"E summary")
    return status


def protected_inputs(base):
    root = base.parents[1]
    inventory = read_json(base/"results/phase3/evidence_inputs.json")
    baseline = "adc35416d4b75a08da1e5b9626e4bfc39fbb4e72"
    compare(inventory["baseline_commit"], baseline, "audited baseline")
    listing = subprocess.check_output(["git", "ls-tree", "-r", "--full-tree", baseline, "--", str(base.relative_to(root))], cwd=root, text=True)
    blobs = {line.split("\t",1)[1]: line.split("\t",1)[0].split()[2] for line in listing.splitlines()}
    prefix = str(base.relative_to(root)) + "/"
    derived = {prefix+"results/phase3/"+stage+"/"+name
               for stage,names in [("gluon_reproduction",["results.json","validation.json","summary.md","design.md"]),
                                   ("gluon_repeated",["results.json","validation.json","summary.md","design.md"]),
                                   ("gluon_timing",["results.json","validation.json","summary.md"]),
                                   ("structural_comparison",["positive_vs_negative.json","summary.md"])]
               for name in names}
    derived.add(prefix+"results/phase3/hypotheses.md")
    expected = {path for path in blobs if path.startswith(prefix+"results/") and path not in derived}
    expected.update(prefix+name for name in ["phase3_audited_annotations.json","audited_phase_annotations.json"])
    if set(inventory["files"]) != expected:
        raise ValueError("Protected inventory omits/adds original evidence paths")
    for path,sha in inventory["files"].items():
        data=(root/path).read_bytes()
        # Bind the new derived inventory to the immutable audited Git tree too.
        git_blob = hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
        if git_blob != blobs[path] or hashlib.sha256(data).hexdigest()!=sha:
            raise ValueError(f"Original evidence changed: {path}")
        if path.endswith(".json"):
            read_json(root/path)
    return len(inventory["files"])


def validate_codegen(base, stage):
    """Reparse actual inputs, then replay only derived reports to a temporary directory.

    Artifact parsing is shared with auditors. Statistical recomputation above is
    independent of the timing auditor; report replay only checks presentation.
    """
    import contextlib
    import io
    import tempfile
    from unittest.mock import patch
    directory = base/"results/phase3"/stage
    raw = read_json(directory/"raw_results.json")
    validation = read_json(directory/"validation.json")
    results = read_json(directory/"results.json")
    if stage=="gluon_repeated":
        from . import audit_gluon_repeated as auditor
        output_attr="STEP_D_DIR"
        for cfg in ac.CONFIGS:
            for c in ac.CANDIDATES:
                actual = ac.inspect_repeated(base,directory/"artifacts"/cfg,cfg,c,raw["configurations"][cfg][c])
                if not actual["all_checks_passed"]:
                    raise ValueError(f"Actual D artifact failed: {cfg}/{c}: {actual['checks']}")
                compare(validation["evaluations"][cfg][c]["offline_artifact_audit"],actual,f"D artifact {cfg}/{c}")
        if validation["timing_gate"]["11_all_structural_conditions"]!="PASS":
            raise ValueError("D complete structural gate failed")
    else:
        from . import audit_gluon_reproduction as auditor
        output_attr="GLUON_DIR"
        for cfg in ac.CONFIGS:
            for c in ac.CANDIDATES:
                arts=directory/"artifacts"/cfg
                hashes=ac.artifact_hashes(arts,c)
                entry=validation["evaluations"][cfg][c]
                compare(entry["artifact_hashes"],hashes,f"C hashes {cfg}/{c}")
                sig=[i["opcode"] for i in ac.initial_load_signature(ac.nonempty_text(arts/f"{c}.ptx"))]
                compare(sig,ac.LOAD_SIGNATURES[cfg,c],f"C LocalLoad {cfg}/{c}")
                compare(entry["localload"]["signature"],sig)
                resources=ac.parse_resource(ac.nonempty_text(arts/f"{c}.resource.txt"))
                rr=raw["configurations"][cfg][c]["resources"]
                for k,v in resources.items():
                    compare(rr["static_shared_bytes" if k=="static_smem_bytes" else k],v,f"C resource {k}")
                if resources["local_bytes"] or resources["stack_bytes"]:
                    raise ValueError("C spill evidence")
                compare(ac.nonempty_text(arts/f"{c}.cubin.sha256").strip(),raw["configurations"][cfg][c]["cubin_sha256"])
                ttgir=ac.nonempty_text(arts/f"{c}.ttgir")
                canon=ac.nonempty_text(base/"results/phase3/fixed_binary_artifacts/canonical"/cfg/f"{c}.ttgir")
                for fam in ("blocked","nvmma_shared"):
                    compare(ac.layout_attrs(ttgir,fam),ac.layout_attrs(canon,fam))
                fp=ac.canonical_fingerprint(base,cfg,c)
                actualfp=ac.fingerprint(ac.ptx_instructions(ac.nonempty_text(arts/f"{c}.ptx")))
                matches=[i for i in range(len(actualfp)-len(fp)+1) if actualfp[i:i+len(fp)]==fp]
                compare(len(matches),1,f"C unique complete fingerprint {cfg}/{c}")
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        with patch.object(auditor,output_attr,Path(tmp)):
            auditor.main()
        compare(validation,read_json(Path(tmp)/"validation.json"),stage+" validation replay")
        compare(results,read_json(Path(tmp)/"results.json"),stage+" results replay")
        for name in ("summary.md","design.md"):
            compare(ac.nonempty_text(directory/name),ac.nonempty_text(Path(tmp)/name),stage+" "+name)
    if any(v.startswith("FAIL") for k,v in validation["criteria"].items() if "Criterion E (Input" not in k):
        raise ValueError(f"{stage}: failed structural criterion")


def validate_structure(base):
    from ..phase3_structural_analysis import build_structural_decomposition_dataset
    dataset=read_json(base/"results/phase3/structural_comparison/positive_vs_negative.json")
    annotations=read_json(base/"phase3_audited_annotations.json")
    compare(dataset,build_structural_decomposition_dataset(annotations),"structural dataset replay")
    for cfg, cd in dataset["configurations"].items():
        for c, row in cd["candidates"].items():
            ttgir=ac.nonempty_text(base/"results/phase3/fixed_binary_artifacts/canonical"/cfg/f"{c}.ttgir")
            counts=ac.elements_per_thread([1,cd["M"],cd["N"]],ac.layout_attrs(ttgir,"blocked"))
            topo=row["reduction_topology"]
            compare(topo["derived_M_elems_per_thread"],counts[1],"M ownership")
            compare(topo["derived_N_elems_per_thread"],counts[2],"N ownership")
            compare(topo["derived_total_elems_per_thread"],math.prod(counts),"total ownership")
            # Independent raw-median -> slope calculation for each structural candidate.
            pilot=read_json(base/"results/phase2/saturation/corrected_pilot_runs.json")
            slopes=[]
            for run in ("run_1","run_2","run_3"):
                grid=pilot[run]["configs"][cfg]["grid_data"]
                meds=[samples_stats(grid[str(b)][c]["raw_samples_us"])["median"] for b in BS]
                slopes.append(fit(BS,meds)[0]*1000)
            compare(row["performance"]["marginal_slope_ns_per_cta"],st.mean(slopes),"structural slope")


def self_test(base):
    """Regression probes intercept reads in memory; original evidence is never edited."""
    import contextlib
    import io
    import re
    import tempfile
    from unittest.mock import patch
    from . import audit_gluon_timing as timing
    directory=base/"results/phase3/gluon_timing"
    original_read=Path.read_text
    checks=[]

    def reject(name, target, transform, validator):
        hits=[]
        def read(path,*args,**kwargs):
            text=original_read(path,*args,**kwargs)
            if path==target:
                hits.append(path)
                return transform(text)
            return text
        with patch.object(Path,"read_text",read), contextlib.redirect_stdout(io.StringIO()):
            try:
                validator()
            except (ValueError,KeyError,FileNotFoundError,TypeError):
                if not hits:
                    raise AssertionError(f"Probe did not reach target: {name}")
            else:
                raise AssertionError(f"Corruption accepted: {name}")
        checks.append(name)

    def json_change(keys,value):
        def change(text):
            data=json.loads(text)
            node=data
            for k in keys[:-1]:node=node[k]
            if value is DELETE:del node[keys[-1]]
            else:node[keys[-1]]=value
            return json.dumps(data)
        return change
    DELETE=object()
    validate_timing(base)
    report=directory/"results.json"
    prefix=["runs",0,"configurations",ac.CONFIGS[0]]
    for name,keys,value in [
        ("per-B std",prefix+["per_b_stats","default","1","16384","std"],999999.0),
        ("per-B mean",prefix+["per_b_stats","4","2","32768","mean"],0.0),
        ("per-B CV",prefix+["per_b_stats","default","0","16384","cv_percent"],0.0),
        ("NaN slope",prefix+["slopes","default","1"],float("nan")),
        ("infinite intercept",prefix+["intercepts","default","1"],float("inf")),
        ("boolean slope",prefix+["slopes","default","1"],True),
        ("missing residual",prefix+["linear_fits","g_r_fit","residual_table"],[]),
        ("missing R",prefix+["slopes","default","8"],DELETE),
        ("cross CV",["cross_invocation_summary","configurations",ac.CONFIGS[0],"g_0","cv"],0.0),
        ("stability label",["primary_analysis","cross_run","alpha","stability_class"],"stable"),
        ("R2 semantics",["primary_analysis","cross_run","cross_mean_fit","r2"],0.999642049199822),
        ("canonical denominator",["canonical_baseline","gap_ns_per_cta"],1.4279),
        ("inconsistent H2a",["h2_evaluation","sub_hypotheses","H2a","status"],"NOT_SUPPORTED")]:
        reject(name,report,json_change(keys,value),lambda:validate_timing(base))
    reject("nonfinite raw sample",directory/"raw_run_1.json",
           json_change(["configurations",ac.CONFIGS[0],"candidates","default","r_timing","0","16384","samples_us",0],float("nan")),lambda:validate_timing(base))
    dp=base/"results/phase3/gluon_repeated/artifacts"/ac.CONFIGS[0]/"default.ptx"
    reject("extra D loop shared load",dp,
           lambda text:re.sub(r'(\$L__BB0_2:[^\n]*\n)',r'\1\tld.shared.b32 %r1, [%r2];\n',text,count=1),
           lambda:validate_codegen(base,"gluon_repeated"))
    ds=dp.with_suffix(".sass")
    reject("explicit D SASS MOV",ds,
           lambda text:re.sub(r'(/\*0500\*/\s+).*?;',r'\1MOV R2, R3;',text,count=1),
           lambda:validate_codegen(base,"gluon_repeated"))
    reject("explicit D SASS IMAD.MOV",ds,
           lambda text:re.sub(r'(/\*0500\*/\s+).*?;',r'\1IMAD.MOV.U32 R2, RZ, RZ, R3;',text,count=1),
           lambda:validate_codegen(base,"gluon_repeated"))
    # Replace a later load only, preserving first opcode and total count.
    cp=base/"results/phase3/gluon_reproduction/artifacts"/ac.CONFIGS[0]/"4.ptx"
    def later_load(text):
        pos=[m.start() for m in re.finditer("ld.shared.v2.b32",text)]
        i=pos[1]
        return text[:i]+text[i:].replace("ld.shared.v2.b32","ld.shared.b32",1)
    reject("later C LocalLoad width",cp,later_load,lambda:validate_codegen(base,"gluon_reproduction"))
    ep=directory/"artifacts"/ac.CONFIGS[0]/"default.resource.txt"
    reject("E resource mismatch",ep,lambda text:text.replace("REG:32","REG:33"),lambda:validate_timing(base))
    def missing(_):raise FileNotFoundError("Synthetic missing E artifact")
    reject("missing E artifact",directory/"artifacts"/ac.CONFIGS[0]/"default.ptx",missing,lambda:validate_timing(base))
    inventory=base/"results/phase3/evidence_inputs.json"
    first=next(iter(read_json(inventory)["files"]))
    reject("incomplete protected inventory",inventory,json_change(["files",first],DELETE),lambda:protected_inputs(base))
    reject("protected hash corruption",inventory,json_change(["files",first],"0"*64),lambda:protected_inputs(base))
    # Synthetic negative observations preserve the actual artifact/SHA identities.
    # The complete timing validator must accept fidelity with NOT_SUPPORTED.
    raws=[read_json(directory/f"raw_run_{i}.json") for i in (1,2,3)]
    swapped=copy.deepcopy(raws)
    for raw in swapped:
        for cfg in ac.CONFIGS:
            cand=raw["configurations"][cfg]["candidates"]
            cand["default"]["r_timing"],cand["4"]["r_timing"]=cand["4"]["r_timing"],cand["default"]["r_timing"]
    synthetic={f"raw_run_{i}.json":json.dumps(raw) for i,raw in enumerate(swapped,1)}
    original_open=open
    original_bytes=Path.read_bytes
    with tempfile.TemporaryDirectory() as tmp:
        def source_open(path,*args,**kwargs):
            if Path(path).name in synthetic:
                return io.StringIO(synthetic[Path(path).name])
            return original_open(path,*args,**kwargs)
        def source_read(path,*args,**kwargs):
            if path.parent==directory and path.name in synthetic:
                return synthetic[path.name]
            if path.parent==directory and path.name in {"results.json","validation.json","summary.md"}:
                return original_read(Path(tmp)/path.name,*args,**kwargs)
            return original_read(path,*args,**kwargs)
        def source_bytes(path):
            if path.parent==directory and path.name in synthetic:
                return synthetic[path.name].encode()
            return original_bytes(path)
        with patch("builtins.open",source_open), patch.object(timing,"TIMING_DIR",Path(tmp)), \
             patch.object(Path,"exists",return_value=True), patch.object(Path,"read_text",source_read), \
             patch.object(Path,"read_bytes",source_bytes), contextlib.redirect_stdout(io.StringIO()):
            timing.main()
            compare(validate_timing(base),"NOT_SUPPORTED","outcome-neutral complete validator")
            result=read_json(Path(tmp)/"results.json")
            compare(result["h2_evaluation"]["sub_hypotheses"]["H2a"]["status"],"NOT_SUPPORTED")
    checks.append("complete validator accepts consistent NOT_SUPPORTED / auditor negative packaging")
    # Shape-aware duplicated ownership must not collapse to tile_size/thread_count.
    compare(ac.elements_per_thread([1,2,4],"sizePerThread = [1,1,4], threadsPerWarp = [1,2,16], warpsPerCTA = [1,8,1]"),[1,1,4])
    checks.append("small-shape duplicated ownership")
    print(f"PASS: {len(checks)} offline regression probes; original samples/artifacts were only read.")
    for name in checks: print(f"  PASS: {name}")
