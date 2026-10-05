"""Unfitted runtime descriptor contrasts and fixed prospective decisions."""
from fractions import Fraction
import json
import math
import statistics as st
import sys
from experiments.tma_reduction_layout.phase8 import common as c, timing_contract as tc
from experiments.tma_reduction_layout.phase6.analyze import decide

METRICS = ("G", "H_static", "D_static", "H_switch", "D_switch", "E_switch", "E_static",
           "G_minus_H_static", "G_minus_H_switch", "G_minus_D_switch", "E_switch_minus_E_static", "zero")


def score(rows,target,predictor):
    errors = [row[predictor]["mean"]-row[target]["mean"] for row in rows]
    return {"MAE": st.mean(abs(e) for e in errors), "RMSE":math.sqrt(st.mean(e*e for e in errors))} if errors else {}


def derive(stage):
    tc.validate_raw(stage)
    cases,binaries,plan=tc.inputs(stage);root=c.OUT/stage
    all_statistics,all_fits,metrics={},{},{}
    for number in (1,2,3):
        raw=c.read(root/f"raw_invocation_{number}.json");samples={}
        for visit in raw["visits"]:samples.setdefault(visit["condition_tag"],[]).extend(visit["samples_us"])
        statistics={tag:{"median_us":st.median(v),"mean_us":st.mean(v),"sample_SD_us":st.stdev(v),
                         "min_us":min(v),"max_us":max(v)} for tag,v in samples.items()}
        fits={}
        for cfg in cases:
            for path in c.PATHS:
                for candidate in c.CANDIDATES:
                    key=f"{cfg}:{path}:{candidate}"
                    fits[key]=c.ols([(b,statistics[key+f":B{b}"]["median_us"]) for b in c.B_VALUES])
            def gap(path):
                return fits[f"{cfg}:{path}:default"]["slope_ns_per_additional_CTA"]-fits[f"{cfg}:{path}:4"]["slope_ns_per_additional_CTA"]
            row={"G":gap("canonical"),"H_static":gap("host_canonical"),"D_static":gap("device_canonical"),
                 "H_switch":gap("switch_host"),"D_switch":gap("switch_device"),"zero":0.0}
            row.update(E_switch=row["D_switch"]-row["H_switch"],E_static=row["D_static"]-row["H_static"],
                       G_minus_H_static=row["G"]-row["H_static"],G_minus_H_switch=row["G"]-row["H_switch"],
                       G_minus_D_switch=row["G"]-row["D_switch"])
            row["E_switch_minus_E_static"]=row["E_switch"]-row["E_static"]
            metrics.setdefault(cfg,[]).append(row)
        all_statistics[str(number)],all_fits[str(number)]=statistics,fits
    rows={cfg:{**case,"invocation_metrics":metrics[cfg],**{name:c.sign([v[name] for v in metrics[cfg]]) for name in METRICS},
               "resource_context":{key.split(":")[1]+":"+key.split(":")[2]:{
                   "num_regs":b["metadata"]["resources"]["num_regs"],"dynamic_smem_bytes":b["metadata"]["dynamic_smem_bytes"],
                   "blocks_per_sm":b["occupancy"]["blocks_per_sm_actual_dynamic_smem"]}
                    for key,b in binaries.items() if key.split(":")[0]==cfg}}
          for cfg,case in cases.items()}
    definitions={"H8_01_SAME_BINARY_DESCRIPTOR_TRANSFER":("G","H_switch","D_switch"),
                 "H8_02_DESCRIPTOR_INCREMENT":("G_minus_H_static","zero","E_switch")}
    comparisons={}
    for scope,pop in (("ALL_ELIGIBLE",list(rows.values())),("PRIMARY",[row for row in rows.values() if row["final_class"]=="PRIMARY"])):
        hypotheses={}
        for name,(target,ref,alt) in definitions.items():
            a,b=score(pop,target,ref),score(pop,target,alt)
            reduction=a["MAE"]-b["MAE"] if a else None
            relative=reduction/a["MAE"] if a and a["MAE"] else None
            hypotheses[name]={"n":len(pop),"target":target,"reference":ref,"alternative":alt,
                              "reference_errors":a,"alternative_errors":b,
                              "decision":decide(a,b,len(pop)) if a and stage=="stage_d" else "DIAGNOSTIC_ONLY",
                              "MAE_decrease_ns_per_CTA":reduction,"relative_MAE_decrease":relative,
                              "practical_threshold_met": bool(reduction is not None and relative is not None and reduction>=0.01 and relative>=0.1),
                              "case_errors":{row["case_id"]:{"reference":row[ref]["mean"]-row[target]["mean"],
                                  "alternative":row[alt]["mean"]-row[target]["mean"]} for row in pop},
                              "leave_one_out":{row["case_id"]:{"reference":score([r for r in pop if r is not row],target,ref),
                                  "alternative":score([r for r in pop if r is not row],target,alt)} for row in pop} if len(pop)>1 else {}}
        comparisons[scope]=hypotheses
    return json.loads(c.encode({"stage":stage,"role":"OLD_CASE_DIAGNOSTIC" if stage=="stage_c" else "PROSPECTIVE_FIXED_UNSEEN_PRIMARY_COHORT",
             "units":"ns/additional CTA; layout gaps and runtime descriptor-mode contrasts; not intrinsic component latency",
             "cases":rows,"condition_statistics":all_statistics,"fits":all_fits,"comparisons":comparisons,
             "protocol_SHA256":c.sha((c.OUT/"stage_a/protocol.json").read_bytes()),
             "raw_manifest_SHA256":c.sha((root/"raw_manifest.json").read_bytes()),
             "limits":["All admitted cases retained; coefficient1/intercept0, no refit or favorable-outcome repeats.",
                       "Mode0/1 share CUBIN/module/function/resource allocation; their descriptor issue paths and runtime branch differ.",
                       "Runtime intervention is not pure tensormap construction latency, a causal share, or equivalence between different binaries.",
                       "Three independent processes may share GPU UUIDs; small/sign-unresolved/negative effects remain in every analysis.",
                       "NCU aggregates are separate descriptive evidence; its duration never enters these event fits.",
                       "Fresh cohort extends warp count; coverage does not establish arbitrary shape/warp generalization.",
                       "H2b/H2c remain UNVERIFIED; no production heuristic or compiler change."]}))


def report(result):
    def val(row,name):
        v=row[name];return f'{v["mean"]:.12g} ± {v["sample_SD"]:.12g} ({v["category"]})'
    lines=["# Phase8 "+result["stage"]+" — Frozen runtime descriptor results","",result["role"],"",result["units"],"",
           c.table(["Case","Class","G","Host static","Device static","Host switch","Device switch","Switch mode effect"],
           [[cfg,row["final_class"],*[val(row,name) for name in ("G","H_static","D_static","H_switch","D_switch","E_switch")]]
            for cfg,row in sorted(result["cases"].items())]),"",
           c.table(["Scope","Hypothesis","n","Reference MAE/RMSE","Alternative MAE/RMSE","Decision","Practical threshold"],
           [[scope,name,h["n"],h["reference_errors"],h["alternative_errors"],h["decision"],h["practical_threshold_met"]]
            for scope,hs in result["comparisons"].items() for name,h in hs.items()]),"",
           "Every invocation value, sign band, OLS intercept/R²/residual, per-case prediction error, leave-one-out result and resource context is retained in results.json.",""]
    lines += ["- "+limit for limit in result["limits"]]
    return "\n".join(lines)+"\n"


def validate(stage):
    root=c.OUT/stage;expected=derive(stage)
    c.require(c.read(root/"results.json") == expected and (root/"summary.md").read_text() == report(expected), "Full independent raw-derived/report closure")
    for number in (1,2,3):
        samples={}
        for visit in c.read(root/f"raw_invocation_{number}.json")["visits"]:samples.setdefault(visit["condition_tag"],[]).extend(visit["samples_us"])
        for tag,values in samples.items():
            values=sorted(values);median=float((Fraction(values[49])+Fraction(values[50]))/2)
            c.require(median==expected["condition_statistics"][str(number)][tag]["median_us"], "Independent order-statistic median")
        for key,fit in expected["fits"][str(number)].items():
            x=list(map(Fraction,c.B_VALUES));y=[Fraction(expected["condition_statistics"][str(number)][key+f":B{b}"]["median_us"]) for b in c.B_VALUES]
            slope=(3*sum(a*b for a,b in zip(x,y))-sum(x)*sum(y))/(3*sum(a*a for a in x)-sum(x)**2)
            c.require(float(slope*1000)==fit["slope_ns_per_additional_CTA"],"Independent rational OLS")
    c.write(root/"analysis_validation.json",{"status":"PASS","stage":stage,"recomputed_cases":len(expected["cases"]),
            "independent_medians_and_OLS":True,"no_refit":True,"all_original_bytes_unchanged":True})
    return expected


if __name__ == "__main__":
    stage=sys.argv[1];root=c.OUT/stage
    if "--validate" in sys.argv:
        result=validate(stage);print("PASS",stage,{scope:{name:h["decision"] for name,h in hs.items()} for scope,hs in result["comparisons"].items()})
    else:
        result=derive(stage);c.write(root/"results.json",result);(root/"summary.md").write_text(report(result));print("Derived",stage)
