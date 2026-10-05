"""Full-kernel effects, fixed practical bands and secondary grid-time fits."""
import argparse
from collections import Counter
import math
import statistics as st
from experiments.tma_reduction_layout.phase9 import common as c, timing_contract as tc


def effect(values):
    mean,sd = st.mean(values),st.stdev(values)
    half = 4.302652729911275*sd/math.sqrt(3)
    low,high = mean-half,mean+half
    status = "BENEFIT" if low > 3 else "REGRESSION" if high < -3 else "WITHIN_PRACTICAL_BAND" if low >= -3 and high <= 3 else "UNRESOLVED"
    return {"invocation_values_percent":values,"mean_percent":mean,"sample_SD_percent":sd,
            "lower_percent":low,"upper_percent":high,"status":status}


def derive(target):
    validation = tc.validate_target(target)
    gate = c.read(c.OUT / "stage_b" / target / "gate.json")
    root = c.OUT / "stage_c" / target
    observations,fits = {},{}
    for invocation in (1,2,3):
        raw = c.read(root / f"raw_invocation_{invocation}.json")
        samples = {}
        for visit in raw["visits"]: samples.setdefault(visit["condition_tag"],[]).extend(visit["samples_us"])
        c.require(all(len(x) == 100 for x in samples.values()),"Exactly100 scalar observations per condition")
        statistics = {tag:{"median_us":st.median(v),"mean_us":st.mean(v),"sample_SD_us":st.stdev(v),
                          "IQR_us":st.quantiles(v,n=4,method="inclusive")[2]-st.quantiles(v,n=4,method="inclusive")[0],
                          "min_us":min(v),"max_us":max(v),"n":len(v)} for tag,v in samples.items()}
        observations[str(invocation)] = statistics
        groups = {}
        for tag,s in statistics.items():
            cfg,h,can,b = tag.split(":")
            groups.setdefault(f"{cfg}:{h}:{can}",[]).append((int(b[1:]),s["median_us"]))
        fits[str(invocation)] = {key:c.ols(sorted(points)) for key,points in groups.items()}
    outcomes = []
    for row in gate["pairs"]:
        outcome = {"pair":row["pair"],"target":target,"eligible":row["eligible"]}
        if not row["eligible"]:
            outcome.update({"status":"UNAVAILABLE","exclusions":row["exclusions"]})
        else:
            outcome["resource_context"] = row
            by_b = {}
            for b in c.B_VALUES:
                values = []
                times = {}
                for invocation in (1,2,3):
                    stat = observations[str(invocation)]
                    d = stat[f'{row["pair"]}:default:B{b}']["median_us"]
                    a = stat[f'{row["pair"]}:4:B{b}']["median_us"]
                    values.append(100*(d-a)/d)
                    times[str(invocation)] = {"default_us":d,"cand4_us":a,"difference_us":d-a}
                by_b[str(b)] = {**effect(values),"times":times}
            outcome["effects_by_B_RUN"] = by_b
            outcome["primary"] = by_b["65536"]
            outcome["status"] = outcome["primary"]["status"]
            slope_differences = [fits[str(i)][row["pair"]+":default"]["slope_ns_per_additional_CTA"]-
                                 fits[str(i)][row["pair"]+":4"]["slope_ns_per_additional_CTA"] for i in (1,2,3)]
            outcome["secondary_grid_slope_difference_ns_per_additional_CTA"] = {"invocation_values":slope_differences,"mean":st.mean(slope_differences),"sample_SD":st.stdev(slope_differences)}
        outcomes.append(outcome)
    counts = {}
    for h in ("reduction","load_copy","store_copy"):
        counts[h] = dict(sorted(Counter(x["status"] for x in outcomes if x["pair"].split(":")[1] == h).items()))
    return {"target":target,"validation":validation,"pairs":outcomes,"status_counts":counts,
        "statistics":observations,"grid_time_fits":fits,"protocol_SHA256":c.sha((c.OUT / "stage_a/protocol.json").read_bytes()),
        "interpretation":"Fixed retrospective cross-architecture contrasts; descriptor construction and compiler responses included. No intrinsic component latency, universal policy, or production patch validation.",
        "original_case":next(x for x in outcomes if x["pair"] == "M32_N64_w8:reduction")}


def summary(result):
    text = f'# Phase9 StageC — {result["target"]} frozen-binary timing\n\n'
    v = result["validation"]
    text += f'{v["eligible_pairs"]} pairs; {v["samples"]} event samples; three independent processes; {len(v["physical_GPU_UUIDs"])} observed physical UUIDs. All valid pairs retained, including changed residency.\n\n'
    text += "Primary effect is100*(T_default-T_cand4)/T_default at grid65536. Positive means candidate4 is faster. Fixed3% practical band and df2 descriptive process uncertainty; no multiple-comparison significance claim. Smaller grids and OLS slopes are secondary.\n\n"
    rows = []
    for r in result["pairs"]:
        if r["eligible"]:
            p = r["primary"]
            rows.append([r["pair"],r["status"],f'{p["mean_percent"]:.6f}',f'[{p["lower_percent"]:.6f},{p["upper_percent"]:.6f}]',r["resource_context"]["resource_stratum"]])
        else: rows.append([r["pair"],"UNAVAILABLE","—","—","actual-target pre-timing artifact failure"])
    text += c.table(["Pair","Decision","Mean improvement %","Process band %","Resources"],rows)+"\n\n"
    text += "All100-sample medians/statistics, individual invocation timings/effects, three-grid OLS intercepts/slopes/R²/residuals, exclusions and resource contexts are retained in results.json. No model fit or case removal. This comparison does not identify a historical bad compiler commit or verify a production fix.\n"
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target",default="all")
    p.add_argument("--validate",action="store_true")
    args = p.parse_args()
    c.protect()
    for target in c.TARGETS if args.target == "all" else [args.target]:
        result = derive(target)
        root = c.OUT / "stage_c" / target
        for name,blob in (("results.json",c.encode(result)),("summary.md",summary(result).encode())):
            if args.validate: c.require((root / name).read_bytes() == blob,"Timing analysis stale: "+name)
            else: (root / name).write_bytes(blob)
        print(target,result["status_counts"],"PASS")


if __name__ == "__main__":
    main()
