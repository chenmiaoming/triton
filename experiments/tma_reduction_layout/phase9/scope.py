"""StageD scope assessment; no policy is fitted to this matrix."""
import argparse
from experiments.tma_reduction_layout.phase9 import common as c


def derive():
    results = {target:c.read(c.OUT / "stage_c" / target / "results.json") for target in c.TARGETS}
    targets = {}
    regressions,unresolved,unavailable = [],[],[]
    secondary_unresolved,secondary_regressions = [],[]
    for target,result in results.items():
        targets[target] = {"original_case":result["original_case"],"status_counts":result["status_counts"],"validation":result["validation"]}
        for row in result["pairs"]:
            item = {"target":target,"pair":row["pair"],"status":row["status"]}
            if row["eligible"]: item["primary"] = row["primary"]
            if row["status"] == "REGRESSION": regressions.append(item)
            elif row["status"] == "UNRESOLVED": unresolved.append(item)
            elif row["status"] == "UNAVAILABLE": unavailable.append(item)
            if row["eligible"]:
                for b,estimate in row["effects_by_B_RUN"].items():
                    if b == "65536": continue
                    secondary = {"target":target,"pair":row["pair"],"B_RUN":int(b),"effect":estimate}
                    if estimate["status"] == "UNRESOLVED": secondary_unresolved.append(secondary)
                    elif estimate["status"] == "REGRESSION": secondary_regressions.append(secondary)
    return {"status":"CROSS_ARCHITECTURE_SCOPE_ASSESSED","targets":targets,
        "regressions":regressions,"unresolved":unresolved,"unavailable":unavailable,
        "secondary_unresolved":secondary_unresolved,"secondary_regressions":secondary_regressions,
        "global_vector4_policy_justified":False,
        "reason":"Observed regressions defeat a global vector4 policy" if regressions else "Finite retrospective matrix and incomplete/uncertain safety coverage do not establish a universal vector4 policy",
        "next_fix_requirements":["identify original good/bad compiler contrast separately from optimization baseline",
            "use explicit target/IR applicability conditions; descriptor load/reduction context analysis belongs in an analysis/transformation",
            "preserve descriptor store and unrelated consumers unless independently justified",
            "freeze a minimal patch and verify it on new cases before claiming held-out policy validation",
            "retain all regressions/resource exclusions; no threshold tuning or negative-case removal"],
        "no_production_change":True,"no_PR_created":True,"PR_requires_user_approval":True,"stop_after_phase9":True,
        "limitations":["same native compiler but architecture-selected ptxas/toolchain identities differ and are recorded",
            "three separate timing processes may reuse a physical GPU; UUIDs measured, not assumed",
            "process bands are descriptive and not familywise corrected",
            "historically selected shapes and a limited BF16/max kernel family; no arbitrary workload or all-TMA safety claim",
            "unavailable shapes are not performance equivalence; throughput effects are not intrinsic descriptor/lane/bank latency"]}


def summary(result):
    text = "# Phase9 StageD — Architecture scope decision\n\n"
    text += "All three requested architectures were allocated and measured with their own archived CUBINs. The exact native core was reused; no compiler/production change or PR was made.\n\n"
    rows = []
    for target,r in result["targets"].items():
        original = r["original_case"]["primary"]
        rows.append([target,original["status"],f'{original["mean_percent"]:.6f}',f'[{original["lower_percent"]:.6f},{original["upper_percent"]:.6f}]',r["validation"]["samples"],r["status_counts"]])
    text += c.table(["Target","Original M32_N64_w8","Mean improvement %","Process band %","Samples","All primary decisions"],rows)+"\n\n"
    text += "A global vector4 policy is not justified. "+result["reason"]+". Whole-kernel benefits include descriptor setup, layout lowering, registers, residency and scheduling. Source-level candidate selection is not a production patch.\n\n"
    if not result["regressions"]:
        text += "No primary pair crossed the preregistered3% regression threshold. This is finite tested coverage, not an all-TMA safety claim.\n\n"
    text += "## Retained regression cases\n\n"
    text += c.table(["Target","Pair","Mean improvement %","Process band %"],
        [[r["target"],r["pair"],f'{r["primary"]["mean_percent"]:.6f}',f'[{r["primary"]["lower_percent"]:.6f},{r["primary"]["upper_percent"]:.6f}]'] for r in result["regressions"]])+"\n\n"
    text += f'{len(result["unresolved"])} unresolved and {len(result["unavailable"])} unavailable pairs remain explicit in results.json. The two unavailable SM120 large tiles exceed actual shared-memory limits; they are not replaced by smaller tiles.\n\n'
    text += "## Secondary grids with unresolved practical effects\n\n"
    text += c.table(["Target","Pair","Grid","Mean improvement %","Process band %"],
        [[r["target"],r["pair"],r["B_RUN"],f'{r["effect"]["mean_percent"]:.6f}',f'[{r["effect"]["lower_percent"]:.6f},{r["effect"]["upper_percent"]:.6f}]'] for r in result["secondary_unresolved"]])+"\n\n"
    text += f'{len(result["secondary_unresolved"])} secondary-grid effects remain unresolved; {len(result["secondary_regressions"])} meet the regression rule. Some bands include decreases beyond3%; the larger-grid result does not establish non-regression at these smaller grids. They remain in the same fixed cohort with no retiming or threshold change.\n\n'
    text += "Next implementation should be scoped to justified target/IR contexts, preserve unrelated descriptor loads/stores, and receive separate fixed-patch validation. Original historical good/bad compiler attribution is still outstanding. Every old sample and artifact remains unchanged. No later phase starts automatically; user approval is mandatory before creating any PR, including draft.\n"
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validate",action="store_true")
    args = p.parse_args()
    c.protect()
    result = derive()
    root = c.OUT / "stage_d"
    root.mkdir(parents=True,exist_ok=True)
    for name,blob in (("results.json",c.encode(result)),("summary.md",summary(result).encode())):
        if args.validate: c.require((root / name).read_bytes() == blob,"Scope report stale: "+name)
        else: (root / name).write_bytes(blob)
    print("Scope report PASS; global vector4 policy justified=False; PR approval required")


if __name__ == "__main__":
    main()
