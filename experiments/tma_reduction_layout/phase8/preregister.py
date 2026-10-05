"""Offline influence audit and prospective contracts; no GPU or compilation."""
import json
import math
import statistics as st
import sys
from experiments.tma_reduction_layout.phase8 import common as c
from experiments.tma_reduction_layout.phase7.contracts import old_ids
from experiments.tma_reduction_layout.phase4.phase5.preregister import predicted_candidate
from experiments.tma_reduction_layout.phase4.preregister import select

DEST = c.OUT / "stage_a"
DIAGNOSTIC = ("M128_N32_w16", "M512_N64_w16", "M1024_N32_w16", "M2048_N32_w16")


def influence():
    previous = c.read(c.BASE / "results/phase7/stage_d/results.json")["cases"]
    def scores(rows):
        return {name: {"MAE": st.mean(abs(row[name]["mean"] - row["G"]["mean"]) for row in rows),
                       "RMSE": math.sqrt(st.mean((row[name]["mean"] - row["G"]["mean"]) ** 2 for row in rows))}
                for name in ("H00", "H01", "H10", "H11")}
    total = sum(abs(r["H00"]["mean"]-r["G"]["mean"])-abs(r["H11"]["mean"]-r["G"]["mean"])
                for r in previous.values())
    return {"original": scores(list(previous.values())), "cases": {
        cfg: {"class": row["final_class"], "absolute_error_reduction_H00_to_H11":
              abs(row["H00"]["mean"]-row["G"]["mean"])-abs(row["H11"]["mean"]-row["G"]["mean"]),
              "signed_share_of_total_error_reduction":
              (abs(row["H00"]["mean"]-row["G"]["mean"])-abs(row["H11"]["mean"]-row["G"]["mean"]))/total,
              "leave_one_out": scores([r for other, r in previous.items() if other != cfg])}
        for cfg, row in previous.items()}, "no_refit": True, "no_new_observations": True}


def derive():
    protected = c.protect()
    historical = old_ids() | {r["config_id"] for r in c.read(c.BASE / "results/phase7/stage_b_prereg/pool.json")["cases"]}
    fresh = [(m, n, 32) for m in (64, 128, 256, 512) for n in (32, 64, 128)]
    fresh += [(m, 256, 16) for m in (32, 64, 128, 256)]
    cases = []
    for role, identities in (("DIAGNOSTIC", DIAGNOSTIC), ("FRESH", [f"M{m}_N{n}_w{w}" for m,n,w in fresh])):
        for cfg in identities:
            m,n,w = [int(part[1:]) for part in cfg.split("_")]
            c.require((cfg in historical) == (role == "DIAGNOSTIC"), "Historical/fresh identity segregation")
            a,b = predicted_candidate(m,n,w), predicted_candidate(m,n,w,4)
            cases.append({"config_id": cfg, "M": m, "N": n, "num_warps": w, "origin": role,
                          "logical_shape": [1,m,n], "reduction_axis": 1, "default": a, "cand4": b})
    protocol = {
        "phase": "PHASE_8", "baseline": c.BASELINE, "stages": ["A_INFLUENCE_PREREG", "B_SAME_BINARY_FEASIBILITY_ARTIFACTS",
        "C_OLD_CASE_DIAGNOSTIC", "D_FRESH_CONFIRMATORY"], "diagnostic": list(DIAGNOSTIC),
        "fresh_domain": {"warp32": {"M": [64,128,256,512], "N": [32,64,128]},
                         "wide_warp16": {"M": [32,64,128,256], "N": [256]}},
        "historical_identities": sorted(historical), "fresh_count": 16,
        "source": "One Gluon kernel per layout with runtime do_not_specialize mode=0/1. Shared allocation/barrier and reduction/output store outside the uniform branch; host descriptor passed by value, device descriptor constructed only in mode1. No lane/warp scalars or hardware IDs.",
        "paths": list(c.PATHS), "canonical_reference": "Unchanged Phase4 Triton canonical source; static host/device canonical-output Gluon controls unchanged from Phase7",
        "dimensions": {"B_DESC": 65536, "B_RUN": list(c.B_VALUES), "invocations": 3, "rounds": 10,
                       "samples_per_visit": 10, "samples_per_condition": 100, "warmups": 3},
        "inputs": "Seed42 BF16 fully initialized descriptor input; same input/output/scratch per shape for every path/layout/mode. Full FP32 output and B_DESC*128 scratch. No amplification or descriptor shrink.",
        "structural_rule": "Reuse the original Phase4 select/valid-layout A-H rules: supported/legal/distinct layouts, unchanged warpPart[M], decreasing lanePart[M]. Keep all16 identities and their structural exclusions in the pool; compile only included identities.",
        "admission": "All exports/actual PTX ABI/full source-bound convert/max fingerprints/shared layout checked; exact same switch CUBIN/module/function/resources across modes; runtime scalar branch present, one shared post-branch reduction; both-mode exact-CUBIN smoke; zero LOCAL/STACK for every binary; actual candidate residency matched within each path; resource failure excludes without replacement; unresolved failure PENDING.",
        "classification": "PRIMARY: all complete reduction opcode sequences match canonical plus all admission gates. SECONDARY: only complete opcode-multiset equivalence plus all gates. Never relax/upgrade by timing. Full operand/SASS hashes retained; same post-branch block verification is separate from opcode classification.",
        "feasibility": "If a valid unspecialized descriptor switch/common reduction cannot be generated, report NOT_IDENTIFIABLE and stop dependent GPU work with recorded stage dispositions; do not substitute invalid scalar programs or two-binary causal claims.",
        "timing": "H100!:1; three separately dispatched single-use processes; CUDA driver events one archived ELF kernel/sample; SHA guard before every launch; no Triton/JIT/compiler imports. Fixed deterministic cyclic schedule offsets17/13. No timing-based selection/reruns.",
        "analysis": "Median100, three-B rational OLS ns/additional CTA; gap default-minus4 for every path. E_switch=gap_switch_device-gap_switch_host; E_static=gap_device_canonical-gap_host_canonical. All three invocation values/means/SD/fixeddf2 sign bands, intercepts/R2/residuals retained. Descriptive leave-one-out and individual-case errors retained.",
        "hypotheses": {"H8_01_SAME_BINARY_DESCRIPTOR_TRANSFER": {"target": "G", "reference": "switch_host", "alternative": "switch_device"},
                       "H8_02_DESCRIPTOR_INCREMENT": {"target": "G_minus_host_canonical", "reference": "zero", "alternative": "E_switch"}},
        "decision": "Coefficient1/intercept0, no fit. ALL_ELIGIBLE and PRIMARY separately; n>=5; both strictly lower MAE/RMSE SUPPORTED, both >= FALSIFIED, mixed INCONCLUSIVE; fewer than5 INCONCLUSIVE_BY_COVERAGE. Practical improvement additionally requires absolute MAE decrease >=0.01 ns/CTA AND relative decrease >=10%; report separately from directional support.",
        "profiling": {"role": "Separate descriptive StageC evidence; never formal timing", "cases": [DIAGNOSTIC[0],DIAGNOSTIC[-1]],
                      "paths": ["switch_host", "switch_device"], "candidates": list(c.CANDIDATES),
                      "metrics": ["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
                                  "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum", "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
                                  "dram__bytes_read.sum", "dram__bytes_write.sum", "smsp__inst_executed.sum"],
                      "settings": {"replay": "kernel", "cache": "all", "clock": "none"}, "missing": "Report unavailable fields/permissions without zeros, substitutions or outcome reruns"},
        "build": "Reuse Phase6 native image excluding experiments; persistent ccache and triton-home; no native source changes/make; exact native/PTX compiler hashes, cache statistics/source snapshot/export provenance retained",
        "repair": "Retain every original return/partial before validation; at most3 whole-invocation infrastructure repairs, never adverse-result repeats. Quota fallback to an already configured Modal profile allowed; all profiles/worker UUIDs recorded.",
        "limits": "Single-binary runtime intervention includes branch/descriptor issue path, not pure tensormap construction latency or a source-free causal share. No equivalence tolerance tuning, production heuristic or compiler changes. H2b/H2c stay UNVERIFIED without their own required mechanism evidence.",
        "stop_after": "Phase8 StageD final validator, four separate stage commits, push existing explore/tma-reduction-layout branch; no Phase9"}
    cases = select(sorted(cases,key=lambda r:r["config_id"]))
    return json.loads(c.encode({"pool": {"cases": cases,
                               "structurally_included": sum(r["included_in_structural_pool"] for r in cases)}, "protocol": protocol,
                               "influence": influence(), "protected_files": len(protected),
                               "protected_inventory_SHA256": c.sha(c.encode(protected))}))


def report(value):
    influence = value["influence"]
    return "\n".join(["# Phase8 StageA — Influence audit and preregistration", "",
        "No GPU, compilation, new timing, or alteration of prior conclusions. Four historical diagnostic cases and16 unseen identities frozen before artifacts or timing.", "",
        c.table(["Previous held-out case", "Class", "H00→H11 absolute error decrease", "Signed share of total decrease"],
                [[cfg,row["class"],row["absolute_error_reduction_H00_to_H11"],row["signed_share_of_total_error_reduction"]]
                 for cfg,row in sorted(influence["cases"].items())]), "",
        "Every leave-one-out MAE/RMSE and negative contribution is retained in contracts.json. The dominant case motivates prospective coverage, not deletion or a fitted correction.", "",
        "StageB must establish a legal block-uniform runtime switch in the same ELF with a shared post-branch reduction. StageC uses old cases for diagnosis; StageD uses the full fixed unseen cohort with n≥5 PRIMARY coverage required for that scope. Failed gates remain excluded without replacement. NCU is separate evidence and profiler duration never enters timing.", "",
        f'All {value["protected_files"]} prior experiment files remain byte-identical. Native core/ccache reuse; no compiler change.']) + "\n"


def main():
    expected = derive()
    if "--validate" in sys.argv:
        c.require(c.read(DEST / "contracts.json") == expected, "Independently recomputed influence/preregistration")
        for name in ("pool", "protocol"):
            c.require(c.read(DEST / (name+".json")) == expected[name], "Frozen " + name)
        c.require((DEST / "summary.md").read_text() == report(expected), "Report closure")
        c.write(DEST / "validation.json", {"status": "PASS", "no_GPU": True, "fresh":16,
                "diagnostic":4, "protected_files":expected["protected_files"]})
        print("PASS Phase8 StageA; four diagnostic and16 unseen cases; no GPU")
    else:
        c.require(not (DEST / "contracts.json").exists(), "Do not overwrite preregistration")
        c.write(DEST / "contracts.json", expected)
        for name in ("pool", "protocol"):
            c.write(DEST / (name+".json"),expected[name])
        (DEST / "summary.md").write_text(report(expected))


if __name__ == "__main__":
    main()
