"""Prepare and check a review draft; no GPU, Modal, compilation or scoring."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GEOMETRIES = [(32,128,4),(32,64,8)]


def make_layout(m,n,warps,vec,within_warp=False):
    if within_warp:
        tn = n // (vec*warps)
        assert n % (vec*warps) == 0 and 1 <= tn <= 32
        tm,wn,wm = 32//tn,warps,1
    else:
        tn = min(32,n//vec)
        tm = 32//tn
        wn = min(warps,n//(vec*tn))
        wm = warps//wn
    base_m,base_n = tm*wm,vec*tn*wn
    assert base_n == n and m % base_m == 0
    spt,tpw,wpc = [1,vec],[tm,tn],[wm,wn]

    def owner(row,col):
        lane_m,lane_n = row % tm,(col//vec) % tn
        warp_m,warp_n = (row//tm) % wm,(col//(vec*tn)) % wn
        return warp_n+wn*warp_m,lane_n+tn*lane_m

    ownership = Counter(owner(r,c) for r in range(m) for c in range(n))
    per_thread = m*n//(32*warps)
    assert len(ownership) == 32*warps and set(ownership.values()) == {per_thread}
    column = Counter(owner(r,0) for r in range(m))
    assert len(column) == tm*wm and sum(column.values()) == m
    return {"sizePerThread":spt,"threadsPerWarp":tpw,"warpsPerCTA":wpc,"order":[1,0],
            "actual_rank":3,"leading_batch_dimension":1,"base_coverage":[base_m,base_n],
            "gluon_blocked_layout":{"size_per_thread":[1,*spt],
                                    "threads_per_warp":[1,*tpw],
                                    "warps_per_cta":[1,*wpc],"order":[2,1,0]},
            "repeats_M":m//base_m,"logical_values_per_thread":per_thread,
            "column_0_contributors":len(column),"column_0_warps":len({w for w,l in column}),
            "column_0_rows":{f"W{w}:L{l}":[r for r in range(m) if owner(r,0)==(w,l)] for w,l in sorted(column)}}


def main():
    cases = []
    for m,n,w in GEOMETRIES:
        for dtype in ("bfloat16","float32"):
            for path in ("host","device"):
                cases.append({"case_id":f"M{m}_N{n}_w{w}:{dtype}:max:{path}",
                              "BM":m,"BN":n,"num_warps":w,"dtype":dtype,"op":"max","path":path,
                              "input_shape":[1,8192,8192],"output_shape":[256,8192],"reduction_axis":1,
                              "full_grid":(8192//m)*(8192//n),"secondary_grid":(8192//m)*(8192//n)//16,
                              "triton_anchor":{"pipeline":"triton","variant":"upstream_default"},
                              "gluon_candidates":{"pipeline":"gluon",
                                                  "variants":["fixed_vec1","fixed_vec2","fixed_vec4","fixed_vec8"],
                                                  "layout_control":"smem.load(register_layout: gl.constexpr)",
                                                  "output_policy":"common_explicit_store_layout"}})
    layouts = []
    for m,n,w in GEOMETRIES:
        for vec in (1,2,4,8):
            layouts.append({"geometry":[m,n,w],"variant":f"fixed_vec{vec}",
                            "layout":make_layout(m,n,w,vec)})
        for vec in (2,8):
            layouts.append({"geometry":[m,n,w],"variant":f"vec{vec}_within_warp",
                            "layout":make_layout(m,n,w,vec,True)})
    assert len(cases)==8 and len(layouts)==12
    record = {
        "status":"REVIEW_DRAFT_NOT_EXECUTABLE_PREREGISTRATION","date":"2026-10-15",
        "purpose":"Measure layout performance and isolate code-generation/communication effects",
        "score_model_used":False,"old_rule_comparator":False,
        "new_GPU_jobs_started":False,"PPT_modified":False,
        "layout_intervention":{"method":"Gluon explicit BlockedLayout",
                               "native_compiler_modified":False,
                               "reuse":"existing unmodified upstream baseline native/image/cache",
                               "new_kernel_compilation_needed":True,
                               "new_Triton_native_build_needed":False,
                               "hardware_ID_backdoors":False},
        "upstream":"20d9d98fead78c2f65e1b3ba5a7881d86a836dad",
        "targets":[{"gpu":"H100","cc":[9,0]},{"gpu":"B200","cc":[10,0]},
                   {"gpu":"RTX PRO 6000","cc":[12,0]}],
        "execution_order":["A_tool_and_metric_coverage","B_H100_layout_sweep",
                           "C_H100_interventions","D_cross_architecture_and_decision"],
        "cases":cases,"candidate_layouts":layouts,
        "core_logical_artifacts_per_target":40,"core_logical_artifacts_all_targets":120,
        "core_artifacts_per_pipeline_per_target":{"triton_anchor":8,"gluon_layout_candidates":32},
        "optional_extra_reproduction_artifacts_per_target_max":8,
        "canonical_gluon_bridge":{"separate_evidence_records":True,
                                 "same_layout_is_binary_equivalence":False,
                                 "required_checks":["shape/axis/warps/full blocked and shared encoding",
                                                    "complete ordered LocalLoad sequence",
                                                    "reduction fingerprint with relevant immediates",
                                                    "output conversion/store and resources",
                                                    "NCU behavior and separate exact-binary timing"],
                                 "if_not_reproduced":"Gluon findings do not establish canonical Triton regression resolution"},
        "mechanism_logical_variants_per_target":16,
        "mechanism_factors":{"geometries":2,"paths":2,"dtypes_ops":["bfloat16/max"],
                             "layouts":["vec8_n_first","vec8_within_warp","vec2_n_first","vec2_within_warp"],
                             "initial_loader":["unmodified_compiler_default"]},
        "loader_opcode_intervention":{"status":"DEFERRED_NOT_REGISTERED_FOR_THIS_ROUND",
                                     "prerequisite":"NCU/code-generation evidence and separately frozen implementation/protocol"},
        "timing":{"worker_invocations":7,"physical_GPU_independence_claimed":False,
                  "primary":"full_grid_single_kernel_launch_CUDA_events",
                  "auxiliary":["sixteenth_grid_single_kernel_launch_CUDA_events"],
                  "kernel_launches_per_scalar_sample":1,"batch_and_divide":False,
                  "graph_timing":False,"primary_comparison_pipeline":"gluon",
                  "unique_binary_aliases":"CUBIN/function/ABI/resource/parameter equality",
                  "order":"2K counterbalanced rows for K unique Gluon candidate binaries; K <= 4",
                  "anchor_and_bridge_order":"separately frozen before execution",
                  "samples_per_round_per_binary":5,
                  "same_case_pairwise_band":"95% simultaneous Bonferroni/t paired log ratios; df=6",
                  "practical_difference_percent":3,"outcome_based_repeats":False},
        "profiling":{"launches_per_unique_binary":3,"graph_profiling":False,
                     "cache_control_requested":"none","clock_control_requested":"none",
                     "metric_mapping":"Unfrozen until capability/coverage probe",
                     "missing_or_unsupported":"UNAVAILABLE, never zero",
                     "phase_attribution":"PC/source evidence or UNATTRIBUTED",
                     "profile_duration_is_performance_sample":False},
        "pending_before_execution":["Validated architecture-specific metric mapping and calibration fixture",
                                    "Adapted explicit-layout Gluon kernel and canonical Triton reproduction bridge",
                                    "Source/unmodified baseline native/assembler/worker snapshots and SHA",
                                    "Exact single-launch/order/statistics implementations and frozen protocol"],
        "plan_SHA256":sha256((ROOT/"PLAN.md").read_bytes()).hexdigest(),
    }
    target = ROOT/"draft_protocol.json"
    target.write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"status":record["status"],"cases":len(cases),"layouts":len(layouts),
                      "core_artifacts_per_target":40,"mechanism_variants_per_target":16,
                      "layout_ownership_checked":True,"GPU_jobs_started":False},ensure_ascii=False))


if __name__ == "__main__":
    main()
