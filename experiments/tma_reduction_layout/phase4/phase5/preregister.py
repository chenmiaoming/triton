"""Phase 5 held-out structural preregistration; no compiler/runtime imports."""
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4 import preregister as stage_a
from experiments.tma_reduction_layout.phase4 import residual_contract as rc

HERE = Path(__file__).parent
OUT = HERE.parent / "results/phase5/preregistration"
DOMAIN = {"M": [16, 256, 512], "N": [16, 32, 64, 128], "num_warps": [4, 8]}
OUTPUT_NAMES = {"structural_pool.json", "exclusions.json", "protocol.json", "cohort_summary.md", "source_bindings.json"}


def predicted_candidate(m, n, warps, forced=None):
    """Replay the existing descriptor coalescing/BlockedEncoding source rules."""
    per_thread = max(m * n // (warps * 32), 1)
    vec = min(per_thread, 8) if forced is None else forced
    legal = forced is None or (vec <= 8 and vec <= per_thread and n % vec == 0 and vec > 0 and not vec & (vec - 1))
    layout = None
    if legal:
        shape, sizes, order = [1, m, n], [1, 1, vec], [2, 1, 0]
        threads, wpc = [0, 0, 0], [0, 0, 0]
        remaining_lanes, remaining_threads, remaining_warps = 32, warps * 32, warps
        prev_lanes = prev_warps = 1
        for i in order[:-1]:
            per_cta = min(remaining_threads, max(1, shape[i] // sizes[i]))
            threads[i] = min(per_cta, remaining_lanes)
            wpc[i] = min(max(per_cta // threads[i], 1), remaining_warps)
            remaining_warps //= wpc[i]
            remaining_lanes //= threads[i]
            remaining_threads //= per_cta
            prev_lanes *= threads[i]
            prev_warps *= wpc[i]
        threads[order[-1]], wpc[order[-1]] = 32 // prev_lanes, warps // prev_warps
        layout = {"sizePerThread": sizes, "threadsPerWarp": threads, "warpsPerCTA": wpc, "order": order}
    return {"legal": bool(legal), "layout": layout, "compiled_num_warps": warps,
        "num_ctas": 1, "hardware_warp_size": 32, "shared_family": "nvmma_shared", "shared_rank": 3,
        "metadata_evidence": "ANALYTIC_EXPECTED_ATTRIBUTES_ONLY_NOT_COMPILED_METADATA",
        "requested_vec": vec}


def source_domain():
    return [{"M": m, "N": n, "num_warps": w} for m in DOMAIN["M"] for n in DOMAIN["N"] for w in DOMAIN["num_warps"]]


def create_pool(source):
    # Only these three scalar structural fields can cross the source boundary.
    projected = []
    for row in source:
        m, n, w = row["M"], row["N"], row["num_warps"]
        rc.require(type(m) is type(n) is type(w) is int, "Integer source dimensions")
        projected.append({"config_id": f"M{m}_N{n}_w{w}", "M": m, "N": n, "num_warps": w,
            "transition": "default -> cand4", "logical_shape": [1, m, n], "reduction_axis": 1,
            "origin": "HELD_OUT_EXTRAPOLATION", "default": predicted_candidate(m, n, w),
            "cand4": predicted_candidate(m, n, w, 4)})
    expected = {(r["M"], r["N"], r["num_warps"]) for r in source_domain()}
    rc.require(len(projected) == 24 and {(r["M"], r["N"], r["num_warps"]) for r in projected} == expected,
               "Exactly once, the entire 24-transition fixed Cartesian domain")
    rows = stage_a.select(sorted(projected, key=lambda r: r["config_id"]))
    for row in rows:
        row["layout_evidence"] = "SOURCE_RULE_PREDICTION_NO_COMPILATION_OR_ACTUAL_TTGIR"
        row["artifact_gate_status"] = "STRUCTURAL_OR_RESOURCE_PENDING" if row["included_in_structural_pool"] else "STRUCTURALLY_EXCLUDED"
        row["timing_eligible"] = False
        row["predicted_resource_geometry"] = {
            "input_BF16_allocation_bytes": 65536 * row["M"] * row["N"] * 2,
            "output_FP32_allocation_bytes": 65536 * row["N"] * 4,
            "logical_shared_tile_bytes": row["M"] * row["N"] * 2,
            "interpretation": "Logical sizes only, not compiled dynamic/static SMEM or spill/occupancy. M512/N128 needs 8GiB input and 128KiB logical shared tile; actual resource legality remains pending. Frozen B_DESC is not reduced."}
    included = [r for r in rows if r["included_in_structural_pool"]]
    counts = Counter(reason for r in rows for reason in r["reason_codes"])
    coverage = {field: dict(sorted(Counter(str(r[field]) for r in included).items())) for field in ("M", "N", "num_warps")}
    coverage["lane_transition"] = dict(sorted(Counter(f'{r["default"]["lanePart_M"]}->{r["cand4"]["lanePart_M"]}' for r in included).items()))
    return {"phase": "PHASE_5_HELD_OUT_CONFIRMATORY_EXTENSION_PREREGISTRATION", "source_domain": DOMAIN,
        "total_source_transitions": 24, "included_structural_pool": len(included), "excluded_structural_pool": 24 - len(included),
        "reason_counts_nonexclusive": {k: counts[k] for k in stage_a.REASONS}, "coverage": coverage,
        "selection_policy": "Unmodified Stage A select/valid_layout A-H rule; inputs projected to dimensions only; no Stage C performance or expected equivalence used for membership",
        "no_compilation": True, "no_artifact_gate": True, "no_timing": True, "transitions": rows}


def protocol(hypotheses, stage_d):
    return {"phase": "PHASE_5_HELD_OUT_VALIDATION", "stage": "PREREGISTRATION_ONLY_NO_EXECUTION",
        "statistical_role": "held-out confirmatory extension for OUTCOME_INFORMED_HYPOTHESIS; Stage C is development/hypothesis-generation data and cannot confirm these hypotheses",
        "domain": DOMAIN, "domain_rationale": "M16 is one power-of-two below the development M32/64/128 range; M256/512 are above it. Deliberate extrapolation, never prune for expected PRIMARY membership or performance.",
        "structural_rule": "Reuse Stage A default/cand4 legal, distinct supported layouts, unchanged warpPart[M], strictly decreasing lanePart[M], M reduction axis, unchanged num_warps. Source-rule predictions require later actual-artifact verification.",
        "artifact_policy": "No artifact gate performed. Future canonical/single/repeated gates reuse Stage A-B fingerprints, LocalLoad semantics, no reload, unspecialized R, zero spill, matched candidate residency and archived exact CUBIN SHA closure. Failed/missing evidence excludes or remains pending without changing structural membership.",
        "hierarchy": {"PRIMARY": "Both candidates exact sequence in canonical-single reproduction AND repeated runtime-body reproduction, with all frozen isolation/resource gates passed",
            "SECONDARY": "All gates passed, but one or more complete bodies are PIPELINED_OPCODE_EQUIVALENT; never upgrade based on outcomes",
            "minimum_PRIMARY_interpretability": 5, "coverage": "n>=5 permits considering generalization only within observed exact-equivalence coverage; does not itself establish it. Report M/N/warp/lane coverage and anchor-free held-out scope.",
            "others": "EXCLUDE_FROM_TIMING or PENDING; no performance-based selection"},
        "future_timing": {"B_DESC": 65536, "B_RUN": [16384, 32768, 65536], "R": [0, 1], "canonical_R": None,
            "independent_invocations": 3, "rounds": 10, "scalar_samples_per_round": 10, "samples_per_condition": 100,
            "primitive": "CUDA events, one exact archived kernel per scalar sample; SHA guard before every launch",
            "warmup_count_per_condition": 3, "warmup_order": "First frozen round, untimed, same rule for candidates/harnesses/R/B",
            "protocol_failures": "Invalid entire invocation; retain failure and partial data; only whole affected invocation repair, never rerun unfavorable outcomes",
            "resource_pending": "If allocation/compiled resources are illegal or unknown, STRUCTURAL_OR_RESOURCE_PENDING; do not change B_DESC/B_RUN/R, add amplification, or silently adapt the protocol"},
        "future_analysis": {"slopes": "Per invocation condition median of 100 samples; 3-B OLS, ns/additional CTA; retain all R2/residuals and cases",
            "metrics": "G=b_canonical_default-b_canonical_4; g0/g1=repeated differentials; D=g1-g0; G-D descriptive only; three invocation values, means, sample SDs",
            "sign_resolution": {"t_df2": 4.302652729911275, "half_width": "t*sample_SD/sqrt(3)", "categories": ["POSITIVE", "NEGATIVE", "SIGN_UNRESOLVED"]},
            "PRIMARY": "Continuous unrounded Spearman(G,D), exact-tie average ranks; n<3/constant UNDEFINED; LOO same minimum n; complete nine-grid signs and coverage; PRIMARY n<5 insufficient for established generalization",
            "SECONDARY": "Separate Pearson(G,D), complete signs/coverage/counterexamples, lower equivalence strength; no merged primary evidence",
            "hypothesis_predictions": "Use the locked Stage D ALL_ELIGIBLE model coefficients for prospective held-out predictions; no refitting or hypothesis changes after Phase 5 outcomes. Evaluate all timing-eligible cases; failed/singular scopes remain INCONCLUSIVE.",
            "threshold_policy": "No p-values, significance claim or tuned support cutoff. Directional/paired-error comparisons below test specified hypotheses and do not automatically establish generalization or causality."},
        "hypotheses": hypotheses, "locked_development_models": {m: v["coefficients"] for m, v in stage_d["models"]["ALL_ELIGIBLE"].items()},
        "execution_authorized": False, "observed_Phase5_timing_samples": 0, "Phase5_artifact_gate_executed": False,
        "scope_preservation": "Stage A-C membership, classes, evidence and scientific status remain unchanged; no production heuristic or PR modification"}


def render(pool, policy):
    lines = ["# Phase 5 — Held-Out Validation Preregistration", "", policy["statistical_role"], "",
        f'24 source transitions; {pool["included_structural_pool"]} structurally included; {pool["excluded_structural_pool"]} excluded. '
        'All layout/resource fields are source-rule predictions. No Phase 5 artifact gate or timing has been executed.', "",
        "Each layout cell is sizePerThread / threadsPerWarp / warpsPerCTA, with order [2,1,0]. Illegal candidates have no predicted layout.", "",
        "| Case | M | N | Warps | Default layout | Cand4 layout | Include/exclude | Reasons | Resource/artifact state |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in pool["transitions"]:
        def layout(c):
            v = r[c]["layout"]
            return "ILLEGAL" if v is None else " / ".join(str(v[k]) for k in ("sizePerThread", "threadsPerWarp", "warpsPerCTA"))
        lines.append(f'| {r["config_id"]} | {r["M"]} | {r["N"]} | {r["num_warps"]} | {layout("default")} | {layout("cand4")} | '
            f'{"INCLUDE" if r["included_in_structural_pool"] else "EXCLUDE"} | {", ".join(r["reason_codes"]) or "PASSED_STAGE_A_STRUCTURAL_RULE"} | {r["artifact_gate_status"]} |')
    lines += ["", "Coverage: `" + json.dumps(pool["coverage"], sort_keys=True) + "`.", "", policy["domain_rationale"], "",
        "The field compiled_num_warps is required by the reused Stage A selector but is explicitly a predicted source attribute here, not a claim of compilation. Actual Stage B artifacts must confirm it.", "",
        "M512/N128 retains B_DESC=65536: 8GiB BF16 input per case and 128KiB logical shared tile. Actual compiled shared allocation, register/spill, occupancy and allocation safety remain pending. No descriptor/grid/resource policy is adjusted.", "",
        "PRIMARY and SECONDARY will be assigned only by the future artifact gate; no predicted PRIMARY count or N64/w8 filter.", "",
        "Full future sampling, analysis, locked prediction coefficients and hypothesis support/falsification observations are in protocol.json. Stage C cannot confirm Stage D-generated hypotheses.", "",
        "NO Phase 5 timing observed. NO Phase 5 artifact gate executed. STOP until a separate artifact-gate authorization.", ""]
    return "\n".join(lines)


def main():
    rc.frozen_inventory()
    pool = create_pool(source_domain())
    stage_d = rc.read(rc.OUT / "results.json")
    policy = protocol(stage_d["hypotheses"], stage_d)
    sources = [Path(__file__), HERE / "validate_preregistration.py", HERE / "PROTOCOL.md",
        rc.BASE / "phase4/preregister.py", rc.BASE / "phase4/residual_contract.py",
        ROOT / "lib/Dialect/TritonGPU/Transforms/Coalesce.cpp", ROOT / "include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td",
        rc.OUT / "results.json"]
    binding = {"starting_HEAD": rc.BASELINE, "source_SHA256": {p.relative_to(ROOT).as_posix(): rc.sha(p.read_bytes()) for p in sources},
        "Stage_D_result_role": "OUTCOME_INFORMED_HYPOTHESIS_AND_LOCKED_MODELS_ONLY_NEVER_MEMBERSHIP_INPUT",
        "protected_prior_inventory_SHA256": rc.sha(rc.encode(rc.frozen_inventory())),
        "Phase5_layout_status": "PREDICTED_FROM_SOURCE_NOT_COMPILED", "Phase5_timing_observed": False}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in (("structural_pool.json", pool), ("exclusions.json", [r for r in pool["transitions"] if not r["included_in_structural_pool"]]),
                        ("protocol.json", policy), ("source_bindings.json", binding)):
        (OUT / name).write_bytes(rc.encode(value))
    (OUT / "cohort_summary.md").write_text(render(pool, policy))
    print(f'PHASE 5 PREREGISTERED: source=24, included={pool["included_structural_pool"]}, excluded={pool["excluded_structural_pool"]}; no artifact gate/timing')


if __name__ == "__main__":
    main()
