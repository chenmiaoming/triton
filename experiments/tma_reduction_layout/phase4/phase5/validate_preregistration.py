"""Independent held-out domain/layout/reason audit; no compilation or GPU imports."""
import ast
from collections import Counter
import copy
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4 import residual_contract as rc
from experiments.tma_reduction_layout.phase4 import preregister as stage_a
from experiments.tma_reduction_layout.phase4.phase5 import preregister as prediction

DOMAIN = {"M": [16, 256, 512], "N": [16, 32, 64, 128], "num_warps": [4, 8]}
REASONS = ("DEFAULT_ILLEGAL", "CAND4_ILLEGAL", "IDENTICAL_LAYOUT", "WARP_PART_CHANGED",
    "LANE_PART_NOT_REDUCED", "REDUCTION_AXIS_MISMATCH", "WARP_COUNT_MISMATCH", "UNSUPPORTED_LAYOUT")
FILES = {"structural_pool.json", "exclusions.json", "protocol.json", "cohort_summary.md", "source_bindings.json"}


def equal(a, b, label):
    rc.require(a == b, label)


def independent_candidate(m, n, warps, forced=None):
    """Closed form of the rank-three power-of-two [1,M,N] source construction."""
    vec = min(max(m * n // (32 * warps), 1), 8) if forced is None else forced
    legal = forced is None or (0 < vec <= 8 and vec <= max(m * n // (32 * warps), 1)
        and n % vec == 0 and vec & (vec - 1) == 0)
    layout = None
    if legal:
        lanes_n = min(32, n // vec)
        lanes_m = min(m, 32 // lanes_n)
        warps_n = min(warps, max((n // vec) // lanes_n, 1))
        warps_m = min(warps // warps_n, max(m // lanes_m, 1))
        layout = {"sizePerThread": [1, 1, vec],
            "threadsPerWarp": [32 // (lanes_m * lanes_n), lanes_m, lanes_n],
            "warpsPerCTA": [warps // (warps_m * warps_n), warps_m, warps_n], "order": [2, 1, 0]}
    return {"legal": bool(legal), "layout": layout, "compiled_num_warps": warps,
        "num_ctas": 1, "hardware_warp_size": 32, "shared_family": "nvmma_shared", "shared_rank": 3,
        "metadata_evidence": "ANALYTIC_EXPECTED_ATTRIBUTES_ONLY_NOT_COMPILED_METADATA", "requested_vec": vec,
        "vec": layout["sizePerThread"][2] if layout else None,
        "lanePart_M": layout["threadsPerWarp"][1] if layout else None,
        "warpPart_M": layout["warpsPerCTA"][1] if layout else None}


def supported(c, warps):
    v = c["layout"]
    return (v is not None and set(v) == {"sizePerThread", "threadsPerWarp", "warpsPerCTA", "order"}
        and all(len(v[k]) == 3 and all(type(x) is int and x > 0 and x & (x - 1) == 0 for x in v[k])
                for k in ("sizePerThread", "threadsPerWarp", "warpsPerCTA"))
        and v["order"] == [2, 1, 0] and math.prod(v["threadsPerWarp"]) == 32
        and math.prod(v["warpsPerCTA"]) == warps
        and all(v[k][0] == 1 for k in ("sizePerThread", "threadsPerWarp", "warpsPerCTA"))
        and c["num_ctas"] == 1 and c["hardware_warp_size"] == 32
        and c["shared_family"] == "nvmma_shared" and c["shared_rank"] == 3)


def independent_reasons(row):
    d, c, w = row["default"], row["cand4"], row["num_warps"]
    errors = set()
    if not d["legal"]: errors.add("DEFAULT_ILLEGAL")
    if not c["legal"]: errors.add("CAND4_ILLEGAL")
    valid = all(supported(v, w) for v in (d, c) if v["legal"])
    if not valid: errors.add("UNSUPPORTED_LAYOUT")
    if row["reduction_axis"] != 1: errors.add("REDUCTION_AXIS_MISMATCH")
    if any(v["legal"] and v["compiled_num_warps"] != w for v in (d, c)): errors.add("WARP_COUNT_MISMATCH")
    if d["legal"] and c["legal"] and valid:
        if d["layout"] == c["layout"]: errors.add("IDENTICAL_LAYOUT")
        if d["warpPart_M"] != c["warpPart_M"]: errors.add("WARP_PART_CHANGED")
        if d["lanePart_M"] <= c["lanePart_M"]: errors.add("LANE_PART_NOT_REDUCED")
    return [r for r in REASONS if r in errors]


def validate(pool, exclusions, policy, stage_d):
    equal(pool["phase"], "PHASE_5_HELD_OUT_CONFIRMATORY_EXTENSION_PREREGISTRATION", "new held-out statistical role")
    equal(pool["source_domain"], DOMAIN, "literal fixed domain")
    equal(pool["total_source_transitions"], 24, "24 source transitions")
    rows = pool["transitions"]
    expected = {(m, n, w) for m in DOMAIN["M"] for n in DOMAIN["N"] for w in DOMAIN["num_warps"]}
    equal(len(rows), 24, "source accounted once")
    equal({(r["M"], r["N"], r["num_warps"]) for r in rows}, expected, "complete fixed Cartesian domain")
    equal([r["config_id"] for r in rows], sorted(f"M{m}_N{n}_w{w}" for m, n, w in expected), "sorted unique identities")
    equal(tuple(stage_a.REASONS), REASONS, "unaltered Stage A reason family")
    for r in rows:
        m, n, w = r["M"], r["N"], r["num_warps"]
        equal(r["config_id"], f"M{m}_N{n}_w{w}", "identity")
        equal(r["logical_shape"], [1, m, n], "logical shape")
        equal(r["reduction_axis"], 1, "same M axis")
        equal(r["origin"], "HELD_OUT_EXTRAPOLATION", "no development anchors")
        equal(r["transition"], "default -> cand4", "fixed transition")
        for c, forced in (("default", None), ("cand4", 4)):
            equal(r[c], independent_candidate(m, n, w, forced), "independent source layout/legality " + r["config_id"] + c)
        reasons = independent_reasons(r)
        equal(r["reason_codes"], reasons, "all nonexclusive exclusions")
        equal(r["included_in_structural_pool"], not reasons, "eight Stage A structural criteria only")
        equal(r["layout_evidence"], "SOURCE_RULE_PREDICTION_NO_COMPILATION_OR_ACTUAL_TTGIR", "predicted layout label")
        equal(r["artifact_gate_status"], "STRUCTURAL_OR_RESOURCE_PENDING" if not reasons else "STRUCTURALLY_EXCLUDED", "no actual gate")
        equal(r["timing_eligible"], False, "no premature timing eligibility")
        resources = r["predicted_resource_geometry"]
        equal(resources["input_BF16_allocation_bytes"], 65536 * m * n * 2, "fixed descriptor input allocation")
        equal(resources["output_FP32_allocation_bytes"], 65536 * n * 4, "output allocation")
        equal(resources["logical_shared_tile_bytes"], m * n * 2, "logical shared tile, not compiled SMEM")
    included = [r for r in rows if r["included_in_structural_pool"]]
    equal(pool["included_structural_pool"], len(included), "included accounted")
    equal(pool["excluded_structural_pool"], 24 - len(included), "excluded accounted")
    equal(exclusions, [r for r in rows if not r["included_in_structural_pool"]], "complete machine-readable exclusions")
    coverage = {k: dict(sorted(Counter(str(r[k]) for r in included).items())) for k in ("M", "N", "num_warps")}
    coverage["lane_transition"] = dict(sorted(Counter(f'{r["default"]["lanePart_M"]}->{r["cand4"]["lanePart_M"]}' for r in included).items()))
    equal(pool["coverage"], coverage, "coverage recomputed")
    counts = Counter(e for r in rows for e in r["reason_codes"])
    equal(pool["reason_counts_nonexclusive"], {k: counts[k] for k in REASONS}, "reason counts")
    for k in ("no_compilation", "no_artifact_gate", "no_timing"): equal(pool[k], True, "offline-only preregistration")
    equal(policy["phase"], "PHASE_5_HELD_OUT_VALIDATION", "future phase")
    equal(policy["stage"], "PREREGISTRATION_ONLY_NO_EXECUTION", "stage")
    equal(policy["domain"], DOMAIN, "future domain")
    for k in ("execution_authorized", "Phase5_artifact_gate_executed"): equal(policy[k], False, "no execution")
    equal(policy["observed_Phase5_timing_samples"], 0, "unmeasured Phase 5")
    timing = policy["future_timing"]
    fixed = {"B_DESC": 65536, "B_RUN": [16384, 32768, 65536], "R": [0, 1], "canonical_R": None,
        "independent_invocations": 3, "rounds": 10, "scalar_samples_per_round": 10,
        "samples_per_condition": 100, "warmup_count_per_condition": 3}
    for k, v in fixed.items(): equal(timing[k], v, "fixed future timing " + k)
    rc.require("CUDA events" in timing["primitive"] and "SHA guard before every launch" in timing["primitive"], "exact frozen binary sampling")
    rc.require("STRUCTURAL_OR_RESOURCE_PENDING" in timing["resource_pending"] and "do not change B_DESC/B_RUN/R" in timing["resource_pending"], "no resource-driven protocol adaptation")
    equal(policy["hierarchy"]["minimum_PRIMARY_interpretability"], 5, "necessary primary minimum")
    rc.require("exact sequence" in policy["hierarchy"]["PRIMARY"] and "all frozen" in policy["hierarchy"]["PRIMARY"], "all exact body gates")
    rc.require("PIPELINED_OPCODE_EQUIVALENT" in policy["hierarchy"]["SECONDARY"], "pipelined remains secondary")
    rc.require("does not itself establish" in policy["hierarchy"]["coverage"], "no automatic generalization")
    equal(policy["hypotheses"], stage_d["hypotheses"], "outcome-informed hypotheses locked verbatim")
    equal(policy["locked_development_models"], {m: v["coefficients"] for m, v in stage_d["models"]["ALL_ELIGIBLE"].items()}, "prospective prediction coefficients locked")
    rc.require("cannot confirm" in policy["statistical_role"] and "no refitting" in policy["future_analysis"]["hypothesis_predictions"], "separate development/confirmation")
    rc.require(not any(k in pool for k in ("timing_results", "samples", "expected_PRIMARY", "speedup")), "no outcome selection or observed timings")


def historical_replay():
    rows = rc.read(stage_a.OUTPUT / "structural_pool.json")["transitions"]
    for r in rows:
        for c, forced in (("default", None), ("cand4", 4)):
            predicted = independent_candidate(r["M"], r["N"], r["num_warps"], forced)
            equal(predicted["legal"], r[c]["legal"], "source prediction agrees with historical legality")
            equal(predicted["layout"], r[c]["layout"], "source prediction agrees with frozen actual Stage A layout")
    return len(rows) * 2


class GuardedDimensions(dict):
    """Fail immediately on any input access outside the three structural keys."""
    def __getitem__(self, key):
        rc.require(key in {"M", "N", "num_warps"}, "Nonstructural membership input accessed: " + key)
        return super().__getitem__(key)

    def get(self, *args): raise AssertionError("Membership must use the three explicit structural keys")
    def __iter__(self): raise AssertionError("Membership cannot iterate nonstructural input")
    def keys(self): raise AssertionError("Membership cannot inspect arbitrary input fields")
    def values(self): raise AssertionError("Membership cannot inspect arbitrary input fields")
    def items(self): raise AssertionError("Membership cannot inspect arbitrary input fields")


def poison_tests(expected):
    domain = [{"M": m, "N": n, "num_warps": w} for m in DOMAIN["M"] for n in DOMAIN["N"] for w in DOMAIN["num_warps"]]
    for sign in (-1, 1):
        rows = [GuardedDimensions({**r, "samples": [sign * 1e99], "timing_us": sign * 1e99,
            "G": sign * 1e99, "D": -sign * 1e99, "g0": sign * 1e99, "speedup": sign * 1e99,
            "expected_PRIMARY": sign > 0, "equivalence": "EXACT_SEQUENCE_EQUIVALENT" if sign > 0 else "MISMATCH",
            "final_class": "PRIMARY" if sign > 0 else "EXCLUDE_FROM_TIMING", "eligible": sign > 0}) for r in reversed(domain)]
        equal(prediction.create_pool(rows), expected, "performance poison and guard leave complete structural pool unchanged")
    return "PASS_BOTH_EXTREME_SIGNS_AND_NONSTRUCTURAL_ACCESS_GUARD"


def file_audit(pool, policy, protected):
    equal({p.name for p in prediction.OUT.iterdir()}, FILES, "only preregistration outputs, no future timing/artifacts")
    equal({p.relative_to(prediction.OUT.parent).as_posix() for p in prediction.OUT.parent.rglob("*") if p.is_file()},
        {"preregistration/" + n for n in FILES} | ({"validation.json"} if (prediction.OUT.parent / "validation.json").exists() else set()),
        "no Phase 5 artifact gate or timing result exists")
    for extra in (rc.BASE / "phase5", rc.BASE / "results/phase5"):
        rc.require(not extra.exists(), "no parallel Phase 5 result/execution tree")
    binding = rc.read(prediction.OUT / "source_bindings.json")
    sources = [prediction.HERE / "preregister.py", prediction.HERE / "validate_preregistration.py", prediction.HERE / "PROTOCOL.md",
        rc.BASE / "phase4/preregister.py", rc.BASE / "phase4/residual_contract.py",
        ROOT / "lib/Dialect/TritonGPU/Transforms/Coalesce.cpp", ROOT / "include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td",
        rc.OUT / "results.json"]
    equal(binding, {"starting_HEAD": rc.BASELINE, "source_SHA256": {p.relative_to(ROOT).as_posix(): rc.sha(p.read_bytes()) for p in sources},
        "Stage_D_result_role": "OUTCOME_INFORMED_HYPOTHESIS_AND_LOCKED_MODELS_ONLY_NEVER_MEMBERSHIP_INPUT",
        "protected_prior_inventory_SHA256": rc.sha(rc.encode(protected)),
        "Phase5_layout_status": "PREDICTED_FROM_SOURCE_NOT_COMPILED", "Phase5_timing_observed": False}, "all source/development/protected hashes")
    equal((prediction.OUT / "cohort_summary.md").read_text(), prediction.render(pool, policy), "all 24 report rows bind validated pool")
    equal(policy, prediction.protocol(rc.read(rc.OUT / "results.json")["hypotheses"], rc.read(rc.OUT / "results.json")), "complete text/analysis protocol lock")
    tree = ast.parse((prediction.HERE / "preregister.py").read_text())
    for node in ast.walk(tree):
        modules = [v.name for v in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
        rc.require(all(n.split(".")[0] in {"collections", "json", "math", "pathlib", "sys", "experiments"} for n in modules), "no compiler/GPU/Modal import")
        if isinstance(node, ast.FunctionDef) and node.name in {"create_pool", "predicted_candidate", "source_domain"}:
            for call in (n for n in ast.walk(node) if isinstance(n, ast.Call)):
                rc.require(not isinstance(call.func, ast.Attribute) or call.func.attr not in {"read", "read_text", "read_bytes", "open", "run", "compile", "launch"}, "membership cannot consult artifacts/outcomes")


def main():
    protected = rc.frozen_inventory()
    pool, exclusions, policy = [rc.read(prediction.OUT / n) for n in ("structural_pool.json", "exclusions.json", "protocol.json")]
    development = rc.read(rc.OUT / "results.json")
    validate(pool, exclusions, policy, development)
    file_audit(pool, policy, protected)
    replayed = historical_replay()
    poison = poison_tests(pool)
    probes = [
        ("missing source", lambda p, e, q: p["transitions"].pop()),
        ("duplicated source", lambda p, e, q: p["transitions"].__setitem__(0, copy.deepcopy(p["transitions"][1]))),
        ("predicted layout", lambda p, e, q: p["transitions"][0]["default"]["layout"]["threadsPerWarp"].__setitem__(1, 999)),
        ("missing exclusion reason", lambda p, e, q: next(r for r in p["transitions"] if r["reason_codes"])["reason_codes"].clear()),
        ("premature timing", lambda p, e, q: q.__setitem__("observed_Phase5_timing_samples", 100)),
        ("adapted descriptor", lambda p, e, q: q["future_timing"].__setitem__("B_DESC", 8192)),
        ("unlocked model", lambda p, e, q: q["locked_development_models"]["C"].__setitem__("warp8", 999.)),
        ("retroactive hypothesis confirmation", lambda p, e, q: q["hypotheses"][0].__setitem__("status", "SUPPORTED"))]
    corruption = []
    for name, mutate in probes:
        bad = copy.deepcopy((pool, exclusions, policy)); mutate(*bad)
        try:
            validate(*bad, development)
        except (ValueError, AssertionError, KeyError):
            corruption.append({"probe": name, "status": "PASS_CORRUPTION_REJECTED"})
        else:
            raise AssertionError("Corruption accepted: " + name)
    (prediction.OUT.parent / "validation.json").write_bytes(rc.encode({"status": "PASS", "source_transitions": 24,
        "included": pool["included_structural_pool"], "excluded": pool["excluded_structural_pool"],
        "independent_method": "Closed-form blocked layouts and independent A-H exclusions; frozen historical layout replay",
        "historical_candidate_layouts_checked": replayed, "performance_poison": poison,
        "corruption_probes": corruption, "protected_prior_files": len(protected),
        "Phase5_timing_samples": 0, "Phase5_artifact_gate_executed": False, "outcome_acceptance_requirements": []}))
    print("PHASE 5 PREREGISTRATION VALIDATOR PASS; 24 source transitions, independent layouts/reasons, guarded performance poison; no gate/timing")


if __name__ == "__main__":
    main()
