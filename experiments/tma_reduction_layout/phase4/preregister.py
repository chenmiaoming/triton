"""Phase 4 Stage A: performance-field-independent programmatic cohort selection. No GPU or compiler imports.

Selection sees a structural projection of the mixed historical sweep JSON.
Resource/artifact eligibility is a separate layer and cannot change membership.
"""
import ast
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASE = ROOT / "experiments/tma_reduction_layout"
OUTPUT = BASE / "results/phase4/preregistration"
BASELINE = "ce04afd7d9095349d54a007e61ec2b5d4fd557b6"
ORIGINAL_STAGE_A = "e37c5ff70d4993434eb781ff64817e8c509422a7"
# Recorded from the original commit before amendment; validation survives Git GC
# without requiring the superseded, unpushed commit object to remain reachable.
ORIGINAL_STAGE_A_SHA256 = {
    "structural_pool.json": "2252d3d98b6013b0fdcdf88e1cf6cc46f2a9f342f79da9f2e145ac803bfa6f53",
    "exclusions.json": "5303c7e8d4e3e470bc1eef2cd870b4c196db48de15a3e14aa571f3e7c38ef915",
    "rotation_schedule.json": "3ef7d10b23b4861bc428560c1b2058c6901eaf98080866a0b99f0b4dc23f3913",
}
SWEEP = BASE / "results/phase2/sweep/results.json"
FIELDS = ("sizePerThread", "threadsPerWarp", "warpsPerCTA", "order")
ANCHORS = ("M32_N64_w8", "M32_N128_w4")
REASONS = ("DEFAULT_ILLEGAL", "CAND4_ILLEGAL", "IDENTICAL_LAYOUT",
           "WARP_PART_CHANGED", "LANE_PART_NOT_REDUCED", "REDUCTION_AXIS_MISMATCH",
           "WARP_COUNT_MISMATCH", "UNSUPPORTED_LAYOUT")
OUTPUT_NAMES = ("structural_pool.json", "exclusions.json", "canonical_eligibility.json",
                "reproduction_plan.json", "rotation_schedule.json", "protocol.json",
                "cohort_summary.md", "protocol.md", "source_bindings.json")


def read_json(path):
    return json.loads(path.read_text())


def dump(data):
    return json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"


def kernel_contract():
    """Read only source AST constants and the sweep kernel's logical shape/axis."""
    tree = ast.parse((BASE / "phase2_benchmark.py").read_text())
    sweep = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.name == "run_sweep_remote")
    kernel = next(n for n in ast.walk(sweep) if isinstance(n, ast.FunctionDef)
                  and n.name == "kernel")
    maxima = [n for n in ast.walk(kernel) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "max"]
    if len(maxima) != 1:
        raise ValueError("Expected one max reduction in sweep source")
    axis = next(ast.literal_eval(k.value) for k in maxima[0].keywords if k.arg == "axis")
    blocks = [k.value for n in ast.walk(kernel) if isinstance(n, ast.Call)
              for k in n.keywords if k.arg == "block_shape"]
    if axis != 1 or len(blocks) != 1 or ast.unparse(blocks[0]) != "[1, M, N]":
        raise ValueError("Sweep source logical [1,M,N]/M-axis contract changed")
    domains = {}
    for n in ast.walk(sweep):
        if isinstance(n, ast.Assign):
            for target in n.targets:
                if isinstance(target, ast.Name) and target.id in ("M_LIST", "N_LIST", "WARPS_LIST"):
                    domains[target.id] = ast.literal_eval(n.value)
    return {"logical_shape": [1, "M", "N"], "reduction_axis": axis,
            "reduction_axis_evidence": "source AST; historical parsed reduce_ops are empty; actual TTGIR required by Stage B",
            "source_function": "run_sweep_remote.get_kernel.kernel", "domains": domains}


def structural_projection(source, contract):
    """Explicit key whitelist. Never reads measured/raw samples/derived transitions.

    Resources and recorded SHA fields are not exposed to the selector either.
    """
    rows = []
    combinations = 0
    configs = source["configs"]
    for config_id in sorted(configs):
        cfg = configs[config_id]
        m, n, warps = cfg["M"], cfg["N"], cfg["num_warps"]
        if any(type(x) is not int or x <= 0 for x in (m, n, warps)):
            raise ValueError("Malformed source configuration dimensions")
        if config_id != f"M{m}_N{n}_w{warps}":
            raise ValueError("Configuration identity mismatch")
        candidates = cfg["candidates"]
        if set(candidates) != {"default", "8", "4", "2", "1"}:
            raise ValueError("Historical five-candidate domain changed")
        combinations += len(candidates)
        row = dict(config_id=config_id, M=m, N=n, num_warps=warps,
                   transition="default -> cand4", logical_shape=[1, m, n],
                   reduction_axis=contract["reduction_axis"],
                   origin="ANCHOR_EXISTING" if config_id in ANCHORS else "NEW_GENERALIZATION")
        for name, key in (("default", "default"), ("cand4", "4")):
            candidate = candidates[key]
            legal = candidate["is_legal"]
            if type(legal) is not bool:
                raise ValueError("Legality must be boolean")
            observed = candidate.get("observed", {}).get("ttgir") if legal else None
            observed = observed if isinstance(observed, dict) else {}
            blocked = observed.get("blocked_encoding")
            layout = {f: blocked.get(f) for f in FIELDS} if isinstance(blocked, dict) else None
            attrs = observed.get("module_attributes")
            shared = observed.get("shared_encoding")
            attrs = attrs if isinstance(attrs, dict) else {}
            shared = shared if isinstance(shared, dict) else {}
            row[name] = dict(legal=legal, layout=layout,
                             compiled_num_warps=attrs.get("num_warps"),
                             num_ctas=attrs.get("num_ctas"),
                             hardware_warp_size=attrs.get("threads_per_warp"),
                             shared_family=shared.get("family"), shared_rank=shared.get("rank"))
        rows.append(row)
    domains = contract["domains"]
    expected = {f"M{m}_N{n}_w{w}" for m in domains["M_LIST"]
                for n in domains["N_LIST"] for w in domains["WARPS_LIST"]}
    if set(configs) != expected:
        raise ValueError("Source sweep does not account for its entire declared Cartesian domain")
    return rows, combinations


def valid_layout(candidate, warps):
    layout = candidate["layout"]
    if not isinstance(layout, dict) or set(layout) != set(FIELDS):
        return False
    for field in FIELDS:
        values = layout[field]
        if not isinstance(values, list) or len(values) != 3 or any(type(x) is not int for x in values):
            return False
        if field != "order" and any(x <= 0 or x & (x-1) for x in values):
            return False
    return (sorted(layout["order"]) == [0, 1, 2]
            and math.prod(layout["threadsPerWarp"]) == 32
            and math.prod(layout["warpsPerCTA"]) == warps
            and all(layout[f][0] == 1 for f in FIELDS[:3])
            and candidate["num_ctas"] == 1 and candidate["hardware_warp_size"] == 32
            and candidate["shared_family"] == "nvmma_shared" and candidate["shared_rank"] == 3)


def select(rows):
    """Pure logical eligibility A-H; no artifact/resource or performance inputs."""
    result = []
    for source_row in rows:
        row = dict(source_row)
        default, cand4 = row["default"], row["cand4"]
        errors = []
        if not default["legal"]:
            errors.append("DEFAULT_ILLEGAL")
        if not cand4["legal"]:
            errors.append("CAND4_ILLEGAL")
        supported = all(valid_layout(c, row["num_warps"]) for c in (default, cand4) if c["legal"])
        if not supported:
            errors.append("UNSUPPORTED_LAYOUT")
        if row["reduction_axis"] != 1:
            errors.append("REDUCTION_AXIS_MISMATCH")
        if any(c["legal"] and c["compiled_num_warps"] != row["num_warps"] for c in (default, cand4)):
            errors.append("WARP_COUNT_MISMATCH")
        if default["legal"] and cand4["legal"] and supported:
            d, c = default["layout"], cand4["layout"]
            if d == c:
                errors.append("IDENTICAL_LAYOUT")
            if d["warpsPerCTA"][1] != c["warpsPerCTA"][1]:
                errors.append("WARP_PART_CHANGED")
            if d["threadsPerWarp"][1] <= c["threadsPerWarp"][1]:
                errors.append("LANE_PART_NOT_REDUCED")
        row["included_in_structural_pool"] = not errors
        row["reason_codes"] = [r for r in REASONS if r in errors]
        for name in ("default", "cand4"):
            layout = row[name]["layout"]
            def component(field, index):
                values=layout.get(field) if isinstance(layout, dict) else None
                return values[index] if isinstance(values, list) and len(values)==3 else None
            row[name] = {**row[name], "vec": component("sizePerThread",2),
                         "lanePart_M": component("threadsPerWarp",1),
                         "warpPart_M": component("warpsPerCTA",1)}
        result.append(row)
    return result


def coverage(rows):
    def counts(values):
        return dict(sorted(Counter(map(str, values)).items()))
    included = [r for r in rows if r["included_in_structural_pool"]]
    result = {name: counts(r[name] for r in included) for name in ("M", "N", "num_warps", "origin")}
    result["lanePart_M_transitions"] = counts(f"{r['default']['lanePart_M']}->{r['cand4']['lanePart_M']}" for r in included)
    result["warpPart_M_values"] = counts(r["default"]["warpPart_M"] for r in included)
    result["sizePerThread_transitions"] = counts(f"{r['default']['layout']['sizePerThread']}->{r['cand4']['layout']['sizePerThread']}" for r in included)
    result["N256_audit"] = [{"config_id": r["config_id"], "reason_codes": r["reason_codes"]}
                            for r in rows if r["N"] == 256]
    return result


def create_pool(source):
    contract = kernel_contract()
    projected, combinations = structural_projection(source, contract)
    rows = select(projected)
    reasons = Counter(r for row in rows for r in row["reason_codes"])
    return {"stage": "PHASE_4_STAGE_A_STRUCTURAL_POOL_FROZEN", "source_git_head": BASELINE,
            "source_path": str(SWEEP.relative_to(ROOT)), "unit": "(M,N,num_warps,default->cand4)",
            "source_combinations": combinations, "total_source_transitions": len(rows),
            "included_structural_pool": sum(r["included_in_structural_pool"] for r in rows),
            "excluded_structural_pool": sum(not r["included_in_structural_pool"] for r in rows),
            "reason_counts_nonexclusive": {r: reasons[r] for r in REASONS},
            "logical_contract": contract, "coverage": coverage(rows), "transitions": rows}


def canonical_inventory(pool):
    """Audit existing artifacts separately. Missing artifacts remain PENDING, not exclusions."""
    from experiments.tma_reduction_layout.gluon import artifact_checks as ac
    from experiments.tma_reduction_layout.analyze_ir import analyze_ttgir
    entries = []
    for row in pool["transitions"]:
        config = row["config_id"]
        item = {"config_id": config, "included_in_structural_pool": row["included_in_structural_pool"],
                "canonical_tier": "FROZEN_PHASE3_FIXED_BINARY_REFERENCE" if config in ANCHORS else "PHASE2_EXPLORATORY_REFERENCE_ONLY",
                "candidates": {}, "pre_timing_eligible": False,
                "gate_status": "PENDING_STAGE_B_NO_NEW_COMPILE_OR_TIMING_AUTHORIZED",
                "pending_reason_codes": ["MISSING_PHASE4_EXACT_BINARIES", "MISSING_PHASE4_EXACT_BINARY_OCCUPANCY", "MISSING_PHASE4_COMPLETE_PROVENANCE"]}
        root = BASE / "results/phase3/fixed_binary_artifacts/canonical" / config if config in ANCHORS else BASE / "results/phase2/representatives" / config
        for label, name in (("default", "default"), ("cand4", "4")):
            existing = {ext: root / f"{name}.{ext}" for ext in ac.EXTENSIONS}
            present = all(p.exists() and p.stat().st_size for p in existing.values())
            text_present = all(existing[ext].exists() and existing[ext].stat().st_size for ext in ("ptx","ttgir","sass","resource.txt"))
            candidate = {"reference_artifacts_complete": bool(present),
                         "reference_paths": {ext: str(path.relative_to(ROOT)) for ext,path in existing.items() if path.exists()},
                         "expected_canonical_localload_opcodes": None,
                         "observed_gluon_localload_opcodes": None,
                         "historical_reproduction_classification": None,
                         "historical_repeated_classification": None,
                         "resource_text": None, "dynamic_shared_bytes": None,
                         "exact_measured_cubin_occupancy_closed": False}
            if text_present:
                parsed = ac.parse_resource(existing["resource.txt"].read_text())
                candidate["resource_text"] = parsed
                candidate["archived_zero_spill"] = parsed["local_bytes"] == parsed["stack_bytes"] == 0
                ir = analyze_ttgir(existing["ttgir"].read_text())
                blocked = ir["blocked_encodings"][ir["local_load_dest_layout"]]
                if {f:blocked[f] for f in FIELDS} != row[label]["layout"]:
                    raise ValueError(f"Archived reference layout disagrees with sweep: {config}/{label}")
                candidate["reference_layout_matches_sweep"] = True
                candidate["expected_canonical_localload_opcodes"] = [i["opcode"] for i in ac.initial_load_signature(existing["ptx"].read_text())]
                # The source sweep records structural resource metadata; samples are not read.
                source_candidate = read_json(SWEEP)["configs"][config]["candidates"][name]
                candidate["dynamic_shared_bytes"] = source_candidate["observed"]["physical_resources"]["triton_launch_shared_bytes"]
                if config in ANCHORS:
                    c_path = BASE / "results/phase3/gluon_reproduction/artifacts" / config / f"{name}.ptx"
                    candidate["observed_gluon_localload_opcodes"] = [i["opcode"] for i in ac.initial_load_signature(c_path.read_text())]
                    from experiments.tma_reduction_layout.phase4.artifact_gate import body_fingerprint, equivalence
                    canonical_fp = body_fingerprint(existing["ptx"].read_text())
                    if canonical_fp != ac.canonical_fingerprint(BASE,config,name):
                        raise ValueError("Archival source-stage extraction disagrees with audited canonical phases")
                    candidate["historical_reproduction_classification"] = equivalence(canonical_fp, body_fingerprint(c_path.read_text(),minimal_output=True))
                    from experiments.tma_reduction_layout.phase4.artifact_gate import repeated
                    d_root = BASE/"results/phase3/gluon_repeated/artifacts"/config
                    d_bundle = {ext:(d_root/f"{name}.{ext}").read_text() for ext in ("ptx","ttgir","sass","resource.txt")}
                    canonical = {ext:existing[ext].read_text() for ext in ("ptx","ttgir","resource.txt")}
                    d_audit = repeated(canonical,d_bundle,row["logical_shape"],row["num_warps"],(BASE/"gluon/kernel_repeated.py").read_text())
                    candidate["historical_repeated_classification"] = d_audit["classification"]
                    candidate["canonical_local_load_match"] = d_audit["canonical_local_load_match"]
                    candidate["historical_repeated_new_gate_checks"] = d_audit["checks"]
                    candidate["historical_repeated_satisfies_new_gate"] = d_audit["structure_pass"]
                    candidate["observed_repeated_localload_opcodes"] = [i["opcode"] for i in ac.initial_load_signature(d_bundle["ptx"])]
                    candidate["historical_repeated_copy_audit"] = {key:d_audit[key] for key in ("copy_count","copy_opcode_sequence","copy_operand_sequence","copy_pairs","all_loop_movs","sass_audit","indirect_barrier_effect")}
                candidate["artifact_sha256"] = {ext: hashlib.sha256(path.read_bytes()).hexdigest() for ext,path in existing.items() if path.exists()}
            item["candidates"][label] = candidate
        entries.append(item)
    return {"layer": "CANONICAL_ARTIFACT_RESOURCE_REFERENCE_AUDIT_ONLY", "pre_timing_eligible_count": 0,
            "resource_policy": "LOCAL=STACK=0; primary candidate blocks/SM and active warps/SM equal; exact timing CUBIN occupancy closure required",
            "entries": entries}


def protocol():
    return {"version": 2,
            "selection_blinding": "performance-field-independent programmatic cohort selection, frozen before any new Phase 4 mechanism measurements; historical canonical outcomes existed before this preregistration, so human-level complete outcome blinding cannot be claimed",
            "canonical_single_gate": "CANONICAL_SINGLE_REPRODUCTION_GATE: exact blocked/NVMMA shared layouts and entire initial LocalLoad opcode sequence; complete reduction fingerprint exact or classified secondary; zero spill and resource provenance",
            "repeated_gate": "REPEATED_HARNESS_ISOLATION_GATE: one TMA/LocalLoad before one unspecialized R loop; no tile reload, extra memory/global effect or accumulator inside; complete reduction fingerprint preserved, zero spill, matched residency, R=0 present and one binary/PTX/LocalLoad sequence across runtime R/B",
            "repeated_local_load": "record EXACT/DIFFERENT_ENCODING/MISMATCH; DIFFERENT_ENCODING is not automatic exclusion when TTGIR semantics and full byte payload agree, load remains one-time/outside loop, R=0 and fixed binary exist and all isolation/resource gates pass",
            "R0_limitation": "R=0 subtraction removes fixed one-time harness differentials, but does not prove absence of interaction through register allocation, live ranges, or scheduling",
            "harness_description": "canonical reduction-body-equivalent controlled harness", "stage": "A_ONLY_NO_EXECUTION", "cohort_rule": "legal, distinct supported layouts; same M reduction axis and warps; warpPart_M unchanged; lanePart_M strictly reduced",
            "fingerprint_families": ["max.*", "cvt.*", "shfl.*", "st.shared*", "ld.shared*", "ldmatrix*", "bar.sync", "selp.*"],
            "fingerprint_stage": "canonical convert/max source stage including terminal exchange, excluding store-layout conversion; Gluon minimal-output complete region from after tile load to global store; repeated complete runtime loop. Future canonical source-stage lines must be SHA-bound and AST-verified. Missing/ambiguous stage => pending, never choose a subsequence by desired match",
            "fingerprint_normalization": "retain shuffle/barrier immediates; remove registers, other addresses and predicates; complete filtered sequence only, no dataflow/full-PTX/SASS/CUBIN equality",
            "primary_classification": "CANONICAL_SINGLE_EXACT_LAYOUT_AND_LOCALLOAD; BOTH_CANDIDATES_SINGLE_AND_RUNTIME_LOOP_EXACT_SEQUENCE_EQUIVALENT; ALL_ISOLATION_RESOURCE_RUNTIME_GATES_PASS; REPEATED_PRELOOP_ENCODING_NEED_NOT_EQUAL_CANONICAL",
            "secondary_classification": "NO_MISMATCH_AND_AT_LEAST_ONE_PIPELINED_OPCODE_EQUIVALENT_AND_ALL_PRE_GATES_PASS; opcode multiset is not topology proof",
            "unsupported_topology_class": "TOPOLOGY_EQUIVALENT_NONEXACT is not assigned by automatic multiset comparison; remains pending/excluded absent separately preregistered dependency/address proof",
            "exclude_classification": ["REDUCTION_FINGERPRINT_MISMATCH", "CANONICAL_SINGLE_LOCALLOAD_SIGNATURE_MISMATCH", "REPEATED_LOCALLOAD_SEMANTIC_MISMATCH", "RUNTIME_BINARY_OR_LOCALLOAD_CHANGED", "MISSING_R0", "LAYOUT_MISMATCH", "SPILLS", "RESIDENCY_MISMATCH"],
            "pending_policy": "Missing evidence is PENDING, not an experimental counterexample or structural-pool removal; retain all failures and reasons",
            "future_sampling": {"B_DESC": 65536, "B_RUN": [16384,32768,65536], "R": [0,1],
                                "rounds": 10, "samples_per_round": 10, "samples_per_condition": 100, "invocations": 3,
                                "binary_scope": "one exact compiled binary per case/candidate/harness across R and B_RUN; canonical and Gluon are separate binaries",
                                "replication_label": "same UUID => same-device temporal replication; different UUIDs must be reported explicitly"},
            "metric_definitions": {"G_canonical": "OLS_B(sample_median_canonical_default)*1000 - OLS_B(sample_median_canonical_cand4)*1000",
                                   "delta_g1": "[b_default(1)-b_cand4(1)]-[b_default(0)-b_cand4(0)]",
                                   "units": "ns/additional CTA (marginal grid-time slope, not single-CTA latency)",
                                   "aggregation": "paired differential per invocation, then arithmetic mean; sample SD across three invocations"},
            "sign_resolution": {"machine_label": "SIGN_UNRESOLVED", "display_alias": "NEAR_ZERO",
                          "band": "two-sided 95% Student-t sign-resolution band, df=2; sample SD uses ddof=1","recommended": "uncertainty-aware absolute signed-difference interval, separately for G_canonical and delta_g1",
                          "threshold": "h = 4.302652729911275 * sample_SD(three paired invocation differentials) / sqrt(3)",
                          "positive": "mean > h", "negative": "mean < -h", "sign_unresolved": "-h <= mean <= h",
                          "interpretation": "SIGN_UNRESOLVED does not mean practical equivalence; sign is unresolved at chosen temporal-replication precision; descriptive t(df=2) interval assumptions and quantization limits must be reported",
                          "relative_alternative": "3% is an existing Phase 2 fixed-B median repeat-trigger, not established universal practical significance for slopes; do not use it as the primary category threshold",
                          "absolute_alternative": "fixed ns/CTA floor has no prior calibration; an uncertainty-scaled absolute half-width avoids inventing one",
                          "precedent": "phase2_benchmark.py run_sweep_remote: abs(diff_pct)>3.0 triggers repeat, diff_pct=(candidate median-default median)/default median*100"},
            "association": {"primary": "Spearman = Pearson of one-based average ranks of continuous unrounded mean delta_g1 and mean G_canonical among all future PRIMARY cases; exact ties receive average ranks; no sign-category discretization or numerical zeroing before ranking",
                            "secondary": "Pearson of the same unrounded means; separate SECONDARY cohort results; PRIMARY newly-generalized-only sensitivity",
                            "anchors": "included in primary if gates pass, labeled separately; no automatic anchor deletion",
                            "minimum": "fewer than 3 pairs or constant variable => UNDEFINED, no threshold/cohort changes",
                            "inference": "structured factorial design, not an IID random population sample; descriptive association only, no correlation p-values as population-level random-sample inference or additive causal evidence",
                            "leave_one_out": "PRIMARY only: report every case_i/rho_minus_i using the same continuous ranks; report min/max/median of defined values, defined/undefined counts; no deletion or refitting of the full-cohort analysis",
                            "minimum_interpretability": "PRIMARY n<5: report per-case results and association metrics but do not claim cross-shape generalization established; n>=5 remains descriptive evidence over this defined cohort, not population inference; no required rho threshold",
                            "sign_agreement": "three-category exact agreement / all eligible pairs; additionally report resolved-sign-only rate with numerator, denominator and unresolved coverage",
                            "sign_unresolved_and_negative": "retain unrounded numeric values in correlations and all cases in taxonomy; never replace near-zero values with zero"},
            "counterexamples": {"POSITIVE/POSITIVE": "A_MECHANISM_ALIGNED_POSITIVE", "SIGN_UNRESOLVED/SIGN_UNRESOLVED": "B_MECHANISM_NEUTRAL_UNRESOLVED",
                                "POSITIVE/SIGN_UNRESOLVED": "C_MASKED_LOCAL_ADVANTAGE_CANDIDATE", "POSITIVE/NEGATIVE": "D_OPPOSED_CANONICAL_OUTCOME",
                                "SIGN_UNRESOLVED/POSITIVE": "E_OTHER_MECHANISM_CANDIDATE", "NEGATIVE/NEGATIVE": "MECHANISM_ALIGNED_NEGATIVE",
                                "NEGATIVE/POSITIVE": "OPPOSED_CANONICAL_POSITIVE", "NEGATIVE/SIGN_UNRESOLVED": "MASKED_LOCAL_DISADVANTAGE_CANDIDATE",
                                "SIGN_UNRESOLVED/NEGATIVE": "CANONICAL_NEGATIVE_WITH_UNRESOLVED_BODY"},
            "counterexample_axis_order": "delta_g1 category / G_canonical category; interpretations are candidates, not causal diagnoses",
            "outliers": {"score": "abs(average_rank(delta_g1)-average_rank(G_canonical))/(n-1)", "report": "rank every case; discuss top min(5,n), descending score with config_id ascending for ties; label anchors",
                         "handling": "no removal, refitting after deletion, or post-result definition change"},
            "scope": "lane-partition-reduction class with unchanged warpPart_M; does not generalize to all partial-reduction layouts or verify H2b/H2c",
            "stop": "Stage A review only; no Modal, H100 compile, performance collection or push"}


def reproduction_plan(pool):
    return {"stage": "B_REQUIRES_SEPARATE_AUTHORIZATION", "cases": [
        {"config_id": r["config_id"], "origin": r["origin"], "logical_shape": r["logical_shape"],
         "num_warps": r["num_warps"], "default_layout": r["default"]["layout"], "cand4_layout": r["cand4"]["layout"],
         "canonical_artifact": "refresh/complete exact canonical reference with actual CUBIN, stage bounds and resource/provenance archive; do not reuse exploratory fixed-B results",
         "gluon_reproduction": "native TMA + NVMMA shared + blocked layout + one smem.load + float32 max(axis=1); explicit full LocalLoad and complete body fingerprint audit",
         "repeated_harness": "canonical reduction-body-equivalent controlled harness; record pre-loop LocalLoad EXACT/DIFFERENT_ENCODING/MISMATCH separately from canonical single exactness; one fixed binary across R/B with R=0; TMA/LocalLoad once before one runtime R loop; unspecialized R, no tile reload/global effects/accumulator inside, terminal reduction exchanges inside; tied empty asm and sink audited observationally",
         "status": "PENDING_NO_COMPILE_EXECUTED"} for r in pool["transitions"] if r["included_in_structural_pool"]]}


def rotation_schedule(pool):
    """Freeze the complete master condition order now. No observations are collected."""
    conditions = []
    for row in pool["transitions"]:
        if not row["included_in_structural_pool"]:
            continue
        for harness in ("canonical", "repeated"):
            for candidate in ("default", "4"):
                for r in ([None] if harness=="canonical" else [0,1]):
                    for b in (16384,32768,65536):
                        conditions.append(dict(config_id=row["config_id"], harness=harness, candidate=candidate, R=r, B_RUN=b))
    identifiers = [f"{c['config_id']}:{c['harness']}:{c['candidate']}:R{c['R']}:B{c['B_RUN']}" for c in conditions]
    runs = []
    for invocation in range(1,4):
        rounds = []
        for round_id in range(1,11):
            shift = ((invocation-1)*97+(round_id-1)*37) % len(identifiers)
            order = identifiers[shift:]+identifiers[:shift]
            if invocation % 2 == 0:
                order = list(reversed(order))
            rounds.append(dict(round=round_id, shift=shift, order=order, samples_per_visit=10))
        runs.append(dict(invocation=invocation, rounds=rounds))
    return {"execution": "FUTURE_ONLY_NO_SAMPLES", "master_conditions": dict(zip(identifiers,conditions)),
            "rule": "case/candidate/harness/R/B master list sorted by frozen pool; shift=(invocation-1)*97+(round-1)*37 mod condition_count; reverse invocation 2",
            "gate_filter": "only structural/resource gates may filter; preserve master relative order and archive both master and executed schedule; no outcome-dependent retry/removal",
            "invocations": runs}


def schedule_accounting(schedule):
    """Count future plan entries from the serialized schema, never observations."""
    master=schedule["master_conditions"]
    n_invocations=len(schedule["invocations"])
    by_harness={}
    for harness in ("canonical","repeated"):
        conditions=[value for value in master.values() if value["harness"]==harness]
        visits=sum(master[identifier]["harness"]==harness for invocation in schedule["invocations"]
                   for round_ in invocation["rounds"] for identifier in round_["order"])
        samples=sum(round_["samples_per_visit"] for invocation in schedule["invocations"]
                    for round_ in invocation["rounds"] for identifier in round_["order"]
                    if master[identifier]["harness"]==harness)
        by_harness[harness]={"cases":len({c["config_id"] for c in conditions}),
                             "candidates":len({c["candidate"] for c in conditions}),
                             "R_values":sorted({c["R"] for c in conditions},key=str),
                             "B_RUN_values":sorted({c["B_RUN"] for c in conditions}),
                             "measurement_conditions_per_invocation":len(conditions),
                             "candidate_level_invocation_conditions":len(conditions)*n_invocations,
                             "round_order_visits":visits,"planned_scalar_timing_samples":samples}
    return {"master_source_domain":"structurally included transitions before artifact/resource filtering; all source transitions retained in structural accounting",
            "master_case_count":len({value["config_id"] for value in master.values()}),
            "invocations":n_invocations,
            "rounds_per_invocation":[len(invocation["rounds"]) for invocation in schedule["invocations"]],
            "samples_per_visit":sorted({round_["samples_per_visit"] for invocation in schedule["invocations"] for round_ in invocation["rounds"]}),
            "stored_candidate_paired_scheduling_units":0,"by_harness":by_harness,
            "total_measurement_conditions_per_invocation":len(master),
            "total_candidate_level_invocation_conditions":len(master)*n_invocations,
            "total_round_order_visits":sum(value["round_order_visits"] for value in by_harness.values()),
            "total_planned_scalar_timing_samples":sum(value["planned_scalar_timing_samples"] for value in by_harness.values()),
            "execution":"FUTURE_PLAN_COUNTS_ONLY_NO_MEASUREMENTS"}


def render_summary(pool, inventory):
    lines = ["# Phase 4 Stage A — Structural cohort freeze", "",
             f"Source: {pool['total_source_transitions']} transitions / {pool['source_combinations']} historical combinations; structural pool: {pool['included_structural_pool']} included, {pool['excluded_structural_pool']} excluded.",
             "Performance-field-independent programmatic cohort selection, frozen before any new Phase 4 mechanism measurements. Membership uses legality and TTGIR layout metadata; resource/reproduction readiness is separate. Historical canonical outcomes existed before this preregistration, so human-level complete outcome blinding cannot be claimed.", "",
             "Layouts use SPT / TPW / WPC / order, each in [B,M,N] axes. Vec is SPT[N].", "",
             "| Case | M | N | Warps | Default layout | Cand4 layout | lanePart M | warpPart M | Include | Reasons | Origin |",
             "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |"]
    def layout(c):
        return "/".join(str(c["layout"][f]).replace(" ","") for f in FIELDS) if c["layout"] else "illegal / unavailable"
    for row in pool["transitions"]:
        d,c = row["default"],row["cand4"]
        lines.append(f"| {row['config_id']} | {row['M']} | {row['N']} | {row['num_warps']} | {layout(d)} | {layout(c)} | {d['lanePart_M']}→{c['lanePart_M']} | {d['warpPart_M']}→{c['warpPart_M']} | {'YES' if row['included_in_structural_pool'] else 'NO'} | {', '.join(row['reason_codes']) or 'PASS'} | {row['origin']} |")
    lines += ["", "## Coverage (included pool)", "", "```json", dump(pool["coverage"]).rstrip(), "```", "",
              "Exclusion reason counts are nonexclusive; a case can satisfy more than one exclusion.", "", "```json", dump(pool["reason_counts_nonexclusive"]).rstrip(), "```", "",
              "All six N256 cases change warpPart M and do not reduce lanePart M; this is a structural exclusion, not performance filtering.",
              "The two existing anchors are retained. New generalization cases are reported separately in future association sensitivity analyses.",
              f"Current pre-timing eligible count: {inventory['pre_timing_eligible_count']}. Actual Phase 4 CUBIN/occupancy/provenance closure is absent; pending cases remain in the frozen pool.",
              "Historical sweep reduce_ops metadata is empty. Source AST supplies the initial [1,M,N]/M-axis contract; Stage B must check actual TTGIR per artifact. No missing historical metadata is invented.",
              "Stop after Stage A review. No timing, H100/Modal work, or push."]
    return "\n".join(lines)+"\n"


def source_bindings(inventory):
    paths = {SWEEP, BASE/"phase2_benchmark.py", BASE/"analyze_ir.py", BASE/"phase3_audited_annotations.json",
             BASE/"audited_phase_annotations.json", BASE/"gluon/artifact_checks.py", BASE/"gluon/kernel_repeated.py", BASE/"gluon/kernel.py"}
    for entry in inventory["entries"]:
        for c in entry["candidates"].values():
            paths.update(ROOT/path for path in c["reference_paths"].values())
        if entry["config_id"] in ANCHORS:
            for name in ("default","4"):
                for ext in ("ptx","ttgir","sass","resource.txt","cubin.sha256"):
                    paths.add(BASE/"results/phase3/gluon_reproduction/artifacts"/entry["config_id"]/f"{name}.{ext}")
                    paths.add(BASE/"results/phase3/gluon_repeated/artifacts"/entry["config_id"]/f"{name}.{ext}")
    paths.update(BASE/"phase4"/name for name in ("preregister.py","artifact_gate.py","analysis_contract.py","validate_preregistration.py","README.md","PROTOCOL.md"))
    return {"source_git_head": BASELINE, "original_stage_a_commit": ORIGINAL_STAGE_A,
            "original_stage_a_preserved_SHA256": ORIGINAL_STAGE_A_SHA256,
            "scope": "entire source bytes bound; selector reads structural whitelist only",
            "files": [{"path":str(p.relative_to(ROOT)),"SHA256":hashlib.sha256(p.read_bytes()).hexdigest(),
                       "git_head":BASELINE if "/phase4/" not in str(p) else None,
                       "role":"PREREGISTRATION_CODE_OR_PROTOCOL" if "/phase4/" in str(p) else "ARCHIVED_STRUCTURAL_SOURCE"}
                      for p in sorted(paths)]}


def generate():
    pool = create_pool(read_json(SWEEP))
    inventory = canonical_inventory(pool)
    protocol_data = protocol()
    schedule = rotation_schedule(pool)
    protocol_data["schedule_semantics"] = {
        "measurement_condition":"(case,harness,candidate,R,B_RUN); canonical R=null, repeated R in {0,1}; one candidate per entry",
        "candidate_paired_condition":"not stored as a scheduling unit; default/cand4 are separate measurement conditions",
        "round_order":"ordered visit to every retained measurement condition once per round; each visit collects samples_per_round scalar timings",
        "invocation":"one temporal benchmark run containing ten round orders; same condition repeated across rounds",
        "sample":"one scalar timing observation; 100 per measurement condition per invocation, not 100 across all invocations",
        "eligible_n_formulas":{"repeated_conditions_per_invocation":"n*2*2*3 = 12n",
                               "repeated_invocation_conditions":"12n*3 = 36n",
                               "repeated_scalar_samples":"36n*10*10 = 3600n",
                               "canonical_conditions_per_invocation":"n*2*1*3 = 6n (R=null)",
                               "canonical_invocation_conditions":"6n*3 = 18n",
                               "canonical_scalar_samples":"18n*10*10 = 1800n",
                               "combined_conditions_per_invocation":"18n",
                               "combined_invocation_conditions":"54n",
                               "combined_scalar_samples":"5400n"},
        "filter":"only artifact/resource eligibility filtering of the original 20-case master order; preserve relative order, never regenerate after Stage B gates"}
    protocol_data["schedule_cardinality"] = schedule_accounting(schedule)
    docs = {"structural_pool.json":dump(pool),
            "exclusions.json":dump({"reason_counts_nonexclusive":pool["reason_counts_nonexclusive"],
                                    "transitions":[r for r in pool["transitions"] if not r["included_in_structural_pool"]]}),
            "canonical_eligibility.json":dump(inventory), "reproduction_plan.json":dump(reproduction_plan(pool)),
            "rotation_schedule.json":dump(schedule), "protocol.json":dump(protocol_data),
            "cohort_summary.md":render_summary(pool,inventory), "protocol.md":(BASE/"phase4/PROTOCOL.md").read_text(),
            "source_bindings.json":dump(source_bindings(inventory))}
    return docs


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for name,text in generate().items():
        (OUTPUT/name).write_text(text)
    print("Stage A frozen: structural metadata only; no compiler, GPU, Modal or measurements invoked.")
