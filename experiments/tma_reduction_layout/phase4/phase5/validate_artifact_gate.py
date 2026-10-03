"""Phase 5 independent raw/body/class/coverage checks and corruption probes; no outcome evaluation."""
import copy
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4.phase5 import stage_b_audit as audit
from experiments.tma_reduction_layout.phase4.phase5 import gate_contract as contract
from collections import Counter
from experiments.tma_reduction_layout.phase4 import artifact_gate as gate
from experiments.tma_reduction_layout.gluon import artifact_checks as ac

CHECKS = []
PROBES = []


def check(name, test):
    if not test:
        raise AssertionError(name)
    CHECKS.append(name)
    print("PASS:", name)


def probe(name, test):
    check("corruption: " + name, test)
    PROBES.append({"name": name, "result": "PASS"})


def rejected(fn):
    try:
        fn()
    except (ValueError, KeyError, FileNotFoundError, AssertionError):
        return True
    return False


def regression_probes(root, frozen, context):
    cases = [c for c in frozen["structural_pool.json"]["transitions"] if c["included_in_structural_pool"]]
    # Select by fixture properties, never case names or future admission class.
    fixtures = []
    for case in cases:
        for candidate in audit.CANDIDATES:
            directory = root / "repeated" / case["config_id"] / candidate
            if all((directory / name).is_file() for name in audit.REQUIRED):
                blobs = {n: (directory / n).read_bytes() for n in audit.REQUIRED}
                exact = audit.exact_bundle(blobs, case, "repeated", candidate, context)
                canonical_dir = root / "canonical" / case["config_id"] / candidate
                single_dir = root / "single" / case["config_id"] / candidate
                if all((canonical_dir / n).is_file() and (single_dir / n).is_file() for n in audit.REQUIRED):
                    canonical_blobs = {n: (canonical_dir / n).read_bytes() for n in audit.REQUIRED}
                    single_blobs = {n: (single_dir / n).read_bytes() for n in audit.REQUIRED}
                    canonical = audit.exact_bundle(canonical_blobs, case, "canonical", candidate, context)
                    single = audit.exact_bundle(single_blobs, case, "single", candidate, context)
                    try:
                        repeated = gate.repeated(canonical["texts"], exact["texts"], case["logical_shape"], case["num_warps"], context["source"], canonical["stage"])
                    except ValueError:
                        continue
                    fixtures.append((case, candidate, blobs, canonical, single, exact, repeated))
    check("actual exported fixtures available for corruption tests", bool(fixtures))
    case, candidate, blobs, canonical, single, repeated, baseline = fixtures[0]
    changed = dict(blobs)
    raw = bytearray(changed["kernel.cubin"])
    raw[-1] ^= 1
    changed["kernel.cubin"] = bytes(raw)
    probe("1 archived CUBIN byte mutation fails closure", rejected(lambda: audit.exact_bundle(changed, case, "repeated", candidate, context)))
    changed = dict(blobs)
    occupancy = json.loads(changed["occupancy.json"])
    occupancy["queried_cubin_sha256"] = "0" * 64
    changed["occupancy.json"] = audit.json_bytes(occupancy)
    probe("2 wrong occupancy CUBIN SHA fails", rejected(lambda: audit.exact_bundle(changed, case, "repeated", candidate, context)))
    multi = next(f for f in fixtures if len(ac.initial_load_signature(f[4]["texts"]["ptx"])) >= 2)
    c, _, _, canon, single_, _, _ = multi
    changed_texts = copy.deepcopy(single_["texts"])
    loads = ac.initial_load_signature(changed_texts["ptx"])
    lines = changed_texts["ptx"].splitlines()
    second = loads[1]
    replacement = "ld.shared.b16" if second["opcode"] != "ld.shared.b16" else "ld.shared.b32"
    lines[second["line"] - 1] = lines[second["line"] - 1].replace(second["opcode"], replacement)
    changed_texts["ptx"] = "\n".join(lines)
    probe("3 second LocalLoad width alone fails single gate", not gate.reproduction(canon["texts"], changed_texts, c["logical_shape"], c["num_warps"], canon["stage"], single_["stage"])["localload_match"])
    def run_changed(texts):
        return gate.repeated(canonical["texts"], texts, case["logical_shape"], case["num_warps"], context["source"], canonical["stage"])
    changed_texts = copy.deepcopy(repeated["texts"])
    load_line = next(line for line in changed_texts["ttgir"].splitlines() if "ttg.local_load" in line)
    changed_texts["ttgir"] = changed_texts["ttgir"].replace(load_line + "\n", "", 1)
    loop_line = next(line for line in changed_texts["ttgir"].splitlines() if "scf.for" in line)
    changed_texts["ttgir"] = changed_texts["ttgir"].replace(loop_line, loop_line + "\n" + load_line, 1)
    probe("4 TTGIR LocalLoad moved into R loop fails", not run_changed(changed_texts)["checks"]["one_preloop_localload_no_tile_reload"])
    loop = next(e for e in ac.ptx_backedges(repeated["texts"]["ptx"]) if e["kind"] == "compiler_loop")
    def insert_loop(extra):
        texts = copy.deepcopy(repeated["texts"])
        lines = texts["ptx"].splitlines()
        lines.insert(loop["start"], extra)
        texts["ptx"] = "\n".join(lines)
        return texts
    probe("5 extra loop ld.shared fails complete memory gate", not run_changed(insert_loop("ld.shared.b32 %r999, [%r998];"))["checks"]["one_preloop_localload_no_tile_reload"])
    # Move an actual terminal shared load from the loop to immediately after it.
    terminal = [i for i in ac.ptx_instructions(repeated["texts"]["ptx"]) if loop["start"] <= i["line"] <= loop["end"] and i["opcode"].startswith(("ld.shared", "ldmatrix"))][-1]
    changed_texts = copy.deepcopy(repeated["texts"])
    lines = changed_texts["ptx"].splitlines()
    instruction = lines.pop(terminal["line"] - 1)
    lines.insert(loop["end"] - 1, instruction)
    changed_texts["ptx"] = "\n".join(lines)
    terminal_audit = run_changed(changed_texts)
    probe("6 actual terminal exchange sunk outside loop fails", not terminal_audit["checks"]["terminal_exchanges_inside"] and not terminal_audit["structure_pass"])
    probe("7 FP32 add accumulator fails", not run_changed(insert_loop("add.f32 %f999, %f998, %f997;"))["checks"]["no_accumulator_or_global_effect"])
    changed = dict(blobs)
    smoke = json.loads(changed["smoke.json"])
    smoke["runtime_bindings"][1]["after_cubin_sha256"] = "f" * 64
    changed["smoke.json"] = audit.json_bytes(smoke)
    invariant = audit.exact_bundle(changed, case, "repeated", candidate, context)["binary_invariant"]
    real_loader = audit.load_bundle
    def altered_r_loader(root_, case_, harness_, candidate_, context_):
        if harness_ == "repeated" and candidate_ == candidate:
            return audit.exact_bundle(changed, case_, harness_, candidate_, context_)
        return real_loader(root_, case_, harness_, candidate_, context_)
    with patch.object(audit, "load_bundle", altered_r_loader):
        specialized_case = audit.audit_case(root, case, context)
    probe("8 R-dependent binary SHA fails case admission", not invariant and specialized_case["final_class"] == "EXCLUDE_FROM_TIMING" and "BINARY_NOT_INVARIANT" in specialized_case["reason_codes"])
    changed_texts = copy.deepcopy(repeated["texts"])
    lines = changed_texts["ptx"].splitlines()
    maxima = next(i for i in ac.ptx_instructions(repeated["texts"]["ptx"]) if loop["start"] <= i["line"] <= loop["end"] and i["opcode"].startswith("max."))
    lines[maxima["line"] - 1] = lines[maxima["line"] - 1].replace(maxima["opcode"], "min.f32")
    changed_texts["ptx"] = "\n".join(lines)
    probe("9 one repeated fingerprint instruction changed fails", run_changed(changed_texts)["classification"] == "REDUCTION_FINGERPRINT_MISMATCH")
    fp = ac.fingerprint([i for i in ac.ptx_instructions(repeated["texts"]["ptx"]) if loop["start"] <= i["line"] <= loop["end"]])
    changed_fp = list(fp)
    unequal = next(i for i in range(1, len(fp)) if fp[i] != fp[0])
    changed_fp[0], changed_fp[unequal] = changed_fp[unequal], changed_fp[0]
    # Sequence-tier regression is outcome-independent even if the actual cohort
    # contains no exact body or no PRIMARY case. Use the actual canonical family
    # sequence to construct a controlled exact projection and reorder its multiset.
    canonical_sequence = list(canonical["observed_layout"] and gate.body_fingerprint(canonical["texts"]["ptx"], canonical["stage"]))
    reordered_sequence = list(canonical_sequence)
    other = next(i for i in range(1, len(canonical_sequence)) if canonical_sequence[i] != canonical_sequence[0])
    reordered_sequence[0], reordered_sequence[other] = reordered_sequence[other], reordered_sequence[0]
    equal_body = {"structure_pass": True, "classification": "EXACT_SEQUENCE_EQUIVALENT", "copy_opcode_sequence": baseline["copy_opcode_sequence"]}
    reordered_body = {**equal_body, "classification": gate.equivalence(canonical_sequence, reordered_sequence)}
    probe("10 exact sequence reorder preserving multiset remains SECONDARY, never upgraded",
        reordered_body["classification"] == "PIPELINED_OPCODE_EQUIVALENT" and
        gate.body_gate_tier({k: equal_body for k in contract.CANDIDATES}, {"default": reordered_body, "4": equal_body}) == "SECONDARY")
    base_case = audit.audit_case(root, case, context)
    loader = audit.load_bundle
    forbidden = ("performance", "timing", "speedup", "winner", "regret", "marginal_ns", "G", "D", "g0", "g1", "residual", "Stage_C_outcomes", "expected_PRIMARY")
    class Poison(dict):
        def __getitem__(self, key):
            if key in forbidden: raise AssertionError("Outcome access: " + key)
            return super().__getitem__(key)
        def get(self, key, default=None):
            if key in forbidden: raise AssertionError("Outcome access: " + key)
            return super().get(key, default)
    def poisoned_loader(*args):
        value = copy.deepcopy(loader(*args))
        for key in ("metadata", "occupancy", "smoke"):
            if key in value:
                value[key] = Poison({**value[key], **{k: {"POISON": [999999, -999999]} for k in forbidden}})
        return value
    with patch.object(audit, "load_bundle", poisoned_loader):
        poisoned_case = audit.audit_case(root, Poison({**case, **{k: "POISON" for k in forbidden}}), context)
    # Outcomes in returned smoke/occupancy observations are stripped solely for
    # comparison; admission class/reasons are computed without accessing them.
    def remove_poison(obj):
        if isinstance(obj, dict): return {k: remove_poison(v) for k, v in dict.items(obj) if k not in forbidden}
        if isinstance(obj, list): return [remove_poison(v) for v in obj]
        return obj
    probe("11 denied Stage C performance fields leave admission unchanged", remove_poison(poisoned_case) == base_case)
    changed = dict(blobs)
    del changed["occupancy.json"]
    probe("12 missing occupancy cannot establish timing eligibility", rejected(lambda: audit.exact_bundle(changed, case, "repeated", candidate, context)))
    master = copy.deepcopy(contract.read(root / "master_schedule_preview.json"))
    order = master["invocations"][0]["rounds"][0]["order"]
    order[0], order[1] = order[1], order[0]
    digest = audit.sha((root / "frozen_rotation_schedule.json").read_bytes())
    check("unmodified master serialization preserves archived SHA", audit.sha(audit.json_bytes(contract.read(root / "master_schedule_preview.json"))) == digest)
    probe("13 reordered master schedule fails byte binding", audit.sha(audit.json_bytes(master)) != digest)


    bad_case = copy.deepcopy(case)
    bad_case["default"]["layout"]["sizePerThread"][2] *= 2
    drift = audit.audit_case(root, bad_case, context)
    probe("14 predicted-vs-observed canonical layout mutation excludes", drift["final_class"] == "EXCLUDE_FROM_TIMING" and "CANONICAL_LAYOUT_DRIFT" in drift["reason_codes"])
    models = ("model_A_prediction", "model_B_prediction", "model_C_prediction", "locked_development_models", "hypothesis_direction", "hypothesis_support")
    class ModelPoison(dict):
        def __getitem__(self, key):
            if key in models: raise AssertionError("Development model access: " + key)
            return super().__getitem__(key)
        def get(self, key, default=None):
            if key in models: raise AssertionError("Development model access: " + key)
            return super().get(key, default)
    for sign in (-1, 1):
        poisoned = ModelPoison({**case, **{key: sign * 1e99 for key in models}})
        probe("15 development model prediction poison sign " + str(sign) + " leaves eligibility unchanged",
            audit.audit_case(root, poisoned, context) == base_case)
    pool_bytes = (contract.PREREG / "structural_pool.json").read_bytes()
    changed_pool = json.loads(pool_bytes)
    next(r for r in changed_pool["transitions"] if r["included_in_structural_pool"])["included_in_structural_pool"] = False
    original_read = Path.read_bytes
    def changed_read(path):
        return contract.encode(changed_pool) if path == contract.PREREG / "structural_pool.json" else original_read(path)
    with patch.object(Path, "read_bytes", changed_read):
        probe("16 structural pool mutation fails frozen binding", rejected(lambda: audit.frozen_files(root)))


def independently_check_case(root, structural, recorded, context):
    """Reconstruct family sequences/counts and final tier independently of audit_case."""
    bundles = {h: {k: audit.load_bundle(root, structural, h, k, context) for k in contract.CANDIDATES}
        for h in contract.HARNESSES}
    pending = [a for values in bundles.values() for a in values.values() if a.get("pending")]
    good = not pending
    observed_failure = any(a["failure_reasons"] for a in pending)
    all_classes, repeated = [], {}
    for h, values in bundles.items():
        full = [a for a in values.values() if not a.get("pending")]
        observed_failure |= any(a["resources"]["local_bytes"] != 0 or a["resources"]["stack_bytes"] != 0
            or not a["smoke"]["passed"] or not a["binary_invariant"] for a in full)
        if len(full) == 2:
            residency = [(a["occupancy"]["blocks_per_sm_actual_dynamic_smem"], a["occupancy"]["active_warps_per_sm"]) for a in full]
            observed_failure |= residency[0] != residency[1]
        for k, value in values.items():
            if value.get("pending"): continue
            recorded_value = recorded[h][k]
            check(structural["config_id"] + "/" + h + "/" + k + " exact bytes/resources/occupancy/smoke", recorded_value["smoke"] == value["smoke"]
                and recorded["binary_hashes"][h][k] == value["cubin_sha256"] and recorded["resources"][h][k] == value["resources"]
                and recorded["occupancy"][h][k] == value["occupancy"])
    for k in contract.CANDIDATES:
        canonical = bundles["canonical"][k]
        if canonical.get("pending"): continue
        expected = structural["default" if k == "default" else "cand4"]
        try:
            actual = gate.ttgir_contract(canonical["texts"]["ttgir"], structural["logical_shape"], structural["num_warps"])
            observed_failure |= actual["blocked"] != expected["layout"] or actual["shared"]["family"] != expected["shared_family"] or actual["shared"]["rank"] != expected["shared_rank"]
            canonical_fp = ac.fingerprint(gate.body_instructions(canonical["texts"]["ptx"], canonical["stage"]))
            check(structural["config_id"] + "/" + k + " complete canonical source-bound family", recorded["canonical"][k]["reduction_fingerprint"] == canonical_fp)
            for h in ("single", "repeated"):
                value = bundles[h][k]
                if value.get("pending"): continue
                if h == "single":
                    observations = gate.reproduction(canonical["texts"], value["texts"], structural["logical_shape"], structural["num_warps"], canonical["stage"], value["stage"])
                    actual_fp = ac.fingerprint(gate.body_instructions(value["texts"]["ptx"], value["stage"], minimal_output=True))
                    check(structural["config_id"] + "/" + k + " single entire LocalLoad sequence", observations["localload_match"] ==
                        ([i["opcode"] for i in ac.initial_load_signature(canonical["texts"]["ptx"])] == [i["opcode"] for i in ac.initial_load_signature(value["texts"]["ptx"])]))
                else:
                    observations = gate.repeated(canonical["texts"], value["texts"], structural["logical_shape"], structural["num_warps"], context["source"], canonical["stage"])
                    edges = ac.ptx_backedges(value["texts"]["ptx"])
                    loops = [e for e in edges if e["kind"] == "compiler_loop"]
                    check(structural["config_id"] + "/" + k + " all backedges enumerated / one runtime loop", edges == recorded[h][k]["all_ptx_backedges"] and len(loops) == 1)
                    loop = [i for i in ac.ptx_instructions(value["texts"]["ptx"]) if loops[0]["start"] <= i["line"] <= loops[0]["end"]]
                    actual_fp = ac.fingerprint(loop)
                    forbidden = any(i["opcode"].startswith(("ld.global", "st.global", "atom", "red.", "cp.", "prefetch", "tensormap")) for i in loop)
                    observed_failure |= forbidden
                    repeated[k] = observations
                classification = "EXACT_SEQUENCE_EQUIVALENT" if actual_fp == canonical_fp else "PIPELINED_OPCODE_EQUIVALENT" if Counter(actual_fp) == Counter(canonical_fp) else "REDUCTION_FINGERPRINT_MISMATCH"
                check(structural["config_id"] + "/" + h + "/" + k + " sequence versus multiset recomputed", recorded[h][k]["classification"] == classification)
                observed_failure |= not observations["structure_pass"]
                all_classes.append(classification)
        except ValueError:
            observed_failure = True
    if len(repeated) == 2:
        observed_failure |= repeated["default"]["copy_opcode_sequence"] != repeated["4"]["copy_opcode_sequence"]
    if observed_failure:
        final = "EXCLUDE_FROM_TIMING"
    elif not good:
        final = "PENDING"
    else:
        check(structural["config_id"] + " all four body reproductions independently available", len(all_classes) == 4)
        final = "PRIMARY" if all(v == "EXACT_SEQUENCE_EQUIVALENT" for v in all_classes) else "SECONDARY"
    check(structural["config_id"] + " final tier independently reconstructed", recorded["final_class"] == final
        and recorded["pre_timing_eligible"] == (final in ("PRIMARY", "SECONDARY")))
    check(structural["config_id"] + " machine-readable reasons", bool(recorded["reason_codes"]) == (final in ("EXCLUDE_FROM_TIMING", "PENDING"))
        and set(recorded["reason_codes"]) <= set(audit.REASONS))
    if recorded["pre_timing_eligible"]:
        for h in ("canonical", "repeated"):
            check(structural["config_id"] + " future " + h + " exact binary contract",
                recorded[h + "_timing_binary_sha"] == {name: bundles[h][key]["cubin_sha256"] for name, key in (("default", "default"), ("cand4", "4"))})


def main():
    root = audit.OUT
    protected = contract.protected_inventory()
    frozen = audit.frozen_files(root)
    context = audit.verify_sources(root)
    check("all prior 2230 files protected against baseline Git tree", len(protected) == 2230)
    check("new raw compiler archive is immutable", contract.read(root / "archive_bindings.json") == audit.raw_inventory(root))
    check("offline audit source identities", contract.read(root / "derivation_provenance.json") == audit.derivation_provenance(root))
    real_read = contract.read
    def outcome_blind_read(path):
        path = Path(path)
        if path.name == "results.json" and ("timing" in path.parts or "residual_analysis" in path.parts):
            raise AssertionError("Gate attempted to decode development outcomes")
        if path == contract.PREREG / "protocol.json":
            raise AssertionError("Gate attempted to decode development models/hypothesis content")
        return real_read(path)
    with patch.object(contract, "read", outcome_blind_read):
        results, cohort, future, preview = audit.derive(root)
    for name, data in (("gate_results.json", results), ("cohort_after_gate.json", cohort),
                       ("hypothesis_feasibility.json", future), ("eligible_schedule_preview.json", preview)):
        check("independently regenerated " + name, contract.read(root / name) == json.loads(contract.encode(data)))
    check("complete report reproduces artifacts/coverage", (root / "summary.md").read_text() == audit.summary_text(results, cohort, future, root))
    check("19 cases / 114 original attempts / 5 exclusions untouched", len(cohort["cases"]) == 19 and results["original_binary_attempts"] == 114 and cohort["structural_excluded"] == 5)
    rows = contract.population(frozen["structural_pool.json"])
    for row, recorded in zip(rows, cohort["cases"]):
        active = root / recorded["active_archive_root"]
        current_context = context if active == root else audit.verify_sources(active)
        independently_check_case(active, row, recorded, current_context)
    counts = Counter(r["final_class"] for r in cohort["cases"])
    check("all class counts recomputed", results["counts"] == {k: counts[k] for k in ("PRIMARY", "SECONDARY", "EXCLUDE_FROM_TIMING", "PENDING")})
    eligible = [r for r in cohort["cases"] if r["pre_timing_eligible"]]
    warps = {str(w): sum(r["num_warps"] == w for r in eligible) for w in (4, 8)}
    check("coverage-only feasibility, no hypothesis/model evaluation", future["eligible_by_warps"] == warps
        and future["hypothesis_outcome_evaluation"] is False and future["model_predictions_or_errors_computed"] is False)
    for key, condition in (("H5_01_EXACT_BODY_DIRECTIONAL_TRACKING", counts["PRIMARY"] >= 5),
                           ("H5_02_FIXED_DIFFERENTIAL_ALONE_INSUFFICIENT", len(eligible) >= 5),
                           ("H5_03_WARP_REGIME_CONTEXT_PREDICTION", warps["4"] >= 3 and warps["8"] >= 3)):
        check(key + " frozen coverage requirement only", future["hypotheses"][key]["coverage_feasibility"] ==
            ("TESTABLE" if condition else "INCONCLUSIVE_BY_COVERAGE_BEFORE_TIMING") and future["hypotheses"][key]["status"] == "PREREGISTERED_HELD_OUT_NOT_YET_TESTED")
    master = contract.read(root / "master_schedule_preview.json")
    check("342 master conditions; stable eligible preview never executed", len(master["master_conditions"]) == 342
        and preview == audit.frozen_audit.stable_filter(master, {r["case_id"] for r in eligible}) and preview["executed"] is False)
    for invocation in preview["invocations"]:
        check("uniform 3x10x10 future schedule", len(invocation["rounds"]) == 10 and all(r["samples_per_visit"] == 10 for r in invocation["rounds"]))
    check("three future invocations/three warmups and zero timing", len(preview["invocations"]) == 3 and preview["warmup_count_per_condition"] == 3
        and cohort["timing_samples"] == results["timing_samples"] == future["timing_samples"] == 0)
    check("no Phase 5 timing tree or result", not any((contract.BASE / "phase4/results" / p).exists()
        for p in ("phase5/timing", "phase5_timing", "phase5_artifact_gate/timing")))
    import ast
    tree = ast.parse((Path(__file__).parent / "run_artifact_gate.py").read_text())
    forbidden_apis = {"do_bench", "elapsed_time", "Event", "cuEventElapsedTime", "perf_counter", "benchmark"}
    check("collector has no timing/event/benchmark API", not any(isinstance(n, ast.Attribute) and n.attr in forbidden_apis for n in ast.walk(tree)))
    regression_probes(root, {**frozen, "rotation_schedule.json": master}, context)
    check("all required corruption probes PASS", len(PROBES) >= 17)
    report = {"status": "PASS", "checks_passed": len(CHECKS), "checks": CHECKS, "corruption_probes": PROBES,
        "counts": results["counts"], "protected_prior_files": len(protected), "no_performance_observation": True,
        "hypothesis_outcome_evaluation": False, "timing_samples": 0,
        "independent_methods": "Exact archive/source closure; raw family sequences/multisets; full LocalLoad and loop/body/isolation/residency; manual final class; coverage-only feasibility; denied outcomes/models; retained Phase 4 corruption semantics"}
    (root / "validator_report.json").write_bytes(contract.encode(report))
    print(f"PHASE 5 ARTIFACT GATE VALIDATOR PASS: {len(CHECKS)} checks; {len(PROBES)} corruption probes; NO timing")


if __name__ == "__main__": main()
