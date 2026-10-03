"""Outcome-blind Phase 5 gate, reusing the unchanged Phase 4 byte/body auditors."""
from collections import Counter
import io
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4.phase5 import gate_contract as c
from experiments.tma_reduction_layout.phase4 import stage_b_audit as frozen_audit
from experiments.tma_reduction_layout.phase4 import artifact_gate as gate
from experiments.tma_reduction_layout.gluon import artifact_checks as ac
from experiments.tma_reduction_layout.source_provenance import compute_manifest_digest

OUT = c.OUT
REQUIRED = frozen_audit.REQUIRED
KERNEL_PATH = frozen_audit.KERNEL_PATH
CANDIDATES = c.CANDIDATES
HARNESSES = c.HARNESSES
sha = c.sha
json_bytes = c.encode
REASONS = {**frozen_audit.REASONS,
    "COMPILE_FAILURE_DETERMINISTIC": "Deterministic compiler failure, never infrastructure PENDING",
    "RESOURCE_UNSUPPORTED": "Deterministic descriptor/compile/launch resource incompatibility",
    "INFRASTRUCTURE_FAILURE": "Temporary infrastructure failure or fragmentation; at most one entire-case fresh-container repair",
    "UNRESOLVED_TOOL_FAILURE": "Artifact export or occupancy tool failure; no complete timing eligibility"}


def frozen_files(root):
    binding = c.read(root / "source_bindings.json")
    result = {}
    for name in c.FROZEN:
        live, archived = (c.PREREG / name).read_bytes(), (root / ("frozen_" + name)).read_bytes()
        c.require(live == archived and c.sha(live) == binding["frozen_SHA256"][name], "Frozen Phase 5 preregistration bytes: " + name)
        # Do not decode hypothesis text or locked development-model coefficients.
        if name in ("structural_pool.json", "exclusions.json"):
            result[name] = json.loads(live)
    return result


def verify_sources(root):
    bindings, provenance, env, dispatch = [c.read(root / n) for n in
        ("source_bindings.json", "local_source_provenance.json", "environment.json", "modal_dispatch.json")]
    raw = (root / "uploaded_source.zip").read_bytes()
    c.require(c.sha(raw) == bindings["uploaded_source_archive_sha256"], "Uploaded source ZIP SHA")
    manifest = provenance["source_manifest"]
    digest = compute_manifest_digest(manifest)
    c.require(digest == provenance["source_manifest_sha256"] == bindings["local_source_manifest_sha256"]
        == bindings["uploaded_source_subset_sha256"] == env["source_verification"]["remote_source_subset_sha256"], "Full local/upload/build manifest closure")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        c.require(len(archive.namelist()) == len(set(archive.namelist())) and set(archive.namelist()) == set(manifest), "Uploaded source membership")
        for path, digest in manifest.items():
            c.require(c.sha(archive.read(path)) == digest, "Compiled source bytes: " + path)
        source = archive.read(KERNEL_PATH).decode()
    c.require(c.sha(source.encode()) == bindings["kernel_source_sha256"] == manifest[KERNEL_PATH]
        == c.sha((ROOT / KERNEL_PATH).read_bytes()), "Unchanged reused kernel source")
    c.require(bindings["build_identity"] == provenance["composite_digest_sha256"]
        and provenance["git_head_sha"] == bindings["git_head"] == c.BASELINE, "Phase 5 starting HEAD/build identity")
    for path in ("lib/Dialect/TritonGPU/Transforms/Coalesce.cpp", "include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td",
                 "experiments/tma_reduction_layout/phase4/phase5/run_artifact_gate.py",
                 "experiments/tma_reduction_layout/phase4/phase5/gate_contract.py"):
        c.require(c.sha((ROOT / path).read_bytes()) == manifest[path], "Current/compiled source closure: " + path)
    c.require("H100" in env["gpu_name"] and env["compute_capability"] == [9, 0]
        and env["gpu_uuid"].startswith("GPU-") and env["no_performance_observation"] is True, "Strict H100 / no timing")
    c.require(dispatch["resolved_image_id"].startswith("im-") and dispatch["function_id"].startswith("fu-")
        and dispatch["profile"] == "miaomingc" and dispatch["strict_gpu"] == "H100!:1", "Actual Modal image/function/profile")
    if env["modal_image"] != "UNAVAILABLE":
        c.require(env["modal_image"] == dispatch["resolved_image_id"], "Remote/dispatch image identity")
    c.require(c.sha((root / "raw_export.zip").read_bytes()) == dispatch["raw_export_zip_sha256"], "Immutable returned export ZIP")
    with zipfile.ZipFile(root / "raw_export.zip") as archive:
        for path in archive.namelist():
            c.require(archive.read(path) == (root / path).read_bytes(), "Original export bytes: " + path)
    for key in ("actual_compiler_ptxas", "built_triton_native_extension"):
        tool = env["toolchain"][key]
        c.require(tool["path"].startswith("/opt/triton-src/") and bool(re.fullmatch(r"[0-9a-f]{64}", tool["sha256"])), "Actual compiler/native identity")
    c.require(bool(env["toolchain"]["actual_compiler_ptxas"]["version"]) and env["CUDA_runtime_version"] > 0
        and env["cudaRuntimeGetVersion_return_code"] == 0 and all(v["return_code"] == 0 for v in env["checked_cuda_calls"]), "Checked toolchain/runtime/driver environment")
    for key, value in {"CCACHE_DIR": "/cache/ccache", "TRITON_HOME": "/cache/triton-home",
            "TRITON_BUILD_WITH_CCACHE": "true", "TRITON_BUILD_WITH_CLANG_LLD": "true"}.items():
        c.require(env["persistent_cache"][key] == value, "Frozen build-cache infrastructure")
    return {"source": source, "manifest": manifest, "environment": env, "bindings": bindings, "dispatch": dispatch}


def exact_bundle(blobs, case, harness, candidate, context):
    resources = ac.parse_resource(blobs["kernel.resource.txt"].decode())
    occupancy = json.loads(blobs["occupancy.json"])
    original_require = frozen_audit.require
    def checked_require(test, message):
        if message == "Driver/resource register/local attributes":
            # The old cohort had LOCAL=STACK=0. With a nonzero stack frame,
            # the driver reports LOCAL+STACK, while cuobjdump reports them
            # separately. Preserve both observations and exclude nonzero
            # LOCAL/STACK below; never rewrite the archive or hide the stack.
            original_require(occupancy["num_regs"] == resources["num_regs"] and
                occupancy["local_bytes"] == resources["local_bytes"] + resources["stack_bytes"], message)
        else:
            original_require(test, message)
    with patch.object(frozen_audit, "require", checked_require):
        result = frozen_audit.exact_bundle(blobs, case, harness, candidate, context)
    result["resources"]["driver_local_bytes"] = occupancy["local_bytes"]
    smoke = result["smoke"]
    c.require(smoke["B_DESC"] == 65536 and smoke["full_descriptor_allocation"] is True
        and smoke["input_allocation_shape"] == [65536, case["M"], case["N"]]
        and smoke["input_allocation_bytes"] == 65536 * case["M"] * case["N"] * 2
        and smoke["uninitialized_regions_never_launched_or_read"] is True, "Full frozen descriptor allocation; only initialized four-tile prefix accessed")
    c.require(not any(k in smoke for k in ("samples_us", "elapsed_time", "latency", "throughput", "G", "D", "g0")), "Smoke has no performance observation")
    return result


def load_bundle(root, case, harness, candidate, context):
    directory = root / harness / case["config_id"] / candidate
    attempt = c.read(directory / "attempt.json")
    c.require((attempt["case_id"], attempt["candidate"], attempt["harness"], attempt["attempted"]) ==
        (case["config_id"], candidate, harness, True), "Attempt identity")
    actual = {p.name: c.sha(p.read_bytes()) for p in directory.iterdir() if p.is_file() and p.name != "attempt.json"}
    c.require(attempt["artifact_SHA256"] == actual, "Every immutable compiler export hash")
    if attempt["status"] == "EXPORTED":
        return exact_bundle({n: (directory / n).read_bytes() for n in REQUIRED}, case, harness, candidate, context)
    c.require(attempt["status"] in ("PENDING", "FAILED_DETERMINISTIC") and attempt["error"]
        == (directory / "error.txt").read_text(), "Retained complete failure")
    reasons = []
    if attempt["status"] == "FAILED_DETERMINISTIC": reasons.append(attempt["failure_kind"])
    if (directory / "kernel.cubin").exists():
        data = (directory / "kernel.cubin").read_bytes()
        c.require(data.startswith(b"\x7fELF") and (directory / "kernel.cubin.sha256").read_text().strip() == c.sha(data), "Failed-attempt partial CUBIN identity")
    if (directory / "kernel.resource.txt").exists():
        resources = ac.parse_resource((directory / "kernel.resource.txt").read_text())
        if resources["local_bytes"] or resources["stack_bytes"]: reasons.append("SPILL")
    if (directory / "metadata.json").exists():
        meta = c.read(directory / "metadata.json")
        if meta["dynamic_smem_bytes"] > context["environment"]["device_limits"]["max_shared_memory_per_block_optin"]:
            reasons.append("RESOURCE_UNSUPPORTED")
    if harness == "canonical" and (directory / "kernel.ttgir").exists():
        try:
            observed = gate.ttgir_contract((directory / "kernel.ttgir").read_text(), case["logical_shape"], case["num_warps"])
            if observed["blocked"] != case["default" if candidate == "default" else "cand4"]["layout"]:
                reasons.append("CANONICAL_LAYOUT_DRIFT")
        except ValueError:
            reasons.append("CANONICAL_LAYOUT_DRIFT")
    return {"pending": True, "error": attempt["error"], "binary_generated": attempt.get("binary_generated", False),
        "failure_reasons": sorted(set(reasons)), "failure_kind": attempt["failure_kind"], "failure_stage": attempt["failure_stage"]}


def audit_case(root, case, context, loader=None):
    loader = loader or load_bundle
    with patch.object(frozen_audit, "load_bundle", loader):
        result = frozen_audit.audit_case(root, case, context)
    additional, incomplete = [], False
    for h in c.HARNESSES:
        for k in c.CANDIDATES:
            value = loader(root, case, h, k, context)
            if value.get("pending"):
                incomplete = True; additional += value["failure_reasons"]
                result[h][k].update({"failure_kind": value["failure_kind"], "failure_stage": value["failure_stage"]})
                if not value["failure_reasons"]:
                    result.setdefault("pending_reason_codes", []).append("INFRASTRUCTURE_FAILURE" if value["failure_kind"] == "INFRASTRUCTURE_FRAGMENTATION"
                        else "EXACT_BINARY_OCCUPANCY_UNAVAILABLE" if value["failure_stage"] == "OCCUPANCY" else "UNRESOLVED_TOOL_FAILURE")
            elif h == "canonical":
                expected_geometry = {"logical_shape": case["logical_shape"], "reduction_axis": 1,
                    "num_warps": case["num_warps"], "num_ctas": 1, "threads_per_warp": 32}
                if value["observed_geometry"] != expected_geometry: additional.append("CANONICAL_LAYOUT_DRIFT")
    reasons = sorted(set(result["reason_codes"] + additional) - {"COMPILER_EXPORT_FAILURE"})
    if reasons:
        result["final_class"] = "EXCLUDE_FROM_TIMING"
    elif incomplete:
        result["final_class"] = "PENDING"
        reasons = sorted(set(result.get("pending_reason_codes", ["UNRESOLVED_TOOL_FAILURE"])))
    result["reason_codes"] = reasons
    result["pre_timing_eligible"] = result["final_class"] in ("PRIMARY", "SECONDARY")
    result.pop("Stage_C_launch_gate", None)
    if result["pre_timing_eligible"]:
        result["canonical_timing_binary_sha"] = {name: result["binary_hashes"]["canonical"][key] for name, key in (("default", "default"), ("cand4", "4"))}
        result["repeated_timing_binary_sha"] = {name: result["binary_hashes"]["repeated"][key] for name, key in (("default", "default"), ("cand4", "4"))}
        result["Phase5_future_launch_gate"] = "Every future launch must hash exact archived CUBIN bytes; mismatch abort before timing. All B_RUN and R0/R1 reuse these bytes. No timing executed."
    return result


def feasibility(cases):
    eligible = [r for r in cases if r["pre_timing_eligible"]]
    counts = {str(w): sum(r["num_warps"] == w for r in eligible) for w in (4, 8)}
    primary = sum(r["final_class"] == "PRIMARY" for r in cases)
    definitions = [("H5_01_EXACT_BODY_DIRECTIONAL_TRACKING", primary >= 5, {"PRIMARY_n": primary, "required": 5}),
        ("H5_02_FIXED_DIFFERENTIAL_ALONE_INSUFFICIENT", len(eligible) >= 5, {"eligible_n": len(eligible), "required": 5}),
        ("H5_03_WARP_REGIME_CONTEXT_PREDICTION", all(counts[str(w)] >= 3 for w in (4, 8)), {"eligible_by_warps": counts, "required_each": 3})]
    return {"phase": "PHASE_5_STAGE_B_COVERAGE_ONLY", "eligible_by_warps": counts,
        "hypotheses": {name: {"coverage_feasibility": "TESTABLE" if ok else "INCONCLUSIVE_BY_COVERAGE_BEFORE_TIMING",
            "coverage": info, "status": "PREREGISTERED_HELD_OUT_NOT_YET_TESTED"} for name, ok, info in definitions},
        "future_nonconstant_variables_unknown": True, "hypothesis_outcome_evaluation": False,
        "model_predictions_or_errors_computed": False, "timing_samples": 0}


def derive(root=OUT):
    protected = c.protected_inventory()
    frozen = frozen_files(root); context = verify_sources(root)
    cases = c.population(frozen["structural_pool.json"])
    attempts = c.read(root / "attempts.json")
    expected = {(r["config_id"], h, k) for r in cases for h in c.HARNESSES for k in c.CANDIDATES}
    c.require(len(attempts) == 114 and {(a["case_id"], a["harness"], a["candidate"]) for a in attempts} == expected, "114 attempts: all 19 cases, no structural exclusions compiled")
    audited = []
    for row in cases:
        for a in (a for a in attempts if a["case_id"] == row["config_id"]):
            c.require(a == c.read(root / a["harness"] / a["case_id"] / a["candidate"] / "attempt.json"), "Original attempt ledger")
        active_root, active_context = root, context
        retry = root / "infrastructure_retries" / row["config_id"]
        if retry.exists():
            original = [a for a in attempts if a["case_id"] == row["config_id"]]
            c.require(len(original) == 6 and all(a.get("failure_kind") == "INFRASTRUCTURE_FRAGMENTATION" for a in original), "Entire case infrastructure-only repair")
            active_context = verify_sources(retry); active_root = retry
            replay = c.read(retry / "attempts.json")
            c.require(len(replay) == 6 and {(a["harness"], a["candidate"]) for a in replay} == {(h, k) for h in c.HARNESSES for k in c.CANDIDATES}
                and active_context["environment"]["modal_task_id"] != context["environment"]["modal_task_id"], "Exactly one fresh-container entire-case retry")
        audit = audit_case(active_root, row, active_context)
        audit["active_archive_root"] = active_root.relative_to(root).as_posix()
        audited.append(audit)
    counts = {key: sum(r["final_class"] == key for r in audited) for key in ("PRIMARY", "SECONDARY", "EXCLUDE_FROM_TIMING", "PENDING")}
    cohort = {"stage": "PHASE_5_STAGE_B", "structural_source": 24, "structural_included": 19, "structural_excluded": 5,
        "counts": counts, "cases": audited, "artifact_gate_executed": True, "no_performance_observation": True, "timing_samples": 0,
        "hypothesis_status": "PREREGISTERED_HELD_OUT_NOT_YET_TESTED", "protected_prior_inventory_SHA256": c.sha(c.encode(protected))}
    results = {"stage": "PHASE_5_STAGE_B", "artifact_gate_executed": True, "original_binary_attempts": 114, "counts": counts,
        "generated_binaries": {h: sum(bool(a.get("binary_generated")) for a in attempts if a["harness"] == h) for h in c.HARNESSES},
        "exported_bundles": {h: sum(a["status"] == "EXPORTED" for a in attempts if a["harness"] == h) for h in c.HARNESSES},
        "reason_semantics": REASONS, "residency_contract": "Reuse frozen Stage A/B: matched candidate blocks/SM and active warps/SM for canonical, single and repeated",
        "resource_fields": "Driver static SMEM, dynamic launch SMEM and cuobjdump SHARED stay separate; driver local bytes and raw cuobjdump LOCAL/STACK stay separate (driver local=LOCAL+STACK); nonzero LOCAL or STACK excludes; exact-CUBIN occupancy only",
        "barrier_contract": "Tied-copy mapping/count/symmetry and SASS loop MOV/IMAD.MOV remain observed; indirect effects UNISOLATED_NOT_ASSUMED_ZERO",
        "fingerprint_contract": "Unchanged frozen family: exact full sequence or exact full multiset; not operand/dataflow/whole-binary equivalence",
        "R0_limitation": "R=0 subtraction controls the fixed one-time differential, not absence of live-range/register/scheduling interactions",
        "no_performance_observation": True, "hypothesis_outcome_evaluation": False, "timing_samples": 0}
    startup = c.BASE / "phase4/results/phase5_artifact_gate_startup_failures"
    results["retained_startup_failures_SHA256"] = {p.relative_to(ROOT).as_posix(): c.sha(p.read_bytes())
        for p in sorted(startup.rglob("*")) if p.is_file()} if startup.exists() else {}
    master = c.read(root / "master_schedule_preview.json")
    c.require(master == c.master_schedule(frozen["structural_pool.json"]), "Full 19-case future master schedule fixed before performance")
    preview = frozen_audit.stable_filter(master, {r["case_id"] for r in audited if r["pre_timing_eligible"]})
    return results, cohort, feasibility(audited), preview


def raw_inventory(root=OUT):
    return {p.relative_to(root).as_posix(): c.sha(p.read_bytes()) for p in sorted(root.rglob("*"))
        if p.is_file() and p.relative_to(root).as_posix() not in c.DERIVED}


def derivation_provenance(root=OUT):
    files = [Path(__file__), Path(__file__).with_name("validate_artifact_gate.py"), Path(c.__file__),
        Path(frozen_audit.__file__), Path(gate.__file__), Path(ac.__file__)]
    return {"offline_source_SHA256": {p.relative_to(ROOT).as_posix(): c.sha(p.read_bytes()) for p in files},
        "compiled_source_bindings_SHA256": c.sha((root / "source_bindings.json").read_bytes()),
        "outcome_independent": True}


def summary_text(results, cohort, future, root=OUT):
    environment = c.read(root / "environment.json")
    lines = ["# PHASE 5 ARTIFACT GATE REPORT", "", "Starting HEAD: `" + c.BASELINE + "`.", "",
        "Held-out confirmatory extension, artifact/coverage stage only. Stage C remains development data. No timing, latency, throughput, slopes, G/D/g0, locked-model prediction or hypothesis outcome evaluation.", "",
        "All 19 structural included cases attempted; 114 original binary attempts. Five structural excluded cases never compiled. Counts: `" + json.dumps(results["counts"], sort_keys=True) + "`.", "",
        "GPU UUID: `" + environment["gpu_uuid"] + "`; " + environment["gpu_name"] + "; CC=" + str(environment["compute_capability"]) + ".", "",
        "Toolchain and actual image/build identities: environment.json, modal_dispatch.json and source_bindings.json. Actual compiler ptxas: `" + environment["toolchain"]["actual_compiler_ptxas"]["version"].replace("\n", "; ") + "`.", "",
        "| Case | Class | Single d/4 body | Repeated d/4 body | Repeated preload d/4 | Registers canonical/single/repeated d→4 | Blocks/SM canonical/single/repeated d→4 | Reasons |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in cohort["cases"]:
        def pair(field): return "/".join(row[field][k] for k in c.CANDIDATES)
        def resources(field, occupancy=False):
            container = row["occupancy"] if occupancy else row["resources"]
            return "; ".join(h + ":" + "→".join(str(container[h].get(k, {}).get(field, "UNAVAILABLE")) for k in c.CANDIDATES) for h in c.HARNESSES)
        lines.append("| " + " | ".join([row["case_id"], row["final_class"], pair("single_reduction_equivalence"), pair("repeated_reduction_equivalence"),
            pair("repeated_localload_encoding"), resources("num_regs"), resources("blocks_per_sm_actual_dynamic_smem", True), ", ".join(row["reason_codes"]) or "ALL_GATES_PASS"]) + " |")
    lines += ["", results["residency_contract"] + ".", "", results["resource_fields"] + ".", "",
        results["barrier_contract"] + ".", "", results["fingerprint_contract"] + ".", "", results["R0_limitation"] + ".", "",
        "Actual CUBIN bytes/SHAs, source stages, full LocalLoad sequences, body fingerprints, every backedge, tied-copy mappings and SMEM/resources/occupancy are retained for every exported bundle. All failed attempts and partial exports remain immutable.", "",
        "## M512 resources and correctness", "",
        "M512/N128 uses full B_DESC=65536, 8GiB BF16 input and 128KiB logical shared tile. No smaller descriptor, altered grid or resource workaround. Every smoke allocates the full descriptor and reads only four initialized prefix tiles. Repeated R0/R1 checks launch safety only.", ""]
    for row in cohort["cases"]:
        if row["M"] == 512:
            lines += [row["case_id"] + ": " + row["final_class"] + "; " + (", ".join(row["reason_codes"]) or "all gates pass") + "; resources `" + json.dumps(row["resources"], sort_keys=True) + "`.", ""]
    correctness = []
    for row in cohort["cases"]:
        for h in c.HARNESSES:
            for k in c.CANDIDATES:
                smoke = row[h][k].get("smoke")
                correctness.append({"case": row["case_id"], "harness": h, "candidate": k,
                    "passed": smoke["passed"] if smoke else None, "max_abs_diff": smoke.get("max_abs_diff") if smoke else None,
                    "purpose": smoke.get("purpose") if smoke else "NO_COMPLETED_SMOKE"})
    lines += ["Complete correctness/launch-safety ledger:", "", "```json", json.dumps(correctness, indent=2, sort_keys=True), "```", "",
        "## Future coverage feasibility only", "", "```json", json.dumps(future, indent=2, sort_keys=True), "```", "",
        "G/D variance is unknown until separately authorized timing. TESTABLE means coverage only, never support/falsification. No H5 outcome evaluation or model prediction/error calculation.", "",
        "Full 19-case master schedule and stable eligibility filter are preview only: three invocations, ten rounds, ten samples/visit, three warmups, unchanged B_DESC/B_RUN/R. Neither schedule was executed.", "",
        "Protected evidence is checked against the entire starting Git tree: Phase 3, Phase 4 A-D and Phase 5 preregistration unchanged. Scientific status and all hypothesis definitions are unchanged.", "",
        "The frozen preregistration validator inventories its entire phase5 results parent. New raw gate exports therefore live in the sibling phase4/results/phase5_artifact_gate/, preserving that validator and every frozen byte.", "",
        "Validator evidence: validator_report.json and validator_suite.json. Runtime Git commit/local/remote/clean checks are reported after normal push.", "",
        "NO Phase 5 timing. NO hypothesis outcome evaluation. NO production heuristic or PR #11991 change. STOP before timing.", ""]
    return "\n".join(lines).rstrip() + "\n"


def main():
    results, cohort, future, preview = derive()
    inventory = raw_inventory()
    ledger = OUT / "archive_bindings.json"
    if ledger.exists(): c.require(c.read(ledger) == inventory, "Immutable new raw artifact archive")
    else: ledger.write_bytes(c.encode(inventory))
    for name, data in (("gate_results.json", results), ("cohort_after_gate.json", cohort),
                       ("hypothesis_feasibility.json", future), ("eligible_schedule_preview.json", preview)):
        (OUT / name).write_bytes(c.encode(data))
    (OUT / "summary.md").write_text(summary_text(results, cohort, future))
    (OUT / "derivation_provenance.json").write_bytes(c.encode(derivation_provenance()))
    print(json.dumps({"counts": results["counts"], "feasibility": future}, indent=2))


if __name__ == "__main__": main()
