"""Derive Stage B admission exclusively from frozen structure and byte archives.

Canonical, single and repeated residency are all checked, preserving the frozen
Stage A archive contract. Driver cuOccupancy (not runtime cudaOccupancy) is used.
No performance data is opened. Export failures remain PENDING; observed
structural failures are EXCLUDE_FROM_TIMING. Nothing runs a future schedule.
"""
import ast
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4 import artifact_gate as gate
from experiments.tma_reduction_layout.gluon import artifact_checks as ac
from experiments.tma_reduction_layout.analyze_ir import analyze_ttgir
from experiments.tma_reduction_layout.source_provenance import compute_manifest_digest

OUT = Path(__file__).parent / "results/artifact_gate"
PREREG = ROOT / "experiments/tma_reduction_layout/results/phase4/preregistration"
HARNESSES = ("canonical", "single", "repeated")
CANDIDATES = ("default", "4")
REQUIRED = ("kernel.ttgir", "kernel.ptx", "kernel.sass", "kernel.resource.txt",
            "kernel.cubin", "kernel.cubin.sha256", "metadata.json", "occupancy.json", "smoke.json", "observed_structure.json")
DERIVED = {"archive_bindings.json", "gate_results.json", "cohort_after_gate.json",
           "summary.md", "eligible_schedule_preview.json", "validator_report.json", "derivation_provenance.json"}
FROZEN = ("structural_pool.json", "exclusions.json", "rotation_schedule.json", "protocol.json", "source_bindings.json")
KERNEL_PATH = "experiments/tma_reduction_layout/phase4/kernels_stage_b.py"
# Define reason semantics before observing the cohort; no case-name overrides.
REASONS = {
    "CANONICAL_LAYOUT_DRIFT": "Observed canonical layout differs from frozen expectation",
    "SINGLE_LAYOUT_MISMATCH": "Single logical/layout/execution contract differs",
    "SINGLE_LOCALLOAD_MISMATCH": "Complete ordered single LocalLoad sequence differs",
    "SINGLE_REDUCTION_MISMATCH": "Frozen complete filtered fingerprint multiset differs",
    "REPEATED_REDUCTION_MISMATCH": "Repeated complete loop fingerprint multiset differs",
    "REPEATED_TILE_RELOAD": "TMA or initial tile LocalLoad not isolated before runtime loop",
    "REPEATED_EXTRA_MEMORY_EFFECT": "Complete runtime loop memory multiset differs from canonical reduction",
    "REDUCTION_TEMPLATE_MISMATCH": "Canonical terminal exchange missing from runtime loop",
    "REPEATED_ACCUMULATOR_CONFOUND": "FP32 accumulator, global/atomic or other forbidden loop effect",
    "R_SPECIALIZED": "Runtime R unspecialized source/IR contract fails",
    "BINARY_NOT_INVARIANT": "R0/R1 loaded/archived CUBIN bindings differ",
    "SPILL": "Nonzero LOCAL or STACK",
    "RESIDENCY_MISMATCH": "Candidate blocks/SM or active warps/SM differ for any frozen harness",
    "BARRIER_CONTRACT_FAILURE": "Empty asm, tied-copy dataflow, symmetry or explicit loop MOV gate fails",
    "CORRECTNESS_FAILURE": "Non-timed reduction correctness or launch smoke failed",
    "COMPILER_EXPORT_FAILURE": "Compiler/export failure; evidence incomplete (PENDING)",
    "PROVENANCE_INCOMPLETE": "Missing/mismatched source or binary evidence (PENDING)",
    "EXACT_BINARY_OCCUPANCY_UNAVAILABLE": "Missing/unbound occupancy evidence (PENDING)",
    "STRUCTURAL_CONTRACT_FAILURE": "Actual IR/PTX/SASS fails a frozen structural contract",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()


def require(test, message):
    if not test:
        raise ValueError(message)


def frozen_files(root):
    bindings = json.loads((root / "source_bindings.json").read_text())
    result = {}
    for name in FROZEN:
        live, archived = (PREREG / name).read_bytes(), (root / ("frozen_" + name)).read_bytes()
        require(live == archived and sha(live) == bindings["frozen_SHA256"][name], "Frozen Stage A binding: " + name)
        result[name] = json.loads(live)
    return result


def verify_sources(root):
    bindings = json.loads((root / "source_bindings.json").read_text())
    prov = json.loads((root / "local_source_provenance.json").read_text())
    env = json.loads((root / "environment.json").read_text())
    dispatch = json.loads((root / "modal_dispatch.json").read_text())
    raw = (root / "uploaded_source.zip").read_bytes()
    require(sha(raw) == bindings["uploaded_source_archive_sha256"], "Full uploaded source archive SHA")
    manifest = prov["source_manifest"]
    digest = compute_manifest_digest(manifest)
    require(digest == prov["source_manifest_sha256"] == bindings["local_source_manifest_sha256"]
            == bindings["uploaded_source_subset_sha256"]
            == env["source_verification"]["remote_source_subset_sha256"], "Uploaded/local source subset closure")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        require(len(z.namelist()) == len(set(z.namelist())) and set(z.namelist()) == set(manifest), "Full source ZIP membership")
        for path, digest in manifest.items():
            require(sha(z.read(path)) == digest, "Uploaded source byte SHA: " + path)
        source = z.read(KERNEL_PATH).decode()
    require(sha(source.encode()) == bindings["kernel_source_sha256"] == manifest[KERNEL_PATH], "Kernel source closure")
    require(bindings["build_identity"] == prov["composite_digest_sha256"], "Build identity")
    require(prov["git_head_sha"] == "e430b24b31d484440cc10ca7c79576dec88b13a0", "Frozen starting HEAD")
    require("H100" in env["gpu_name"] and env["compute_capability"] == [9, 0], "Strict H100 environment")
    require(env["gpu_uuid"].startswith("GPU-") and env["no_performance_observation"] is True, "GPU/provenance fields")
    require(dispatch["resolved_image_id"].startswith("im-") and dispatch["function_id"].startswith("fu-"), "Resolved Modal identity")
    if env["modal_image"] != "UNAVAILABLE":
        require(env["modal_image"] == dispatch["resolved_image_id"], "Remote/dispatch image identity")
    for key in ("actual_compiler_ptxas", "built_triton_native_extension"):
        tool = env["toolchain"][key]
        require(tool["path"].startswith("/opt/triton-src/") and bool(re.fullmatch(r"[0-9a-f]{64}", tool["sha256"])), "Actual build tool/library SHA: " + key)
    require(bool(env["toolchain"]["actual_compiler_ptxas"]["version"]), "Actual compiler version")
    require(sha((ROOT / KERNEL_PATH).read_bytes()) == bindings["kernel_source_sha256"], "Current/compiled kernel source unchanged")
    require(all(c["return_code"] == 0 for c in env["checked_cuda_calls"]), "Environment CUDA returns")
    for key, val in {"CCACHE_DIR": "/cache/ccache", "TRITON_HOME": "/cache/triton-home",
                     "TRITON_BUILD_WITH_CCACHE": "true", "TRITON_BUILD_WITH_CLANG_LLD": "true"}.items():
        require(env["persistent_cache"][key] == val, "Persistent build cache: " + key)
    return {"source": source, "manifest": manifest, "environment": env, "bindings": bindings, "dispatch": dispatch}


def bind_stage(metadata, ptx, source, harness):
    stage = metadata["reduction_stage"]
    require(stage["source_path"] == KERNEL_PATH and stage["source_sha256"] == sha(source.encode()), "Reduction source byte binding")
    expected_fn = {"canonical": "canonical_kernel", "single": "single_kernel", "repeated": "repeated_kernel"}[harness]
    require(stage["kernel_function"] == expected_fn, "Source harness identity")
    fns = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == expected_fn]
    require(len(fns) == 1, "Unique source kernel")
    ops = [n for n in ast.walk(fns[0]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("to", "max")]
    require(sorted(n.lineno for n in ops) == stage["source_lines"] and len(ops) == 2, "Complete convert/max AST stage")
    maxima = [n for n in ops if n.func.attr == "max"]
    require(len(maxima) == 1 and any(k.arg == "axis" and isinstance(k.value, ast.Constant) and k.value.value == 1 for k in maxima[0].keywords), "Source reduction axis")
    files = {int(idx): path for idx, path in re.findall(r'\.file\s+(\d+)\s+"([^"]+)"', ptx)}
    require(files.get(stage["file_id"]) == stage["ptx_source_path"] and stage["ptx_source_path"].endswith(KERNEL_PATH), "PTX source debug mapping")
    return stage


def geometry_from_ttgir(text):
    loads = re.findall(r'ttg\.local_load[^\n]*?-> tensor<(\d+x\d+x\d+)xbf16,', text)
    axes = re.findall(r'"tt\.reduce"\([^\n]+?\)\s*<\{axis\s*=\s*(\d+)\s*:\s*i32', text)
    require(len(loads) == len(axes) == 1, "Unique actual LocalLoad shape/reduction axis")
    attrs = analyze_ttgir(text)["module_attributes"]
    return {"logical_shape": list(map(int, loads[0].split("x"))), "reduction_axis": int(axes[0]),
            "num_warps": attrs["num_warps"], "num_ctas": attrs["num_ctas"],
            "threads_per_warp": attrs["threads_per_warp"]}


def exact_bundle(blobs, case, harness, candidate, context):
    """Pure byte audit; used unchanged by mutation probes and the real gate."""
    require(all(blobs.get(n) for n in REQUIRED), "Missing exact archive file")
    cubin = blobs["kernel.cubin"]
    digest = sha(cubin)
    require(cubin.startswith(b"\x7fELF") and blobs["kernel.cubin.sha256"].decode().strip() == digest, "Actual CUBIN byte/SHA closure")
    texts = {key: blobs["kernel." + key].decode() for key in ("ttgir", "ptx", "sass", "resource.txt")}
    resources = ac.parse_resource(texts["resource.txt"])
    metadata, occupancy, smoke = (json.loads(blobs[key]) for key in ("metadata.json", "occupancy.json", "smoke.json"))
    require((metadata["case_id"], metadata["harness"], metadata["candidate"]) == (case["config_id"], harness, candidate), "Case/harness/candidate identity")
    require(metadata["logical_shape"] == case["logical_shape"] and metadata["num_warps"] == case["num_warps"] and metadata["B_DESC"] == 65536 and metadata["reduction_axis"] == 1, "Binary geometry binding")
    require(metadata["cubin_sha256"] == occupancy["cubin_sha256"] == occupancy["queried_cubin_sha256"] == digest, "Exact-binary occupancy SHA binding")
    require(occupancy["query_api"] == "cuOccupancyMaxActiveBlocksPerMultiprocessor"
            and occupancy["load_method"] == "cuModuleLoadData(buffer_of_archived_file_bytes)", "Archived-byte occupancy API")
    require(occupancy["function_name"] == metadata["function_name"], "Queried function binding")
    require(occupancy["resource_sha256"] == sha(blobs["kernel.resource.txt"]) and metadata["resources"] == resources, "Resource byte binding")
    require(occupancy["num_regs"] == resources["num_regs"] and occupancy["local_bytes"] == resources["local_bytes"], "Driver/resource register/local attributes")
    # Legacy parse_resource labels cuobjdump SHARED as static_smem_bytes. Keep
    # that raw observation, but do not equate it to CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES.
    # In this build the values are 1024 and 0 respectively. Both come from the
    # same SHA-bound CUBIN; neither is used as a substitute occupancy estimate.
    require(type(occupancy["static_smem_bytes"]) is int and occupancy["static_smem_bytes"] >= 0, "Driver static SMEM attribute")
    require(occupancy["dynamic_smem_bytes"] == metadata["dynamic_smem_bytes"] and type(metadata["dynamic_smem_bytes"]) is int and metadata["dynamic_smem_bytes"] >= 0, "Actual dynamic SMEM binding")
    require(occupancy["num_warps"] == case["num_warps"] and occupancy["block_threads"] == case["num_warps"] * 32 and occupancy["active_warps_per_sm"] == occupancy["blocks_per_sm_actual_dynamic_smem"] * case["num_warps"], "Occupancy geometry")
    require(all(type(occupancy[k]) is int and occupancy[k] > 0 for k in ("blocks_per_sm_actual_dynamic_smem", "blocks_per_sm_zero_dynamic_smem", "active_warps_per_sm")), "Positive theoretical residency")
    require(occupancy["gpu_uuid"] == context["environment"]["gpu_uuid"] and occupancy["device_limits"] == context["environment"]["device_limits"] and occupancy["build_identity"] == context["bindings"]["build_identity"], "Occupancy environment/build binding")
    calls = occupancy["checked_cuda_calls"]
    require(all(c["return_code"] == 0 for c in calls) and Counter(c["api"] for c in calls) == Counter({"cuModuleLoadData": 1, "cuModuleGetFunction": 1, "cuFuncGetAttribute": 3, "cuOccupancyMaxActiveBlocksPerMultiprocessor": 2, "cuModuleUnload": 1, **({"cuFuncSetAttribute": 1} if metadata["dynamic_smem_bytes"] > 49152 else {})}), "All exact occupancy CUDA return codes")
    stage = bind_stage(metadata, texts["ptx"], context["source"], harness)
    try:
        observed = {"layout": gate.ttgir_contract(texts["ttgir"], case["logical_shape"], case["num_warps"]),
                    "initial_localload": ac.initial_load_signature(texts["ptx"]),
                    "all_ptx_backedges": ac.ptx_backedges(texts["ptx"])}
        if harness == "repeated":
            loops = [e for e in observed["all_ptx_backedges"] if e["kind"] == "compiler_loop"]
            require(len(loops) == 1, "Unique runtime loop missing")
            observed["reduction_fingerprint"] = ac.fingerprint([i for i in ac.ptx_instructions(texts["ptx"])
                if loops[0]["start"] <= i["line"] <= loops[0]["end"]])
        else:
            observed["reduction_fingerprint"] = gate.body_fingerprint(texts["ptx"], stage, minimal_output=harness == "single")
    except ValueError as exc:
        observed = {"structural_parse_error": str(exc)}
    require(json.loads(blobs["observed_structure.json"]) == observed, "Independently re-extracted layout/LocalLoad/fingerprint observations")
    require(smoke["timed"] is False and smoke["compile_calls"] == metadata["compile_calls"] == 1
            and smoke["launch_api"].startswith("CompiledKernel.__getitem__") and smoke["launched_cubin_sha256"] == digest, "Compile-once direct launch closure")
    require(smoke["B_RUN"] == smoke["initialized_prefix_tiles"] == 4 and smoke["input_dtype"] == "bfloat16" and smoke["output_dtype"] == "float32", "Smoke geometry/type")
    binary_invariant = True
    if harness == "repeated":
        bindings = smoke["runtime_bindings"]
        require([b["R"] for b in bindings] == [0, 1], "Both unspecialized R launch records")
        binary_invariant = all(b[k] == digest for b in bindings for k in ("before_cubin_sha256", "after_cubin_sha256", "archived_cubin_sha256"))
        require("not reduction correctness" in smoke["purpose"], "Repeated launch-safety qualification")
    else:
        require(smoke["after_cubin_sha256"] == digest and smoke["purpose"] == "reduction correctness", "Correctness binary closure")
        require(smoke["passed"] == (smoke["max_abs_diff"] == 0.0), "Correctness result consistency")
    reported = {**resources, "cuobjdump_reported_shared_bytes": resources["static_smem_bytes"],
        "static_smem_bytes": occupancy["static_smem_bytes"], "dynamic_smem_bytes": metadata["dynamic_smem_bytes"],
        "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed"}
    try:
        geometry = geometry_from_ttgir(texts["ttgir"])
    except ValueError:
        geometry = None
    return {"texts": texts, "resources": reported, "occupancy": occupancy, "metadata": metadata,
            "observed_geometry": geometry, "observed_layout": observed.get("layout"),
            "smoke": smoke, "stage": stage, "cubin_sha256": digest, "binary_invariant": binary_invariant}


def load_bundle(root, case, harness, candidate, context):
    directory = root / harness / case["config_id"] / candidate
    attempt = json.loads((directory / "attempt.json").read_text())
    require((attempt["case_id"], attempt["candidate"], attempt["harness"], attempt["attempted"]) == (case["config_id"], candidate, harness, True), "Attempt membership")
    actual = {p.name: sha(p.read_bytes()) for p in directory.iterdir() if p.is_file() and p.name != "attempt.json"}
    require(attempt["artifact_SHA256"] == actual, "Immutable compiler export hashes: " + str(directory))
    if attempt["status"] != "EXPORTED":
        require(attempt["status"] == "PENDING" and attempt.get("error") and (directory / "error.txt").read_text() == attempt["error"], "Complete retained failure")
        return {"pending": True, "error": attempt["error"], "binary_generated": attempt.get("binary_generated", False)}
    blobs = {n: (directory / n).read_bytes() for n in REQUIRED}
    return exact_bundle(blobs, case, harness, candidate, context)


def memory_signature(instructions):
    return dict(sorted(Counter(i["opcode"] for i in instructions if i["opcode"].startswith(("ld.", "st.", "atom", "red.", "cp."))).items()))


def audit_case(root, case, context):
    cfg = case["config_id"]
    audits = {h: {c: load_bundle(root, case, h, c, context) for c in CANDIDATES} for h in HARNESSES}
    result = {"case_id": cfg, "structural_pool_status": "INCLUDED", "origin": case["origin"],
              "M": case["M"], "N": case["N"], "num_warps": case["num_warps"],
              "lanePart_transition": f'{case["default"]["lanePart_M"]}->{case["cand4"]["lanePart_M"]}',
              "reason_codes": [], "canonical": {}, "single": {}, "repeated": {},
              "resources": {}, "occupancy": {}, "binary_hashes": {}, "exact_binary_closure": True,
              "residency_comparisons": {}}
    reasons = result["reason_codes"]
    for h, candidates in audits.items():
        result["resources"][h], result["occupancy"][h], result["binary_hashes"][h] = {}, {}, {}
        for c, a in candidates.items():
            if a.get("pending"):
                result[h][c] = {"status": "PENDING", "error": a["error"], "binary_generated": a["binary_generated"]}
                result["exact_binary_closure"] = False
            else:
                result["resources"][h][c] = a["resources"]
                result["occupancy"][h][c] = a["occupancy"]
                result["binary_hashes"][h][c] = a["cubin_sha256"]
                result[h][c] = {"status": "EXPORTED", "smoke": a["smoke"],
                               "observed_geometry": a["observed_geometry"], "observed_layout": a["observed_layout"]}
                if a["resources"]["local_bytes"] or a["resources"]["stack_bytes"]:
                    reasons.append("SPILL")
                if not a["smoke"]["passed"]:
                    reasons.append("CORRECTNESS_FAILURE")
                if not a["binary_invariant"]:
                    reasons.append("BINARY_NOT_INVARIANT")
        if all(not a.get("pending") for a in candidates.values()):
            residency = [(a["occupancy"]["blocks_per_sm_actual_dynamic_smem"], a["occupancy"]["active_warps_per_sm"]) for a in candidates.values()]
            result["residency_comparisons"][h] = {"default": residency[0], "4": residency[1], "matched": residency[0] == residency[1]}
            if residency[0] != residency[1]:
                reasons.append("RESIDENCY_MISMATCH")
    singles, reps = {}, {}
    for c in CANDIDATES:
        canonical = audits["canonical"][c]
        if canonical.get("pending"):
            continue
        try:
            observed = gate.ttgir_contract(canonical["texts"]["ttgir"], case["logical_shape"], case["num_warps"])
            expected = case["default" if c == "default" else "cand4"]
            match = observed["blocked"] == expected["layout"] and observed["shared"]["family"] == expected["shared_family"] and observed["shared"]["rank"] == expected["shared_rank"]
            result["canonical"][c].update({"expected": expected, "observed": observed, "layout_identity_pass": match,
                "initial_localload": ac.initial_load_signature(canonical["texts"]["ptx"]),
                "reduction_fingerprint": gate.body_fingerprint(canonical["texts"]["ptx"], canonical["stage"]),
                "all_ptx_backedges": ac.ptx_backedges(canonical["texts"]["ptx"])})
            if not match:
                reasons.append("CANONICAL_LAYOUT_DRIFT")
            single = audits["single"][c]
            if not single.get("pending"):
                singles[c] = gate.reproduction(canonical["texts"], single["texts"], case["logical_shape"], case["num_warps"], canonical["stage"], single["stage"])
                result["single"][c].update(singles[c])
                for field, code in (("layout_match", "SINGLE_LAYOUT_MISMATCH"), ("localload_match", "SINGLE_LOCALLOAD_MISMATCH")):
                    if not singles[c][field]: reasons.append(code)
                if singles[c]["classification"] == "REDUCTION_FINGERPRINT_MISMATCH": reasons.append("SINGLE_REDUCTION_MISMATCH")
            repeated = audits["repeated"][c]
            if not repeated.get("pending"):
                reps[c] = gate.repeated(canonical["texts"], repeated["texts"], case["logical_shape"], case["num_warps"], context["source"], canonical["stage"])
                edges = reps[c]["all_ptx_backedges"]
                runtime = next(e for e in edges if e["kind"] == "compiler_loop")
                loop = [i for i in ac.ptx_instructions(repeated["texts"]["ptx"]) if runtime["start"] <= i["line"] <= runtime["end"]]
                expected_mem = memory_signature(gate.body_instructions(canonical["texts"]["ptx"], canonical["stage"]))
                actual_mem = memory_signature(loop)
                # Strengthen explicit forbidden-effects enumeration without changing frozen families.
                forbidden = [i for i in loop if i["opcode"].startswith(("ld.global", "st.global", "atom", "red.", "cp.", "prefetch", "tensormap"))]
                reps[c]["checks"]["no_forbidden_global_async_memory"] = not forbidden
                reps[c]["structure_pass"] = all(reps[c]["checks"].values())
                reps[c].update({"expected_reduction_memory_signature": expected_mem,
                    "observed_loop_memory_signature": actual_mem, "forbidden_loop_instructions": forbidden,
                    "runtime_loop_fingerprint": ac.fingerprint(loop),
                    "preloop_localload_full_instructions": ac.initial_load_signature(repeated["texts"]["ptx"]),
                    "payload_bytes_per_thread": sum(int(re.search(r'\.[bus](\d+)$', op).group(1)) // 8 * int((re.search(r'\.v([124])\.', op) or [None, "1"])[1]) for op in reps[c]["observed_repeated_localload_opcodes"]),
                    "initial_load_encoding_semantics": "same actual TTGIR logical descriptor view and blocked/shared layouts; full per-thread byte extent; encoding equality does not prove address/dataflow equality"})
                result["repeated"][c].update(reps[c])
                codes = {"one_preloop_tma": "REPEATED_TILE_RELOAD", "one_preloop_localload_no_tile_reload": "REPEATED_EXTRA_MEMORY_EFFECT",
                    "runtime_R_unspecialized": "R_SPECIALIZED", "layouts_match": "STRUCTURAL_CONTRACT_FAILURE",
                    "preloop_localload_semantics": "REPEATED_TILE_RELOAD", "complete_body_matches": "REPEATED_REDUCTION_MISMATCH",
                    "terminal_exchanges_inside": "REDUCTION_TEMPLATE_MISMATCH", "no_accumulator_or_global_effect": "REPEATED_ACCUMULATOR_CONFOUND",
                    "zero_spills": "SPILL", "no_forbidden_global_async_memory": "REPEATED_EXTRA_MEMORY_EFFECT"}
                for check, passed in reps[c]["checks"].items():
                    if not passed: reasons.append(codes.get(check, "BARRIER_CONTRACT_FAILURE"))
        except ValueError as exc:
            result.setdefault("structural_errors", {})[c] = str(exc)
            reasons.append("STRUCTURAL_CONTRACT_FAILURE")
    if len(reps) == 2 and reps["default"]["copy_opcode_sequence"] != reps["4"]["copy_opcode_sequence"]:
        reasons.append("BARRIER_CONTRACT_FAILURE")
    incomplete = any(a.get("pending") for cs in audits.values() for a in cs.values())
    # Any observed real failure excludes, even if another bundle is incomplete.
    if reasons:
        final = "EXCLUDE_FROM_TIMING"
    elif incomplete:
        final = "PENDING"
        reasons.append("COMPILER_EXPORT_FAILURE")
    else:
        require(len(singles) == len(reps) == 2, "All structurally audited candidates")
        final = gate.body_gate_tier(singles, reps)
    result.update({"final_class": final, "pre_timing_eligible": final in ("PRIMARY", "SECONDARY"),
        "reason_codes": sorted(set(reasons)), "zero_spills": "SPILL" not in reasons and not incomplete,
        "single_reduction_equivalence": {c: result["single"][c].get("classification", "UNAVAILABLE") for c in CANDIDATES},
        "repeated_reduction_equivalence": {c: result["repeated"][c].get("classification", "UNAVAILABLE") for c in CANDIDATES},
        "repeated_localload_encoding": {c: result["repeated"][c].get("canonical_local_load_match", "UNAVAILABLE") for c in CANDIDATES},
        "single_localload_gate": {c: result["single"][c].get("localload_match", False) for c in CANDIDATES}})
    if result["pre_timing_eligible"]:
        result["canonical_timing_binary_sha"] = result["binary_hashes"]["canonical"]
        result["repeated_timing_binary_sha"] = result["binary_hashes"]["repeated"]
        result["Stage_C_launch_gate"] = "Compute actual runtime CUBIN SHA and compare the corresponding archived eligible SHA. Mismatch: ABORT BEFORE TIMING. Repeated R=0/1 and B=16384/32768/65536 reuse the same bound binary."
    return result


def stable_filter(master, eligible):
    # Preserve every invocation, round and every retained item in original order.
    result = json.loads(json.dumps(master))
    for invocation in result["invocations"]:
        for round_ in invocation["rounds"]:
            round_["order"] = [item for item in round_["order"] if item.split(":", 1)[0] in eligible]
    result["stage_b_preview_only"] = True
    result["executed"] = False
    return result


def coverage(cases, classification):
    selected = [c for c in cases if c["final_class"] == classification]
    return {key: dict(sorted(Counter(str(c[key]) for c in selected).items()))
            for key in ("M", "N", "num_warps", "lanePart_transition")}


def derive(root=OUT):
    frozen = frozen_files(root)
    context = verify_sources(root)
    cases = [c for c in frozen["structural_pool.json"]["transitions"] if c["included_in_structural_pool"]]
    require(len(cases) == 20 and frozen["structural_pool.json"]["excluded_structural_pool"] == 10, "Frozen structural membership")
    attempts = json.loads((root / "attempts.json").read_text())
    expected = {(c["config_id"], h, k) for c in cases for h in HARNESSES for k in CANDIDATES}
    require(len(attempts) == 120 and {(a["case_id"], a["harness"], a["candidate"]) for a in attempts} == expected and all(a["attempted"] for a in attempts), "120 attempts, no duplicates or omissions")
    for a in attempts:
        require(a == json.loads((root / a["harness"] / a["case_id"] / a["candidate"] / "attempt.json").read_text()), "Attempt ledger binding")
    audited = [audit_case(root, c, context) for c in cases]
    counts = {cls: sum(c["final_class"] == cls for c in audited) for cls in ("PRIMARY", "SECONDARY", "EXCLUDE_FROM_TIMING", "PENDING")}
    eligible = {c["case_id"] for c in audited if c["pre_timing_eligible"]}
    cohort = {"stage": "PHASE_4_STAGE_B", "structural_included": 20, "structural_excluded": 10,
        "counts": counts, "cases": audited, "no_performance_observation": True,
        "scientific_status_unchanged": {"H2a": "SUPPORTED_AT_REDUCTION_BODY_LEVEL", "H2b": "UNVERIFIED", "H2c": "UNVERIFIED"}}
    results = {"stage": "PHASE_4_STAGE_B", "attempts": 120, "candidate_variants": 40, "counts": counts,
        "generated_binaries": {h: sum(bool(a.get("binary_generated")) for a in attempts if a["harness"] == h) for h in HARNESSES},
        "exported_bundles": {h: sum(a["status"] == "EXPORTED" for a in attempts if a["harness"] == h) for h in HARNESSES},
        "eligible_coverage": {cls: coverage(audited, cls) for cls in ("PRIMARY", "SECONDARY")},
        "reason_semantics": REASONS, "residency_contract": "Matched candidate blocks/SM and active warps/SM for canonical, single, repeated; register counts may differ",
        "fingerprint_contract": "Frozen family/immediate sequence or full multiset; operands/dataflow/topology not proven by a matching projection",
        "barrier_contract": "Empty input/sink asm; observed tied copies and SASS MOV/IMAD.MOV; frozen no-explicit-loop-MOV gate retained; indirect effects UNISOLATED_NOT_ASSUMED_ZERO",
        "resource_field_contract": "Root case resources.static_smem_bytes is the driver attribute; cuobjdump_reported_shared_bytes retains raw SHARED. Raw metadata and frozen body-helper resources retain legacy parser names. Here driver static=0 and cuobjdump SHARED=1024; both are archived separately without assuming equality or using a proxy occupancy calculation.",
        "no_performance_observation": True,
        "R0_limitation": "R=0 subtraction controls fixed one-time harness differential. It does not prove absence of interactions via register allocation, live ranges, instruction scheduling, or compiler decisions.",
        "Stage_C_contract": "For canonical and repeated, runtime CUBIN SHA must equal archived timing-eligible SHA; mismatch ABORT BEFORE TIMING. Stage C not started.",
        "master_schedule_sha256": context["bindings"]["frozen_SHA256"]["rotation_schedule.json"]}
    results["structural_observations"] = {
        "single_pipelined": [c["case_id"] + ":" + k for c in audited for k in CANDIDATES if c["single_reduction_equivalence"][k] == "PIPELINED_OPCODE_EQUIVALENT"],
        "repeated_pipelined": [c["case_id"] + ":" + k for c in audited for k in CANDIDATES if c["repeated_reduction_equivalence"][k] == "PIPELINED_OPCODE_EQUIVALENT"],
        "single_localload_failures": [c["case_id"] for c in audited if not all(c["single_localload_gate"].values())],
        "canonical_layout_drift": [c["case_id"] for c in audited if "CANONICAL_LAYOUT_DRIFT" in c["reason_codes"]],
        "residency_mismatches": {c["case_id"]: {h: v for h, v in c["residency_comparisons"].items() if not v["matched"]} for c in audited if "RESIDENCY_MISMATCH" in c["reason_codes"]}}
    preview = stable_filter(frozen["rotation_schedule.json"], eligible)
    return results, cohort, preview


def raw_inventory(root):
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name not in DERIVED}


def summary_text(results, cohort, root):
    env = json.loads((root / "environment.json").read_text())
    lines = ["# PHASE 4 STAGE B REPORT", "", "Outcome-blind structural eligibility; no Stage C execution or new timing.", "",
        f"GPU: {env['gpu_name']} / {env['gpu_uuid']}; CC {env['compute_capability']}.",
        f"Counts: {results['counts']}", f"Generated: {results['generated_binaries']}; exported: {results['exported_bundles']}", "",
        "E = EXACT_SEQUENCE_EQUIVALENT; P = PIPELINED_OPCODE_EQUIVALENT; M = fingerprint mismatch; U = unavailable.",
        "Load encodings are default/cand4. Blocks/SM refer to repeated actual dynamic SMEM.", "",
        "|case|origin|single d/4|single LL|repeated d/4|preload d/4|spill|blocks d/4|closure|class|reasons|",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    short = {"EXACT_SEQUENCE_EQUIVALENT": "E", "PIPELINED_OPCODE_EQUIVALENT": "P", "REDUCTION_FINGERPRINT_MISMATCH": "M", "UNAVAILABLE": "U"}
    for c in cohort["cases"]:
        eqs = ["/".join(short[c[field][k]] for k in CANDIDATES) for field in ("single_reduction_equivalence", "repeated_reduction_equivalence")]
        blocks = "/".join(str(c["occupancy"]["repeated"].get(k, {}).get("blocks_per_sm_actual_dynamic_smem", "U")) for k in CANDIDATES)
        lines.append(f"|{c['case_id']}|{c['origin']}|{eqs[0]}|{all(c['single_localload_gate'].values())}|{eqs[1]}|{'/'.join(c['repeated_localload_encoding'][k] for k in CANDIDATES)}|{'0' if c['zero_spills'] else 'see record'}|{blocks}|{c['exact_binary_closure']}|{c['final_class']}|{', '.join(c['reason_codes']) or '—'}|")
    lines += ["", "## Exact-binary closure", "",
        "Each exported bundle archives actual ELF CUBIN bytes and its SHA. Occupancy loads a buffer read from that file and queries the bound function using the checked CUDA driver API. R0/R1 smoke calls the same CompiledKernel directly after one warmup; hashes before/after both launches are retained. Export failures are preserved and cannot be eligible.", "",
        "Canonical/single deterministic four-tile prefix smoke compares float32 max over M. The descriptor retains B_DESC=65536; other tiles are uninitialized and not launched/read. Repeated zero output proves launch safety only.", "",
        "All backward/self edges, complete loop memory signatures, terminal exchanges, tied copy pairs and explicit SASS MOV/IMAD.MOV observations are retained in cohort_after_gate.json. Filtered opcode equality does not establish topology/dataflow equality. No zero-overhead barrier claim.", "",
        results["R0_limitation"], "", results["Stage_C_contract"], "",
        "## Resource field interpretation", "", results["resource_field_contract"], "",
        "## Structural observations", "", "```json", json.dumps(results["structural_observations"], indent=2), "```", "",
        "## Eligible coverage", "", "```json", json.dumps(results["eligible_coverage"], indent=2), "```", "",
        "## Provenance and frozen files", "",
        "source_bindings.json, local_source_provenance.json, uploaded_source.zip and modal_dispatch.json bind source bytes, local Git/diff, uploaded subset, frozen protocol/pool/schedule and resolved image/function identity. Per-bundle attempt.json binds immutable exports. Stage A master files are byte-identical. The original Stage A validator restricts new paths; Stage B outputs therefore live under phase4/results/artifact_gate.", "",
        "## No-performance attestation", "", "No new timing / latency / throughput / speedup / slope was collected or used. H2a/H2b/H2c are unchanged. The schedule preview is a stable filter and was not executed. Local commit only; no push; STOP before Stage C.", ""]
    return "\n".join(lines)


def derivation_provenance(root):
    files = [Path(__file__), Path(__file__).with_name("validate_artifact_gate.py"),
             Path(__file__).with_name("STAGE_B.md"), Path(gate.__file__), Path(ac.__file__)]
    return {"offline_source_SHA256": {p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in files},
        "compiled_source_bindings_sha256": sha((root / "source_bindings.json").read_bytes()),
        "note": "Offline auditors were corrected after export to distinguish raw cuobjdump SHARED from driver static SMEM and validate serialized observations. No compiler/kernel source or raw export was changed; all cases are re-audited."}


def main():
    results, cohort, preview = derive()
    inventory = raw_inventory(OUT)
    ledger = OUT / "archive_bindings.json"
    if ledger.exists():
        require(json.loads(ledger.read_text()) == inventory, "Immutable Stage B raw archive inventory")
    else:
        ledger.write_bytes(json_bytes(inventory))
    for name, obj in (("gate_results.json", results), ("cohort_after_gate.json", cohort), ("eligible_schedule_preview.json", preview)):
        (OUT / name).write_bytes(json_bytes(obj))
    (OUT / "summary.md").write_text(summary_text(results, cohort, OUT))
    (OUT / "derivation_provenance.json").write_bytes(json_bytes(derivation_provenance(OUT)))
    print(json.dumps({"counts": results["counts"], "generated": results["generated_binaries"]}, indent=2))


if __name__ == "__main__":
    main()
