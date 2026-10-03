"""Frozen Stage C fidelity contracts; no scientific outcome requirements."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4 import stage_b_audit as stage_b

BASELINE = "1db7ee4e6eb9be0d545a497b4c70492029f8c398"
GATE = stage_b.OUT
OUT = Path(__file__).parent / "results/timing"
WARMUP = 3
EXPECTED_COUNTS = {"eligible_cases": 18, "PRIMARY": 3, "SECONDARY": 15,
    "conditions_per_invocation": 324, "canonical_per_invocation": 108,
    "repeated_per_invocation": 216, "invocation_conditions": 972,
    "round_visits": 9720, "scalar_samples": 97200,
    "canonical_samples": 32400, "repeated_samples": 64800}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(obj):
    return (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def require(test, message):
    if not test:
        raise ValueError(message)


def condition(tag):
    cfg, harness, candidate, r, b = tag.split(":")
    return {"config_id": cfg, "harness": harness, "candidate": candidate,
            "R": None if r == "RNone" else int(r[1:]), "B_RUN": int(b[1:])}


def inputs():
    cohort = json.loads((GATE / "cohort_after_gate.json").read_text())
    preview = json.loads((GATE / "eligible_schedule_preview.json").read_text())
    cases = {c["case_id"]: c for c in cohort["cases"] if c["pre_timing_eligible"]}
    master = json.loads((stage_b.PREREG / "rotation_schedule.json").read_text())
    require(preview == stage_b.stable_filter(master, set(cases)), "Frozen stable eligibility filter")
    plan = json.loads(json.dumps(preview))
    plan["executed"] = True
    validate_plan(plan, cases, preview)
    return cases, plan, preview


def validate_plan(plan, cases, preview):
    expected = json.loads(json.dumps(preview))
    expected["executed"] = True
    require(plan == expected, "Executed plan exactly equals frozen preview, with executed=true only")
    require(len(cases) == 18 and Counter(c["final_class"] for c in cases.values()) == Counter({"PRIMARY": 3, "SECONDARY": 15}), "Frozen 18-case/3-primary/15-secondary cohort")
    require([i["invocation"] for i in plan["invocations"]] == [1, 2, 3], "Three invocation identities")
    all_visits = []
    for inv in plan["invocations"]:
        require([r["round"] for r in inv["rounds"]] == list(range(1, 11)), "Ten round identities")
        for round_ in inv["rounds"]:
            tags = round_["order"]
            require(len(tags) == len(set(tags)) == 324 and round_["samples_per_visit"] == 10, "324 unique conditions / ten scalar samples per visit")
            parsed = [condition(tag) for tag in tags]
            expected_conditions = {(cfg, h, c, r, b) for cfg in cases for h in ("canonical", "repeated")
                for c in ("default", "4") for r in ((None,) if h == "canonical" else (0, 1)) for b in (16384, 32768, 65536)}
            require({tuple(p[k] for k in ("config_id", "harness", "candidate", "R", "B_RUN")) for p in parsed} == expected_conditions, "Exact condition domains and membership")
            require(Counter(p["harness"] for p in parsed) == Counter({"canonical": 108, "repeated": 216}), "Harness cardinalities")
            all_visits.extend(parsed)
    require(len(all_visits) == 9720 and len(all_visits) * 10 == 97200, "9720 visits / 97200 samples preflight")
    return EXPECTED_COUNTS


def binary_map(cases):
    result = {}
    for cfg, case in cases.items():
        for h in ("canonical", "repeated"):
            for c in ("default", "4"):
                directory = GATE / h / cfg / c
                digest = case[h + "_timing_binary_sha"][c]
                cubin = directory / "kernel.cubin"
                require(sha(cubin.read_bytes()) == digest, "Stage B actual CUBIN SHA: " + str(cubin))
                meta = json.loads((directory / "metadata.json").read_text())
                ttgir = (directory / "kernel.ttgir").read_text()
                import re
                scratch = re.findall(r'ttg.global_scratch_alloc\s*\{alignment = (\d+) : i32, nbytes = (\d+) : i32\}', ttgir)
                require((h == "canonical" and scratch == [("128", "128")]) or (h == "repeated" and not scratch), "Frozen scratch ABI")
                result[f"{cfg}:{h}:{c}"] = {"archive_path": cubin.relative_to(ROOT).as_posix(),
                    "archive_sha256": digest, "metadata": meta,
                    "shared_layout": case[h][c]["observed_layout"]["shared"],
                    "scratch_bytes_per_cta": 128 if h == "canonical" else 0,
                    "ptx_sha256": sha((directory / "kernel.ptx").read_bytes())}
    require(len(result) == 72, "72 timing binaries")
    return result


def validate_invocation(raw, planned, cases, binaries):
    number = planned["invocation"]
    require(raw["status"] == "VALID_PROTOCOL_RUN" and raw["invocation"] == number, "Valid invocation identity")
    require(raw["recompilation"] is False and raw["timing_primitive"] == "CUDA_DRIVER_EVENTS_ONE_KERNEL_PER_SAMPLE", "Frozen binary / event primitive")
    require(raw["warmup_policy"] == {"count_per_condition": WARMUP, "condition_order": "first frozen round", "timed": False}, "Uniform frozen warmup policy")
    env = raw["environment"]
    require("H100" in env["gpu_name"] and env["compute_capability"] == [9, 0] and env["gpu_uuid"].startswith("GPU-"), "Strict H100 identity")
    require(env["driver_API_version"] > 0 and env["CUDA_runtime_version"] > 0
            and env["cudaRuntimeGetVersion_return_code"] == 0, "Checked driver/runtime versions")
    require(all(code == 0 for code in raw["cuda_return_codes"]) and bool(raw["cuda_call_counts"]), "All checked CUDA API returns")
    require(len(raw["loaded_binaries"]) == 72 and set(raw["loaded_binaries"]) == set(binaries), "All 72 loaded modules")
    for key, record in raw["loaded_binaries"].items():
        require(record["archive_sha256"] == record["runtime_loaded_cubin_sha256"] == binaries[key]["archive_sha256"]
                and record["archive_path"] == binaries[key]["archive_path"]
                and record["function_name"] == binaries[key]["metadata"]["function_name"], "Loaded archive/function closure: " + key)
        require(record["module_identity"] and record["function_identity"],
                "Module identity/all launches: " + key)
        # Each binary: 3 B * (1 canonical or 2 repeated R) * (100 samples + 3 warmups).
        cfg, harness, _ = key.split(":")
        required_launches = (3 if harness == "canonical" else 6) * (100 + WARMUP)
        require(record["sha_guard_count"] == required_launches and record["launch_count"] == required_launches, "Every launch SHA guard: " + key)
    expected_tags = planned["rounds"][0]["order"]
    require([r["condition_tag"] for r in raw["warmups"]] == expected_tags, "Warmup order and exact condition domain")
    for record in raw["warmups"]:
        p = condition(record["condition_tag"])
        key = f'{p["config_id"]}:{p["harness"]}:{p["candidate"]}'
        require(record["count"] == WARMUP and record["binary_sha256"] == binaries[key]["archive_sha256"], "Warmup count/binary closure")
    visits = raw["visits"]
    expected = [(round_["round"], tag) for round_ in planned["rounds"] for tag in round_["order"]]
    require([(r["round"], r["condition_tag"]) for r in visits] == expected, "Exact visit order; no missing/duplicate conditions")
    require(len(visits) == 3240, "3240 visits per invocation")
    counts = Counter()
    for visit in visits:
        p = condition(visit["condition_tag"])
        require(all(visit[k] == v for k, v in p.items()), "Raw condition/runtime arguments")
        case = cases[p["config_id"]]
        require(visit["origin"] == case["origin"] and visit["stage_b_class"] == case["final_class"]
                and visit["invocation"] == number and visit["gpu_uuid"] == env["gpu_uuid"], "Case/class/origin/invocation/GPU binding")
        values = visit["samples_us"]
        require(len(values) == 10 and all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in values), "Ten positive finite scalar samples")
        key = f'{p["config_id"]}:{p["harness"]}:{p["candidate"]}'
        digest = binaries[key]["archive_sha256"]
        require(visit["archived_cubin_sha256"] == visit["runtime_loaded_cubin_sha256"] == digest
                and visit["sample_launch_sha256"] == [digest] * 10 and visit["binary_sha_match"] is True, "Every sample launch SHA contract")
        counts[visit["condition_tag"]] += len(values)
    require(len(counts) == 324 and set(counts.values()) == {100}, "100 samples per exact condition")
    require(raw["cuda_call_counts"]["cuLaunchKernel"] == 324 * (100 + WARMUP)
            and raw["cuda_call_counts"]["cuEventElapsedTime"] == 32400
            and raw["cuda_call_counts"]["cuEventRecord"] == 64800
            and raw["cuda_call_counts"]["cuEventSynchronize"] == 32400
            and raw["cuda_call_counts"]["cuModuleLoadData"] == 72
            and raw["cuda_call_counts"]["cuModuleUnload"] == 72,
            "One kernel per event interval; warmups separate; exact module loading")
    return {"conditions": len(counts), "samples": sum(counts.values()), "gpu_uuid": env["gpu_uuid"]}


def protected_bindings():
    names = [stage_b.PREREG / n for n in stage_b.FROZEN]
    names += [GATE / n for n in ("cohort_after_gate.json", "eligible_schedule_preview.json", "archive_bindings.json")]
    return {p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in names}


def raw_names(root):
    derived = {"results.json", "validation.json", "summary.md", "raw_validation.json"}
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.name not in derived and p.name != "raw_manifest.json")


def validate_all(root=OUT):
    cases, plan, preview = inputs()
    require(json.loads((root / "executed_schedule.json").read_text()) == plan, "Frozen executed schedule binding")
    manifest = json.loads((root / "raw_manifest.json").read_text())
    require(manifest["raw_SHA256"] == {n: sha((root / n).read_bytes()) for n in raw_names(root)}, "Immutable raw-data bytes and complete inventory")
    require(manifest["protected_SHA256"] == protected_bindings() and manifest["starting_HEAD"] == BASELINE, "Frozen prior evidence lineage")
    require(stage_b.raw_inventory(GATE) == json.loads((GATE / "archive_bindings.json").read_text()), "All Stage B raw artifacts unchanged")
    binaries = binary_map(cases)
    require(manifest["Stage_B_binary_SHA256"] == {k: b["archive_sha256"] for k, b in binaries.items()}, "Stage B timing binary map")
    require(manifest["protocol_SHA256"] == sha((stage_b.PREREG / "protocol.json").read_bytes())
            and manifest["expected_counts"] == EXPECTED_COUNTS, "Frozen protocol SHA and expected counts")
    source = json.loads((root / "source_bindings.json").read_text())
    provenance = source["provenance"]
    require(provenance["git_head_sha"] == BASELINE and provenance["branch"] == "explore/tma-reduction-layout"
            and source["protected_SHA256"] == protected_bindings()
            and source["warmup_count_per_condition"] == WARMUP
            and source["timing_kernel_recompilation"] is False, "Source and pre-measurement policy lineage")
    from experiments.tma_reduction_layout.source_provenance import compute_manifest_digest
    require(compute_manifest_digest(provenance["source_manifest"]) == provenance["source_manifest_sha256"],
            "Source manifest digest fidelity")
    actual = []
    invocations = []
    for planned in plan["invocations"]:
        raw = json.loads((root / f'raw_invocation_{planned["invocation"]}.json').read_text())
        env = raw["environment"]
        verification = env["source_verification"]
        require(env["source_HEAD"] == BASELINE
                and env["source_manifest_sha256"] == provenance["source_manifest_sha256"]
                and env["archived_payload_sha256"] == source["archived_payload_sha256"]
                and verification["status"] == "PASS"
                and verification["uploaded_source_fidelity_verified"] is True
                and verification["missing_count"] == verification["mismatched_count"] == 0
                and verification["remote_source_subset_sha256"] == provenance["source_manifest_sha256"],
                "Per-invocation remote source/archive verification")
        actual.append(validate_invocation(raw, planned, cases, binaries))
        invocations.append(raw)
    require(len({r["dispatch_id"] for r in invocations}) == 3 and len({r["process_identity"] for r in invocations}) == 3, "Three separately dispatched single-use-container invocations")
    require(sum(r["samples"] for r in actual) == 97200 and sum(r["conditions"] for r in actual) == 972, "Full scalar/condition cardinalities")
    invalid = list((root / "invalid_invocations").glob("*.json")) if (root / "invalid_invocations").exists() else []
    require(all(json.loads(p.read_text())["status"] == "INVALID_PROTOCOL_RUN" for p in invalid), "Retained invalid invocation metadata")
    return {"counts": EXPECTED_COUNTS, "valid_invocations": 3, "invalid_invocations": len(invalid),
        "gpu_uuids": [r["gpu_uuid"] for r in actual], "recompilation": False,
        "replication_mode": "same-device temporal replication across distinct invocations" if len({r["gpu_uuid"] for r in actual}) == 1 else "distinct invocations on multiple physical UUIDs"}
