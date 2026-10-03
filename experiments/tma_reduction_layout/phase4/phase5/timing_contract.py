"""Phase 5 exact-binary timing fidelity; admission and order are frozen at Stage B."""
from collections import Counter
import hashlib
import json
import math
import subprocess
import io
import zipfile
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4 import stage_b_audit as stage_b

BASELINE = "56128d73233b234dacd191c569971b418680b857"
from experiments.tma_reduction_layout.phase4.phase5 import gate_contract as gc
GATE = gc.OUT
PREREG = gc.PREREG
OUT = GATE.parent / "phase5_timing"
WARMUP = 3


def dimensions(n, candidates=2, b_values=3, repeated_r=2, invocations=3, rounds=10, samples=10):
    canonical = n * candidates * b_values
    repeated = canonical * repeated_r
    conditions = canonical + repeated
    return {"eligible_cases": n, "PRIMARY": 2, "SECONDARY": 11,
        "conditions_per_invocation": conditions, "canonical_per_invocation": canonical,
        "repeated_per_invocation": repeated, "invocation_conditions": conditions * invocations,
        "round_visits": conditions * invocations * rounds,
        "scalar_samples": conditions * invocations * rounds * samples,
        "canonical_samples": canonical * invocations * rounds * samples,
        "repeated_samples": repeated * invocations * rounds * samples}


EXPECTED_COUNTS = dimensions(13)
DERIVED = {"results.json", "summary.md", "validation.json", "raw_validation.json", "validator_suite.json"}


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
    require(Counter(c["final_class"] for c in cohort["cases"]) == Counter(
        {"PRIMARY": 2, "SECONDARY": 11, "EXCLUDE_FROM_TIMING": 6}), "Frozen complete gate membership")
    timing = json.loads((PREREG / "protocol.json").read_text())["future_timing"]
    require(timing["B_DESC"] == 65536 and timing["B_RUN"] == [16384, 32768, 65536]
        and timing["R"] == [0, 1] and timing["canonical_R"] is None
        and timing["independent_invocations"] == 3 and timing["rounds"] == 10
        and timing["scalar_samples_per_round"] == 10 and timing["samples_per_condition"] == 100
        and timing["warmup_count_per_condition"] == WARMUP, "Immutable registered dimensions")
    require(EXPECTED_COUNTS == dimensions(len(cases), len(gc.CANDIDATES), len(timing["B_RUN"]),
        len(timing["R"]), timing["independent_invocations"], timing["rounds"],
        timing["scalar_samples_per_round"]), "Counts derived from frozen protocol dimensions")
    master = json.loads((GATE / "master_schedule_preview.json").read_text())
    require(preview == stage_b.stable_filter(master, set(cases)), "Frozen stable eligibility filter")
    plan = json.loads(json.dumps(preview))
    plan["executed"] = True
    validate_plan(plan, cases, preview)
    return cases, plan, preview


def validate_plan(plan, cases, preview):
    expected = json.loads(json.dumps(preview))
    expected["executed"] = True
    require(plan == expected, "Executed plan exactly equals frozen preview, with executed=true only")
    require(len(cases) == 13 and Counter(c["final_class"] for c in cases.values()) == Counter({"PRIMARY": 2, "SECONDARY": 11}), "Frozen 13-case/2-primary/11-secondary cohort")
    require([i["invocation"] for i in plan["invocations"]] == [1, 2, 3], "Three invocation identities")
    all_visits = []
    for inv in plan["invocations"]:
        require([r["round"] for r in inv["rounds"]] == list(range(1, 11)), "Ten round identities")
        for round_ in inv["rounds"]:
            tags = round_["order"]
            require(len(tags) == len(set(tags)) == EXPECTED_COUNTS["conditions_per_invocation"] and round_["samples_per_visit"] == 10, "Exact unique conditions / ten scalar samples per visit")
            parsed = [condition(tag) for tag in tags]
            expected_conditions = {(cfg, h, c, r, b) for cfg in cases for h in ("canonical", "repeated")
                for c in ("default", "4") for r in ((None,) if h == "canonical" else (0, 1)) for b in (16384, 32768, 65536)}
            require({tuple(p[k] for k in ("config_id", "harness", "candidate", "R", "B_RUN")) for p in parsed} == expected_conditions, "Exact condition domains and membership")
            require(Counter(p["harness"] for p in parsed) == Counter({"canonical": EXPECTED_COUNTS["canonical_per_invocation"], "repeated": EXPECTED_COUNTS["repeated_per_invocation"]}), "Harness cardinalities")
            all_visits.extend(parsed)
    require(len(all_visits) == EXPECTED_COUNTS["round_visits"] and len(all_visits) * 10 == EXPECTED_COUNTS["scalar_samples"], "Mechanically derived visits / scalar samples preflight")
    return EXPECTED_COUNTS


def binary_map(cases):
    result = {}
    for cfg, case in cases.items():
        for h in ("canonical", "repeated"):
            for c in ("default", "4"):
                directory = GATE / h / cfg / c
                digest = case[h + "_timing_binary_sha"]["cand4" if c == "4" else c]
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
    require(len(result) == len(cases) * 2 * 2, "Four exact timing binaries per eligible case")
    return result


def validate_invocation(raw, planned, cases, binaries):
    number = planned["invocation"]
    require(raw["status"] == "VALID_PROTOCOL_RUN" and raw["invocation"] == number, "Valid invocation identity")
    require(raw["recompilation"] is False and raw["timing_primitive"] == "CUDA_DRIVER_EVENTS_ONE_KERNEL_PER_SAMPLE", "Frozen binary / event primitive")
    require(raw["compilation_audit"] == {"compiler_subprocess_calls": 0, "triton_imported": False,
        "jit_compile_calls": 0, "load_input": "ELF_CUBIN_ONLY"}, "No compiler/JIT activity")
    require(raw["warmup_policy"] == {"count_per_condition": WARMUP, "condition_order": "first frozen round", "timed": False}, "Uniform frozen warmup policy")
    env = raw["environment"]
    require("H100" in env["gpu_name"] and env["compute_capability"] == [9, 0] and env["gpu_uuid"].startswith("GPU-"), "Strict H100 identity")
    require(env["driver_API_version"] > 0 and env["CUDA_runtime_version"] > 0
            and env["cudaRuntimeGetVersion_return_code"] == 0, "Checked driver/runtime versions")
    require(all(code == 0 for code in raw["cuda_return_codes"]) and bool(raw["cuda_call_counts"]), "All checked CUDA API returns")
    require(raw["subprocess_commands"] and all(Path(c[0]).name == "nvidia-smi" for c in raw["subprocess_commands"]), "Only telemetry subprocesses")
    require(set(raw["allocations"]) == set(cases), "All and only eligible full allocations")
    require(set(raw["tensor_maps"]) == {k for k in binaries if k.split(":")[1] == "repeated"}, "Exact repeated descriptor membership")
    for key, t in raw["tensor_maps"].items():
        cfg = key.split(":")[0]; m, n = cases[cfg]["M"], cases[cfg]["N"]
        sw = binaries[key]["shared_layout"]["swizzlingByteWidth"]
        require(t["global_dims_n_m_b"] == [n, m, 65536]
            and t["global_strides_bytes"] == [n * 2, m * n * 2]
            and t["message_box_n_m_b"] == [min(n, sw // 2), min(m, 256), 1]
            and t["element_strides"] == [1, 1, 1] and t["swizzling_bytes"] == sw
            and len(t["descriptor_SHA256"]) == 64, "Frozen host TMA ABI with unchanged full descriptor shape")
    for cfg, case in cases.items():
        a = raw["allocations"][cfg]; m, n = case["M"], case["N"]
        require(a["input_shape"] == [65536, m, n] and a["input_dtype"] == "torch.bfloat16"
            and a["input_bytes"] == 65536 * m * n * 2 and a["all_descriptor_elements_initialized"] is True
            and a["output_shape"] == [65536, n] and a["output_dtype"] == "torch.float32"
            and a["output_bytes"] == 65536 * n * 4 and a["scratch_bytes"] == 65536 * 128
            and all(a[k] > 0 and a[k] % 128 == 0 for k in ("input_pointer", "output_pointer", "scratch_pointer")), "Full descriptor-backed allocation: " + cfg)
    require(len(raw["loaded_binaries"]) == len(binaries) and set(raw["loaded_binaries"]) == set(binaries), "All exact loaded modules")
    for key, record in raw["loaded_binaries"].items():
        require(record["archive_sha256"] == record["runtime_loaded_cubin_sha256"] == binaries[key]["archive_sha256"]
                and record["archive_path"] == binaries[key]["archive_path"]
                and record["function_name"] == binaries[key]["metadata"]["function_name"], "Loaded archive/function closure: " + key)
        require(record["module_identity"] and record["function_identity"],
                "Module identity/all launches: " + key)
        require(record["load_api"] == "cuModuleLoadData(archived_file_bytes)", "ELF load API")
        dynamic = binaries[key]["metadata"]["dynamic_smem_bytes"]
        require(record["dynamic_smem_bytes"] == dynamic
            and record["dynamic_smem_optin"] == (dynamic > 49152), "Exact large-SMEM launch opt-in")
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
        if p["harness"] == "canonical":
            require(record["loaded_ABI_correctness_max_abs_diff"] == 0.0, "Loaded canonical ABI correctness")
    visits = raw["visits"]
    expected = [(round_["round"], tag) for round_ in planned["rounds"] for tag in round_["order"]]
    require([(r["round"], r["condition_tag"]) for r in visits] == expected, "Exact visit order; no missing/duplicate conditions")
    require(len(visits) == EXPECTED_COUNTS["conditions_per_invocation"] * 10, "Exact visits per invocation")
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
    require(len(counts) == EXPECTED_COUNTS["conditions_per_invocation"] and set(counts.values()) == {100}, "100 samples per exact condition")
    require(raw["cuda_call_counts"]["cuLaunchKernel"] == EXPECTED_COUNTS["conditions_per_invocation"] * (100 + WARMUP)
            and raw["cuda_call_counts"]["cuEventElapsedTime"] == EXPECTED_COUNTS["conditions_per_invocation"] * 100
            and raw["cuda_call_counts"]["cuEventRecord"] == EXPECTED_COUNTS["conditions_per_invocation"] * 200
            and raw["cuda_call_counts"]["cuEventSynchronize"] == EXPECTED_COUNTS["conditions_per_invocation"] * 100
            and raw["cuda_call_counts"]["cuModuleLoadData"] == len(binaries)
            and raw["cuda_call_counts"]["cuModuleUnload"] == len(binaries),
            "One kernel per event interval; warmups separate; exact module loading")
    return {"conditions": len(counts), "samples": sum(counts.values()), "gpu_uuid": env["gpu_uuid"]}


def protected_bindings():
    names = [PREREG / n for n in gc.FROZEN]
    names += [GATE / n for n in ("cohort_after_gate.json", "eligible_schedule_preview.json", "master_schedule_preview.json", "archive_bindings.json", "source_bindings.json", "hypothesis_feasibility.json")]
    return {p.relative_to(ROOT).as_posix(): sha(p.read_bytes()) for p in names}


def protected_inventory():
    tree = subprocess.check_output(["git", "ls-tree", "-rz", BASELINE, "--", "experiments/tma_reduction_layout"], cwd=ROOT)
    result = {}
    import hashlib
    for entry in tree.split(b"\0"):
        if not entry: continue
        metadata, name = entry.split(b"\t", 1)
        path = name.decode(); data = (ROOT / path).read_bytes()
        require(hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            == metadata.decode().split()[2], "Protected baseline evidence changed: " + path)
        result[path] = sha(data)
    return result


def raw_names(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and not (p.parent == root and p.name in DERIVED | {"raw_manifest.json"}))


def validate_all(root=OUT):
    cases, plan, preview = inputs()
    require(json.loads((root / "executed_schedule.json").read_text()) == plan, "Frozen executed schedule binding")
    manifest = json.loads((root / "raw_manifest.json").read_text())
    require(manifest["raw_SHA256"] == {n: sha((root / n).read_bytes()) for n in raw_names(root)}, "Immutable raw-data bytes and complete inventory")
    require(manifest["protected_SHA256"] == protected_bindings() and manifest["starting_HEAD"] == BASELINE, "Frozen prior evidence lineage")
    require(manifest["protected_prior_inventory_SHA256"] == sha(encode(protected_inventory())), "All baseline experiment bytes unchanged")
    require({p.relative_to(GATE).as_posix(): sha(p.read_bytes()) for p in GATE.rglob("*") if p.is_file() and not (p.parent == GATE and p.name in gc.DERIVED)} == json.loads((GATE / "archive_bindings.json").read_text()), "All Stage B raw artifacts unchanged")
    binaries = binary_map(cases)
    require(manifest["Stage_B_binary_SHA256"] == {k: b["archive_sha256"] for k, b in binaries.items()}, "Stage B timing binary map")
    require(manifest["protocol_SHA256"] == sha((PREREG / "protocol.json").read_bytes())
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
    with zipfile.ZipFile(root / "uploaded_source.zip") as z:
        require(set(z.namelist()) == set(provenance["source_manifest"]), "Uploaded source archive membership")
        require(all(sha(z.read(n)) == d for n, d in provenance["source_manifest"].items()), "Uploaded source bytes")
    require(source["uploaded_source_SHA256"] == sha((root / "uploaded_source.zip").read_bytes()), "Source archive SHA")
    require(source["archived_payload_sha256"] == sha((root / "archived_timing_payload.zip").read_bytes()), "Archived launch payload SHA")
    with zipfile.ZipFile(root / "archived_timing_payload.zip") as z:
        names = {b["archive_path"] for b in binaries.values()}
        names |= {str(Path(n).with_suffix(".ptx")) for n in names}
        require(set(z.namelist()) == names, "Only eligible CUBIN/PTX ABI files uploaded")
        require(all(sha(z.read(b["archive_path"])) == b["archive_sha256"]
            and sha(z.read(str(Path(b["archive_path"]).with_suffix(".ptx")))) == b["ptx_sha256"]
            for b in binaries.values()), "Payload exact CUBIN/ABI bytes")
    runtime_paths = ["experiments/tma_reduction_layout/phase4/phase5/" + n for n in
        ("run_timing.py", "timing_contract.py", "validate_timing_raw.py")]
    runtime_paths += ["experiments/tma_reduction_layout/phase4/" + n for n in ("archived_launch.py", "exact_cuda.py")]
    require(all(sha((ROOT / n).read_bytes()) == provenance["source_manifest"][n] for n in runtime_paths), "Exact timing implementation unchanged")
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
    require(sum(r["samples"] for r in actual) == EXPECTED_COUNTS["scalar_samples"] and sum(r["conditions"] for r in actual) == EXPECTED_COUNTS["invocation_conditions"], "Full scalar/condition cardinalities")
    invalid = list((root / "invalid_invocations").glob("*.json")) if (root / "invalid_invocations").exists() else []
    require(all(json.loads(p.read_text())["status"] == "INVALID_PROTOCOL_RUN" for p in invalid), "Retained invalid invocation metadata")
    environment = json.loads((root / "environment.json").read_text())
    launches = json.loads((root / "launch_bindings.json").read_text())
    require(set(environment["invocations"]) == set(launches) == {"1", "2", "3"}, "All provenance invocation identities")
    for raw in invocations:
        k = str(raw["invocation"])
        require(launches[k] == raw["loaded_binaries"]
            and environment["invocations"][k] == {"environment": raw["environment"],
                "start": raw["telemetry_start"], "end": raw["telemetry_end"],
                "dispatch_id": raw["dispatch_id"], "process_identity": raw["process_identity"]}, "Aggregate environment and launch closure")
    return {"counts": EXPECTED_COUNTS, "valid_invocations": 3, "invalid_invocations": len(invalid),
        "gpu_uuids": [r["gpu_uuid"] for r in actual], "recompilation": False,
        "replication_mode": "same-device temporal replication across distinct invocations" if len({r["gpu_uuid"] for r in actual}) == 1 else "distinct invocations on multiple physical UUIDs"}
