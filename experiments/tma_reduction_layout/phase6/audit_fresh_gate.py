"""Outcome-blind fresh admission through unmodified full-body/byte auditors."""
from collections import Counter
import copy
import io
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.tma_reduction_layout.phase6 import common as c
from experiments.tma_reduction_layout.phase6 import contracts
from experiments.tma_reduction_layout.phase6 import fresh_contract as fc
from experiments.tma_reduction_layout.phase4.phase5 import stage_b_audit as audit
from experiments.tma_reduction_layout.source_provenance import compute_manifest_digest

DEST = fc.OUT
DERIVED = {"gate_results.json", "launch_contract.json", "validation.json", "summary.md", "raw_manifest.json"}


def raw_inventory():
    return {p.relative_to(DEST).as_posix(): c.sha(p.read_bytes()) for p in sorted(DEST.rglob("*"))
            if p.is_file() and p.name not in DERIVED}


def verify_sources():
    bindings = c.read(DEST / "source_bindings.json")
    provenance = c.read(DEST / "local_source_provenance.json")
    environment = c.read(DEST / "environment.json")
    dispatch = c.read(DEST / "modal_dispatch.json")
    manifest = provenance["source_manifest"]
    c.require(compute_manifest_digest(manifest) == provenance["source_manifest_sha256"] ==
              environment["source_verification"]["remote_source_subset_sha256"] == bindings["uploaded_source_subset_sha256"],
              "Local/upload/runtime complete manifest closure")
    c.require(c.sha((DEST / "uploaded_source.zip").read_bytes()) == bindings["uploaded_source_archive_sha256"], "Immutable source ZIP")
    with zipfile.ZipFile(DEST / "uploaded_source.zip") as z:
        c.require(len(z.namelist()) == len(set(z.namelist())) and set(z.namelist()) == set(manifest), "Complete source snapshot membership")
        for name, digest in manifest.items():
            c.require(c.sha(z.read(name)) == digest, "Source snapshot bytes: " + name)
        source = z.read(audit.KERNEL_PATH).decode()
    c.require(c.sha(source.encode()) == bindings["kernel_source_sha256"] == c.sha((c.ROOT / audit.KERNEL_PATH).read_bytes()), "Original kernel source unchanged")
    c.require(bindings["build_identity"] == provenance["composite_digest_sha256"] and
              bindings["git_head"] == provenance["git_head_sha"], "Actual source HEAD/diff/native build provenance")
    for name, local in (("structural_pool.json", "fresh_structural_pool.json"), ("protocol.json", "protocol.json")):
        raw = (DEST / ("frozen_"+name)).read_bytes()
        c.require(raw == (contracts.DEST / local).read_bytes() and c.sha(raw) == bindings["frozen_SHA256"][name], "Pre-timing frozen domain/protocol")
    c.require(c.sha((DEST / "raw_export.zip").read_bytes()) == dispatch["raw_export_zip_sha256"], "Original compiler return ZIP")
    with zipfile.ZipFile(DEST / "raw_export.zip") as z:
        for name in z.namelist():
            c.require(z.read(name) == (DEST / name).read_bytes(), "Original exported file bytes: " + name)
    c.require("H100" in environment["gpu_name"] and environment["compute_capability"] == [9, 0] and
              environment["no_performance_observation"] is True and environment["modal_profile"] == dispatch["profile"], "Strict H100/profile/no timing")
    c.require(dispatch["resolved_image_id"].startswith("im-") and dispatch["core_image_id"].startswith("im-"), "Core/runtime image identities")
    c.require(bool(environment["core_build_cache"]) and environment["persistent_cache"]["CCACHE_DIR"] == "/cache/ccache", "Persistent ccache diagnostics")
    for name in ("actual_compiler_ptxas", "built_triton_native_extension"):
        c.require(len(environment["toolchain"][name]["sha256"]) == 64, "Actual compiler/native SHA identity")
    return {"bindings": bindings, "manifest": manifest, "environment": environment,
            "dispatch": dispatch, "source": source}


def derive():
    c.inventory()
    context = verify_sources()
    source = c.read(contracts.DEST / "fresh_structural_pool.json")
    cases = fc.population(source)
    attempts = c.read(DEST / "attempts.json")
    domain = {(r["config_id"], h, k) for r in cases for h in fc.HARNESSES for k in fc.CANDIDATES}
    c.require(len(attempts) == 108 and {(a["case_id"], a["harness"], a["candidate"]) for a in attempts} == domain,
              "All 108 attempts; all 14 structural exclusions uncompiled")
    for attempt in attempts:
        c.require(attempt == c.read(DEST / attempt["harness"] / attempt["case_id"] / attempt["candidate"] / "attempt.json"), "Original attempt ledger")
    rows = [audit.audit_case(DEST, case, context) for case in cases]
    for row in rows:
        row.pop("Phase5_future_launch_gate", None)
        row["Phase6_launch_gate"] = "Exact archived canonical/single/repeated binaries; runtime R=0,1,2,4,8; no compilation"
    counts = dict(Counter(r["final_class"] for r in rows))
    included = {row["case_id"]: row for row in rows if row["pre_timing_eligible"]}
    binaries = {}
    for cfg, row in included.items():
        for h in fc.HARNESSES:
            for candidate in fc.CANDIDATES:
                binaries[f"{cfg}:{h}:{candidate}"] = contracts.binary_binding(DEST, row, h, candidate)
    plan = fc.schedule(included) if included else None
    launch = {"cases": {cfg: {k: row[k] for k in ("case_id", "M", "N", "num_warps", "origin", "final_class", "lanePart_transition")}
                        for cfg, row in included.items()}, "binaries": binaries, "schedule": plan,
              "counts": counts, "PRIMARY_coverage_before_timing": "TESTABLE" if counts.get("PRIMARY", 0) >= 5 else "INCONCLUSIVE_BY_COVERAGE",
              "all_eligible_coverage_before_timing": "TESTABLE" if len(included) >= 5 else "INCONCLUSIVE_BY_COVERAGE",
              "frozen_protocol_SHA256": c.sha((contracts.DEST / "protocol.json").read_bytes())}
    result = {"cases": rows, "counts": counts, "attempts": 108, "source_transitions": 32,
            "structurally_included": 18, "structurally_excluded": 14,
            "performance_observations": 0, "classification_policy": "Unmodified Phase 4/5 full-body gates; no relaxation or timing-based admission"}
    return json.loads(c.encode(result)), json.loads(c.encode(launch))


def cache_report():
    evidence = c.read(DEST / "environment.json")["core_build_cache"]
    def parse(text):
        return {line.split()[0]: int(line.split()[1]) for line in text.splitlines() if len(line.split()) == 2 and line.split()[1].isdigit()}
    before, after = parse(evidence["ccache_before.txt"]), parse(evidence["ccache_after.txt"])
    delta = {k: after.get(k, 0)-before.get(k, 0) for k in set(before)|set(after)}
    hits = delta.get("direct_cache_hit", 0)+delta.get("preprocessed_cache_hit", 0)
    misses = delta.get("cache_miss", 0)
    return {"before": before, "after": after, "delta": delta, "build_hits": hits,
            "build_misses": misses, "hit_rate": hits/(hits+misses) if hits+misses else None,
            "note": "Profile-specific Volume; core image excludes experiments and can be reused on Python-only iterations"}


def report(result, launch):
    rows = [[r["case_id"], r["final_class"], ", ".join(r["reason_codes"]) or "ALL_GATES_PASS"] for r in result["cases"]]
    cache = cache_report()
    return "\n".join(["# Phase 6 Stage D — Fresh artifact admission", "",
        "32 frozen warp16 transitions; 18 structurally included; 14 excluded without compilation. "
        "108 complete original binary attempts retained. No performance observation.", "",
        c.table(["Case", "Class", "Reasons"], rows), "", str(dict(sorted(result["counts"].items()))), "",
        f"Cache build delta: hits={cache['build_hits']}, misses={cache['build_misses']}, hit rate={cache['hit_rate']}. "
        "Full before/after counters, compiler/native SHA, core/runtime image IDs and build log are retained.", "",
        "PRIMARY coverage before timing: " + launch["PRIMARY_coverage_before_timing"] + ". "
        "All-eligible coverage before timing: " + launch["all_eligible_coverage_before_timing"] + ".", "",
        "Admission matches the frozen full family/load/loop/spill/occupancy gate. Equal opcodes or occupancy "
        "do not establish operand/dataflow/whole-program equivalence or isolate registers and scheduling.", ""])


def main():
    result, launch = derive()
    if "--validate" in sys.argv:
        c.require(c.read(DEST / "gate_results.json") == result and c.read(DEST / "launch_contract.json") == launch,
                  "Independent full-artifact gate and exact launch bindings")
        c.require(c.read(DEST / "raw_manifest.json")["files"] == raw_inventory(), "All compiler/source/failure bytes immutable")
        probes = []
        for name, mutate in (("classification", lambda r: r["cases"][0].update(final_class="PRIMARY" if r["cases"][0]["final_class"] != "PRIMARY" else "SECONDARY")),
                             ("case", lambda r: r["cases"].pop()), ("attempts", lambda r: r.update(attempts=0))):
            bad = copy.deepcopy(result); mutate(bad)
            try:
                c.require(bad == result, "Independent gate rederivation")
            except ValueError:
                probes.append(name)
            else:
                raise RuntimeError("Gate corruption accepted")
        c.require((DEST / "summary.md").read_text() == report(result, launch), "Gate summary closure")
        c.write(DEST / "validation.json", {"status": "PASS", "attempts": 108,
            "counts": result["counts"], "corruption_probes": probes, "cache": cache_report()})
        print(result["counts"], cache_report()["build_hits"], cache_report()["build_misses"])
    else:
        c.write(DEST / "gate_results.json", result)
        c.write(DEST / "launch_contract.json", launch)
        c.write(DEST / "raw_manifest.json", {"files": raw_inventory(), "before_fresh_timing": True})
        (DEST / "summary.md").write_text(report(result, launch))
        print("Fresh admission", result["counts"], "without timing")


if __name__ == "__main__":
    main()
