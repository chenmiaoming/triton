"""Offline byte/source/structural gate validation plus 13 corruption probes."""
import copy
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4 import stage_b_audit as audit
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
    exact_fixture = next(f for f in fixtures if f[6]["classification"] == "EXACT_SEQUENCE_EQUIVALENT")
    exact_case, _, _, exact_canon, _, exact_rep, _ = exact_fixture
    exact_loop = next(e for e in ac.ptx_backedges(exact_rep["texts"]["ptx"]) if e["kind"] == "compiler_loop")
    projected = [i for i in ac.ptx_instructions(exact_rep["texts"]["ptx"]) if exact_loop["start"] <= i["line"] <= exact_loop["end"] and i["opcode"].startswith(ac.FP_FAMILIES)]
    other = next(i for i in range(1, len(projected)) if ac.normalize(projected[i]) != ac.normalize(projected[0]))
    reordered = copy.deepcopy(exact_rep["texts"])
    lines = reordered["ptx"].splitlines()
    a, b = projected[0]["line"] - 1, projected[other]["line"] - 1
    lines[a], lines[b] = lines[b], lines[a]
    reordered["ptx"] = "\n".join(lines)
    reorder_audit = gate.repeated(exact_canon["texts"], reordered, exact_case["logical_shape"], exact_case["num_warps"], context["source"], exact_canon["stage"])
    probe("10 actual exact sequence reordered with same multiset becomes SECONDARY equivalence", gate.equivalence(fp, changed_fp) == "PIPELINED_OPCODE_EQUIVALENT" and reorder_audit["classification"] == "PIPELINED_OPCODE_EQUIVALENT" and reorder_audit["structure_pass"])
    base_case = audit.audit_case(root, case, context)
    loader = audit.load_bundle
    forbidden = ("performance", "timing", "speedup", "winner", "regret", "marginal_ns")
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
    probe("11 denied performance fields leave admission unchanged", remove_poison(poisoned_case) == base_case)
    changed = dict(blobs)
    del changed["occupancy.json"]
    probe("12 missing occupancy cannot establish timing eligibility", rejected(lambda: audit.exact_bundle(changed, case, "repeated", candidate, context)))
    master = copy.deepcopy(frozen["rotation_schedule.json"])
    order = master["invocations"][0]["rounds"][0]["order"]
    order[0], order[1] = order[1], order[0]
    digest = audit.sha((root / "frozen_rotation_schedule.json").read_bytes())
    check("unmodified master serialization preserves archived SHA", audit.sha(audit.json_bytes(frozen["rotation_schedule.json"])) == digest)
    probe("13 reordered master schedule fails byte binding", audit.sha(audit.json_bytes(master)) != digest)


def main():
    root = audit.OUT
    frozen = audit.frozen_files(root)
    check("all five Stage A master files byte-identical", len(frozen) == 5)
    context = audit.verify_sources(root)
    check("full source bytes/upload subset/kernel/toolchain/H100/cache/image closure", bool(context))
    check("immutable raw compiler/binary/source archive inventory", json.loads((root / "archive_bindings.json").read_text()) == audit.raw_inventory(root))
    check("current offline auditors bound separately from actual compiled source", json.loads((root / "derivation_provenance.json").read_text()) == audit.derivation_provenance(root))
    results, cohort, preview = audit.derive(root)
    for filename, expected in (("gate_results.json", results), ("cohort_after_gate.json", cohort), ("eligible_schedule_preview.json", preview)):
        check("independently recomputed " + filename, json.loads((root / filename).read_text()) == json.loads(audit.json_bytes(expected)))
    check("summary recomputed from actual artifacts", (root / "summary.md").read_text() == audit.summary_text(results, cohort, root))
    check("20 included / 10 excluded / 120 attempted / 40 variants", len(cohort["cases"]) == 20 and cohort["structural_excluded"] == 10 and results["attempts"] == 120 and results["candidate_variants"] == 40)
    for case in cohort["cases"]:
        check(case["case_id"] + " all candidate/harness evidence and final class recomputed", case["final_class"] in ("PRIMARY", "SECONDARY", "EXCLUDE_FROM_TIMING", "PENDING") and len(case["canonical"]) == len(case["single"]) == len(case["repeated"]) == 2)
        if case["pre_timing_eligible"]:
            check(case["case_id"] + " future exact canonical/repeated launch gate", case["canonical_timing_binary_sha"] == case["binary_hashes"]["canonical"] and case["repeated_timing_binary_sha"] == case["binary_hashes"]["repeated"] and case["exact_binary_closure"])
    check("stable relative-order schedule filter, never executed", preview == audit.stable_filter(frozen["rotation_schedule.json"], {c["case_id"] for c in cohort["cases"] if c["pre_timing_eligible"]}) and preview["executed"] is False)
    regression_probes(root, frozen, context)
    check("all thirteen required corruption probes PASS", len(PROBES) == 13)
    report = {"checks_passed": len(CHECKS), "checks": CHECKS, "corruption_probes": PROBES,
              "counts": results["counts"], "no_performance_observation": True}
    (root / "validator_report.json").write_bytes(audit.json_bytes(report))
    print(f"{len(CHECKS)}/{len(CHECKS)} PASS; 13/13 corruption probes PASS")


if __name__ == "__main__":
    main()
