"""Lossless instruction/operand inventories and descriptive whole-kernel context."""
from collections import Counter
import copy
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.tma_reduction_layout.phase7 import common as c
from experiments.tma_reduction_layout.gluon import artifact_checks as ac

DEST = c.OUT / "stage_a"


def inspect(directory):
    ptx, ir, sass = [(directory / ("kernel." + ext)).read_text() for ext in ("ptx", "ttgir", "sass")]
    instructions = ac.ptx_instructions(ptx)
    names = {}
    normalized = []
    for inst in instructions:
        def rename(match):
            value = match[0]
            return names.setdefault(value, f"%v{len(names)}")
        normalized.append(re.sub(r"%\w+", rename, inst["text"]))
    machine = []
    for line_number, line in enumerate(sass.splitlines(), 1):
        m = re.match(r"\s*/\*([0-9a-f]+)\*/\s+(?:(@!?\w+)\s+)?([A-Z][A-Z0-9_.]*)\s*(.*?)\s*;\s*/\*(.*?)\*/", line)
        if m:
            machine.append({"line": line_number, "address": m[1], "predicate": m[2], "opcode": m[3],
                            "operands": m[4], "instruction_word": m[5],
                            "following_control_word_line": sass.splitlines()[line_number] if line_number < len(sass.splitlines()) else ""})
    c.require(machine and instructions, "Complete nonempty machine and PTX inventory")
    effects = [i for i in instructions if i["opcode"].startswith(("tensormap.", "fence.", "cp.async.", "mbarrier.", "bar.", "ld.shared", "st.shared", "st.global"))]
    ssa = [{"line": i, "text": line.strip(), "SSA_tokens": re.findall(r"%[\w]+", line)}
           for i, line in enumerate(ir.splitlines(), 1) if re.search(r"ttg\.convert_layout|tt\.store|tt\.reduce|tt\.make_tensor_descriptor|ttg\.local_load", line)]
    resources = ac.parse_resource((directory / "kernel.resource.txt").read_text())
    return {"file_SHA256": {p.name: c.sha(p.read_bytes()) for p in sorted(directory.iterdir()) if p.is_file()},
            "PTX_all_instructions_with_operands": instructions, "PTX_first_occurrence_register_renaming": normalized,
            "SASS_all_instructions_with_operands_and_control_word_lines": machine,
            "PTX_opcode_counts": dict(sorted(Counter(i["opcode"] for i in instructions).items())),
            "SASS_opcode_counts": dict(sorted(Counter(i["opcode"] for i in machine).items())),
            "descriptor_sync_shared_store_instructions": effects, "IR_operations_with_SSA_tokens": ssa,
            "output_convert_layout_count": len(re.findall(r"ttg\.convert_layout", ir)),
            "device_descriptor_construction": "tt.make_tensor_descriptor" in ir,
            "resources": resources, "occupancy": c.read(directory / "occupancy.json"),
            "PTX_operand_sensitive_SHA256": c.sha(c.encode(normalized)),
            "limitations": "Static lossless operands/control-word inventories, not CFG/liveness analysis, instruction latency attribution or semantic equivalence proof."}


def derive():
    protected = c.inventory()
    prior = c.read(c.PRIOR / "stage_d/results.json")
    rows = {}
    for cfg, result in prior["cases"].items():
        binaries = {h + ":" + k: inspect(c.PRIOR / "stage_d_gate" / h / cfg / k)
                    for h in ("canonical", "single", "repeated") for k in c.CANDIDATES}
        rows[cfg] = {"class": result["final_class"], "G": result["G"], "S": result["S"],
                     "G_minus_S": result["G_minus_S"], "all_binaries": binaries,
                     "whole_PTX_operand_identity_canonical_vs_single": {
                         k: binaries["canonical:" + k]["PTX_operand_sensitive_SHA256"] ==
                         binaries["single:" + k]["PTX_operand_sensitive_SHA256"] for k in c.CANDIDATES}}
    return json.loads(c.encode({"role": "OUTCOME_INFORMED_OFFLINE_DIAGNOSTIC", "cases": rows,
        "diagnostic_PRIMARY_cases": c.diagnostic_cases(), "binary_count": len(rows)*6,
        "protected_files": len(protected), "protected_inventory_SHA256": c.sha(c.encode(protected)),
        "candidate_interventions": ["Host versus device descriptor construction within one Gluon program family",
                                    "Inherited reduction output layout versus actual canonical one-dimensional store layout",
                                    "Two-factor interaction in layout gap; no additive cost assumption"],
        "limits": ["Existing PRIMARY is opcode sequence equivalence of the reduction stage only.",
                   "Operands, full-program control/dataflow, schedules and CUBIN equivalence remain separate questions.",
                   "Static differences and G-S are descriptive; no component latency or causal share inferred.",
                   "All 17 cases retained; all nine PRIMARY form the intervention diagnostic cohort.",
                   "No GPU, timing, compiler execution or new model fitting in Stage A."]}))


def report(result):
    rows = []
    for cfg, row in result["cases"].items():
        b = row["all_binaries"]
        rows.append([cfg, row["class"], row["G_minus_S"]["mean"],
                     [b["canonical:"+k]["output_convert_layout_count"] for k in c.CANDIDATES],
                     [b["single:"+k]["output_convert_layout_count"] for k in c.CANDIDATES],
                     [b["canonical:"+k]["device_descriptor_construction"] for k in c.CANDIDATES]])
    return "\n".join(["# Phase 7 Stage A — Existing complete-kernel audit", "",
        "All 17 eligible Phase 6 cases / 102 actual binary bundles retained. "
        "Complete parsed PTX operands/predicates, SASS operands/instruction words/control-word lines, "
        "descriptor/synchronization/shared/store operations, IR SSA-token records and resources are in results.json.", "",
        c.table(["Case", "Class", "G−S ns/additional CTA", "Canonical output conversions [default,4]", "Single conversions", "Canonical device descriptor"], rows), "",
        "These are static instruction inventories; no liveness/CFG solver or whole-program semantic-equivalence proof is claimed. "
        "The descriptor and output-layout differences motivate independently controlled Gluon interventions. "
        "The all-nine PRIMARY diagnostic cohort is outcome-informed; it is not a new confirmatory cohort.", "",
        *["- " + s for s in result["limits"]], ""])


def main():
    expected = derive()
    if "--validate" in sys.argv:
        stored = c.read(DEST / "results.json")
        c.require(stored == expected, "All raw byte/operand/context observations independently rederive")
        probes = []
        for name, mutate in (
            ("operand", lambda r: next(iter(r["cases"].values()))["all_binaries"]["canonical:default"]["PTX_all_instructions_with_operands"][0].update(operands="CORRUPT")),
            ("case", lambda r: r["cases"].pop(next(iter(r["cases"])))),
            ("cohort", lambda r: r["diagnostic_PRIMARY_cases"].pop(next(iter(r["diagnostic_PRIMARY_cases"]))))):
            bad = copy.deepcopy(stored); mutate(bad)
            c.require(bad != expected, "Offline corruption rejected: " + name); probes.append(name)
        c.require((DEST / "summary.md").read_text() == report(expected), "Offline report closure")
        result = {"status": "PASS", "binary_count": expected["binary_count"], "cases": 17, "PRIMARY": 9,
                  "corruption_probes": probes, "old_files_unchanged": expected["protected_files"], "no_GPU": True}
        c.write(DEST / "validation.json", result); print(result)
    else:
        c.write(DEST / "results.json", expected)
        (DEST / "summary.md").write_text(report(expected)); print("Stage A completed without GPU/compilation")


if __name__ == "__main__":
    main()
