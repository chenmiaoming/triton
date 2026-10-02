"""Offline contracts for archived compiler artifacts; never compiles or launches kernels.

Exact sequence means equality of the complete filtered reduction fingerprint.
It does not establish operand/dataflow, full PTX, SASS, or CUBIN equivalence.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

CONFIGS = ("M32_N64_w8", "M32_N128_w4")
CANDIDATES = ("default", "4")
LOAD_SIGNATURES = {
    ("M32_N64_w8", "default"): ["ld.shared.v4.b32"],
    ("M32_N64_w8", "4"): ["ld.shared.v2.b32"] * 2,
    ("M32_N128_w4", "default"): ["ld.shared.v4.b32"] * 4,
    ("M32_N128_w4", "4"): ["ld.shared.v2.b32"] * 8,
}
REPEATED_LOAD_SIGNATURES = {key: [op.replace("v2.b32", "v4.b16") for op in value]
                            for key, value in LOAD_SIGNATURES.items()}
EXTENSIONS = ("ptx", "ttgir", "sass", "resource.txt", "cubin.sha256")
FP_FAMILIES = ("shfl.", "max.", "cvt.", "st.shared", "ld.shared", "ldmatrix", "bar.sync", "selp.")


def nonempty_text(path):
    text = Path(path).read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Empty artifact: {path}")
    return text


def artifact_hashes(directory, candidate):
    hashes = {}
    for ext in EXTENSIONS:
        path = directory / f"{candidate}.{ext}"
        nonempty_text(path)
        hashes[ext] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def ptx_instructions(text):
    result = []
    for line, original in enumerate(text.splitlines(), 1):
        s = original.split("//", 1)[0].strip()
        m = re.fullmatch(r"(?:@([^\s]+)\s+)?([a-z][\w.]*)\s*(.*?)\s*;", s)
        if m:
            result.append({"line": line, "opcode": m[2], "operands": m[3], "predicate": m[1], "text": s})
    return result


def normalize(inst):
    if isinstance(inst, str):
        parsed = ptx_instructions(inst)
        if not parsed:
            return ""
        inst = parsed[0]
    op = inst["opcode"]
    if op.startswith(("shfl.", "bar.sync")):
        args = re.sub(r"%\w+", "", inst["operands"])
        return op + " " + " ".join(re.findall(r"0x[\da-fA-F]+|-?\d+", args))
    return op


def fingerprint(instructions):
    return [normalize(i) for i in instructions if i["opcode"].startswith(FP_FAMILIES)]


def ptx_backedges(text):
    labels = {m[1]: n for n, line in enumerate(text.splitlines(), 1)
              if (m := re.match(r"\s*([\w$]+):", line))}
    result = []
    for inst in ptx_instructions(text):
        target = inst["operands"]
        if inst["opcode"] in ("bra", "bra.uni") and target in labels and labels[target] <= inst["line"]:
            result.append({"start": labels[target], "end": inst["line"], "target": target,
                           "predicate": inst["predicate"],
                           "kind": "compiler_loop" if target.startswith("$L__BB") else "inline_loop"})
    return result


def ttgir_loop(text):
    lines = text.splitlines()
    starts = [n for n, line in enumerate(lines) if re.search(r"\bscf\.(for|while)\b", line)]
    if len(starts) != 1 or "scf.for" not in lines[starts[0]]:
        raise ValueError("Expected exactly one scf.for and no other structured runtime loops")
    start = starts[0]
    depth = 0
    end = None
    for n in range(start, len(lines)):
        depth += lines[n].count("{") - lines[n].count("}")
        if depth == 0:
            end = n
            break
    if end is None:
        raise ValueError("Unterminated TTGIR runtime loop")
    return "\n".join(lines[:start]), "\n".join(lines[start:end + 1]), "\n".join(lines[end + 1:])


def layout_attrs(text, family):
    matches = re.findall(r"#ttg\." + family + r"<[^>]+>", text)
    if not matches:
        raise ValueError(f"Missing {family} layout")
    return matches[0]


def elements_per_thread(shape, layout):
    """Shape-aware blocked-layout storage extent, including replicated small tiles.

    Each dimension has sizePerThread * ceil(shape / (spt * lanes * warps))
    elements per thread. Count vector width once, including partial layout tiles.
    """
    fields = {}
    for name in ("sizePerThread", "threadsPerWarp", "warpsPerCTA"):
        m = re.search(name + r"\s*=\s*\[([^]]+)\]", layout)
        if not m:
            raise ValueError(f"Missing blocked field {name}")
        fields[name] = [int(x) for x in m[1].split(",")]
    if any(len(v) != len(shape) for v in fields.values()) or any(x <= 0 for v in fields.values() for x in v):
        raise ValueError("Invalid blocked-layout dimensions")
    counts = []
    for n, s, t, w in zip(shape, fields["sizePerThread"], fields["threadsPerWarp"], fields["warpsPerCTA"]):
        if type(n) is not int or n <= 0:
            raise ValueError("Invalid logical shape")
        tile = s * t * w
        counts.append(s * ((n + tile - 1) // tile))
    return counts


def parse_resource(text):
    result = {}
    for field, key in (("REG", "num_regs"), ("LOCAL", "local_bytes"), ("STACK", "stack_bytes"),
                       ("SHARED", "static_smem_bytes")):
        values = re.findall(r"\b" + field + r":(\d+)\b", text)
        if len(values) != 1:
            raise ValueError(f"Expected one resource field {field}")
        result[key] = int(values[0])
    return result


def canonical_fingerprint(base, cfg, candidate):
    ann = json.loads(nonempty_text(base / "phase3_audited_annotations.json"))
    phases = ann["configurations"][cfg][candidate]["phases"]
    start = phases["thread_local_reduction_arithmetic"]["lines"][0]
    end = phases["cross_warp_reduction_communication"]["lines"][1]
    text = nonempty_text(base / "results/phase3/fixed_binary_artifacts/canonical" / cfg / f"{candidate}.ptx")
    fp = fingerprint([i for i in ptx_instructions(text) if start <= i["line"] <= end])
    if not fp:
        raise ValueError("Empty canonical fingerprint")
    return fp


def initial_load_signature(text):
    insts = ptx_instructions(text)
    starts = [n for n, i in enumerate(insts) if i["opcode"].startswith("mbarrier.inval")]
    if len(starts) != 1:
        raise ValueError("Expected one mbarrier invalidation")
    loads = []
    for inst in insts[starts[0] + 1:]:
        if inst["opcode"].startswith(("max.", "cvt.", "bar.sync")):
            break
        if inst["opcode"].startswith(("ld.shared", "ldmatrix")):
            loads.append(inst)
    return loads


def inspect_sass(text):
    insts = []
    for line in text.splitlines():
        m = re.match(r"\s*/\*([\da-fA-F]+)\*/\s+(.*?);", line)
        if m:
            insts.append((int(m[1], 16), m[2].strip()))
    if not insts or len({a for a, _ in insts}) != len(insts):
        raise ValueError("Missing SASS instructions or duplicate addresses")
    edges = []
    repetition = []
    for address, inst in insts:
        m = re.search(r"\bBRA(?:\.\w+)*\s+0x([\da-fA-F]+)", inst)
        if not m or int(m[1], 16) > address:
            continue
        target = int(m[1], 16)
        body = [(a, i) for a, i in insts if target <= a <= address]
        conditional = inst.startswith("@")
        is_reduction = conditional and any(re.search(r"\b(FMNMX|SHFL)\b", i) for _, i in body)
        kind = "reduction" if is_reduction else ("terminal_self_loop" if target == address else
               "barrier_poll" if any("TRYWAIT" in i for _, i in body) and conditional else "return_edge")
        edge = {"start_addr": target, "end_addr": address, "kind": kind, "instruction": inst}
        edges.append(edge)
        if is_reduction:
            repetition.append(body)
    if len(repetition) != 1:
        raise ValueError(f"Expected one conditional SASS reduction backedge, found {len(repetition)}")
    loop = repetition[0]
    movs = [(f"0x{a:x}", i) for a, i in loop if re.search(r"(?:^|\s)MOV(?:\.\w+)*\s", i) or "IMAD.MOV" in i]
    return {"all_backward_edges": edges, "loop_start_addr": f"0x{loop[0][0]:x}",
            "loop_end_addr": f"0x{loop[-1][0]:x}", "loop_sass_instruction_count": len(loop),
            "loop_body_mov_instructions": movs,
            "explicit_loop_mov_count": len(movs),
            "sass_barrier_classification": "NO_EXPLICIT_LOOP_MOV_OBSERVED" if not movs else "EXPLICIT_LOOP_MOV_OBSERVED",
            "sass_barrier_observation": "PTX tied copies exist; no extra explicit SASS MOV was observed." if not movs else "Explicit MOV observed in the repetition region.",
            "attribution_limit": "No proof of zero indirect live-range, register-assignment, or scheduling effects."}


def inspect_repeated(base, directory, cfg, candidate, raw, require_r_hashes=True):
    """Re-extract actual artifacts. A failed structural check cannot unlock timing."""
    texts = {ext: nonempty_text(directory / f"{candidate}.{ext}") for ext in EXTENSIONS}
    ptx, ttgir = texts["ptx"], texts["ttgir"]
    ins = ptx_instructions(ptx)
    edges = ptx_backedges(ptx)
    compiler_edges = [e for e in edges if e["kind"] == "compiler_loop"]
    if len(compiler_edges) != 1:
        raise ValueError(f"Expected one PTX reduction loop; found {len(compiler_edges)}")
    loop_edge = compiler_edges[0]
    lo, hi = loop_edge["start"], loop_edge["end"]
    before, inside, after = ttgir_loop(ttgir)
    loop = [i for i in ins if lo <= i["line"] <= hi]
    pre_loads = [i for i in ins if i["line"] < lo and i["opcode"].startswith(("ld.shared", "ldmatrix"))]
    post_loads = [i for i in ins if i["line"] > hi and i["opcode"].startswith(("ld.shared", "ldmatrix"))]
    canon = canonical_fingerprint(base, cfg, candidate)
    fp = fingerprint(loop)
    kind = "EXACT_SEQUENCE_EQUIVALENT" if fp == canon else (
        "PIPELINED_OPCODE_EQUIVALENT" if Counter(fp) == Counter(canon) else "MISMATCH")
    fp_lines = [i["line"] for i in loop if i["opcode"].startswith(FP_FAMILIES)]
    copies = [i for i in loop if i["opcode"] == "mov.b16"]
    copy_operands = [tuple(re.findall(r"%[\w]+", i["operands"])) for i in copies]
    copies_one_to_one = all(len(pair) == 2 for pair in copy_operands) and len({p[0] for p in copy_operands}) == len(copies) and len({p[1] for p in copy_operands}) == len(copies)
    memory_ops = [i["opcode"] for i in loop if i["opcode"].startswith(("ld.", "st.", "atom", "red.", "cp."))]
    expected_memory = [op.split()[0] for op in canon if op.startswith(("ld.", "st.", "atom", "red.", "cp."))]
    asm_regions = []
    current = None
    for line, text in enumerate(ptx.splitlines(), 1):
        if lo <= line <= hi:
            if "// begin inline asm" in text:
                if current is not None:
                    raise ValueError("Nested inline asm marker")
                current = {"start": line, "instructions": []}
            elif "// end inline asm" in text:
                if current is None:
                    raise ValueError("Unmatched inline asm marker")
                current["end"] = line
                asm_regions.append(current)
                current = None
            elif current is not None:
                current["instructions"].extend(ptx_instructions(text))
    if current is not None:
        raise ValueError("Unterminated inline asm marker")
    parsed = parse_resource(texts["resource.txt"])
    expected_resources = raw["resources"]
    resource_match = all(type(expected_resources[k]) is int and parsed[k] == expected_resources[k] for k in parsed)
    sha = texts["cubin.sha256"].strip()
    cubin_match = bool(re.fullmatch(r"[\da-f]{64}", sha)) and sha == raw["cubin_sha256"]
    if require_r_hashes:
        hashes = raw["r_cubin_hashes"]
        cubin_match = cubin_match and set(hashes) == {"1", "2", "4", "8"} and all(v == sha for v in hashes.values())
    canon_ttgir = nonempty_text(base / "results/phase3/fixed_binary_artifacts/canonical" / cfg / f"{candidate}.ttgir")
    sass = inspect_sass(texts["sass"])
    checks = {
        "tma_once_before_loop": before.count("ttng.async_tma_copy_global_to_local") == 1 and
            not any("ttng.async_tma_copy_global_to_local" in s for s in (inside, after)),
        "local_load_once_before_loop": before.count("ttg.local_load") == 1 and not any("ttg.local_load" in s for s in (inside, after)),
        "runtime_r": bool(re.search(r"scf.for .* to %num_reductions .*: i32", inside)) and bool(re.search(r"%num_reductions: i32", ttgir)),
        "one_runtime_loop": len(compiler_edges) == 1 and all(e["target"] == "waitLoop" for e in edges if e["kind"] == "inline_loop"),
        "layouts": all(layout_attrs(ttgir, f) == layout_attrs(canon_ttgir, f) for f in ("blocked", "nvmma_shared")),
        "initial_load_signature": [i["opcode"] for i in pre_loads] == REPEATED_LOAD_SIGNATURES[cfg, candidate],
        "complete_reduction_fingerprint": kind != "MISMATCH" and (cfg != "M32_N64_w8" or kind == "EXACT_SEQUENCE_EQUIVALENT"),
        "complete_memory_signature": Counter(memory_ops) == Counter(expected_memory) and not post_loads,
        "no_extra_float_add_or_global_effect": not any(i["opcode"].startswith("add.") and "f32" in i["opcode"] or
            i["opcode"].startswith(("st.global", "atom", "red.")) for i in loop),
        "barriers": len(asm_regions) == 2 and all(not a["instructions"] for a in asm_regions) and
            len(copies) == (8 if cfg == "M32_N64_w8" else 32) and copies_one_to_one and all(i["line"] < asm_regions[0]["start"] for i in copies),
        "no_explicit_sass_mov": sass["explicit_loop_mov_count"] == 0,
        "resources": resource_match and parsed["local_bytes"] == parsed["stack_bytes"] == 0,
        ("single_recorded_binary" if require_r_hashes else "recorded_candidate_binary_binding"): cubin_match,
    }
    return {"checks": checks, "all_checks_passed": all(checks.values()), "all_ptx_backedges": edges,
            "loop_start": lo, "loop_end": hi, "fingerprint": fp, "canonical_fingerprint": canon,
            "reduction_match_type": kind, "red_start": min(fp_lines) if fp_lines else None,
            "red_end": max(fp_lines) if fp_lines else None,
            "initial_load_signature": [i["opcode"] for i in pre_loads],
            "loop_memory_signature": memory_ops, "expected_loop_memory_signature": expected_memory,
            "post_loop_shared_loads": [i["text"] for i in post_loads],
            "copy_signature": [i["opcode"] for i in copies], "copy_operands": copy_operands,
            "copy_operands_one_to_one": copies_one_to_one,
            "resources_from_artifact": parsed, "sass_audit": sass,
            "artifact_hashes": artifact_hashes(directory, candidate)}
