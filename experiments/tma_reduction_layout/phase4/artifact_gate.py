"""Offline Stage B archive contracts. No compilation, CUDA calls or timing.

Future manifests must bind real compiled CUBIN files and exact-binary resource/
occupancy evidence. Stage A only audits already archived text references.
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
from experiments.tma_reduction_layout.analyze_ir import analyze_ttgir
from experiments.tma_reduction_layout.gluon import artifact_checks as ac

ARTIFACT_KEYS = ("ttgir", "ptx", "sass", "resource.txt", "cubin", "cubin.sha256",
                 "occupancy.json", "source_manifest.json", "build_provenance.json", "compile_metadata.json", "environment.json")


def body_instructions(ptx, stage=None, minimal_output=False):
    """Complete convert/max stage, including terminal exchange, excluding store layout.

    Future compile metadata supplies SHA-bound source file/convert/max lines.
    Archival preview only may infer the adjacent convert/max source locations;
    its result must be checked against the existing audited phase annotations.
    Never choose a subsequence by matching a desired fingerprint.
    """
    loads = ac.initial_load_signature(ptx)
    if not loads:
        raise ValueError("Missing initial tile-load sequence")
    start = loads[-1]["line"] + 1
    if minimal_output:
        instructions=ac.ptx_instructions(ptx)
        stores=[i["line"] for i in instructions if i["line"]>=start and i["opcode"].startswith("st.global")]
        if not stores:raise ValueError("Minimal Gluon output store missing")
        return [i for i in instructions if start<=i["line"]<stores[0]]
    locations=[]
    for line,text in enumerate(ptx.splitlines(),1):
        match=re.match(r"\s*\.loc\s+(\d+)\s+(\d+)\s+",text)
        if match and line>=start:
            locations.append((line,int(match[1]),int(match[2])))
    if not locations:
        raise ValueError("Reduction stage source locations missing")
    if stage is None:
        _,file_id,first=locations[0]
        last=first+1
    else:
        file_id=stage["file_id"]
        first,last=min(stage["source_lines"]),max(stage["source_lines"])
        if not any(f==file_id and first<=source_line<=last for _,f,source_line in locations):
            raise ValueError("Source-bound reduction stage absent in PTX")
    boundaries=[line for line,f,source_line in locations if f==file_id and source_line>last]
    if not boundaries:
        raise ValueError("Post-reduction source boundary missing; Stage B must resolve it before timing")
    end=boundaries[0]
    if end<=start:
        raise ValueError("Invalid reduction source stage")
    return [i for i in ac.ptx_instructions(ptx) if start<=i["line"]<end]


def body_fingerprint(ptx, stage=None, minimal_output=False):
    fp = ac.fingerprint(body_instructions(ptx,stage,minimal_output))
    if not fp:
        raise ValueError("Empty reduction fingerprint")
    return fp


def equivalence(canonical, observed):
    if not canonical or not observed:
        return "REDUCTION_FINGERPRINT_MISMATCH"
    if canonical == observed:
        return "EXACT_SEQUENCE_EQUIVALENT"
    if Counter(canonical) == Counter(observed):
        return "PIPELINED_OPCODE_EQUIVALENT"
    return "REDUCTION_FINGERPRINT_MISMATCH"


def localload_encoding(expected, observed, same_semantics, payload_bytes):
    """Opcode encoding observation, backed by TTGIR semantics and byte extent.

    Equal payload widths alone do not prove identical PTX addresses/dataflow.
    Different encodings are allowed only in the isolated repeated prelude.
    """
    def extent(opcodes):
        total = 0
        for opcode in opcodes:
            match = re.fullmatch(r"ld\.shared(?:\.v([124]))?\.[bus](16|32|64)", opcode)
            if not match:
                return None
            total += int(match[1] or 1) * int(match[2]) // 8
        return total
    if not same_semantics or not observed or extent(expected) != payload_bytes or extent(observed) != payload_bytes:
        return "MISMATCH"
    return "EXACT" if expected == observed else "DIFFERENT_ENCODING"


def ttgir_contract(text, shape, warps):
    info = analyze_ttgir(text)
    local_alias = info["local_load_dest_layout"]
    shared_alias = info["local_load_src_shared_layout"]
    if not local_alias or not shared_alias:
        raise ValueError("TTGIR LocalLoad origin/destination missing")
    axes = re.findall(r'"tt\.reduce"\([^\n]+?\)\s*<\{axis\s*=\s*(\d+)\s*:\s*i32',text)
    if axes != ["1"]:
        raise ValueError("Actual artifact must contain one M-axis reduction")
    dims = "x".join(map(str,shape))
    if not re.search(r"ttg\.local_load.*?tensor<"+dims+r"xbf16,",text):
        raise ValueError("TTGIR logical tile shape mismatch")
    attrs=info["module_attributes"]
    if attrs["num_warps"] != warps or attrs["num_ctas"] != 1 or attrs["threads_per_warp"] != 32:
        raise ValueError("TTGIR execution geometry mismatch")
    blocked = info["blocked_encodings"][local_alias]
    shared = info["shared_encodings"][shared_alias]
    return {"blocked": {key:blocked[key] for key in ("sizePerThread","threadsPerWarp","warpsPerCTA","order")},
            "shared": {key:shared[key] for key in ("family","swizzlingByteWidth","transposed","elementBitWidth","rank")}}


def reproduction(canonical, observed, shape, warps, canonical_stage=None, observed_stage=None):
    """Full layout, LocalLoad and fingerprint comparison of archived single bodies."""
    layout_match = ttgir_contract(canonical["ttgir"],shape,warps) == ttgir_contract(observed["ttgir"],shape,warps)
    expected = [i["opcode"] for i in ac.initial_load_signature(canonical["ptx"])]
    actual = [i["opcode"] for i in ac.initial_load_signature(observed["ptx"])]
    canonical_fp, actual_fp = body_fingerprint(canonical["ptx"],canonical_stage),body_fingerprint(observed["ptx"],observed_stage,minimal_output=True)
    classification = equivalence(canonical_fp,actual_fp)
    resources = {name:ac.parse_resource(bundle["resource.txt"]) for name,bundle in (("canonical",canonical),("gluon",observed))}
    no_spills = all(r["local_bytes"]==r["stack_bytes"]==0 for r in resources.values())
    return {"layout_match": layout_match, "expected_canonical_localload_opcodes":expected,
            "observed_gluon_localload_opcodes":actual, "localload_match":expected==actual,
            "classification":classification, "canonical_fingerprint":canonical_fp,
            "gluon_fingerprint":actual_fp,"resources":resources,"zero_spills":no_spills,
            "structure_pass":layout_match and expected==actual and no_spills and classification!="REDUCTION_FINGERPRINT_MISMATCH"}


def repeated(canonical, bundle, shape, warps, source, canonical_stage=None):
    """Generic complete-loop contract, independent of the Phase 3 anchor sizes."""
    ptx,ttgir = bundle["ptx"],bundle["ttgir"]
    before,inside,after = ac.ttgir_loop(ttgir)
    edges = ac.ptx_backedges(ptx)
    loops = [e for e in edges if e["kind"]=="compiler_loop"]
    if len(loops)!=1 or any(e["target"]!="waitLoop" for e in edges if e["kind"]=="inline_loop"):
        raise ValueError("One runtime reduction loop plus the documented TMA poll required")
    low,high = loops[0]["start"],loops[0]["end"]
    instructions=ac.ptx_instructions(ptx)
    loop=[i for i in instructions if low<=i["line"]<=high]
    projected=ac.fingerprint(loop)
    expected=body_fingerprint(canonical["ptx"],canonical_stage)
    expected_mem=Counter(i["opcode"] for i in body_instructions(canonical["ptx"],canonical_stage) if i["opcode"].startswith(("ld.","st.","atom","red.","cp.")))
    actual_mem=Counter(i["opcode"] for i in loop if i["opcode"].startswith(("ld.","st.","atom","red.","cp.")))
    copies=[i for i in loop if i["opcode"] in ("mov.b16","mov.b32")]
    assembly=[]
    assembly_lines=[]
    current=None
    for n,line in enumerate(ptx.splitlines(),1):
        if not low<=n<=high:
            continue
        if "// begin inline asm" in line:
            if current is not None:raise ValueError("Nested inline asm markers")
            current=[]
            assembly_lines.append(n)
        elif "// end inline asm" in line:
            if current is None:raise ValueError("Unmatched inline asm end")
            assembly.append(current)
            current=None
        elif current is not None:
            current.extend(ac.ptx_instructions(line))
    input_copies=[i for i in copies if assembly_lines and i["line"]<assembly_lines[0]]
    copy_pairs=[]
    for instruction in input_copies:
        pair=re.fullmatch(r"\s*(%\w+)\s*,\s*(%\w+)\s*",instruction["operands"])
        if pair:copy_pairs.append(pair.groups())
    sass=ac.inspect_sass(bundle["sass"])
    resources=ac.parse_resource(bundle["resource.txt"])
    contract=ttgir_contract(ttgir,shape,warps)
    blocked=contract["blocked"]
    payload_elements=math.prod(s*((n+s*t*w-1)//(s*t*w)) for n,s,t,w in zip(shape,blocked["sizePerThread"],blocked["threadsPerWarp"],blocked["warpsPerCTA"]))
    same_semantics=contract==ttgir_contract(canonical["ttgir"],shape,warps)
    initial_loads=ac.initial_load_signature(ptx)
    canonical_loads=[i["opcode"] for i in ac.initial_load_signature(canonical["ptx"])]
    observed_loads=[i["opcode"] for i in initial_loads]
    encoding=localload_encoding(canonical_loads,observed_loads,same_semantics,payload_elements*2)
    checks={"one_preloop_tma":before.count("ttng.async_tma_copy_global_to_local")==1 and all("ttng.async_tma_copy_global_to_local" not in x for x in (inside,after)),
            "one_preloop_localload_no_tile_reload":before.count("ttg.local_load")==1 and all("ttg.local_load" not in x for x in (inside,after)) and actual_mem==expected_mem,
            "runtime_R_unspecialized":bool(re.search(r"do_not_specialize\s*=\s*\[(['\"])num_reductions\1\]",source)) and bool(re.search(r"scf.for .* to %num_reductions .*: i32",inside)),
            "layouts_match":ttgir_contract(ttgir,shape,warps)==ttgir_contract(canonical["ttgir"],shape,warps),
            "preloop_localload_semantics":encoding!="MISMATCH" and all(i["line"]<low for i in initial_loads),
            "complete_body_matches":equivalence(expected,projected)!="REDUCTION_FINGERPRINT_MISMATCH",
            "terminal_exchanges_inside":not any(i["line"]>high and i["opcode"].startswith(("ld.shared","ldmatrix")) for i in instructions),
            "no_accumulator_or_global_effect":not any((i["opcode"].startswith("add.") and "f32" in i["opcode"]) or i["opcode"].startswith(("st.global","atom","red.")) for i in loop),
            "empty_input_and_sink_asm":current is None and len(assembly)==2 and all(not a for a in assembly),
            "input_copies_one_to_one":len(input_copies)==payload_elements and all(i["opcode"]=="mov.b16" for i in input_copies) and len(copy_pairs)==len(input_copies) and len({p[0] for p in copy_pairs})==len(input_copies) and len({p[1] for p in copy_pairs})==len(input_copies),
            "no_sink_copy":bool(assembly_lines) and not any(i["line"]>assembly_lines[-1] for i in copies),
            "zero_spills":resources["local_bytes"]==resources["stack_bytes"]==0,
            "no_explicit_sass_mov_observed":sass["explicit_loop_mov_count"]==0}
    return {"checks":checks,"structure_pass":all(checks.values()),"classification":equivalence(expected,projected),
            "canonical_local_load_match":encoding,
            "expected_canonical_localload_opcodes":canonical_loads,
            "observed_repeated_localload_opcodes":observed_loads,
            "harness_description":"canonical reduction-body-equivalent controlled harness",
            "R0_limit":"R=0 subtraction removes fixed one-time harness differentials, but does not prove absence of interaction through register allocation, live ranges, or scheduling",
            "all_ptx_backedges":edges,"sass_audit":sass,"copy_count":len(input_copies),
            "copy_operand_sequence":[i["operands"] for i in input_copies],"copy_pairs":copy_pairs,
            "copy_opcode_sequence":[i["opcode"] for i in input_copies],
            "all_loop_movs":[{"opcode":i["opcode"],"operands":i["operands"]} for i in copies],
            "indirect_barrier_effect":"UNISOLATED_NOT_ASSUMED_ZERO"}


def body_gate_tier(single, repeated_bodies):
    """No config-name exceptions; archive/residency/runtime closure is separate."""
    all_bodies=list(single.values())+list(repeated_bodies.values())
    good=all(body["structure_pass"] for body in all_bodies)
    symmetric=repeated_bodies["default"]["copy_opcode_sequence"]==repeated_bodies["4"]["copy_opcode_sequence"]
    if not good or not symmetric:
        return "EXCLUDE_FROM_TIMING"
    if all(body["classification"]=="EXACT_SEQUENCE_EQUIVALENT" for body in all_bodies):
        return "PRIMARY"
    return "SECONDARY"


def runtime_binary_contract(records, archive):
    """Future pre-timing plan binds R=0/1 and all B to one archived binary/PTX.

    The execution log must later verify the plan; this does not collect timing.
    """
    expected={(r,b) for r in (0,1) for b in (16384,32768,65536)}
    actual=[(record["R"],record["B_RUN"]) for record in records]
    if len(actual)!=6 or set(actual)!=expected:
        raise ValueError("R=0/1 and complete B runtime plan required")
    loads=[i["opcode"] for i in ac.initial_load_signature(archive["texts"]["ptx"])]
    for record in records:
        if record["cubin_sha256"]!=archive["hashes"]["cubin"] or record["ptx_sha256"]!=archive["hashes"]["ptx"]:
            raise ValueError("Repeated binary/PTX changes across runtime R/B")
        if record["pre_loop_localload_opcodes"]!=loads:
            raise ValueError("Repeated pre-loop LocalLoad changes across runtime R/B")
    return {"R0_present":True,"fixed_binary_across_R_and_B":True,"condition_count":6}


def source_stage(metadata, source_manifest, ptx):
    """Bind complete convert/max AST lines to archived source and PTX file ID."""
    stage=metadata["reduction_stage"]
    path=ROOT/stage["source_path"]
    source=path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=stage["source_sha256"]:
        raise ValueError("Reduction source SHA mismatch")
    bound={entry["path"]:entry["SHA256"] for entry in source_manifest["files"]}
    if bound.get(stage["source_path"])!=stage["source_sha256"]:
        raise ValueError("Reduction source absent from source manifest")
    entry=next(entry for entry in source_manifest["files"] if entry["path"]==stage["source_path"])
    if entry["ptx_source_path"]!=stage["ptx_source_path"]:
        raise ValueError("Compiler source-path mapping mismatch")
    functions=[node for node in ast.walk(ast.parse(source)) if isinstance(node,ast.FunctionDef) and node.name==stage["kernel_function"]]
    if len(functions)!=1:raise ValueError("Ambiguous source kernel identity")
    operations=[node for node in ast.walk(functions[0]) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ("to","max")]
    maxima=[node for node in operations if node.func.attr=="max"]
    if len(maxima)!=1 or not any(k.arg=="axis" and isinstance(k.value,ast.Constant) and k.value.value==1 for k in maxima[0].keywords):
        raise ValueError("Source M-axis max contract missing")
    lines=sorted({node.lineno for node in operations})
    if stage["source_lines"]!=lines or len(lines)!=2:
        raise ValueError("Complete convert/max AST lines required")
    files=dict((int(n),p) for n,p in re.findall(r'\.file\s+(\d+)\s+"([^"]+)"',ptx))
    recorded=files.get(stage["file_id"])
    if not recorded or recorded!=stage["ptx_source_path"]:
        raise ValueError("PTX debug file binding mismatch")
    return stage


def exact_archive(manifest):
    """Validate future byte archives and logged exact-binary occupancy binding.

    This never calls the API itself. Query authenticity still depends on the
    archived invocation provenance, not on a recomputed resource proxy.
    """
    paths={key:ROOT/manifest["paths"][key] for key in ARTIFACT_KEYS}
    blobs={key:path.read_bytes() for key,path in paths.items()}
    if any(not b for b in blobs.values()):raise ValueError("Empty exact-binary archive")
    hashes={key:hashlib.sha256(blob).hexdigest() for key,blob in blobs.items()}
    if hashes!=manifest["SHA256"]:raise ValueError("Archive manifest hash mismatch")
    cubin_sha=hashes["cubin"]
    if not blobs["cubin"].startswith(b"\x7fELF") or blobs["cubin.sha256"].decode().strip()!=cubin_sha:
        raise ValueError("Actual ELF CUBIN and its recorded SHA are required")
    resources=ac.parse_resource(blobs["resource.txt"].decode())
    occupancy=json.loads(blobs["occupancy.json"])
    metadata=json.loads(blobs["compile_metadata.json"])
    env=json.loads(blobs["environment.json"])
    provenance=json.loads(blobs["build_provenance.json"])
    source_manifest=json.loads(blobs["source_manifest.json"])
    for entry in source_manifest["files"]:
        if hashlib.sha256((ROOT/entry["path"]).read_bytes()).hexdigest()!=entry["SHA256"]:
            raise ValueError("Archived source bundle SHA mismatch")
    if occupancy["queried_cubin_sha256"]!=cubin_sha or occupancy["resource_sha256"]!=hashes["resource.txt"]:
        raise ValueError("Occupancy must query the exact archived CUBIN/resource generation")
    if occupancy["query_api"]!="cudaOccupancyMaxActiveBlocksPerMultiprocessor":
        raise ValueError("Official occupancy API query record required")
    if occupancy["num_warps"]!=metadata["num_warps"] or occupancy["dynamic_shared_bytes"]!=metadata["dynamic_shared_bytes"]:
        raise ValueError("Occupancy launch geometry/shared-memory mismatch")
    if occupancy["resources"]!=resources or metadata["resources"]!=resources:
        raise ValueError("Exact-binary resource metadata mismatch")
    if occupancy["block_threads"]!=metadata["num_warps"]*32:
        raise ValueError("Occupancy block thread count mismatch")
    if type(metadata["dynamic_shared_bytes"]) is not int or metadata["dynamic_shared_bytes"]<0:
        raise ValueError("Invalid dynamic shared-memory size")
    if occupancy["active_warps_per_sm"]!=occupancy["blocks_per_sm"]*metadata["num_warps"]:
        raise ValueError("Occupancy warps/blocks mismatch")
    if any(type(occupancy[k]) is not int or occupancy[k]<=0 for k in ("blocks_per_sm","active_warps_per_sm","num_warps")):
        raise ValueError("Invalid occupancy geometry")
    if provenance["source_manifest_sha256"]!=hashes["source_manifest.json"] or not source_manifest["files"]:
        raise ValueError("Full source manifest/provenance binding required")
    for name in ("gpu_uuid","driver","CUDA","toolchain"):
        if not env[name]:raise ValueError(f"Missing environment field {name}")
    if not provenance["modal_image"] or not provenance["build_identity"]:
        raise ValueError("Modal image/build provenance required")
    if provenance["toolchain"]!=env["toolchain"] or occupancy["gpu_uuid"]!=env["gpu_uuid"] or occupancy["build_identity"]!=provenance["build_identity"]:
        raise ValueError("Occupancy/environment/build identity mismatch")
    exports={key:hashes[key] for key in ("ttgir","ptx","sass","resource.txt","cubin","cubin.sha256","compile_metadata.json")}
    if provenance["export_SHA256"]!=exports:
        raise ValueError("Compiler export provenance hash mismatch")
    source_stage(metadata,source_manifest,blobs["ptx"].decode())
    if resources["local_bytes"] or resources["stack_bytes"]:
        raise ValueError("Spill-free exact timing binary required")
    return {"hashes":hashes,"resources":resources,"occupancy":occupancy,"compile_metadata":metadata,"source_manifest":source_manifest,"environment":env,
            "texts":{key:blobs[key].decode() for key in ("ttgir","ptx","sass","resource.txt")}}


def future_case_gate(manifest):
    """Fail closed for all three canonical/single-reproduction/repeated bundles."""
    shape,warps=manifest["logical_shape"],manifest["num_warps"]
    audits={h:{c:exact_archive(manifest[h][c]) for c in ("default","4")}
            for h in ("canonical","gluon_reproduction","gluon_repeated")}
    for harness,candidates in audits.items():
        d,c=candidates["default"]["occupancy"],candidates["4"]["occupancy"]
        if (d["blocks_per_sm"],d["active_warps_per_sm"])!=(c["blocks_per_sm"],c["active_warps_per_sm"]):
            raise ValueError(f"RESIDENCY_MISMATCH: {harness}")
        for audit in candidates.values():
            if audit["compile_metadata"]["logical_shape"]!=shape or audit["compile_metadata"]["num_warps"]!=warps:
                raise ValueError("Exact-binary shape/warps mismatch")
    source=(ROOT/manifest["repeated_source_path"]).read_text()
    if hashlib.sha256(source.encode()).hexdigest()!=manifest["repeated_source_sha256"]:
        raise ValueError("Repeated harness source binding mismatch")
    for c in ("default","4"):
        bound={entry["path"]:entry["SHA256"] for entry in audits["gluon_repeated"][c]["source_manifest"]["files"]}
        if bound.get(manifest["repeated_source_path"])!=manifest["repeated_source_sha256"]:
            raise ValueError("Repeated source not bound to actual compilation")
    if len({audit["environment"]["gpu_uuid"] for candidates in audits.values() for audit in candidates.values()})!=1:
        raise ValueError("Case exact-binary occupancy queried on different GPUs")
    rep,reps={},{}
    for c in ("default","4"):
        canonical=audits["canonical"][c]["texts"]
        rep[c]=reproduction(canonical,audits["gluon_reproduction"][c]["texts"],shape,warps,
                            audits["canonical"][c]["compile_metadata"]["reduction_stage"],
                            audits["gluon_reproduction"][c]["compile_metadata"]["reduction_stage"])
        reps[c]=repeated(canonical,audits["gluon_repeated"][c]["texts"],shape,warps,source,
                          audits["canonical"][c]["compile_metadata"]["reduction_stage"])
    runtime={c:runtime_binary_contract(manifest["gluon_repeated"][c]["runtime_conditions"],audits["gluon_repeated"][c]) for c in ("default","4")}
    symmetric=reps["default"]["copy_opcode_sequence"]==reps["4"]["copy_opcode_sequence"]
    classification=body_gate_tier(rep,reps)
    return {"classification":classification,"pre_timing_eligible":classification in ("PRIMARY","SECONDARY"),
            "reproduction":rep,"repeated":reps,"copy_symmetry":symmetric,"runtime_contract":runtime}


if __name__=="__main__":
    if len(sys.argv)!=2:
        raise SystemExit("Offline archive audit only: artifact_gate.py FUTURE_CASE_MANIFEST.json")
    result=future_case_gate(json.loads(Path(sys.argv[1]).read_text()))
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["pre_timing_eligible"] else 1)
