"""
IR and PTX analysis utilities for TMA reduction layout experiments.

Provides auditable parsing strictly conforming to:
- RAW: hashes, samples, raw outputs
- OBSERVED: TTGIR layout attributes, PTX whole-kernel opcode counts, LocalLoad lowering
- DERIVED: mathematical partitioning formulas
- INFERRED: semantic phase interpretations with cited line ranges
- MEASURED: statistical timing metrics
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# TTGIR Analyzer
# ---------------------------------------------------------------------------
def analyze_ttgir(ttgir_text: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "blocked_encodings": {},
        "shared_encodings": {},
        "module_attributes": {},
        "local_load_dest_layout": None,
        "local_load_src_shared_layout": None,
        "reduce_ops": [],
        "convert_layout_ops": [],
    }

    # 1. Module attributes: "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32
    num_ctas_m = re.search(r'"ttg\.num-ctas"\s*=\s*(\d+)', ttgir_text)
    num_warps_m = re.search(r'"ttg\.num-warps"\s*=\s*(\d+)', ttgir_text)
    tpw_m = re.search(r'"ttg\.threads-per-warp"\s*=\s*(\d+)', ttgir_text)
    info["module_attributes"]["num_ctas"] = int(num_ctas_m.group(1)) if num_ctas_m else 1
    info["module_attributes"]["num_warps"] = int(num_warps_m.group(1)) if num_warps_m else 4
    info["module_attributes"]["threads_per_warp"] = int(tpw_m.group(1)) if tpw_m else 32

    # 2. Blocked layout definitions: #name = #ttg.blocked<{...}>
    blocked_pattern = re.compile(
        r"#(?P<name>\w+)\s*=\s*#ttg\.blocked<\{(?P<attrs>[^}]+)\}>"
    )
    for m in blocked_pattern.finditer(ttgir_text):
        name = m.group("name")
        attrs_str = m.group("attrs")
        entry = {"name": name}
        for attr in ["sizePerThread", "threadsPerWarp", "warpsPerCTA", "order"]:
            match = re.search(rf"{attr}\s*=\s*\[([^\]]+)\]", attrs_str)
            if match:
                entry[attr] = [int(x.strip()) for x in match.group(1).split(",")]
        info["blocked_encodings"][name] = entry

    # 3. Shared memory encodings: #name = #ttg.(nvmma_shared|swizzled_shared|shared)<{...}>
    shared_pattern = re.compile(
        r"#(?P<name>\w+)\s*=\s*#ttg\.(?P<family>nvmma_shared|swizzled_shared|shared)<\{(?P<attrs>[^}]+)\}>"
    )
    for m in shared_pattern.finditer(ttgir_text):
        name = m.group("name")
        family = m.group("family")
        attrs_str = m.group("attrs")
        entry: Dict[str, Any] = {
            "name": name,
            "family": family,
            "raw_attrs": attrs_str.strip(),
        }
        # Parse common attributes
        swizzle_m = re.search(r"swizzlingByteWidth\s*=\s*(\d+)", attrs_str)
        if swizzle_m:
            entry["swizzlingByteWidth"] = int(swizzle_m.group(1))
        trans_m = re.search(r"transposed\s*=\s*(true|false)", attrs_str)
        if trans_m:
            entry["transposed"] = trans_m.group(1) == "true"
        elem_bits_m = re.search(r"elementBitWidth\s*=\s*(\d+)", attrs_str)
        if elem_bits_m:
            entry["elementBitWidth"] = int(elem_bits_m.group(1))
        rank_m = re.search(r"rank\s*=\s*(\d+)", attrs_str)
        if rank_m:
            entry["rank"] = int(rank_m.group(1))
        vec_m = re.search(r"vec\s*=\s*(\d+)", attrs_str)
        if vec_m:
            entry["vec"] = int(vec_m.group(1))
        per_phase_m = re.search(r"perPhase\s*=\s*(\d+)", attrs_str)
        if per_phase_m:
            entry["perPhase"] = int(per_phase_m.group(1))
        max_phase_m = re.search(r"maxPhase\s*=\s*(\d+)", attrs_str)
        if max_phase_m:
            entry["maxPhase"] = int(max_phase_m.group(1))

        info["shared_encodings"][name] = entry

    # 4. Explicitly find ttg.local_load op and its destination layout alias
    # e.g.: %x_6 = ttg.local_load %x : !ttg.memdesc<1x32x128xbf16, #shared, #smem, mutable> -> tensor<1x32x128xbf16, #blocked>
    local_load_m = re.search(
        r"ttg\.local_load\s+%\w+\s*:\s*!ttg\.memdesc<[^,]+,\s*#(?P<shared_alias>\w+)[^>]*>\s*->\s*tensor<[^,]+,\s*#(?P<blocked_alias>\w+)>",
        ttgir_text,
    )
    if local_load_m:
        info["local_load_src_shared_layout"] = local_load_m.group("shared_alias")
        info["local_load_dest_layout"] = local_load_m.group("blocked_alias")

    # 5. Explicitly find tt.reduce ops and their input layout
    # e.g.: %y = "tt.reduce"(%x_fp32) <{axis = 1 : i32}> ... : (tensor<1x32x128xf32, #blocked>) -> tensor<1x128xf32, ...>
    reduce_pattern = re.compile(
        r'"tt\.reduce"\s*\([^)]*\)\s*<\{axis\s*=\s*(?P<axis>\d+)\s*:\s*i32\}>.*:\s*\([^,]+,\s*#(?P<in_layout>\w+)\)\s*->\s*(?P<ret_type>tensor<[^>]+>)',
        re.DOTALL,
    )
    for m in reduce_pattern.finditer(ttgir_text):
        info["reduce_ops"].append({
            "axis": int(m.group("axis")),
            "input_layout_alias": m.group("in_layout"),
            "return_type": m.group("ret_type"),
        })

    # 6. Convert layout ops
    convert_pattern = re.compile(
        r"ttg\.convert_layout\s+%\w+\s*:\s*tensor<[^,]+,\s*(?P<from_layout>[^>]+)>\s*->\s*tensor<[^,]+,\s*(?P<to_layout>[^>]+)>"
    )
    for m in convert_pattern.finditer(ttgir_text):
        info["convert_layout_ops"].append({
            "from_layout": m.group("from_layout").strip(),
            "to_layout": m.group("to_layout").strip(),
        })

    return info


# ---------------------------------------------------------------------------
# PTX Analyzer
# ---------------------------------------------------------------------------
def analyze_ptx(ptx_text: str) -> Dict[str, Any]:
    lines = ptx_text.splitlines()

    stats: Dict[str, Any] = {
        "total_lines": len(lines),
        "whole_kernel_opcode_counts": {
            "ld_shared_total": 0,
            "ld_shared_details": {},
            "st_shared_total": 0,
            "st_shared_details": {},
            "ldmatrix_total": 0,
            "ldmatrix_details": {},
            "shfl_total": 0,
            "shfl_details": {},
            "bar_sync_total": 0,
            "bar_warp_sync_total": 0,
            "cp_async_bulk_total": 0,
            "cp_async_bulk_details": {},
            "tensormap_total": 0,
            "mbarrier_total": 0,
            "st_global_total": 0,
            "st_global_details": {},
            "arithmetic_mix": {},
        },
        "virtual_register_declarations": {
            "max_virtual_b32_regs": None,
            "max_virtual_b16_regs": None,
            "max_virtual_b64_regs": None,
            "max_virtual_pred_regs": None,
            "note": "Virtual register declarations in PTX (.reg .b32 %r<N>), NOT physical registers per thread.",
        },
        "observed_shfl_breakdown": {
            "setup_shfl_count": 0,
            "reduction_region_shfl_count": 0,
            "details": [],
        },
        "initial_local_load": {
            "instruction": None,
            "count": 0,
            "line_range": None,
            "details": [],
        },
    }

    reg_pattern = re.compile(r"\.reg\s+\.(?P<type>b\d+|pred)\s+%\w+<(?P<count>\d+)>")

    raw_instructions: List[Tuple[int, str, str]] = []  # (line_num_1indexed, stripped_line, opcode)

    for idx, line in enumerate(lines):
        line_num = idx + 1
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue

        # Virtual register declarations
        reg_m = reg_pattern.search(stripped)
        if reg_m:
            rtype = reg_m.group("type")
            rcount = int(reg_m.group("count"))
            if rtype == "b32":
                stats["virtual_register_declarations"]["max_virtual_b32_regs"] = rcount
            elif rtype == "b16":
                stats["virtual_register_declarations"]["max_virtual_b16_regs"] = rcount
            elif rtype == "b64":
                stats["virtual_register_declarations"]["max_virtual_b64_regs"] = rcount
            elif rtype == "pred":
                stats["virtual_register_declarations"]["max_virtual_pred_regs"] = rcount
            continue

        if stripped.startswith("."):
            continue

        # Strip predicate if present: e.g. @%p2 or @!%p2
        code_part = stripped
        if code_part.startswith("@"):
            parts = code_part.split(maxsplit=1)
            if len(parts) > 1:
                code_part = parts[1]
            else:
                continue

        opcode = code_part.split()[0]
        raw_instructions.append((line_num, stripped, opcode))

    # Second pass: count opcode families and inspect phases
    wk = stats["whole_kernel_opcode_counts"]
    shared_loads: List[Tuple[int, str, str]] = []

    for line_num, stripped, opcode in raw_instructions:
        # ld.shared
        if opcode.startswith("ld.shared"):
            wk["ld_shared_total"] += 1
            wk["ld_shared_details"][opcode] = wk["ld_shared_details"].get(opcode, 0) + 1
            shared_loads.append((line_num, stripped, opcode))

        # st.shared
        elif opcode.startswith("st.shared") or "st.shared" in stripped:
            wk["st_shared_total"] += 1
            # preserve predicate in detail if present
            store_key = stripped.split()[0] if not stripped.startswith("@") else f"{stripped.split()[0]} {stripped.split()[1]}"
            wk["st_shared_details"][store_key] = wk["st_shared_details"].get(store_key, 0) + 1

        # ldmatrix
        elif opcode.startswith("ldmatrix"):
            wk["ldmatrix_total"] += 1
            wk["ldmatrix_details"][opcode] = wk["ldmatrix_details"].get(opcode, 0) + 1
            shared_loads.append((line_num, stripped, opcode))

        # shfl.sync
        elif opcode.startswith("shfl.sync"):
            wk["shfl_total"] += 1
            wk["shfl_details"][opcode] = wk["shfl_details"].get(opcode, 0) + 1
            if opcode == "shfl.sync.idx.b32":
                stats["observed_shfl_breakdown"]["setup_shfl_count"] += 1
            else:
                stats["observed_shfl_breakdown"]["reduction_region_shfl_count"] += 1
            stats["observed_shfl_breakdown"]["details"].append({
                "line": line_num,
                "opcode": opcode,
                "instruction": stripped,
            })

        # bar.sync / bar.warp.sync
        elif opcode.startswith("bar.sync"):
            wk["bar_sync_total"] += 1
        elif opcode.startswith("bar.warp.sync"):
            wk["bar_warp_sync_total"] += 1

        # cp.async.bulk
        elif "cp.async.bulk" in stripped:
            wk["cp_async_bulk_total"] += 1
            bulk_op = stripped.split()[1] if stripped.startswith("@") else opcode
            wk["cp_async_bulk_details"][bulk_op] = wk["cp_async_bulk_details"].get(bulk_op, 0) + 1

        # tensormap
        elif "tensormap" in stripped:
            wk["tensormap_total"] += 1

        # mbarrier
        elif "mbarrier" in stripped:
            wk["mbarrier_total"] += 1

        # st.global
        elif opcode.startswith("st.global"):
            wk["st_global_total"] += 1
            wk["st_global_details"][opcode] = wk["st_global_details"].get(opcode, 0) + 1

        # Arithmetic mix
        elif opcode in ["max.bf16", "max.bf16x2", "max.f32", "cvt.f32.bf16"]:
            wk["arithmetic_mix"][opcode] = wk["arithmetic_mix"].get(opcode, 0) + 1

    # 3. Detect Initial LocalLoad lowering
    # The initial LocalLoad occurs right after the TMA barrier sequence
    # Find the contiguous block of initial shared memory load instructions
    if shared_loads:
        first_load_line, first_load_str, first_load_op = shared_loads[0]
        contiguous_loads = [shared_loads[0]]
        for i in range(1, len(shared_loads)):
            curr_line, curr_str, curr_op = shared_loads[i]
            prev_line = contiguous_loads[-1][0]
            # Must be closely contiguous (within 5 lines of each other) and have matching opcode family
            if (curr_line - prev_line <= 5) and (curr_op == first_load_op):
                contiguous_loads.append(shared_loads[i])
            else:
                break

        stats["initial_local_load"] = {
            "instruction": first_load_op,
            "count": len(contiguous_loads),
            "line_range": [contiguous_loads[0][0], contiguous_loads[-1][0]],
            "raw_examples": [x[1] for x in contiguous_loads[:2]],
        }

    return stats


# ---------------------------------------------------------------------------
# Derived Metrics
# ---------------------------------------------------------------------------
def compute_derived_layout_metrics(
    shape: List[int],
    size_per_thread: List[int],
    threads_per_warp: List[int],
    warps_per_cta: List[int],
    num_ctas: int,
    reduce_axis: int,
) -> Dict[str, Any]:
    """
    Computes analytical quantities derived from the BlockedEncoding and module attributes.
    """
    rank = len(shape)
    num_threads = 1
    for t in threads_per_warp:
        num_threads *= t
    for w in warps_per_cta:
        num_threads *= w

    total_elements = 1
    for s in shape:
        total_elements *= s

    elements_per_thread = total_elements // num_threads

    # Partitioning along reduce axis
    lane_partitions = threads_per_warp[reduce_axis]
    warp_partitions = warps_per_cta[reduce_axis]
    cta_partitions = num_ctas  # Derived from module "ttg.num-ctas"
    total_reduce_partitions = lane_partitions * warp_partitions * cta_partitions

    # Elements per partition along reduce axis
    elements_along_reduce_axis = shape[reduce_axis]
    derived_reduce_elems_per_partition = elements_along_reduce_axis // total_reduce_partitions

    # Contiguous axis
    contig_axis = rank - 1
    contig_lane_partitions = threads_per_warp[contig_axis]
    contig_warp_partitions = warps_per_cta[contig_axis]
    total_contig_partitions = contig_lane_partitions * contig_warp_partitions
    derived_contig_elems_per_partition = shape[contig_axis] // total_contig_partitions

    return {
        "total_elements": total_elements,
        "num_threads": num_threads,
        "derived_elements_per_thread_total": elements_per_thread,
        "lane_partitions_reduce_axis": lane_partitions,
        "formula_lane_partitions": "threadsPerWarp[reduce_axis]",
        "warp_partitions_reduce_axis": warp_partitions,
        "formula_warp_partitions": "warpsPerCTA[reduce_axis]",
        "cta_partitions_reduce_axis": cta_partitions,
        "reason_cta_partitions": f"ttg.num-ctas = {num_ctas}, therefore cross-CTA reduction is absent",
        "total_partitions_reduce_axis": total_reduce_partitions,
        "derived_reduce_elems_per_partition": derived_reduce_elems_per_partition,
        "formula_reduce_elems": "shape[reduce_axis] // (lane_partitions * warp_partitions * cta_partitions)",
        "derived_contig_elems_per_partition": derived_contig_elems_per_partition,
    }


# ---------------------------------------------------------------------------
# SASS and Resource Usage Parser
# ---------------------------------------------------------------------------
def parse_resource_usage(res_text: str) -> Dict[str, Any]:
    if not res_text or "UNKNOWN" in res_text:
        return {
            "physical_regs_per_thread": "UNKNOWN",
            "shared_memory_bytes": "UNKNOWN",
            "local_memory_bytes": "UNKNOWN",
            "stack_bytes": "UNKNOWN",
            "raw_text": res_text,
        }

    reg_m = re.search(r"\bREG:(\d+)\b", res_text)
    shared_m = re.search(r"\bSHARED:(\d+)\b", res_text)
    local_m = re.search(r"\bLOCAL:(\d+)\b", res_text)
    stack_m = re.search(r"\bSTACK:(\d+)\b", res_text)

    return {
        "physical_regs_per_thread": int(reg_m.group(1)) if reg_m else "UNKNOWN",
        "shared_memory_bytes": int(shared_m.group(1)) if shared_m else "UNKNOWN",
        "local_memory_bytes": int(local_m.group(1)) if local_m else "UNKNOWN",
        "stack_bytes": int(stack_m.group(1)) if stack_m else "UNKNOWN",
        "raw_text": res_text.strip(),
    }
