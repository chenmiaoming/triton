"""
IR and PTX analysis utilities for TMA reduction layout experiments.

Extracts:
- TTGIR: BlockedEncoding attributes, shared encodings, TMA swizzle, reduce op
- PTX: Instruction counts (ld.shared, st.shared, shfl.sync, bar.sync), registers
- Derived partition metrics along reduce axis and contiguous axis
"""

import re
from typing import Any, Dict, List, Optional


def analyze_ttgir(ttgir_text: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "blocked_encodings": [],
        "shared_encodings": [],
        "reduce_ops": [],
        "descriptor_load_encoding": None,
    }

    # Find blocked layout definitions: #ttg.blocked<{sizePerThread = [...], threadsPerWarp = [...], warpsPerCTA = [...], order = [...]}>
    blocked_pattern = re.compile(
        r"#(?P<name>\w+)\s*=\s*#ttg\.blocked<\{(?P<attrs>[^}]+)\}>"
    )
    for m in blocked_pattern.finditer(ttgir_text):
        name = m.group("name")
        attrs_str = m.group("attrs")
        entry = {"name": name}
        for attr in ["sizePerThread", "threadsPerWarp", "warpsPerCTA", "order", "CTAsPerCGA"]:
            match = re.search(rf"{attr}\s*=\s*\[([^\]]+)\]", attrs_str)
            if match:
                entry[attr] = [int(x.strip()) for x in match.group(1).split(",")]
        info["blocked_encodings"].append(entry)

    # Find shared memory encodings: #ttg.shared<{...}> or #triton_gpu.shared<{...}>
    shared_pattern = re.compile(
        r"#(?P<name>\w+)\s*=\s*#ttg\.shared<\{(?P<attrs>[^}]+)\}>"
    )
    for m in shared_pattern.finditer(ttgir_text):
        info["shared_encodings"].append({
            "name": m.group("name"),
            "attrs": m.group("attrs").strip(),
        })

    # Find descriptor load ops and their output types/encodings
    # e.g.: %x = tt.descriptor_load %desc [...] : ... -> tensor<1x32x128xbf16, #blocked>
    desc_load_pattern = re.compile(
        r"tt\.descriptor_load\s+.*->\s*(?P<type>tensor<[^>]+>)"
    )
    for m in desc_load_pattern.finditer(ttgir_text):
        info["descriptor_load_type"] = m.group("type")

    # Find reduce ops
    reduce_pattern = re.compile(
        r"tt\.reduce\s+.*axis\s*=\s*(?P<axis>\d+)"
    )
    for m in reduce_pattern.finditer(ttgir_text):
        info["reduce_ops"].append({"axis": int(m.group("axis"))})

    return info


def analyze_ptx(ptx_text: str) -> Dict[str, Any]:
    lines = ptx_text.splitlines()

    stats = {
        "total_lines": len(lines),
        "ld_shared_total": 0,
        "ld_shared_details": {},
        "st_shared_total": 0,
        "st_shared_details": {},
        "shfl_total": 0,
        "shfl_details": {},
        "bar_sync_total": 0,
        "tma_bulk_total": 0,
        "max_regs": None,
    }

    reg_pattern = re.compile(r"\.reg\s+\.b\d+\s+%\w+<(\d+)>")
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue

        # Registers
        reg_match = reg_pattern.search(stripped)
        if reg_match:
            n_regs = int(reg_match.group(1))
            if stats["max_regs"] is None or n_regs > stats["max_regs"]:
                stats["max_regs"] = n_regs

        # ld.shared
        if "ld.shared" in stripped:
            stats["ld_shared_total"] += 1
            op = stripped.split()[0]
            stats["ld_shared_details"][op] = stats["ld_shared_details"].get(op, 0) + 1

        # st.shared
        if "st.shared" in stripped:
            stats["st_shared_total"] += 1
            op = stripped.split()[0]
            stats["st_shared_details"][op] = stats["st_shared_details"].get(op, 0) + 1

        # shfl.sync
        if "shfl.sync" in stripped:
            stats["shfl_total"] += 1
            op = stripped.split()[0]
            stats["shfl_details"][op] = stats["shfl_details"].get(op, 0) + 1

        # bar.sync
        if "bar.sync" in stripped:
            stats["bar_sync_total"] += 1

        # cp.async.bulk
        if "cp.async.bulk" in stripped:
            stats["tma_bulk_total"] += 1

    return stats


def compute_derived_layout_metrics(
    shape: List[int],
    size_per_thread: List[int],
    threads_per_warp: List[int],
    warps_per_cta: List[int],
    ctas_per_cga: List[int],
    reduce_axis: int,
) -> Dict[str, Any]:
    """
    Computes all analytical and derived metrics strictly from the layout specifications.
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
    cta_partitions = ctas_per_cga[reduce_axis] if ctas_per_cga else 1
    total_reduce_partitions = lane_partitions * warp_partitions * cta_partitions

    # Ownership / repetitions along reduce axis (M)
    elements_along_reduce_axis = shape[reduce_axis]
    # Each thread owns elements_along_reduce_axis / total_reduce_partitions
    thread_ownership_reduce_axis = elements_along_reduce_axis // total_reduce_partitions

    # Contiguous axis (last dimension)
    contig_axis = rank - 1
    contig_lane_partitions = threads_per_warp[contig_axis]
    contig_warp_partitions = warps_per_cta[contig_axis]
    total_contig_partitions = contig_lane_partitions * contig_warp_partitions
    thread_ownership_contig_axis = shape[contig_axis] // total_contig_partitions

    # Repetitions
    # Base distributed tile size in thread
    base_tile_thread = 1
    for spt in size_per_thread:
        base_tile_thread *= spt

    repetitions_total = elements_per_thread // base_tile_thread
    repetitions_reduce_axis = thread_ownership_reduce_axis // size_per_thread[reduce_axis]
    repetitions_contig_axis = thread_ownership_contig_axis // size_per_thread[contig_axis]

    # Communication mechanism classification
    if lane_partitions > 1:
        intra_warp_mechanism = f"shuffle ({lane_partitions} lanes per output)"
    else:
        intra_warp_mechanism = "none (1 lane per output, thread-local)"

    if warp_partitions > 1:
        cross_warp_mechanism = f"shared memory reduction ({warp_partitions} warps)"
    else:
        cross_warp_mechanism = "none (1 warp per output, warp-local)"

    if cta_partitions > 1:
        cross_cta_mechanism = f"global/inter-CTA reduction ({cta_partitions} CTAs)"
    else:
        cross_cta_mechanism = "none (single CTA)"

    return {
        "total_elements": total_elements,
        "num_threads": num_threads,
        "elements_per_thread": elements_per_thread,
        "lane_partitions_reduce_axis": lane_partitions,
        "warp_partitions_reduce_axis": warp_partitions,
        "cta_partitions_reduce_axis": cta_partitions,
        "total_partitions_reduce_axis": total_reduce_partitions,
        "thread_ownership_reduce_axis": thread_ownership_reduce_axis,
        "thread_ownership_contig_axis": thread_ownership_contig_axis,
        "repetitions_reduce_axis": repetitions_reduce_axis,
        "repetitions_contig_axis": repetitions_contig_axis,
        "repetitions_total": repetitions_total,
        "intra_warp_mechanism": intra_warp_mechanism,
        "cross_warp_mechanism": cross_warp_mechanism,
        "cross_cta_mechanism": cross_cta_mechanism,
    }
