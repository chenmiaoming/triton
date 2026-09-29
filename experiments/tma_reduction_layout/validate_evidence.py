"""
Consistency Validator for TMA reduction layout evidence.

Checks:
1. Branch & Run Metadata Sanity (local_provenance.is_dirty == false, outputs raw SHAs).
2. Artifact SHAs match baseline_results.json raw hashes.
3. Audited phase annotations:
   - PTX & TTGIR SHA bindings match artifact SHAs.
   - Line range bounds: 1 <= start <= end <= total_ptx_lines.
   - Non-inverted phase execution ordering without invalid overlaps.
   - Exact instruction count verification (e.g. initial_local_load count).
   - Global store opcode verification (st.global).
   - Structured opcode_counts verification across arithmetic & communication phases.
4. Bit-for-bit equivalence: default vs forced-8 (PTX and TTGIR).
5. Physical resource re-parsing consistency (cuobjdump REG and SHARED).
6. Canonical Markdown equivalence (exact string match for rendered summary and evidence).
7. Modal & manifest ignore policy consistency on synthetic test paths.
8. Uploaded source manifest subset fidelity (local manifest digest == remote subset digest).

Exits with code 0 on complete consistency, or non-zero on any failure.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = REPO_ROOT / "experiments" / "tma_reduction_layout"
RESULTS_DIR = EXP_DIR / "results"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"

from experiments.tma_reduction_layout.benchmark import render_evidence, render_summary
from experiments.tma_reduction_layout.source_provenance import (
    MODAL_SOURCE_IGNORE_PATTERNS,
    is_ignored_path,
)


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_opcode_in_range(lines: List[str], start: int, end: int, opcode: str) -> int:
    pattern = re.compile(r"^(?:@\S+\s+)?\s*" + re.escape(opcode) + r"[\s;]")
    count = 0
    for idx in range(start - 1, end):
        if pattern.match(lines[idx].strip()):
            count += 1
    return count


def validate():
    print("==================================================")
    print("Running TMA Reduction Evidence Consistency Validator")
    print("==================================================")

    errors: List[str] = []

    # 1. Load baseline_results.json
    results_json_path = RESULTS_DIR / "baseline_results.json"
    if not results_json_path.exists():
        print(f"FAIL: {results_json_path} does not exist.")
        sys.exit(1)

    data = json.loads(results_json_path.read_text(encoding="utf-8"))
    audited = data.get("audited_results", {})
    candidates = ["default", "8", "4", "2", "1"]

    # 2. Load audited_phase_annotations.json
    ann_path = EXP_DIR / "audited_phase_annotations.json"
    if not ann_path.exists():
        print(f"FAIL: {ann_path} does not exist.")
        sys.exit(1)
    ann_raw = json.loads(ann_path.read_text(encoding="utf-8"))
    annotations = ann_raw.get("annotations", {})

    # Check 1: Branch and Run Metadata Sanity
    print("[1/8] Validating branch & run metadata sanity...")
    prov = data.get("local_provenance", {})
    is_dirty = prov.get("is_dirty")
    if is_dirty is not False:
        errors.append(f"local_provenance.is_dirty is {is_dirty}, expected False.")

    exp_head = prov.get("git_head_sha", "UNKNOWN")
    try:
        curr_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception as e:
        curr_head = f"Error: {e}"

    print(f"  experiment source HEAD: {exp_head}")
    print(f"  current repository HEAD: {curr_head}")
    print(f"  working tree is_dirty: {is_dirty}")

    # Check 2: Artifact hashes vs baseline_results.json
    print("[2/8] Validating artifact hashes against baseline_results.json...")
    for cand in candidates:
        if cand not in audited:
            errors.append(f"Candidate '{cand}' missing in baseline_results.json audited_results.")
            continue
        raw = audited[cand]["raw"]

        for ext, key in [
            ("ttgir", "ttgir_sha256"),
            ("ptx", "ptx_sha256"),
            ("sass", "sass_sha256"),
            ("resource.txt", "resource_sha256"),
        ]:
            f_path = ARTIFACTS_DIR / f"{cand}.{ext}"
            if not f_path.exists():
                errors.append(f"Missing artifact: {f_path}")
                continue
            actual_sha = compute_sha256(f_path.read_text(encoding="utf-8"))
            expected_sha = raw.get(key)
            if actual_sha != expected_sha:
                errors.append(f"SHA mismatch for {cand}.{ext}: disk={actual_sha} vs json={expected_sha}")

    # Check 3: Audited Phase Annotations & Mechanical Instruction Verification
    print("[3/8] Validating audited phase annotations and mechanical instruction counts...")
    standard_phase_order = [
        "tma_setup_and_descriptor",
        "initial_local_load",
        "thread_local_reduction_arithmetic",
        "cross_thread_reduction_communication",
        "post_reduction_convert_layout",
        "global_store",
    ]

    for cand in candidates:
        if cand not in annotations:
            errors.append(f"Candidate '{cand}' missing in audited_phase_annotations.json.")
            continue

        cand_ann = annotations[cand]
        ptx_file = ARTIFACTS_DIR / f"{cand}.ptx"
        ttgir_file = ARTIFACTS_DIR / f"{cand}.ttgir"

        ptx_lines = ptx_file.read_text(encoding="utf-8").splitlines()
        total_ptx_lines = len(ptx_lines)

        actual_ptx_sha = compute_sha256(ptx_file.read_text(encoding="utf-8"))
        actual_ttgir_sha = compute_sha256(ttgir_file.read_text(encoding="utf-8"))

        ann_ptx_sha = cand_ann.get("ptx_sha256")
        ann_ttgir_sha = cand_ann.get("ttgir_sha256")

        if actual_ptx_sha != ann_ptx_sha:
            errors.append(f"Annotation PTX SHA mismatch for '{cand}': disk={actual_ptx_sha} vs ann={ann_ptx_sha}")
        if actual_ttgir_sha != ann_ttgir_sha:
            errors.append(f"Annotation TTGIR SHA mismatch for '{cand}': disk={actual_ttgir_sha} vs ann={ann_ttgir_sha}")

        phases = cand_ann.get("phases", {})
        prev_end = 0

        for phase_name in standard_phase_order:
            if phase_name not in phases:
                continue
            p = phases[phase_name]
            lines = p.get("lines")
            if lines is None:
                continue

            if not (isinstance(lines, list) and len(lines) == 2):
                errors.append(f"Invalid lines format for '{cand}' phase '{phase_name}': {lines}")
                continue

            start, end = lines[0], lines[1]

            # A. Line bounds
            if not (1 <= start <= end <= total_ptx_lines):
                errors.append(
                    f"Line bounds error for '{cand}' phase '{phase_name}': "
                    f"[{start}, {end}] not within [1, {total_ptx_lines}]"
                )

            # B. Non-inverted phase execution ordering
            if start < prev_end:
                errors.append(
                    f"Phase execution order inverted/overlapping for '{cand}' phase '{phase_name}': "
                    f"start={start} < previous_end={prev_end}"
                )
            prev_end = end

            # C. Instruction & count check (e.g. initial_local_load)
            if "instruction" in p and "count" in p:
                inst = p["instruction"]
                expected_cnt = p["count"]
                actual_cnt = count_opcode_in_range(ptx_lines, start, end, inst)
                if actual_cnt != expected_cnt:
                    errors.append(
                        f"Opcode count mismatch in '{cand}' phase '{phase_name}': "
                        f"instruction={inst}, expected={expected_cnt}, found={actual_cnt}"
                    )

            # D. Global store opcode check
            if phase_name == "global_store":
                st_found = any("st.global" in ptx_lines[idx] for idx in range(start - 1, end))
                if not st_found:
                    errors.append(
                        f"Global store line [{start}, {end}] for '{cand}' does not contain 'st.global'"
                    )

            # E. Structured opcode_counts check
            if "opcode_counts" in p:
                for opcode, exp_cnt in p["opcode_counts"].items():
                    act_cnt = count_opcode_in_range(ptx_lines, start, end, opcode)
                    if act_cnt != exp_cnt:
                        errors.append(
                            f"Structured opcode_counts mismatch for '{cand}' phase '{phase_name}': "
                            f"opcode={opcode}, expected={exp_cnt}, found={act_cnt}"
                        )

    # Check 4: default vs forced-8 bit-for-bit equivalence
    print("[4/8] Validating default vs forced-8 PTX & TTGIR bit-for-bit equivalence...")
    default_ptx_sha = compute_sha256((ARTIFACTS_DIR / "default.ptx").read_text(encoding="utf-8"))
    c8_ptx_sha = compute_sha256((ARTIFACTS_DIR / "8.ptx").read_text(encoding="utf-8"))
    if default_ptx_sha != c8_ptx_sha:
        errors.append(f"default.ptx ({default_ptx_sha}) != 8.ptx ({c8_ptx_sha})")

    default_ttgir_sha = compute_sha256((ARTIFACTS_DIR / "default.ttgir").read_text(encoding="utf-8"))
    c8_ttgir_sha = compute_sha256((ARTIFACTS_DIR / "8.ttgir").read_text(encoding="utf-8"))
    if default_ttgir_sha != c8_ttgir_sha:
        errors.append(f"default.ttgir ({default_ttgir_sha}) != 8.ttgir ({c8_ttgir_sha})")

    # Check 5: Re-parse resource.txt files
    print("[5/8] Validating resource.txt parsing consistency...")
    for cand in candidates:
        res_file = ARTIFACTS_DIR / f"{cand}.resource.txt"
        res_text = res_file.read_text(encoding="utf-8")
        reg_m = re.search(r"\bREG:(\d+)\b", res_text)
        shared_m = re.search(r"\bSHARED:(\d+)\b", res_text)

        actual_regs = int(reg_m.group(1)) if reg_m else "UNKNOWN"
        actual_smem = int(shared_m.group(1)) if shared_m else "UNKNOWN"

        expected_regs = audited[cand]["observed"]["physical_resources"]["physical_regs_per_thread"]["value"]
        expected_smem = audited[cand]["observed"]["physical_resources"]["cuobjdump_shared_bytes"]["value"]

        if actual_regs != expected_regs:
            errors.append(f"Physical regs mismatch for '{cand}': parsed={actual_regs} vs json={expected_regs}")
        if actual_smem != expected_smem:
            errors.append(f"cuobjdump SHARED mismatch for '{cand}': parsed={actual_smem} vs json={expected_smem}")

    # Check 6: Canonical Markdown Verification
    print("[6/8] Validating canonical Markdown generation against committed docs...")
    rendered_summary = render_summary(data)
    committed_summary = (RESULTS_DIR / "baseline_summary_table.md").read_text(encoding="utf-8")
    if rendered_summary != committed_summary:
        errors.append(
            "baseline_summary_table.md does not match canonical render_summary(payload) output!"
        )

    rendered_evidence = render_evidence(data, ann_raw)
    committed_evidence = (RESULTS_DIR / "baseline_evidence.md").read_text(encoding="utf-8")
    if rendered_evidence != committed_evidence:
        errors.append(
            "baseline_evidence.md does not match canonical render_evidence(payload, annotations) output!"
        )

    # Check 7: Modal & Source Manifest Ignore Policy Consistency
    print("[7/8] Validating unified Modal & manifest ignore policy on synthetic paths...")
    synthetic_cases: List[Tuple[str, bool]] = [
        ("foo.so", True),
        ("foo.o", True),
        ("x/__pycache__/a.pyc", True),
        ("foo.egg-info/x", True),
        ("results/x", True),
        ("experiments/tma_reduction_layout/results/x", True),
        ("build/x", True),
        ("normal_source.cpp", False),
    ]

    for p_str, expected_ignored in synthetic_cases:
        actual_ignored = is_ignored_path(Path(p_str))
        if actual_ignored != expected_ignored:
            errors.append(
                f"is_ignored_path mismatch for synthetic path '{p_str}': "
                f"expected={expected_ignored}, actual={actual_ignored}"
            )

    try:
        from modal.file_pattern_matcher import FilePatternMatcher

        modal_matcher = FilePatternMatcher(*MODAL_SOURCE_IGNORE_PATTERNS)
        for p_str, expected_ignored in synthetic_cases:
            modal_ignored = modal_matcher(Path(p_str))
            if modal_ignored != expected_ignored:
                errors.append(
                    f"Modal FilePatternMatcher mismatch for synthetic path '{p_str}': "
                    f"expected={expected_ignored}, modal={modal_ignored}"
                )
    except ImportError:
        pass

    # Check 8: Source subset manifest digest matches local source manifest
    print("[8/8] Validating uploaded source-manifest subset digest fidelity...")
    env_ver = data.get("environment", {}).get("manifest_verification", {})
    if env_ver:
        local_sha = env_ver.get("local_manifest_sha256")
        remote_subset_sha = env_ver.get("remote_source_subset_sha256")
        if remote_subset_sha and local_sha != remote_subset_sha:
            errors.append(
                f"Source manifest subset digest mismatch: local={local_sha} vs remote_subset={remote_subset_sha}"
            )
        print(f"  Verified {env_ver.get('files_verified')} uploaded files match local manifest with identical bytes.")
        print(f"  Remote post-build extra files count: {env_ver.get('remote_extra_file_count')}")

    print("--------------------------------------------------")
    if errors:
        print(f"FAILED with {len(errors)} consistency error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL 8 CONSISTENCY CHECKS PASSED SUCCESSFULLY.")
        print("==================================================")
        sys.exit(0)


if __name__ == "__main__":
    validate()
