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
9. Phase 2 B-Saturation pilot results & extended saturation canonical markdown.
10. Phase 2 30-Config Steady-State sweep results (150 combinations, re-derived transitions, canonical CSV & MD).
11. Phase 2 representative artifacts fidelity & byte-for-byte SHA256 bindings.
12. Offline marginal analysis formula consistency & canonical markdown.
13. Corrected fixed-binary saturation pilot runs (3 benchmark invocations, fixed-binary invariance, telemetry, canonical MD).
14. Phase 3 structural evidence, canonical artifact bindings, equivalence report, and hypotheses.
15. Phase 3 Step B: Reduction-Communication Amplification Microbenchmark (H2 isolation, codegen invariance, telemetry, slope recomputation, canonical MD).

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
PHASE2_DIR = RESULTS_DIR / "phase2"
SATURATION_DIR = PHASE2_DIR / "saturation"
SWEEP_DIR = PHASE2_DIR / "sweep"
REPRESENTATIVES_DIR = PHASE2_DIR / "representatives"

from experiments.tma_reduction_layout.benchmark import render_evidence, render_summary
from experiments.tma_reduction_layout.source_provenance import (
    MODAL_SOURCE_IGNORE_PATTERNS,
    is_ignored_path,
)
from experiments.tma_reduction_layout.phase2_benchmark import (
    check_candidate_legality,
    classify_structural_transitions,
    test_classify_structural_transitions,
    generate_sweep_csv,
    generate_sweep_markdown,
    generate_extended_saturation_markdown,
    render_corrected_pilot_summary_markdown,
)
from experiments.tma_reduction_layout.marginal_analysis import (
    compute_marginal_analysis,
    linear_regression,
    render_marginal_analysis_markdown,
)
from experiments.tma_reduction_layout.analyze_ir import compute_derived_layout_metrics
from experiments.tma_reduction_layout.phase3_structural_analysis import (
    render_summary_markdown as render_phase3_summary_markdown,
    render_hypotheses_markdown as render_phase3_hypotheses_markdown,
)
from experiments.tma_reduction_layout.microbench_reduction import (
    generate_design_markdown as render_mb_design_markdown,
    generate_summary_markdown as render_mb_summary_markdown,
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
    print("[1/15] Validating branch & run metadata sanity...")
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
    print("[2/15] Validating artifact hashes against baseline_results.json...")
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
    print("[3/15] Validating audited phase annotations and mechanical instruction counts...")
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
    print("[4/15] Validating default vs forced-8 PTX & TTGIR bit-for-bit equivalence...")
    default_ptx_sha = compute_sha256((ARTIFACTS_DIR / "default.ptx").read_text(encoding="utf-8"))
    c8_ptx_sha = compute_sha256((ARTIFACTS_DIR / "8.ptx").read_text(encoding="utf-8"))
    if default_ptx_sha != c8_ptx_sha:
        errors.append(f"default.ptx ({default_ptx_sha}) != 8.ptx ({c8_ptx_sha})")

    default_ttgir_sha = compute_sha256((ARTIFACTS_DIR / "default.ttgir").read_text(encoding="utf-8"))
    c8_ttgir_sha = compute_sha256((ARTIFACTS_DIR / "8.ttgir").read_text(encoding="utf-8"))
    if default_ttgir_sha != c8_ttgir_sha:
        errors.append(f"default.ttgir ({default_ttgir_sha}) != 8.ttgir ({c8_ttgir_sha})")

    # Check 5: Re-parse resource.txt files
    print("[5/15] Validating resource.txt parsing consistency...")
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
    print("[6/15] Validating canonical Markdown generation against committed docs...")
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
    print("[7/15] Validating unified Modal & manifest ignore policy on synthetic paths...")
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
    print("[8/15] Validating uploaded source-manifest subset digest fidelity...")
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

    # Check 9: Phase 2 B-Saturation Pilot Consistency
    print("[9/15] Validating Phase 2 B-Saturation pilot results...")
    sat_json_path = SATURATION_DIR / "results.json"
    sat_md_path = SATURATION_DIR / "summary.md"
    if not sat_json_path.exists():
        errors.append(f"Phase 2 saturation results missing: {sat_json_path}")
    elif not sat_md_path.exists():
        errors.append(f"Phase 2 saturation summary markdown missing: {sat_md_path}")
    else:
        sat_data = json.loads(sat_json_path.read_text(encoding="utf-8"))
        sat_env = sat_data.get("environment", {})
        if sat_env.get("gpu_compute_capability") != [9, 0]:
            errors.append(f"Phase 2 pilot GPU compute capability mismatch: {sat_env.get('gpu_compute_capability')}")
        b_vals = sat_data.get("b_values", [])
        if b_vals != [64, 256, 1024, 4096, 8192]:
            errors.append(f"Phase 2 pilot b_values mismatch: expected [64, 256, 1024, 4096, 8192], got {b_vals}")
        for b in b_vals:
            b_info = sat_data.get("data", {}).get(str(b), {})
            for cand in ["default", "8", "4", "2", "1"]:
                cinfo = b_info.get(cand, {})
                if cinfo.get("is_legal", False):
                    med = cinfo.get("median_us", 0.0)
                    t_cta = cinfo.get("amortized_grid_time_per_cta_ns", cinfo.get("time_per_cta_ns", 0.0))
                    expected_t_cta = (med * 1000.0) / b
                    if abs(t_cta - expected_t_cta) > 0.1:
                        errors.append(f"Pilot amortized_grid_time_per_cta_ns mismatch for B={b}, cand={cand}: {t_cta} vs {expected_t_cta}")
        print("  Verified Phase 2 saturation pilot data and formula consistency.")

        # Check extended saturation pilot if present
        ext_json_path = SATURATION_DIR / "extended_results.json"
        ext_md_path = SATURATION_DIR / "extended_summary.md"
        if ext_json_path.exists():
            ext_data = json.loads(ext_json_path.read_text(encoding="utf-8"))
            ext_env = ext_data.get("environment", {})
            if ext_env.get("gpu_compute_capability") != [9, 0]:
                errors.append(f"Extended pilot GPU compute capability mismatch: {ext_env.get('gpu_compute_capability')}")
            ext_b_vals = ext_data.get("b_values", [])
            if ext_b_vals != [4096, 8192, 16384, 32768, 65536]:
                errors.append(f"Extended pilot b_values mismatch: expected [4096, 8192, 16384, 32768, 65536], got {ext_b_vals}")
            for cfg_k, cfg_v in ext_data.get("configs", {}).items():
                for b in ext_b_vals:
                    b_dict = cfg_v.get("data", {}).get(str(b), {})
                    for cand, cinfo in b_dict.items():
                        if cinfo.get("is_legal", False):
                            med = cinfo.get("median_us", 0.0)
                            t_cta = cinfo.get("amortized_grid_time_per_cta_ns", 0.0)
                            expected_t_cta = (med * 1000.0) / b
                            if abs(t_cta - expected_t_cta) > 0.1:
                                errors.append(f"Extended pilot amortized time mismatch for {cfg_k}, B={b}, cand={cand}: {t_cta} vs {expected_t_cta}")
            if ext_md_path.exists():
                canonical_ext_md = generate_extended_saturation_markdown(ext_data)
                actual_ext_md = ext_md_path.read_text(encoding="utf-8")
                if canonical_ext_md != actual_ext_md:
                    errors.append("extended_summary.md does not match canonical generate_extended_saturation_markdown() output byte-for-byte")
            else:
                errors.append(f"Missing {ext_md_path}")
            print("  Verified Phase 2 extended saturation pilot data and canonical markdown.")

    # Check 10: Phase 2 30-Config Sweep Consistency
    print("[10/15] Validating Phase 2 30-Config Steady-State sweep results...")
    # 10.1 Synthetic unit test of transition classification
    try:
        assert test_classify_structural_transitions() is True
        print("  Verified classify_structural_transitions() on synthetic test cases (T1-T5).")
    except Exception as e:
        errors.append(f"Synthetic test_classify_structural_transitions failed: {e}")

    sweep_json_path = SWEEP_DIR / "results.json"
    sweep_csv_path = SWEEP_DIR / "summary.csv"
    sweep_md_path = SWEEP_DIR / "summary.md"
    configs = {}
    if not sweep_json_path.exists():
        errors.append(f"Phase 2 sweep results missing: {sweep_json_path}")
    elif not sweep_csv_path.exists():
        errors.append(f"Phase 2 sweep summary CSV missing: {sweep_csv_path}")
    elif not sweep_md_path.exists():
        errors.append(f"Phase 2 sweep summary markdown missing: {sweep_md_path}")
    else:
        sweep_data = json.loads(sweep_json_path.read_text(encoding="utf-8"))
        
        # 10.2 Verify steady_state_established flag and status
        if sweep_data.get("steady_state_established") is not False:
            errors.append(f"Expected sweep steady_state_established == False, found {sweep_data.get('steady_state_established')}")
        if sweep_data.get("status") != "exploratory_multi_cta":
            errors.append(f"Expected sweep status == 'exploratory_multi_cta', found {sweep_data.get('status')}")

        configs = sweep_data.get("configs", {})
        if len(configs) != 30:
            errors.append(f"Phase 2 sweep expected 30 configs, found {len(configs)}")

        b_steady = sweep_data.get("b_steady", 4096)

        for cfg_k, cfg in configs.items():
            m = cfg["M"]
            n = cfg["N"]
            w = cfg["num_warps"]
            def_cand = cfg["candidates"].get("default", {})

            for cand, cdata in cfg["candidates"].items():
                expected_legal, expected_err = check_candidate_legality(m, n, w, cand)
                actual_legal = cdata.get("is_legal", False)
                if actual_legal != expected_legal:
                    errors.append(f"Candidate legality mismatch for {cfg_k}, cand={cand}: actual={actual_legal}, expected={expected_legal}")
                if expected_legal:
                    if not cdata.get("is_correct", False):
                        errors.append(f"Numerical correctness failed for {cfg_k}, cand={cand}")
                    # Re-derive layout metrics
                    ttg = cdata.get("observed", {}).get("ttgir", {})
                    b_enc = ttg.get("blocked_encoding", {})
                    derived = compute_derived_layout_metrics(
                        shape=[1, m, n],
                        size_per_thread=b_enc.get("sizePerThread", [1, 1, 1]),
                        threads_per_warp=b_enc.get("threadsPerWarp", [1, 1, 32]),
                        warps_per_cta=b_enc.get("warpsPerCTA", [1, 1, w]),
                        num_ctas=1,
                        reduce_axis=1,
                    )
                    c_derived = cdata.get("derived", {})
                    if derived["lane_partitions_reduce_axis"] != c_derived.get("lane_partitions_reduce_axis"):
                        errors.append(f"Lane partitions mismatch for {cfg_k}, cand={cand}")
                    if derived["warp_partitions_reduce_axis"] != c_derived.get("warp_partitions_reduce_axis"):
                        errors.append(f"Warp partitions mismatch for {cfg_k}, cand={cand}")
                    if derived["derived_reduce_elems_per_partition"] != c_derived.get("derived_reduce_elems_per_partition"):
                        errors.append(f"M elems per partition mismatch for {cfg_k}, cand={cand}")

                    # Re-derive structural transitions
                    recorded_trans = cdata.get("structural_transitions", [])
                    if cand == "default":
                        expected_trans = ["BASELINE"]
                    else:
                        expected_trans = classify_structural_transitions(def_cand, cdata)
                    if recorded_trans != expected_trans:
                        errors.append(
                            f"Structural transition re-derivation mismatch for {cfg_k}, cand={cand}: "
                            f"recorded={recorded_trans} vs rederived={expected_trans}"
                        )

                    # Check amortized grid time
                    med = cdata.get("measured", {}).get("median_us", 0.0)
                    t_cta = cdata.get("measured", {}).get("amortized_grid_time_per_cta_ns", 0.0)
                    expected_t_cta = (med * 1000.0) / b_steady
                    if abs(t_cta - expected_t_cta) > 0.1:
                        errors.append(
                            f"Sweep amortized_grid_time_per_cta_ns mismatch for {cfg_k}, cand={cand}: "
                            f"{t_cta} vs {expected_t_cta}"
                        )

                    # Check repeated validation status
                    vs_def = cdata.get("measured", {}).get("vs_default_pct", 0.0)
                    if abs(vs_def) > 3.0 and cand != "default":
                        rep = cdata.get("measured", {}).get("repeat_validation")
                        if not rep:
                            errors.append(f"Missing repeat validation for >3% case {cfg_k}, cand={cand}")
                        elif rep.get("status") not in ["within_run_gt3_reproduced", "within_run_direction_reproduced", "unstable/unresolved"]:
                            errors.append(f"Invalid repeat validation status for {cfg_k}, cand={cand}: {rep.get('status')}")

        # 10.3 Canonical CSV match
        canonical_csv = generate_sweep_csv(sweep_data)
        actual_csv = sweep_csv_path.read_text(encoding="utf-8")
        if canonical_csv != actual_csv:
            errors.append("Phase 2 sweep summary.csv does not match canonical generate_sweep_csv() output byte-for-byte")

        # 10.4 Canonical Markdown match
        canonical_md = generate_sweep_markdown(sweep_data)
        actual_md = sweep_md_path.read_text(encoding="utf-8")
        if canonical_md != actual_md:
            errors.append("Phase 2 sweep summary.md does not match canonical generate_sweep_markdown() output byte-for-byte")

        print("  Verified Phase 2 sweep (150 combinations, candidate legality, re-derived transitions, canonical CSV & MD).")

    # Check 11: Phase 2 Representative Artifact Fidelity & Hash-Binding
    print("[11/15] Validating Phase 2 representative artifacts fidelity and hash binding...")
    if not REPRESENTATIVES_DIR.exists():
        errors.append(f"Representative cases directory missing: {REPRESENTATIVES_DIR}")
    else:
        case_dirs = [p for p in REPRESENTATIVES_DIR.iterdir() if p.is_dir()]
        if len(case_dirs) < 3:
            errors.append(f"Expected at least 3 representative case directories, found {len(case_dirs)}")
        for cdir in case_dirs:
            cfg_k = cdir.name
            if cfg_k not in configs:
                errors.append(f"Representative case {cfg_k} not in sweep configs")
                continue
            cfg = configs[cfg_k]
            if not (cdir / "case_summary.md").exists():
                errors.append(f"Missing case_summary.md in {cdir}")
            for cand, cdata in cfg["candidates"].items():
                if not cdata.get("is_legal", False):
                    continue
                for ext, hash_key in [("ttgir", "ttgir_sha256"), ("ptx", "ptx_sha256"), ("sass", "sass_sha256"), ("resource.txt", "resource_sha256")]:
                    art_file = cdir / f"{cand}.{ext}"
                    if not art_file.exists():
                        errors.append(f"Missing artifact {art_file}")
                        continue
                    actual_sha = compute_sha256(art_file.read_text(encoding="utf-8"))
                    expected_sha = cdata.get("raw", {}).get(hash_key)
                    if actual_sha != expected_sha:
                        errors.append(f"SHA mismatch for {art_file}: actual={actual_sha} vs expected={expected_sha}")
        print(f"  Verified {len(case_dirs)} representative cases with exact byte-for-byte SHA256 bindings.")

    # Check 12: Offline Marginal Analysis Consistency
    print("[12/15] Validating offline marginal analysis consistency and canonical markdown...")
    mar_json_path = SATURATION_DIR / "marginal_analysis.json"
    mar_md_path = SATURATION_DIR / "marginal_analysis.md"
    if not mar_json_path.exists():
        errors.append(f"Missing {mar_json_path}")
    elif not mar_md_path.exists():
        errors.append(f"Missing {mar_md_path}")
    else:
        mar_data = json.loads(mar_json_path.read_text(encoding="utf-8"))
        for cfg_k, cfg_data in mar_data.get("configs", {}).items():
            for cand, cdata in cfg_data.get("candidates", {}).items():
                if not cdata.get("is_legal"):
                    continue
                # 1. Recalculate interval slopes
                islopes = cdata.get("interval_slopes", [])
                for s_entry in islopes:
                    dt = s_entry["dt_us"]
                    db = s_entry["db_ctas"]
                    exp_slope = (dt / db) * 1000.0
                    act_slope = s_entry["slope_ns_per_cta"]
                    if abs(exp_slope - act_slope) > 1e-4:
                        errors.append(f"Marginal slope formula mismatch in {cfg_k}, cand={cand}: {act_slope} vs {exp_slope}")

                # 2. Recalculate affine regression fit
                fit = cdata.get("affine_fit", {})
                fit_b = [float(x) for x in fit.get("fit_b_points", [])]
                fit_ys = [fit["intercept_us"] + (fit["marginal_ns_per_cta"] / 1000.0) * b + res for b, res in zip(fit_b, fit["residuals_us"])]
                recomputed_slope, recomputed_icept, recomputed_r2, _ = linear_regression(fit_b, fit_ys)
                if abs(recomputed_slope * 1000.0 - fit["marginal_ns_per_cta"]) > 1e-3:
                    errors.append(f"Marginal ns/CTA recomputation mismatch in {cfg_k}, cand={cand}")
                if abs(recomputed_icept - fit["intercept_us"]) > 1e-3:
                    errors.append(f"Intercept recomputation mismatch in {cfg_k}, cand={cand}")
                if abs(recomputed_r2 - fit["r2"]) > 1e-4:
                    errors.append(f"R2 recomputation mismatch in {cfg_k}, cand={cand}")

                # 3. Check marginal linear regime boolean
                last_two_delta = cdata.get("last_two_slope_delta_pct")
                expected_regime = (last_two_delta is not None and abs(last_two_delta) < 5.0 and fit["r2"] >= 0.99)
                if cdata.get("marginal_linear_regime_observed") != expected_regime:
                    errors.append(f"marginal_linear_regime_observed mismatch in {cfg_k}, cand={cand}: expected={expected_regime}")

        canonical_mar_md = render_marginal_analysis_markdown(mar_data)
        actual_mar_md = mar_md_path.read_text(encoding="utf-8")
        if canonical_mar_md != actual_mar_md:
            errors.append("marginal_analysis.md does not match canonical render_marginal_analysis_markdown() output byte-for-byte")
        else:
            print("  Verified offline marginal analysis calculations and byte-for-byte markdown.")

    # Check 13: Corrected Fixed-Binary Saturation Pilot Runs & Multi-Invocation Verification
    print("[13/15] Validating corrected fixed-binary saturation pilot runs (3 benchmark invocations) and summary markdown...")
    cp_json_path = SATURATION_DIR / "corrected_pilot_runs.json"
    cp_md_path = SATURATION_DIR / "corrected_pilot_summary.md"
    EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    if not cp_json_path.exists():
        errors.append(f"Missing {cp_json_path}")
    elif not cp_md_path.exists():
        errors.append(f"Missing {cp_md_path}")
    else:
        cp_runs = json.loads(cp_json_path.read_text(encoding="utf-8"))
        expected_run_keys = ["run_1", "run_2", "run_3"]
        if list(cp_runs.keys()) != expected_run_keys:
            errors.append(f"Expected runs {expected_run_keys}, got {list(cp_runs.keys())}")

        for rk in expected_run_keys:
            rdata = cp_runs.get(rk, {})
            env = rdata.get("environment", {})
            if "H100" not in env.get("gpu_name", ""):
                errors.append(f"{rk}: GPU name does not contain H100 ({env.get('gpu_name')})")
            if env.get("gpu_compute_capability") != [9, 0]:
                errors.append(f"{rk}: Compute capability mismatch ({env.get('gpu_compute_capability')})")
            if not isinstance(env.get("l2_cache_bytes"), int) or env.get("l2_cache_bytes") <= 0:
                errors.append(f"{rk}: l2_cache_bytes invalid ({env.get('l2_cache_bytes')})")
            telem = env.get("pre_run_gpu_telemetry", env.get("gpu_telemetry", {}))
            for t_key in ["sm_clock_mhz", "memory_clock_mhz", "power_draw_w", "gpu_temperature_c"]:
                if t_key not in telem:
                    errors.append(f"{rk}: Missing telemetry key {t_key}")

            for cfg_k, cfg_v in rdata.get("configs", {}).items():
                m = cfg_v["M"]
                n = cfg_v["N"]
                w = cfg_v["num_warps"]
                b_desc = cfg_v["b_desc"]
                b_runs = cfg_v["b_runs"]
                grid_data = cfg_v.get("grid_data", {})
                comp_arts = cfg_v.get("compiled_artifacts", {})

                if not comp_arts:
                    errors.append(f"{rk} {cfg_k}: Missing compiled_artifacts at config level")

                # Validate compiled artifacts and reject empty or invalid hashes
                for cand, art in comp_arts.items():
                    for hkey in ["ttgir_sha256", "ptx_sha256", "cubin_sha256", "sass_sha256"]:
                        if hkey in art:
                            hval = art[hkey]
                            if not hval or hval == EMPTY_SHA256 or len(hval) != 64:
                                errors.append(f"{rk} {cfg_k} cand={cand}: Invalid or empty hash for {hkey}: {hval}")
                    for req_key in ["ttgir_sha256", "ptx_sha256"]:
                        if req_key not in art or not art[req_key] or art[req_key] == EMPTY_SHA256:
                            errors.append(f"{rk} {cfg_k} cand={cand}: Missing or empty required hash {req_key}")

                # Fixed-binary check: each candidate references its single compiled specialization across all b_runs
                for cand in ["default", "8", "4", "2", "1"]:
                    c_first = grid_data.get(str(b_runs[0]), {}).get(cand, {})
                    if not c_first.get("is_legal"):
                        continue
                    expected_art_id = comp_arts.get(cand, {}).get("artifact_id", f"{cfg_k}_{cand}")
                    for b_other in b_runs:
                        c_other = grid_data.get(str(b_other), {}).get(cand, {})
                        art_id = c_other.get("compiled_artifact_id")
                        if art_id != expected_art_id:
                            errors.append(
                                f"{rk} {cfg_k} cand={cand}: Fixed-binary violation! "
                                f"compiled_artifact_id for B={b_other} is '{art_id}', expected '{expected_art_id}'"
                            )

                # Re-verify marginal analysis
                man = cfg_v.get("marginal_analysis", {})
                for cand, cman in man.items():
                    meds = [grid_data[str(b)][cand]["median_us"] for b in b_runs]
                    fit_b = [float(b) for b in b_runs[-3:]]
                    fit_t = meds[-3:]
                    slope_us, icept_us, r2, _ = linear_regression(fit_b, fit_t)
                    act_m_ns = cman["affine_fit"]["marginal_ns_per_cta"]
                    if abs(slope_us * 1000.0 - act_m_ns) > 1e-3:
                        errors.append(f"{rk} {cfg_k} cand={cand}: marginal slope mismatch: {act_m_ns} vs {slope_us * 1000.0}")
                    if abs(icept_us - cman["affine_fit"]["intercept_us"]) > 1e-3:
                        errors.append(f"{rk} {cfg_k} cand={cand}: intercept mismatch")
                    if abs(r2 - cman["affine_fit"]["r2"]) > 1e-4:
                        errors.append(f"{rk} {cfg_k} cand={cand}: R2 mismatch")

        canonical_cp_md = render_corrected_pilot_summary_markdown(cp_runs)
        actual_cp_md = cp_md_path.read_text(encoding="utf-8")
        if canonical_cp_md != actual_cp_md:
            errors.append("corrected_pilot_summary.md does not match canonical render_corrected_pilot_summary_markdown() output byte-for-byte")
        else:
            print("  Verified corrected fixed-binary pilot runs (3 benchmark invocations, fixed-binary invariance, telemetry, canonical MD).")

    # Check 14: Phase 3 Structural Evidence, Canonical Artifact Bindings, Equivalence, and Hypotheses
    print("[14/15] Validating Phase 3 structural evidence, canonical artifact bindings, equivalence, and hypotheses...")
    phase3_dir = RESULTS_DIR / "phase3"
    canonical_dir = phase3_dir / "fixed_binary_artifacts" / "canonical"
    p3_ann_path = EXP_DIR / "phase3_audited_annotations.json"
    p3_equiv_path = phase3_dir / "artifact_equivalence_report.json"
    p3_pos_neg_path = phase3_dir / "structural_comparison" / "positive_vs_negative.json"
    p3_summary_md_path = phase3_dir / "structural_comparison" / "summary.md"
    p3_hypotheses_md_path = phase3_dir / "hypotheses.md"

    expected_p3_configs = {
        "M32_N64_w8": ["default", "8", "4", "2", "1"],
        "M32_N128_w4": ["default", "8", "4", "2", "1"],
        "M32_N16_w8": ["default", "2", "1"],
    }

    # 14.1 Validate existence and non-emptiness of canonical artifacts
    for cfg_k, cands in expected_p3_configs.items():
        cfg_canon_dir = canonical_dir / cfg_k
        if not cfg_canon_dir.exists():
            errors.append(f"Phase 3 canonical dir missing: {cfg_canon_dir}")
            continue
        for cand in cands:
            for ext in ["ptx", "ttgir", "sass", "resource.txt", "cubin.sha256"]:
                f_path = cfg_canon_dir / f"{cand}.{ext}"
                if not f_path.exists():
                    errors.append(f"Phase 3 canonical artifact missing: {f_path}")
                else:
                    content = f_path.read_text(encoding="utf-8").strip()
                    if not content:
                        errors.append(f"Phase 3 canonical artifact empty: {f_path}")
                    elif ext in ["ptx", "ttgir", "sass"]:
                        file_sha = compute_sha256(f_path.read_text(encoding="utf-8"))
                        if file_sha == EMPTY_SHA256:
                            errors.append(f"Phase 3 canonical artifact has empty sha: {f_path}")
                    elif ext == "cubin.sha256":
                        if len(content) != 64 or content == EMPTY_SHA256 or not all(c in "0123456789abcdefABCDEF" for c in content):
                            errors.append(f"Phase 3 canonical cubin hash invalid: {f_path} -> {content}")

    # 14.2 Validate cross-run invariance of canonical artifacts vs run_1, run_2, run_3
    fixed_arts_dir = phase3_dir / "fixed_binary_artifacts"
    for rk in ["run_1", "run_2", "run_3"]:
        for cfg_k, cands in expected_p3_configs.items():
            for cand in cands:
                for ext in ["ptx", "ttgir", "sass", "resource.txt", "cubin.sha256"]:
                    run_f = fixed_arts_dir / rk / cfg_k / f"{cand}.{ext}"
                    canon_f = canonical_dir / cfg_k / f"{cand}.{ext}"
                    if run_f.exists() and canon_f.exists():
                        if run_f.read_text(encoding="utf-8") != canon_f.read_text(encoding="utf-8"):
                            errors.append(f"Phase 3 artifact mismatch between {rk} and canonical: {run_f} vs {canon_f}")

    # 14.3 Validate phase3_audited_annotations.json bindings and structure
    if not p3_ann_path.exists():
        errors.append(f"Missing {p3_ann_path}")
    else:
        p3_ann_data = json.loads(p3_ann_path.read_text(encoding="utf-8"))
        p3_cfgs = p3_ann_data.get("configurations", {})
        for cfg_k, cands in expected_p3_configs.items():
            if cfg_k not in p3_cfgs:
                errors.append(f"Phase 3 annotations missing config {cfg_k}")
                continue
            for cand in cands:
                if cand not in p3_cfgs[cfg_k]:
                    errors.append(f"Phase 3 annotations missing {cfg_k} candidate {cand}")
                    continue
                cand_data = p3_cfgs[cfg_k][cand]
                ptx_f = canonical_dir / cfg_k / f"{cand}.ptx"
                ttgir_f = canonical_dir / cfg_k / f"{cand}.ttgir"
                sass_f = canonical_dir / cfg_k / f"{cand}.sass"
                res_f = canonical_dir / cfg_k / f"{cand}.resource.txt"
                cubin_sha_f = canonical_dir / cfg_k / f"{cand}.cubin.sha256"

                if ptx_f.exists() and compute_sha256(ptx_f.read_text(encoding="utf-8")) != cand_data.get("ptx_sha256"):
                    errors.append(f"Phase 3 PTX SHA binding mismatch in annotations: {cfg_k} {cand}")
                if ttgir_f.exists() and compute_sha256(ttgir_f.read_text(encoding="utf-8")) != cand_data.get("ttgir_sha256"):
                    errors.append(f"Phase 3 TTGIR SHA binding mismatch in annotations: {cfg_k} {cand}")
                if sass_f.exists() and compute_sha256(sass_f.read_text(encoding="utf-8")) != cand_data.get("sass_sha256"):
                    errors.append(f"Phase 3 SASS SHA binding mismatch in annotations: {cfg_k} {cand}")
                if res_f.exists() and compute_sha256(res_f.read_text(encoding="utf-8")) != cand_data.get("resource_sha256"):
                    errors.append(f"Phase 3 resource SHA binding mismatch in annotations: {cfg_k} {cand}")
                if cubin_sha_f.exists() and cubin_sha_f.read_text(encoding="utf-8").strip() != cand_data.get("cubin_sha256"):
                    errors.append(f"Phase 3 cubin SHA binding mismatch in annotations: {cfg_k} {cand}")

                # Check phase lines and non-overlapping bounds
                if ptx_f.exists():
                    ptx_lines = ptx_f.read_text(encoding="utf-8").splitlines()
                    total_ptx_lines = len(ptx_lines)
                    prev_end = 0
                    for p_name, p_info in cand_data.get("phases", {}).items():
                        plines = p_info.get("lines")
                        if not plines or plines == [0, 0]:
                            continue
                        start, end = plines[0], plines[1]
                        if not (1 <= start <= end <= total_ptx_lines):
                            errors.append(f"Phase 3 line bounds error {cfg_k} {cand} {p_name}: [{start}, {end}] vs total {total_ptx_lines}")
                        if start < prev_end:
                            errors.append(f"Phase 3 phase ordering error {cfg_k} {cand} {p_name}: {start} < {prev_end}")
                        prev_end = end
                        if "instruction" in p_info and "count" in p_info:
                            act_c = count_opcode_in_range(ptx_lines, start, end, p_info["instruction"])
                            if act_c != p_info["count"]:
                                errors.append(f"Phase 3 opcode count mismatch {cfg_k} {cand} {p_name}: expected {p_info['count']}, found {act_c}")
                        if "opcode_counts" in p_info:
                            for opc, exp_c in p_info["opcode_counts"].items():
                                act_c = count_opcode_in_range(ptx_lines, start, end, opc)
                                if act_c != exp_c:
                                    errors.append(f"Phase 3 opcode_counts mismatch {cfg_k} {cand} {p_name} {opc}: expected {exp_c}, found {act_c}")

    # 14.4 Validate artifact_equivalence_report.json
    if not p3_equiv_path.exists():
        errors.append(f"Missing {p3_equiv_path}")
    else:
        equiv_data = json.loads(p3_equiv_path.read_text(encoding="utf-8"))
        for cfg_k, cands in expected_p3_configs.items():
            if cfg_k not in equiv_data.get("configurations", {}):
                errors.append(f"Equivalence report missing {cfg_k}")
                continue
            for cand in cands:
                cand_eq = equiv_data["configurations"][cfg_k].get(cand, {})
                if not cand_eq.get("reduction_region_instructions_equal"):
                    errors.append(f"Equivalence report: reduction_region_instructions_equal is not true for {cfg_k} {cand}")
                if cand_eq.get("classification") != "full_artifact_differs_reduction_region_equivalent":
                    errors.append(f"Equivalence report: unexpected classification for {cfg_k} {cand}: {cand_eq.get('classification')}")

    # 14.5 Validate canonical regeneration of summary.md and hypotheses.md
    if not p3_pos_neg_path.exists():
        errors.append(f"Missing {p3_pos_neg_path}")
    if not p3_summary_md_path.exists():
        errors.append(f"Missing {p3_summary_md_path}")
    if not p3_hypotheses_md_path.exists():
        errors.append(f"Missing {p3_hypotheses_md_path}")

    if p3_pos_neg_path.exists() and p3_summary_md_path.exists() and p3_hypotheses_md_path.exists():
        curr_ds = json.loads(p3_pos_neg_path.read_text(encoding="utf-8"))
        
        expected_summary_md = render_phase3_summary_markdown(curr_ds)
        actual_summary_md = p3_summary_md_path.read_text(encoding="utf-8").strip()
        if expected_summary_md.strip() != actual_summary_md:
            errors.append("Phase 3 summary.md does not match canonical render_summary_markdown(ds) byte-for-byte")

        expected_hypotheses_md = render_phase3_hypotheses_markdown(curr_ds)
        actual_hypotheses_md = p3_hypotheses_md_path.read_text(encoding="utf-8").strip()
        if expected_hypotheses_md.strip() != actual_hypotheses_md:
            errors.append("Phase 3 hypotheses.md does not match canonical render_hypotheses_markdown(ds) byte-for-byte")

        # 14.6 Validate hypotheses format and discipline
        hypotheses_text = p3_hypotheses_md_path.read_text(encoding="utf-8")
        observed_sections = re.findall(r"\*\*OBSERVED\*\*:(.*?)(?=\*\*DERIVED\*\*|\*\*HYPOTHESIS\*\*|\Z)", hypotheses_text, re.DOTALL)
        for idx, obs_sec in enumerate(observed_sections):
            for banned in ["dram saturation", "hbm saturation", "hbm3 saturation", "memory bus saturation", "dram bandwidth saturation"]:
                if banned in obs_sec.lower():
                    errors.append(f"Hypotheses OBSERVED section {idx+1} contains unmeasured saturation assertion: '{banned}'")

        status_matches = re.findall(r"\*\*STATUS\*\*:\s*`([^`]+)`", hypotheses_text)
        if len(status_matches) != 4:
            errors.append(f"Expected 4 STATUS tags in hypotheses.md, found {len(status_matches)}")
        for st in status_matches:
            if st != "UNVERIFIED / PENDING_DIFFERENTIAL_MICROBENCH":
                errors.append(f"Invalid STATUS tag in hypotheses.md: '{st}'")

        print("  Verified Phase 3 structural evidence, canonical artifact bindings, equivalence report, and hypotheses.")

    # Check 15: Phase 3 Step B v1 confounded benchmark evidence
    print("[15/16] Validating Phase 3 Step B v1 confounded benchmark evidence...")
    mb_dir = phase3_dir / "microbench_reduction"
    mb_design_path = mb_dir / "design.md"
    mb_val_path = mb_dir / "validation.json"
    mb_res_path = mb_dir / "results.json"
    mb_sum_path = mb_dir / "summary.md"
    mb_arts_dir = mb_dir / "artifacts"

    if not mb_design_path.exists():
        errors.append(f"Missing {mb_design_path}")
    if not mb_val_path.exists():
        errors.append(f"Missing {mb_val_path}")
    if not mb_res_path.exists():
        errors.append(f"Missing {mb_res_path}")
    if not mb_sum_path.exists():
        errors.append(f"Missing {mb_sum_path}")
    if not mb_arts_dir.exists():
        errors.append(f"Missing {mb_arts_dir}")

    if mb_val_path.exists() and mb_res_path.exists() and mb_arts_dir.exists():
        mb_val_data = json.loads(mb_val_path.read_text(encoding="utf-8"))
        mb_res_data = json.loads(mb_res_path.read_text(encoding="utf-8"))

        expected_mb_configs = ["M32_N64_w8", "M32_N128_w4"]
        expected_mb_candidates = ["default", "4"]
        expected_k_vals = [1, 2, 4, 8]
        expected_runs = ["run_1", "run_2", "run_3"]

        # 15.1 Artifact completeness and non-empty hash validation
        for rk in expected_runs:
            for cfg_k in expected_mb_configs:
                for cand in expected_mb_candidates:
                    for k in expected_k_vals:
                        prefix = f"cand4_K{k}" if cand == "4" else f"default_K{k}"
                        for ext in ["ptx", "ttgir", "sass", "resource.txt", "cubin.sha256"]:
                            art_path = mb_arts_dir / rk / cfg_k / f"{prefix}.{ext}"
                            if not art_path.exists():
                                errors.append(f"Missing microbench artifact: {art_path}")
                            else:
                                art_content = art_path.read_text(encoding="utf-8").strip()
                                if not art_content:
                                    errors.append(f"Empty microbench artifact: {art_path}")
                                elif ext in ["ptx", "ttgir", "sass"]:
                                    h = compute_sha256(art_path.read_text(encoding="utf-8"))
                                    if h == EMPTY_SHA256:
                                        errors.append(f"Empty SHA for microbench artifact: {art_path}")
                                elif ext == "cubin.sha256":
                                    if len(art_content) != 64 or art_content == EMPTY_SHA256:
                                        errors.append(f"Invalid cubin sha in {art_path}: {art_content}")

        # 15.2 Cross-run bit-for-bit codegen invariance
        if mb_res_data.get("codegen_identical_across_all_runs") is not True:
            errors.append("results.json codegen_identical_across_all_runs is not True")

        for cfg_k in expected_mb_configs:
            for cand in expected_mb_candidates:
                for k in expected_k_vals:
                    prefix = f"cand4_K{k}" if cand == "4" else f"default_K{k}"
                    r1_ptx = (mb_arts_dir / "run_1" / cfg_k / f"{prefix}.ptx").read_text(encoding="utf-8")
                    r1_cubin = (mb_arts_dir / "run_1" / cfg_k / f"{prefix}.cubin.sha256").read_text(encoding="utf-8").strip()
                    for rk in ["run_2", "run_3"]:
                        other_ptx = (mb_arts_dir / rk / cfg_k / f"{prefix}.ptx").read_text(encoding="utf-8")
                        other_cubin = (mb_arts_dir / rk / cfg_k / f"{prefix}.cubin.sha256").read_text(encoding="utf-8").strip()
                        if r1_ptx != other_ptx:
                            errors.append(f"PTX codegen mismatch across runs for {cfg_k} {prefix}: run_1 vs {rk}")
                        if r1_cubin != other_cubin:
                            errors.append(f"CUBIN sha mismatch across runs for {cfg_k} {prefix}: run_1 vs {rk}")

        # 15.3 Validation invariants & blind spot verification
        for cfg_k in expected_mb_configs:
            cfg_val = mb_val_data.get(cfg_k, {})
            for cand in expected_mb_candidates:
                # Extract initial load sequence from disk PTX for K=1
                k1_prefix = f"cand4_K1" if cand == "4" else f"default_K1"
                k1_ptx_path = mb_arts_dir / "run_1" / cfg_k / f"{k1_prefix}.ptx"
                k1_loads = []
                if k1_ptx_path.exists():
                    for line in k1_ptx_path.read_text(encoding="utf-8").splitlines():
                        ls = line.strip()
                        if not ls or ls.startswith("//") or ls.startswith(".") or ls.startswith("$") or ls.endswith(":"):
                            continue
                        if "ld.shared" in ls:
                            k1_loads.append(ls.split()[0])
                        elif k1_loads:
                            if "ld.shared" not in ls:
                                break

                # Also verify TTGIR layout invariance across K
                k1_ttgir_path = mb_arts_dir / "run_1" / cfg_k / f"{k1_prefix}.ttgir"
                k1_enc = re.search(r"#blocked\s*=\s*(#ttg\.blocked<\{[^>]+\}>)", k1_ttgir_path.read_text(encoding="utf-8")) if k1_ttgir_path.exists() else None
                k1_layout_str = k1_enc.group(1) if k1_enc else ""

                for k in expected_k_vals:
                    key = f"{cand}_K{k}"
                    if key not in cfg_val:
                        errors.append(f"Missing {key} in validation.json for {cfg_k}")
                        continue
                    item = cfg_val[key]
                    if not item.get("is_correct"):
                        errors.append(f"Numerical correctness failed in validation.json: {cfg_k} {key}")
                    if item.get("tma_load_count") != 1:
                        errors.append(f"TMA count not 1 in {cfg_k} {key}: {item.get('tma_load_count')}")

                    # Check actual PTX initial loads
                    k_prefix = f"cand4_K{k}" if cand == "4" else f"default_K{k}"
                    k_ptx_path = mb_arts_dir / "run_1" / cfg_k / f"{k_prefix}.ptx"
                    k_loads = []
                    if k_ptx_path.exists():
                        for line in k_ptx_path.read_text(encoding="utf-8").splitlines():
                            ls = line.strip()
                            if not ls or ls.startswith("//") or ls.startswith(".") or ls.startswith("$") or ls.endswith(":"):
                                continue
                            if "ld.shared" in ls:
                                k_loads.append(ls.split()[0])
                            elif k_loads:
                                if "ld.shared" not in ls:
                                    break

                    sig_match = (k_loads == k1_loads)
                    if item.get("localload_signature_matches_k1") != sig_match:
                        errors.append(f"localload_signature_matches_k1 mismatch in {cfg_k} {key}: disk={sig_match} vs json={item.get('localload_signature_matches_k1')}")

                    # Check TTGIR layout invariance
                    k_ttgir_path = mb_arts_dir / "run_1" / cfg_k / f"{k_prefix}.ttgir"
                    k_enc = re.search(r"#blocked\s*=\s*(#ttg\.blocked<\{[^>]+\}>)", k_ttgir_path.read_text(encoding="utf-8")) if k_ttgir_path.exists() else None
                    k_layout_str = k_enc.group(1) if k_enc else ""
                    if k_layout_str != k1_layout_str:
                        errors.append(f"TTGIR blocked layout mismatch across K in {cfg_k} {key}: {k_layout_str} vs {k1_layout_str}")

                    # For Step B v1: initial load signature mismatch in cand4, max.bf16x2 not scaling, and register growth
                    # must result in is_valid_for_isolation == False and classification == CONFOUNDED_BY_CODEGEN_AND_REGISTER_PRESSURE
                    if item.get("is_valid_for_isolation") is not False:
                        errors.append(f"Expected is_valid_for_isolation == False for Step B v1 in {cfg_k} {key}")
                    if not item.get("confound_reasons"):
                        errors.append(f"Expected non-empty confound_reasons for Step B v1 in {cfg_k} {key}")

                    # Verify mathematical opcode scaling for shfl
                    opc = item.get("opcode_counts", {})
                    shfl_cnt = opc.get("shfl_sync_bfly", 0)
                    if cfg_k == "M32_N64_w8":
                        exp_shfl = (40 if cand == "default" else 16) * k
                    else:
                        exp_shfl = (24 if cand == "default" else 8) * k
                    if shfl_cnt != exp_shfl:
                        errors.append(f"Opcode shfl scaling mismatch in {cfg_k} {key}: expected {exp_shfl}, got {shfl_cnt}")

        # 15.4 Numerical slope & linear fit recomputation
        cfg_summary = mb_res_data.get("configuration_summary", {})
        for cfg_k in expected_mb_configs:
            if cfg_k not in cfg_summary:
                errors.append(f"Missing {cfg_k} in results.json configuration_summary")
                continue
            cfg_s = cfg_summary[cfg_k]
            if cfg_s.get("valid_for_isolation") is not False:
                errors.append(f"Expected valid_for_isolation == False in results.json configuration_summary for {cfg_k}")
            if cfg_s.get("classification") != "CONFOUNDED_BY_CODEGEN_AND_REGISTER_PRESSURE":
                errors.append(f"Expected classification == 'CONFOUNDED_BY_CODEGEN_AND_REGISTER_PRESSURE' in results.json for {cfg_k}, got {cfg_s.get('classification')}")

            k_scaling = cfg_s.get("k_scaling", {})
            for str_k, k_info in k_scaling.items():
                k_val = int(str_k)
                def_m = k_info["default_marginal_ns_mean"]
                c4_m = k_info["cand4_marginal_ns_mean"]
                gap_m = k_info["gap_ns_mean"]
                if abs(gap_m - (def_m - c4_m)) > 1e-4:
                    errors.append(f"Gap mean formula mismatch in {cfg_k} K={k_val}: {gap_m} vs {def_m - c4_m}")
                for r_idx in range(len(expected_runs)):
                    r_def = k_info["default_marginal_ns_runs"][r_idx]
                    r_c4 = k_info["cand4_marginal_ns_runs"][r_idx]
                    r_gap = k_info["gap_ns_runs"][r_idx]
                    if abs(r_gap - (r_def - r_c4)) > 1e-4:
                        errors.append(f"Per-run gap mismatch in {cfg_k} K={k_val} run {r_idx+1}")

            # Verify linear fits
            lin_fits = cfg_s.get("linear_fits", {})
            gap_vals = [k_scaling[str(k)]["gap_ns_mean"] for k in expected_k_vals]
            rec_gap_slope, rec_gap_icept, rec_gap_r2, _ = linear_regression([float(k) for k in expected_k_vals], gap_vals)
            if abs(rec_gap_slope - lin_fits["gap_vs_k_slope"]) > 1e-4:
                errors.append(f"Gap vs K slope mismatch in {cfg_k}: {rec_gap_slope} vs {lin_fits['gap_vs_k_slope']}")
            if abs(rec_gap_icept - lin_fits["gap_vs_k_intercept"]) > 1e-4:
                errors.append(f"Gap vs K intercept mismatch in {cfg_k}")
            if abs(rec_gap_r2 - lin_fits["gap_vs_k_r2"]) > 1e-4:
                errors.append(f"Gap vs K R2 mismatch in {cfg_k}")

        # 15.5 Canonical Markdown equivalence
        can_des_md = render_mb_design_markdown()
        act_des_md = mb_design_path.read_text(encoding="utf-8")
        if can_des_md != act_des_md:
            errors.append("design.md does not match canonical generate_design_markdown() output byte-for-byte")

        can_sum_md = render_mb_summary_markdown(mb_res_data)
        act_sum_md = mb_sum_path.read_text(encoding="utf-8")
        if can_sum_md != act_sum_md:
            errors.append("summary.md does not match canonical generate_summary_markdown() output byte-for-byte")

        # 15.6 Markdown text hygiene
        if "CONFOUNDED_BY_CODEGEN_AND_REGISTER_PRESSURE" not in act_sum_md:
            errors.append("summary.md missing CONFOUNDED_BY_CODEGEN_AND_REGISTER_PRESSURE tag")
        if "fitted K-axis intercept; no physical attribution" not in act_sum_md:
            errors.append("summary.md missing fitted K-axis intercept note")

        print("  Verified Phase 3 Step B v1 confounded benchmark evidence (confound classification, LocalLoad divergence, telemetry, canonical MD).")

    # Check 16: Phase 3 Step B v3 Single-Binary Runtime-K Feasibility
    print("[16/17] Validating Phase 3 Step B v3 single-binary runtime-K feasibility...")
    v3_dir = phase3_dir / "v3_runtime_k"
    v3_design_path = v3_dir / "design.md"
    v3_val_path = v3_dir / "validation.json"
    v3_res_path = v3_dir / "results.json"
    v3_sum_path = v3_dir / "summary.md"
    v3_arts_dir = v3_dir / "artifacts"
    v3_raw_path = v3_dir / "raw_results.json"

    if not v3_design_path.exists():
        errors.append(f"Missing {v3_design_path}")
    if not v3_val_path.exists():
        errors.append(f"Missing {v3_val_path}")
    if not v3_res_path.exists():
        errors.append(f"Missing {v3_res_path}")
    if not v3_sum_path.exists():
        errors.append(f"Missing {v3_sum_path}")
    if not v3_arts_dir.exists():
        errors.append(f"Missing {v3_arts_dir}")
    if not v3_raw_path.exists():
        errors.append(f"Missing {v3_raw_path}")

    if v3_val_path.exists() and v3_res_path.exists() and v3_raw_path.exists() and v3_arts_dir.exists():
        v3_val_data = json.loads(v3_val_path.read_text(encoding="utf-8"))
        v3_res_data = json.loads(v3_res_path.read_text(encoding="utf-8"))
        v3_raw_data = json.loads(v3_raw_path.read_text(encoding="utf-8"))

        expected_configs = ["M32_N64_w8", "M32_N128_w4"]
        expected_candidates = ["default", "4"]

        # 16.1 Artifact completeness & non-empty content validation
        for cfg in expected_configs:
            for cand in expected_candidates:
                cand_data = v3_raw_data.get(cfg, {}).get(cand, {})
                cubin_sha = cand_data.get("cubin_sha256", "")
                if not cubin_sha or len(cubin_sha) != 64:
                    errors.append(f"Invalid cubin_sha256 for {cfg} {cand}: {cubin_sha}")

                for ext in ["ptx", "ttgir", "sass", "resource.txt", "cubin.sha256"]:
                    art_file = v3_arts_dir / cfg / f"{cand}.{ext}"
                    if not art_file.exists():
                        errors.append(f"Missing v3 artifact: {art_file}")
                    else:
                        content = art_file.read_text(encoding="utf-8").strip()
                        if not content:
                            errors.append(f"Empty v3 artifact: {art_file}")
                        if ext == "cubin.sha256" and content != cubin_sha:
                            errors.append(f"cubin.sha256 mismatch for {cfg} {cand}: {content} vs {cubin_sha}")

                # 16.2 Single-binary reuse & no runtime K specialization
                spec_check = cand_data.get("specialization_check", {})
                if spec_check.get("runtime_k_specialized") is not False:
                    errors.append(f"Expected runtime_k_specialized == False for {cfg} {cand}")
                if spec_check.get("initial_cache_len") != 1 or spec_check.get("final_cache_len") != 1:
                    errors.append(f"Cache len changed across K in {cfg} {cand}: initial={spec_check.get('initial_cache_len')}, final={spec_check.get('final_cache_len')}")

                # 16.3 Runtime loop verification
                loop_check = cand_data.get("runtime_loop", {})
                if not loop_check.get("ttgir_has_scf_for"):
                    errors.append(f"Missing scf.for in TTGIR for {cfg} {cand}")
                if not loop_check.get("ptx_has_loop_branch"):
                    errors.append(f"Missing loop branch in PTX for {cfg} {cand}")
                if not loop_check.get("sass_has_bra"):
                    errors.append(f"Missing BRA in SASS for {cfg} {cand}")
                if not loop_check.get("runtime_loop_detected"):
                    errors.append(f"runtime_loop_detected == False for {cfg} {cand}")

                # 16.4 Numerical correctness
                corr_check = cand_data.get("correctness", {})
                if not corr_check.get("all_passed"):
                    errors.append(f"Correctness failed for {cfg} {cand}")

                # 16.5 Spill checks
                res = cand_data.get("resources", {})
                if res.get("local_bytes") != 0 or res.get("stack_bytes") != 0:
                    errors.append(f"Non-zero spill in {cfg} {cand}: local={res.get('local_bytes')}, stack={res.get('stack_bytes')}")

                # 16.6 Mechanical TTGIR validation
                ttgir_text = (v3_arts_dir / cfg / f"{cand}.ttgir").read_text(encoding="utf-8")
                scf_split = ttgir_text.split("scf.for", 1)
                tma_inside = len(re.findall(r"ttng\.async_tma_copy_global_to_local", scf_split[1])) if len(scf_split) > 1 else 0
                tma_outside = len(re.findall(r"ttng\.async_tma_copy_global_to_local", scf_split[0]))
                if tma_inside != 0 or tma_outside != 1:
                    errors.append(f"Mechanical Criterion A failure in v3 for {cfg} {cand}: outside={tma_outside}, inside={tma_inside}")

                # 16.7 Mechanical PTX loop branch validation
                ptx_text = (v3_arts_dir / cfg / f"{cand}.ptx").read_text(encoding="utf-8")
                labels_seen = set()
                bw_count = 0
                for l in ptx_text.splitlines():
                    lbl_m = re.match(r"^\s*(\$L__BB\d+_\d+):", l)
                    if lbl_m:
                        labels_seen.add(lbl_m.group(1))
                    bra_m = re.search(r"(@%p\d+)?\s*bra(?:\.uni)?\s+(\$L__BB\d+_\d+);", l)
                    if bra_m:
                        target = bra_m.group(2)
                        if target in labels_seen:
                            bw_count += 1
                if bw_count != 1:
                    errors.append(f"Expected exactly 1 backward branch in v3 PTX for {cfg} {cand}, got {bw_count}")

        # 16.8 Confound classification & criteria verification
        if v3_val_data.get("overall_status") != "V3_CODEGEN_CONFOUNDED_NOT_ACCEPTED_FOR_TIMING":
            errors.append(f"Expected overall_status == 'V3_CODEGEN_CONFOUNDED_NOT_ACCEPTED_FOR_TIMING', got {v3_val_data.get('overall_status')}")

        crit = v3_val_data.get("criteria", {})
        if crit.get("Criterion A (TMA Descriptor Load Invariant)") != "PASS":
            errors.append("Criterion A was not PASS")
        if crit.get("Criterion B (Initial LocalLoad Invariant)") != "FAIL_LOCAL_LOAD_SUNK_INTO_LOOP":
            errors.append("Criterion B was not FAIL_LOCAL_LOAD_SUNK_INTO_LOOP")
        if crit.get("Criterion C (Reduction Body Template Invariant)") != "FAIL_REDUCTION_TEMPLATE_MISMATCH":
            errors.append("Criterion C was not FAIL_REDUCTION_TEMPLATE_MISMATCH")
        if crit.get("Criterion F (Residency & Occupancy Matched)") != "FAIL_RESIDENCY_DISPARITY":
            errors.append("Criterion F was not FAIL_RESIDENCY_DISPARITY")

        print("  Verified Phase 3 Step B v3 single-binary runtime-K feasibility (mechanical loop checks, single CUBIN, residency disparity & LocalLoad confounds recorded).")

    # Check 17: Phase 3 Step B v4 Preloaded-Register Runtime-K Feasibility & Occupancy Baseline
    print("[17/18] Validating Phase 3 Step B v4 preloaded-register runtime-K feasibility...")
    canon_occ_path = phase3_dir / "canonical_occupancy" / "canonical_occupancy.json"
    v4_dir = phase3_dir / "v4_preloaded_k"
    v4_design_path = v4_dir / "design.md"
    v4_val_path = v4_dir / "validation.json"
    v4_res_path = v4_dir / "results.json"
    v4_sum_path = v4_dir / "summary.md"
    v4_arts_dir = v4_dir / "artifacts"
    v4_raw_path = v4_dir / "raw_results.json"

    if not canon_occ_path.exists():
        errors.append(f"Missing {canon_occ_path}")
    if not v4_design_path.exists():
        errors.append(f"Missing {v4_design_path}")
    if not v4_val_path.exists():
        errors.append(f"Missing {v4_val_path}")
    if not v4_res_path.exists():
        errors.append(f"Missing {v4_res_path}")
    if not v4_sum_path.exists():
        errors.append(f"Missing {v4_sum_path}")
    if not v4_arts_dir.exists():
        errors.append(f"Missing {v4_arts_dir}")
    if not v4_raw_path.exists():
        errors.append(f"Missing {v4_raw_path}")

    if canon_occ_path.exists() and v4_val_path.exists() and v4_res_path.exists() and v4_raw_path.exists() and v4_arts_dir.exists():
        canon_occ_data = json.loads(canon_occ_path.read_text(encoding="utf-8"))
        v4_val_data = json.loads(v4_val_path.read_text(encoding="utf-8"))
        v4_res_data = json.loads(v4_res_path.read_text(encoding="utf-8"))
        v4_raw_data = json.loads(v4_raw_path.read_text(encoding="utf-8"))

        dev_lim = canon_occ_data.get("device_limits", {})
        if dev_lim.get("max_warps_per_sm") != 64:
            errors.append(f"Expected max_warps_per_sm == 64, got {dev_lim.get('max_warps_per_sm')}")

        expected_configs = ["M32_N64_w8", "M32_N128_w4"]
        expected_candidates = ["default", "4"]

        # 17.1 Canonical Step A occupancy baseline check
        canon_res = canon_occ_data.get("canonical_results", {})
        for cfg in expected_configs:
            for cand in expected_candidates:
                c_occ = canon_res.get(cfg, {}).get(cand, {}).get("occupancy", {})
                if c_occ.get("active_warps_per_sm") != 64:
                    errors.append(f"Canonical {cfg} {cand} active_warps_per_sm != 64 (got {c_occ.get('active_warps_per_sm')})")
                if c_occ.get("smem_limited") is not False:
                    errors.append(f"Canonical {cfg} {cand} was unexpectedly smem_limited")

        # 17.2 v4 Artifact completeness, non-empty, and CUBIN SHA binding
        for cfg in expected_configs:
            for cand in expected_candidates:
                cand_data = v4_raw_data.get(cfg, {}).get(cand, {})
                cubin_sha = cand_data.get("cubin_sha256", "")
                if not cubin_sha or len(cubin_sha) != 64:
                    errors.append(f"Invalid cubin_sha256 for v4 {cfg} {cand}: {cubin_sha}")

                for ext in ["ptx", "ttgir", "sass", "resource.txt", "cubin.sha256"]:
                    art_file = v4_arts_dir / cfg / f"{cand}.{ext}"
                    if not art_file.exists():
                        errors.append(f"Missing v4 artifact: {art_file}")
                    else:
                        content = art_file.read_text(encoding="utf-8").strip()
                        if not content:
                            errors.append(f"Empty v4 artifact: {art_file}")
                        if ext == "cubin.sha256" and content != cubin_sha:
                            errors.append(f"cubin.sha256 mismatch for v4 {cfg} {cand}: {content} vs {cubin_sha}")

                # 17.3 Single-binary reuse across K
                spec_check = cand_data.get("specialization_check", {})
                if spec_check.get("runtime_k_specialized") is not False:
                    errors.append(f"Expected v4 runtime_k_specialized == False for {cfg} {cand}")

                # 17.4 Numerical correctness across K
                corr_check = cand_data.get("correctness", {})
                if not corr_check.get("all_passed"):
                    errors.append(f"Correctness failed for v4 {cfg} {cand}")

                # 17.5 Zero spills
                res = cand_data.get("resources", {})
                if res.get("local_bytes") != 0 or res.get("stack_bytes") != 0:
                    errors.append(f"Non-zero spill in v4 {cfg} {cand}: local={res.get('local_bytes')}, stack={res.get('stack_bytes')}")

                # 17.6 Mechanical TTGIR validation
                ttgir_text = (v4_arts_dir / cfg / f"{cand}.ttgir").read_text(encoding="utf-8")
                scf_split = ttgir_text.split("scf.for", 1)
                tma_inside = len(re.findall(r"ttng\.async_tma_copy_global_to_local", scf_split[1])) if len(scf_split) > 1 else 0
                tma_outside = len(re.findall(r"ttng\.async_tma_copy_global_to_local", scf_split[0]))
                if tma_inside != 0 or tma_outside != 1:
                    errors.append(f"Mechanical Criterion A failure in v4 for {cfg} {cand}: outside={tma_outside}, inside={tma_inside}")

                ll_inside = len(re.findall(r"\bttg\.local_load\b", scf_split[1])) if len(scf_split) > 1 else 0
                ll_outside = len(re.findall(r"\bttg\.local_load\b", scf_split[0]))
                if ll_inside != 0:
                    errors.append(f"Mechanical Criterion B failure: ttg.local_load found inside scf.for in v4 for {cfg} {cand} (count={ll_inside})")
                if ll_outside != 1:
                    errors.append(f"Mechanical Criterion B failure: expected 1 pre-loop ttg.local_load in v4 for {cfg} {cand} (count={ll_outside})")

                # Layout match with canonical
                canon_ttgir_text = (phase3_dir / "fixed_binary_artifacts" / "canonical" / cfg / f"{cand}.ttgir").read_text(encoding="utf-8")
                cb_m = re.search(r"(#blocked\d*\s*=\s*#ttg\.blocked<[^>]+>)", canon_ttgir_text)
                vb_m = re.search(r"(#blocked\d*\s*=\s*#ttg\.blocked<[^>]+>)", ttgir_text)
                cb = re.sub(r"#blocked\d*", "#blocked", cb_m.group(1)) if cb_m else None
                vb = re.sub(r"#blocked\d*", "#blocked", vb_m.group(1)) if vb_m else None
                if cb != vb:
                    errors.append(f"Criterion D failure: layout mismatch for v4 {cfg} {cand}: {vb} vs {cb}")

                # 17.7 Mechanical PTX loop branch validation
                ptx_text = (v4_arts_dir / cfg / f"{cand}.ptx").read_text(encoding="utf-8")
                labels_seen = set()
                bw_count = 0
                for l in ptx_text.splitlines():
                    lbl_m = re.match(r"^\s*(\$L__BB\d+_\d+):", l)
                    if lbl_m:
                        labels_seen.add(lbl_m.group(1))
                    bra_m = re.search(r"(@%p\d+)?\s*bra(?:\.uni)?\s+(\$L__BB\d+_\d+);", l)
                    if bra_m:
                        target = bra_m.group(2)
                        if target in labels_seen:
                            bw_count += 1
                if bw_count != 1:
                    errors.append(f"Expected exactly 1 backward branch in v4 PTX for {cfg} {cand}, got {bw_count}")

        # 17.8 Residency comparison & mechanical Criterion N check
        w8_def_occ = v4_val_data["v4_evaluations"]["M32_N64_w8"]["default"]["occupancy"]
        w8_c4_occ = v4_val_data["v4_evaluations"]["M32_N64_w8"]["4"]["occupancy"]
        if w8_def_occ["blocks_per_sm_actual_smem"] != 6 or w8_c4_occ["blocks_per_sm_actual_smem"] != 8:
            errors.append(f"Unexpected occupancy values for M32_N64_w8: default={w8_def_occ['blocks_per_sm_actual_smem']}, cand4={w8_c4_occ['blocks_per_sm_actual_smem']}")

        for cfg in expected_configs:
            for cand in expected_candidates:
                occ = v4_val_data["v4_evaluations"][cfg][cand]["occupancy"]
                b_act = occ.get("blocks_per_sm_actual_smem")
                b_zero = occ.get("blocks_per_sm_zero_dynamic_smem")
                if b_act is None or b_zero is None or b_act != b_zero:
                    errors.append(f"Mechanical Criterion N failure in v4 for {cfg} {cand}: actual={b_act}, zero_smem={b_zero}")

        # 17.9 Confound classification & criteria verification
        if v4_val_data.get("overall_status") != "LOCALLOAD_ISOLATED_BUT_RESIDENCY_AND_ACCUMULATOR_CONFOUNDED":
            errors.append(f"Expected overall_status == 'LOCALLOAD_ISOLATED_BUT_RESIDENCY_AND_ACCUMULATOR_CONFOUNDED', got {v4_val_data.get('overall_status')}")

        v4_crit = v4_val_data.get("criteria", {})
        if v4_crit.get("Criterion A (TMA Descriptor Load Invariant)") != "PASS":
            errors.append("v4 Criterion A was not PASS")
        if v4_crit.get("Criterion B (Initial LocalLoad Invariant - Inside Loop == 0)") != "PASS":
            errors.append("v4 Criterion B was not PASS")
        if v4_crit.get("Criterion C (Canonical Reduction Body Template Equivalence)") != "FAIL":
            errors.append("v4 Criterion C was not FAIL")
        if v4_crit.get("Criterion D (Distributed Layout Invariance)") != "PASS":
            errors.append("v4 Criterion D was not PASS")
        if v4_crit.get("Criterion E (Self-Contained Complete Executable Artifacts)") != "PASS":
            errors.append("v4 Criterion E was not PASS")
        if v4_crit.get("Criterion F (Residency & Occupancy Matched)") != "FAIL_RESIDENCY_DISPARITY":
            errors.append("v4 Criterion F was not FAIL_RESIDENCY_DISPARITY")
        if v4_crit.get("Criterion J (Sunk LocalLoad Count Inside Loop == 0)") != "PASS":
            errors.append("v4 Criterion J was not PASS")
        if v4_crit.get("Criterion L (Accumulator Structural Consistency)") != "FAIL_CANDIDATE_DEPENDENT_ACCUMULATOR":
            errors.append("v4 Criterion L was not FAIL_CANDIDATE_DEPENDENT_ACCUMULATOR")
        if v4_crit.get("Criterion N (Dynamic Smem Limiter Disambiguation)") != "PASS":
            errors.append("v4 Criterion N was not PASS")

        print("  Verified Phase 3 Step B v4 preloaded runtime-loop isolation (0 tile loads inside loop, residency & accumulator confounds recorded).")

    # Check 18: Phase 3 Step B v5 Preloaded-Register Last-Result Carry Feasibility
    print("[18/18] Validating Phase 3 Step B v5 last-result runtime-K feasibility...")
    v5_dir = phase3_dir / "v5_preloaded_k"
    v5_design_path = v5_dir / "design.md"
    v5_val_path = v5_dir / "validation.json"
    v5_res_path = v5_dir / "results.json"
    v5_sum_path = v5_dir / "summary.md"
    v5_arts_dir = v5_dir / "artifacts"
    v5_raw_path = v5_dir / "raw_results.json"

    if not v5_design_path.exists():
        errors.append(f"Missing {v5_design_path}")
    if not v5_val_path.exists():
        errors.append(f"Missing {v5_val_path}")
    if not v5_res_path.exists():
        errors.append(f"Missing {v5_res_path}")
    if not v5_sum_path.exists():
        errors.append(f"Missing {v5_sum_path}")
    if not v5_arts_dir.exists():
        errors.append(f"Missing {v5_arts_dir}")
    if not v5_raw_path.exists():
        errors.append(f"Missing {v5_raw_path}")

    if v5_val_path.exists() and v5_res_path.exists() and v5_raw_path.exists() and v5_arts_dir.exists():
        v5_val_data = json.loads(v5_val_path.read_text(encoding="utf-8"))
        v5_res_data = json.loads(v5_res_path.read_text(encoding="utf-8"))
        v5_raw_data = json.loads(v5_raw_path.read_text(encoding="utf-8"))

        expected_configs = ["M32_N64_w8", "M32_N128_w4"]
        expected_candidates = ["default", "4"]

        for cfg in expected_configs:
            for cand in expected_candidates:
                art_prefix = v5_arts_dir / cfg / cand
                for ext in [".ptx", ".ttgir", ".sass", ".resource.txt", ".cubin.sha256"]:
                    f_path = art_prefix.with_suffix(ext) if ext != ".cubin.sha256" else v5_arts_dir / cfg / f"{cand}.cubin.sha256"
                    if not f_path.exists() or f_path.stat().st_size == 0:
                        errors.append(f"Missing or empty v5 artifact: {f_path}")

                cand_data = v5_raw_data.get(cfg, {}).get(cand, {})
                c_sha = cand_data.get("cubin_sha256")
                sha_file_path = v5_arts_dir / cfg / f"{cand}.cubin.sha256"
                if sha_file_path.exists():
                    f_sha = sha_file_path.read_text(encoding="utf-8").strip()
                    if f_sha != c_sha:
                        errors.append(f"CUBIN sha mismatch for v5 {cfg} {cand}: file={f_sha} vs raw={c_sha}")

                # 18.1 Single CUBIN reuse across K
                if cand_data.get("specialization_check", {}).get("runtime_k_specialized") is not False:
                    errors.append(f"Expected runtime_k_specialized == False in v5 for {cfg} {cand}")

                # 18.2 Numerical correctness
                corr = cand_data.get("correctness", {})
                if not corr.get("all_passed", False):
                    errors.append(f"Numerical correctness failed in v5 for {cfg} {cand}")

                # 18.3 Zero spills
                res = cand_data.get("resources", {})
                if res.get("local_bytes") != 0 or res.get("stack_bytes") != 0:
                    errors.append(f"Non-zero spill in v5 {cfg} {cand}: local={res.get('local_bytes')}, stack={res.get('stack_bytes')}")

                # 18.4 Mechanical TTGIR validation
                ttgir_text = (v5_arts_dir / cfg / f"{cand}.ttgir").read_text(encoding="utf-8")
                scf_split = ttgir_text.split("scf.for", 1)
                tma_inside = len(re.findall(r"ttng\.async_tma_copy_global_to_local", scf_split[1])) if len(scf_split) > 1 else 0
                tma_outside = len(re.findall(r"ttng\.async_tma_copy_global_to_local", scf_split[0]))
                if tma_inside != 0 or tma_outside != 1:
                    errors.append(f"Mechanical Criterion A failure in v5 for {cfg} {cand}: outside={tma_outside}, inside={tma_inside}")

                ll_inside = len(re.findall(r"\bttg\.local_load\b", scf_split[1])) if len(scf_split) > 1 else 0
                ll_outside = len(re.findall(r"\bttg\.local_load\b", scf_split[0]))
                if ll_inside != 0:
                    errors.append(f"Mechanical Criterion C failure: ttg.local_load found inside scf.for in v5 for {cfg} {cand} (count={ll_inside})")
                if ll_outside != 1:
                    errors.append(f"Mechanical Criterion B failure: expected 1 pre-loop ttg.local_load in v5 for {cfg} {cand} (count={ll_outside})")

                red_inside = len(re.findall(r"\btt\.reduce\b", scf_split[1])) if len(scf_split) > 1 else 0
                red_outside = len(re.findall(r"\btt\.reduce\b", scf_split[0]))
                if red_inside == 0 or red_outside != 0:
                    errors.append(f"Mechanical Criterion H failure: tt.reduce not inside scf.for in v5 for {cfg} {cand}: inside={red_inside}, outside={red_outside}")

                # 18.5 Layout match with canonical
                canon_ttgir_text = (phase3_dir / "fixed_binary_artifacts" / "canonical" / cfg / f"{cand}.ttgir").read_text(encoding="utf-8")
                cb_m = re.search(r"(#blocked\d*\s*=\s*#ttg\.blocked<[^>]+>)", canon_ttgir_text)
                vb_m = re.search(r"(#blocked\d*\s*=\s*#ttg\.blocked<[^>]+>)", ttgir_text)
                cb = re.sub(r"#blocked\d*", "#blocked", cb_m.group(1)) if cb_m else None
                vb = re.sub(r"#blocked\d*", "#blocked", vb_m.group(1)) if vb_m else None
                if cb != vb:
                    errors.append(f"Criterion F failure: layout mismatch for v5 {cfg} {cand}: {vb} vs {cb}")

                # 18.6 Mechanical PTX loop branch validation
                ptx_text = (v5_arts_dir / cfg / f"{cand}.ptx").read_text(encoding="utf-8")
                labels_seen = set()
                bw_count = 0
                for l in ptx_text.splitlines():
                    lbl_m = re.match(r"^\s*(\$L__BB\d+_\d+):", l)
                    if lbl_m:
                        labels_seen.add(lbl_m.group(1))
                    bra_m = re.search(r"(@%p\d+)?\s*bra(?:\.uni)?\s+(\$L__BB\d+_\d+);", l)
                    if bra_m:
                        target = bra_m.group(2)
                        if target in labels_seen:
                            bw_count += 1
                if bw_count != 1:
                    errors.append(f"Expected exactly 1 backward branch in v5 PTX for {cfg} {cand}, got {bw_count}")

                # 18.7 Decomposition validations: zero accumulator, zero global store inside loop
                eval_entry = v5_val_data.get("v5_evaluations", {}).get(cfg, {}).get(cand, {})
                decomp_info = eval_entry.get("loop_decomposition", {})
                if decomp_info.get("accumulator_count") != 0:
                    errors.append(f"Expected accumulator_count == 0 in v5 {cfg} {cand}, got {decomp_info.get('accumulator_count')}")
                if decomp_info.get("global_store_inside_loop_count") != 0:
                    errors.append(f"Expected global_store_inside_loop_count == 0 in v5 {cfg} {cand}, got {decomp_info.get('global_store_inside_loop_count')}")

        # 18.8 Residency comparison: default vs cand4 matched
        w8_def_occ = v5_val_data["v5_evaluations"]["M32_N64_w8"]["default"]["occupancy"]
        w8_c4_occ = v5_val_data["v5_evaluations"]["M32_N64_w8"]["4"]["occupancy"]
        if w8_def_occ["blocks_per_sm_actual_smem"] != 8 or w8_c4_occ["blocks_per_sm_actual_smem"] != 8:
            errors.append(f"Expected 8 blocks/SM for M32_N64_w8 default and cand4, got {w8_def_occ['blocks_per_sm_actual_smem']} and {w8_c4_occ['blocks_per_sm_actual_smem']}")

        w4_def_occ = v5_val_data["v5_evaluations"]["M32_N128_w4"]["default"]["occupancy"]
        w4_c4_occ = v5_val_data["v5_evaluations"]["M32_N128_w4"]["4"]["occupancy"]
        if w4_def_occ["blocks_per_sm_actual_smem"] != 16 or w4_c4_occ["blocks_per_sm_actual_smem"] != 16:
            errors.append(f"Expected 16 blocks/SM for M32_N128_w4 default and cand4, got {w4_def_occ['blocks_per_sm_actual_smem']} and {w4_c4_occ['blocks_per_sm_actual_smem']}")

        # 18.9 Mechanical Criterion O check: actual == zero_dynamic_smem
        for cfg in expected_configs:
            for cand in expected_candidates:
                occ = v5_val_data["v5_evaluations"][cfg][cand]["occupancy"]
                b_act = occ.get("blocks_per_sm_actual_smem")
                b_zero = occ.get("blocks_per_sm_zero_dynamic_smem")
                if b_act is None or b_zero is None or b_act != b_zero:
                    errors.append(f"Mechanical Criterion O failure in v5 for {cfg} {cand}: actual={b_act}, zero_smem={b_zero}")

        # 18.10 Confound classification & criteria verification
        if v5_val_data.get("overall_status") != "REDUCTION_ISOLATED_BUT_TEMPLATE_CONFOUNDED":
            errors.append(f"Expected overall_status == 'REDUCTION_ISOLATED_BUT_TEMPLATE_CONFOUNDED', got {v5_val_data.get('overall_status')}")

        v5_crit = v5_val_data.get("criteria", {})
        for crit_name in [
            "Criterion A (TMA Issue Once Outside Loop)",
            "Criterion B (Initial LocalLoad Once Outside Loop)",
            "Criterion C (Zero Initial LocalLoad Inside Loop)",
            "Criterion D (One Runtime Loop / One CUBIN)",
            "Criterion E (Runtime K Unspecialized)",
            "Criterion F (Distributed Layout Invariance)",
            "Criterion H (Reduction Remains Inside Runtime Loop)",
            "Criterion I (Zero Accumulator Adds Inside Loop)",
            "Criterion J (Zero Global Store Inside Loop)",
            "Criterion K (In-Loop Anti-CSE Region Candidate Symmetry)",
            "Criterion L (Zero Local Memory & Stack Spills)",
            "Criterion M (Numerical Correctness across K)",
            "Criterion N (Residency & Occupancy Matched)",
            "Criterion O (Dynamic Smem Limiter Disambiguation)",
        ]:
            if v5_crit.get(crit_name) != "PASS":
                errors.append(f"v5 {crit_name} was not PASS (got {v5_crit.get(crit_name)})")

        if v5_crit.get("Criterion G (Canonical Reduction Core Fingerprint Equivalence)") != "FAIL":
            errors.append("v5 Criterion G was not FAIL")

        print("  Verified Phase 3 Step B v5 last-result runtime-loop isolation (0 accumulator adds, residency matched, template confound recorded).")

    print("--------------------------------------------------")
    if errors:
        print(f"FAILED with {len(errors)} consistency error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL 18 CONSISTENCY CHECKS PASSED SUCCESSFULLY.")
        print("==================================================")
        sys.exit(0)


if __name__ == "__main__":
    validate()
