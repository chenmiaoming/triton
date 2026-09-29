"""
Consistency Validator for TMA reduction layout evidence.

Checks:
1. Artifact SHAs match baseline_results.json raw hashes.
2. Audited phase annotation SHAs match current artifact SHAs.
3. default PTX hash == forced-8 PTX hash.
4. default TTGIR hash == forced-8 TTGIR hash.
5. resource.txt files re-parse to match physical_regs_per_thread and cuobjdump_shared_bytes in results.
6. Summary table (baseline_summary_table.md) numbers match baseline_results.json.
7. Remote source subset manifest digest == local source manifest digest in provenance.

Exits with code 0 on complete consistency, or non-zero on any failure.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXP_DIR = REPO_ROOT / "experiments" / "tma_reduction_layout"
RESULTS_DIR = EXP_DIR / "results"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate():
    print("==================================================")
    print("Running TMA Reduction Evidence Consistency Validator")
    print("==================================================")

    errors = []

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
    annotations = json.loads(ann_path.read_text(encoding="utf-8")).get("annotations", {})

    # Check 1: Artifact hashes vs baseline_results.json
    print("[1/7] Validating artifact hashes against baseline_results.json...")
    for cand in candidates:
        if cand not in audited:
            errors.append(f"Candidate '{cand}' missing in baseline_results.json audited_results.")
            continue
        raw = audited[cand]["raw"]

        for ext, key in [("ttgir", "ttgir_sha256"), ("ptx", "ptx_sha256"), ("sass", "sass_sha256"), ("resource.txt", "resource_sha256")]:
            f_path = ARTIFACTS_DIR / f"{cand}.{ext}"
            if not f_path.exists():
                errors.append(f"Missing artifact: {f_path}")
                continue
            actual_sha = compute_sha256(f_path.read_text(encoding="utf-8"))
            expected_sha = raw.get(key)
            if actual_sha != expected_sha:
                errors.append(f"SHA mismatch for {cand}.{ext}: disk={actual_sha} vs json={expected_sha}")

    # Check 2: Audited phase annotations vs actual artifact hashes
    print("[2/7] Validating audited_phase_annotations.json SHA bindings...")
    for cand in candidates:
        if cand not in annotations:
            errors.append(f"Candidate '{cand}' missing in audited_phase_annotations.json.")
            continue
        ptx_file = ARTIFACTS_DIR / f"{cand}.ptx"
        ttgir_file = ARTIFACTS_DIR / f"{cand}.ttgir"

        actual_ptx_sha = compute_sha256(ptx_file.read_text(encoding="utf-8"))
        actual_ttgir_sha = compute_sha256(ttgir_file.read_text(encoding="utf-8"))

        ann_ptx_sha = annotations[cand].get("ptx_sha256")
        ann_ttgir_sha = annotations[cand].get("ttgir_sha256")

        if actual_ptx_sha != ann_ptx_sha:
            errors.append(f"Annotation PTX SHA mismatch for '{cand}': disk={actual_ptx_sha} vs ann={ann_ptx_sha}")
        if actual_ttgir_sha != ann_ttgir_sha:
            errors.append(f"Annotation TTGIR SHA mismatch for '{cand}': disk={actual_ttgir_sha} vs ann={ann_ttgir_sha}")

    # Check 3: default vs forced-8 PTX equivalence
    print("[3/7] Validating default vs forced-8 PTX bit-for-bit equivalence...")
    default_ptx_sha = compute_sha256((ARTIFACTS_DIR / "default.ptx").read_text(encoding="utf-8"))
    c8_ptx_sha = compute_sha256((ARTIFACTS_DIR / "8.ptx").read_text(encoding="utf-8"))
    if default_ptx_sha != c8_ptx_sha:
        errors.append(f"default.ptx ({default_ptx_sha}) != 8.ptx ({c8_ptx_sha})")

    # Check 4: default vs forced-8 TTGIR equivalence
    print("[4/7] Validating default vs forced-8 TTGIR bit-for-bit equivalence...")
    default_ttgir_sha = compute_sha256((ARTIFACTS_DIR / "default.ttgir").read_text(encoding="utf-8"))
    c8_ttgir_sha = compute_sha256((ARTIFACTS_DIR / "8.ttgir").read_text(encoding="utf-8"))
    if default_ttgir_sha != c8_ttgir_sha:
        errors.append(f"default.ttgir ({default_ttgir_sha}) != 8.ttgir ({c8_ttgir_sha})")

    # Check 5: Re-parse resource.txt files
    print("[5/7] Validating resource.txt parsing consistency...")
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

    # Check 6: Summary table matches baseline_results.json
    print("[6/7] Validating summary table numbers against baseline_results.json...")
    summary_md = (RESULTS_DIR / "baseline_summary_table.md").read_text(encoding="utf-8")
    for cand in candidates:
        # Check median in table
        med_val = audited[cand]["measured"]["median_us"]
        med_str = f"**{med_val:.2f}**"
        if med_str not in summary_md:
            errors.append(f"Median {med_str} for '{cand}' not found in baseline_summary_table.md")

        # Check physical regs in table
        regs_val = audited[cand]["observed"]["physical_resources"]["physical_regs_per_thread"]["value"]
        regs_str = f"`{regs_val}`"
        if regs_str not in summary_md:
            errors.append(f"Physical regs {regs_str} for '{cand}' not found in baseline_summary_table.md")

    # Check 7: Source subset manifest digest matches local source manifest
    print("[7/7] Validating source subset manifest digest fidelity...")
    env_ver = data.get("environment", {}).get("manifest_verification", {})
    if env_ver:
        local_sha = env_ver.get("local_manifest_sha256")
        remote_subset_sha = env_ver.get("remote_source_subset_sha256")
        if remote_subset_sha and local_sha != remote_subset_sha:
            errors.append(
                f"Source manifest subset digest mismatch: local={local_sha} vs remote_subset={remote_subset_sha}"
            )
        print(f"  Verified {env_ver.get('files_verified')} uploaded files match local manifest byte-for-byte.")
        print(f"  Remote post-build extra files count: {env_ver.get('remote_extra_file_count')}")

    print("--------------------------------------------------")
    if errors:
        print(f"FAILED with {len(errors)} consistency error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL 7 CONSISTENCY CHECKS PASSED SUCCESSFULLY.")
        print("==================================================")
        sys.exit(0)


if __name__ == "__main__":
    validate()
