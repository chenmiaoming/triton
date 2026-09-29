"""
Source provenance generator and verification helper for TMA reduction layout experiments.

Provides:
- Git metadata (HEAD SHA, branch, porcelain status).
- Binary diff SHA256 against HEAD.
- Tracking and hashing of untracked working-tree files.
- Deterministic source manifest SHA256 covering all source tree files uploaded to Modal.
- Remote container verification asserting byte-for-byte fidelity of /opt/triton-src.
"""

import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def get_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path("/home/chenmiaoming/triton-exp")


# Standard ignore patterns matching modal_runner.py's add_local_dir ignore list
DEFAULT_IGNORE_PREFIXES = {
    ".git",
    ".venv",
    "build",
    "experiments/tma_reduction_layout/results",
}

DEFAULT_IGNORE_NAMES = {
    "__pycache__",
    ".pytest_cache",
}

DEFAULT_IGNORE_SUFFIXES = {
    ".pyc",
    ".egg-info",
    ".so",
    ".a",
    ".o",
}


def is_ignored_path(
    rel_path: Path,
    ignore_prefixes: Set[str] = DEFAULT_IGNORE_PREFIXES,
    ignore_names: Set[str] = DEFAULT_IGNORE_NAMES,
    ignore_suffixes: Set[str] = DEFAULT_IGNORE_SUFFIXES,
) -> bool:
    parts = rel_path.parts
    # Check directory or file component names
    for p in parts:
        if p in ignore_names:
            return True
        for suff in ignore_suffixes:
            if p.endswith(suff):
                return True

    posix = rel_path.as_posix()
    for prefix in ignore_prefixes:
        if posix == prefix or posix.startswith(prefix + "/"):
            return True

    return False


def hash_file(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_source_manifest(repo_root: Path) -> Dict[str, str]:
    """
    Computes a deterministic mapping of relative_path -> sha256
    for all files in the source tree intended for Modal image build.
    """
    manifest: Dict[str, str] = {}
    for p in sorted(repo_root.rglob("*")):
        if p.is_file():
            try:
                rel = p.relative_to(repo_root)
            except ValueError:
                continue
            if is_ignored_path(rel):
                continue
            manifest[rel.as_posix()] = hash_file(p)
    return manifest


def compute_manifest_digest(manifest: Dict[str, str]) -> str:
    sorted_items = sorted(manifest.items())
    content = json.dumps(sorted_items, separators=(",", ":"))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def generate_provenance(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    if repo_root is None:
        repo_root = get_repo_root()

    def run_git(args: List[str]) -> str:
        try:
            return subprocess.check_output(
                ["git"] + args, cwd=repo_root, text=True, stderr=subprocess.PIPE
            ).strip()
        except Exception:
            return ""

    head_sha = run_git(["rev-parse", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status_porcelain = run_git(["status", "--porcelain"])

    # Binary diff against HEAD
    try:
        diff_bytes = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=repo_root
        )
    except Exception:
        diff_bytes = b""
    diff_sha256 = hashlib.sha256(diff_bytes).hexdigest()

    # Untracked files
    untracked_lines = run_git(["ls-files", "--others", "--exclude-standard"])
    untracked_files: List[Dict[str, str]] = []
    if untracked_lines:
        for u_path in untracked_lines.splitlines():
            u_p = repo_root / u_path
            if u_p.is_file():
                untracked_files.append({
                    "path": u_path,
                    "sha256": hash_file(u_p),
                })
    untracked_digest = hashlib.sha256(
        json.dumps(untracked_files, sort_keys=True).encode("utf-8")
    ).hexdigest()

    # Source manifest for complete working tree (uploaded to Modal)
    manifest = compute_source_manifest(repo_root)
    manifest_sha256 = compute_manifest_digest(manifest)

    # Composite digest covering HEAD + tracked diff + untracked files
    composite_raw = f"{head_sha}:{diff_sha256}:{untracked_digest}"
    composite_digest = hashlib.sha256(composite_raw.encode("utf-8")).hexdigest()

    is_dirty = bool(status_porcelain) or bool(untracked_files)

    provenance = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_head_sha": head_sha,
        "branch": branch,
        "is_dirty": is_dirty,
        "git_status_porcelain": status_porcelain.splitlines() if status_porcelain else [],
        "git_diff_head_sha256": diff_sha256,
        "untracked_files": untracked_files,
        "untracked_digest_sha256": untracked_digest,
        "composite_digest_sha256": composite_digest,
        "source_manifest_sha256": manifest_sha256,
        "source_manifest_file_count": len(manifest),
        "source_manifest": manifest,
        "repo_root": str(repo_root),
    }
    return provenance


def verify_remote_source_manifest(
    local_provenance: Dict[str, Any],
    remote_root: Path = Path("/opt/triton-src"),
) -> Dict[str, Any]:
    """
    Runs inside the remote Modal container.
    Verifies that all files in the local upload/source manifest exist at the exact
    same paths in remote_root with identical SHA256 hashes.
    Post-build extra files generated on the remote container are recorded separately
    and do not cause failure.
    """
    assert remote_root.exists(), f"Remote source directory {remote_root} does not exist!"

    local_manifest_sha256 = local_provenance.get("source_manifest_sha256", "")
    local_manifest = local_provenance.get("source_manifest", {})

    missing_files: List[str] = []
    mismatched_files: List[Dict[str, str]] = []
    remote_source_subset_manifest: Dict[str, str] = {}

    # Verify every file from the local source manifest
    for rel_path, local_hash in local_manifest.items():
        remote_file = remote_root / rel_path
        if not remote_file.exists():
            missing_files.append(rel_path)
        else:
            remote_hash = hash_file(remote_file)
            remote_source_subset_manifest[rel_path] = remote_hash
            if remote_hash != local_hash:
                mismatched_files.append({
                    "path": rel_path,
                    "local_sha256": local_hash,
                    "remote_sha256": remote_hash,
                })

    remote_source_subset_sha256 = compute_manifest_digest(remote_source_subset_manifest)

    # Compute full remote manifest to identify post-build generated files
    remote_full_manifest = compute_source_manifest(remote_root)
    remote_full_manifest_sha256 = compute_manifest_digest(remote_full_manifest)

    local_keys = set(local_manifest.keys())
    remote_full_keys = set(remote_full_manifest.keys())
    remote_extra_files = sorted(list(remote_full_keys - local_keys))

    verification_passed = (
        len(missing_files) == 0
        and len(mismatched_files) == 0
        and remote_source_subset_sha256 == local_manifest_sha256
    )

    if not verification_passed:
        err_msg = (
            f"CRITICAL: Remote uploaded source-file fidelity verification failed!\n"
            f"Missing files ({len(missing_files)}): {missing_files[:5]}\n"
            f"Mismatched files ({len(mismatched_files)}): {mismatched_files[:5]}\n"
            f"Local manifest digest: {local_manifest_sha256}\n"
            f"Remote subset digest:  {remote_source_subset_sha256}"
        )
        raise RuntimeError(err_msg)

    return {
        "status": "PASS",
        "uploaded_source_fidelity_verified": True,
        "verification_statement": (
            "Every file in the local upload/source manifest was found at the "
            "same path in /opt/triton-src and had identical bytes."
        ),
        "local_manifest_sha256": local_manifest_sha256,
        "remote_source_subset_sha256": remote_source_subset_sha256,
        "remote_full_manifest_sha256": remote_full_manifest_sha256,
        "files_verified": len(local_manifest),
        "missing_count": len(missing_files),
        "mismatched_count": len(mismatched_files),
        "remote_extra_file_count": len(remote_extra_files),
        "remote_extra_files": remote_extra_files,
    }


def save_provenance(provenance: Dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Strip full manifest from compact json dump to save space if needed
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)


if __name__ == "__main__":
    prov = generate_provenance()
    print(f"Git HEAD: {prov['git_head_sha']}")
    print(f"Branch: {prov['branch']}")
    print(f"Is Dirty: {prov['is_dirty']}")
    print(f"Untracked Count: {len(prov['untracked_files'])}")
    print(f"Manifest Files: {prov['source_manifest_file_count']}")
    print(f"Manifest SHA256: {prov['source_manifest_sha256']}")
