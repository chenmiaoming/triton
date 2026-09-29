"""
Source provenance generator and verification helper for TMA reduction layout experiments.
"""

import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict


def get_repo_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return Path("/home/chenmiaoming/triton-exp")


def generate_provenance(repo_root: Path = None) -> Dict[str, Any]:
    if repo_root is None:
        repo_root = get_repo_root()

    def run_git(args):
        return subprocess.check_output(["git"] + args, cwd=repo_root, text=True).strip()

    head_sha = run_git(["rev-parse", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status_porcelain = run_git(["status", "--porcelain"])

    # Compute SHA256 of git diff HEAD
    diff_bytes = subprocess.check_output(["git", "diff", "HEAD"], cwd=repo_root)
    diff_sha256 = hashlib.sha256(diff_bytes).hexdigest()

    is_dirty = bool(status_porcelain)

    provenance = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_head_sha": head_sha,
        "branch": branch,
        "is_dirty": is_dirty,
        "git_status_porcelain": status_porcelain.splitlines() if status_porcelain else [],
        "git_diff_head_sha256": diff_sha256,
        "repo_root": str(repo_root),
    }
    return provenance


def save_provenance(provenance: Dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)


if __name__ == "__main__":
    prov = generate_provenance()
    print(json.dumps(prov, indent=2))
