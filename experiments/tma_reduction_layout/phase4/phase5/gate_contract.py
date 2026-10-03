"""Phase 5 Stage B population and source contracts; no outcome/model inputs."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "experiments/tma_reduction_layout"
BASELINE = "113748ae8f8e4817f8070847156f885c7a92a860"
PREREG = BASE / "phase4/results/phase5/preregistration"
# A sibling preserves the frozen preregistration validator's exact directory inventory.
OUT = BASE / "phase4/results/phase5_artifact_gate"
FROZEN = ("structural_pool.json", "exclusions.json", "protocol.json", "cohort_summary.md", "source_bindings.json")
HARNESSES = ("canonical", "single", "repeated")
CANDIDATES = ("default", "4")
DERIVED = {"gate_results.json", "cohort_after_gate.json", "hypothesis_feasibility.json", "archive_bindings.json",
    "summary.md", "validator_report.json", "derivation_provenance.json", "eligible_schedule_preview.json", "validator_suite.json"}


def sha(data): return hashlib.sha256(data).hexdigest()
def encode(value): return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
def read(path): return json.loads(path.read_text())
def require(test, message):
    if not test: raise ValueError(message)


def protected_inventory():
    tree = subprocess.check_output(["git", "ls-tree", "-rz", BASELINE, "--", "experiments/tma_reduction_layout"], cwd=ROOT)
    result = {}
    for entry in tree.split(b"\0"):
        if not entry: continue
        metadata, name = entry.split(b"\t", 1)
        path = name.decode(); data = (ROOT / path).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        require(blob == metadata.decode().split()[2], "Protected baseline evidence changed: " + path)
        result[path] = sha(data)
    return result


def population(pool):
    """Explicit whitelist: never expose development outcomes/hypothesis/model fields."""
    require(pool["total_source_transitions"] == 24 and pool["included_structural_pool"] == 19
        and pool["excluded_structural_pool"] == 5, "Frozen 24/19/5 population")
    rows = []
    for original in pool["transitions"]:
        if not original["included_in_structural_pool"]: continue
        row = {k: original[k] for k in ("config_id", "M", "N", "num_warps", "logical_shape", "reduction_axis", "origin")}
        for c in ("default", "cand4"):
            row[c] = {k: original[c][k] for k in ("legal", "layout", "compiled_num_warps", "num_ctas", "hardware_warp_size",
                "shared_family", "shared_rank", "lanePart_M", "warpPart_M", "vec")}
        rows.append(row)
    require(len(rows) == 19 and len({r["config_id"] for r in rows}) == 19, "All 19 included exactly once")
    return sorted(rows, key=lambda r: r["config_id"])


def master_schedule(pool):
    from experiments.tma_reduction_layout.phase4.preregister import rotation_schedule
    # Stage A deterministic order/rotation rule; only structural rows enter this helper.
    rows = [{**r, "included_in_structural_pool": True} for r in population(pool)]
    schedule = rotation_schedule({"transitions": rows})
    schedule.update({"phase": "PHASE_5", "preview_only": True, "executed": False,
        "warmup_count_per_condition": 3, "B_DESC": 65536})
    return schedule
