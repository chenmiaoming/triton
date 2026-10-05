"""Phase 9 cross-architecture evidence; every earlier experiment is read-only."""
from pathlib import Path
from experiments.tma_reduction_layout.phase6 import common as prior

ROOT, BASE = prior.ROOT, prior.BASE
OUT = BASE / "results/phase9"
BASELINE = "af64d2f478f3d09c1fb1966307cd7a8e86293fd1"
sha, encode, read, write, require, table = prior.sha, prior.encode, prior.read, prior.write, prior.require, prior.table
ols = prior.ols
TARGETS = {
    "sm90": {"gpu": "H100!:1", "name_contains": "H100", "cc": [9, 0]},
    "sm100": {"gpu": "B200:1", "name_contains": "B200", "cc": [10, 0]},
    "sm120": {"gpu": "RTX-PRO-6000:1", "name_contains": "RTX PRO 6000", "cc": [12, 0]},
}
B_DESC = 65536
B_VALUES = (16384, 32768, 65536)
CANDIDATES = ("default", "4")
DERIVED = {"gate.json", "results.json", "summary.md", "validation.json", "raw_manifest.json", "suite.json"}


def protect():
    return prior.inventory(BASELINE)


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in sorted(Path(root).rglob("*"))
            if p.is_file() and p.name not in DERIVED}


def freeze(root):
    require(not (Path(root) / "raw_manifest.json").exists(), "Never overwrite a raw freeze")
    write(Path(root) / "raw_manifest.json", {"files": inventory(root), "baseline": BASELINE,
          "protected_inventory_SHA256": sha(encode(protect()))})


def validate_freeze(root):
    record = read(Path(root) / "raw_manifest.json")
    require(record["files"] == inventory(root), "Frozen original bytes: " + str(root))
    require(record["baseline"] == BASELINE and record["protected_inventory_SHA256"] == sha(encode(protect())),
            "Every pre-Phase9 experiment byte unchanged")


def protocol():
    return read(OUT / "stage_a/protocol.json")


def condition(tag):
    cfg, harness, candidate, b = tag.split(":")
    return {"case_id": cfg, "harness": harness, "candidate": candidate, "B_RUN": int(b[1:])}


def schedule(pair_keys):
    # Rotate the full predeclared master; stable eligibility filtering only.
    master = protocol()["master_conditions"]
    keep = set(pair_keys)
    invocations = []
    for invocation in (1, 2, 3):
        rounds = []
        for number in range(1, 11):
            offset = ((invocation - 1) * 17 + (number - 1) * 13) % len(master)
            order = [tag for tag in master[offset:] + master[:offset]
                     if ":".join(tag.split(":")[:2]) in keep]
            rounds.append({"round": number, "order": order, "samples_per_visit": 10})
        invocations.append({"invocation": invocation, "rounds": rounds})
    return {"invocations": invocations, "conditions": len(keep) * 6,
            "samples": len(keep) * 1800, "warmups": 3}
