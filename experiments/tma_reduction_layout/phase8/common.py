"""Phase 8 evidence closure. All previous experiment bytes are immutable."""
from experiments.tma_reduction_layout.phase6 import common as prior

ROOT, BASE = prior.ROOT, prior.BASE
OUT = BASE / "results/phase8"
BASELINE = "7907562cae9d6fc76184ef17d07fb9e28c364ecf"
sha, encode, read, write, require, table = prior.sha, prior.encode, prior.read, prior.write, prior.require, prior.table
ols, sign = prior.ols, prior.sign
B_VALUES = (16384, 32768, 65536)
PATHS = ("canonical", "host_canonical", "device_canonical", "switch_host", "switch_device")
CANDIDATES = ("default", "4")
DERIVED = {"results.json", "summary.md", "validation.json", "raw_manifest.json", "gate.json",
           "launch_stage_c.json", "launch_stage_d.json", "analysis_validation.json", "suite.json"}


def protect():
    return prior.inventory(BASELINE)


def inventory(root):
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name not in DERIVED}


def freeze(root):
    require(not (root / "raw_manifest.json").exists(), "Never overwrite a raw freeze")
    write(root / "raw_manifest.json", {"files": inventory(root), "baseline": BASELINE,
          "protected_inventory_SHA256": sha(encode(protect()))})


def validate_freeze(root):
    stored = read(root / "raw_manifest.json")
    require(stored["files"] == inventory(root), "Frozen original bytes: " + str(root))
    require(stored["baseline"] == BASELINE and stored["protected_inventory_SHA256"] ==
            sha(encode(protect())), "Every pre-Phase8 experiment byte unchanged")


def tags(cases):
    return sorted(f"{cfg}:{path}:{candidate}:B{b}" for cfg in cases for path in PATHS
                  for candidate in CANDIDATES for b in B_VALUES)


def schedule(cases):
    master = tags({r["config_id"]: r for r in read(OUT / "stage_a/pool.json")["cases"]})
    invocations = []
    for invocation in (1, 2, 3):
        rounds = []
        for number in range(1, 11):
            offset = ((invocation - 1) * 17 + (number - 1) * 13) % len(master)
            order = [tag for tag in master[offset:] + master[:offset] if tag.split(":")[0] in cases]
            rounds.append({"round": number, "order": order, "samples_per_visit": 10})
        invocations.append({"invocation": invocation, "rounds": rounds})
    return {"invocations": invocations, "conditions": len(cases) * 30,
            "samples": len(cases) * 9000, "warmups": 3}


def condition(tag):
    cfg, path, candidate, b = tag.split(":")
    return {"case_id": cfg, "path": path, "candidate": candidate, "B_RUN": int(b[1:]),
            "mode": 1 if path == "switch_device" else 0 if path == "switch_host" else None}
