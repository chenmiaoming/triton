"""Freeze completed raw evidence after the CLI log is closed, before analysis."""
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.tma_reduction_layout.phase7 import common as c
from experiments.tma_reduction_layout.phase7 import timing_contract as tc


def main():
    stage = sys.argv[1]
    root = c.OUT / stage
    if len(sys.argv) > 2:
        log = root / "dispatch_logs" / Path(sys.argv[2]).name
        log.parent.mkdir(exist_ok=True)
        c.require(not log.exists(), "Never overwrite a retained dispatch log")
        shutil.copyfile(sys.argv[2], log)
    c.require(not (root / "raw_manifest.json").exists(), "Raw evidence already frozen")
    cases, binaries, plan = tc.inputs(stage)
    for p in plan["invocations"]:
        tc.validate_invocation(c.read(root / f'raw_invocation_{p["invocation"]}.json'), p, cases, binaries)
    c.write(root / "raw_manifest.json", {"files": tc.raw_inventory(root),
        "prior_inventory_SHA256": c.sha(c.encode(c.inventory())),
        "protocol_SHA256": c.sha((c.OUT / "stage_b_prereg/protocol.json").read_bytes()),
        "source_HEAD": c.read(root / "source_bindings.json")["provenance"]["git_head_sha"],
        "role": "RAW_FROZEN_BEFORE_ANALYSIS"})
    print(tc.validate_raw(stage))


if __name__ == "__main__":
    main()
