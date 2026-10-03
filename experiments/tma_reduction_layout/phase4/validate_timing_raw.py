"""Stage C raw fidelity only; no timing/scientific outcome thresholds."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4 import timing_contract as tc


def rejected(fn):
    try:
        fn()
    except (ValueError, KeyError, AssertionError):
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cases, plan, preview = tc.inputs()
    binaries = tc.binary_map(cases)
    print("PASS: frozen 18-case cohort; 72 exact binaries; 324 conditions/invocation; 9720 visits; 97200 samples planned")
    if args.preflight:
        print("STAGE C PREFLIGHT PASS; no timing or scientific analysis")
        return
    report = tc.validate_all()
    first = json.loads((tc.OUT / "raw_invocation_1.json").read_text())
    planned = plan["invocations"][0]
    tests = []
    for name, mutate in (
        ("remove one scalar sample", lambda r: r["visits"][0]["samples_us"].pop()),
        ("duplicate one condition visit", lambda r: r["visits"].insert(1, copy.deepcopy(r["visits"][0]))),
        ("change runtime CUBIN SHA", lambda r: r["visits"][0].update(runtime_loaded_cubin_sha256="0" * 64)),
        ("swap two schedule condition visits", lambda r: r["visits"].__setitem__(slice(0, 2), r["visits"][:2][::-1])),
    ):
        changed = copy.deepcopy(first)
        mutate(changed)
        if not rejected(lambda: tc.validate_invocation(changed, planned, cases, binaries)):
            raise AssertionError(name)
        tests.append({"probe": name, "result": "PASS"})
        print("PASS corruption:", name)
    environment = json.loads((tc.OUT / "environment.json").read_text())
    launches = json.loads((tc.OUT / "launch_bindings.json").read_text())
    for i in (1, 2, 3):
        raw = json.loads((tc.OUT / f"raw_invocation_{i}.json").read_text())
        if launches[str(i)] != raw["loaded_binaries"] or environment["invocations"][str(i)]["environment"] != raw["environment"]:
            raise AssertionError("Aggregate provenance/launch binding mismatch")
    report["corruption_probes"] = tests
    (tc.OUT / "raw_validation.json").write_bytes(tc.encode(report))
    print("STAGE C RAW VALIDATOR PASS; 4/4 corruption probes; 97200 scalar samples")


if __name__ == "__main__":
    main()
