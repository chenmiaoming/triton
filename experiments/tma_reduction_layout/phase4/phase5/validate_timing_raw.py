"""Phase 5 Stage C raw fidelity only; no timing/scientific outcome thresholds."""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4.phase5 import timing_contract as tc


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
    paths = [Path(__file__).with_name("run_timing.py"), Path(tc.__file__),
        tc.ROOT / "experiments/tma_reduction_layout/phase4/archived_launch.py",
        tc.ROOT / "experiments/tma_reduction_layout/phase4/exact_cuda.py"]
    for path in paths:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                if any(n == "triton" or n.startswith("triton.") for n in names):
                    raise AssertionError("Timing runtime imports Triton: " + str(path))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("compile", "warmup", "do_bench"):
                raise AssertionError("Forbidden JIT/benchmark call: " + str(path))
    print("PASS: frozen 13-case cohort; 52 exact binaries; 234 conditions/invocation; 7020 visits; 70200 samples planned")
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
        ("nonfinite scalar sample", lambda r: r["visits"][0]["samples_us"].__setitem__(0, float("nan"))),
        ("nonpositive scalar sample", lambda r: r["visits"][0]["samples_us"].__setitem__(0, 0.0)),
        ("change descriptor allocation", lambda r: next(iter(r["allocations"].values())).update(input_bytes=1024)),
        ("remove launch SHA guard", lambda r: next(iter(r["loaded_binaries"].values())).update(sha_guard_count=0)),
        ("claim recompilation", lambda r: r.update(recompilation=True)),
        ("remove large-SMEM opt-in", lambda r: next(v for v in r["loaded_binaries"].values() if v["dynamic_smem_optin"]).update(dynamic_smem_optin=False)),
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
    print("PHASE 5 RAW VALIDATOR PASS; 10/10 corruption probes; 70200 scalar samples")


if __name__ == "__main__":
    main()
