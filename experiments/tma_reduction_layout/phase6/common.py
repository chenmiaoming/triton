"""Phase 6 evidence utilities. Historical evidence is read-only."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/tma_reduction_layout"
OUT = BASE / "results/phase6"
BASELINE = "fbc9ae41eb1d79f196bb28b196039a4b3f9671e6"
sys.path.insert(0, str(ROOT))
B_VALUES = (16384, 32768, 65536)
R_VALUES = (0, 1, 2, 4, 8)
FAMILY = tuple(f"M{m}_N64_w8" for m in (32, 64, 128, 256, 512))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(value))


def require(value, message):
    if not value:
        raise ValueError(message)


def inventory(commit=BASELINE):
    tree = subprocess.check_output(["git", "ls-tree", "-rz", commit, "--",
                                    "experiments/tma_reduction_layout"], cwd=ROOT)
    result = {}
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        name = name.decode()
        data = (ROOT / name).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        require(blob == metadata.decode().split()[2], "Historical bytes changed: " + name)
        result[name] = sha(data)
    return result


def sign(values):
    mean, sd = st.mean(values), st.stdev(values)
    half = 4.302652729911275 * sd / math.sqrt(3)
    return {"mean": mean, "sample_SD": sd, "invocation_values": values,
            "half_width": half, "category": "POSITIVE" if mean > half else (
                "NEGATIVE" if mean < -half else "SIGN_UNRESOLVED")}


def ols(points):
    """Exact rational sums of the input floats; three-point grid-time OLS."""
    xs = [Fraction(x) for x, _ in points]
    ys = [Fraction(y) for _, y in points]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    b = sum((x-xm)*(y-ym) for x, y in zip(xs, ys)) / sum((x-xm)**2 for x in xs)
    a = ym - b*xm
    residuals = [y-a-b*x for x, y in zip(xs, ys)]
    sst = sum((y-ym)**2 for y in ys)
    return {"intercept_us": float(a), "slope_ns_per_additional_CTA": float(b*1000),
            "R_squared": float(1-sum(r*r for r in residuals)/sst) if sst else None,
            "residuals_us": list(map(float, residuals)), "points": points}


def raw_fits(raw):
    samples = {}
    for visit in raw["visits"]:
        samples.setdefault(visit["condition_tag"], []).extend(visit["samples_us"])
    require(all(len(v) == 100 for v in samples.values()), "100 scalar samples per condition")
    statistics = {tag: {"median_us": st.median(v), "mean_us": st.mean(v),
                       "sample_SD_us": st.stdev(v), "min_us": min(v), "max_us": max(v)}
                  for tag, v in samples.items()}
    groups = {}
    for tag, values in samples.items():
        cfg, harness, candidate, r, b = tag.split(":")
        groups.setdefault((cfg, harness, candidate, r), []).append((int(b[1:]), st.median(values)))
    fits = {":".join(k): ols(sorted(points)) for k, points in groups.items()}
    return statistics, fits


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join("---" for _ in headers) + " |"] +
                      ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])


def run_legacy_suite(destination):
    """Replay frozen stage-scoped validators; prove current prior bytes separately."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    prior = read(BASE / "phase4/results/phase5_timing/validator_suite.json")
    results = []
    paths = {}
    parent = Path(tempfile.mkdtemp(prefix="tma-phase6-legacy-"))
    try:
        for commit in (BASELINE, "56128d73233b234dacd191c569971b418680b857"):
            path = parent / commit[:8]
            subprocess.run(["git", "worktree", "add", "--detach", str(path), commit],
                           cwd=ROOT, check=True, capture_output=True)
            paths[commit] = path
        for index, item in enumerate(prior["validators"]):
            command = item["command"]
            historical = command[-1].endswith("phase5/validate_artifact_gate.py")
            commit = "56128d73233b234dacd191c569971b418680b857" if historical else BASELINE
            cwd = paths[commit]
            run = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT)
            log = destination / f"validator_{index+1:02}.log"
            log.write_text(run.stdout)
            results.append({"command": command, "return_code": run.returncode,
                            "log": log.relative_to(ROOT).as_posix(), "log_SHA256": sha(log.read_bytes()),
                            "scope": "UNCHANGED_CHECKER_AT_TRUSTED_STAGE_BASELINE", "checkout_commit": commit})
            print(f"Legacy validator {index+1}/{len(prior['validators'])}: {run.returncode}", flush=True)
            require(run.returncode == 0, "Legacy validator failed: " + str(command))
    finally:
        for path in paths.values():
                subprocess.run(["git", "worktree", "remove", str(path)], cwd=ROOT,
                               check=True, capture_output=True)
        parent.rmdir()
    result = {"status": "PASS", "validators": results,
              "historical_scope_note": "Frozen Phase 4 preregistration forbids additions outside Phase 4; Phase 5 Stage B forbids later timing trees. Unchanged checkers replayed at trusted stage baselines. Current protected inventory proves every prior file byte identical.",
              "protected_prior_files": len(inventory())}
    write(destination / "suite.json", result)
    return result
