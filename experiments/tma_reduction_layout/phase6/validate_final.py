"""Final evidence closure, including positive raw tampering in an isolated copy."""
import copy
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.tma_reduction_layout.phase6 import common as c
from experiments.tma_reduction_layout.phase6 import timing_contract as tc

DEST = c.OUT / "final_validation"
COMMANDS = (
    ("stage_a", "diagnose.py", "--validate"),
    ("stage_b", "contracts.py", "--validate"),
    ("stage_c_raw", "timing_contract.py", "stage_c"),
    ("stage_c_analysis", "analyze.py", "stage_c", "--validate"),
    ("stage_d_gate", "audit_fresh_gate.py", "--validate"),
    ("stage_d_raw", "timing_contract.py", "stage_d"),
    ("stage_d_analysis", "analyze.py", "stage_d", "--validate"),
)


def first_freeze(stage):
    root = c.OUT / stage
    relative = root.relative_to(c.ROOT).as_posix()
    commits = subprocess.check_output(["git", "log", "--diff-filter=A", "--format=%H", "--",
                                       relative + "/raw_manifest.json"], cwd=c.ROOT, text=True).splitlines()
    c.require(len(commits) == 1, "Unique first raw freeze: " + stage)
    commit = commits[0]
    tree = subprocess.check_output(["git", "ls-tree", "-rz", commit, "--", relative], cwd=c.ROOT)
    blobs = {}
    for entry in tree.split(b"\0"):
        if entry:
            meta, name = entry.split(b"\t", 1)
            blobs[name.decode()] = meta.decode().split()[2]
    files = set(c.read(root / "raw_manifest.json")["files"]) | {"raw_manifest.json"}
    digests = {}
    for name in sorted(files):
        data = (root / name).read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        c.require(blobs.get(relative + "/" + name) == digest, "First-commit raw bytes changed: " + stage + "/" + name)
        digests[name] = c.sha(data)
    return {"first_raw_commit": commit, "protected_files": len(files), "inventory_SHA256": c.sha(c.encode(digests))}


def isolated_probes():
    cases, binaries, plan = tc.inputs("stage_d")
    original = c.read(c.OUT / "stage_d/raw_invocation_1.json")
    altered = copy.deepcopy(original)
    sample = altered["visits"][0]["samples_us"][0]
    altered["visits"][0]["samples_us"][0] = math.nextafter(sample, math.inf)
    c.require(altered["visits"][0]["samples_us"][0] > sample, "Positive finite tamper must change the value")
    # This is still a structurally valid positive sample. Its rejection must
    # come from frozen byte identity rather than an artificial negative value.
    tc.validate_invocation(altered, plan["invocations"][0], cases, binaries)
    with tempfile.TemporaryDirectory(prefix="tma-phase6-raw-probe-") as directory:
        isolated = Path(directory)
        shutil.copytree(c.OUT / "stage_d", isolated / "stage_d")
        (isolated / "stage_d_gate").mkdir()
        shutil.copyfile(c.OUT / "stage_d_gate/launch_contract.json", isolated / "stage_d_gate/launch_contract.json")
        c.write(isolated / "stage_d/raw_invocation_1.json", altered)
        with patch.object(c, "OUT", isolated):
            try:
                tc.validate_raw("stage_d")
            except ValueError as exc:
                c.require(str(exc) == "All raw bytes frozen", "Positive tamper rejected for the intended reason")
            else:
                raise RuntimeError("Positive raw tamper accepted")
            contract = c.read(isolated / "stage_d_gate/launch_contract.json")
            order = contract["schedule"]["invocations"][0]["rounds"][0]["order"]
            order[0], order[1] = order[1], order[0]
            c.write(isolated / "stage_d_gate/launch_contract.json", contract)
            try:
                tc.inputs("stage_d")
            except ValueError as exc:
                c.require(str(exc) == "Frozen deterministic complete schedule", "Master-order tamper rejected for the intended reason")
            else:
                raise RuntimeError("Master-order tamper accepted")
    return ["POSITIVE_FINITE_RAW_SAMPLE_BYTE_TAMPER", "FRESH_MASTER_ORDER_TAMPER"]


def report(result):
    fresh = c.read(c.OUT / "stage_d/results.json")
    rows = [[scope, values["n"], predictor, error["MAE"], error["RMSE"]]
            for scope, values in fresh["predictive_comparisons"].items() if scope != "SECONDARY"
            for predictor, error in values["errors"].items()]
    hypotheses = [[scope, row["n"], row["H6_01_SINGLE_CONTEXT_TRANSFER"], row["H6_02_REPETITION_TREND_TRANSFER"]]
                  for scope, row in fresh["hypotheses"].items()]
    return "\n".join([
        "# Phase 6 completion and final evidence closure", "",
        "Stages A–D completed with separate commits. All prior protocols, samples, compiler exports, classifications, "
        "models and Phase 5 hypothesis statuses remain byte-identical.", "",
        "Stage A rederived 31 historical cases and 39 frozen prediction errors without GPU or refitting. "
        "Stage B froze five retrospective supplement cases, three harnesses, all five R values and the unseen warp16 domain before new timing. "
        "Stage C collected 63,000 archived-binary samples; its R curves reveal context dependence and departures from R0/R1 extrapolation. "
        "These old cases are diagnostic supplements, not a new confirmatory cohort.", "",
        "Stage D accounted for all 32 frozen source transitions: 14 structural exclusions uncompiled, "
        "18 cases / 108 binary attempts, one resource exclusion, nine PRIMARY and eight SECONDARY admitted before timing. "
        "All 17 eligible cases retained across canonical, single and repeated harnesses: 214,200 samples, "
        "three independently dispatched single-use process invocations. Two actual physical GPU UUIDs were observed; "
        "invocations 1 and 2 shared one GPU. No dispatch was repeated to obtain a different outcome or hardware UUID.", "",
        c.table(["Scope", "n", "H6_01 single S vs D1", "H6_02 beta vs D1"], hypotheses), "",
        c.table(["Scope", "n", "Fixed predictor", "MAE", "RMSE"], rows), "",
        "Error units: ns/additional CTA. Predictions use coefficient 1 and intercept 0; no cross-case refit or calibration. "
        "S improves over D1 within the frozen tested warp16 domain and retains substantial residual error. "
        "The five-point repetition trend beta worsens both frozen metrics. These comparisons do not isolate lane/warp "
        "communication, register allocation or scheduling. H2b/H2c remain UNVERIFIED; no production heuristic follows from this result.", "",
        "Persistent ccache migration retained 1,043 original files. The completed native-build transaction recorded "
        "136 cache hits and 251 misses (35.1421%); full before/after counters and actual native/compiler SHA identities are retained. "
        "The ZIP export repair reused the exact completed native core image. Core sources exclude experiments, "
        "so experiment Python iterations reuse that image. Timing C and D load archived CUBINs with no Triton import or compilation. "
        "Quota failure, stopped builds, cache-path correction and source-ZIP failure logs/partial returns are retained. "
        "MODAL_PROFILE was selected per command; the globally active profile was not changed.", "",
        f"Final validator: {result['status']}; seven Phase 6 validators, eleven unchanged historical validators "
        "replayed at their trusted stage baselines, two isolated tamper probes, and first-commit byte checks "
        f"for every new raw/artifact manifest. Current protected historical files: {result['protected_prior_files']}.", "",
        "[Supplement analysis](../stage_c/summary.md) · [Fresh admission](../stage_d_gate/summary.md) · "
        "[Fresh analysis](../stage_d/summary.md) · [Complete validator evidence](suite.json)", "",
        "Work stops at Phase 6 Stage D. No next phase has been started.", ""])


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    protected = c.inventory()
    prior_raw = {stage: first_freeze(stage) for stage in ("stage_c", "stage_d_gate", "stage_d")}
    legacy = c.read(DEST / "legacy/suite.json")
    c.require(legacy["status"] == "PASS" and len(legacy["validators"]) == 11, "All historical validators passed")
    for record in legacy["validators"]:
        c.require(record["return_code"] == 0 and record["scope"] == "UNCHANGED_CHECKER_AT_TRUSTED_STAGE_BASELINE"
                  and c.sha((c.ROOT / record["log"]).read_bytes()) == record["log_SHA256"], "Retained historical validator proof")
    validators = []
    for name, script, *arguments in COMMANDS:
        command = [sys.executable, "experiments/tma_reduction_layout/phase6/" + script, *arguments]
        run = subprocess.run(command, cwd=c.ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = DEST / (name + ".log")
        log.write_text(run.stdout)
        validators.append({"command": command, "return_code": run.returncode,
                           "log": log.relative_to(c.ROOT).as_posix(), "log_SHA256": c.sha(log.read_bytes())})
        print(name + ": " + str(run.returncode), flush=True)
        c.require(run.returncode == 0, "Final validator failed: " + name)
    probes = isolated_probes()
    c.require(c.inventory() == protected, "Historical byte identity through final verification")
    c.require({stage: first_freeze(stage) for stage in prior_raw} == prior_raw, "No original raw bytes changed by validators/probes")
    result = {"status": "PASS", "validated_HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.ROOT, text=True).strip(),
              "protected_prior_files": len(protected), "prior_inventory_SHA256": c.sha(c.encode(protected)),
              "first_commit_raw_protection": prior_raw, "validators": validators,
              "legacy_suite_SHA256": c.sha((DEST / "legacy/suite.json").read_bytes()),
              "legacy_scope": legacy["historical_scope_note"], "isolated_corruption_probes": probes,
              "samples": {"retrospective_supplement": 63000, "prospective_fresh": 214200},
              "stop_after": "PHASE_6_STAGE_D", "no_next_phase": True}
    c.write(DEST / "suite.json", result)
    (DEST / "summary.md").write_text(report(result))
    print("Final validation PASS; all original evidence bytes unchanged", flush=True)


if __name__ == "__main__":
    main()
