"""Stage B: frozen archived-binary supplement and unseen structural domain."""
from collections import Counter
import copy
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.tma_reduction_layout.phase6 import common as c
from experiments.tma_reduction_layout.phase4 import artifact_gate as ag
from experiments.tma_reduction_layout.phase4 import preregister as p4
from experiments.tma_reduction_layout.phase4.phase5.preregister import predicted_candidate

DEST = c.OUT / "stage_b"
STAGE_A_COMMIT = "a47887beaa"
FRESH_DOMAIN = {"M": [8, 16, 32, 64, 128, 256, 512, 1024],
                "N": [16, 32, 64, 128], "num_warps": [16]}
HARNESSES = ("canonical", "single", "repeated")
CANDIDATES = ("default", "4")


def abi(ptx):
    entry = re.search(r'\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid', ptx, re.S)
    c.require(entry is not None, "Full PTX entry ABI")
    params = [line for line in entry[1].splitlines() if ".param" in line]
    return ["descriptor128" if ".b8" in x and "[128]" in x else "u32" if ".u32" in x
            else "u64" if ".u64" in x else "UNKNOWN" for x in params]


def expected_abi(harness):
    if harness == "canonical":
        return ["u64", "u64", "u32", "u32", "u64", "u64"]
    tail = ["u64", "u64"] if harness == "single" else ["u32", "u64", "u64"]
    return ["descriptor128", "u32", "u32", "u32", "u64", "u64", "u64", "u64"] + tail


def binary_binding(gate, row, h, candidate):
    directory = gate / h / row["case_id"] / candidate
    path = directory / "kernel.cubin"
    ptx = (directory / "kernel.ptx").read_text()
    metadata = c.read(directory / "metadata.json")
    digest = c.sha(path.read_bytes())
    c.require(path.read_bytes().startswith(b"\x7fELF") and digest == row["binary_hashes"][h][candidate],
              "Actual archived CUBIN bytes match admitted binary")
    c.require(abi(ptx) == expected_abi(h), "Supported exact launch ABI: " + str(path))
    observed = ag.ttgir_contract((directory / "kernel.ttgir").read_text(),
                                 [1, row["M"], row["N"]], row["num_warps"])
    c.require(observed == row[h][candidate]["observed_layout"], "Observed full layout unchanged")
    occupancy = c.read(directory / "occupancy.json")
    c.require(occupancy["cubin_sha256"] == digest and occupancy["queried_cubin_sha256"] == digest,
              "Exact-binary occupancy binding")
    return {"archive_path": path.relative_to(c.ROOT).as_posix(), "archive_sha256": digest,
            "ptx_sha256": c.sha(ptx.encode()), "metadata": metadata, "ABI": abi(ptx),
            "shared_layout": observed["shared"], "occupancy": occupancy,
            "scratch_bytes_per_cta": 128 if h == "canonical" else 0}


def fresh_pool():
    rows = []
    for m in FRESH_DOMAIN["M"]:
        for n in FRESH_DOMAIN["N"]:
            w = 16
            rows.append({"config_id": f"M{m}_N{n}_w{w}", "M": m, "N": n, "num_warps": w,
                         "transition": "default -> cand4", "logical_shape": [1, m, n], "reduction_axis": 1,
                         "origin": "UNMEASURED_WARP16_EXTRAPOLATION",
                         "default": predicted_candidate(m, n, w), "cand4": predicted_candidate(m, n, w, 4)})
    selected = p4.select(sorted(rows, key=lambda row: row["config_id"]))
    return {"source_domain": FRESH_DOMAIN, "total_source_transitions": len(selected),
            "included_structural_pool": sum(r["included_in_structural_pool"] for r in selected),
            "excluded_structural_pool": sum(not r["included_in_structural_pool"] for r in selected),
            "transitions": selected, "role": "NEW_WARP_REGIME_PROSPECTIVE_EXTENSION",
            "selection": "Unmodified Phase 4 structural selector; analytic layout predictions only. No performance filter.",
            "timing_samples_seen": 0}


def protocol():
    return {"phase": "PHASE_6", "starting_baseline": c.BASELINE,
            "supplement_role": "OUTCOME_INFORMED_RETROSPECTIVE_MECHANISM_DIAGNOSTIC",
            "supplement_cases": list(c.FAMILY), "fresh_domain": FRESH_DOMAIN,
            "fresh_role": "PROSPECTIVE_WARP16_EXTRAPOLATION; no prior timing at these case identities",
            "dimensions": {"B_DESC": 65536, "B_RUN": list(c.B_VALUES), "R": list(c.R_VALUES),
                           "canonical_R": None, "single_R": None, "independent_invocations": 3,
                           "rounds": 10, "samples_per_visit": 10, "warmups_per_condition": 3},
            "admission": "Both candidate layouts follow the frozen structural selector; three actual harnesses pass full frozen family/load/loop/resource gates. PRIMARY exact sequence, SECONDARY full opcode multiset. No outcome-based upgrade.",
            "cross_harness_residency": "Report equality across all six case binaries separately. Different occupancy forbids single-factor attribution but does not erase valid predictive observations.",
            "timing": "Strict H100!:1; three independently dispatched single-use containers. CUDA events with exactly one archived kernel per sample. Every launch guarded by full CUBIN SHA. No Triton import or compilation.",
            "repair": "Only documented protocol/infrastructure failure permits replacing an entire invocation; preserve original return, partial samples, source ZIP and logs. No outcome-based repeats. Persist remote returns before retrieval.",
            "ordering": "Sorted full structural master conditions, cyclic deterministic rotations indexed by invocation and round. Eligibility only filters this order stably; no new seed or performance-based sorting.",
            "analysis": {"fit": "Median of 100 samples at each B; three-B OLS with rational input sums; retain intercept, slope, all R2/residuals. Unit ns/additional CTA, not single-CTA latency.",
                         "metrics": "G=canonical default-4 slope; S=single default-4; gR=repeated default-4 at R; DR=gR-g0; etaR=DR-R*D1 for R=2,4,8; beta=OLS slope of gR against R over all five points.",
                         "uncertainty": "Three invocation values plus mean/sample SD; fixed df2 t=4.302652729911275 sign bands. Unresolved is not practical equivalence.",
                         "scope": "No M/lane/warp/register causal decomposition. Single/repeated differences remain cross-harness context diagnostics. No posthoc model or coefficient fitting."},
            "fresh_hypotheses": {
                "H6_01_SINGLE_CONTEXT_TRANSFER": {"population": "ALL_ARTIFACT_ELIGIBLE_FRESH_CASES; case means across three independent invocations",
                    "prediction_S": "G_hat=S (coefficient 1, intercept 0)", "prediction_D": "G_hat=D1 (coefficient 1, intercept 0)",
                    "minimum_cases": 5, "SUPPORTED": "MAE_S<MAE_D AND RMSE_S<RMSE_D",
                    "FALSIFIED": "MAE_S>=MAE_D AND RMSE_S>=RMSE_D", "INCONCLUSIVE": "mixed ordering, undefined, invalid protocol, or n<5"},
                "H6_02_REPETITION_TREND_TRANSFER": {"population": "SAME_ALL_ARTIFACT_ELIGIBLE_FRESH_CASES",
                    "prediction_beta": "G_hat=beta (no cross-case calibration)", "prediction_D": "same D1 predictor as H6_01",
                    "minimum_cases": 5, "SUPPORTED": "MAE_beta<MAE_D AND RMSE_beta<RMSE_D",
                    "FALSIFIED": "MAE_beta>=MAE_D AND RMSE_beta>=RMSE_D", "INCONCLUSIVE": "mixed ordering, undefined, invalid protocol, or n<5"}},
            "fresh_strict_scope": "Report the same metrics separately for PRIMARY; PRIMARY n<5 is outcome-independent INCONCLUSIVE_BY_COVERAGE. n>=5 permits this test only within actual coverage, not universal generalization.",
            "thresholds": "Exact floating comparisons; no tuned tolerance, significance layer, adaptive repeats or case removal. Zero predictor error is also reported as a fixed descriptive reference.",
            "cache": "Reuse archived supplement CUBINs without building Triton. Fresh compilation uses serialized image builds, persistent triton-build-cache Volume, CCACHE_COMPILERCHECK=content and /cache/triton-home; save pre/post ccache stats. Workspace caches are separate.",
            "stop_after": "Stage D prospective analysis and final validation/push; no further stage automatically"}


def schedule(cases):
    tags = sorted(f"{cfg}:{h}:{candidate}:R{r}:B{b}" for cfg in cases for h in HARNESSES
                  for candidate in CANDIDATES for r in (c.R_VALUES if h == "repeated" else (None,))
                  for b in c.B_VALUES)
    invocations = []
    for invocation in (1, 2, 3):
        rounds = []
        for round_ in range(1, 11):
            offset = ((invocation-1)*17 + (round_-1)*13) % len(tags)
            rounds.append({"round": round_, "samples_per_visit": 10,
                           "order": tags[offset:] + tags[:offset]})
        invocations.append({"invocation": invocation, "rounds": rounds})
    return {"invocations": invocations, "conditions_per_invocation": len(tags),
            "scalar_samples": len(tags)*3*100, "warmups_per_condition": 3, "executed": False}


def derive():
    protected = c.inventory(STAGE_A_COMMIT)
    cases, binaries = {}, {}
    for cfg in c.FAMILY:
        phase = 5 if cfg.startswith(("M256_", "M512_")) else 4
        gate = c.BASE / "phase4/results" / ("phase5_artifact_gate" if phase == 5 else "artifact_gate")
        row = next(r for r in c.read(gate / "cohort_after_gate.json")["cases"] if r["case_id"] == cfg)
        c.require(row["final_class"] == "PRIMARY" and row["pre_timing_eligible"], "Frozen historical PRIMARY admission")
        cases[cfg] = {k: row[k] for k in ("case_id", "M", "N", "num_warps", "origin", "final_class", "lanePart_transition")}
        for h in HARNESSES:
            for candidate in CANDIDATES:
                binaries[f"{cfg}:{h}:{candidate}"] = binary_binding(gate, row, h, candidate)
    fresh = fresh_pool()
    old_ids = set(c.read(c.OUT / "stage_a/results.json")["recomputed_cases"])
    old_ids |= set(c.read(c.BASE / "results/phase2/sweep/results.json")["configs"])
    c.require(not old_ids.intersection(r["config_id"] for r in fresh["transitions"]), "Fresh cohort identities never previously timed")
    return {"cases": cases, "binaries": binaries, "protocol": protocol(), "schedule": schedule(cases),
            "fresh_structural_pool": fresh, "prior_inventory_SHA256": c.sha(c.encode(protected)),
            "protected_files": len(protected), "stage_a_commit": subprocess.check_output(
                ["git", "rev-parse", STAGE_A_COMMIT], cwd=c.ROOT, text=True).strip()}


def main():
    expected = derive()
    if "--validate" in sys.argv:
        stored = c.read(DEST / "contracts.json")
        c.require(stored == expected, "All Stage B bindings/protocol/schedules independently regenerate")
        for name, key in (("protocol.json", "protocol"), ("fresh_structural_pool.json", "fresh_structural_pool"),
                          ("schedule.json", "schedule")):
            c.require(c.read(DEST / name) == expected[key], "Frozen file closure: " + name)
        probes = []
        for name, mutate in (("binary", lambda r: next(iter(r["binaries"].values())).update(archive_sha256="0"*64)),
                             ("schedule", lambda r: r["schedule"]["invocations"][0]["rounds"][0]["order"].pop()),
                             ("domain", lambda r: r["fresh_structural_pool"]["transitions"].pop()),
                             ("predictor", lambda r: r["protocol"]["fresh_hypotheses"]["H6_01_SINGLE_CONTEXT_TRANSFER"].update(prediction_S="refit"))):
            bad = copy.deepcopy(stored)
            mutate(bad)
            try:
                c.require(bad == expected, "All Stage B bindings/protocol/schedules independently regenerate")
            except ValueError:
                probes.append(name)
            else:
                raise RuntimeError("Corruption accepted: " + name)
        report = {"status": "PASS", "archived_binaries": len(expected["binaries"]),
                  "supplement_samples": expected["schedule"]["scalar_samples"],
                  "fresh_source_cases": len(expected["fresh_structural_pool"]["transitions"]),
                  "corruption_probes": probes, "no_GPU_or_compilation": True}
        c.write(DEST / "validation.json", report)
        print(report)
    else:
        c.write(DEST / "contracts.json", expected)
        for name, key in (("protocol.json", "protocol"), ("fresh_structural_pool.json", "fresh_structural_pool"),
                          ("schedule.json", "schedule")):
            c.write(DEST / name, expected[key])
        (DEST / "summary.md").write_text("# Phase 6 Stage B — Frozen contracts\n\n"
            "30 actual archived CUBINs: five already-observed PRIMARY cases × three harnesses × two candidates. "
            "All full PTX ABIs, actual layouts and exact-binary occupancy bindings checked without GPU or compilation.\n\n"
            "Supplement: 210 conditions per invocation; three invocations; 63,000 scalar samples. "
            "Single and canonical R=null; repeated runtime R=0,1,2,4,8 shares one archived binary.\n\n"
            "Fresh cohort is frozen now, before supplementary timing: 32 transitions in "
            "M={8,16,32,64,128,256,512,1024}, N={16,32,64,128}, warps=16. "
            f"{expected['fresh_structural_pool']['included_structural_pool']} structurally included; "
            "actual artifacts/resources still pending. Entire fixed domain accounted for, without performance filtering.\n\n"
            "Two prospective, uncalibrated comparisons are frozen: single-context S versus D1, and "
            "five-R repetition slope beta versus the same D1. Both use case-level MAE/RMSE, no refit. "
            "All-eligible predictive scope and strict PRIMARY scope remain distinct.\n\n"
            "No new samples, compiler artifacts or hypothesis outcomes produced at this stage. "
            "Old protocols and evidence remain byte-identical.\n")
        print("Stage B contracts derived; no GPU or compilation")


if __name__ == "__main__":
    main()
