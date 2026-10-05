"""Freeze architecture targets, retrospective cases, controls and decision rules."""
import argparse
from experiments.tma_reduction_layout.phase9 import common as c

SPECS = [
    (32, 64, 8, "original canonical positive case"),
    (32, 128, 4, "initial baseline, narrower lanes"),
    (32, 16, 4, "same-layout negative control: default already vector4"),
    (32, 256, 4, "wide tile, warp ownership changes"),
    (256, 64, 8, "Phase5 PRIMARY unresolved-sign counterexample"),
    (512, 64, 8, "Phase5 PRIMARY unresolved-sign counterexample"),
    (128, 32, 16, "Phase8 old positive descriptor-path diagnostic"),
    (1024, 32, 16, "Phase8 old near-zero/opposed gap diagnostic"),
    (2048, 32, 16, "Phase7 dominant influence, Phase8 descriptor diagnostic"),
    (128, 64, 32, "Phase8 new PRIMARY positive case"),
    (512, 128, 32, "Phase8 PRIMARY residual counterexample, large tile"),
    (32, 128, 8, "historical warp8 positive case"),
]
CONTROLS = {(32, 64, 8), (32, 256, 4), (128, 32, 16)}


def render():
    cases = []
    for m, n, w, reason in SPECS:
        vector = min(max(m*n//(w*32), 1), 8)
        cases.append({"case_id": f"M{m}_N{n}_w{w}", "M": m, "N": n, "num_warps": w,
            "selection_reason": reason, "historical_shape": True,
            "harnesses": ["reduction"] + (["load_copy", "store_copy"] if (m,n,w) in CONTROLS else []),
            "expected_default_vector": vector, "candidate4_legal": n % 4 == 0 and m*n//(w*32) >= 4})
    master = sorted(f'{x["case_id"]}:{h}:{candidate}:B{b}' for x in cases
                    for h in x["harnesses"] for candidate in c.CANDIDATES for b in c.B_VALUES)
    return {"phase": "9_CROSS_ARCHITECTURE_REPRODUCTION", "baseline": c.BASELINE,
        "targets": c.TARGETS, "cases": cases, "candidates": list(c.CANDIDATES),
        "B_DESC": c.B_DESC, "B_RUN": list(c.B_VALUES), "master_conditions": master,
        "input_dtype": "BF16", "reduction": "max over M; FP32 output; axis1 in [1,M,N]",
        "controls": {"load_copy": "device TMA descriptor load -> ordinary BF16 pointer store; no reduction",
                     "store_copy": "ordinary BF16 pointer load -> device TMA descriptor store; no reduction"},
        "same_source_across_targets": True, "per_target_CUBINs": True,
        "compiler": {"native": "reuse exact Phase6 core image/native SHA; no native/compiler change",
                     "candidate": "existing experimental Coalesce override; fresh JIT/cache directory per attempt",
                     "allowed_tool_repair": "newer cuobjdump/nvdisasm read-only disassembly; preserve first failure and tool hashes",
                     "no_forced_architecture": "compile for actual verified worker CC; strict target name/CC"},
        "artifact_gate": {"include_pair": "both candidates compile, archive ELF, checked occupancy and prefix4 correctness pass; complete PTX ABI available",
            "no_timing_exclusion": True, "no_replacement": True,
            "resource_strata": "matched zero-spill residency / changed residency / local memory or spill; all valid pairs timed",
            "structural_observations": "actual LocalLoad register/shared layouts, LocalLoad opcode signature, reduction phase binding; no cross-architecture equivalence assumption"},
        "timing": {"invocations_per_target": 3, "rounds": 10, "samples_per_visit": 10, "warmup_launches_per_condition": 3,
            "schedule": "cyclic master offsets (invocation-1)*17+(round-1)*13; stable eligible-pair filter",
            "primitive": "one archived CUBIN launch between CUDA driver events; compiler-free worker",
            "input": "seed42 contiguous torch.randn BF16 full descriptor extent initialized; unique(M,N) reused across warps; separate BF16 copy outputs",
            "SHA_guard": "archive and loaded buffer checked before every launch outside event interval",
            "correctness": "artifact smoke and archived-ABI warmup prefix4 with output reset outside timing",
            "UUID": "record actual worker UUID in artifacts and every timing sample visit; no attempts to force distinct physical cards",
            "repair": "at most three whole-invocation infrastructure repairs; original complete returns retained; no adverse-outcome reruns",
            "quota": "stop and ask user to switch profile; no automatic fallback"},
        "decision": {"unit": "one case/harness on one target; no pooled cross-architecture score",
            "primary_B_RUN": 65536, "effect_percent": "100*(median_default_us-median_cand4_us)/median_default_us per invocation",
            "band": "mean effect +/- t(df2)=4.302652729911275 * sampleSD/sqrt(3); descriptive three-process band, no familywise claim",
            "practical_percent": 3.0,
            "BENEFIT": "lower bound > +3 percent", "REGRESSION": "upper bound < -3 percent",
            "WITHIN_PRACTICAL_BAND": "both bounds within [-3,+3] percent; tested setup only",
            "UNRESOLVED": "all remaining complete pairs", "UNAVAILABLE": "pre-timing gate fails or target cannot be dispatched",
            "secondary": "same effects at B16384/B32768; rational OLS grid-time slopes, residuals and resource strata; no ranking within noise",
            "global_policy": "no global cand4 recommendation if any REGRESSION or unavailable/unresolved non-regression coverage; tested successes alone never prove universal safety",
            "scope": "retrospective shapes; first B200/SM120 measurements are cross-architecture evidence, not held-out validation of a fitted production policy"},
        "stop": "after architecture scope report and final validator; no production heuristic, native/compiler edit, PR creation (including draft), or later phase",
        "PR_permission": "user must approve before any PR is created or submitted",
        "historical_regression": "default-vs-cand4 is an optimization contrast; does not identify a historical good/bad compiler commit"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    protected = c.protect()
    root = c.OUT / "stage_a"
    data = render()
    summary = "# Phase9 StageA — Cross-architecture preregistration\n\n" + (
        "H100/SM90, B200/SM100 and RTX PRO6000 Blackwell/SM120 are separate strict targets. "
        "Twelve retrospective reduction identities and six non-reduction load/store controls are frozen before dispatch. "
        "No observed outcome can remove or replace a legal pair. Each target gets its own archived binaries.\n\n"
        "Primary outcome: full-kernel time at B_RUN65536; three invocation effects use a fixed3% practical band and df2 descriptive uncertainty. "
        "Grid16384/32768 and marginal OLS slopes are secondary. A missing/ambiguous result is not a zero or evidence of safety.\n\n"
        "The existing experimental vector4 override tests a candidate, not a production patch. "
        "Different targets may change LocalLoad lowering, registers and residency; all valid pairs are retained in explicit resource strata. "
        "Historical compiler regression attribution and held-out patch validation remain outstanding.\n\n"
        "Reuse the exact Phase6 native core and persistent ccache. Python-only work does not run make. "
        "If quota is exhausted stop and ask the user to switch profile. "
        "No compiler edit or PR (including draft); user approval is mandatory before PR creation. Stop after the scope report.\n\n")
    summary += c.table(["Case", "Warps", "Harnesses", "Reason"],
        [[x["case_id"],x["num_warps"],", ".join(x["harnesses"]),x["selection_reason"]] for x in data["cases"]]) + "\n"
    files = {"protocol.json": c.encode(data), "prior_inventory.json": c.encode(protected), "summary.md": summary.encode()}
    for name, blob in files.items():
        if args.validate:
            c.require((root / name).read_bytes() == blob, "Preregistration changed: " + name)
        else:
            root.mkdir(parents=True, exist_ok=True)
            c.require(not (root / name).exists(), "Never overwrite preregistration: " + name)
            (root / name).write_bytes(blob)
    print(f"StageA {'PASS' if args.validate else 'frozen'}; {len(protected)} prior files protected; 18 pairs/target")


if __name__ == "__main__":
    main()
