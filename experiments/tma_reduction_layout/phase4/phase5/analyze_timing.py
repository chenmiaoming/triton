"""Held-out case-level predictions with preregistered coefficients; no refitting."""
from collections import Counter
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4.phase5 import timing_contract as tc
from experiments.tma_reduction_layout.phase4.analyze_timing import stats, fit, structure
from experiments.tma_reduction_layout.phase4.analysis_contract import sign_resolution

IDS = ("H5_01_EXACT_BODY_DIRECTIONAL_TRACKING", "H5_02_FIXED_DIFFERENTIAL_ALONE_INSUFFICIENT",
    "H5_03_WARP_REGIME_CONTEXT_PREDICTION")


def raw_commit():
    commit = subprocess.check_output(["git", "log", "--format=%H", "--max-count=1",
        "--grep=^experiment: record Phase 5 held-out timing$"], cwd=ROOT, text=True).strip()
    tc.require(bool(commit), "Commit raw timing before any held-out hypothesis evaluation")
    for name in tc.raw_names(tc.OUT) + ["raw_manifest.json"]:
        relative = (tc.OUT / name).relative_to(ROOT).as_posix()
        committed = subprocess.check_output(["git", "show", commit + ":" + relative], cwd=ROOT)
        tc.require(committed == (tc.OUT / name).read_bytes(), "Raw commit bytes changed: " + name)
    return commit


def locked_models():
    protocol_path = tc.PREREG / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    models = protocol["locked_development_models"]
    expected = {"A": {"intercept", "D"}, "B": {"intercept", "D", "g0"}, "C": {"intercept", "D", "g0", "warp8"}}
    tc.require(set(models) == set(expected) and all(set(models[m]) == keys for m, keys in expected.items()), "Locked model terms")
    tc.require(all(type(v) in (int, float) and math.isfinite(v) for m in models.values() for v in m.values()), "Finite locked coefficients")
    binding = json.loads((tc.PREREG / "source_bindings.json").read_text())
    source = "experiments/tma_reduction_layout/phase4/results/residual_analysis/results.json"
    digest = binding["source_SHA256"][source]
    tc.require(tc.sha((ROOT / source).read_bytes()) == digest, "Frozen Stage D source-result SHA")
    return models, {"coefficients": models, "protocol_SHA256": tc.sha(protocol_path.read_bytes()),
        "Stage_D_source_result_path": source, "Stage_D_source_result_SHA256": digest,
        "preregistration_source_bindings_SHA256": tc.sha((tc.PREREG / "source_bindings.json").read_bytes()),
        "coefficient_source": "Frozen Phase 5 protocol.json; no refit/calibration"}


def prediction(models, d, g0, warp8):
    values = {"D": d, "g0": g0, "warp8": warp8}
    return {name: model["intercept"] + sum(model[k] * values[k] for k in ("D", "g0", "warp8") if k in model)
        for name, model in models.items()}


def errors(rows):
    return {m: {"n": len(rows), "unit": "one held-out case; three-invocation means",
        "MAE": st.mean(r["predictions"][m]["abs_error"] for r in rows),
        "RMSE": math.sqrt(st.mean(r["predictions"][m]["squared_error"] for r in rows))}
        for m in ("A", "B", "C")}


def decisions(metrics, primary_n, warp_counts, n, valid_protocol=True):
    finite = valid_protocol and all(v is not None and math.isfinite(v)
        for m in metrics.values() for k, v in m.items() if k in ("MAE", "RMSE"))
    h2 = h3 = "INCONCLUSIVE"
    if finite and n >= 5:
        a, b = metrics["A"], metrics["B"]
        if b["MAE"] >= a["MAE"] and b["RMSE"] >= a["RMSE"]: h2 = "SUPPORTED"
        elif b["MAE"] < a["MAE"] and b["RMSE"] < a["RMSE"]: h2 = "FALSIFIED"
    if finite and all(warp_counts.get(str(w), 0) >= 3 for w in (4, 8)):
        b, c = metrics["B"], metrics["C"]
        if c["MAE"] < b["MAE"] and c["RMSE"] < b["RMSE"]: h3 = "SUPPORTED"
        elif c["MAE"] >= b["MAE"] and c["RMSE"] >= b["RMSE"]: h3 = "FALSIFIED"
    return {
        IDS[0]: {"status": "INCONCLUSIVE_BY_COVERAGE", "PRIMARY_n": primary_n, "required": 5,
            "pre_timing_feasibility": "INCONCLUSIVE_BY_COVERAGE_BEFORE_TIMING", "Spearman_computed": False},
        IDS[1]: {"status": h2, "n": n, "population": "ALL_TIMING_ELIGIBLE_HELD_OUT_CASES",
            "Model_A": metrics["A"], "Model_B": metrics["B"]},
        IDS[2]: {"status": h3, "n": n, "population": "ALL_TIMING_ELIGIBLE_HELD_OUT_CASES",
            "eligible_by_warps": warp_counts, "Model_B": metrics["B"], "Model_C": metrics["C"]}}


def derive(raws, cases, models):
    statistics, fits, runs, fit_records = {}, {}, {}, []
    for raw in raws:
        inv = str(raw["invocation"])
        samples = {}
        for v in raw["visits"]: samples.setdefault(v["condition_tag"], []).extend(v["samples_us"])
        tc.require(len(samples) == tc.EXPECTED_COUNTS["conditions_per_invocation"]
            and all(len(v) == 100 for v in samples.values()), "Full raw condition cardinality")
        statistics[inv] = {tag: stats(v) for tag, v in samples.items()}
        fits[inv], runs[inv] = {}, {}
        for cfg in sorted(cases):
            fits[inv][cfg] = {"canonical": {}, "repeated": {}}
            slopes = {}
            for h in ("canonical", "repeated"):
                for c in ("default", "4"):
                    if h == "repeated": fits[inv][cfg][h][c] = {}
                    for r in ((None,) if h == "canonical" else (0, 1)):
                        points = [(b, statistics[inv][f"{cfg}:{h}:{c}:R{r}:B{b}"]["median_us"])
                            for b in (16384, 32768, 65536)]
                        f = fit(points)
                        if h == "canonical": fits[inv][cfg][h][c] = f
                        else: fits[inv][cfg][h][c][str(r)] = f
                        slopes[h, c, r] = f["slope_ns_per_additional_CTA"]
                        fit_records.append({"invocation": int(inv), "case_id": cfg, "harness": h,
                            "candidate": c, "R": r, "R_squared": f["R_squared"], "residuals_us": f["residuals_us"]})
            G = slopes["canonical", "default", None] - slopes["canonical", "4", None]
            g0 = slopes["repeated", "default", 0] - slopes["repeated", "4", 0]
            g1 = slopes["repeated", "default", 1] - slopes["repeated", "4", 1]
            runs[inv][cfg] = {"G": G, "g0": g0, "g1": g1, "D": g1 - g0, "G_minus_D": G - (g1 - g0)}
    rows = {}
    for cfg, case in sorted(cases.items()):
        row = {"case_id": cfg, "class": case["final_class"], "origin": case["origin"],
            **{k: case[k] for k in ("M", "N", "num_warps", "lanePart_transition")},
            "structural_attributes": structure(case), "warp8_indicator": int(case["num_warps"] == 8),
            "invocation_differentials": {i: runs[i][cfg] for i in runs}}
        for metric in ("G", "g0", "g1", "D", "G_minus_D"):
            values = [runs[str(i)][cfg][metric] for i in (1, 2, 3)]
            row[metric] = {**sign_resolution(values), "invocation_values": values}
        g, d = row["G"]["category"], row["D"]["category"]
        row["sign_relationship"] = "BOTH_SIGN_UNRESOLVED" if g == d == "SIGN_UNRESOLVED" else (
            "ONE_SIGN_UNRESOLVED" if "SIGN_UNRESOLVED" in (g, d) else "RESOLVED_AGREEMENT" if g == d else "RESOLVED_OPPOSITION")
        row["predictions"] = {}
        for model, pred in prediction(models, row["D"]["mean"], row["g0"]["mean"], row["warp8_indicator"]).items():
            e = row["G"]["mean"] - pred
            row["predictions"][model] = {"G_hat": pred, "error_G_minus_prediction": e,
                "abs_error": abs(e), "squared_error": e * e}
        for h in ("canonical", "repeated"):
            row["lowest_" + h + "_R_squared"] = min((f["R_squared"] for f in fit_records
                if f["case_id"] == cfg and f["harness"] == h and f["R_squared"] is not None), default=None)
        rows[cfg] = row
    metrics = errors(list(rows.values()))
    primary_n = sum(r["class"] == "PRIMARY" for r in rows.values())
    warp_counts = dict(sorted(Counter(str(r["num_warps"]) for r in rows.values()).items()))
    hypotheses = decisions(metrics, primary_n, warp_counts, len(rows))
    ranked = [{"case_id": n, "model": m, "class": r["class"], "M": r["M"], "N": r["N"],
        "num_warps": r["num_warps"], **r["predictions"][m]} for n, r in rows.items() for m in ("A", "B", "C")]
    ranked.sort(key=lambda r: (-r["abs_error"], r["case_id"], r["model"]))
    groups = {"PRIMARY": [r for r in rows.values() if r["class"] == "PRIMARY"],
        "SECONDARY": [r for r in rows.values() if r["class"] == "SECONDARY"]}
    groups.update({f"warps{w}": [r for r in rows.values() if r["num_warps"] == w] for w in (4, 8)})
    groups.update({f"M{m}": [r for r in rows.values() if r["M"] == m] for m in (16, 256, 512)})
    return {"units": "ns/additional CTA; marginal grid-time slope, not single-CTA latency",
        "condition_statistics": statistics, "fits": fits, "cases": rows, "prediction_errors": metrics,
        "hypotheses": hypotheses, "all_prediction_errors_ranked": ranked, "top_five_prediction_errors": ranked[:5],
        "descriptive_strata": {g: {"n": len(v), "case_ids": sorted(r["case_id"] for r in v),
            "prediction_errors": errors(v), "interpretation": "DESCRIPTIVE_ONLY; confirmatory decisions use all 13 eligible cases"}
            for g, v in groups.items()},
        "fit_diagnostics": {"all_fit_count": len(fit_records), "lowest_five": {h: sorted(
            [f for f in fit_records if f["harness"] == h and f["R_squared"] is not None],
            key=lambda r: (r["R_squared"], r["case_id"], r["invocation"], r["candidate"], str(r["R"])))[:5]
            for h in ("canonical", "repeated")}, "handling": "All fits and cases retained; no R2 threshold or outcome-based retry"},
        "scientific_status": {"Phase5_role": "HELD_OUT_CONFIRMATORY_EXTENSION",
            "Phase4_primary_generalization": "NOT_ESTABLISHED_PRIMARY_N3",
            "H2a_historical": "SUPPORTED_AT_REDUCTION_BODY_LEVEL", "H2b": "UNVERIFIED", "H2c": "UNVERIFIED",
            "model_refit": False, "production_heuristic": False, "PR_11991_modified": False,
            "residual_interpretation": "G-D is a cross-harness descriptive residual, not a causal remainder",
            "sign_unresolved_interpretation": "SIGN_UNRESOLVED is not practical equivalence"}}


def report(result, execution):
    def num(v): return "UNDEFINED" if v is None else format(v, ".12g")
    def metric(v): return num(v["mean"]) + " ± " + num(v["sample_SD"])
    def table(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
            + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows])
    rows = result["cases"]
    environments = json.loads((tc.OUT / "environment.json").read_text())["invocations"]
    text = ["# PHASE 5 HELD-OUT TIMING REPORT", "",
        "Starting HEAD: `" + tc.BASELINE + "`. Raw commit: `" + result["lineage"]["raw_timing_commit"] + "`.", "",
        "13 eligible cases: 2 PRIMARY, 11 SECONDARY. Six artifact exclusions absent. Three independent single-use Modal H100 invocations; "
        "234 conditions/invocation (78 canonical, 156 repeated); 702 invocation-level conditions; 70,200 scalar CUDA-event samples. "
        "10 rounds × 10 one-kernel samples; three untimed warmups per condition/invocation.", "",
        "UUIDs: " + ", ".join(execution["gpu_uuids"]) + ". Replication: " + execution["replication_mode"] + ".", "",
        table(["Invocation", "GPU / CC", "Driver", "CUDA runtime", "Temperature C start→end", "Power W start→end", "SM / memory clock MHz start→end"],
            [[i, r["environment"]["gpu_name"] + " / " + str(r["environment"]["compute_capability"]),
                r["start"].get("driver_version", "UNAVAILABLE"), r["environment"]["CUDA_runtime_version"],
                r["start"].get("temperature.gpu", "UNAVAILABLE") + "→" + r["end"].get("temperature.gpu", "UNAVAILABLE"),
                r["start"].get("power.draw", "UNAVAILABLE") + "→" + r["end"].get("power.draw", "UNAVAILABLE"),
                r["start"].get("clocks.current.sm", "UNAVAILABLE") + "/" + r["start"].get("clocks.current.memory", "UNAVAILABLE") + "→"
                + r["end"].get("clocks.current.sm", "UNAVAILABLE") + "/" + r["end"].get("clocks.current.memory", "UNAVAILABLE")]
                for i, r in environments.items()]), "",
        "All runtime CUBIN SHAs match frozen Stage B bytes; no JIT, Triton import, ptxas or recompilation. "
        "Exact full descriptor allocation and registered R/B dimensions retained. Invalid attempt records: " + str(execution["invalid_invocations"]) + ". "
        "Initial import-path startup failure occurred before benchmark entry; original source snapshot and full logs remain retained. "
        "A subsequent INVALID_PROTOCOL_RUN return was lost after a local invalid-attempt filename collision; consumed Modal output was unavailable on retrieval, "
        "so its original error/partial samples cannot be recovered. Its call ID, logs and source snapshot remain retained and none of its samples is accepted. "
        "Three later cuTensorMapEncodeTiled failures retain complete returns with zero warmups/visits. "
        "Repair reproduces frozen compiler getTMABlockShapeTiled message-box clipping at 256, without altering full descriptor bounds or kernel bytes. "
        "All repairs rerun the entire invocation; no outcome-based repeats.", "",
        "All differential values are ns/additional CTA, mean ± sample SD across three invocations. "
        "Sign resolution uses frozen t(df=2)=4.302652729911275; unresolved signs are not practical equivalence.", "",
        table(["Case", "Class", "Warps", "G mean±SD", "g0 mean±SD", "D mean±SD", "G/D categories",
            "A prediction / abs error", "B prediction / abs error", "C prediction / abs error"],
            [[n, r["class"], r["num_warps"], metric(r["G"]), metric(r["g0"]), metric(r["D"]),
                r["G"]["category"] + "/" + r["D"]["category"],
                *[num(r["predictions"][m]["G_hat"]) + " / " + num(r["predictions"][m]["abs_error"]) for m in ("A", "B", "C")]]
                for n, r in rows.items()]), "",
        "All individual invocation values, medians/means/SD/IQR/min/max, 234 OLS fits, intercepts, slopes, R², three residuals, "
        "G/g0/g1/D/G-D and every squared prediction error are preserved at full JSON precision in results.json.", "",
        table(["Case", "Squared error A", "Squared error B", "Squared error C"],
            [[n, *[num(r["predictions"][m]["squared_error"]) for m in ("A", "B", "C")]] for n, r in rows.items()]), "",
        "## Locked coefficients and confirmatory decisions", "", "`" + json.dumps(result["locked_model_bindings"], sort_keys=True) + "`", "",
        "Case is the prediction unit: target mean G, predictors mean D and mean g0. Same Model B metrics and same all-eligible n=13 population in H5_02 and H5_03. No refit/calibration/tolerance/significance layer.", "",
        table(["Model", "n", "MAE", "RMSE"], [[m, v["n"], num(v["MAE"]), num(v["RMSE"])] for m, v in result["prediction_errors"].items()]), ""]
    for h, decision in result["hypotheses"].items():
        text += [h + ": **" + decision["status"] + "**. `" + json.dumps(decision, sort_keys=True) + "`", ""]
    if result["hypotheses"][IDS[1]]["status"] == "FALSIFIED":
        text += ["Adding g0 in locked Model B lowered both held-out MAE and RMSE relative to locked Model A, contradicting H5_02's registered non-improvement prediction.", ""]
    if result["hypotheses"][IDS[2]]["status"] == "FALSIFIED":
        text += ["Locked Model C did not improve either registered error metric relative to the same locked Model B, contradicting H5_03's registered prospective extension prediction.", ""]
    text += ["H5_01 remains outcome-independent INCONCLUSIVE_BY_COVERAGE (PRIMARY n=2<5); no n=2 Spearman computed.", "",
        "## Prediction failures (descriptive only)", "",
        table(["Case", "Model", "Class", "M×N / warps", "Absolute error", "Squared error"],
            [[r["case_id"], r["model"], r["class"], f'{r["M"]}×{r["N"]} / {r["num_warps"]}', num(r["abs_error"]), num(r["squared_error"])]
                for r in result["top_five_prediction_errors"]]), "",
        "All 39 case/model prediction errors are ranked and retained. No counterexample, opposed sign or poor fit removed.", "",
        "## Fit diagnostics", ""]
    for h, fits in result["fit_diagnostics"]["lowest_five"].items():
        text += [h + ":", "", table(["Case", "Invocation", "Candidate", "R", "R²", "Three residuals (us)"],
            [[f["case_id"], f["invocation"], f["candidate"], f["R"], num(f["R_squared"]), ", ".join(num(x) for x in f["residuals_us"])] for f in fits]), ""]
    text += ["## Interpretation and validation", "",
        "Deterministic prospective prediction comparisons do not establish a causal mechanism or production policy. "
        "G-D remains a cross-harness descriptive residual. Barrier copies may interact with live ranges/registers/scheduling; no zero differential assumption. "
        "Phase 4 PRIMARY n=3 generalization remains not established; H2a historical SUPPORTED_AT_REDUCTION_BODY_LEVEL; H2b/H2c UNVERIFIED. "
        "No production heuristic, Coalesce.cpp change, or PR #11991 modification.", "",
        "Raw manifest/validation: raw_manifest.json and raw_validation.json. Independent rational-moment/OLS/prediction validation and corruption/synthetic branches: validation.json. "
        "Final old/new validator suite: validator_suite.json. All baseline experiment bytes and first-commit raw bytes remain identical. "
        "The unmodified Phase 5 Stage B validator explicitly rejects a later phase5_timing directory. It is replayed in a detached checkout of trusted baseline 56128d...; "
        "the Stage C validator separately proves every one of the 3514 prior experiment files in the current checkout equals that baseline. "
        "The initial stage-scope failure and complete diagnostic output are retained in validator_suite.json; no frozen checker or gate was changed. "
        "executed_schedule.json retains the original Stage B preview's descriptive metadata; executed=true and the three VALID_PROTOCOL_RUN records identify actual execution. "
        "The master condition dictionary retains the full frozen structural domain; only eligibility-filtered round orders are launched. "
        "Raw and analysis commit SHAs are verified separately in the final Git report. Stop after this Stage C.", ""]
    return "\n".join(text)


def main():
    execution = tc.validate_all()
    commit = raw_commit()
    cases, _, _ = tc.inputs()
    models, binding = locked_models()
    raws = [json.loads((tc.OUT / f"raw_invocation_{i}.json").read_text()) for i in (1, 2, 3)]
    result = derive(raws, cases, models)
    result["locked_model_bindings"] = binding
    result["lineage"] = {"raw_timing_commit": commit, "raw_manifest_SHA256": tc.sha((tc.OUT / "raw_manifest.json").read_bytes()),
        "analysis_source_SHA256": {p.relative_to(ROOT).as_posix(): tc.sha(p.read_bytes()) for p in
            (Path(__file__), Path(__file__).with_name("validate_timing_analysis.py"))}}
    (tc.OUT / "results.json").write_bytes(tc.encode(result))
    (tc.OUT / "summary.md").write_text(report(result, execution))
    print("Derived all 13 held-out cases using unchanged locked coefficients; no refit or omission")


if __name__ == "__main__": main()
