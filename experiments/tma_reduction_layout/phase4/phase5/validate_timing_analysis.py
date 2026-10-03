"""Independent raw moments, rational OLS and locked case-level hypothesis checks."""
from collections import Counter
import copy
from fractions import Fraction as F
import json
import math
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4.phase5 import timing_contract as tc
from experiments.tma_reduction_layout.phase4.phase5 import analyze_timing as analysis
from experiments.tma_reduction_layout.phase4.validate_timing_analysis import (
    same, condition_statistics, ols, band)

IDS = analysis.IDS


def locked_binding():
    protocol = tc.PREREG / "protocol.json"
    p = json.loads(protocol.read_text())
    s = json.loads((tc.PREREG / "source_bindings.json").read_text())
    source = "experiments/tma_reduction_layout/phase4/results/residual_analysis/results.json"
    digest = tc.sha((ROOT / source).read_bytes())
    tc.require(digest == s["source_SHA256"][source], "Frozen Stage D coefficient source SHA")
    return {"coefficients": p["locked_development_models"], "protocol_SHA256": tc.sha(protocol.read_bytes()),
        "Stage_D_source_result_path": source, "Stage_D_source_result_SHA256": digest,
        "preregistration_source_bindings_SHA256": tc.sha((tc.PREREG / "source_bindings.json").read_bytes()),
        "coefficient_source": "Frozen Phase 5 protocol.json; no refit/calibration"}


def predict(coefficients, d, g0, w):
    a, b, c = coefficients["A"], coefficients["B"], coefficients["C"]
    return {"A": a["intercept"] + a["D"] * d,
        "B": b["intercept"] + b["D"] * d + b["g0"] * g0,
        "C": c["intercept"] + c["D"] * d + c["g0"] * g0 + c["warp8"] * w}


def metric(rows):
    return {m: {"n": len(rows), "unit": "one held-out case; three-invocation means",
        "MAE": sum(r["predictions"][m]["abs_error"] for r in rows) / len(rows),
        "RMSE": math.sqrt(sum(r["predictions"][m]["squared_error"] for r in rows) / len(rows))}
        for m in ("A", "B", "C")}


def oracle(metrics, primary_n, warps, n, valid=True):
    defined = valid and all(type(metrics[m][k]) in (int, float) and math.isfinite(metrics[m][k])
        for m in ("A", "B", "C") for k in ("MAE", "RMSE"))
    a, b, c = [metrics[m] for m in ("A", "B", "C")]
    d2 = "INCONCLUSIVE"
    if defined and n >= 5:
        comparisons = [b[k] >= a[k] for k in ("MAE", "RMSE")]
        d2 = "SUPPORTED" if all(comparisons) else "FALSIFIED" if not any(comparisons) else "INCONCLUSIVE"
    d3 = "INCONCLUSIVE"
    if defined and min(warps.get("4", 0), warps.get("8", 0)) >= 3:
        comparisons = [c[k] < b[k] for k in ("MAE", "RMSE")]
        d3 = "SUPPORTED" if all(comparisons) else "FALSIFIED" if not any(comparisons) else "INCONCLUSIVE"
    return {
        IDS[0]: {"status": "INCONCLUSIVE_BY_COVERAGE", "PRIMARY_n": primary_n, "required": 5,
            "pre_timing_feasibility": "INCONCLUSIVE_BY_COVERAGE_BEFORE_TIMING", "Spearman_computed": False},
        IDS[1]: {"status": d2, "n": n, "population": "ALL_TIMING_ELIGIBLE_HELD_OUT_CASES",
            "Model_A": a, "Model_B": b},
        IDS[2]: {"status": d3, "n": n, "population": "ALL_TIMING_ELIGIBLE_HELD_OUT_CASES",
            "eligible_by_warps": warps, "Model_B": b, "Model_C": c}}


def recompute(raws, cases, coefficients):
    statistics, fits, runs, records = {}, {}, {}, []
    for raw in raws:
        inv = str(raw["invocation"])
        pooled = {}
        for v in raw["visits"]: pooled.setdefault(v["condition_tag"], []).extend(v["samples_us"])
        tc.require(len(pooled) == 234 and all(len(v) == 100 for v in pooled.values()), "Independent raw cardinality")
        statistics[inv] = {tag: condition_statistics(v) for tag, v in pooled.items()}
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
                        f = ols(points)
                        if h == "canonical": fits[inv][cfg][h][c] = f
                        else: fits[inv][cfg][h][c][str(r)] = f
                        slopes[h, c, r] = f["slope_ns_per_additional_CTA"]
                        records.append({"invocation": int(inv), "case_id": cfg, "harness": h,
                            "candidate": c, "R": r, "R_squared": f["R_squared"], "residuals_us": f["residuals_us"]})
            G = slopes["canonical", "default", None] - slopes["canonical", "4", None]
            g0 = slopes["repeated", "default", 0] - slopes["repeated", "4", 0]
            g1 = slopes["repeated", "default", 1] - slopes["repeated", "4", 1]
            runs[inv][cfg] = {"G": G, "g0": g0, "g1": g1, "D": g1 - g0, "G_minus_D": G - (g1 - g0)}
    rows = {}
    for cfg, c in sorted(cases.items()):
        structural = {"repeated_reduction_equivalence": c["repeated_reduction_equivalence"],
            "lanePart_transition": c["lanePart_transition"], "layouts": {}}
        for h in ("canonical", "repeated"):
            structural["layouts"][h] = {}
            for candidate in ("default", "4"):
                layout = c[h][candidate]["observed_layout"]
                structural["layouts"][h][candidate] = {"lanePart_M": layout["blocked"]["threadsPerWarp"][1],
                    "warpPart_M": layout["blocked"]["warpsPerCTA"][1], "vec": layout["blocked"]["sizePerThread"][2],
                    "swizzlingByteWidth": layout["shared"]["swizzlingByteWidth"],
                    "blocks_per_sm_actual_dynamic_smem": c["occupancy"][h][candidate]["blocks_per_sm_actual_dynamic_smem"]}
        row = {"case_id": cfg, "class": c["final_class"], "origin": c["origin"],
            **{k: c[k] for k in ("M", "N", "num_warps", "lanePart_transition")},
            "structural_attributes": structural, "warp8_indicator": 1 if c["num_warps"] == 8 else 0,
            "invocation_differentials": {i: runs[i][cfg] for i in runs}}
        for name in ("G", "g0", "g1", "D", "G_minus_D"):
            row[name] = band([runs[str(i)][cfg][name] for i in (1, 2, 3)])
        g, d = row["G"]["category"], row["D"]["category"]
        row["sign_relationship"] = "BOTH_SIGN_UNRESOLVED" if g == d == "SIGN_UNRESOLVED" else (
            "ONE_SIGN_UNRESOLVED" if "SIGN_UNRESOLVED" in (g, d) else "RESOLVED_AGREEMENT" if g == d else "RESOLVED_OPPOSITION")
        row["predictions"] = {}
        for m, v in predict(coefficients, row["D"]["mean"], row["g0"]["mean"], row["warp8_indicator"]).items():
            e = row["G"]["mean"] - v
            row["predictions"][m] = {"G_hat": v, "error_G_minus_prediction": e, "abs_error": abs(e), "squared_error": e * e}
        for h in ("canonical", "repeated"):
            row["lowest_" + h + "_R_squared"] = min((f["R_squared"] for f in records
                if f["case_id"] == cfg and f["harness"] == h and f["R_squared"] is not None), default=None)
        rows[cfg] = row
    metrics = metric(list(rows.values()))
    primary = sum(r["class"] == "PRIMARY" for r in rows.values())
    warps = dict(sorted(Counter(str(r["num_warps"]) for r in rows.values()).items()))
    ranked = [{"case_id": n, "model": m, "class": r["class"], "M": r["M"], "N": r["N"], "num_warps": r["num_warps"],
        **r["predictions"][m]} for n, r in rows.items() for m in ("A", "B", "C")]
    ranked.sort(key=lambda r: (-r["abs_error"], r["case_id"], r["model"]))
    groups = {"PRIMARY": [r for r in rows.values() if r["class"] == "PRIMARY"],
        "SECONDARY": [r for r in rows.values() if r["class"] == "SECONDARY"]}
    groups.update({f"warps{w}": [r for r in rows.values() if r["num_warps"] == w] for w in (4, 8)})
    groups.update({f"M{m}": [r for r in rows.values() if r["M"] == m] for m in (16, 256, 512)})
    return {"units": "ns/additional CTA; marginal grid-time slope, not single-CTA latency",
        "condition_statistics": statistics, "fits": fits, "cases": rows, "prediction_errors": metrics,
        "hypotheses": oracle(metrics, primary, warps, len(rows)), "all_prediction_errors_ranked": ranked,
        "top_five_prediction_errors": ranked[:5], "descriptive_strata": {g: {"n": len(v),
            "case_ids": sorted(r["case_id"] for r in v), "prediction_errors": metric(v),
            "interpretation": "DESCRIPTIVE_ONLY; confirmatory decisions use all 13 eligible cases"} for g, v in groups.items()},
        "fit_diagnostics": {"all_fit_count": len(records), "lowest_five": {h: sorted(
            [f for f in records if f["harness"] == h and f["R_squared"] is not None],
            key=lambda r: (r["R_squared"], r["case_id"], r["invocation"], r["candidate"], str(r["R"])))[:5]
            for h in ("canonical", "repeated")}, "handling": "All fits and cases retained; no R2 threshold or outcome-based retry"},
        "scientific_status": {"Phase5_role": "HELD_OUT_CONFIRMATORY_EXTENSION",
            "Phase4_primary_generalization": "NOT_ESTABLISHED_PRIMARY_N3", "H2a_historical": "SUPPORTED_AT_REDUCTION_BODY_LEVEL",
            "H2b": "UNVERIFIED", "H2c": "UNVERIFIED", "model_refit": False, "production_heuristic": False,
            "PR_11991_modified": False, "residual_interpretation": "G-D is a cross-harness descriptive residual, not a causal remainder",
            "sign_unresolved_interpretation": "SIGN_UNRESOLVED is not practical equivalence"}}


def validate(result, expected, cases, binding):
    tc.require(set(result["cases"]) == set(cases) and len(result["cases"]) == 13, "Exact all-eligible n13 population")
    tc.require(result["locked_model_bindings"] == binding, "Exact frozen coefficient/source SHA binding; no refit")
    stripped = {k: v for k, v in result.items() if k not in ("lineage", "locked_model_bindings")}
    same(stripped, expected)
    for cfg, case in cases.items():
        tc.require(result["cases"][cfg]["warp8_indicator"] == int(case["num_warps"] == 8), "Frozen warp indicator")
    metrics = result["prediction_errors"]
    decisions = oracle(metrics, 2, {"4": 5, "8": 8}, 13)
    tc.require(result["hypotheses"] == decisions, "Exact unrounded deterministic decisions")
    tc.require(result["hypotheses"][IDS[1]]["Model_B"] == result["hypotheses"][IDS[2]]["Model_B"] == metrics["B"], "One shared Model B metric pair")


def rejected(fn):
    try: fn()
    except (AssertionError, ValueError, KeyError): return True
    return False


def synthetic_branches(cases, models):
    rows = []
    for i, (cfg, case) in enumerate(sorted(cases.items())):
        pred = predict(models, (i - 6) / 3, (i + 1) / 20, int(case["num_warps"] == 8))
        rows.append({"case_id": cfg, "prediction": pred})
    report = []
    for hypothesis, left, right, support_target in ((IDS[1], "A", "B", "A"), (IDS[2], "B", "C", "C")):
        for wanted in ("SUPPORTED", "FALSIFIED", "INCONCLUSIVE"):
            def data_metrics(targets):
                data = [{"predictions": {m: {"abs_error": abs(g - r["prediction"][m]),
                    "squared_error": (g - r["prediction"][m]) ** 2} for m in ("A", "B", "C")}}
                    for g, r in zip(targets, rows)]
                return metric(data)
            target_model = support_target if wanted == "SUPPORTED" else (right if support_target == left else left)
            targets = [r["prediction"][target_model] for r in rows]
            if wanted == "INCONCLUSIVE":
                found = False
                # Fixed offline fixture search on synthetic values; no actual timing input or coefficient refit.
                for j in range(len(rows)):
                    for direction in (-1, 1):
                        for scale in (1, 2, 4, 8, 16, 32, 64, 128):
                            trial = [r["prediction"][left] for r in rows]
                            trial[j] = rows[j]["prediction"][right] + direction * scale
                            metrics = data_metrics(trial)
                            if oracle(metrics, 2, {"4": 5, "8": 8}, 13)[hypothesis]["status"] == wanted:
                                targets, found = trial, True; break
                        if found: break
                    if found: break
                tc.require(found, "Synthetic mixed-metric fixture exists")
            metrics = data_metrics(targets)
            expected = oracle(metrics, 2, {"4": 5, "8": 8}, 13)
            observed = analysis.decisions(metrics, 2, {"4": 5, "8": 8}, 13)
            tc.require(expected == observed and observed[hypothesis]["status"] == wanted, "Synthetic decision branch: " + wanted)
            report.append({"probe": hypothesis + " synthetic " + wanted, "metrics": metrics, "result": "PASS"})
    _, plan, _ = tc.inputs()
    primary_names = sorted(n for n, c in cases.items() if c["final_class"] == "PRIMARY")
    synthetic_raw = []
    for inv in plan["invocations"]:
        visits = []
        for round_ in inv["rounds"]:
            for tag in round_["order"]:
                p = tc.condition(tag)
                primary = cases[p["config_id"]]["final_class"] == "PRIMARY"
                differential = 1000000.0 * (1 + primary_names.index(p["config_id"])) if primary else 1.0
                slope = 100.0
                if p["candidate"] == "default" and (p["harness"] == "canonical" or p["R"] == 1):
                    slope += differential
                visits.append({"condition_tag": tag, "samples_us": [20.0 + slope * p["B_RUN"] / 1000.0] * 10})
        synthetic_raw.append({"invocation": inv["invocation"], "visits": visits})
    spectacular_result = analysis.derive(synthetic_raw, cases, models)
    tc.require(spectacular_result["hypotheses"][IDS[0]]["status"] == "INCONCLUSIVE_BY_COVERAGE"
        and not spectacular_result["hypotheses"][IDS[0]]["Spearman_computed"], "Spectacular n2 cannot confirm H5_01")
    tc.require(len({spectacular_result["cases"][n]["G"]["mean"] for n in primary_names}) == 2
        and all(spectacular_result["cases"][n][k]["category"] == "POSITIVE"
            for n in primary_names for k in ("G", "D")), "Distinct spectacular positive PRIMARY outcomes")
    report.append({"probe": "spectacular PRIMARY n2 remains coverage-inconclusive", "result": "PASS"})
    spectacular = {m: {"MAE": 0.0, "RMSE": 0.0} for m in ("A", "B", "C")}
    equal = analysis.decisions(spectacular, 2, {"4": 5, "8": 8}, 13)
    tc.require(equal[IDS[1]]["status"] == "SUPPORTED" and equal[IDS[2]]["status"] == "FALSIFIED", "Exact equality branch; no tolerance")
    report.append({"probe": "equal metrics use exact frozen comparison", "result": "PASS"})
    for valid, n, warps, label in ((False, 13, {"4": 5, "8": 8}, "invalid protocol"),
        (True, 4, {"4": 2, "8": 2}, "insufficient case/warp coverage")):
        decision = analysis.decisions(spectacular, 2, warps, n, valid)
        tc.require(decision[IDS[1]]["status"] == decision[IDS[2]]["status"] == "INCONCLUSIVE", label)
        report.append({"probe": label + " stays inconclusive", "result": "PASS"})
    return report


def main():
    execution = tc.validate_all()
    commit = analysis.raw_commit()
    cases, _, _ = tc.inputs()
    binding = locked_binding()
    raws = [json.loads((tc.OUT / f"raw_invocation_{i}.json").read_text()) for i in (1, 2, 3)]
    result = json.loads((tc.OUT / "results.json").read_text())
    expected = recompute(raws, cases, binding["coefficients"])
    validate(result, expected, cases, binding)
    tc.require(result["lineage"] == {"raw_timing_commit": commit,
        "raw_manifest_SHA256": tc.sha((tc.OUT / "raw_manifest.json").read_bytes()),
        "analysis_source_SHA256": {p.relative_to(ROOT).as_posix(): tc.sha(p.read_bytes())
            for p in (Path(analysis.__file__), Path(__file__))}}, "Raw commit and offline implementation lineage")
    tc.require((tc.OUT / "summary.md").read_text() == analysis.report(result, execution), "Derived complete summary")
    first = sorted(cases)[0]
    probes = []
    mutations = (
        ("one slope", lambda r: r["fits"]["1"][first]["canonical"]["default"].update(slope_ns_per_additional_CTA=9999.0)),
        ("D", lambda r: r["cases"][first]["D"].update(mean=9999.0)),
        ("frozen coefficient", lambda r: r["locked_model_bindings"]["coefficients"]["B"].update(g0=0.0)),
        ("Phase5 refit coefficient", lambda r: r["locked_model_bindings"]["coefficients"]["C"].update(warp8=1234.0)),
        ("drop eligible case", lambda r: r["cases"].pop(first)),
        ("add excluded case", lambda r: r["cases"].update(M512_N128_w4=copy.deepcopy(r["cases"][first]))),
        ("warp indicator", lambda r: r["cases"][first].update(warp8_indicator=1-r["cases"][first]["warp8_indicator"])),
        ("one prediction", lambda r: r["cases"][first]["predictions"]["A"].update(G_hat=9999.0)),
        ("MAE/RMSE", lambda r: r["prediction_errors"]["B"].update(MAE=9999.0, RMSE=9999.0)),
        ("shared Model B population", lambda r: r["hypotheses"][IDS[2]]["Model_B"].update(n=39)),
    )
    for label, mutate in mutations:
        changed = copy.deepcopy(result); mutate(changed)
        tc.require(rejected(lambda: validate(changed, expected, cases, binding)), "Reject corruption: " + label)
        probes.append({"probe": label, "result": "PASS"})
    changed = copy.deepcopy(raws)
    changed[0]["visits"][0]["samples_us"][0] += 10000.0
    tc.require(rejected(lambda: validate(result, recompute(changed, cases, binding["coefficients"]), cases, binding)), "Raw sample corruption changes independent analysis")
    probes.append({"probe": "raw scalar sample mutation", "result": "PASS"})
    probes += synthetic_branches(cases, binding["coefficients"])
    validation = {"status": "PASS", "independent_recomputation": "raw rational moments, quantiles, normal-equation OLS, paired differences, bands, case-mean predictions/errors and deterministic rules",
        "counts": tc.EXPECTED_COUNTS, "all_case_count": len(cases), "all_OLS_fit_count": 234,
        "locked_coefficients_bound": True, "raw_commit_bytes_unchanged": True,
        "protected_prior_files": len(tc.protected_inventory()), "corruption_and_synthetic_probes": probes}
    (tc.OUT / "validation.json").write_bytes(tc.encode(validation))
    print(f'PHASE 5 ANALYSIS VALIDATOR PASS; {len(probes)}/{len(probes)} corruption/synthetic probes; all 13 cases and locked models')


if __name__ == "__main__": main()
