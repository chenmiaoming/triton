"""Independent rational-moment/OLS analysis fidelity; adverse outcomes accepted."""
from collections import Counter
import copy
from fractions import Fraction as F
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3] if Path(__file__).parent.name == "phase4" else Path.cwd()
sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4 import timing_contract as tc
if Path(__file__).parent.name == "phase4":
    from experiments.tma_reduction_layout.phase4 import analyze_timing as analysis
else:
    spec = importlib.util.spec_from_file_location("analysis_fixture", Path(__file__).with_name("analyze_timing.py"))
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)


def same(actual, expected, path="result"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected): raise AssertionError(path + " keys")
        for key in expected: same(actual[key], expected[key], path + "." + str(key))
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected): raise AssertionError(path + " list")
        for i, (a, e) in enumerate(zip(actual, expected)): same(a, e, path + f"[{i}]")
    elif type(expected) is float:
        if type(actual) not in (int, float) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-10):
            raise AssertionError(f"{path}: {actual} != {expected}")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual} != {expected}")


def moments(v):
    values = [F(x) for x in v]
    mean = sum(values) / len(values)
    sd = math.sqrt(float(sum((x - mean) ** 2 for x in values) / (len(values) - 1)))
    return float(mean), sd


def quantile(values, fraction):
    data = sorted(F(x) for x in values)
    index = (len(data) - 1) * fraction
    low = index.numerator // index.denominator
    high = min(low + 1, len(data) - 1)
    return float(data[low] * (1 - (index - low)) + data[high] * (index - low))


def condition_statistics(values):
    mean, sd = moments(values)
    return {"count": len(values), "median_us": quantile(values, F(1, 2)), "mean_us": mean,
        "sample_SD_us": sd, "IQR_us": quantile(values, F(3, 4)) - quantile(values, F(1, 4)),
        "min_us": min(values), "max_us": max(values), "p10_us": quantile(values, F(1, 10)), "p90_us": quantile(values, F(9, 10))}


def ols(points):
    xs, ys = [F(x) for x, _ in points], [F(y) for _, y in points]
    n = len(points)
    # Independent uncentered normal-equation formula, evaluated rationally.
    b = (n * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys)) / (n * sum(x * x for x in xs) - sum(xs) ** 2)
    a = (sum(ys) - b * sum(xs)) / n
    residuals = [y - a - b * x for x, y in zip(xs, ys)]
    total = sum(y * y for y in ys) - sum(ys) ** 2 / n
    return {"points": [{"B_RUN": x, "median_us": y} for x, y in points],
        "intercept_us": float(a), "slope_ns_per_additional_CTA": float(b * 1000),
        "residuals_us": list(map(float, residuals)), "R_squared": float(1 - sum(x * x for x in residuals) / total) if total else None}


def band(values):
    mean, sd = moments(values)
    h = 4.302652729911275 * sd / math.sqrt(3)
    label = "POSITIVE" if mean > h else "NEGATIVE" if mean < -h else "SIGN_UNRESOLVED"
    return {"mean": mean, "sample_SD": sd, "half_width": h, "interval": [mean - h, mean + h],
            "category": label, "invocation_values": values}


def ranks(values):
    return [1. + sum(w < v for w in values) + (sum(w == v for w in values) - 1) / 2 for v in values]


def correlation(x, y):
    if len(x) < 3: return None
    x, y = list(map(F, x)), list(map(F, y))
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx, sy = sum((v - mx) ** 2 for v in x), sum((v - my) ** 2 for v in y)
    return float(sum((a - mx) * (b - my) for a, b in zip(x, y))) / math.sqrt(float(sx * sy)) if sx and sy else None


def signs(rows):
    cats = ("POSITIVE", "NEGATIVE", "SIGN_UNRESOLVED")
    grid = {d: {g: 0 for g in cats} for d in cats}
    pairs = [(r["delta_g1"]["category"], r["G_canonical"]["category"]) for r in rows]
    for d, g in pairs: grid[d][g] += 1
    resolved = [(d, g) for d, g in pairs if d != "SIGN_UNRESOLVED" and g != "SIGN_UNRESOLVED"]
    numerator = sum(d == g for d, g in resolved)
    return {"axis_order": "delta_g1/G_canonical", "matrix": grid,
        "three_category_exact_agreement": {"numerator": sum(d == g for d, g in pairs), "denominator": len(pairs)},
        "resolved_sign_agreement": {"numerator": numerator, "denominator": len(resolved), "rate": numerator / len(resolved) if resolved else None},
        "SIGN_UNRESOLVED_pair_coverage": {"count": len(pairs) - len(resolved), "denominator": len(pairs)}}


def validate(result, raws, cases, protocol):
    if set(result["cases"]) != set(cases): raise AssertionError("All 18 cases and counterexamples must remain")
    independent_runs, fit_records = {}, []
    for raw in raws:
        inv = str(raw["invocation"])
        values = {}
        for v in raw["visits"]: values.setdefault(v["condition_tag"], []).extend(v["samples_us"])
        computed = {tag: condition_statistics(v) for tag, v in values.items()}
        same(result["condition_statistics"][inv], computed, "statistics." + inv)
        independent_runs[inv] = {}
        for cfg in cases:
            slopes = {}
            for h in ("canonical", "repeated"):
                for c in ("default", "4"):
                    for r in ((None,) if h == "canonical" else (0, 1)):
                        points = [(b, computed[f"{cfg}:{h}:{c}:R{r}:B{b}"]["median_us"]) for b in (16384, 32768, 65536)]
                        expected = ols(points)
                        observed = result["fits"][inv][cfg][h][c]
                        if h == "repeated": observed = observed[str(r)]
                        same(observed, expected, f"OLS.{inv}.{cfg}.{h}.{c}.{r}")
                        slopes[h, c, r] = expected["slope_ns_per_additional_CTA"]
                        fit_records.append({"invocation": int(inv), "case_id": cfg, "harness": h, "candidate": c, "R": r,
                            "R_squared": observed["R_squared"], "residuals_us": observed["residuals_us"]})
            G = slopes["canonical", "default", None] - slopes["canonical", "4", None]
            g0 = slopes["repeated", "default", 0] - slopes["repeated", "4", 0]
            g1 = slopes["repeated", "default", 1] - slopes["repeated", "4", 1]
            independent_runs[inv][cfg] = {"G_canonical": G, "g0": g0, "g1": g1, "delta_g1": g1 - g0}
    for cfg, case in cases.items():
        row = result["cases"][cfg]
        for metric in ("G_canonical", "delta_g1"):
            same(row[metric], band([independent_runs[str(i)][cfg][metric] for i in (1, 2, 3)]), cfg + "." + metric)
        same(row["invocation_differentials"], {i: independent_runs[i][cfg] for i in independent_runs}, cfg + ".paired_formulas")
        for field in ("origin", "M", "N", "num_warps", "lanePart_transition"): same(row[field], case[field], cfg + field)
        structure = {"repeated_reduction_equivalence": case["repeated_reduction_equivalence"],
            "lanePart_transition": case["lanePart_transition"], "layouts": {}}
        for h in ("canonical", "repeated"):
            structure["layouts"][h] = {}
            for c in ("default", "4"):
                observed = case[h][c]["observed_layout"]["blocked"]
                structure["layouts"][h][c] = {"lanePart_M": observed["threadsPerWarp"][1],
                    "warpPart_M": observed["warpsPerCTA"][1], "vec": observed["sizePerThread"][2]}
                structure["layouts"][h][c].update({
                    "swizzlingByteWidth": case[h][c]["observed_layout"]["shared"]["swizzlingByteWidth"],
                    "blocks_per_sm_actual_dynamic_smem": case["occupancy"][h][c]["blocks_per_sm_actual_dynamic_smem"]})
        same(row["structural_attributes"], structure, "frozen structural attributes")
        same(row["class"], case["final_class"], "class")
        d, g = row["delta_g1"]["category"], row["G_canonical"]["category"]
        same(row["counterexample_grid"], d + "/" + g, "nine-grid")
        same(row["counterexample_taxonomy"], protocol["counterexamples"][d + "/" + g], "taxonomy")
        relationship = "BOTH_SIGN_UNRESOLVED" if d == g == "SIGN_UNRESOLVED" else "ONE_SIGN_UNRESOLVED" if "SIGN_UNRESOLVED" in (d, g) else "RESOLVED_AGREEMENT" if d == g else "RESOLVED_OPPOSITION"
        same(row["sign_relationship"], relationship, "relationship")
        for h in ("canonical", "repeated"):
            lowest = min((f["R_squared"] for f in fit_records if f["case_id"] == cfg and f["harness"] == h and f["R_squared"] is not None), default=None)
            same(row["lowest_" + h + "_R_squared"], lowest, "case fit diagnostic")
    for name in ("PRIMARY", "SECONDARY"):
        selected = {cfg: r for cfg, r in result["cases"].items() if r["class"] == name}
        names = sorted(selected)
        x, y = [selected[n]["delta_g1"]["mean"] for n in names], [selected[n]["G_canonical"]["mean"] for n in names]
        rx, ry = ranks(x), ranks(y)
        s = result["strata"][name]
        expected_fields = {"n", "case_ids", "association", "average_ranks", "sign_accounting", "counterexample_distribution",
            "rank_disagreement", "top_rank_disagreement", "coverage"}
        if name == "PRIMARY": expected_fields |= {"leave_one_out", "new_only_sensitivity"}
        if set(s) != expected_fields: raise AssertionError("Frozen stratum analysis scope: " + name)
        same(s["n"], len(names), "stratum cardinality")
        same(s["case_ids"], names, "stratum membership")
        same(s["association"], {"method": "Spearman" if name == "PRIMARY" else "Pearson",
             "coefficient": correlation(rx, ry) if name == "PRIMARY" else correlation(x, y)}, "association")
        same(s["average_ranks"], {n: {"delta_g1": a, "G_canonical": b} for n, a, b in zip(names, rx, ry)}, "exact tie ranks")
        same(s["sign_accounting"], signs(list(selected.values())), "sign matrix")
        same(s["counterexample_distribution"], dict(Counter(r["counterexample_taxonomy"] for r in selected.values())), "all counterexample distribution")
        ranked = [{"case_id": n, "origin": selected[n]["origin"], "delta_rank": a, "G_rank": b,
                   "score": abs(a - b) / (len(names) - 1)} for n, a, b in zip(names, rx, ry)]
        ranked.sort(key=lambda r: (-r["score"], r["case_id"]))
        same(s["rank_disagreement"], ranked, "outlier rank/score/order")
        same(s["top_rank_disagreement"], ranked[:min(5, len(names))], "top min(5,n)")
        for item in ranked: same(selected[item["case_id"]]["outlier_score"], item["score"], "per-case outlier score")
        same(s["coverage"], {field: dict(Counter(str(r[field]) for r in selected.values())) for field in ("M", "N", "num_warps", "lanePart_transition")}, "coverage")
        if name == "PRIMARY":
            rhos = {n: correlation(ranks(x[:i] + x[i + 1:]), ranks(y[:i] + y[i + 1:])) for i, n in enumerate(names)}
            defined = sorted(v for v in rhos.values() if v is not None)
            same(s["leave_one_out"], {"rho_minus_i": rhos, "min": min(defined) if defined else None,
                "max": max(defined) if defined else None, "median": quantile(defined, F(1, 2)) if defined else None,
                "defined_count": len(defined), "undefined_count": len(rhos) - len(defined)}, "LOO")
            new = [n for n in names if selected[n]["origin"] == "NEW_GENERALIZATION"]
            same(s["new_only_sensitivity"], {"n": len(new), "case_ids": new,
                "Spearman": correlation(ranks([selected[n]["delta_g1"]["mean"] for n in new]), ranks([selected[n]["G_canonical"]["mean"] for n in new]))}, "new-only minimum n")
    same(result["all_eligible_descriptive_sign_accounting"], signs(list(result["cases"].values())), "pooled descriptive signs only")
    same(result["fit_diagnostics"]["all_fit_count"], len(fit_records), "all 324 fits retained")
    for h in ("canonical", "repeated"):
        low = sorted([f for f in fit_records if f["harness"] == h and f["R_squared"] is not None], key=lambda r: (r["R_squared"], r["case_id"], r["invocation"], r["candidate"], str(r["R"])))[:5]
        same(result["fit_diagnostics"]["lowest_five"][h], low, "lowest R-squared diagnostics")
    same(result["scientific_status"]["phase4_primary_generalization_status"], "INSUFFICIENT_PRIMARY_COHORT_SIZE_FOR_ESTABLISHED_GENERALIZATION", "primary n<5 limitation")
    for h in ("H2b", "H2c"): same(result["scientific_status"][h], "UNVERIFIED", h)
    same(result["scientific_status"]["H2a_historical"], "SUPPORTED_AT_REDUCTION_BODY_LEVEL", "historical H2a unchanged")
    return True


def probes(cases, plan, protocol):
    raw = analysis.synthetic(cases, plan)
    result = analysis.derive(raw, cases, protocol)
    validate(result, raw, cases, protocol)
    if result["strata"]["PRIMARY"]["association"]["coefficient"] >= 0 or result["strata"]["SECONDARY"]["association"]["coefficient"] >= 0:
        raise AssertionError("Synthetic fixture must be adversely associated")
    tests = [{"probe": "negative-correlation synthetic raw dataset", "result": "PASS derivation fidelity"}]
    tied = analysis.synthetic(cases, plan, tied=True)
    tied_result = analysis.derive(tied, cases, protocol)
    validate(tied_result, tied, cases, protocol)
    tied_primary = sorted(n for n in cases if cases[n]["final_class"] == "PRIMARY")
    tie_ranks = tied_result["strata"]["PRIMARY"]["average_ranks"]
    same([tie_ranks[n]["G_canonical"] for n in tied_primary], [1.5, 1.5, 3.], "synthetic exact ties must use average ranks")
    tests.append({"probe": "exact average-rank ties synthetic dataset", "result": "PASS"})
    first = sorted(cases)[0]
    tag = next(iter(result["condition_statistics"]["1"]))
    primary = sorted(n for n in cases if cases[n]["final_class"] == "PRIMARY")[0]
    changes = (
        ("reported median", lambda r: r["condition_statistics"]["1"][tag].__setitem__("median_us", r["condition_statistics"]["1"][tag]["median_us"] + 1)),
        ("slope NaN", lambda r: r["fits"]["1"][first]["canonical"]["default"].__setitem__("slope_ns_per_additional_CTA", float("nan"))),
        ("incorrect G formula", lambda r: r["cases"][first]["invocation_differentials"]["1"].__setitem__("G_canonical", 12345.)),
        ("delta sign flip", lambda r: r["cases"][first]["delta_g1"].__setitem__("mean", -r["cases"][first]["delta_g1"]["mean"])),
        ("sign-resolution category", lambda r: r["cases"][first]["delta_g1"].__setitem__("category", "POSITIVE")),
        ("Spearman exact tie rank", lambda r: r["strata"]["PRIMARY"]["average_ranks"][primary].__setitem__("delta_g1", 42.)),
        ("deleted counterexample", lambda r: r["cases"].pop(first)),
    )
    for name, mutate in changes:
        fixture_result, fixture_raw = (tied_result, tied) if name == "Spearman exact tie rank" else (result, raw)
        bad = copy.deepcopy(fixture_result)
        mutate(bad)
        try:
            validate(bad, fixture_raw, cases, protocol)
        except (AssertionError, KeyError, ValueError):
            tests.append({"probe": name, "result": "PASS corruption rejected"})
        else:
            raise AssertionError("Corruption accepted: " + name)
    return tests


def main():
    tc.validate_all()
    cases, plan, _ = tc.inputs()
    protocol = json.loads((tc.stage_b.PREREG / "protocol.json").read_text())
    raws = [json.loads((tc.OUT / f"raw_invocation_{i}.json").read_text()) for i in (1, 2, 3)]
    result = json.loads((tc.OUT / "results.json").read_text())
    validate(result, raws, cases, protocol)
    commit = analysis.raw_commit()
    same(result["lineage"], {"raw_timing_commit": commit,
        "raw_manifest_sha256": tc.sha((tc.OUT / "raw_manifest.json").read_bytes()),
        "protocol_sha256": tc.sha((tc.stage_b.PREREG / "protocol.json").read_bytes())}, "raw-before-analysis lineage")
    # Temporal anchor comparison is separate from the registered associations.
    historic_path = tc.ROOT / "experiments/tma_reduction_layout/results/phase3/gluon_timing/results.json"
    historic = json.loads(historic_path.read_text())
    anchor = result["anchor_reproducibility"]
    cfg = historic["primary_analysis"]["config"]
    same(anchor["case_id"], cfg, "anchor origin")
    same(anchor["historical_evidence_sha256"], tc.sha(historic_path.read_bytes()), "protected anchor evidence")
    same(anchor["historical_evidence_path"], historic_path.relative_to(tc.ROOT).as_posix(), "anchor evidence path")
    old = historic["primary_analysis"]["cross_run"]["delta_g_1"]
    same(anchor["Phase3_delta_g1"], {"mean": old["mean"], "sample_SD": old["std"]}, "historical anchor differential")
    baseline = historic["canonical_baseline"]
    _, historic_sd = moments([v["default"] - v["4"] for v in baseline["per_run_slopes"].values()])
    same(anchor["Phase3_historical_canonical_baseline"], {"mean": baseline["gap_ns_per_cta"],
        "sample_SD": historic_sd, "origin": "Phase 2 historical canonical baseline reused descriptively in Phase 3"},
        "historical baseline is descriptive only")
    for field in ("G_canonical", "delta_g1"):
        same(anchor["Phase4_" + field], result["cases"][cfg][field], "fresh anchor " + field)
    G = result["cases"][cfg]["G_canonical"]["mean"]
    same(anchor["Phase4_delta_over_fresh_canonical"], result["cases"][cfg]["delta_g1"]["mean"] / G if G != 0 else None,
         "fresh Stage C canonical denominator")
    # Verify that analysis has not rewritten any raw file since the first commit.
    manifest = json.loads((tc.OUT / "raw_manifest.json").read_text())
    for name, digest in manifest["raw_SHA256"].items():
        path = (tc.OUT / name).relative_to(tc.ROOT).as_posix()
        data = subprocess.check_output(["git", "show", commit + ":" + path], cwd=tc.ROOT)
        if tc.sha(data) != digest: raise AssertionError("Raw differs from frozen first commit: " + name)
    tests = probes(cases, plan, protocol)
    if (tc.OUT / "summary.md").read_text() != analysis.report(result, tc.validate_all()):
        raise AssertionError("Summary differs from independently validated complete results")
    (tc.OUT / "validation.json").write_bytes(tc.encode({"derivation_fidelity": "PASS", "negative_outcome_synthetic": "PASS",
        "analysis_corruption_probes": tests, "measured_scalar_samples": 97200,
        "cohort_unchanged": True, "outcome_requirements": []}))
    print("STAGE C ANALYSIS VALIDATOR PASS; independent raw recomputation; adverse synthetic outcome accepted")


if __name__ == "__main__":
    main()
