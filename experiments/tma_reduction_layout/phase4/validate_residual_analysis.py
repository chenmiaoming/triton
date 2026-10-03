"""Independent rational OLS/LOOCV and archive extraction; no outcome gate."""
from collections import Counter
import copy
from fractions import Fraction as F
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4 import residual_contract as rc
from experiments.tma_reduction_layout.phase4 import analyze_residual_mechanisms as analysis
from experiments.tma_reduction_layout.phase4 import validate_timing_analysis as independent_c
from experiments.tma_reduction_layout.phase4 import timing_contract as tc
from experiments.tma_reduction_layout.phase4 import artifact_gate as gate
from experiments.tma_reduction_layout.gluon import artifact_checks as ac
from experiments.tma_reduction_layout.analyze_ir import analyze_ttgir

same = independent_c.same


def independent_metrics(raws):
    rows = {}
    for raw in raws:
        samples = {}
        for v in raw["visits"]:
            samples.setdefault(v["condition_tag"], []).extend(v["samples_us"])
        for cfg in sorted({s.split(":")[0] for s in samples}):
            slopes = {}
            for h in ("canonical", "repeated"):
                for c in ("default", "4"):
                    for r in ((None,) if h == "canonical" else (0, 1)):
                        points = [(b, independent_c.quantile(samples[f"{cfg}:{h}:{c}:R{r}:B{b}"], F(1, 2))) for b in (16384, 32768, 65536)]
                        slopes[h, c, r] = independent_c.ols(points)["slope_ns_per_additional_CTA"]
            G = slopes["canonical", "default", None] - slopes["canonical", "4", None]
            g0 = slopes["repeated", "default", 0] - slopes["repeated", "4", 0]
            g1 = slopes["repeated", "default", 1] - slopes["repeated", "4", 1]
            D = g1 - g0
            values = rows.setdefault(cfg, {k: [] for k in ("G", "g0", "g1", "D", "residual")})
            for k, v in {"G": G, "g0": g0, "g1": g1, "D": D, "residual": G - D}.items():
                values[k].append(v)
    return {n: {k: {"invocation_values": v, "mean": independent_c.moments(v)[0], "sample_SD": independent_c.moments(v)[1]} for k, v in row.items()} for n, row in rows.items()}


def independent_features(cases):
    result = {}
    for cfg, case in sorted(cases.items()):
        row = {k: case[k] for k in ("M", "N", "num_warps")}
        row.update({"canonical_reproduction_equivalence": case["single_reduction_equivalence"],
            "repeated_equivalence": case["repeated_reduction_equivalence"], "preload_encoding": case["repeated_localload_encoding"], "harnesses": {}})
        for h in ("canonical", "repeated"):
            pair = {}
            for c in ("default", "4"):
                root = rc.GATE / h / cfg / c
                ptx, ttgir = (root / "kernel.ptx").read_text(), (root / "kernel.ttgir").read_text()
                info = analyze_ttgir(ttgir)
                blocked = {k: info["blocked_encodings"][info["local_load_dest_layout"]][k] for k in ("sizePerThread", "threadsPerWarp", "warpsPerCTA", "order")}
                layout = gate.ttgir_contract(ttgir, [1, case["M"], case["N"]], case["num_warps"])
                if h == "canonical":
                    meta = rc.read(root / "metadata.json")
                    fp = ac.fingerprint(gate.body_instructions(ptx, meta["reduction_stage"]))
                else:
                    loop = next(e for e in ac.ptx_backedges(ptx) if e["kind"] == "compiler_loop")
                    fp = ac.fingerprint([i for i in ac.ptx_instructions(ptx) if loop["start"] <= i["line"] <= loop["end"]])
                families = {k: 0 for k in rc.FAMILIES}
                for instruction in fp:
                    op = instruction.split()[0]
                    for family in families:
                        if (family == "bar.sync" and op == "bar.sync") or (family != "bar.sync" and op.startswith(family[:-1])):
                            families[family] += 1
                loads = ac.initial_load_signature(ptx)
                opcodes = [i["opcode"] for i in loads]
                widths, payloads = [], []
                for op in opcodes:
                    match = re.fullmatch(r"ld\.shared(?:\.v([124]))?\.[bus](16|32|64)", op)
                    widths.append(int(match[1] or 1) if match else None)
                    payloads.append(int(match[2]) * int(match[1] or 1) // 8 if match else None)
                resources = ac.parse_resource((root / "kernel.resource.txt").read_text())
                occ, meta = rc.read(root / "occupancy.json"), rc.read(root / "metadata.json")
                resources = {**resources, "cuobjdump_reported_shared_bytes": resources["static_smem_bytes"],
                    "static_smem_bytes": occ["static_smem_bytes"], "dynamic_smem_bytes": meta["dynamic_smem_bytes"],
                    "shared_field_note": "cuobjdump SHARED and driver static SMEM retained separately; no equivalence assumed"}
                pair[c] = {"lanePart_M": blocked["threadsPerWarp"][1], "warpPart_M": blocked["warpsPerCTA"][1],
                    "vec": blocked["sizePerThread"][2], "blocked_layout": blocked, "shared_layout": layout["shared"],
                    "resources": resources, "blocks_per_SM": occ["blocks_per_sm_actual_dynamic_smem"],
                    "LocalLoad": {"opcode_sequence": opcodes, "instruction_count": len(opcodes), "vector_widths": widths,
                        "total_payload_bytes_per_thread": sum(payloads) if None not in payloads else None,
                        "payload_status": "MECHANICALLY_DERIVED_PER_THREAD" if None not in payloads else "UNDEFINED_FOR_UNSUPPORTED_OPCODE",
                        "lines": [i["line"] for i in loads]},
                    "fingerprint_sequence": fp, "opcode_multiset": dict(Counter(fp)), "family_counts": families,
                    "archive_cubin_sha256": rc.sha((root / "kernel.cubin").read_bytes())}
            d, c = pair["default"], pair["4"]
            diff = {"num_regs": d["resources"]["num_regs"] - c["resources"]["num_regs"], "blocks_per_SM": d["blocks_per_SM"] - c["blocks_per_SM"],
                "LocalLoad_instruction_count": d["LocalLoad"]["instruction_count"] - c["LocalLoad"]["instruction_count"],
                "LocalLoad_payload_bytes_per_thread": d["LocalLoad"]["total_payload_bytes_per_thread"] - c["LocalLoad"]["total_payload_bytes_per_thread"]
                    if all(v["LocalLoad"]["total_payload_bytes_per_thread"] is not None for v in (d, c)) else None}
            for field in ("opcode_multiset", "family_counts"):
                diff[field] = {k: d[field].get(k, 0) - c[field].get(k, 0) for k in set(d[field]) | set(c[field])}
            pair["delta_default_minus_cand4"] = diff
            pair["candidate_pair_sequence_equivalence"] = "EXACT_SEQUENCE_EQUIVALENT" if d["fingerprint_sequence"] == c["fingerprint_sequence"] else "PIPELINED_OPCODE_EQUIVALENT" if Counter(d["fingerprint_sequence"]) == Counter(c["fingerprint_sequence"]) else "REDUCTION_FINGERPRINT_MISMATCH"
            row["harnesses"][h] = pair
        d, c = row["harnesses"]["canonical"]["default"], row["harnesses"]["canonical"]["4"]
        row["lane_transition"] = str(d["lanePart_M"]) + "->" + str(c["lanePart_M"])
        row["lanePart_reduction_ratio"] = d["lanePart_M"] / c["lanePart_M"]
        row["equivalence_group"] = "EXACT" if all(v == "EXACT_SEQUENCE_EQUIVALENT" for k in ("canonical_reproduction_equivalence", "repeated_equivalence") for v in row[k].values()) else "PIPELINED"
        row["preload_group"] = "EXACT" if set(row["preload_encoding"].values()) == {"EXACT"} else "DIFFERENT_ENCODING"
        result[cfg] = row
    return result


def exact_fit(X, y):
    """Independent rational normal equations/Gauss-Jordan, no QR reuse."""
    n, p = len(X), len(X[0])
    if n < p:
        return None
    X, y = [[F(v) for v in row] for row in X], list(map(F, y))
    a = [[sum(row[i] * row[j] for row in X) for j in range(p)] + [sum(row[i] * value for row, value in zip(X, y))] for i in range(p)]
    for col in range(p):
        pivot = next((i for i in range(col, p) if a[i][col]), None)
        if pivot is None:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        divisor = a[col][col]
        a[col] = [v / divisor for v in a[col]]
        for i in range(p):
            if i != col:
                scale = a[i][col]
                a[i] = [v - scale * w for v, w in zip(a[i], a[col])]
    return [float(a[i][-1]) for i in range(p)]


def independent_model(rows, terms):
    rows = sorted(rows, key=lambda r: r["case_id"])
    X = [[{"intercept": 1., "D": r["metrics"]["D"]["mean"], "g0": r["metrics"]["g0"]["mean"],
           "warp8": float(r["features"]["num_warps"] == 8)}[t] for t in terms] for r in rows]
    y = [r["metrics"]["G"]["mean"] for r in rows]
    b = exact_fit(X, y)
    predicted = [sum(v * coef for v, coef in zip(row, b)) for row in X] if b is not None else None
    folds = []
    for i, row in enumerate(rows):
        beta = exact_fit([x for j, x in enumerate(X) if j != i], [v for j, v in enumerate(y) if j != i])
        p = sum(x * a for x, a in zip(X[i], beta)) if beta is not None else None
        folds.append({"held_out_case": row["case_id"], "training_case_ids": [r["case_id"] for j, r in enumerate(rows) if j != i],
            "coefficients": dict(zip(terms, beta)) if beta is not None else None, "observed_G": y[i], "predicted_G": p,
            "error_predicted_minus_observed": p - y[i] if p is not None else None,
            "status": "DEFINED" if p is not None else "UNDEFINED_SINGULAR_OR_INSUFFICIENT_OBSERVATIONS"})
    errors = [f["error_predicted_minus_observed"] for f in folds]
    complete = None not in errors
    yy = list(map(F, y)); mean = sum(yy) / len(yy); total = sum((v - mean) ** 2 for v in yy)
    return {"n": len(rows), "terms": terms, "coefficients": dict(zip(terms, b)) if b is not None else None,
        "fit_status": "DEFINED" if b is not None else "UNDEFINED_SINGULAR_OR_INSUFFICIENT_OBSERVATIONS",
        "in_sample_predictions": dict(zip([r["case_id"] for r in rows], predicted)) if predicted is not None else None,
        "R_squared": float(1 - sum((F(a) - F(p)) ** 2 for a, p in zip(y, predicted)) / total) if predicted is not None and total else None,
        "LOOCV": {"folds": folds, "MAE": float(sum(F(abs(e)) for e in errors) / len(errors)) if complete else None,
            "RMSE": math.sqrt(float(sum(F(e) ** 2 for e in errors) / len(errors))) if complete else None,
            "defined_folds": sum(e is not None for e in errors), "undefined_folds": sum(e is None for e in errors)}}


def independent_agreement(rows):
    rr = [(r["G_category"], r["D_category"]) for r in rows if "SIGN_UNRESOLVED" not in (r["G_category"], r["D_category"])]
    match = sum(g == d for g, d in rr)
    return {"numerator": match, "denominator": len(rr), "rate": match / len(rr) if rr else None, "unresolved_pairs": len(rows) - len(rr)}


def validate(result, raws, cases, fmap):
    metrics = independent_metrics(raws)
    same(set(result["cases"]), set(cases), "all eighteen cases")
    for cfg, case in cases.items():
        row = result["cases"][cfg]
        same(row["metrics"], metrics[cfg], cfg + ".five_paired_metrics")
        same(row["features"], fmap[cfg], cfg + ".actual_archive_features")
        for field, key in (("class", "final_class"), ("origin", "origin")):
            same(row[field], case[key], "frozen " + field)
        for key, field in (("G", "G_category"), ("D", "D_category")):
            same(row[field], independent_c.band(metrics[cfg][key]["invocation_values"])["category"], "frozen sign rule")
        policy = rc.read(tc.stage_b.PREREG / "protocol.json")
        same(row["taxonomy"], policy["counterexamples"][row["D_category"] + "/" + row["G_category"]], "complete taxonomy")
    rows = list(result["cases"].values())
    strata = {"ALL_ELIGIBLE": rows, **{s: [r for r in rows if r["class"] == s] for s in ("PRIMARY", "SECONDARY")}}
    same(set(result["models"]), set(strata), "fixed model scopes")
    for s, rr in strata.items():
        same(set(result["models"][s]), {"A", "B", "C"}, "exact three model families")
        for name, terms in rc.MODEL_TERMS.items():
            same(result["models"][s][name], independent_model(rr, terms), "independent OLS/each LOOCV fold " + s + "/" + name)
        G = [r["metrics"]["G"]["mean"] for r in rr]
        scores = {"D": [r["metrics"]["D"]["mean"] for r in rr], "g0": [r["metrics"]["g0"]["mean"] for r in rr],
                  "D_plus_g0": [r["metrics"]["D"]["mean"] + r["metrics"]["g0"]["mean"] for r in rr]}
        expected = {k: {"Pearson": independent_c.correlation(G, v),
            "Spearman": independent_c.correlation(independent_c.ranks(G), independent_c.ranks(v))} for k, v in scores.items()}
        same(result["correlations"][s], expected, "continuous/exact-tie correlations")
    fields = ("num_warps", "M", "N", "lane_transition", "equivalence_group", "preload_group")
    same(set(result["groups"]), set(fields), "fixed group family")
    for field in fields:
        expected = {}
        for label in sorted({str(r["features"][field]) for r in rows}):
            rr = [r for r in rows if str(r["features"][field]) == label]
            expected[label] = {"n": len(rr), "small_n": len(rr) < 3,
                "mean_G": independent_c.moments([r["metrics"]["G"]["mean"] for r in rr])[0] if len(rr) > 1 else rr[0]["metrics"]["G"]["mean"],
                "mean_D": float(sum(F(r["metrics"]["D"]["mean"]) for r in rr) / len(rr)),
                "mean_descriptive_residual": float(sum(F(r["metrics"]["residual"]["mean"]) for r in rr) / len(rr)),
                "resolved_sign_agreement": independent_agreement(rr), "case_ids": sorted(r["case_id"] for r in rr)}
        same(result["groups"][field], expected, "all groups " + field)
    same(result["taxonomy_distribution"], dict(Counter(r["taxonomy"] for r in rows)), "all counterexamples/neutral cases")
    same(result["analysis_status"], "OUTCOME_INFORMED_EXPLORATORY_ANALYSIS", "statistical role")
    return True


def synthetic(cases, plan, fmap):
    """Fixed adverse fixture: g0 orthogonal; negative D/G; extra warp regressor overfits."""
    names = sorted(cases)
    D = [(i - 8.5) / 4 for i in range(18)]
    G = [-2 * d + .2 * math.sin(i * 1.7) for i, d in enumerate(D)]
    vec = [math.cos(i * .91) for i in range(18)]
    basis = []
    for column in ([1.] * 18, D, G):
        column = list(column)
        for q in basis:
            coefficient = sum(a * b for a, b in zip(column, q))
            column = [a - coefficient * b for a, b in zip(column, q)]
        norm = math.sqrt(sum(v * v for v in column))
        basis.append([v / norm for v in column])
    for q in basis:
        coefficient = sum(a * b for a, b in zip(vec, q))
        vec = [a - coefficient * b for a, b in zip(vec, q)]
    targets = {n: {"G": G[i], "D": D[i], "g0": vec[i] * .05} for i, n in enumerate(names)}
    raws = []
    for planned in plan["invocations"]:
        visits = []
        for rr in planned["rounds"]:
            for tag in rr["order"]:
                p = tc.condition(tag)
                t = targets[p["config_id"]]
                slope = 100.
                if p["candidate"] == "default":
                    slope += t["G"] if p["harness"] == "canonical" else t["g0"] + (t["D"] if p["R"] == 1 else 0)
                v = 20 + slope * p["B_RUN"] / 1000
                visits.append({"condition_tag": tag, "samples_us": [v] * 10})
        raws.append({"invocation": planned["invocation"], "visits": visits})
    policy = rc.read(tc.stage_b.PREREG / "protocol.json")
    rows = analysis.rows_from_raw(raws, cases, fmap, policy["counterexamples"])
    result = rc.derive(rows)
    validate(result, raws, cases, fmap)
    A, B, C = (result["models"]["ALL_ELIGIBLE"][k] for k in ("A", "B", "C"))
    rc.require(result["correlations"]["ALL_ELIGIBLE"]["D"]["Pearson"] < 0, "Adverse D/G fixture")
    rc.require(abs(B["coefficients"]["g0"]) < 1e-9 and abs(B["R_squared"] - A["R_squared"]) < 1e-10, "g0 is useless in adverse fixture")
    rc.require(C["LOOCV"]["RMSE"] > A["LOOCV"]["RMSE"], "Model C worse than A on adverse LOOCV fixture")
    return {"status": "PASS_DERIVATION_FIDELITY", "Pearson_G_D": result["correlations"]["ALL_ELIGIBLE"]["D"]["Pearson"],
        "g0_coefficient_model_B": B["coefficients"]["g0"], "LOOCV_RMSE_A": A["LOOCV"]["RMSE"], "LOOCV_RMSE_C": C["LOOCV"]["RMSE"],
        "interpretation": "Nested OLS in-sample R2 cannot worsen, so 'C worse' is tested in LOOCV; no measured outcome acceptance threshold", "saved_synthetic_samples": 0}


def validate_annotations(result):
    """Annotations bind already independently validated numbers, not favorable outcomes."""
    same(result["units"], "ns/additional CTA", "units")
    same(result["residual_interpretation"], "G-D is a cross-harness descriptive residual, NOT an additive causal remainder.", "descriptive residual only")
    same(result["g0_interpretation"], "fixed-harness differential; may include pre-loop LocalLoad, fixed work and compiler/register/scheduling interactions", "fixed harness interpretation")
    same(result["score_interpretation"], "D+g0 is a descriptive constructed score; no causal decomposition", "constructed score only")
    hs = result["hypotheses"]
    rc.require(0 < len(hs) <= 3 and len({h["hypothesis_id"] for h in hs}) == len(hs), "At most three unique hypotheses")
    for h in hs:
        for field in ("hypothesis_id", "observed_motivation", "mechanistic_rationale", "variables",
                      "supporting_future_observation", "falsifying_future_observation", "scope", "inconclusive", "lock_policy"):
            rc.require(bool(h[field]), "Complete testable hypothesis field: " + field)
        same(h["status"], "OUTCOME_INFORMED_HYPOTHESIS", "hypothesis not retroactively confirmed")
        same(h["development_data"], "Stage C; hypothesis generation only", "development role")
        same(h["confirmation_data"], "Unmeasured Phase 5 held-out domain; no use of Stage C for confirmation", "held-out role")
    same(hs, analysis.hypotheses(result), "motivation numbers and future hypothesis locks")
    audit = result["focused_audit"]
    same(audit["resolved_opposition"]["target"], "M128_N16_w4", "required opposed case")
    for section, pairs in (("resolved_opposition", [("M128_N16_w4", "M64_N16_w4"), ("M128_N16_w4", "M128_N16_w8")]),
                           ("canonical_positive_D_unresolved", [("M128_N32_w8", "M64_N32_w4"), ("M32_N64_w4", "M32_N64_w8")])):
        comparisons = audit[section]["comparisons"]
        same([(v["target"], v["reference"]) for v in comparisons], pairs, "required contextual comparisons")
        for value in comparisons:
            a, b = [result["cases"][value[k]] for k in ("target", "reference")]
            same(value["metric_mean_target_minus_reference"], {k: a["metrics"][k]["mean"] - b["metrics"][k]["mean"]
                for k in ("G", "g0", "g1", "D", "residual")}, "focused metrics")
            same(value["warps_target_reference"], [r["features"]["num_warps"] for r in (a, b)], "focused warp regime")
            same(value["canonical_preload_payload_default_target_reference"], [r["features"]["harnesses"]["canonical"]["default"]["LocalLoad"]["total_payload_bytes_per_thread"]
                for r in (a, b)], "focused LocalLoad payload")
            same(value["register_deltas_target_reference"], {h: [r["features"]["harnesses"][h]["delta_default_minus_cand4"]["num_regs"] for r in (a, b)]
                for h in ("canonical", "repeated")}, "focused register deltas")
            same(value["canonical_family_delta_target_reference"], [r["features"]["harnesses"]["canonical"]["delta_default_minus_cand4"]["family_counts"]
                for r in (a, b)], "focused fingerprint deltas")
    same(audit["canonical_positive_D_unresolved"]["case_ids"], ["M128_N32_w8", "M32_N64_w4", "M64_N32_w4"], "all unresolved D counterexamples")
    same(audit["neutral_cases"]["case_ids"], sorted(["M128_N64_w8", "M128_N128_w4", "M128_N128_w8", "M128_N32_w4",
        "M32_N128_w4", "M64_N128_w8", "M64_N64_w4"]), "seven neutral cases retained")
    same(audit, analysis.focused_audit(result), "complete focused annotations")


def main():
    protected = rc.frozen_inventory()
    tc.validate_all()
    cases, plan, _ = tc.inputs()
    raws = [rc.read(rc.TIMING / f"raw_invocation_{i}.json") for i in (1, 2, 3)]
    result = rc.read(rc.OUT / "results.json")
    fmap = independent_features(cases)
    validate(result, raws, cases, fmap)
    validate_annotations(result)
    rc.require((rc.OUT / "summary.md").read_text() == analysis.report(result), "Report must reproduce all validated 18-case/model/group/focus/hypothesis/24-transition content deterministically")
    same(result["frozen_stage_c_scientific_status"], rc.read(rc.TIMING / "results.json")["scientific_status"], "Stage C scientific status unchanged")
    same(result["lineage"], {"starting_HEAD": rc.BASELINE, "Stage_C_results_SHA256": rc.sha((rc.TIMING / "results.json").read_bytes()),
        "Stage_C_raw_manifest_SHA256": rc.sha((rc.TIMING / "raw_manifest.json").read_bytes()), "Stage_B_cohort_SHA256": rc.sha((rc.GATE / "cohort_after_gate.json").read_bytes()),
        "protected_prior_inventory_SHA256": rc.sha(rc.encode(protected)),
        "Stage_D_source_SHA256": {p.relative_to(ROOT).as_posix(): rc.sha(p.read_bytes()) for p in
            (Path(analysis.__file__), Path(__file__), Path(rc.__file__))}}, "complete raw/result/archive/derivation source lineage")
    adverse = synthetic(cases, plan, fmap)
    tests = []
    first = sorted(cases)[0]
    changes = [("residual formula", lambda r: r["cases"][first]["metrics"]["residual"].__setitem__("mean", 123.)),
        ("drop neutral case", lambda r: r["cases"].pop("M128_N64_w8")),
        ("LocalLoad count", lambda r: r["cases"][first]["features"]["harnesses"]["canonical"]["default"]["LocalLoad"].__setitem__("instruction_count", 999)),
        ("shuffle multiset delta", lambda r: r["cases"][first]["features"]["harnesses"]["canonical"]["delta_default_minus_cand4"]["family_counts"].__setitem__("shfl.*", 999)),
        ("OLS coefficient", lambda r: r["models"]["ALL_ELIGIBLE"]["A"]["coefficients"].__setitem__("D", 999.)),
        ("LOOCV prediction", lambda r: r["models"]["ALL_ELIGIBLE"]["B"]["LOOCV"]["folds"][0].__setitem__("predicted_G", 999.)),
        ("group count", lambda r: r["groups"]["num_warps"]["4"].__setitem__("n", 999)),
        ("correlation sign", lambda r: r["correlations"]["ALL_ELIGIBLE"]["D"].__setitem__("Pearson", -r["correlations"]["ALL_ELIGIBLE"]["D"]["Pearson"])),
        ("retroactive hypothesis confirmation", lambda r: r["hypotheses"][0].__setitem__("status", "SUPPORTED")),
        ("drop neutral focused audit", lambda r: r["focused_audit"]["neutral_cases"]["case_ids"].pop())]
    for name, mutate in changes:
        bad = copy.deepcopy(result); mutate(bad)
        try:
            validate(bad, raws, cases, fmap)
            validate_annotations(bad)
        except (AssertionError, KeyError, ValueError):
            tests.append({"probe": name, "result": "PASS_CORRUPTION_REJECTED"})
        else:
            raise AssertionError("Corruption accepted: " + name)
    (rc.OUT / "validation.json").write_bytes(rc.encode({"fidelity": "PASS", "protected_prior_files": len(protected),
        "independent_methods": "Rational raw-median slopes; rational normal equations/Gauss-Jordan; every LOOCV fold; independent archive features/counts; exact-tie ranks/groups",
        "adverse_synthetic": adverse, "corruption_probes": tests, "outcome_requirements": [], "no_GPU_Modal_compilation_or_new_timing": True}))
    print("STAGE D VALIDATOR PASS; all raw metrics/features/OLS/LOOCV/groups/correlations; adverse synthetic accepted")


if __name__ == "__main__":
    main()
