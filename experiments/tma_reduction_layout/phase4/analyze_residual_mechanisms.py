"""Stage D: outcome-informed offline analysis of frozen Stage B/C archives."""
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4 import residual_contract as rc
from experiments.tma_reduction_layout.phase4 import timing_contract as tc
from experiments.tma_reduction_layout.phase4 import stage_b_audit as sb
from experiments.tma_reduction_layout.phase4 import artifact_gate as gate
from experiments.tma_reduction_layout.gluon import artifact_checks as ac


def family_counts(fingerprint):
    counts = {}
    for family in rc.FAMILIES:
        prefix = family.rstrip("*").rstrip(".") if family == "bar.sync" else family.rstrip("*")
        counts[family] = sum(i.split()[0] == prefix if family == "bar.sync" else i.split()[0].startswith(prefix) for i in fingerprint)
    return counts


def local_load(instructions):
    opcodes = [i["opcode"] for i in instructions]
    widths, payloads = [], []
    for opcode in opcodes:
        match = re.fullmatch(r"ld\.shared(?:\.v([124]))?\.[bus](16|32|64)", opcode)
        widths.append(int(match[1] or 1) if match else None)
        payloads.append(int(match[1] or 1) * int(match[2]) // 8 if match else None)
    return {"opcode_sequence": opcodes, "instruction_count": len(opcodes), "vector_widths": widths,
        "total_payload_bytes_per_thread": sum(payloads) if all(v is not None for v in payloads) else None,
        "payload_status": "MECHANICALLY_DERIVED_PER_THREAD" if all(v is not None for v in payloads) else "UNDEFINED_FOR_UNSUPPORTED_OPCODE",
        "lines": [i["line"] for i in instructions]}


def difference(default, cand4):
    return {k: default.get(k, 0) - cand4.get(k, 0) for k in sorted(set(default) | set(cand4))}


def features(cases):
    frozen = sb.frozen_files(rc.GATE)
    structural = {c["config_id"]: c for c in frozen["structural_pool.json"]["transitions"]}
    context = sb.verify_sources(rc.GATE)
    result = {}
    for cfg, old in sorted(cases.items()):
        value = {k: old[k] for k in ("M", "N", "num_warps")}
        value.update({"canonical_reproduction_equivalence": old["single_reduction_equivalence"],
            "repeated_equivalence": old["repeated_reduction_equivalence"],
            "preload_encoding": old["repeated_localload_encoding"], "harnesses": {}})
        for h in ("canonical", "repeated"):
            pair = {}
            for c in ("default", "4"):
                a = sb.load_bundle(rc.GATE, structural[cfg], h, c, context)
                layout = gate.ttgir_contract(a["texts"]["ttgir"], [1, old["M"], old["N"]], old["num_warps"])
                blocked = layout["blocked"]
                loads = ac.initial_load_signature(a["texts"]["ptx"])
                if h == "canonical":
                    fp = gate.body_fingerprint(a["texts"]["ptx"], a["stage"])
                else:
                    loops = [e for e in ac.ptx_backedges(a["texts"]["ptx"]) if e["kind"] == "compiler_loop"]
                    rc.require(len(loops) == 1, "Frozen single unspecialized runtime R loop")
                    edge = loops[0]
                    fp = ac.fingerprint([i for i in ac.ptx_instructions(a["texts"]["ptx"]) if edge["start"] <= i["line"] <= edge["end"]])
                expected = old["canonical"][c]["reduction_fingerprint"] if h == "canonical" else old["repeated"][c]["runtime_loop_fingerprint"]
                rc.require(fp == expected, "Frozen fingerprint extraction: " + cfg + ":" + h + ":" + c)
                pair[c] = {"lanePart_M": blocked["threadsPerWarp"][1], "warpPart_M": blocked["warpsPerCTA"][1],
                    "vec": blocked["sizePerThread"][2], "blocked_layout": blocked, "shared_layout": layout["shared"],
                    "resources": a["resources"], "blocks_per_SM": a["occupancy"]["blocks_per_sm_actual_dynamic_smem"],
                    "LocalLoad": local_load(loads), "fingerprint_sequence": fp,
                    "opcode_multiset": dict(sorted(Counter(fp).items())), "family_counts": family_counts(fp),
                    "archive_cubin_sha256": a["cubin_sha256"]}
            d, c = pair["default"], pair["4"]
            pair["delta_default_minus_cand4"] = {
                "num_regs": d["resources"]["num_regs"] - c["resources"]["num_regs"],
                "blocks_per_SM": d["blocks_per_SM"] - c["blocks_per_SM"],
                "LocalLoad_instruction_count": d["LocalLoad"]["instruction_count"] - c["LocalLoad"]["instruction_count"],
                "LocalLoad_payload_bytes_per_thread": d["LocalLoad"]["total_payload_bytes_per_thread"] - c["LocalLoad"]["total_payload_bytes_per_thread"]
                    if d["LocalLoad"]["total_payload_bytes_per_thread"] is not None and c["LocalLoad"]["total_payload_bytes_per_thread"] is not None else None,
                "opcode_multiset": difference(d["opcode_multiset"], c["opcode_multiset"]),
                "family_counts": difference(d["family_counts"], c["family_counts"])}
            pair["candidate_pair_sequence_equivalence"] = gate.equivalence(d["fingerprint_sequence"], c["fingerprint_sequence"])
            value["harnesses"][h] = pair
        d, c = value["harnesses"]["canonical"]["default"], value["harnesses"]["canonical"]["4"]
        value["lane_transition"] = f'{d["lanePart_M"]}->{c["lanePart_M"]}'
        value["lanePart_reduction_ratio"] = d["lanePart_M"] / c["lanePart_M"]
        equivalences = [v for field in ("canonical_reproduction_equivalence", "repeated_equivalence") for v in value[field].values()]
        value["equivalence_group"] = "EXACT" if all(v == "EXACT_SEQUENCE_EQUIVALENT" for v in equivalences) else "PIPELINED"
        value["preload_group"] = "EXACT" if all(v == "EXACT" for v in value["preload_encoding"].values()) else "DIFFERENT_ENCODING"
        result[cfg] = value
    return result


def rows_from_raw(raws, cases, feature_map, taxonomy):
    metrics = rc.timing_metrics(raws)
    rows = []
    for cfg, case in sorted(cases.items()):
        v = metrics[cfg]
        G, D = rc.sign(v["G"]["invocation_values"]), rc.sign(v["D"]["invocation_values"])
        rows.append({"case_id": cfg, "class": case["final_class"], "origin": case["origin"],
            "metrics": v, "G_category": G, "D_category": D, "taxonomy": taxonomy[D + "/" + G],
            "features": feature_map[cfg]})
    return rows


def hypotheses(result):
    models = result["models"]["ALL_ELIGIBLE"]
    common = {"status": "OUTCOME_INFORMED_HYPOTHESIS", "development_data": "Stage C; hypothesis generation only",
        "confirmation_data": "Unmeasured Phase 5 held-out domain; no use of Stage C for confirmation",
        "lock_policy": "Do not alter hypothesis, scope, equations or comparison criteria after Phase 5 timing"}
    return [
        {**common, "hypothesis_id": "H5_01_EXACT_BODY_DIRECTIONAL_TRACKING",
         "hypothesis": "Within the newly gated exact-sequence scope, G and D have a positive continuous rank association.",
         "observed_motivation": {"PRIMARY_Spearman_G_D": result["correlations"]["PRIMARY"]["D"]["Spearman"],
            "case_ids": sorted(n for n, r in result["cases"].items() if r["class"] == "PRIMARY"),
            "limitation": "Development n=3, one anchor and two new shapes; LOO/new-only undefined. This is not established generalization."},
         "mechanistic_rationale": "Exact body reproduction preserves the frozen instruction sequence projection; variation in a body advantage may track canonical marginal cost within that equivalence scope, subject to context interactions.",
         "variables": ["G", "D", "exact_sequence_gate", "coverage"],
         "supporting_future_observation": "Held-out PRIMARY n>=5 with nonconstant G/D and positive Spearman(G,D) supports the directional prediction; report full sign agreement, effect sizes and coverage, without automatic established-generalization status.",
         "falsifying_future_observation": "Held-out PRIMARY n>=5 with nonconstant G/D and negative Spearman(G,D) contradicts the directional prediction.",
         "inconclusive": "PRIMARY n<5, constant/undefined correlation, or rho=0; no rescue by adding development data or upgrading PIPELINED cases.",
         "scope": "Every Phase 5 PRIMARY case admitted by the future frozen-equivalence artifact gate, across the full 24-source domain; no N64/w8 structural filter."},
        {**common, "hypothesis_id": "H5_02_FIXED_DIFFERENTIAL_ALONE_INSUFFICIENT",
         "hypothesis": "Adding the fixed-harness differential alone does not improve prospective prediction over the body-only linear model.",
         "observed_motivation": {"LOOCV_A_MAE": models["A"]["LOOCV"]["MAE"], "LOOCV_B_MAE": models["B"]["LOOCV"]["MAE"],
            "LOOCV_A_RMSE": models["A"]["LOOCV"]["RMSE"], "LOOCV_B_RMSE": models["B"]["LOOCV"]["RMSE"],
            "focused_cases": ["M128_N16_w4", "M128_N32_w8", "M32_N64_w4", "M64_N32_w4"]},
         "mechanistic_rationale": "Small R=0 differentials coexist with opposed/unresolved-body canonical outcomes. Fixed work need not capture register/live-range/scheduling interactions between the repeated and canonical execution contexts.",
         "variables": ["G", "D", "g0", "locked_model_A_predictions", "locked_model_B_predictions"],
         "supporting_future_observation": "Using the locked development A/B coefficients on all held-out timing-eligible cases (n>=5), MAE_B>=MAE_A and RMSE_B>=RMSE_A support the specified non-improvement prediction.",
         "falsifying_future_observation": "On that same held-out set, MAE_B<MAE_A and RMSE_B<RMSE_A contradict the specified non-improvement prediction.",
         "inconclusive": "Mixed MAE/RMSE ordering, fewer than five eligible cases, invalid protocol or undefined predictions; no refitting or performance-driven exclusions.",
         "scope": "All Phase 5 timing-eligible PRIMARY/SECONDARY cases; separately display strata and coverage. A/B are frozen descriptive predictors, not causal cost components."},
        {**common, "hypothesis_id": "H5_03_WARP_REGIME_CONTEXT_PREDICTION",
         "hypothesis": "The registered warp-regime covariate carries additional predictive context beyond D and g0 on held-out extrapolation cases.",
         "observed_motivation": {"Model_C_warp8_coefficient": models["C"]["coefficients"]["warp8"],
            "LOOCV_B_MAE": models["B"]["LOOCV"]["MAE"], "LOOCV_C_MAE": models["C"]["LOOCV"]["MAE"],
            "LOOCV_B_RMSE": models["B"]["LOOCV"]["RMSE"], "LOOCV_C_RMSE": models["C"]["LOOCV"]["RMSE"]},
         "mechanistic_rationale": "Warp count changes per-thread work and inter-warp execution context. The covariate may proxy those changes or other correlated codegen features; it is not an identified causal warp effect or an always-warp8 rule.",
         "variables": ["G", "D", "g0", "I(num_warps==8)", "locked_model_B_predictions", "locked_model_C_predictions"],
         "supporting_future_observation": "Using locked development B/C coefficients on all held-out timing-eligible cases with at least three cases in each warp regime, MAE_C<MAE_B and RMSE_C<RMSE_B support the specified predictive extension.",
         "falsifying_future_observation": "Under that coverage, MAE_C>=MAE_B and RMSE_C>=RMSE_B contradict the specified predictive extension.",
         "inconclusive": "Mixed MAE/RMSE ordering, fewer than three cases in either warp regime or undefined predictions; no new model terms or refitting.",
         "scope": "Entire Phase 5 timing-eligible cohort, across both warp regimes and full new-M domain. Does not select structural membership or establish production heuristics."}]


def focused_audit(result):
    def comparison(target, other):
        a, b = result["cases"][target], result["cases"][other]
        return {"target": target, "reference": other,
            "metric_mean_target_minus_reference": {k: a["metrics"][k]["mean"] - b["metrics"][k]["mean"] for k in ("G", "g0", "g1", "D", "residual")},
            "warps_target_reference": [a["features"]["num_warps"], b["features"]["num_warps"]],
            "canonical_preload_payload_default_target_reference": [a["features"]["harnesses"]["canonical"]["default"]["LocalLoad"]["total_payload_bytes_per_thread"],
                b["features"]["harnesses"]["canonical"]["default"]["LocalLoad"]["total_payload_bytes_per_thread"]],
            "register_deltas_target_reference": {h: [a["features"]["harnesses"][h]["delta_default_minus_cand4"]["num_regs"],
                b["features"]["harnesses"][h]["delta_default_minus_cand4"]["num_regs"]] for h in ("canonical", "repeated")},
            "canonical_family_delta_target_reference": [a["features"]["harnesses"]["canonical"]["delta_default_minus_cand4"]["family_counts"],
                b["features"]["harnesses"]["canonical"]["delta_default_minus_cand4"]["family_counts"]]}
    return {"resolved_opposition": {"target": "M128_N16_w4", "comparisons": [comparison("M128_N16_w4", n) for n in ("M64_N16_w4", "M128_N16_w8")],
        "plausible_mechanisms": "All three have exact preloads, the same lane transition and the same canonical/repeated register deltas. The opposed case has 32B/thread preload versus 16B in both neighbors, larger per-thread M work than the warp8 neighbor, and different absolute/family shuffle structure. These distinguish execution contexts but do not identify a cause; g0 does not capture all body/context interactions."},
        "canonical_positive_D_unresolved": {"case_ids": ["M128_N32_w8", "M32_N64_w4", "M64_N32_w4"],
            "comparisons": [comparison("M128_N32_w8", "M64_N32_w4"), comparison("M32_N64_w4", "M32_N64_w8")],
            "plausible_mechanisms": "The three cases span both warp regimes and both EXACT/DIFFERENT_ENCODING preload groups. Their small fixed-harness differentials coexist with positive G and unresolved D. Initial load width/count, body instruction mix, register allocation and scheduling interactions are candidate explanations; encoding mismatch and warp count alone do not distinguish all three."},
        "neutral_cases": {"case_ids": sorted(n for n, r in result["cases"].items() if r["G_category"] == r["D_category"] == "SIGN_UNRESOLVED"),
            "interpretation": "Retain all seven neutral cases and their signed continuous means/SDs. They may locate expression boundaries, but SIGN_UNRESOLVED is not practical equivalence and cannot justify declaring the optimization irrelevant."}}


def report(result):
    def number(v): return "UNDEFINED" if v is None else f"{v:.9g}"
    def metric(row, key):
        v = row["metrics"][key]
        return number(v["mean"]) + " ± " + number(v["sample_SD"])
    def table(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
            + ["| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in row) + " |" for row in rows])
    eq = {"EXACT_SEQUENCE_EQUIVALENT": "E", "PIPELINED_OPCODE_EQUIVALENT": "P", "REDUCTION_FINGERPRINT_MISMATCH": "M"}
    lines = ["# STAGE D REPORT", "", "OUTCOME-INFORMED EXPLORATORY ANALYSIS + HELD-OUT CONFIRMATORY EXTENSION PREREGISTRATION", "",
        "Stage C is the development/hypothesis-generation set. No Stage D finding is retroactively preregistered or confirmed by Stage C. "
        "No Modal/GPU, compilation, new timing, production-code/heuristic change or PR update was performed.", "",
        "Starting HEAD: `" + rc.BASELINE + "`. All pre-existing experiment files are checked byte-for-byte against that Git tree. "
        "Stage C remains PRIMARY=3, SECONDARY=15, EXCLUDE=2; PRIMARY n=3<5 remains insufficient for established generalization. "
        "H2a historical remains SUPPORTED_AT_REDUCTION_BODY_LEVEL; H2b/H2c remain UNVERIFIED.", "",
        result["residual_interpretation"], "", result["g0_interpretation"], "",
        "All values below are mean ± sample SD over the three paired invocation differentials, in ns/additional CTA. "
        "Invocation values for G/g0/g1/D/G-D, all extracted sequences/multisets, resource fields, every model/LOOCV fold and all correlations are retained in results.json.", "",
        "## Complete 18-case mechanism table", "",
        table(["Case", "Class / origin", "G", "g0", "g1", "D", "G-D", "G/D categories", "Taxonomy", "Lane transition"],
            [[n, r["class"] + "/" + r["origin"], metric(r, "G"), metric(r, "g0"), metric(r, "g1"), metric(r, "D"), metric(r, "residual"),
              r["G_category"] + "/" + r["D_category"], r["taxonomy"], r["features"]["lane_transition"]] for n, r in sorted(result["cases"].items())]), "",
        "E/P below retains EXACT_SEQUENCE_EQUIVALENT versus PIPELINED_OPCODE_EQUIVALENT relative to the canonical body. "
        "The canonical equivalence column is canonical-to-single reproduction, not default-to-cand4 body equivalence. "
        "Instruction counts/multisets and sequence order classifications are separate features. Deltas mean default−cand4, never weighted instruction costs.", "",
        table(["Case", "M/N/warps", "Vec d/4; warpPart d/4", "Canonical equivalence d/4", "Repeated equivalence d/4", "Preload d/4",
               "Registers canonical/repeated d→4", "Blocks/SM canonical/repeated d→4", "Initial LL count delta canonical/repeated", "LL bytes/thread canonical d/4; repeated d/4",
               "Canonical shfl/bar/ld/st/max/cvt/ldmatrix/selp deltas"],
            [[n, f'{r["features"]["M"]}/{r["features"]["N"]}/{r["features"]["num_warps"]}',
              f'{r["features"]["harnesses"]["canonical"]["default"]["vec"]}/{r["features"]["harnesses"]["canonical"]["4"]["vec"]}; '
              f'{r["features"]["harnesses"]["canonical"]["default"]["warpPart_M"]}/{r["features"]["harnesses"]["canonical"]["4"]["warpPart_M"]}',
              "/".join(eq[r["features"]["canonical_reproduction_equivalence"][c]] for c in ("default", "4")),
              "/".join(eq[r["features"]["repeated_equivalence"][c]] for c in ("default", "4")),
              "/".join(r["features"]["preload_encoding"][c] for c in ("default", "4")),
              "; ".join(h + ":" + "→".join(str(r["features"]["harnesses"][h][c]["resources"]["num_regs"]) for c in ("default", "4")) for h in ("canonical", "repeated")),
              "; ".join(h + ":" + "→".join(str(r["features"]["harnesses"][h][c]["blocks_per_SM"]) for c in ("default", "4")) for h in ("canonical", "repeated")),
              "/".join(str(r["features"]["harnesses"][h]["delta_default_minus_cand4"]["LocalLoad_instruction_count"]) for h in ("canonical", "repeated")),
              "; ".join(h + ":" + "/".join(str(r["features"]["harnesses"][h][c]["LocalLoad"]["total_payload_bytes_per_thread"]) for c in ("default", "4")) for h in ("canonical", "repeated")),
              "/".join(str(r["features"]["harnesses"]["canonical"]["delta_default_minus_cand4"]["family_counts"][f]) for f in ("shfl.*", "bar.sync", "ld.shared*", "st.shared*", "max.*", "cvt.*", "ldmatrix*", "selp.*"))]
              for n, r in sorted(result["cases"].items())]), "",
        "All shared fields remain distinct: driver static SMEM, dynamic launch SMEM and cuobjdump-reported SHARED. "
        "Here driver static is 0 and raw cuobjdump SHARED is 1024; neither is substituted for the other or used to infer occupancy. "
        "All LocalLoad sequences and resource fields follow; complete body fingerprint sequences and multisets are archived in results.json.", "",
        table(["Case / harness", "Lane reduction ratio", "Register delta", "Dynamic SMEM d/4", "Driver static SMEM d/4", "cuobjdump SHARED d/4", "LocalLoad opcode sequence d/4", "Vector widths d/4"],
            [[n + "/" + h, r["features"]["lanePart_reduction_ratio"], r["features"]["harnesses"][h]["delta_default_minus_cand4"]["num_regs"],
              *["/".join(str(r["features"]["harnesses"][h][c]["resources"][field]) for c in ("default", "4"))
                for field in ("dynamic_smem_bytes", "static_smem_bytes", "cuobjdump_reported_shared_bytes")],
              " / ".join(json.dumps(r["features"]["harnesses"][h][c]["LocalLoad"]["opcode_sequence"]) for c in ("default", "4")),
              " / ".join(json.dumps(r["features"]["harnesses"][h][c]["LocalLoad"]["vector_widths"]) for c in ("default", "4"))]
              for n, r in sorted(result["cases"].items()) for h in ("canonical", "repeated")]), "",
        "## Only the registered exploratory model family", "", result["model_policy"], "",
        "A: G~1+D; B: G~1+D+g0; C: G~1+D+g0+I(warps==8). The QR rank tolerance 1e-12 is numerical only. "
        "All fits use case means, with all invocation values retained separately; no p-values or significance claims.", "",
        table(["Scope", "Model", "n", "OLS coefficients", "In-sample R²", "LOOCV MAE", "LOOCV RMSE", "Defined/undefined folds"],
            [[s, name, m["n"], json.dumps(m["coefficients"], sort_keys=True) if m["coefficients"] is not None else "UNDEFINED",
              number(m["R_squared"]), number(m["LOOCV"]["MAE"]), number(m["LOOCV"]["RMSE"]), f'{m["LOOCV"]["defined_folds"]}/{m["LOOCV"]["undefined_folds"]}']
             for s, models in sorted(result["models"].items()) for name, m in sorted(models.items())]), "",
        "PRIMARY B is a saturated three-point fit; R²=1 is not validation and all its LOOCV folds are undefined. PRIMARY C is insufficient/singular. "
        "On all 18 development cases B has slightly higher LOOCV error than A, while C has lower error than A/B. These observations only motivate hypotheses; no model winner or causal warp claim.", "",
        "## Groupwise descriptive audit", "",
        "EXACT means all canonical-single and repeated candidate body reproductions are E; PIPELINED means at least one P. "
        "Preload EXACT means both candidates EXACT; otherwise DIFFERENT_ENCODING is present. Neither aggregation upgrades a Stage C class.", ""]
    for field, groups in sorted(result["groups"].items()):
        lines += [field + ":", "", table(["Group", "n", "small-n", "Mean G", "Mean D", "Mean G-D", "Resolved sign matches/denominator", "Unresolved pairs"],
            [[label, g["n"], g["small_n"], number(g["mean_G"]), number(g["mean_D"]), number(g["mean_descriptive_residual"]),
              f'{g["resolved_sign_agreement"]["numerator"]}/{g["resolved_sign_agreement"]["denominator"]}', g["resolved_sign_agreement"]["unresolved_pairs"]]
              for label, g in groups.items()]), ""]
    lines += ["## Exploratory correlations", "", result["score_interpretation"], "",
        table(["Scope", "Score versus G", "Pearson", "Spearman"], [[s, k, number(v["Pearson"]), number(v["Spearman"])] for s, scores in sorted(result["correlations"].items()) for k, v in sorted(scores.items())]), "",
        "## Focused opposition/unresolved/neutral audit", ""]
    audit = result["focused_audit"]
    for key in ("resolved_opposition", "canonical_positive_D_unresolved"):
        a = audit[key]
        lines += [key + ": " + a["plausible_mechanisms"], ""]
        for comparison in a["comparisons"]:
            lines += ["```json", json.dumps(comparison, indent=2, sort_keys=True), "```", ""]
    for target in ["M128_N16_w4", "M128_N32_w8", "M32_N64_w4", "M64_N32_w4"]:
        r = result["cases"][target]
        lines += [target + ": G=" + metric(r, "G") + "; g0=" + metric(r, "g0") + "; D=" + metric(r, "D") + "; residual=" + metric(r, "residual") + ".", ""]
        for h in ("canonical", "repeated"):
            f = r["features"]["harnesses"][h]
            lines += [h + " LocalLoad d/4: `" + json.dumps({c: f[c]["LocalLoad"]["opcode_sequence"] for c in ("default", "4")}) + "`; family counts d/4: `"
                + json.dumps({c: f[c]["family_counts"] for c in ("default", "4")}, sort_keys=True) + "`.", ""]
    lines += [audit["neutral_cases"]["interpretation"], "", table(["Neutral case", "G", "g0", "D", "G-D", "Lane", "Preload group"],
        [[n, metric(result["cases"][n], "G"), metric(result["cases"][n], "g0"), metric(result["cases"][n], "D"), metric(result["cases"][n], "residual"),
          result["cases"][n]["features"]["lane_transition"], result["cases"][n]["features"]["preload_group"]] for n in audit["neutral_cases"]["case_ids"]]), "",
        "## Outcome-informed hypotheses, for new held-out data only", ""]
    for h in result["hypotheses"]:
        lines += ["### " + h["hypothesis_id"], "", "OUTCOME_INFORMED_HYPOTHESIS", "", h["hypothesis"], "",
            "Observed development motivation: `" + json.dumps(h["observed_motivation"], sort_keys=True) + "`.", "",
            "Mechanistic rationale: " + h["mechanistic_rationale"], "", "Variables: " + ", ".join(h["variables"]) + ".", "",
            "Future support: " + h["supporting_future_observation"], "", "Future falsification: " + h["falsifying_future_observation"], "",
            "Inconclusive: " + h["inconclusive"], "", "Scope: " + h["scope"], ""]
    lines += ["## Phase 5 preregistration and STOP", "",
        "Full 24-source transition/layout/inclusion/reason table: ../phase5/preregistration/cohort_summary.md; future sampling/analysis/hypothesis locks: ../phase5/preregistration/protocol.json. "
        "No performance-dependent membership, expected PRIMARY filtering or N64/w8-only domain. Layouts are source-rule predictions, actual artifact/resource status pending.", "",
        "The frozen Stage A validator allows new paths only under phase4/. Phase 5 code therefore lives in phase4/phase5/ and outputs in phase4/results/phase5/preregistration/. "
        "This preserves every Stage A-C byte, including the validator SHA, and retains a distinct Phase 5 statistical role.", "",
        "Eight validators must pass: Phase 3 self-test/evidence, Stage A, Stage B, Stage C raw/analysis, Stage D residual analysis, Phase 5 preregistration. "
        "validation.json includes independent full recomputation, corruption probes, and a synthetic negative D/G, useless-g0, worse-C-LOOCV dataset accepted by fidelity checks.", "",
        "NO Phase 5 timing observed. NO Phase 5 artifact gate executed. No production heuristic, Coalesce.cpp modification or PR #11991 update. STOP.", ""]
    from experiments.tma_reduction_layout.phase4.phase5 import preregister as heldout
    pool = heldout.create_pool(heldout.source_domain())
    policy = heldout.protocol(result["hypotheses"], result)
    lines += ["## Full held-out 24-transition pool", "", heldout.render(pool, policy), ""]
    return "\n".join(lines).rstrip() + "\n"


def main():
    protected = rc.frozen_inventory()
    tc.validate_all()
    cases, _, _ = tc.inputs()
    raws = [rc.read(rc.TIMING / f"raw_invocation_{i}.json") for i in (1, 2, 3)]
    protocol = rc.read(sb.PREREG / "protocol.json")
    rows = rows_from_raw(raws, cases, features(cases), protocol["counterexamples"])
    result = rc.derive(rows)
    stage_c = rc.read(rc.TIMING / "results.json")
    for row in rows:
        original = stage_c["cases"][row["case_id"]]
        rc.require(row["class"] == original["class"] and row["origin"] == original["origin"]
            and row["G_category"] == original["G_canonical"]["category"] and row["D_category"] == original["delta_g1"]["category"]
            and row["taxonomy"] == original["counterexample_taxonomy"], "Stage C interpretation frozen")
    result["frozen_stage_c_scientific_status"] = stage_c["scientific_status"]
    result["lineage"] = {"starting_HEAD": rc.BASELINE, "Stage_C_results_SHA256": rc.sha((rc.TIMING / "results.json").read_bytes()),
        "Stage_C_raw_manifest_SHA256": rc.sha((rc.TIMING / "raw_manifest.json").read_bytes()),
        "Stage_B_cohort_SHA256": rc.sha((rc.GATE / "cohort_after_gate.json").read_bytes()),
        "protected_prior_inventory_SHA256": rc.sha(rc.encode(protected)),
        "Stage_D_source_SHA256": {p.relative_to(ROOT).as_posix(): rc.sha(p.read_bytes()) for p in
            (Path(__file__), Path(__file__).with_name("validate_residual_analysis.py"), Path(rc.__file__))}}
    result["hypotheses"] = hypotheses(result)
    result["focused_audit"] = focused_audit(result)
    rc.OUT.mkdir(parents=True, exist_ok=True)
    (rc.OUT / "results.json").write_bytes(rc.encode(result))
    (rc.OUT / "summary.md").write_text(report(result))
    print(json.dumps({s: {m: {k: a[k] for k in ("coefficients", "R_squared", "fit_status")} | {"LOOCV_MAE": a["LOOCV"]["MAE"], "LOOCV_RMSE": a["LOOCV"]["RMSE"]} for m, a in models.items()} for s, models in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
