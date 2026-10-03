"""Stage C frozen descriptive analysis, independent of the timing runner."""
from collections import Counter
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3] if Path(__file__).parent.name == "phase4" else Path.cwd()
sys.path.insert(0, str(ROOT))
from experiments.tma_reduction_layout.phase4 import timing_contract as tc
from experiments.tma_reduction_layout.phase4 import analysis_contract as frozen


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def stats(values):
    return {"count": len(values), "median_us": st.median(values), "mean_us": st.mean(values),
        "sample_SD_us": st.stdev(values), "IQR_us": percentile(values, .75) - percentile(values, .25),
        "min_us": min(values), "max_us": max(values), "p10_us": percentile(values, .1), "p90_us": percentile(values, .9)}


def fit(points):
    x, y = [p[0] for p in points], [p[1] for p in points]
    mx, my = st.mean(x), st.mean(y)
    b = sum((a - mx) * (v - my) for a, v in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    a = my - b * mx
    residuals = [v - (a + b * size) for size, v in zip(x, y)]
    total = sum((v - my) ** 2 for v in y)
    return {"points": [{"B_RUN": b_run, "median_us": value} for b_run, value in points],
        "intercept_us": a, "slope_ns_per_additional_CTA": b * 1000,
        "R_squared": 1 - sum(r * r for r in residuals) / total if total else None,
        "residuals_us": residuals}


def pearson(x, y):
    if len(x) < 3: return None
    mx, my = st.mean(x), st.mean(y)
    xx, yy = [v - mx for v in x], [v - my for v in y]
    sx, sy = sum(v * v for v in xx), sum(v * v for v in yy)
    return sum(a * b for a, b in zip(xx, yy)) / math.sqrt(sx * sy) if sx and sy else None


def sign_table(rows):
    categories = ("POSITIVE", "NEGATIVE", "SIGN_UNRESOLVED")
    matrix = {d: {g: 0 for g in categories} for d in categories}
    numerator = denominator = unresolved = exact = 0
    for row in rows:
        d, g = row["delta_g1"]["category"], row["G_canonical"]["category"]
        matrix[d][g] += 1
        exact += d == g
        if "SIGN_UNRESOLVED" in (d, g): unresolved += 1
        else:
            denominator += 1
            numerator += d == g
    return {"axis_order": "delta_g1/G_canonical", "matrix": matrix,
        "three_category_exact_agreement": {"numerator": exact, "denominator": len(rows)},
        "resolved_sign_agreement": {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None},
        "SIGN_UNRESOLVED_pair_coverage": {"count": unresolved, "denominator": len(rows)}}


def stratum(rows, primary):
    names = sorted(rows)
    x, y = [rows[n]["delta_g1"]["mean"] for n in names], [rows[n]["G_canonical"]["mean"] for n in names]
    rx, ry = frozen.average_ranks(x), frozen.average_ranks(y)
    rankings = [{"case_id": name, "origin": rows[name]["origin"], "delta_rank": a, "G_rank": b,
                 "score": abs(a - b) / (len(names) - 1) if len(names) > 1 else None} for name, a, b in zip(names, rx, ry)]
    rankings.sort(key=lambda r: (-r["score"], r["case_id"]))
    result = {"n": len(names), "case_ids": names,
        "association": {"method": "Spearman" if primary else "Pearson",
            "coefficient": frozen.spearman(x, y) if primary else pearson(x, y)},
        "average_ranks": {n: {"delta_g1": a, "G_canonical": b} for n, a, b in zip(names, rx, ry)},
        "sign_accounting": sign_table(list(rows.values())),
        "counterexample_distribution": dict(sorted(Counter(r["counterexample_taxonomy"] for r in rows.values()).items())),
        "rank_disagreement": rankings, "top_rank_disagreement": rankings[:min(5, len(names))],
        "coverage": {field: dict(sorted(Counter(str(r[field]) for r in rows.values()).items())) for field in ("M", "N", "num_warps", "lanePart_transition")}}
    if primary:
        result["leave_one_out"] = frozen.leave_one_out_spearman(dict(zip(names, zip(x, y))))
        new_names = [n for n in names if rows[n]["origin"] == "NEW_GENERALIZATION"]
        result["new_only_sensitivity"] = {"n": len(new_names), "case_ids": new_names,
            "Spearman": frozen.spearman([rows[n]["delta_g1"]["mean"] for n in new_names], [rows[n]["G_canonical"]["mean"] for n in new_names])}
    return result


def structure(case):
    return {"repeated_reduction_equivalence": case["repeated_reduction_equivalence"],
        "lanePart_transition": case["lanePart_transition"],
        "layouts": {h: {c: {"lanePart_M": case[h][c]["observed_layout"]["blocked"]["threadsPerWarp"][1],
            "warpPart_M": case[h][c]["observed_layout"]["blocked"]["warpsPerCTA"][1],
            "vec": case[h][c]["observed_layout"]["blocked"]["sizePerThread"][2],
            "swizzlingByteWidth": case[h][c]["observed_layout"]["shared"]["swizzlingByteWidth"],
            "blocks_per_sm_actual_dynamic_smem": case["occupancy"][h][c]["blocks_per_sm_actual_dynamic_smem"]}
            for c in ("default", "4")} for h in ("canonical", "repeated")}}


def derive(raws, cases, protocol):
    condition_stats, fits, rows, fit_records = {}, {}, {}, []
    runs = {}
    for raw in raws:
        inv = str(raw["invocation"])
        samples = {}
        for visit in raw["visits"]:
            samples.setdefault(visit["condition_tag"], []).extend(visit["samples_us"])
        if any(len(v) != 100 for v in samples.values()): raise ValueError("Exactly 100 raw samples per condition")
        condition_stats[inv] = {tag: stats(v) for tag, v in samples.items()}
        fits[inv], runs[inv] = {}, {}
        for cfg in sorted(cases):
            fits[inv][cfg] = {"canonical": {}, "repeated": {}}
            for h in ("canonical", "repeated"):
                for c in ("default", "4"):
                    if h == "repeated": fits[inv][cfg][h][c] = {}
                    for r in ((None,) if h == "canonical" else (0, 1)):
                        points = [(b, condition_stats[inv][f"{cfg}:{h}:{c}:R{r}:B{b}"]["median_us"]) for b in (16384, 32768, 65536)]
                        value = fit(points)
                        if h == "canonical": fits[inv][cfg][h][c] = value
                        else: fits[inv][cfg][h][c][str(r)] = value
                        fit_records.append({"invocation": int(inv), "case_id": cfg, "harness": h,
                            "candidate": c, "R": r, "R_squared": value["R_squared"], "residuals_us": value["residuals_us"]})
            f = fits[inv][cfg]
            G = f["canonical"]["default"]["slope_ns_per_additional_CTA"] - f["canonical"]["4"]["slope_ns_per_additional_CTA"]
            g = {r: f["repeated"]["default"][str(r)]["slope_ns_per_additional_CTA"] - f["repeated"]["4"][str(r)]["slope_ns_per_additional_CTA"] for r in (0, 1)}
            runs[inv][cfg] = {"G_canonical": G, "g0": g[0], "g1": g[1], "delta_g1": g[1] - g[0]}
    for cfg, case in cases.items():
        G = frozen.sign_resolution([runs[str(i)][cfg]["G_canonical"] for i in (1, 2, 3)])
        D = frozen.sign_resolution([runs[str(i)][cfg]["delta_g1"] for i in (1, 2, 3)])
        G["invocation_values"] = [runs[str(i)][cfg]["G_canonical"] for i in (1, 2, 3)]
        D["invocation_values"] = [runs[str(i)][cfg]["delta_g1"] for i in (1, 2, 3)]
        categories = D["category"] + "/" + G["category"]
        relationship = "BOTH_SIGN_UNRESOLVED" if categories == "SIGN_UNRESOLVED/SIGN_UNRESOLVED" else "ONE_SIGN_UNRESOLVED" if "SIGN_UNRESOLVED" in categories else "RESOLVED_AGREEMENT" if D["category"] == G["category"] else "RESOLVED_OPPOSITION"
        rows[cfg] = {"case_id": cfg, "class": case["final_class"], "origin": case["origin"],
            **{k: case[k] for k in ("M", "N", "num_warps", "lanePart_transition")},
            "structural_attributes": structure(case),
            "G_canonical": G, "delta_g1": D, "invocation_differentials": {i: runs[i][cfg] for i in runs},
            "sign_relationship": relationship, "counterexample_grid": categories,
            "counterexample_taxonomy": protocol["counterexamples"][categories],
            "lowest_canonical_R_squared": min((r["R_squared"] for r in fit_records if r["case_id"] == cfg and r["harness"] == "canonical" and r["R_squared"] is not None), default=None),
            "lowest_repeated_R_squared": min((r["R_squared"] for r in fit_records if r["case_id"] == cfg and r["harness"] == "repeated" and r["R_squared"] is not None), default=None)}
    strata = {name: stratum({n: r for n, r in rows.items() if r["class"] == name}, name == "PRIMARY") for name in ("PRIMARY", "SECONDARY")}
    for name, stratum_ in strata.items():
        for item in stratum_["rank_disagreement"]:
            rows[item["case_id"]]["outlier_score"] = item["score"]
    lowest = {}
    for h in ("canonical", "repeated"):
        selected = [f for f in fit_records if f["harness"] == h and f["R_squared"] is not None]
        lowest[h] = sorted(selected, key=lambda r: (r["R_squared"], r["case_id"], r["invocation"], r["candidate"], str(r["R"])))[:5]
    return {"units": "ns/additional CTA; marginal grid-time slope, not single-CTA latency",
        "condition_statistics": condition_stats, "fits": fits, "cases": rows, "strata": strata,
        "all_eligible_descriptive_sign_accounting": sign_table(list(rows.values())),
        "fit_diagnostics": {"all_fit_count": len(fit_records), "lowest_five": lowest,
            "handling": "All fits/cases retained; three B points per fit; no R-squared eligibility threshold or performance-driven retry"},
        "scientific_status": {"H2a_historical": "SUPPORTED_AT_REDUCTION_BODY_LEVEL", "H2b": "UNVERIFIED", "H2c": "UNVERIFIED",
            "phase4_primary_generalization_status": "INSUFFICIENT_PRIMARY_COHORT_SIZE_FOR_ESTABLISHED_GENERALIZATION",
            "interpretation": "PRIMARY n=3<5: cross-shape generalization is not established. SECONDARY is a separate broader, lower-equivalence-strength descriptive stratum; structured factorial cohort, no population correlation inference or additive causal evidence."}}


def synthetic(cases, plan, tied=False):
    names = sorted(cases)
    primary = sorted(n for n in names if cases[n]["final_class"] == "PRIMARY")
    raws = []
    for inv in plan["invocations"]:
        visits = []
        for round_ in inv["rounds"]:
            for tag in round_["order"]:
                p = tc.condition(tag)
                g = float(names.index(p["config_id"]) + 1)
                d = -g
                if tied and p["config_id"] in primary:
                    index = primary.index(p["config_id"])
                    g, d = [1., 1., 2.][index], [3., 2., 2.][index]
                slope = 100.
                if p["candidate"] == "default":
                    if p["harness"] == "canonical": slope += g
                    elif p["R"] == 1: slope += d
                value = 20. + slope * p["B_RUN"] / 1000.
                visits.append({"condition_tag": tag, "round": round_["round"], "samples_us": [value] * 10})
        raws.append({"invocation": inv["invocation"], "visits": visits})
    return raws


def raw_commit():
    value = subprocess.check_output(["git", "log", "--format=%H", "--max-count=1", "--grep=^experiment: record Phase 4 cross-shape timing$"], cwd=ROOT, text=True).strip()
    if not value: raise ValueError("Freeze actual raw timing in the required first commit before analysis")
    return value


def anchor_comparison(result):
    path = ROOT / "experiments/tma_reduction_layout/results/phase3/gluon_timing/results.json"
    historical = json.loads(path.read_text())
    cfg = historical["primary_analysis"]["config"]
    now = result["cases"][cfg]
    old = historical["primary_analysis"]["cross_run"]["delta_g_1"]
    base = historical["canonical_baseline"]
    return {"case_id": cfg, "historical_evidence_path": path.relative_to(ROOT).as_posix(),
        "historical_evidence_sha256": tc.sha(path.read_bytes()),
        "Phase3_delta_g1": {"mean": old["mean"], "sample_SD": old["std"]},
        "Phase3_historical_canonical_baseline": {"mean": base["gap_ns_per_cta"],
            "sample_SD": st.stdev(v["default"] - v["4"] for v in base["per_run_slopes"].values()),
            "origin": "Phase 2 historical canonical baseline reused descriptively in Phase 3"},
        "Phase4_G_canonical": now["G_canonical"], "Phase4_delta_g1": now["delta_g1"],
        "Phase4_delta_over_fresh_canonical": now["delta_g1"]["mean"] / now["G_canonical"]["mean"] if now["G_canonical"]["mean"] != 0 else None,
        "interpretation": "Descriptive temporal comparison only. Phase 4 uses its fresh canonical denominator; no historical-denominator attribution or additive causal claim."}


def report(result, execution):
    def number(x): return "UNDEFINED" if x is None else f"{x:.9g}"
    def metric(m): return f'{number(m["mean"])} ± {number(m["sample_SD"])}'
    def table(headers, rows):
        return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
            + ["| " + " | ".join(str(x).replace("|", "\\|").replace("\n", " ") for x in row) + " |" for row in rows])
    invalid_records = [json.loads(p.read_text()) for p in (tc.OUT / "invalid_invocations").glob("*.json")]
    startup_failures = sum(r.get("invocation") is None for r in invalid_records)
    dispatched_failures = len(invalid_records) - startup_failures
    raw_invocations = [json.loads((tc.OUT / f"raw_invocation_{i}.json").read_text()) for i in (1, 2, 3)]
    text = ["# Phase 4 Stage C — Frozen-Binary Timing and Pre-Registered Analysis", "",
        "Starting HEAD: `" + tc.BASELINE + "`. Raw timing commit: `" + result["lineage"]["raw_timing_commit"] + "`.", "",
        "18 eligible cases (3 PRIMARY, 15 SECONDARY); 2 exclusions absent. Three separately dispatched single-use Modal H100 benchmark containers. "
        "Each invocation has 324 conditions (108 canonical, 216 repeated), 10 rounds × 10 scalar CUDA-event samples per condition: 972 invocation-conditions, 9,720 visits, 97,200 samples (32,400 canonical; 64,800 repeated).", "",
        f'Invalid infrastructure/protocol records retained: {execution["invalid_invocations"]} '
        f'({startup_failures} Modal App startup failures before any dispatch or GPU sample; '
        f'{dispatched_failures} invalid dispatched benchmark invocations). '
        'Every timed and warmup launch is guarded against the Stage B archived CUBIN SHA; canonical reuses its exact binary across B, repeated across R/B. No timing-kernel compilation. '
        'Three untimed warmup launches per condition, in first frozen round order, were fixed before collecting outcomes.', "",
        'GPU UUIDs, in invocation order: ' + ', '.join('`' + u + '`' for u in execution['gpu_uuids']) + '. Replication: ' + execution['replication_mode'] + '.', "",
        table(['Invocation', 'Modal profile', 'GPU / CC', 'Driver API / driver', 'Queried libcudart / Torch CUDA build', 'PState start→end',
               'SM / memory clock MHz start→end', 'Power W start→end', 'Temperature C start→end'],
            [[r['invocation'], r['environment']['modal_profile'], r['environment']['gpu_name'] + ' / ' + str(r['environment']['compute_capability']),
              str(r['environment']['driver_API_version']) + ' / ' + r['telemetry_start'].get('driver_version', 'UNAVAILABLE'),
              str(r['environment']['CUDA_runtime_version']) + ' / ' + r['environment']['torch_cuda'],
              r['telemetry_start'].get('pstate', 'UNAVAILABLE') + '→' + r['telemetry_end'].get('pstate', 'UNAVAILABLE'),
              r['telemetry_start'].get('clocks.current.sm', 'UNAVAILABLE') + '/' + r['telemetry_start'].get('clocks.current.memory', 'UNAVAILABLE') + '→'
                + r['telemetry_end'].get('clocks.current.sm', 'UNAVAILABLE') + '/' + r['telemetry_end'].get('clocks.current.memory', 'UNAVAILABLE'),
              r['telemetry_start'].get('power.draw', 'UNAVAILABLE') + '→' + r['telemetry_end'].get('power.draw', 'UNAVAILABLE'),
              r['telemetry_start'].get('temperature.gpu', 'UNAVAILABLE') + '→' + r['telemetry_end'].get('temperature.gpu', 'UNAVAILABLE')]
             for r in raw_invocations]), "",
        "All displayed G and Δg(1) values are mean ± sample SD in ns/additional CTA. This is a marginal grid-time slope, not single-CTA latency. "
        "The frozen t band (t=4.302652729911275, df=2; half-width=t×SD/√3) is used only for sign resolution. Continuous values remain unrounded for association and rank calculations.", "",
        "## Complete eligible cohort", "",
        table(["Case", "Class", "Origin", "G canonical mean ± SD", "G category", "Min canonical R²", "Δg(1) mean ± SD", "D category", "Min repeated R²", "Sign relationship", "Taxonomy", "Outlier score"],
            [[n, r['class'], r['origin'], metric(r['G_canonical']), r['G_canonical']['category'], number(r['lowest_canonical_R_squared']),
              metric(r['delta_g1']), r['delta_g1']['category'], number(r['lowest_repeated_R_squared']), r['sign_relationship'], r['counterexample_taxonomy'], number(r['outlier_score'])]
             for n, r in sorted(result['cases'].items())])]
    for name, s in result['strata'].items():
        text += ["", "## " + name, "", f'n={s["n"]}; {s["association"]["method"]}(mean Δg(1), mean G canonical)={number(s["association"]["coefficient"])}.', ""]
        if name == 'PRIMARY':
            text += ["PRIMARY n=3<5: cross-shape generalization is not established, regardless of observed rho. "
                'Leave-one-case-out results: ' + ', '.join(n + '=' + number(v) for n, v in s['leave_one_out']['rho_minus_i'].items())
                + '; defined=' + str(s['leave_one_out']['defined_count']) + ', undefined=' + str(s['leave_one_out']['undefined_count']) + '.', "",
                f'New-generalization-only n={s["new_only_sensitivity"]["n"]}, Spearman={number(s["new_only_sensitivity"]["Spearman"])}. '
                'The anchor remains in PRIMARY; new-only sensitivity is undefined below n=3.', ""]
        else:
            direction = 'undefined' if s['association']['coefficient'] is None else 'positive' if s['association']['coefficient'] > 0 else 'negative' if s['association']['coefficient'] < 0 else 'zero'
            text += ['SECONDARY is the broader, lower-equivalence-strength descriptive stratum. Observed Pearson direction: ' + direction + '. '
                'No post-hoc correlation threshold or support category is applied. PRIMARY and SECONDARY are separate evidence strata within a structured factorial cohort.', ""]
        a = s['sign_accounting']; categories = ('POSITIVE', 'NEGATIVE', 'SIGN_UNRESOLVED')
        text += [table(['D category / G category', *categories], [[d, *[a['matrix'][d][g] for g in categories]] for d in categories]), "",
            f'Resolved sign agreement: {a["resolved_sign_agreement"]["numerator"]}/{a["resolved_sign_agreement"]["denominator"]} '
            f'(rate={number(a["resolved_sign_agreement"]["rate"])}). SIGN_UNRESOLVED pair coverage: '
            f'{a["SIGN_UNRESOLVED_pair_coverage"]["count"]}/{a["SIGN_UNRESOLVED_pair_coverage"]["denominator"]}. '
            f'Three-category exact agreement: {a["three_category_exact_agreement"]["numerator"]}/{a["three_category_exact_agreement"]["denominator"]}.', "",
            'Coverage: `' + json.dumps(s['coverage'], sort_keys=True) + '`.', "",
            'Nine-grid case distribution: `' + json.dumps(s['counterexample_distribution'], sort_keys=True) + '`.', "",
            'Rank-disagreement audit (score=|average rank(D)−average rank(G)|/(n−1); descending score, then case ID):', "",
            table(['Case', 'Origin', 'D rank', 'G rank', 'Score'], [[r['case_id'], r['origin'], number(r['delta_rank']), number(r['G_rank']), number(r['score'])] for r in s['rank_disagreement']]), "",
            'Top min(5,n): ' + ', '.join(r['case_id'] for r in s['top_rank_disagreement']) + '. Every case is retained.']
    anchor = result['anchor_reproducibility']; old = anchor['Phase3_historical_canonical_baseline']
    text += ["", "## Anchor temporal comparison", "",
        anchor['case_id'] + ': Phase 4 G=' + metric(anchor['Phase4_G_canonical']) + ', Δg(1)=' + metric(anchor['Phase4_delta_g1']) + '.', "",
        'Phase 3 historical Δg(1)=' + metric(anchor['Phase3_delta_g1']) + '; its Phase 2 canonical baseline G=' + metric(old) + '. '
        'Fresh Phase 4 descriptive D/G=' + number(anchor['Phase4_delta_over_fresh_canonical']) + '. ' + anchor['interpretation'], "",
        "## Full counterexample and structural audit", "",
        'All nine-grid classifications are listed, including aligned, unresolved, masked, and opposed cases. No category is dropped.', "",
        table(['Case', 'G', 'D', 'D/G categories', 'Taxonomy', 'M×N / warps / lanePart', 'Warp partition default→4; canonical / repeated', 'Body equivalence'],
            [[n, metric(r['G_canonical']), metric(r['delta_g1']), r['counterexample_grid'], r['counterexample_taxonomy'],
              f'{r["M"]}×{r["N"]} / {r["num_warps"]} / {r["lanePart_transition"]}',
              '; '.join(h + ': ' + str(r['structural_attributes']['layouts'][h]['default']['warpPart_M']) + '→' + str(r['structural_attributes']['layouts'][h]['4']['warpPart_M']) for h in ('canonical', 'repeated')),
              r['structural_attributes']['repeated_reduction_equivalence']]
             for n, r in sorted(result['cases'].items())]), "",
        "## Fit diagnostics", "",
        'All 324 OLS fits use exactly three B points and retain all cases, slopes, intercepts, R² and residuals in results.json. '
        'No R² cutoff is introduced. These are the five lowest finite R² fits per harness for inspection; constant-response fits have undefined R² and remain in the full results.', ""]
    for h, fits in result['fit_diagnostics']['lowest_five'].items():
        text += [h + ':', "", table(['Case', 'Invocation', 'Candidate', 'R', 'R²', 'Residuals (us)'],
            [[f['case_id'], f['invocation'], f['candidate'], f['R'], number(f['R_squared']), ', '.join(number(v) for v in f['residuals_us'])] for f in fits]), ""]
    text += ['All-eligible pooled signs are descriptive only: `' + json.dumps(result['all_eligible_descriptive_sign_accounting'], sort_keys=True) + '`.', "",
        "## Fidelity and scientific status", "",
        'Raw integrity and corruption probes: raw_validation.json. Independent rational-moment/OLS/rank analysis recomputation and adverse-outcome synthetic acceptance: validation.json. '
        'Final verification commands: validate_evidence.py --self-test; validate_evidence.py; validate_preregistration.py; '
        'validate_artifact_gate.py; validate_timing_raw.py; validate_timing_analysis.py. '
        'All six commands must pass after the analysis commit before the normal push. No raw file is changed after the first raw timing commit.', "",
        'Phase 4 PRIMARY status: `' + result['scientific_status']['phase4_primary_generalization_status'] + '`. '
        'H2a historical status remains SUPPORTED_AT_REDUCTION_BODY_LEVEL; H2b and H2c remain UNVERIFIED. '
        'No production heuristic is proposed, no upstream PR is modified, and Stage D is not started.', ""]
    return "\n".join(text)


def main():
    tc.validate_all()
    commit = raw_commit()
    cases, plan, _ = tc.inputs()
    raws = [json.loads((tc.OUT / f"raw_invocation_{i}.json").read_text()) for i in (1, 2, 3)]
    protocol = json.loads((tc.stage_b.PREREG / "protocol.json").read_text())
    result = derive(raws, cases, protocol)
    result["lineage"] = {"raw_timing_commit": commit, "raw_manifest_sha256": tc.sha((tc.OUT / "raw_manifest.json").read_bytes()),
        "protocol_sha256": tc.sha((tc.stage_b.PREREG / "protocol.json").read_bytes())}
    result["anchor_reproducibility"] = anchor_comparison(result)
    (tc.OUT / "results.json").write_bytes(tc.encode(result))
    (tc.OUT / "summary.md").write_text(report(result, tc.validate_all()))
    print("Derived all 18 cases from frozen raw medians; no fit or counterexample removed")


if __name__ == "__main__":
    main()
