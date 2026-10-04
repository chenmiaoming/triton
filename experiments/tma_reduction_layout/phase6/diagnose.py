"""Stage A: rederive existing measurements and audit all prediction failures."""
from collections import Counter
import copy
import json
import math
from pathlib import Path
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.tma_reduction_layout.phase6 import common as c

DEST = c.OUT / "stage_a"


def derive():
    historical = c.inventory()
    all_rows, raw_bindings = {}, {}
    for phase, gate_name, timing_name in ((4, "artifact_gate", "timing"),
                                          (5, "phase5_artifact_gate", "phase5_timing")):
        gate = c.BASE / "phase4/results" / gate_name
        timing = c.BASE / "phase4/results" / timing_name
        cohort = c.read(gate / "cohort_after_gate.json")
        originals = c.read(timing / "results.json")["cases"]
        cases = [row for row in cohort["cases"] if row["pre_timing_eligible"]]
        run_metrics = {}
        for invocation in (1, 2, 3):
            raw_path = timing / f"raw_invocation_{invocation}.json"
            raw = c.read(raw_path)
            raw_bindings[raw_path.relative_to(c.ROOT).as_posix()] = c.sha(raw_path.read_bytes())
            _, fits = c.raw_fits(raw)
            for row in cases:
                cfg = row["case_id"]
                def slope(h, candidate, r):
                    return fits[f"{cfg}:{h}:{candidate}:R{r}"]["slope_ns_per_additional_CTA"]
                G = slope("canonical", "default", None)-slope("canonical", "4", None)
                g = {r: slope("repeated", "default", r)-slope("repeated", "4", r) for r in (0, 1)}
                run_metrics.setdefault(cfg, []).append({"G": G, "g0": g[0], "g1": g[1], "D": g[1]-g[0]})
        for row in cases:
            cfg = row["case_id"]
            summary = {metric: c.sign([run[metric] for run in run_metrics[cfg]])
                       for metric in ("G", "g0", "g1", "D")}
            old = originals[cfg]
            for metric, old_name in (("G", "G_canonical" if phase == 4 else "G"),
                                     ("D", "delta_g1" if phase == 4 else "D")):
                c.require(math.isclose(summary[metric]["mean"], old[old_name]["mean"], abs_tol=1e-9),
                          "Independent raw recomputation disagrees: " + cfg + metric)
            context = {"resources": row["resources"], "layout": {}, "body_counts": {},
                       "blocks_per_sm": {}, "preload_bytes_per_thread": {}, "body_equivalence": {}}
            for h in ("canonical", "single", "repeated"):
                context["layout"][h] = {candidate: row[h][candidate]["observed_layout"]
                                         for candidate in ("default", "4")}
                context["blocks_per_sm"][h] = {candidate: row["occupancy"][h][candidate]["blocks_per_sm_actual_dynamic_smem"]
                                                for candidate in ("default", "4")}
                field = {"canonical": "reduction_fingerprint", "single": "gluon_fingerprint",
                         "repeated": "runtime_loop_fingerprint"}[h]
                context["body_counts"][h] = {candidate: dict(Counter(row[h][candidate][field]))
                                              for candidate in ("default", "4")}
            context["body_equivalence"] = {h: {candidate: row[h][candidate]["classification"]
                                               for candidate in ("default", "4")} for h in ("single", "repeated")}
            context["preload_bytes_per_thread"] = {candidate: row["repeated"][candidate]["payload_bytes_per_thread"]
                                                    for candidate in ("default", "4")}
            occupancy = [v for h in context["blocks_per_sm"].values() for v in h.values()]
            all_rows[cfg] = {"phase": phase, "class": row["final_class"],
                             **{k: row[k] for k in ("M", "N", "num_warps", "lanePart_transition")},
                             **summary, "context": context,
                             "cross_harness_residency_matched": len(set(occupancy)) == 1,
                             "gate_path": gate.relative_to(c.ROOT).as_posix()}
    policy = c.read(c.BASE / "phase4/results/phase5/preregistration/protocol.json")
    coefficients = policy["locked_development_models"]
    errors = []
    for cfg, row in all_rows.items():
        if row["phase"] != 5:
            continue
        row["frozen_predictions"] = {}
        for model, co in coefficients.items():
            terms = {"intercept": co["intercept"], "D": co.get("D", 0)*row["D"]["mean"],
                     "g0": co.get("g0", 0)*row["g0"]["mean"],
                     "warp8": co.get("warp8", 0)*int(row["num_warps"] == 8)}
            pred = sum(terms.values())
            error = pred-row["G"]["mean"]
            record = {"case_id": cfg, "model": model, "M": row["M"], "num_warps": row["num_warps"],
                      "prediction_terms": terms, "prediction": pred, "prediction_minus_G": error,
                      "absolute_error": abs(error), "squared_error": error*error}
            row["frozen_predictions"][model] = record
            errors.append(record)
    groups = {}
    for model in coefficients:
        for label, predicate in (("ALL_13", lambda r: True),
                                 ("M256_OR_512_W8", lambda r: r["M"] >= 256 and r["num_warps"] == 8),
                                 ("W4", lambda r: r["num_warps"] == 4), ("W8", lambda r: r["num_warps"] == 8)):
            rows = [r for r in errors if r["model"] == model and predicate(r)]
            groups[f"{model}:{label}"] = {"n": len(rows), "MAE": st.mean(r["absolute_error"] for r in rows),
                                        "RMSE": math.sqrt(st.mean(r["squared_error"] for r in rows)),
                                        "mean_prediction_bias": st.mean(r["prediction_minus_G"] for r in rows)}
    return {"phase": "PHASE_6_STAGE_A_RETROSPECTIVE_DIAGNOSTIC", "baseline": c.BASELINE,
            "protected_inventory_SHA256": c.sha(c.encode(historical)), "protected_files": len(historical),
            "raw_input_SHA256": raw_bindings, "recomputed_cases": all_rows,
            "diagnostic_family": {cfg: all_rows[cfg] for cfg in c.FAMILY},
            "all_39_prediction_errors": sorted(errors, key=lambda r: (-r["absolute_error"], r["case_id"], r["model"])),
            "descriptive_error_groups": groups, "frozen_coefficients": coefficients,
            "limits": ["Outcome-informed diagnostic selection; all five cases already observed. No new confirmation.",
                       "Changing M changes payload/resources/body together. No isolated M/lane/warp causality.",
                       "Opcode sequence equivalence is not operand/dataflow or whole-program equivalence.",
                       "Same blocks/SM does not establish same register live ranges or scheduling.",
                       "SIGN_UNRESOLVED is not practical equivalence; G-D is descriptive.",
                       "No model refit; original H5_01/02/03 decisions and H2 statuses unchanged."]}


def report(result):
    rows = []
    for cfg in c.FAMILY:
        r = result["diagnostic_family"][cfg]
        ctx = r["context"]
        regs = "; ".join(h + ":" + str(ctx["resources"][h]["default"]["num_regs"]) + "→" +
                         str(ctx["resources"][h]["4"]["num_regs"]) for h in ("canonical", "single", "repeated"))
        blocks = "; ".join(h + ":" + json.dumps(ctx["blocks_per_sm"][h], sort_keys=True) for h in ("canonical", "single", "repeated"))
        rows.append([cfg, r["G"]["mean"], r["G"]["category"], r["D"]["mean"], r["D"]["category"],
                     json.dumps(ctx["preload_bytes_per_thread"], sort_keys=True), regs, blocks, r["cross_harness_residency_matched"]])
    text = ["# Phase 6 Stage A — Existing-evidence diagnostic", "",
            "31 cases recomputed from six historical raw invocations, with exact-rational OLS sums. "
            "All 39 Phase 5 frozen-model errors retained. No GPU, compiler execution, new timing or model refit.", "",
            c.table(["Case", "G", "G sign", "D", "D sign", "Preload B/thread d/4", "Registers d→4",
                     "Blocks/SM", "Across-harness residency matched"], rows), "",
            "Units: ns/additional CTA; not single-CTA latency. Five historical PRIMARY cases are a "
            "retrospective diagnostic family, never a newly confirmed n=5 cohort.", "",
            c.table(["Model / descriptive group", "n", "MAE", "RMSE", "Mean prediction−G"],
                    [[g, *[r[k] for k in ("n", "MAE", "RMSE", "mean_prediction_bias")]]
                     for g, r in sorted(result["descriptive_error_groups"].items())]), "",
            "Model C's fixed warp8 contribution is examined alongside all its other frozen terms. "
            "This failure audit identifies a context-transfer question, not a causal warp explanation.", "",
            "Next experiment: reuse all 30 archived binaries in canonical/single/repeated harnesses; "
            "measure unspecialized runtime R=0,1,2,4,8 in one repeated CUBIN per candidate. "
            "Check all resources and full family matches before timing. Freeze a fresh cohort before seeing any new outcomes.", ""]
    text += ["- " + limit for limit in result["limits"]]
    return "\n".join(text) + "\n"


def validate(result):
    expected = derive()
    c.require(result == expected, "Stage A must independently rederive from historical raw evidence")
    c.require(len(result["recomputed_cases"]) == 31 and len(result["all_39_prediction_errors"]) == 39,
              "Full historical population and error inventory")
    return {"status": "PASS", "raw_recomputed_cases": 31, "prediction_errors": 39,
            "protected_files": result["protected_files"]}


def main():
    result = derive()
    if "--validate" in sys.argv:
        loaded = c.read(DEST / "results.json")
        validation = validate(loaded)
        probes = []
        for label, mutate in (("metric", lambda r: r["diagnostic_family"][c.FAMILY[0]]["G"].update(mean=99)),
                              ("resource", lambda r: r["diagnostic_family"][c.FAMILY[0]]["context"]["resources"]["single"]["default"].update(num_regs=0)),
                              ("error", lambda r: r["all_39_prediction_errors"][0].update(absolute_error=0))):
            bad = copy.deepcopy(loaded)
            mutate(bad)
            try:
                validate(bad)
            except ValueError:
                probes.append(label)
            else:
                raise RuntimeError("Corruption accepted: " + label)
        validation["corruption_probes"] = probes
        c.require((DEST / "summary.md").read_text() == report(loaded), "Reproducible report")
        c.write(DEST / "validation.json", validation)
        print(validation)
    else:
        c.write(DEST / "results.json", result)
        (DEST / "summary.md").write_text(report(result))
        print("Derived all 31 cases and 39 errors without GPU execution")


if __name__ == "__main__":
    main()
