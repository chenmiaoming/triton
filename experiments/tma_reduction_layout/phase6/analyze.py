"""Phase 6 independent raw analysis. No cross-case model fitting."""
import copy
from fractions import Fraction
import math
import json
from pathlib import Path
import statistics as st
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.tma_reduction_layout.phase6 import common as c
from experiments.tma_reduction_layout.phase6 import timing_contract as tc


def decide(reference, alternative, n):
    if n < 5 or not all(math.isfinite(x) for v in (reference, alternative) for x in v.values()):
        return "INCONCLUSIVE_BY_COVERAGE" if n < 5 else "INCONCLUSIVE"
    if alternative["MAE"] < reference["MAE"] and alternative["RMSE"] < reference["RMSE"]:
        return "SUPPORTED"
    if alternative["MAE"] >= reference["MAE"] and alternative["RMSE"] >= reference["RMSE"]:
        return "FALSIFIED"
    return "INCONCLUSIVE"


def errors(rows):
    return {predictor: {"MAE": st.mean(abs(r[predictor]["mean"]-r["G"]["mean"]) for r in rows),
                       "RMSE": math.sqrt(st.mean((r[predictor]["mean"]-r["G"]["mean"])**2 for r in rows))}
            for predictor in ("S", "D1", "beta", "zero")} if rows else {}


def derive(stage):
    tc.validate_raw(stage)
    cases, binaries, plan = tc.inputs(stage)
    root = c.OUT / stage
    fits, statistics, metrics = {}, {}, {}
    for invocation in (1, 2, 3):
        raw = c.read(root / f"raw_invocation_{invocation}.json")
        statistics[str(invocation)], fits[str(invocation)] = c.raw_fits(raw)
        for cfg in cases:
            def slope(h, candidate, r):
                return fits[str(invocation)][f"{cfg}:{h}:{candidate}:R{r}"]["slope_ns_per_additional_CTA"]
            G = slope("canonical", "default", None)-slope("canonical", "4", None)
            S = slope("single", "default", None)-slope("single", "4", None)
            g = {r: slope("repeated", "default", r)-slope("repeated", "4", r) for r in c.R_VALUES}
            D = {r: g[r]-g[0] for r in c.R_VALUES}
            trend = c.ols([(r, g[r]) for r in c.R_VALUES])
            beta = trend["slope_ns_per_additional_CTA"]/1000
            value = {"G": G, "S": S, "g0": g[0], "D1": D[1], "beta": beta, "zero": 0.0,
                     "G_minus_S": G-S, "S_minus_D1": S-D[1], "G_minus_D1": G-D[1],
                     "repetition_fit": {"intercept_ns_per_CTA": trend["intercept_us"],
                        "beta_ns_per_CTA_per_repetition": beta, "R_squared": trend["R_squared"],
                        "residuals_ns_per_CTA": trend["residuals_us"]}}
            value.update({f"g{r}": g[r] for r in c.R_VALUES})
            value.update({f"D{r}": D[r] for r in c.R_VALUES})
            value.update({f"eta{r}": D[r]-r*D[1] for r in (2, 4, 8)})
            metrics.setdefault(cfg, []).append(value)
    rows = {}
    for cfg, case in cases.items():
        row = {**case, "invocation_metrics": metrics[cfg]}
        for name in metrics[cfg][0]:
            if name != "repetition_fit":
                row[name] = c.sign([r[name] for r in metrics[cfg]])
        row["prediction_errors"] = {p: {"prediction": row[p]["mean"],
                                      "error_prediction_minus_G": row[p]["mean"]-row["G"]["mean"],
                                      "absolute_error": abs(row[p]["mean"]-row["G"]["mean"]),
                                      "squared_error": (row[p]["mean"]-row["G"]["mean"])**2}
                                    for p in ("S", "D1", "beta", "zero")}
        row["resource_context"] = {key.split(":")[1]+":"+key.split(":")[2]: {
            "num_regs": binding["metadata"]["resources"]["num_regs"],
            "dynamic_smem_bytes": binding["metadata"]["dynamic_smem_bytes"],
            "blocks_per_sm": binding["occupancy"]["blocks_per_sm_actual_dynamic_smem"]}
            for key, binding in binaries.items() if key.split(":")[0] == cfg}
        row["cross_harness_residency_matched"] = len({v["blocks_per_sm"] for v in row["resource_context"].values()}) == 1
        rows[cfg] = row
    populations = {"ALL_ELIGIBLE": list(rows.values()),
                   "PRIMARY": [r for r in rows.values() if r["final_class"] == "PRIMARY"],
                   "SECONDARY": [r for r in rows.values() if r["final_class"] == "SECONDARY"]}
    comparison = {name: {"n": len(pop), "errors": errors(pop)} for name, pop in populations.items()}
    hypotheses = {}
    if stage == "stage_d":
        for name, result in comparison.items():
            if name == "SECONDARY":
                continue
            e = result["errors"]
            hypotheses[name] = {"H6_01_SINGLE_CONTEXT_TRANSFER": decide(e["D1"], e["S"], result["n"]) if e else "INCONCLUSIVE_BY_COVERAGE",
                                "H6_02_REPETITION_TREND_TRANSFER": decide(e["D1"], e["beta"], result["n"]) if e else "INCONCLUSIVE_BY_COVERAGE",
                                "n": result["n"], "case_level_population": name}
    result = {"stage": stage, "role": "RETROSPECTIVE_DIAGNOSTIC_SUPPLEMENT" if stage == "stage_c" else "PROSPECTIVE_UNMEASURED_WARP16_EXTENSION",
            "units": "ns/additional CTA; beta ns/(additional CTA * repetition); not single-CTA latency",
            "condition_statistics": statistics, "fits": fits, "cases": rows,
            "predictive_comparisons": comparison, "hypotheses": hypotheses,
            "raw_manifest_SHA256": c.sha((root / "raw_manifest.json").read_bytes()),
            "protocol_SHA256": c.sha((c.OUT / "stage_b/protocol.json").read_bytes()),
            "frozen_prior_status": c.read(c.BASE / "phase4/results/phase5_timing/results.json")["hypotheses"],
            "limits": ["No refit/calibration, no case removal, no tuned threshold, no outcome-based repeats.",
                       "Opcode matching and equal theoretical occupancy do not isolate registers/scheduling.",
                       "G-S, S-D1 and G-D1 remain cross-harness descriptive differences, not causal shares.",
                       "etaR measures departure from the R0/R1 extrapolation within one binary; unresolved is not practical equivalence.",
                       "Warp16 tests the specified extension domain only; n>=5 does not establish universal generalization.",
                       "H2b/H2c remain UNVERIFIED; no compiler or production heuristic change."]}
    return json.loads(c.encode(result))


def report(result):
    rows = result["cases"]
    def value(r, key):
        v = r[key]
        return f'{v["mean"]:.12g} ± {v["sample_SD"]:.12g} ({v["category"]})'
    lines = ["# Phase 6 " + result["stage"] + " — Exact-binary results", "", result["role"], "", result["units"], "",
        c.table(["Case", "Class", "G", "Single S", "D1", "beta", "G−S", "S−D1"],
                [[cfg, r["final_class"], *[value(r, k) for k in ("G", "S", "D1", "beta", "G_minus_S", "S_minus_D1")]]
                 for cfg, r in sorted(rows.items())]), "",
        c.table(["Case", "D2", "D4", "D8", "eta2", "eta4", "eta8"],
                [[cfg, *[value(r, k) for k in ("D2", "D4", "D8", "eta2", "eta4", "eta8")]]
                 for cfg, r in sorted(rows.items())]), "",
        c.table(["Population", "n", "Predictor", "MAE", "RMSE"],
                [[name, comp["n"], predictor, error["MAE"], error["RMSE"]]
                 for name, comp in sorted(result["predictive_comparisons"].items())
                 for predictor, error in sorted(comp["errors"].items())]), "",
        "All fits, intercepts, R²/residuals, invocation values, resource contexts and per-case prediction errors are retained in results.json.", ""]
    if result["hypotheses"]:
        lines += [c.table(["Scope", "n", "H6_01 single vs D1", "H6_02 beta vs D1"],
                         [[name, h["n"], h["H6_01_SINGLE_CONTEXT_TRANSFER"], h["H6_02_REPETITION_TREND_TRANSFER"]]
                          for name, h in sorted(result["hypotheses"].items())]), ""]
    lines += ["- " + limit for limit in result["limits"]]
    return "\n".join(lines) + "\n"


def validate(stored, stage):
    expected = derive(stage)
    c.require(stored == expected, "Every new analysis field must independently rederive from raw")
    # Independent median and OLS identities, not derived-to-derived comparison.
    for i in (1, 2, 3):
        raw = c.read(c.OUT / stage / f"raw_invocation_{i}.json")
        samples = {}
        for visit in raw["visits"]:
            samples.setdefault(visit["condition_tag"], []).extend(visit["samples_us"])
        for tag, values in samples.items():
            values = sorted(values)
            median = float((Fraction(values[49])+Fraction(values[50]))/2)
            c.require(median == stored["condition_statistics"][str(i)][tag]["median_us"], "Independent order-statistic median")
        for key, fit in stored["fits"][str(i)].items():
            x = list(map(Fraction, c.B_VALUES))
            y = [Fraction(stored["condition_statistics"][str(i)][key+f":B{b}"]["median_us"]) for b in c.B_VALUES]
            slope = (3*sum(a*b for a,b in zip(x,y))-sum(x)*sum(y))/(3*sum(a*a for a in x)-sum(x)**2)
            c.require(math.isclose(float(slope*1000), fit["slope_ns_per_additional_CTA"], abs_tol=1e-12), "Independent OLS sum identity")
    return expected


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "stage_c"
    root = c.OUT / stage
    if "--validate" in sys.argv:
        stored = c.read(root / "results.json")
        expected = validate(stored, stage)
        probes = []
        for name, mutate in (("slope", lambda r: next(iter(r["fits"]["1"].values())).update(slope_ns_per_additional_CTA=99)),
                             ("difference", lambda r: next(iter(r["cases"].values()))["D1"].update(mean=99)),
                             ("curve", lambda r: next(iter(r["cases"].values()))["eta8"].update(mean=99)),
                             ("case", lambda r: r["cases"].pop(next(iter(r["cases"])))),
                             ("prediction_error", lambda r: next(iter(r["cases"].values()))["prediction_errors"]["S"].update(absolute_error=0)),
                             ("metric", lambda r: r["predictive_comparisons"]["ALL_ELIGIBLE"]["errors"]["S"].update(MAE=99))):
            bad = copy.deepcopy(stored); mutate(bad)
            try:
                c.require(bad == expected, "Independent raw analysis closure")
            except ValueError:
                probes.append(name)
            else:
                raise RuntimeError("Corruption accepted: " + name)
        for label, ref, alt, n, outcome in (
            ("support", {"MAE": 2., "RMSE": 3.}, {"MAE": 1., "RMSE": 2.}, 5, "SUPPORTED"),
            ("falsify_equal", {"MAE": 2., "RMSE": 3.}, {"MAE": 2., "RMSE": 3.}, 5, "FALSIFIED"),
            ("mixed", {"MAE": 2., "RMSE": 3.}, {"MAE": 1., "RMSE": 4.}, 5, "INCONCLUSIVE"),
            ("coverage", {"MAE": 2., "RMSE": 3.}, {"MAE": 0., "RMSE": 0.}, 4, "INCONCLUSIVE_BY_COVERAGE")):
            c.require(decide(ref, alt, n) == outcome, "Prospective branch: " + label)
            probes.append(label)
        c.require((root / "summary.md").read_text() == report(stored), "Reproducible report")
        result = {"status": "PASS", "stage": stage, "raw_recomputed_cases": len(stored["cases"]),
                  "corruption_and_branch_probes": probes, "raw_bytes_unchanged": True}
        c.write(root / "analysis_validation.json", result); print(result)
    else:
        result = derive(stage)
        c.write(root / "results.json", result)
        (root / "summary.md").write_text(report(result))
        print("Independent analysis completed", stage)


if __name__ == "__main__":
    main()
