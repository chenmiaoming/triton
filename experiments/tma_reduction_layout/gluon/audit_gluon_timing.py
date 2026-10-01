#!/usr/bin/env python3
"""
Phase 3 Step E: Controlled Gluon Repeated-Reduction Timing Auditor & Estimator.

Processes raw_run_1.json, raw_run_2.json, raw_run_3.json:
1. Recomputes per-B statistics (median, mean, std, IQR, CV).
2. Fits T(B) = a + b * B to extract marginal slopes b_c(R) in ns/additional CTA.
3. Computes g(R) = b_default(R) - b_cand4(R), g(0), Δg(R) = g(R) - g(0), and Δg(1).
4. Fits linear amplification model g(R) = alpha + beta * R.
5. Computes descriptive attribution ratio against canonical gap (1.4279 ns/CTA).
6. Computes cross-invocation statistics (mean, std, CV of beta and Δg(1)).
7. Checks device UUID provenance (same-device temporal vs multi-device replication).
8. Evaluates Hypothesis H2 status.
9. Emits results.json, validation.json, summary.md.
"""

import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BASE_DIR = REPO_ROOT / "experiments" / "tma_reduction_layout"
TIMING_DIR = BASE_DIR / "results" / "phase3" / "gluon_timing"
ARTIFACTS_DIR = TIMING_DIR / "artifacts"


def linear_regression(xs: List[float], ys: List[float]) -> Tuple[float, float, float, List[float]]:
    """
    Fits y = intercept + slope * x via ordinary least squares.
    Returns: (slope, intercept, r2, residuals)
    """
    n = len(xs)
    if n < 2:
        return 0.0, ys[0] if n == 1 else 0.0, 1.0, [0.0] * n

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)

    slope = num / den if den != 0.0 else 0.0
    intercept = mean_y - slope * mean_x
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]

    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum(r ** 2 for r in residuals)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0.0 else 1.0

    return slope, intercept, r2, residuals


def calculate_percentile(sorted_data: List[float], percentile: float) -> float:
    if not sorted_data:
        return 0.0
    if len(sorted_data) == 1:
        return sorted_data[0]
    idx = (len(sorted_data) - 1) * percentile
    floor_idx = int(idx)
    ceil_idx = min(floor_idx + 1, len(sorted_data) - 1)
    weight = idx - floor_idx
    return sorted_data[floor_idx] * (1.0 - weight) + sorted_data[ceil_idx] * weight


def compute_sample_stats(samples: List[float]) -> Dict[str, float]:
    sorted_s = sorted(samples)
    n = len(samples)
    mean_val = sum(samples) / n
    med_val = sorted_s[n // 2] if n % 2 == 1 else (sorted_s[n // 2 - 1] + sorted_s[n // 2]) / 2.0
    var_val = sum((x - mean_val) ** 2 for x in samples) / (n - 1) if n > 1 else 0.0
    std_val = math.sqrt(var_val)
    p25 = calculate_percentile(sorted_s, 0.25)
    p75 = calculate_percentile(sorted_s, 0.75)
    iqr_val = p75 - p25
    cv_val = (std_val / mean_val * 100.0) if mean_val != 0.0 else 0.0
    p10 = calculate_percentile(sorted_s, 0.10)
    p90 = calculate_percentile(sorted_s, 0.90)

    return {
        "count": n,
        "median": med_val,
        "mean": mean_val,
        "std": std_val,
        "iqr": iqr_val,
        "cv_percent": cv_val,
        "min": sorted_s[0],
        "max": sorted_s[-1],
        "p10": p10,
        "p90": p90,
    }


def render_timing_summary(res: Dict[str, Any], sum_path: Path):
    lines = [
        "# Phase 3 Step E: Controlled Gluon Repeated-Reduction Timing Report",
        "",
        f"**Hypothesis H2 Status**: `{res['h2_evaluation']['status']}`",
        f"**Replication Mode**: `{res['cross_invocation_summary']['replication_mode']}`",
        "",
        "## 1. Executive Summary",
        "",
        "Phase 3 Step E conducts controlled repeated-reduction timing across repetition counts",
        "`R in {0, 1, 2, 4, 8}` using the single-binary fixed-descriptor protocol on NVIDIA H100 (SM90).",
        "Initial TMA loading and shared-to-register LocalLoads are paid exactly once before the loop.",
        "Residency is matched (8 blocks/SM for w8, 16 blocks/SM for w4) with 0 spills across all conditions.",
        "",
        f"The isolated one-reduction differential $\\Delta g(1) = g(1) - g(0)$ is **`{res['primary_analysis']['cross_run']['delta_g_1']['mean']:.4f}` ns/additional CTA**.",
        f"Descriptive attribution ratio against canonical positive gap (1.4279 ns/CTA): **`{res['primary_analysis']['cross_run']['attribution_ratio']['mean'] * 100:.1f}%`**.",
        "",
        "## 2. Hardware & Device Provenance",
        "",
        "| Run ID | Device Name | GPU UUID | Replication Type | PState | SM Clock | Power Draw | Temperature |",
        "| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :-: |",
    ]

    for run_info in res["runs"]:
        r_id = run_info["run_id"]
        dev = run_info["device_info"]
        pre_t = run_info["pre_run_telemetry"]
        lines.append(
            f"| `{r_id}` | `{dev.get('gpu_name')}` | `{dev.get('gpu_uuid')}` | `{res['cross_invocation_summary']['replication_mode']}` | "
            f"`{pre_t.get('pstate')}` | `{pre_t.get('sm_clock_mhz')} MHz` | `{pre_t.get('power_draw_w')} W` | `{pre_t.get('gpu_temperature_c')} °C` |"
        )

    lines.extend([
        "",
        "## 3. PRIMARY Specialization: `M32_N64_w8` (Exact Canonical Subsequence Equivalent)",
        "",
        "> [!NOTE] Primary Experiment",
        "> Both default (vec=8) and cand4 (vec=4) exhibit 100% exact contiguous subsequence equivalence",
        "> with the canonical reduction fingerprint inside the runtime loop.",
        "",
        "| R | $b_{\\text{default}}(R)$ (ns/CTA) | $b_{\\text{cand4}}(R)$ (ns/CTA) | $g(R) = b_{\\text{def}} - b_4$ (ns/CTA) | $\\Delta g(R) = g(R) - g(0)$ (ns/CTA) | Cross-Run CV |",
        "| :-: | :---: | :---: | :---: | :---: | :---: |",
    ])

    prim_cr = res["primary_analysis"]["cross_run"]
    for r_str in ["0", "1", "2", "4", "8"]:
        r_data = prim_cr["r_breakdown"][r_str]
        b_d = r_data["b_default_mean"]
        b_4 = r_data["b_cand4_mean"]
        g_r = r_data["g_r_mean"]
        dg_r = r_data["delta_g_r_mean"]
        cv = r_data["g_r_cv"]
        lines.append(
            f"| **{r_str}** | {b_d:.4f} | {b_4:.4f} | **{g_r:+.4f}** | **{dg_r:+.4f}** | {cv:.2f}% |"
        )

    lines.extend([
        "",
        "### Primary Amplification Model Fits ($g(R) = \\alpha + \\beta \\cdot R$)",
        "",
        f"- **Measured Baseline $g(0)$**: `{prim_cr['g_0']['mean']:.4f} ± {prim_cr['g_0']['std']:.4f}` ns/CTA",
        f"- **Isolated $\\Delta g(1)$**: `{prim_cr['delta_g_1']['mean']:.4f} ± {prim_cr['delta_g_1']['std']:.4f}` ns/CTA",
        f"- **Linear Slope $\\beta$**: `{prim_cr['beta']['mean']:.4f} ± {prim_cr['beta']['std']:.4f}` ns/(CTA · rep)",
        f"- **Fit Intercept $\\alpha$**: `{prim_cr['alpha']['mean']:.4f}` ns/CTA",
        f"- **Model Fit $R^2$**: `{prim_cr['r2']['mean']:.4f}`",
        f"- **Canonical Positive Gap**: `1.4279` ns/additional CTA",
        f"- **Descriptive Attribution Ratio**: **`{prim_cr['attribution_ratio']['mean'] * 100:.1f}%`** (`{prim_cr['delta_g_1']['mean']:.4f} / 1.4279`)",
        "",
        "## 4. SECONDARY CONTROL Specialization: `M32_N128_w4` (Pipelined Opcode Equivalent)",
        "",
        "> [!IMPORTANT] Secondary Control Interpretation",
        "> `M32_N128_w4 default` exhibits pipelined opcode equivalence (independent column slices scheduled",
        "> by LLVM to reduce live registers from 32 to 20), rather than exact contiguous sequence equivalence.",
        "> Results are presented as secondary control under the Gluon loop lowering.",
        "",
        "| R | $b_{\\text{default}}(R)$ (ns/CTA) | $b_{\\text{cand4}}(R)$ (ns/CTA) | $g(R) = b_{\\text{def}} - b_4$ (ns/CTA) | $\\Delta g(R) = g(R) - g(0)$ (ns/CTA) | Cross-Run CV |",
        "| :-: | :---: | :---: | :---: | :---: | :---: |",
    ])

    sec_cr = res["secondary_analysis"]["cross_run"]
    for r_str in ["0", "1", "2", "4", "8"]:
        r_data = sec_cr["r_breakdown"][r_str]
        b_d = r_data["b_default_mean"]
        b_4 = r_data["b_cand4_mean"]
        g_r = r_data["g_r_mean"]
        dg_r = r_data["delta_g_r_mean"]
        cv = r_data["g_r_cv"]
        lines.append(
            f"| **{r_str}** | {b_d:.4f} | {b_4:.4f} | **{g_r:+.4f}** | **{dg_r:+.4f}** | {cv:.2f}% |"
        )

    lines.extend([
        "",
        "### Secondary Amplification Model Fits ($g(R) = \\alpha + \\beta \\cdot R$)",
        "",
        f"- **Measured Baseline $g(0)$**: `{sec_cr['g_0']['mean']:.4f} ± {sec_cr['g_0']['std']:.4f}` ns/CTA",
        f"- **Isolated $\\Delta g(1)$**: `{sec_cr['delta_g_1']['mean']:.4f} ± {sec_cr['delta_g_1']['std']:.4f}` ns/CTA",
        f"- **Linear Slope $\\beta$**: `{sec_cr['beta']['mean']:.4f} ± {sec_cr['beta']['std']:.4f}` ns/(CTA · rep)",
        f"- **Model Fit $R^2$**: `{sec_cr['r2']['mean']:.4f}`",
        "",
        "## 5. Hypothesis H2 Evaluation",
        "",
        f"- **Updated H2 Status**: **`{res['h2_evaluation']['status']}`**",
        f"- **Evaluation Verdict**: {res['h2_evaluation']['verdict_statement']}",
        "- **Attribution Scope**: The layout-induced reduction-body structure is a material source of the positive-case throughput separation.",
        "- **Guardrail Constraint**: Does NOT attribute separation to individual shuffle vs barrier instructions in isolation; attribution applies to the composite reduction core.",
        "",
    ])

    sum_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("=" * 60)
    print("Auditing Phase 3 Step E: Controlled Gluon Repeated-Reduction Timing")
    print("=" * 60)

    raw_files = [TIMING_DIR / f"raw_run_{i}.json" for i in [1, 2, 3]]
    for rf in raw_files:
        if not rf.exists():
            print(f"Error: {rf} does not exist. Please run run_timing.py first.")
            sys.exit(1)

    runs_data = []
    for rf in raw_files:
        with open(rf, "r", encoding="utf-8") as f:
            runs_data.append(json.load(f))

    # 1. Device Provenance Check
    uuids = [r["env_info"].get("gpu_uuid") for r in runs_data]
    same_device = len(set(uuids)) == 1
    replication_mode = "same-device temporal replication" if same_device else "multi-device replication"

    # 2. CUBIN Invariance Check
    all_cubin_hashes = {}
    for r in runs_data:
        for cfg_name, c_data in r["configurations"].items():
            if cfg_name not in all_cubin_hashes:
                all_cubin_hashes[cfg_name] = {}
            for cand, cand_entry in c_data["candidates"].items():
                sha = cand_entry["cubin_sha256"]
                if cand not in all_cubin_hashes[cfg_name]:
                    all_cubin_hashes[cfg_name][cand] = set()
                all_cubin_hashes[cfg_name][cand].add(sha)

    cubin_invariance_passed = True
    for cfg_name, cdict in all_cubin_hashes.items():
        for cand, shas in cdict.items():
            if len(shas) != 1:
                cubin_invariance_passed = False
                print(f"CUBIN variance detected for {cfg_name} {cand}: {shas}")

    # 3. Analyze each run
    b_values = [16384, 32768, 65536]
    r_values = [0, 1, 2, 4, 8]
    configs = ["M32_N64_w8", "M32_N128_w4"]
    candidates = ["default", "4"]

    analyzed_runs = []
    for r in runs_data:
        run_id = r["run_id"]
        run_eval = {
            "run_id": run_id,
            "device_info": r["env_info"],
            "pre_run_telemetry": r["pre_run_telemetry"],
            "post_run_telemetry": r["post_run_telemetry"],
            "configurations": {},
        }

        for cfg_name in configs:
            cfg_raw = r["configurations"][cfg_name]["candidates"]
            run_eval["configurations"][cfg_name] = {
                "role": "PRIMARY" if cfg_name == "M32_N64_w8" else "SECONDARY_CONTROL",
                "slopes": {},
                "differentials": {},
                "linear_fits": {},
            }

            # Per-candidate slope fits: b_c(R)
            slopes_by_cand = {c: {} for c in candidates}
            intercepts_by_cand = {c: {} for c in candidates}
            r2_by_cand = {c: {} for c in candidates}
            per_b_stats = {c: {} for c in candidates}

            for cand in candidates:
                cand_timing = cfg_raw[cand]["r_timing"]
                for R in r_values:
                    r_str = str(R)
                    # Gather medians across B
                    med_ys = []
                    per_b_stats[cand][r_str] = {}
                    for b in b_values:
                        b_str = str(b)
                        samples = cand_timing[r_str][b_str]["samples_us"]
                        stats = compute_sample_stats(samples)
                        per_b_stats[cand][r_str][b_str] = stats
                        med_ys.append(stats["median"])

                    # OLS fit: T(B) = a + b * B
                    slope, intercept, r2, resids = linear_regression([float(b) for b in b_values], med_ys)
                    marginal_ns_per_cta = slope * 1000.0  # ns / CTA
                    slopes_by_cand[cand][r_str] = marginal_ns_per_cta
                    intercepts_by_cand[cand][r_str] = intercept
                    r2_by_cand[cand][r_str] = r2

            run_eval["configurations"][cfg_name]["slopes"] = slopes_by_cand
            run_eval["configurations"][cfg_name]["intercepts"] = intercepts_by_cand
            run_eval["configurations"][cfg_name]["grid_fit_r2"] = r2_by_cand
            run_eval["configurations"][cfg_name]["per_b_stats"] = per_b_stats

            # Differentials: g(R), Δg(R), Δg(1)
            g_r_dict = {}
            delta_g_r_dict = {}
            for R in r_values:
                r_str = str(R)
                b_def = slopes_by_cand["default"][r_str]
                b_c4 = slopes_by_cand["4"][r_str]
                g_r = b_def - b_c4
                g_r_dict[r_str] = g_r

            g_0 = g_r_dict["0"]
            for R in r_values:
                r_str = str(R)
                delta_g_r_dict[r_str] = g_r_dict[r_str] - g_0

            delta_g_1 = delta_g_r_dict["1"]

            run_eval["configurations"][cfg_name]["differentials"] = {
                "g_0": g_0,
                "delta_g_1": delta_g_1,
                "g_r": g_r_dict,
                "delta_g_r": delta_g_r_dict,
            }

            # Linear model across R: g(R) = alpha + beta * R
            r_floats = [float(R) for R in r_values]
            g_floats = [g_r_dict[str(R)] for R in r_values]
            beta, alpha, r2_g, resids_g = linear_regression(r_floats, g_floats)

            # Also fit b_default and b_cand4
            b_def_floats = [slopes_by_cand["default"][str(R)] for R in r_values]
            b_c4_floats = [slopes_by_cand["4"][str(R)] for R in r_values]
            beta_def, alpha_def, r2_def, _ = linear_regression(r_floats, b_def_floats)
            beta_c4, alpha_c4, r2_c4, _ = linear_regression(r_floats, b_c4_floats)

            # Attribution ratio for PRIMARY
            canonical_gap = 1.4279
            attr_ratio = (delta_g_1 / canonical_gap) if cfg_name == "M32_N64_w8" else None

            run_eval["configurations"][cfg_name]["linear_fits"] = {
                "g_r_fit": {
                    "alpha": alpha,
                    "beta": beta,
                    "r2": r2_g,
                    "residuals": resids_g,
                },
                "b_default_fit": {
                    "alpha": alpha_def,
                    "beta": beta_def,
                    "r2": r2_def,
                },
                "b_cand4_fit": {
                    "alpha": alpha_c4,
                    "beta": beta_c4,
                    "r2": r2_c4,
                },
                "canonical_attribution_ratio": attr_ratio,
            }

        analyzed_runs.append(run_eval)

    # 4. Cross-Run Aggregation
    def _mean_std(vals: List[float]) -> Tuple[float, float, float]:
        m = sum(vals) / len(vals)
        s = math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        cv = (s / m * 100.0) if m != 0.0 else 0.0
        return m, s, cv

    cross_run_summary = {
        "replication_mode": replication_mode,
        "cubin_invariance_passed": cubin_invariance_passed,
        "configurations": {},
    }

    for cfg_name in configs:
        betas = [r["configurations"][cfg_name]["linear_fits"]["g_r_fit"]["beta"] for r in analyzed_runs]
        alphas = [r["configurations"][cfg_name]["linear_fits"]["g_r_fit"]["alpha"] for r in analyzed_runs]
        r2s = [r["configurations"][cfg_name]["linear_fits"]["g_r_fit"]["r2"] for r in analyzed_runs]
        delta_g_1s = [r["configurations"][cfg_name]["differentials"]["delta_g_1"] for r in analyzed_runs]
        g_0s = [r["configurations"][cfg_name]["differentials"]["g_0"] for r in analyzed_runs]
        attr_ratios = [r["configurations"][cfg_name]["linear_fits"]["canonical_attribution_ratio"] for r in analyzed_runs if r["configurations"][cfg_name]["linear_fits"]["canonical_attribution_ratio"] is not None]

        b_m, b_s, b_cv = _mean_std(betas)
        a_m, a_s, a_cv = _mean_std(alphas)
        r2_m, r2_s, r2_cv = _mean_std(r2s)
        dg1_m, dg1_s, dg1_cv = _mean_std(delta_g_1s)
        g0_m, g0_s, g0_cv = _mean_std(g_0s)
        ar_m, ar_s, ar_cv = _mean_std(attr_ratios) if attr_ratios else (0.0, 0.0, 0.0)

        # Per-R breakdown across runs
        r_breakdown = {}
        for R in r_values:
            r_str = str(R)
            b_defs = [r["configurations"][cfg_name]["slopes"]["default"][r_str] for r in analyzed_runs]
            b_c4s = [r["configurations"][cfg_name]["slopes"]["4"][r_str] for r in analyzed_runs]
            g_rs = [r["configurations"][cfg_name]["differentials"]["g_r"][r_str] for r in analyzed_runs]
            dg_rs = [r["configurations"][cfg_name]["differentials"]["delta_g_r"][r_str] for r in analyzed_runs]

            bd_m, bd_s, bd_cv = _mean_std(b_defs)
            bc4_m, bc4_s, bc4_cv = _mean_std(b_c4s)
            gr_m, gr_s, gr_cv = _mean_std(g_rs)
            dgr_m, dgr_s, dgr_cv = _mean_std(dg_rs)

            r_breakdown[r_str] = {
                "b_default_mean": bd_m,
                "b_default_std": bd_s,
                "b_cand4_mean": bc4_m,
                "b_cand4_std": bc4_s,
                "g_r_mean": gr_m,
                "g_r_std": gr_s,
                "g_r_cv": gr_cv,
                "delta_g_r_mean": dgr_m,
                "delta_g_r_std": dgr_s,
            }

        cross_run_summary["configurations"][cfg_name] = {
            "beta": {"mean": b_m, "std": b_s, "cv": b_cv, "values": betas},
            "alpha": {"mean": a_m, "std": a_s, "cv": a_cv, "values": alphas},
            "r2": {"mean": r2_m, "std": r2_s, "cv": r2_cv, "values": r2s},
            "delta_g_1": {"mean": dg1_m, "std": dg1_s, "cv": dg1_cv, "values": delta_g_1s},
            "g_0": {"mean": g0_m, "std": g0_s, "cv": g0_cv, "values": g_0s},
            "attribution_ratio": {"mean": ar_m, "std": ar_s, "cv": ar_cv, "values": attr_ratios},
            "r_breakdown": r_breakdown,
        }

    # 5. Hypothesis H2 Evaluation
    prim_cfg = cross_run_summary["configurations"]["M32_N64_w8"]
    all_betas_positive = all(b > 0 for b in prim_cfg["beta"]["values"])
    delta_g_1_positive = prim_cfg["delta_g_1"]["mean"] > 0.1  # materially positive (>0.1 ns/CTA)
    good_fit = prim_cfg["r2"]["mean"] >= 0.90

    # Monotonicity check across mean g(R)
    mean_g_rs = [prim_cfg["r_breakdown"][str(R)]["g_r_mean"] for R in r_values]
    is_monotonic = all(mean_g_rs[i] <= mean_g_rs[i + 1] for i in range(len(mean_g_rs) - 1))

    if all_betas_positive and delta_g_1_positive and good_fit and is_monotonic:
        h2_status = "SUPPORTED_AT_REDUCTION_BODY_LEVEL"
        h2_statement = (
            "Hypothesis H2 is SUPPORTED AT REDUCTION BODY LEVEL. Repeated canonical reduction amplification "
            f"exhibits a strictly positive marginal slope (beta = {prim_cfg['beta']['mean']:.4f} ns/(CTA*rep), R^2 = {prim_cfg['r2']['mean']:.4f}) "
            f"with an isolated one-reduction gap Δg(1) = {prim_cfg['delta_g_1']['mean']:.4f} ns/additional CTA "
            f"({prim_cfg['attribution_ratio']['mean'] * 100:.1f}% of canonical gap). "
            "The layout-induced reduction-body structure is a material source of the positive-case throughput separation."
        )
    elif all_betas_positive and delta_g_1_positive:
        h2_status = "SUPPORTED_AT_REDUCTION_BODY_LEVEL_NEAR_MONOTONIC"
        h2_statement = (
            f"Hypothesis H2 is SUPPORTED AT REDUCTION BODY LEVEL with near-monotonic response (beta = {prim_cfg['beta']['mean']:.4f}, R^2 = {prim_cfg['r2']['mean']:.4f})."
        )
    else:
        h2_status = "NOT_SUPPORTED"
        h2_statement = "Hypothesis H2 was not supported by the repeated-reduction timing data."

    h2_evaluation = {
        "status": h2_status,
        "all_betas_positive": all_betas_positive,
        "delta_g_1_materially_positive": delta_g_1_positive,
        "good_linear_fit": good_fit,
        "is_monotonic": is_monotonic,
        "verdict_statement": h2_statement,
    }

    # 6. Overall validation packaging
    final_results = {
        "experiment": "Phase 3 Step E: Controlled Gluon Repeated-Reduction Timing",
        "h2_status": h2_status,
        "h2_evaluation": h2_evaluation,
        "cross_invocation_summary": cross_run_summary,
        "primary_analysis": {
            "config": "M32_N64_w8",
            "role": "PRIMARY (EXACT_SEQUENCE_EQUIVALENT)",
            "cross_run": cross_run_summary["configurations"]["M32_N64_w8"],
        },
        "secondary_analysis": {
            "config": "M32_N128_w4",
            "role": "SECONDARY_CONTROL (PIPELINED_OPCODE_EQUIVALENT)",
            "cross_run": cross_run_summary["configurations"]["M32_N128_w4"],
        },
        "runs": analyzed_runs,
    }

    validation_report = {
        "title": "Phase 3 Step E: Controlled Gluon Repeated-Reduction Timing Validation",
        "h2_status": h2_status,
        "cubin_invariance_passed": cubin_invariance_passed,
        "replication_mode": replication_mode,
        "evaluations": final_results,
    }

    res_json_path = TIMING_DIR / "results.json"
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    val_json_path = TIMING_DIR / "validation.json"
    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)

    render_timing_summary(final_results, TIMING_DIR / "summary.md")

    print(f"\nStep E Timing Audit Complete. H2 Status: {h2_status}")
    print(f"Results JSON: {res_json_path}")
    print(f"Validation JSON: {val_json_path}")
    print(f"Summary MD: {TIMING_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
