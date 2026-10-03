"""Offline Stage D arithmetic and immutable Stage A-C lineage (stdlib only)."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "experiments/tma_reduction_layout"
BASELINE = "11310ee3b6c92e188c66999a5dacd1ece87d7079"
OUT = BASE / "phase4/results/residual_analysis"
TIMING = BASE / "phase4/results/timing"
GATE = BASE / "phase4/results/artifact_gate"
MODEL_TERMS = {"A": ["intercept", "D"], "B": ["intercept", "D", "g0"],
               "C": ["intercept", "D", "g0", "warp8"]}
FAMILIES = ["max.*", "cvt.*", "shfl.*", "ld.shared*", "st.shared*", "ldmatrix*", "bar.sync", "selp.*"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def frozen_inventory():
    """Check every pre-existing experiment file against the trusted Git blob."""
    tree = subprocess.check_output(["git", "ls-tree", "-rz", BASELINE, "--", "experiments/tma_reduction_layout"], cwd=ROOT)
    result = {}
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        meta, name = entry.split(b"\t", 1)
        path = name.decode()
        blob = (ROOT / path).read_bytes()
        git_sha = hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
        require(git_sha == meta.decode().split()[2], "Frozen Stage A-C/earlier bytes changed: " + path)
        result[path] = sha(blob)
    return result


def summary(values):
    return {"invocation_values": values, "mean": st.mean(values), "sample_SD": st.stdev(values)}


def sign(values):
    mean, sd = st.mean(values), st.stdev(values)
    half_width = 4.302652729911275 * sd / math.sqrt(3)
    return "POSITIVE" if mean > half_width else "NEGATIVE" if mean < -half_width else "SIGN_UNRESOLVED"


def slope(points):
    mx = st.mean(x for x, _ in points)
    my = st.mean(y for _, y in points)
    return 1000 * sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x, _ in points)


def timing_metrics(raws):
    """Recompute paired marginal-slope differences from the ordered raw samples."""
    values = {}
    for raw in raws:
        samples = {}
        for visit in raw["visits"]:
            samples.setdefault(visit["condition_tag"], []).extend(visit["samples_us"])
        require(all(len(v) == 100 for v in samples.values()), "100 samples per timing condition")
        names = sorted({tag.split(":")[0] for tag in samples})
        for cfg in names:
            b = {}
            for h in ("canonical", "repeated"):
                for c in ("default", "4"):
                    for r in ((None,) if h == "canonical" else (0, 1)):
                        points = [(size, st.median(samples[f"{cfg}:{h}:{c}:R{r}:B{size}"])) for size in (16384, 32768, 65536)]
                        b[h, c, r] = slope(points)
            G = b["canonical", "default", None] - b["canonical", "4", None]
            g0 = b["repeated", "default", 0] - b["repeated", "4", 0]
            g1 = b["repeated", "default", 1] - b["repeated", "4", 1]
            D = g1 - g0
            row = values.setdefault(cfg, {k: [] for k in ("G", "g0", "g1", "D", "residual")})
            for key, v in zip(row, (G, g0, g1, D, G - D)):
                row[key].append(v)
    return {cfg: {key: summary(v) for key, v in row.items()} for cfg, row in values.items()}


def ranks(values):
    ordered = sorted(set(values))
    rank = {v: 1 + sum(x < v for x in values) + (values.count(v) - 1) / 2 for v in ordered}
    return [rank[v] for v in values]


def pearson(x, y):
    if len(x) < 3:
        return None
    mx, my = st.mean(x), st.mean(y)
    xx, yy = [v - mx for v in x], [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in xx) * sum(v * v for v in yy))
    return sum(a * b for a, b in zip(xx, yy)) / denom if denom else None


def correlations(rows):
    G = [r["metrics"]["G"]["mean"] for r in rows]
    scores = {"D": [r["metrics"]["D"]["mean"] for r in rows],
              "g0": [r["metrics"]["g0"]["mean"] for r in rows],
              "D_plus_g0": [r["metrics"]["D"]["mean"] + r["metrics"]["g0"]["mean"] for r in rows]}
    return {k: {"Pearson": pearson(G, v), "Spearman": pearson(ranks(G), ranks(v))} for k, v in scores.items()}


def design(row, terms):
    v = {"intercept": 1., "D": row["metrics"]["D"]["mean"],
         "g0": row["metrics"]["g0"]["mean"], "warp8": float(row["features"]["num_warps"] == 8)}
    return [v[t] for t in terms]


def qr_fit(X, y):
    """Scaled, reorthogonalized QR; rank tolerance is numerical, not scientific."""
    n, p = len(X), len(X[0])
    if n < p:
        return None
    scales = [math.sqrt(sum(x[j] ** 2 for x in X)) for j in range(p)]
    if not all(scales):
        return None
    Q, R = [], [[0.] * p for _ in range(p)]
    for j in range(p):
        v = [x[j] / scales[j] for x in X]
        for _ in range(2):
            for i, q in enumerate(Q):
                value = sum(a * b for a, b in zip(q, v))
                R[i][j] += value
                v = [a - value * b for a, b in zip(v, q)]
        length = math.sqrt(sum(a * a for a in v))
        if length <= 1e-12:
            return None
        R[j][j] = length
        Q.append([a / length for a in v])
    rhs = [sum(a * b for a, b in zip(q, y)) for q in Q]
    beta = [0.] * p
    for j in reversed(range(p)):
        beta[j] = (rhs[j] - sum(R[j][k] * beta[k] for k in range(j + 1, p))) / R[j][j]
    return [v / s for v, s in zip(beta, scales)]


def model(rows, terms):
    rows = sorted(rows, key=lambda r: r["case_id"])
    X, y = [design(r, terms) for r in rows], [r["metrics"]["G"]["mean"] for r in rows]
    beta = qr_fit(X, y)
    predicted = [sum(a * b for a, b in zip(x, beta)) for x in X] if beta is not None else None
    total = sum((v - st.mean(y)) ** 2 for v in y)
    folds = []
    for i, row in enumerate(rows):
        b = qr_fit(X[:i] + X[i + 1:], y[:i] + y[i + 1:])
        p = sum(a * value for a, value in zip(X[i], b)) if b is not None else None
        folds.append({"held_out_case": row["case_id"], "training_case_ids": [r["case_id"] for r in rows if r is not row],
            "coefficients": dict(zip(terms, b)) if b is not None else None,
            "observed_G": y[i], "predicted_G": p, "error_predicted_minus_observed": p - y[i] if p is not None else None,
            "status": "DEFINED" if p is not None else "UNDEFINED_SINGULAR_OR_INSUFFICIENT_OBSERVATIONS"})
    errors = [f["error_predicted_minus_observed"] for f in folds]
    complete = all(v is not None for v in errors)
    return {"n": len(rows), "terms": terms, "coefficients": dict(zip(terms, beta)) if beta is not None else None,
        "fit_status": "DEFINED" if beta is not None else "UNDEFINED_SINGULAR_OR_INSUFFICIENT_OBSERVATIONS",
        "in_sample_predictions": dict(zip([r["case_id"] for r in rows], predicted)) if predicted is not None else None,
        "R_squared": 1 - sum((a - b) ** 2 for a, b in zip(y, predicted)) / total if predicted is not None and total else None,
        "LOOCV": {"folds": folds, "MAE": st.mean(abs(e) for e in errors) if complete else None,
            "RMSE": math.sqrt(st.mean(e * e for e in errors)) if complete else None,
            "defined_folds": sum(e is not None for e in errors), "undefined_folds": sum(e is None for e in errors)}}


def agreement(rows):
    resolved = [r for r in rows if r["G_category"] != "SIGN_UNRESOLVED" and r["D_category"] != "SIGN_UNRESOLVED"]
    matches = sum(r["G_category"] == r["D_category"] for r in resolved)
    return {"numerator": matches, "denominator": len(resolved), "rate": matches / len(resolved) if resolved else None,
            "unresolved_pairs": len(rows) - len(resolved)}


def group_summary(rows):
    return {"n": len(rows), "small_n": len(rows) < 3,
        "mean_G": st.mean(r["metrics"]["G"]["mean"] for r in rows),
        "mean_D": st.mean(r["metrics"]["D"]["mean"] for r in rows),
        "mean_descriptive_residual": st.mean(r["metrics"]["residual"]["mean"] for r in rows),
        "resolved_sign_agreement": agreement(rows), "case_ids": sorted(r["case_id"] for r in rows)}


def derive(rows):
    rows = sorted(rows, key=lambda r: r["case_id"])
    strata = {"ALL_ELIGIBLE": rows, **{name: [r for r in rows if r["class"] == name] for name in ("PRIMARY", "SECONDARY")}}
    models = {s: {name: model(rr, terms) for name, terms in MODEL_TERMS.items()} for s, rr in strata.items()}
    groups = {}
    for field in ("num_warps", "M", "N", "lane_transition", "equivalence_group", "preload_group"):
        labels = sorted({str(r["features"][field]) for r in rows})
        groups[field] = {label: group_summary([r for r in rows if str(r["features"][field]) == label]) for label in labels}
    return {"analysis_status": "OUTCOME_INFORMED_EXPLORATORY_ANALYSIS", "units": "ns/additional CTA",
        "residual_interpretation": "G-D is a cross-harness descriptive residual, NOT an additive causal remainder.",
        "g0_interpretation": "fixed-harness differential; may include pre-loop LocalLoad, fixed work and compiler/register/scheduling interactions",
        "score_interpretation": "D+g0 is a descriptive constructed score; no causal decomposition",
        "cases": {r["case_id"]: r for r in rows}, "models": models, "groups": groups,
        "correlations": {s: correlations(rr) for s, rr in strata.items()},
        "model_policy": "Only A/B/C, unweighted OLS on case means; all 18 cases retained; PRIMARY/SECONDARY separate; singular fit/fold UNDEFINED, never partially aggregate failed LOOCV; no p-values, significance, winner threshold or causal inference",
        "taxonomy_distribution": dict(sorted(Counter(r["taxonomy"] for r in rows).items()))}
