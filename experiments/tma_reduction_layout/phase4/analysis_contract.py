"""Frozen descriptive analysis semantics; offline synthetic tests only in Stage A."""
import math
import statistics

T_DF2_975 = 4.302652729911275


def finite_values(values):
    values = list(values)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Finite continuous numerical values required")
    return values


def average_ranks(values):
    """One-based average ranks for exact ties; no rounding or sign binning."""
    values = finite_values(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(x, y):
    """Pearson of continuous-value average ranks. None means UNDEFINED.

    Preserve the existing <3-pair/constant-variable rule. No p-values or IID
    random-sample inference; these are structured factorial cohort descriptions.
    """
    x, y = finite_values(x), finite_values(y)
    if len(x) != len(y):
        raise ValueError("Paired lengths must agree")
    if len(x) < 3:
        return None
    rx, ry = average_ranks(x), average_ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    dx, dy = [v - mx for v in rx], [v - my for v in ry]
    vx, vy = sum(v * v for v in dx), sum(v * v for v in dy)
    if vx == 0 or vy == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(vx * vy)


def leave_one_out_spearman(cases):
    """Each PRIMARY case maps to (continuous mean delta_g1, mean G_canonical).

    Report every omission for sensitivity, never delete it from the full cohort.
    Summaries cover defined rhos; retain undefined omissions and their counts.
    """
    names = sorted(cases)
    if any(len(cases[name]) != 2 for name in names):
        raise ValueError("Each case must supply exactly two continuous means")
    for name in names:
        finite_values(cases[name])
    rhos = {}
    for excluded in names:
        remaining = [cases[name] for name in names if name != excluded]
        rhos[excluded] = spearman([pair[0] for pair in remaining], [pair[1] for pair in remaining])
    defined = [value for value in rhos.values() if value is not None]
    return {
        "rho_minus_i": rhos,
        "min": min(defined) if defined else None,
        "max": max(defined) if defined else None,
        "median": statistics.median(defined) if defined else None,
        "defined_count": len(defined),
        "undefined_count": len(rhos) - len(defined),
    }


def sign_resolution(values):
    """Two-sided 95% Student-t sign-resolution band for three temporal pairs."""
    values = finite_values(values)
    if len(values) != 3:
        raise ValueError("Exactly three paired temporal differentials required")
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half_width = T_DF2_975 * sd / math.sqrt(3)
    finite_values([mean, sd, half_width, mean - half_width, mean + half_width])
    label = "POSITIVE" if mean > half_width else "NEGATIVE" if mean < -half_width else "SIGN_UNRESOLVED"
    return {"mean": mean, "sample_SD": sd, "half_width": half_width,
            "interval": [mean - half_width, mean + half_width], "category": label}
