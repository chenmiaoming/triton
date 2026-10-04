"""Frozen fresh structural population; no timing/model inputs to compilation."""
from experiments.tma_reduction_layout.phase6 import common as c
from experiments.tma_reduction_layout.phase6 import contracts

OUT = c.OUT / "stage_d_gate"
PREREG = contracts.DEST
HARNESSES, CANDIDATES = contracts.HARNESSES, contracts.CANDIDATES
FROZEN = ("structural_pool.json", "protocol.json")
sha, encode, read, require = c.sha, c.encode, c.read, c.require


def population(pool):
    # Explicitly expose only structural whitelist; no coefficients/outcomes.
    require(pool["source_domain"] == contracts.FRESH_DOMAIN and pool["total_source_transitions"] == 32,
            "Entire frozen warp16 Cartesian domain")
    rows = []
    for original in pool["transitions"]:
        if not original["included_in_structural_pool"]:
            continue
        row = {k: original[k] for k in ("config_id", "M", "N", "num_warps", "logical_shape", "reduction_axis", "origin")}
        for name in ("default", "cand4"):
            row[name] = {k: original[name][k] for k in ("legal", "layout", "compiled_num_warps", "num_ctas",
                "hardware_warp_size", "shared_family", "shared_rank", "lanePart_M", "warpPart_M", "vec")}
        rows.append(row)
    require(len(rows) == 18 and len({r["config_id"] for r in rows}) == 18, "Frozen 18 included / 14 excluded")
    return sorted(rows, key=lambda r: r["config_id"])


def schedule(cases):
    """Stable eligibility filter of the preregistered full source-domain order."""
    pool = read(PREREG / "fresh_structural_pool.json")
    included = {r["config_id"] for r in population(pool)}
    require(set(cases) <= included and cases, "Nonempty structurally admitted schedule")
    master = contracts.schedule([r["config_id"] for r in pool["transitions"]])
    for invocation in master["invocations"]:
        for round_ in invocation["rounds"]:
            round_["order"] = [tag for tag in round_["order"] if tag.split(":")[0] in cases]
    count = len(master["invocations"][0]["rounds"][0]["order"])
    master.update(conditions_per_invocation=count, scalar_samples=count*3*100)
    return master
