"""Fixed Phase8 compiler population and PTX ABI."""
from experiments.tma_reduction_layout.phase8 import common as c
from experiments.tma_reduction_layout.phase6.contracts import abi
DEST, GATE = c.OUT / "stage_a", c.OUT / "stage_b"
HARNESSES = ("canonical", "host_canonical", "device_canonical", "switch")


def population(pool):
    return [row for row in pool["cases"] if row["included_in_structural_pool"]]


def expected_abi(harness):
    if harness == "switch":
        return ["descriptor128", "u32", "u32", "u32", "u64", "u64", "u64",
                "u64", "u64", "u32", "u32", "u32", "u64", "u64"]
    from experiments.tma_reduction_layout.phase7.contracts import expected_abi as previous
    return previous(harness)
