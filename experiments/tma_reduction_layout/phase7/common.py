"""Phase 7 utilities; all evidence through Phase 6 is immutable."""
from experiments.tma_reduction_layout.phase6 import common as prior

ROOT, BASE = prior.ROOT, prior.BASE
OUT = BASE / "results/phase7"
BASELINE = "cddc222f9968d254190782bff9836b9b4c814975"
PRIOR = BASE / "results/phase6"
sha, encode, read, write, require, table = prior.sha, prior.encode, prior.read, prior.write, prior.require, prior.table
ols, sign, raw_fits = prior.ols, prior.sign, prior.raw_fits
B_VALUES = prior.B_VALUES
CANDIDATES = ("default", "4")
VARIANTS = ("host_native", "host_canonical", "device_native", "device_canonical")
HARNESSES = ("canonical",) + VARIANTS


def inventory():
    return prior.inventory(BASELINE)


def diagnostic_cases():
    rows = read(PRIOR / "stage_d_gate/launch_contract.json")["cases"]
    return {key: value for key, value in rows.items() if value["final_class"] == "PRIMARY"}
