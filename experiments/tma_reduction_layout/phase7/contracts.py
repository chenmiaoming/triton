"""Pre-timing 2x2 intervention and independent-cohort contracts."""
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.tma_reduction_layout.phase7 import common as c
from experiments.tma_reduction_layout.phase6.contracts import abi
from experiments.tma_reduction_layout.phase4 import preregister as p4
from experiments.tma_reduction_layout.phase4.phase5.preregister import predicted_candidate

DEST = c.OUT / "stage_b_prereg"
GATE = c.OUT / "stage_b_artifacts"
FRESH_DOMAIN = {"M": [1024, 2048], "N": [16, 32, 64, 128], "num_warps": [4, 8, 16]}
HARNESSES, CANDIDATES = c.HARNESSES, c.CANDIDATES


def old_ids():
    ids = set(c.read(c.BASE / "results/phase2/sweep/results.json")["configs"])
    ids |= set(c.read(c.PRIOR / "stage_a/results.json")["recomputed_cases"])
    ids |= {r["config_id"] for r in c.read(c.PRIOR / "stage_b/fresh_structural_pool.json")["transitions"]}
    return ids


def population(frozen):
    return [r for r in frozen["cases"] if r["included_in_structural_pool"]]


def derive():
    prior = c.inventory()
    diagnostic = []
    for cfg, original in c.diagnostic_cases().items():
        m,n,w = original["M"],original["N"],original["num_warps"]
        diagnostic.append({"config_id":cfg,"M":m,"N":n,"num_warps":w,"logical_shape":[1,m,n],"reduction_axis":1,
            "origin":"OUTCOME_INFORMED_PRIMARY_DIAGNOSTIC","default":predicted_candidate(m,n,w),"cand4":predicted_candidate(m,n,w,4)})
    fresh, excluded = [], []
    for m in FRESH_DOMAIN["M"]:
        for n in FRESH_DOMAIN["N"]:
            for w in FRESH_DOMAIN["num_warps"]:
                cfg = f"M{m}_N{n}_w{w}"
                if cfg in old_ids():
                    excluded.append({"config_id":cfg,"reason":"PRIOR_CASE_ID"}); continue
                fresh.append({"config_id":cfg,"M":m,"N":n,"num_warps":w,"logical_shape":[1,m,n],"reduction_axis":1,
                    "origin":"UNMEASURED_LARGE_M_WARP_EXTENSION","default":predicted_candidate(m,n,w),"cand4":predicted_candidate(m,n,w,4)})
    rows = p4.select(sorted(diagnostic+fresh,key=lambda r:r["config_id"]))
    c.require(len(diagnostic)==9 and len(fresh)==20 and len(excluded)==4, "Frozen complete diagnostic and fresh domains")
    c.require(not (set(r["config_id"] for r in fresh)&old_ids()), "Every fresh identity unseen")
    pool={"cases":rows,"diagnostic":9,"fresh":20,"fresh_source_domain":FRESH_DOMAIN,"prior_identity_exclusions":excluded,
          "included":sum(r["included_in_structural_pool"] for r in rows)}
    protocol={"phase":"PHASE_7","baseline":c.BASELINE,"diagnostic_role":"OUTCOME_INFORMED_INTERVENTION_DIAGNOSTIC",
        "fresh_role":"PROSPECTIVE_FIXED_EXTENT_AND_WARP_EXTENSION","fresh_domain":FRESH_DOMAIN,
        "matrix":{"descriptor":["host","device"],"store":["native","canonical"],"frontend":"Gluon for all four matrix cells",
                  "canonical_reference":"Unchanged Phase4 Triton source, compiled once alongside the matrix"},
        "dimensions":{"B_DESC":65536,"B_RUN":list(c.B_VALUES),"R":1,"invocations":3,"rounds":10,"samples_per_visit":10,"warmups":3},
        "inputs":"Seed42 BF16 torch.randn full descriptor; one allocation per unique(M,N) shared across warps/candidates/harnesses; full FP32 output and B_DESC*128 scratch",
        "admission":"Actual same logical tile/layout/shared/TMA/load family; full source-bound convert/max body matches canonical by sequence or complete opcode multiset; all ten binaries zero LOCAL/STACK; exact-CUBIN candidate occupancy matched within each harness; all correctness smokes pass; no PENDING launch",
        "classification":"PRIMARY all reduction fingerprints exact sequence, SECONDARY full opcode multiset; no timing-based upgrade",
        "controls":"Report all full operand-sensitive body inventories, initial-load signatures, resources, cross-cell residency and output/descriptor interventions. Source intervention includes compiler responses; no pure-component latency/causal share claim",
        "ordering":"Full29-case sorted master tags; deterministic cyclic offsets17 per invocation/13 per round; stable filter by diagnostic or fresh eligibility",
        "timing":"Strict H100!:1; three separately dispatched single-use processes; driver events one archived kernel/sample; full CUBIN SHA before every launch; no Triton/JIT/compiler",
        "analysis":"Median100 at three B; rational OLS ns/additional CTA. Hxy=default-4; store=H01-H00; descriptor=H10-H00; interaction=H11-H10-H01+H00. G is canonical gap. Three invocation values/mean/SD/fixeddf2 sign bands; all residuals retained",
        "fresh_hypotheses":{"H7_01_OUTPUT_CONTEXT_TRANSFER":{"alternative":"host_canonical","reference":"host_native"},
                            "H7_02_FULL_CONTEXT_TRANSFER":{"alternative":"device_canonical","reference":"host_native"},
                            "H7_03_DESCRIPTOR_INCREMENT":{"alternative":"device_canonical","reference":"host_canonical"}},
        "decisions":"All fixed coefficient1/intercept0 predictors of G; n>=5; SUPPORTED if both MAE/RMSE strictly lower, FALSIFIED if both >=, mixed INCONCLUSIVE. n<5 INCONCLUSIVE_BY_COVERAGE. ALL_ELIGIBLE and PRIMARY separately; no refit or tuned tolerance",
        "profiling":"Separate StageC diagnostic counters if ncu/driver permits; archive reports/commands/cache/clock/replay settings; unavailable tool/permissions recorded, not replaced by fabricated counters. Profiling duration never enters formal timing",
        "repair":"Only documented infrastructure/protocol failure; original returns and partials durable before retrieval/validation; fixed max3 whole-invocation attempts, no outcome-based rerun",
        "build":"Reuse Phase6 native core image excluding experiments, persistent ccache/triton-home; actual compiler/native SHA and cache stats; source snapshot and original exports retained",
        "stop_after":"StageD final validator/commits/push; no subsequent phase or production compiler heuristic"}
    return json.loads(c.encode({"pool":pool,"protocol":protocol,"protected_files":len(prior),"prior_inventory_SHA256":c.sha(c.encode(prior))}))


def expected_abi(harness):
    if harness=="canonical" or harness.startswith("device_"):
        return ["u64","u64","u32","u32","u64","u64"]
    return ["descriptor128","u32","u32","u32","u64","u64","u64","u64","u64","u64"]


def schedule(cases):
    pool = c.read(DEST/"pool.json")
    tags=sorted(f'{r["config_id"]}:{h}:{k}:RNone:B{b}' for r in pool["cases"] for h in HARNESSES for k in CANDIDATES for b in c.B_VALUES)
    invocations=[]
    for invocation in (1,2,3):
        rounds=[]
        for round_ in range(1,11):
            offset=((invocation-1)*17+(round_-1)*13)%len(tags)
            order=[tag for tag in tags[offset:]+tags[:offset] if tag.split(":")[0] in cases]
            rounds.append({"round":round_,"samples_per_visit":10,"order":order})
        invocations.append({"invocation":invocation,"rounds":rounds})
    return {"invocations":invocations,"conditions_per_invocation":len(cases)*30,"scalar_samples":len(cases)*9000,"warmups_per_condition":3,"executed":False}


def main():
    expected=derive()
    if "--validate" in sys.argv:
        c.require(c.read(DEST/"contracts.json")==expected,"Full Phase7 preregistration independently rederives")
        for name in ("pool","protocol"):c.require(c.read(DEST/(name+".json"))==expected[name],"Frozen "+name)
        probes=[]
        for name,mutate in (("domain",lambda r:r["pool"]["cases"].pop()),("predictor",lambda r:r["protocol"]["fresh_hypotheses"]["H7_01_OUTPUT_CONTEXT_TRANSFER"].update(alternative="REFIT"))):
            bad=copy.deepcopy(expected);mutate(bad);c.require(bad!=expected,"Preregistration tamper rejected");probes.append(name)
        result={"status":"PASS","diagnostic":9,"fresh":20,"prior_excluded":4,"corruption_probes":probes,"no_GPU":True}
        c.write(DEST/"validation.json",result);print(result)
    else:
        c.write(DEST/"contracts.json",expected)
        for name in ("pool","protocol"):c.write(DEST/(name+".json"),expected[name])
        (DEST/"summary.md").write_text("# Phase7 StageB frozen contracts\n\nNine historical PRIMARY diagnostic cases and20 entirely unseen case identities. Full M={1024,2048},N={16,32,64,128},warps={4,8,16} domain accounted for; four prior identities excluded before compilation. Four Gluon matrix cells plus unchanged canonical reference; no model fitting. All three prospective comparisons and both analysis scopes frozen before any new timing.\n")
        print("Frozen Phase7 matrix, diagnostic cohort and all20 fresh identities before timing")


if __name__=="__main__":main()
