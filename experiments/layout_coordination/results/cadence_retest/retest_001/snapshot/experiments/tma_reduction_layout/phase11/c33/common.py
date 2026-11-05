"""Pure CPU C3.3 model paths and immutable C2d evidence binding."""
from experiments.tma_reduction_layout.phase11.c2d import common as prior
from pathlib import Path

ROOT, BASE = prior.ROOT, prior.BASE
SOURCE = Path(__file__).resolve().parent
OUT = BASE/'results/phase11/stage_c33'
read, encode, sha = prior.read, prior.encode, prior.sha
derived, write_once = prior.derived, prior.write_once


def protocol():
    return read(OUT/'protocol.json')


def preregister():
    binding = prior.protocol()
    method = (SOURCE/'PLAN.md').read_bytes()
    assert sha(method)==binding['C33_method_SHA256']
    value = {'stage':'C3.3','date':'2026-10-18','plan_SHA256':sha(method),
        'C2d_protocol_SHA256':sha(encode(binding)),
        'protected_D1':binding['reserved_D1'],'moved_to_development':[16,64,8],
        'ridge_candidates':[0.,.001,.01,.1,1.], 'overlap_eta_candidates':[0.,.25,.5,.75,1.],
        'model_families':['full_structural','partial_overlap'],
        'modes':['full','Load-blind','Comm-blind','Resource/Envelope-blind','Fixed-R-blind'],
        'minimum_distinct_geometries_per_stratum':3,
        'practical_switch_percent':3.,'uncertainty_relative_margin_floor':.01,
        'resource_ridge':.001,'resource_margin_registers':8.,
        'historical_26_6_used':False,'global_grid_feature':False,
        'signed_residual_reflection':False,'D1_compiled':False,'D1_timed':False}
    write_once(OUT/'protocol.json',value)
    print('C3.3 method bound to pre-measurement C2d protocol')


if __name__=='__main__':
    preregister()
