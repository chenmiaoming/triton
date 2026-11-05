"""Record the unchanged blind split and C3.1 development-only protocol."""
from experiments.tma_reduction_layout.phase11.c31 import common as c


def run():
    target=c.OUT/'preregistration/protocol.json'
    assert not target.exists(),'write-once protocol'
    prior=c.prior_manifest()
    split=c.read(c.V0/'preregistration/protocol.json')
    cases=[{k:r[k] for k in ('BM','BN','W','id')} for r in split['blind_cases']]
    c.write(target,{'stage':'C3.1','version':'c3_v1','date':'2026-10-17',
        'parent_commit':c.PARENT,'prior_manifest':prior,'blind_cases':cases,
        'old_split_SHA256':c.sha((c.V0/'preregistration/protocol.json').read_bytes()),
        'old_prediction_SHA256':c.sha((c.V0/'D1_predictions.json').read_bytes()),
        'plan_SHA256':c.sha((c.SOURCE/'PLAN.md').read_bytes()),
        'target':'H100 sm90','dtype':'bfloat16','reduce_axis':1,
        'descriptor':'host-created','threshold':0.03,
        'target_quantity':'MC D1 steady ns per tile per SM, K16/64/256 and multi-grid slopes',
        'model_recipes':{'features':['static_counts','counts_plus_primitive_shapes'],
            'q_response':['inverse_q','hinges_q1_q2_q4'],
            'ridge':[0.0,0.0001,0.01,0.1],
            'selection_loss':'leave-geometry-out paired-difference squared error; ties favor fewer features, stronger ridge'},
        'pairing':['cohort','GPU_UUID','worker_invocation','q_condition','geometry'],
        'uncertainty':'joint (default residual, candidate-minus-default residual), from same-fold LOLO/LOGO pairs and same-condition worker estimates',
        'auxiliary_only':['MO','K1','grid1'],
        'new_GPU_jobs':0,'D1_compiled':False,'D1_timed':False,'stop_before_D1':True})
    c.write(c.OUT/'c3_v0_overconservative/REFERENCE.json',{
        'version':'c3_v0_overconservative','original_directory':str(c.V0.relative_to(c.ROOT)),
        'commit':c.PARENT,'freeze_SHA256':c.sha((c.V0/'freeze.json').read_bytes()),
        'REPORT_SHA256':c.sha((c.V0/'REPORT.md').read_bytes()),
        'original_theta_version':c.read(c.V0/'theta_H100.json')['version'],
        'note':'Historical name alias only. Original files and their older internal version labels are unchanged.'})
    print('C3.1 protocol frozen; same eight blind cases; prior files',len(prior))


if __name__=='__main__':run()
