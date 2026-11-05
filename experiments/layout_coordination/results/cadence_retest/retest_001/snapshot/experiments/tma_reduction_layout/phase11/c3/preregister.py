"""Freeze the split before C3 parameter fitting, never launch a GPU."""
from experiments.tma_reduction_layout.phase11.c3 import common as c
from experiments.tma_reduction_layout.phase11.c3.static import candidates, plan, compact

# Coverage chosen from history metadata, without looking at new timings.
BLIND=((8,64,4),(8,128,8),(16,128,2),(16,256,8),
       (32,256,2),(64,128,2),(64,512,4),(128,64,2))


def main():
    audit=c.read(c.OUT/'history_audit.json')
    rows=[]
    for m,n,w in BLIND:
        history=next(r for r in audit['pool'] if (r['BM'],r['BN'],r['W'])==(m,n,w))
        assert not history['exposed'] and history['static_128B_shape_legal'],history
        plans=[compact(plan(x)) for x in candidates(m,n,w)]
        assert plans and all(p['resources']['shared_legal'] and
                             not p['resources']['definite_register_infeasible'] for p in plans)
        rows.append({'BM':m,'BN':n,'W':w,'id':history['case'],
                     'static_candidates':plans,'exposure_count':0})
    inputs={}
    for name in ['stage_c2a/analysis.json','stage_c2b/analysis.json',
                 'stage_c2b/artifact_gate.json','stage_c2b/counter_gate.json']:
        path=c.BASE/'results/phase11'/name
        inputs[str(path.relative_to(c.ROOT))]=c.digest(path.read_bytes())
    value={'date':'2026-10-17','stage':'C3','target':'H100 sm90',
           'pinned_upstream':c.PIN,'dtype':'bfloat16','reduce_axis':1,
           'descriptor_path':'host-created; original regression path',
           'input_shape':[1,8192,8192],'practical_threshold':0.03,
           'blind_cases':rows,'calibration_input_SHA256':inputs,
           'history_audit_SHA256':c.digest((c.OUT/'history_audit.json').read_bytes()),
           'prior_manifest':{r['path']:r['SHA256'] for r in audit['scanned']},
           'freeze_timing':'After C2; before C3 fitting and before any D1 compilation or timing. Original draft requested pre-C2 split; no such freeze existed. Latest user authorization explicitly requests the audit now.',
           'new_GPU_jobs':0,'native_rebuilds':0,'blind_GPU_timing':False,
           'selection_inputs':['BM','BN','W','dtype','reduce_axis','register_layout','shared_layout'],
           'stop':'After model/parameters/predictions commit; before D1 execution'}
    target=c.OUT/'preregistration/protocol.json'
    assert not target.exists(),'Split is write-once'
    c.write(target,value)
    print('frozen', [r['id'] for r in rows])


if __name__=='__main__':
    main()
