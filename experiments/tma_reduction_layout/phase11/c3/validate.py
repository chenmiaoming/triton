"""Final CPU-only evidence/derivation validator. No fitting is hidden."""
import argparse
import inspect
import json
import subprocess
import sys
from experiments.tma_reduction_layout.phase11.c3 import common as c
from experiments.tma_reduction_layout.phase11.c3.static import (
    candidates,plan,compact,coordinates,shared_offset,upstream)


def validate(committed=False,skip_prior_hash=False):
    protocol=c.read(c.OUT/'preregistration/protocol.json')
    freeze=c.read(c.OUT/'freeze.json')
    for name,digest in freeze['files'].items():
        assert c.digest((c.ROOT/name).read_bytes())==digest,('C3 freeze',name)
    if not skip_prior_hash:
        for name,digest in protocol['prior_manifest'].items():
            assert c.digest((c.ROOT/name).read_bytes())==digest,('prior changed',name)
    # --skip-prior-hash is only for iteration; final validation must hash all.
    if committed:
        assert not skip_prior_hash
        tracked=set(subprocess.check_output(['git','ls-files'],cwd=c.ROOT,text=True).splitlines())
        assert set(freeze['files']).issubset(tracked)
        for name,digest in freeze['files'].items():
            data=subprocess.check_output(['git','show','HEAD:'+name],cwd=c.ROOT)
            assert c.digest(data)==digest,('committed bytes',name)
    audit=c.read(c.OUT/'static_audit.json')
    assert len(audit['rows'])==15 and audit['all_pass']
    assert all(all(r['checks'].values()) for r in audit['rows'])
    from experiments.tma_reduction_layout.phase11.c3.audit import run as static_audit
    assert static_audit()==audit,'static counter re-derivation'
    history=c.read(c.OUT/'history_audit.json')
    assert c.digest((c.OUT/'history_audit.json').read_bytes())==protocol['history_audit_SHA256']
    supplement=c.read(c.OUT/'supplemental_source_audit.json')
    assert supplement['blind_all_zero']
    assert all(not a['project_experiment_members'] for a in supplement['archives'])
    blind=c.read(c.OUT/'D1_predictions.json')
    assert not blind['GPU_timed'] and not blind['compiled'] and len(blind['cases'])==8
    assert blind['theta_SHA256']==c.digest((c.OUT/'theta_H100.json').read_bytes())
    assert blind['protocol_SHA256']==c.digest((c.OUT/'preregistration/protocol.json').read_bytes())
    assert blind['static_plan_revision_SHA256']==c.digest((c.OUT/'preregistration/static_plan_revision_1.json').read_bytes())
    all_cases=[x for g in ((32,64,8),(32,128,4)) for x in candidates(*g)]
    revision=c.read(c.OUT/'preregistration/static_plan_revision_1.json')
    for entry in blind['cases']:
        m,n,w=entry['geometry'];hist=next(r for r in history['pool'] if
            (r['BM'],r['BN'],r['W'])==(m,n,w))
        assert not hist['exposed'] and hist['static_128B_shape_legal']
        assert f'{m}_{n}_{w}' not in supplement['hits']
        cases=candidates(m,n,w)
        updated=next(r for r in revision['cases'] if r['id']==hist['case'])
        assert updated['plans']==[compact(plan(x)) for x in cases]
        assert entry['default']==upstream(m,n,w)
        assert {p['id'] for p in entry['predictions']}=={x['id'] for x in cases}
        all_cases+=cases
    for case in all_cases:
        m,n,w=case['BM'],case['BN'],case['W'];ll=case['register_layout']
        v=ll['size_per_thread'][2];seen=[]
        for rm,rn,warp,coords in coordinates(m,n,w,ll):
            seen.extend((r,col+i) for r,col in coords for i in range(v))
        assert len(seen)==len(set(seen))==m*n,('ownership bijection',case['id'])
        assert set(seen)=={(r,col) for r in range(m) for col in range(n)}
        assert {shared_offset(r,col,n) for r,col in seen}==set(range(m*n))
        p=plan(case)
        assert p['p']*p['u']*p['w']==m
        assert p['p']*p['columns_per_thread']==p['elements_per_thread']
        assert p['ML']['wavefronts']-p['ML']['ideal_wavefronts']==p['ML']['bank_conflicts']
        assert p['resources']['static_shape_legal'] and p['resources']['shared_legal']
        assert not p['resources']['definite_register_infeasible']
    # Deterministic re-derivation uses only frozen C2 and static features.
    from experiments.tma_reduction_layout.phase11.c3.calibrate import calibrate
    from experiments.tma_reduction_layout.phase11.c3.evaluate import evaluate
    from experiments.tma_reduction_layout.phase11.c3.model import predict,select
    assert list(inspect.signature(predict).parameters)==['case','theta','mode']
    expected_theta=c.read(c.OUT/'theta_H100.json')
    expected_dev=c.read(c.OUT/'development.json')
    theta=calibrate();assert theta==expected_theta,'theta re-derivation'
    value=evaluate(theta);assert value==expected_dev,'development re-derivation'
    assert c.read(c.OUT/'D1_predictions.json')==blind,'blind re-derivation'
    assert value['robust_switch_count']==0
    assert value['body_calibration_interval_coverage']==15
    assert value['K1_calibration_interval_coverage']==15
    for row in blind['cases']:
        for p in row['predictions']:
            for key in ('steady_ns','onchip_steady_ns','K1_ns','single_CTA_effective_chain_ns'):
                assert 0<=p[key][0]<=p[key][1],(p['id'],key,p[key])
    assert not any(p.suffix in ('.cubin','.ncu-rep','.ptx','.sass') for p in c.OUT.rglob('*'))
    assert all(name not in sys.modules for name in ('triton','torch','modal'))
    for name,digest in freeze['files'].items():
        assert c.digest((c.ROOT/name).read_bytes())==digest,('recompute changed freeze',name)
    result={'status':'PASS','static_layouts':15,'blind_cases':8,
        'ownership_bijection_layouts':len(all_cases),'prior_files_hashed':0 if skip_prior_hash else len(protocol['prior_manifest']),
        'frozen_files':len(freeze['files']),'committed_bytes_checked':committed,
        'model_selection_utility_validated':False,'new_GPU_jobs':0,
        'D1_started':False,'Triton_native_rebuilds':0,'deterministic_recompute':True}
    if not committed:
        c.write(c.OUT/'validation.json',result)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--committed',action='store_true')
    parser.add_argument('--skip-prior-hash',action='store_true')
    args=parser.parse_args()
    validate(args.committed,args.skip_prior_hash)
