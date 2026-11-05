"""CPU-only evidence and method validation, distinct from D1 validation."""
import copy
import inspect
import json
import subprocess
import sys
from unittest.mock import patch
import numpy as np
from experiments.tma_reduction_layout.phase11.c31 import common as c
from experiments.tma_reduction_layout.phase11.c31 import calibrate,evaluate,model,data
from experiments.tma_reduction_layout.phase11.c31.features import features
from experiments.tma_reduction_layout.phase11.c3.static import compact,plan,upstream


def validate(committed=False,skip_prior=False):
    protocol=c.read(c.OUT/'preregistration/protocol.json')
    frozen=c.read(c.V1/'freeze.json')
    assert frozen['version']=='c3_v1' and not frozen['D1_compiled'] and not frozen['D1_timed']
    for path,digest in frozen['files'].items():assert c.sha((c.ROOT/path).read_bytes())==digest,path
    if not skip_prior:
        for path,digest in protocol['prior_manifest'].items():
            assert c.sha((c.ROOT/path).read_bytes())==digest,('prior modified',path)
    if committed:
        assert not skip_prior
        for path,digest in {**frozen['files'],
            str((c.V1/'freeze.json').relative_to(c.ROOT)):c.sha((c.V1/'freeze.json').read_bytes())}.items():
            assert c.sha(subprocess.check_output(['git','show','HEAD:'+path],cwd=c.ROOT))==digest,path
    old=c.read(c.V0/'freeze.json')
    for path,digest in old['files'].items():assert c.sha((c.ROOT/path).read_bytes())==digest,('C3-v0',path)
    theta=c.read(c.V1/'theta_H100.json');development=c.read(c.V1/'development.json')
    blind=c.read(c.V1/'D1_predictions.json')
    assert blind['theta_SHA256']==c.sha((c.V1/'theta_H100.json').read_bytes())
    assert blind['protocol_SHA256']==c.sha((c.OUT/'preregistration/protocol.json').read_bytes())
    assert not blind['compiled'] and not blind['GPU_timed']
    old_blind=c.read(c.V0/'D1_predictions.json')
    assert [r['geometry'] for r in blind['cases']]==[r['geometry'] for r in old_blind['cases']]
    assert len(blind['cases'])==8
    history=c.read(c.V0/'history_audit.json')
    for r in blind['cases']:
        g=r['geometry'];h=next(t for t in history['pool'] if [t['BM'],t['BN'],t['W']]==g)
        assert not h['exposed'] and h['static_128B_shape_legal']
    cases,workers,cells=data.development()
    assert len(cells)==60 and len(workers)==180
    # Recheck the old static evidence without rewriting any old artifacts.
    audit=c.read(c.V0/'static_audit.json')
    for row in audit['rows']:
        p=plan(row['case']);assert compact(p)==row['static']
        obs=row['observed']
        assert p['ML']['wavefronts']==obs['ML_WF']
        assert p['ML']['bank_conflicts']==obs['ML_BC']
        assert p['MR']['shuffle_warp_instructions']==obs['MR_SHFL']
        assert p['MR']['barrier_warp_instructions']==obs['MR_BAR']
        assert p['MR']['shared_load_wavefronts']==obs['MR_WF']
    # Folds must actually exclude every q/worker of the held group. Refit every
    # saved fold and inspect matching cohort/worker/UUID records.
    paired=0
    for mode,entry in theta['models'].items():
        assert all(v>=0 for v in entry['model']['coefficients'])
        for fold in entry['cross_validation']['fold_models']:
            key='geometry' if fold['level']=='LOGO' else 'layout'
            train=[r for r in cells if r[key]!=fold['held']]
            assert all(r[key]!=fold['held'] for r in train)
            refit=calibrate.fit(train,theta['recipe'],theta['primitive_shape_tables'],mode)
            assert refit==fold['model'],('fold refit',mode,fold['held'])
        for r in entry['cross_validation']['paired_residuals']:
            a=next(x for x in workers if x['layout']==r['layout'] and x['q_condition']==r['q_condition']
                and x['cohort']==r['cohort'] and x['worker']==r['worker'] and x['GPU_UUID']==r['GPU_UUID'])
            d=next(x for x in workers if x['layout']==r['default'] and x['q_condition']==r['q_condition']
                and x['cohort']==r['cohort'] and x['worker']==r['worker'] and x['GPU_UUID']==r['GPU_UUID'])
            assert abs((a['service_ns']-d['service_ns'])-r['observed_delta_ns'])<1e-9
            assert abs((r['observed_delta_ns']-r['predicted_delta_ns'])-r['delta_residual_ns'])<1e-9
            paired+=1
    # Check monotonicity on real candidate plans, including all blind shapes.
    all_cases=list(cases.values())+[p['case'] for r in blind['cases'] for p in r['predictions']]
    monotonic=0
    for case in all_cases:
        p=model.plan(case)
        assert p['resources']['shared_legal'] and not p['resources']['definite_register_infeasible']
        for mode,entry in theta['models'].items():
            values=[calibrate.predict(entry['model'],case,q,theta['primitive_shape_tables']) for q in range(1,65)]
            assert all(a>=b-1e-8 for a,b in zip(values,values[1:])),(case['id'],mode)
            monotonic+=1
    # Every evaluated worst scenario uses the SAME residual/base and TMA in
    # the two service predictions. No marginal-interval subtraction.
    for selection in development['selections']+blind['cases']:
        for comp in selection['comparisons']:
            worst=comp['worst_scenario'];td=max(worst['TMA_ns'],max(0.,worst['default_MC_ns_before_clamp']))
            tc=max(worst['TMA_ns'],max(0.,worst['candidate_MC_ns_before_clamp']))
            assert abs(td-worst['default_tile_ns'])<1e-8 and abs(tc-worst['candidate_tile_ns'])<1e-8
            if comp['robust_switch_eligible']:
                assert comp['resource_gate_pass'] and comp['relative_change_interval'][1]<-.03
        for p in selection['predictions']:
            assert abs(sum(p['feature_contributions_ns'].values())-p['MC_point_ns'])<1e-7
    assert all(development['acceptance'].values())
    baseline=next(s for s in development['selections'] if s['mode']=='full' and s['geometry']==[32,128,4])
    assert baseline['choice']==baseline['default'] and baseline['reason']=='transfer_limited_comparable_set'
    extra=next(s for s in development['selections'] if s['mode']=='Envelope-only-blind' and s['geometry']==[32,128,4])
    assert extra['switched'],'envelope control'
    # Demonstrate auxiliary timings and old fixture ratios cannot change the
    # selected training cells, primitive tables or fitted MC model.
    original_read=c.read
    def poisoned(path):
        value=original_read(path)
        if str(path).endswith('stage_c2b/analysis.json'):
            value=copy.deepcopy(value)
            value['setup_and_K1_services']=[{'poisoned':True}]
            for row in value['services']:
                row['grid1_effective_chain_us']=1e30
                if row['family']=='MO':row['service_ns_per_tile_SM']=1e30
        if str(path).endswith('stage_c3/theta_H100.json'):
            value=copy.deepcopy(value);value['composition']={'poisoned':1e30}
        return value
    with patch.object(c,'read',side_effect=poisoned):
        _,_,unaffected=data.development()
        assert unaffected==cells
        tables=data.primitive_tables();assert tables==theta['primitive_shape_tables']
        assert calibrate.fit(cells,theta['recipe'],tables,'full')==theta['models']['full']['model']
    # Deterministic complete re-derivation into an in-memory sink: frozen files
    # remain write-once even when validating a committed version.
    def check_write(path,value):
        assert c.encode(value)==path.read_bytes(),('rederive',path)
    with patch.object(c,'write',side_effect=check_write):
        assert calibrate.run()==theta
        assert evaluate.run(theta)==development
    assert list(inspect.signature(model.select).parameters)==['m','n','w','theta','mode']
    assert not any(name in sys.modules for name in ('triton','torch','modal'))
    assert not any(p.suffix in ('.cubin','.ptx','.sass','.ncu-rep') for p in c.OUT.rglob('*'))
    for path,digest in frozen['files'].items():assert c.sha((c.ROOT/path).read_bytes())==digest,path
    result={'status':'PASS','version':'c3_v1','prior_files_hashed':0 if skip_prior else len(protocol['prior_manifest']),
        'static_layouts_matched':15,'MC_cells':60,'MC_worker_estimates':180,
        'paired_CV_records_checked':paired,'monotonic_case_models_checked':monotonic,
        'auxiliary_poison_invariance':True,'old_C3_unchanged':True,'D1_cases':8,
        'D1_compiled':False,'D1_timed':False,'new_GPU_jobs':0,'committed_bytes_checked':committed,
        'development_acceptance':development['acceptance'],'held_out_validation_passed':False}
    if not committed:c.write(c.V1/'validation.json',result)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--committed',action='store_true')
    parser.add_argument('--skip-prior',action='store_true');args=parser.parse_args()
    validate(args.committed,args.skip_prior)
