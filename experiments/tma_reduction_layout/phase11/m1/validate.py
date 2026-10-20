"""CPU evidence validator, independent identities and optional final freeze."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ['CUDA_VISIBLE_DEVICES']=''
from collections import defaultdict
import csv
import io
import math
import statistics as st
import subprocess
import sys
import numpy as np
from experiments.tma_reduction_layout.phase11.m1 import common as c
from experiments.tma_reduction_layout.phase11.m1 import profile,pipeline,mechanisms,plot,report


def close(a,b):assert math.isclose(a,b,rel_tol=2e-10,abs_tol=1e-7),(a,b)


def run(freeze=False,committed=False):
    c.protect_prior();p=c.protocol()
    assert p['CPU_only'] and p['GPU_allocations']==0 and not p['new_D1_predictions']
    assert c.sha((c.SOURCE/'PLAN.md').read_bytes())==p['plan_SHA256']
    assert c.sha((c.OUT/'prior_manifest.json').read_bytes())==p['prior_manifest_SHA256']
    prof=c.read(c.OUT/'identifiability.json');pipe=c.read(c.OUT/'pipeline.json');mech=c.read(c.OUT/'mechanisms.json')
    assert all(r['status']=='OK' for r in (prof,pipe,mech))
    assert not prof['D1_used_for_fit'] and len(prof['development_geometries'])==9
    assert pipe['no_refit'] and not pipe['D1_rescored'] and not pipe['control_interpolation']
    assert len(prof['fits'])==10 and len(mech['pairs'])==49 and len(mech['PC_provenance'])==6
    original=c.read(c.source('stage_c34','theta_H100.json'))
    scalar=c.read(c.source('stage_c33','theta_H100.json'))
    for row in prof['CV_curve']:
        scores=[r for r in original['inner_scores'] if r['family']=='partial_overlap' and r['eta']==row['eta']]
        best=min(scores,key=profile.fit.recipe_key)
        assert best['inner_relative_loss']==row['best_existing_inner_CV_loss']
        assert best['ridge']==row['best_existing_inner_CV_ridge']
    D1={c.geo(r['geometry']) for r in c.read(c.source('stage_d1','preregistration/protocol.json'))['D1_cases']}
    for fit in prof['fits']:
        mm=fit['model'];assert not D1.intersection(mm['training_geometries'])
        assert not D1.intersection(mm['control_training_geometries'])
        assert not D1.intersection(mm['MC']['training_geometries'])
        assert set(mm['training_geometries'])==set(prof['development_geometries'])
        assert fit['eta'] in p['eta_grid'] and mm['eta_q_slope']==0
        assert fit['FULL_feature_design']['columns']==27 and fit['FULL_feature_design']['rank']==26
        assert 0<fit['active_column_design']['rank']<=fit['active_column_design']['columns']<=27
        if fit['profile']=='fixed_ridge':assert fit['active_column_design']['rank']==fit['active_column_design']['columns']
        rd=fit['MC_redundancy'];assert rd['MC_clipped_rows']==0
        assert rd['MC_column_max_reconstruction_error_ns']<1e-7 and rd['A_FULL_max_collapse_error_ns']<1e-7
        assert len(rd['collapsed_coefficients'])==26
        if fit['profile']=='fixed_ridge' and fit['eta']==.25:
            assert mm['full']['feature_names']==scalar['model']['full']['feature_names']
            assert np.max(np.abs(np.array(mm['full']['coefficients'])-np.array(scalar['model']['full']['coefficients'])))<1e-8
    bundle=pipeline.load()
    originals={(r['cohort'],r['tag'],r['worker'],r['GPU_UUID'],r['q']):r for r in bundle['full']}
    controls={(r['cohort'],r['geometry'],r['worker'],r['GPU_UUID'],r['q']):r for r in bundle['controls']}
    natural=0
    for r in pipe['records']:
        raw=originals[r['cohort'],r['tag'],r['worker'],r['GPU_UUID'],r['actual_q']]
        assert raw['target_ns']==r['observed_FULL_ns']
        assert raw['condition_aliases']==r['condition_aliases']
        for key,v in r['predictions'].items():
            if v is None:
                assert key=='measured_control' and r['missing_control_at_predicted_q']
                continue
            if v['partial_overlap']:close(v['tile_ns'],max(v['control_ns'],v['A_FULL_ns']+(1-v['eta'])*v['control_ns']))
            if key in ('measured_control','measured_q_and_control'):
                q=r['predicted_q'] if key=='measured_control' else r['actual_q']
                ctrl=controls[r['cohort'],r['geometry'],r['worker'],r['GPU_UUID'],q]
                assert v['control_ns']==ctrl['target_ns'] and v['q']==q
        if r['q_condition']=='natural':natural+=1
        if all(v is not None for v in r['predictions'].values()):
            v=r['predictions'];close(r['substitution_interaction_ns'],v['measured_q_and_control']['tile_ns']-
                v['measured_q']['tile_ns']-v['measured_control']['tile_ns']+v['static']['tile_ns'])
    assert natural==1080==pipe['natural_frozen_predictions_reproduced']
    for summary in pipe['summary']:
        rows=[r for r in pipe['records'] if r['model']==summary['model'] and r['cohort']==summary['cohort'] and
              (summary['endpoint']!='natural' or r['q_condition']=='natural')]
        for key in pipeline.VARIANTS:
            groups=defaultdict(list)
            for row in rows:
                if row['predictions'][key] is not None:groups[row['layout'],row['actual_q']].append(row)
            errors=[]
            for vals in groups.values():errors.append(st.median(v['predictions'][key]['tile_ns'] for v in vals)/st.median(v['observed_FULL_ns'] for v in vals)-1)
            close(summary['all_available'][key]['FULL_relative_RMSE_percent'],100*math.sqrt(sum(e*e for e in errors)/len(errors)))
            assert summary['all_available'][key]['conditions']==len(groups)
    for pair in mech['pairs']:
        nf=bundle['gates'][pair['cohort']]['metadata'][pair['N_first']]
        wi=bundle['gates'][pair['cohort']]['metadata'][pair['within']]
        assert nf['vec']==wi['vec']==pair['vec']
        for row in pair['same_worker_same_q_contrasts']:
            a=originals[pair['cohort'],pair['N_first'],row['worker'],row['GPU_UUID'],row['q']]
            b=originals[pair['cohort'],pair['within'],row['worker'],row['GPU_UUID'],row['q']]
            close(row['within_minus_N_first_ns'],b['target_ns']-a['target_ns'])
            close(row['within_gain_percent'],100*(1-b['target_ns']/a['target_ns']))
    for record in mech['PC_provenance']:
        path=c.ROOT/record['disassembly'];assert c.sha(path.read_bytes())==record['disassembly_SHA256']
        raw=c.read(path);assert raw['CPU_only'] and raw['returncode']==0
        counters=bundle['counters'][record['cohort']]
        tag=counters['profile_aliases'].get(record['tag'],record['tag'])
        dynamic=next(r for r in counters['single_pass'] if r['tag']==tag)
        assert len(record['per_PC'])==len(dynamic['per_PC'])
        sums=defaultdict(int)
        for row in record['per_PC']:
            orig=dynamic['per_PC'][row['relative_PC']]
            assert row['instruction']==orig['instruction'] and row['count']==orig['count']
            for family in row['families']:sums[family]+=row['count']
        for family,count in dynamic['counts'].items():assert sums[family]==count
    names=('identifiability.json','pipeline.json','mechanisms.json','REPORT.md')
    assert not any(ord(ch)<32 and ch not in '\n\r\t' for ch in (c.OUT/'REPORT.md').read_text())
    before={name:(c.OUT/name).read_bytes() for name in names}
    profile.run();pipeline.run();mechanisms.run();report.run()
    assert all((c.OUT/name).read_bytes()==blob for name,blob in before.items()),'Non-reproducible audit output'
    for name in ('eta_CV_profile','design_and_response_terms','D1_substitution_errors','FULL_q_response'):
        assert (c.OUT/'figures'/(name+'.png')).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert (c.OUT/'figures'/(name+'.pdf')).read_bytes().startswith(b'%PDF-')
    for path in (c.OUT/'tables').glob('*.csv'):
        rows=list(csv.DictReader(io.StringIO(path.read_text())));assert rows,path
    assert not any(n=='triton' or n.startswith('triton.') or n in ('torch','modal') for n in sys.modules)
    if freeze and not (c.OUT/'freeze.json').exists():
        files={str(path.relative_to(c.ROOT)):c.sha(path.read_bytes()) for root in (c.SOURCE,c.OUT)
            for path in sorted(root.rglob('*')) if path.is_file() and '__pycache__' not in path.parts and
            path.name not in ('freeze.json','validation.json')}
        c.write_once(c.OUT/'freeze.json',{'files':files,'stage':'M1','CPU_only':True,'GPU_allocations':0})
    f=c.read(c.OUT/'freeze.json') if (c.OUT/'freeze.json').exists() else None
    if f:
        for name,digest in f['files'].items():assert c.sha((c.ROOT/name).read_bytes())==digest,name
    if committed:
        assert f
        tracked=set(subprocess.check_output(['git','ls-files'],cwd=c.ROOT,text=True).splitlines());assert set(f['files'])<=tracked
        subprocess.run(['git','diff','--exit-code','HEAD','--',str(c.SOURCE),str(c.OUT)],cwd=c.ROOT,check=True)
    result={'status':'PASS','stage':'M1','CPU_only':True,'GPU_allocations':0,'native_rebuilds':0,
        'prior_files_preserved':len(c.read(c.OUT/'prior_manifest.json')['files']),
        'development_only_fits':10,'frozen_natural_worker_predictions_reproduced':natural,
        'diagnostic_worker_rows':len(pipe['records']),'same_vec_layout_pairs':49,'original_CUBIN_source_maps':6,
        'D1_refit':False,'D1_rescored':False,'frozen_files':len(f['files']) if f else 0,'D2_started':False}
    c.derived(c.OUT/'validation.json',result)
    print('M1 VALIDATION PASS: prior hashes, development-only fits, frozen responses, exact-q controls and PC count conservation',flush=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--committed',action='store_true')
    args=parser.parse_args();run(args.freeze,args.committed)
