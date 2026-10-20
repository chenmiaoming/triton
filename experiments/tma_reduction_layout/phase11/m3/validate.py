"""Final CPU integrity and independent three-footprint contrast checks."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ['CUDA_VISIBLE_DEVICES']=''
from collections import defaultdict
import math
import statistics as st
import subprocess
from experiments.tma_reduction_layout.phase11.m3 import common as c
from experiments.tma_reduction_layout.phase11.m3 import analyze, diagnose, offline_import, report
from experiments.tma_reduction_layout.phase11.m3.static import plan
from experiments.tma_reduction_layout.phase11.c3.static import candidates


def close(a,b):
    assert math.isclose(a,b,rel_tol=2e-10,abs_tol=1e-7),(a,b)


def run(freeze=False,committed=False):
    c.protect_prior()
    p=c.protocol()
    c.verify_sources(p)
    assert p['pinned_upstream']==c.PIN and p['profile']=='chenmiaoming'
    assert not p['selector_refit'] and not p['new_blind_claim']
    assert c.sha((c.SOURCE/'PLAN.md').read_bytes())==p['plan_SHA256']
    assert c.sha((c.OUT/'preregistration/prior_manifest.json').read_bytes())==p['prior_manifest_SHA256']
    assert c.sha((c.OUT/'preregistration/static_predictions.json').read_bytes())==p['predictions_SHA256']
    predictions=c.read(c.OUT/'preregistration/static_predictions.json')
    for r in predictions['layouts']:
        case=next(case for g in {(x['M'],x['N'],x['num_warps']) for x in p['cases']}
                  for case in candidates(*g) if case['id']==r['layout_key'])
        assert plan(case)==r
    assert len(p['cases'])==34 and sum('reuse_source' in r for r in p['cases'])==22
    cg=c.read(c.OUT/'compile_gate.json')
    assert cg['new_compiles']==12 and cg['reused']==22 and not cg['GPU_allocated_for_compile']
    assert len(cg['metadata'])==34
    for meta in cg['metadata'].values():
        folder=c.artifact_folder(meta,cg)
        assert c.sha((folder/'kernel.cubin').read_bytes())==meta['cubin_SHA256']
        if meta.get('reuse_source'):
            assert (c.ROOT/meta['reuse_source']/'kernel.cubin').read_bytes()==(folder/'kernel.cubin').read_bytes()
        if meta['output_policy']=='native':
            ir=(folder/'kernel.glir').read_text()
            assert 'tt.reduce' in ir and 'tt.store' in ir
    gate=c.read(c.OUT/'artifact_gate.json')
    eligibility=c.read(c.ROOT/gate['eligibility_path'])
    assert eligibility['status']=='OK' and len(eligibility['checks'])==102
    assert all(r['pass'] and r['tail_canary_pass'] and r['bit_exact'] for r in eligibility['checks'])
    assert eligibility['driver_return_codes']==[0] and not eligibility['Triton_imported']
    assert eligibility['payload_SHA256']==c.sha((c.OUT/'compiled_payload.zip').read_bytes())
    originals={}
    samples=0
    workers=[]
    for i in range(1,p['timing']['workers']+1):
        raw=c.read(c.OUT/'timing'/f'invocation_{i}'/'raw.json')
        assert raw['status']=='OK' and raw['driver_return_codes']==[0] and not raw['Triton_imported']
        assert not raw['profiler_attached'] and raw['protocol_SHA256']==c.sha(c.encode(p))
        assert raw['payload_SHA256']==eligibility['payload_SHA256']
        assert raw['environment']['SM_count']==132 and raw['environment']['cc']==[9,0]
        workers.append(raw['environment'])
        for tag,binding in raw['bindings'].items():
            assert binding['CUBIN_SHA256']==binding['loaded_CUBIN_SHA256']==cg['metadata'][tag]['cubin_SHA256']
        grouped=defaultdict(dict)
        for row in raw['cells']:
            assert not cg['metadata'][row['tag']].get('profile_only')
            assert len(row['samples_us'])==12 and row['correctness']['pass']
            assert all(v>0 and math.isfinite(v) for v in row['samples_us'])
            assert len(row['visits'])==3 and all(sorted(v['node_types'])==[0,7,7] for v in row['visits'])
            assert row['samples_us']==[x for v in row['visits'] for x in v['samples_us']]
            samples+=len(row['samples_us'])
            for alias in [row['cell_id']]+row['aliases']:
                tag,qtext,gtext,_=alias.split('|')
                grouped[tag,qtext[1:]][int(gtext[1:])]=st.median(row['samples_us'])
        for (tag,qcond),values in grouped.items():
            g0,g1,g2=sorted(values)
            assert g1-g0==g2-g1
            # Independent equally-spaced OLS identity: middle point changes
            # intercept/residuals but not the endpoint difference slope.
            slope=(values[g2]-values[g0])*132000/(g2-g0)
            capacity=raw['bindings'][tag]['reservations'][qcond]['capacity']
            originals[tag,qcond,i,raw['environment']['gpu_uuid']]=(slope,capacity)
    analysis=c.read(c.OUT/'analysis.json')
    assert samples==analysis['event_samples']
    for row in analysis['oneshot']:
        slope,capacity=originals[row['tag'],row['q_condition'],row['invocation'],row['GPU_UUID']]
        close(slope,row['tile_slope_ns']);assert capacity==row['q']
    diag=c.read(c.OUT/'diagnostics.json')
    assert diag['status']=='OK' and len(diag['layout_tests'])==12 and len(diag['consumer_tests'])==4
    rows=diagnose.unique_services(analysis['oneshot'])
    idx={(r['tag'],r['q'],r['invocation'],r['GPU_UUID']):r for r in rows}
    for row in diag['pairs']:
        m,n,w=row['geometry'];v=row['vec'];key=row['q'],row['worker'],row['GPU_UUID']
        prefix=f'M{m}_N{n}_w{w}:vec{v}:'
        def value(place,policy):return idx[prefix+place+':'+policy+':ONE_FULL',*key]['tile_slope_ns']
        fixed=value('within','fixed')-value('n_first','fixed')
        native=value('within','native')-value('n_first','native')
        close(fixed,row['fixed_gap_ns']);close(native,row['native_gap_ns'])
        close(fixed-native,row['interaction_ns'])
    ct=c.read(c.OUT/'counter_gate.json')
    for row in diag['layout_tests']:
        key=row['layout_key'];expected=row['prediction']
        fixed=row['hardware'][key+':fixed:ONE_FULL']['sectors_per_CTA']
        native=row['hardware'][key+':native:ONE_FULL']['sectors_per_CTA']
        assert row['fixed_sector_prediction_match']==math.isclose(fixed,expected['predicted_fixed_store_sectors_per_CTA'])
        assert row['native_sector_prediction_match']==math.isclose(native,expected['predicted_native_store_sectors_per_CTA'])
    for row in diag['consumer_tests']:
        assert row['prediction_match']==(row['broadcast_LDS']==row['expectation']['predicted_broadcast_whole_LDS'])
    for folder in (c.OUT/'profiles').glob('group_*/attempt_*/export'):
        if c.read(folder/'result.json')['status']!='OK':continue
        for mode in ('software','hardware'):
            for cell in c.read(folder/mode/'job.json')['cells']:
                if gate['metadata'][cell['tag']].get('profile_only'):assert cell['grid']==1
    for record in diag['code']:
        summed=defaultdict(int)
        for pc in record['per_PC']:
            for family in pc['families']:summed[family]+=pc['count']
        for family,value in record['whole_counts'].items():assert summed[family]==value
    assert all(r['all_metrics_and_source_match'] for r in c.read(c.OUT/'offline_import_validation.json')['reports'])
    for record in ct['reports']:
        assert c.sha((c.ROOT/record['path']).read_bytes())==record['SHA256']
    names=('analysis.json','counter_gate.json','diagnostics.json','REPORT.md')
    before={name:(c.OUT/name).read_bytes() for name in names}
    analyze.services();analyze.counters();diagnose.main();report.main()
    assert all((c.OUT/name).read_bytes()==blob for name,blob in before.items())
    if freeze and not (c.OUT/'freeze.json').exists():
        files={str(path.relative_to(c.ROOT)):c.sha(path.read_bytes()) for root in (c.SOURCE,c.OUT)
               for path in sorted(root.rglob('*')) if path.is_file() and '__pycache__' not in path.parts
               and path.name not in ('freeze.json','validation.json')}
        c.write_once(c.OUT/'freeze.json',{'stage':'M3','files':files})
    frozen=c.read(c.OUT/'freeze.json') if (c.OUT/'freeze.json').exists() else None
    if frozen:
        for name,digest in frozen['files'].items():assert c.sha((c.ROOT/name).read_bytes())==digest,name
    if committed:
        assert frozen
        tracked=set(subprocess.check_output(['git','ls-files'],cwd=c.ROOT,text=True).splitlines())
        assert set(frozen['files'])<=tracked
        subprocess.run(['git','diff','--exit-code','HEAD','--',str(c.SOURCE),str(c.OUT)],cwd=c.ROOT,check=True)
    result={'status':'PASS','stage':'M3','programs':34,'timed_programs':30,'profile_only_programs':4,'new_CPU_compiles':12,'reused_CUBINs':22,
        'native_rebuilds':0,'complete_output_correctness_checks':102,'raw_timing_samples':samples,
        'same_worker_interactions':len(diag['pairs']),'conversion_matches':diag['conversion_matches'],'sector_matches':diag['sector_matches'],'consumer_matches':diag['consumer_matches'],'NCU_reports':len(ct['reports']),
        'prior_files_preserved':len(c.read(c.OUT/'preregistration/prior_manifest.json')['files']),
        'worker_GPU_UUIDs':[r['gpu_uuid'] for r in workers],
        'selector_refit':False,'new_blind_claim':False,'D2_started':False,
        'frozen_files':len(frozen['files']) if frozen else 0}
    c.derived(c.OUT/'validation.json',result)
    print('M3 VALIDATION PASS: prior hashes, CUBIN bindings, complete oracles, independent slopes, paired contrasts and offline NCU imports',flush=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--committed',action='store_true')
    args=parser.parse_args();run(args.freeze,args.committed)
