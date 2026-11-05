"""Outcome-independent local A10 validation and paired temporal analysis."""
import argparse
import gzip
from pathlib import Path

import numpy as np

from experiments.tma_reduction_layout.phase11.operator_gpu import common as c
from .a10_prepare import DEFAULT,UUID,cpu_gate


def results(root,eligibility=False):
    for invocation in ([0] if eligibility else [1,2,3]):
        folder=Path(root)/('check/invocation_0' if invocation==0 else f'timing/invocation_{invocation}')
        yield folder,c.read(folder/'result.json')


def validate(root,eligibility=False):
    root=Path(root)
    p=c.read(root/'preregistration/protocol.json')
    cpu=cpu_gate(p,(root/'preregistration/payload.zip').read_bytes(),root/'snapshot')
    programs=p['suites']['a10']['programs']
    expected_contexts={panel+':'+geom for panel in ('layout','versions')
        for geom in {x['geometry'] for x in programs}}
    expected_by_tag={x['tag']:x for x in programs}
    total=0;contexts=0;proofs=0;checks=0;pids=[];sms=[];telemetry_rows=0
    for folder,r in results(root,eligibility):
        assert r['status']=='OK' and not r['cleanup_errors'] and not r['partial_errors']
        assert r['protocol_SHA256']==c.sha(c.encode(p))
        assert r['payload_SHA256']==p['suites']['a10']['payload_SHA256']
        assert r['environment']['gpu_uuid']==UUID and r['environment']['cc']==[8,6]
        assert r['environment']['name']=='NVIDIA A10'
        assert r['environment']['CUDA_VISIBLE_DEVICES']==UUID
        pids.append(r['environment']['PID']);sms.append(r['environment']['SM_count'])
        assert r['driver_return_codes']==[0] and not r['Triton_imported']
        assert r['compile_calls']==r['native_rebuilds']==0 and not r['profiler_attached']
        idle=c.read(folder/'idle_before_context.json')
        assert idle['status']=='PASS' and idle['gpu_uuid']==UUID
        assert all(q['returncode']==0 for q in idle['queries'])
        if not eligibility:
            binding=c.read(folder/'eligibility_binding.json')
            assert binding['eligibility_gate_SHA256']==c.sha((root/'check/eligibility_gate.json').read_bytes())
        assert len(r['process_boundaries'])==8
        assert all(row['pass'] and row['returncode']==0 and row['PID']==r['environment']['PID']
                   for row in r['process_boundaries'])
        assert len(r['bindings'])==28
        bytag={x['tag']:x for x in r['bindings']}
        assert set(bytag)==set(expected_by_tag)
        for program in programs:
            b=bytag[program['tag']]
            assert b['CUBIN_SHA256']==b['loaded_SHA256']==program['metadata']['CUBIN_SHA256']
            assert b['ABI_types']==program['ABI'] and b['attributes']['n_regs']==program['metadata']['num_regs']
            assert b['attributes']['local_bytes']==0
            replays=2 if eligibility else 2+10*(3+31*(2 if program['role']=='U' else 1))
            assert b['captures']==1 and b['graph_launches']==replays*9
            assert b['launches']==10 and b['sha_guards']==(2 if eligibility else 12)
        for pair in p['suites']['a10']['G0_equivalence']:
            assert bytag[pair['U']]['attributes']==bytag[pair['G0']]['attributes']
            assert bytag[pair['U']]['ABI_types']==bytag[pair['G0']]['ABI_types']
        assert len(r['shards'])==4 and len(r['graph_proofs'])==28
        assert {row['context']['id'] for row in r['shards']}==expected_contexts
        assert {row['tag'] for row in r['graph_proofs']}==set(expected_by_tag)
        for proof in r['graph_proofs']:
            assert sorted(proof['node_types'])==[0]*9+[7,7]
            assert proof['primes']==8 and proof['expected_nodes_per_semantic_iteration']==dict(kernels=1,memsets=0,copies=0)
            original=expected_by_tag[proof['tag']]
            assert proof['role']==original['role']
            assert proof['replays']==(2 if eligibility else 2+10*(3+31*(2 if original['role']=='U' else 1)))
            proofs+=1
        assert len(r['checks'])==88
        assert {(x['tag'],x['phase']) for x in r['checks'] if x['phase']!='readonly_input_after'}=={
            (tag,phase) for tag in expected_by_tag for phase in ('before','graph_before','graph_after')}
        for check in r['checks']:
            assert check['pass'],check
            if check['phase']!='readonly_input_after':
                assert check['all_output_elements']==2097152 and check['canaries_pass']
                assert check['finite'] and check['exact_equal'] and check['output_SHA256']==check['expected_SHA256']
            checks+=1
        for record in r['shards']:
            path=folder/record['path'];assert c.sha(path.read_bytes())==record['SHA256']
            shard=c.read(path);assert 'incomplete_visit' not in shard
            context=shard['context']
            selected=[x for x in programs if x['panel']==context['panel'] and x['geometry']==context['geometry']]
            expected_cells={x['tag']:x['role'] for x in selected}
            expected_cells.update({x['tag']+':repeat':'U_repeat' for x in selected if x['role']=='U'})
            case=selected[0]['metadata']['case']
            assert [context[k] for k in ('M','N','W','grid')]==[case[k] for k in ('BM','BN','num_warps','grid')]
            bycell={x['tag']:x for x in shard['cells']}
            assert len(bycell)==len(shard['cells']) and set(bycell)==set(expected_cells)
            for cell in bycell.values():
                assert cell['role']==expected_cells[cell['tag']]
                assert len(cell['visits'])==(0 if eligibility else 10)
                assert [v['visit'] for v in cell['visits']]==([] if eligibility else list(range(10)))
                if cell['sham_of']:
                    assert cell['role']=='U_repeat' and cell['sham_of'] in bycell
                for visit in cell['visits']:
                    assert len(visit['samples_us'])==31
                    assert all(np.isfinite(v) and v>0 for v in visit['samples_us'])
                    assert all(v>=1_000_000 for v in visit['host_gaps_ns'])
                    assert all(end>start for start,end in zip(visit['launch_ns'],visit['completion_ns']))
                    total+=31
            if not eligibility:
                assert len(shard['orders'])==10
                assert [v['visit'] for v in shard['orders']]==list(range(10))
                for order in shard['orders']:
                    assert len(order['round_orders'])==31
                    assert all(set(x)==set(bycell) and len(x)==len(bycell) for x in order['round_orders'])
                    previous=None
                    for sample,tags in enumerate(order['round_orders']):
                        for tag in tags:
                            visit=bycell[tag]['visits'][order['visit']]
                            launch=visit['launch_ns'][sample];completion=visit['completion_ns'][sample]
                            assert visit['start_ns']<=launch<completion<=visit['end_ns']
                            if previous is not None:
                                assert launch-previous==visit['host_gaps_ns'][sample]>=1_000_000
                            previous=completion
            contexts+=1
        trace=folder/r['telemetry']['path']
        assert c.sha(trace.read_bytes())==r['telemetry']['SHA256']
        import json
        rows=[json.loads(line) for line in gzip.decompress(trace.read_bytes()).splitlines()]
        assert rows and len(rows)==r['telemetry']['rows']
        # Missing hardware fields remain explicit; do not replace them with zero.
        for name in ('sm_clock_mhz','memory_clock_mhz','temperature_c','power_usage_mw'):
            assert any(row.get(name,{}).get('rc')==0 for row in rows),name
        telemetry_rows+=len(rows)
    assert len(set(pids))==len(pids) and len(set(sms))==1
    assert total==(0 if eligibility else 33480)
    return dict(status='PASS',protocol_SHA256=c.sha(c.encode(p)),
        scope='Eligibility graph/correctness only' if eligibility else 'Complete frozen local A10 matrix; no favorable timing requirement',
        physical_GPU_count=1,process_cohorts=len(pids),PIDs=pids,gpu_uuid=UUID,SM_count=sms[0],
        contexts=contexts,graph_proofs=proofs,checks=checks,raw_samples=total,
        telemetry_rows=telemetry_rows,cpu_gate=cpu)


def analyze(root):
    root=Path(root);p=c.read(root/'preregistration/protocol.json')
    cube={}
    for folder,result in results(root):
        for record in result['shards']:
            shard=c.read(folder/record['path'])
            key=shard['context']['id']
            for cell in shard['cells']:
                cube.setdefault((key,cell['tag']),[]).append([
                    float(np.median(v['samples_us'])) for v in cell['visits']])
    rng=np.random.default_rng(p['timing']['bootstrap_seed'])
    draws=p['timing']['bootstrap_draws']
    cohorts=rng.integers(0,3,size=(draws,3))
    visits=rng.integers(0,10,size=(draws,3,10))

    def compare(key,a,b):
        av=np.asarray(cube[(key,a)]);bv=np.asarray(cube[(key,b)])
        assert av.shape==bv.shape==(3,10)
        # Every resample preserves candidate/control process and visit pairing.
        ratio=av/bv
        boots=ratio[cohorts[:,:,None],visits].mean(axis=(1,2))
        low,high=np.quantile(boots,[.025,.975])
        point=float(ratio.mean())
        return dict(candidate=a,reference=b,ratio=point,ratio_CI95=[float(low),float(high)],
            gain_percent=100*(1-point),gain_CI95_percent=[100*(1-float(high)),100*(1-float(low))],
            practical_faster=bool(high<.97), practical_slower=bool(low>1.03),
            equivalent_within_3percent=bool(low>=.97 and high<=1.03),
            candidate_median_us=float(np.median(av)),reference_median_us=float(np.median(bv)),
            per_cohort_gain_percent=[float(100*(1-x.mean())) for x in ratio])

    rows=[];controls=[];versions=[]
    programs=p['suites']['a10']['programs']
    for geometry in sorted({x['geometry'] for x in programs}):
        key='layout:'+geometry
        selected=[x for x in programs if x['panel']=='layout' and x['geometry']==geometry]
        u=next(x['tag'] for x in selected if x['role']=='U')
        g=next(x['tag'] for x in selected if x['role']=='G0')
        controls.extend([dict(context=key,kind='U_repeat',**compare(key,u+':repeat',u)),
                         dict(context=key,kind='G0_U',**compare(key,g,u))])
        for x in selected:
            if x['role']!='candidate':continue
            primary=x['metadata']['case']['path']=='direct'
            ownbase=g if primary else geometry+':cp_async:vec8'
            row=dict(context=key,tag=x['tag'],scope='direct_layout' if primary else 'cp_async_changed_path',
                vs_U=compare(key,x['tag'],u),vs_G0=compare(key,x['tag'],g),
                within_path=compare(key,x['tag'],ownbase))
            row['certified_vs_U_and_G0']=bool(primary and row['vs_U']['practical_faster'] and
                row['vs_G0']['practical_faster'] and all(cn['equivalent_within_3percent']
                for cn in controls if cn['context']==key))
            rows.append(row)
        key='versions:'+geometry
        for version in ('3.5.1','3.6.0','3.7.1'):
            tag=version+':'+geometry
            controls.append(dict(context=key,kind='release_U_repeat',**compare(key,tag+':repeat',tag)))
            if version!='3.5.1':
                comparison=compare(key,tag,'3.5.1:'+geometry)
                shams=[compare(key,current+':'+geometry+':repeat',current+':'+geometry)
                       for current in ('3.5.1',version)]
                valid=all(row['equivalent_within_3percent'] for row in shams)
                versions.append(dict(context=key,version=version,**comparison,
                    controls_equivalent=valid,certified_regression=bool(valid and comparison['practical_slower']),
                    interpretation='Control-qualified' if valid else 'Inconclusive: at least one release U/U_repeat control failed equivalence'))
    return dict(status='COMPLETE',single_physical_device=True,temporal_processes=3,
        metric='Mean paired ratio of per-visit median single-target GPU event times',
        bootstrap=dict(draws=draws,seed=p['timing']['bootstrap_seed'],unit='process then visits, paired'),
        controls=controls,layout_comparisons=rows,release_comparisons=versions,
        caveat='Read-only nominal clocks are sampled, not locked and not per-kernel frequency proof. Old/new changes are not uniquely attributable to cadence.')


def report(root):
    root=Path(root)
    validation=validate(root)
    summary=analyze(root)
    c.write_once(root/'validation.json',validation)
    c.write_once(root/'analysis.json',summary)
    lines=['# Local A10 cadence remeasurement','',
        'Reused 28 original archived CUBINs without compilation. Three sequential fresh processes ran on **one physical A10 GPU0**, with eight in-graph priming iterations and a fixed minimum 1 ms host gap before every graph. Exactly one final kernel is inside each timing interval. All 33,480 observations are retained.','',
        'All full outputs, captured graph outputs before/after, read-only input digests and byte guards passed. The latest direct-load canonical U and Gluon direct vec8 G0 have identical allocated ELF, pointer ABI and compiled resources for both shapes. Older releases have U-only controls; no older-release G0 is fabricated.','',
        'Intervals below resample paired process/visit medians. They describe temporal repetition on one GPU. cp_async changes the memory path and is a secondary within-path comparison.','',
        '| Context | Control | Gain % | 95% CI % | Within ±3% |',
        '| --- | --- | ---: | --- | --- |']
    for row in summary['controls']:
        lo,hi=row['gain_CI95_percent']
        lines.append(f"| {row['context']} | {row['kind']} | {row['gain_percent']:.2f} | [{lo:.2f}, {hi:.2f}] | {row['equivalent_within_3percent']} |")
    lines+=['','| Context / candidate | Path | U median µs | Candidate median µs | Gain vs U % (95% CI) | Gain vs G0 % | Certified direct gain |',
        '| --- | --- | ---: | ---: | --- | ---: | --- |']
    for row in summary['layout_comparisons']:
        r=row['vs_U'];lo,hi=r['gain_CI95_percent']
        lines.append(f"| {row['tag']} | {row['scope']} | {r['reference_median_us']:.3f} | {r['candidate_median_us']:.3f} | {r['gain_percent']:.2f} [{lo:.2f}, {hi:.2f}] | {row['vs_G0']['gain_percent']:.2f} | {row['certified_vs_U_and_G0']} |")
    lines+=['','| Geometry | Release vs 3.5.1 | Time ratio (95% CI) | >3% regression |',
        '| --- | --- | --- | --- |']
    for r in summary['release_comparisons']:
        lo,hi=r['ratio_CI95']
        lines.append(f"| {r['context']} | {r['version']} | {r['ratio']:.5f} [{lo:.5f}, {hi:.5f}] | {r['certified_regression']} ({r['interpretation']}) |")
    lines+=['',summary['caveat'],
        'The frozen legacy data remain unchanged. These results do not refit any frozen H100 model or turn post-D1 diagnostics into a new blind test. No conclusion about independent A10 devices or Blackwell follows from this local suite.','']
    c.write_once(root/'REPORT.md','\n'.join(lines).encode())
    print('A10 VALIDATED',validation['raw_samples'],'samples',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=DEFAULT)
    parser.add_argument('--eligibility',action='store_true')
    args=parser.parse_args()
    if args.eligibility:print(validate(args.root,True))
    else:report(args.root)
