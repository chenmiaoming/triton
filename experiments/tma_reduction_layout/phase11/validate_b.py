"""Independent event/sample/binary/profiler checks and immutable evidence gate."""
import argparse
from collections import Counter
import csv
import io
import json
import math
import statistics
import subprocess
import zipfile
from experiments.tma_reduction_layout.phase11 import common as c
from experiments.tma_reduction_layout.phase11.protocol import load_b
from experiments.tma_reduction_layout.phase11.schedule import schedule
from experiments.tma_reduction_layout.phase11.validate_a import csv_metrics, source_loads


def midpoint(values):
    values = sorted(values)
    i = len(values)//2
    return values[i] if len(values)%2 else (values[i-1]+values[i])/2


def close(a,b):
    assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10), (a,b)


def check_stats(raw,analysis):
    values = {}
    for run in raw:
        groups = {}
        for visit in run['visits']:
            key = (visit['cohort'],visit['group'],visit['fraction'],visit['tag'])
            groups.setdefault(key,[]).extend(visit['samples_us'])
        for key,samples in groups.items():
            values.setdefault(key,[]).append(midpoint(samples))
    for row in analysis['layout_results']:
        group,fraction = row['case_id'],row['fraction']
        gate = c.read(c.OUT/'stage_b/artifact_gate.json')
        for tag,actual in row['worker_medians_us'].items():
            expected = values['gluon_layout',group,fraction,gate['aliases'][tag]]
            assert actual==expected
            close(row['median_of_worker_medians_us'][tag],midpoint(expected))
        for pair in row['pairs']:
            a,b = pair['a'],pair['b']
            logs = [math.log(x/y) for x,y in zip(row['worker_medians_us'][a],row['worker_medians_us'][b])]
            mean = math.fsum(logs)/7
            sd = math.sqrt(math.fsum((v-mean)**2 for v in logs)/6)
            rad = pair['t_critical']*sd/math.sqrt(7)
            close(pair['time_reduction_percent'],100*(1-math.exp(mean)))
            close(pair['time_reduction_interval_percent'][0],100*(1-math.exp(mean+rad)))
            close(pair['time_reduction_interval_percent'][1],100*(1-math.exp(mean-rad)))
    # Independently integrate the df6 PDF to verify the Bonferroni critical.
    critical = analysis['layout_results'][0]['pairs'][0]['t_critical']
    n = 20000
    h = critical/n
    density = lambda x:math.gamma(3.5)/(math.sqrt(6*math.pi)*math.gamma(3))*(1+x*x/6)**-3.5
    integral = h/3*(density(0)+density(critical)+4*sum(density(i*h) for i in range(1,n,2))+
                    2*sum(density(i*h) for i in range(2,n,2)))
    close(.5+integral,1-.05/12)


def check_profiles():
    root = c.OUT/'stage_b/archived_profiles'
    request = c.read(root/'request.json')
    assert request['repeats']==3 and set(request['targets'])=={'sm90'}
    original = c.read(root/'sm90/export/result.json')
    assert original['status']=='OK' and original['formal_timing_sample'] is False
    assert original['request_SHA256']==c.sha(c.encode(request))
    result = {}
    for tag,meta in request['metadata']['sm90'].items():
        repeats = []
        for repeat in range(1,4):
            for group in ('hardware','software'):
                folder = root/'sm90/export'/tag.replace(':','_')/f'repeat_{repeat}'/group
                record = c.read(folder/'collect.json')
                assert record['returncode']==0
                line = next(l for l in record['output'].splitlines() if l.startswith('PROFILE_BINDING '))
                binding = json.loads(line[len('PROFILE_BINDING '):])
                assert binding['CUBIN_SHA256']==meta['cubin_SHA256']
                assert binding['binding']['loaded_CUBIN_SHA256']==meta['cubin_SHA256']
                assert binding['binding']['launches']==binding['binding']['sha_guards']==2
                assert binding['correctness_prefix128'] and binding['compile_calls']==0
                assert not binding['Triton_imported']
                assert binding['environment']['gpu_uuid']==original['environment']['gpu_uuid']
                assert (folder/'report.ncu-rep').stat().st_size>0
            folder = root/'sm90/export'/tag.replace(':','_')/f'repeat_{repeat}'
            hardware = csv_metrics(c.read(folder/'hardware/import.json')['output'])
            loads = source_loads(c.read(folder/'software/source.json')['output'])
            repeats.append({'dynamic_shared_load_warp_instructions':sum(r['dynamic_warp_instructions'] for r in loads),
                'wavefronts':float(hardware['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum'].replace(',','')),
                'bank_conflicts':float(hardware['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum'].replace(',','')),
                'source_loads':loads,'scope':'whole-kernel shared loads; phase attribution is not inferred from totals'})
        result[tag] = repeats
    return {'profiled_binaries':len(result),'reports':len(result)*3*2,'binaries':result,
            'durations_enter_timing_statistics':False,'GPU_UUID':original['environment']['gpu_uuid']}


def derive():
    c.protect_prior()
    protocol,protocol_path = load_b()
    gate = c.read(c.OUT/'stage_b/artifact_gate.json')
    assert gate['environment']['protocol_SHA256']==c.sha(c.encode(protocol))
    assert gate['environment']['native_SHA256']==protocol['baseline']['native_SHA256']
    assert gate['environment']['ccache_before']==gate['environment']['ccache_after']
    snapshot = protocol_path.with_name('source_snapshot.zip') if protocol_path.parent.name!='preregistration' else protocol_path.with_name('worker_source_snapshot.zip')
    with zipfile.ZipFile(snapshot) as z:
        for name,digest in protocol['worker_source_SHA256'].items():
            assert c.sha(z.read(name))==digest,name
    payload = (c.OUT/'stage_b/frozen_payload.zip').read_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for tag,meta in gate['metadata'].items():
            assert c.sha(z.read(tag.replace(':','/')+'/kernel.cubin'))==meta['cubin_SHA256']
    raw = []
    total = 0
    for invocation in range(1,8):
        root = c.OUT/'stage_b/timing'/f'invocation_{invocation}'
        record = c.read(root/'raw.json')
        assert record['status']=='OK' and record['invocation']==invocation
        assert record['protocol_SHA256']==c.sha(c.encode(protocol))
        assert record['payload_SHA256']==c.sha(payload)
        assert record['gate_SHA256']==c.sha(c.encode(gate))
        assert record['call_id']==c.read(root/'dispatch.json')['call_id']
        assert record['environment']['cc']==[9,0] and 'H100' in record['environment']['name']
        assert record['compilation']=={'Triton_imported':False,'subprocess_calls':0,'compile_calls':0}
        assert record['CUDA_return_codes']==[0]
        warmups = record['warmups']
        assert len(warmups)==len(gate['metadata'])
        assert all(w['count']==3 and w['correctness']['pass'] for w in warmups)
        expected = []
        for cohort,groups in [('gluon_layout',gate['gluon_groups']),('canonical_bridge',gate['bridge_groups'])]:
            slots = schedule(groups,invocation)
            assert record[cohort+'_schedule_SHA256']==c.sha(c.encode(slots))
            expected += [{**slot,'cohort':cohort} for slot in slots]
        assert len(record['visits'])==len(expected)
        counts = Counter()
        for visit,slot in zip(record['visits'],expected):
            assert all(visit[k]==v for k,v in slot.items())
            meta = gate['metadata'][visit['tag']]
            assert visit['grid']==meta['full_grid']//visit['fraction']
            assert visit['gpu_uuid']==record['environment']['gpu_uuid']
            assert len(visit['samples_us'])==len(visit['launch_SHA256'])==5
            assert all(math.isfinite(x) and x>0 for x in visit['samples_us'])
            assert set(visit['launch_SHA256'])=={meta['cubin_SHA256']}
            assert visit['binding']['CUBIN_SHA256']==visit['binding']['loaded_CUBIN_SHA256']==meta['cubin_SHA256']
            counts[visit['tag']] += 5
        samples = sum(counts.values())
        assert record['CUDA_call_counts']['cuLaunchKernel']==samples+3*len(warmups)
        assert record['CUDA_call_counts']['cuEventRecord']==samples*2
        assert record['CUDA_call_counts']['cuEventElapsedTime']==samples
        assert record['CUDA_call_counts']['cuEventSynchronize']==samples
        for tag,binding in record['binaries'].items():
            assert binding['launches']==binding['sha_guards']==counts[tag]+3
        total += samples
        raw.append(record)
    assert len({r['call_id'] for r in raw})==7
    analysis = c.read(c.OUT/'stage_b/analysis.json')
    check_stats(raw,analysis)
    assert analysis['raw_timing_samples']==total and analysis['stop_after_B']
    profiles = check_profiles()
    accepted = c.read(c.OUT/'stage_b/artifacts'/gate['accepted_attempt']/'export/attempts.json')
    compiler_events = [event for row in accepted for event in row['compiler_cache_events']]
    return {'stage':'B','status':'PASS','logical_artifacts':len(accepted),'full_output_correctness_checks':len(accepted)*3,
            'timing_invocations':7,'timing_samples':total,'raw_timing_SHA256':[
                c.sha((c.OUT/'stage_b/timing'/f'invocation_{i}/raw.json').read_bytes()) for i in range(1,8)],
            'kernel_cache_hits':sum(e['cache_hit'] for e in compiler_events),'kernel_cache_events':len(compiler_events),
            'native_rebuilds':0,'old_score_model_used':False,'profile_validation':profiles,
            'prior_files_protected':len(c.read(c.OUT/'stage_a/preregistration/prior_manifest.json')['files']),
            'C_and_D_executed':False,'PR_created':False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--committed',action='store_true')
    args = parser.parse_args()
    result = derive()
    root = c.OUT/'stage_b'
    if args.freeze:
        manifest = {n:d for n,d in c.inventory(root).items() if n not in
                    ('raw_manifest.json','validation.json','analysis.json','summary.md')}
        c.write_once(root/'raw_manifest.json',{'files':manifest})
        c.write_once(root/'validation.json',result)
    else:
        assert c.read(root/'validation.json')==result
    manifest = c.read(root/'raw_manifest.json')['files']
    derived = {'raw_manifest.json','validation.json','analysis.json','summary.md','interpretation.md'}
    assert set(c.inventory(root))-derived==set(manifest), 'Unregistered raw evidence file'
    for name,digest in manifest.items():
        assert c.sha((root/name).read_bytes())==digest,name
    if args.committed:
        for name,digest in manifest.items():
            path = (root/name).relative_to(c.ROOT).as_posix()
            assert c.sha(subprocess.check_output(['git','show','HEAD:'+path],cwd=c.ROOT))==digest,path
    print('Stage B PASS:',result['logical_artifacts'],'artifacts,',result['timing_samples'],
          'single-kernel samples,',result['profile_validation']['reports'],'separate NCU reports; stop after B')


if __name__=='__main__':
    main()
