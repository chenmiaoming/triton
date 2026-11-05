"""Reproducible paired analysis; no fitting or selection-model updates."""
from collections import defaultdict
import csv
import io
from pathlib import Path
import re
import numpy as np
from . import common as c


def timing_analysis(write=True):
    protocol=c.read(c.OUT/'stage_g2/preregistration/protocol.json')
    compiled=c.read(c.OUT/'stage_g2/compile_gate.json')
    expected_samples=protocol['timing']['visits']*protocol['timing']['samples_per_visit']
    paired=defaultdict(dict)
    environments=[]
    sources=[]
    sample_count=0
    physical_cells=0
    for worker in range(1,protocol['timing']['workers']+1):
        path=c.OUT/'stage_g2/timing'/f'invocation_{worker}'/'raw.json'
        raw=c.read(path)
        assert raw['status']=='OK' and not raw['Triton_imported'] and raw['driver_return_codes']==[0]
        assert raw['protocol_SHA256']==c.sha(c.encode(protocol))
        assert raw['payload_SHA256']==c.sha((c.OUT/'stage_g2/compiled_payload.zip').read_bytes())
        environments.append({'worker':worker,**raw['environment']})
        sources.append({'path':path.relative_to(c.ROOT).as_posix(),'SHA256':c.sha(path.read_bytes())})
        for cell in raw['cells']:
            assert cell['complete_output_check']['pass']
            assert len(cell['samples_us'])==expected_samples
            assert all(value>0 and np.isfinite(value) for value in cell['samples_us'])
            assert cell['samples_us']==[value for visit in cell['visits'] for value in visit['samples_us']]
            assert len(cell['visits'])==protocol['timing']['visits']
            for visit in cell['visits']:
                assert len(visit['samples_us'])==protocol['timing']['samples_per_visit']
                kernels=2 if cell['mode']=='decode' else 1
                assert sorted(visit['node_types'])==[0]*kernels+[7,7]
            values=[float(np.median(visit['samples_us'])) for visit in cell['visits']]
            for tag in [cell['tag']]+cell['aliases']:
                key=(cell['geometry'],cell['batch'],cell['length'],cell['mode'],tag)
                assert worker not in paired[key]
                paired[key][worker]=values
            sample_count+=len(cell['samples_us'])
            physical_cells+=1
        for tag,binding in raw['bindings'].items():
            row=next(row for row in compiled['records'] if row['tag']==tag)
            assert binding['cubin_SHA256']==binding['loaded_SHA256']==row['cubin_SHA256']
    workers=list(range(1,protocol['timing']['workers']+1))
    arrays={key:np.asarray([values[w] for w in workers]) for key,values in paired.items()}
    assert all(set(values)==set(workers) for values in paired.values())
    grouped=defaultdict(list)
    for key in arrays:
        grouped[key[:4]].append(key)
    rows=[]
    for work,keys in sorted(grouped.items()):
        geometry,batch,length,mode=work
        u=arrays[(*work,geometry+':upstream')]
        g0=arrays[(*work,geometry+':gluon_anchor')]
        seed=20261021+int(c.sha(c.encode(list(work)))[:8],16)
        rng=np.random.default_rng(seed)
        wi=rng.integers(0,len(workers),(4096,len(workers)))
        vi=rng.integers(0,u.shape[1],(4096,len(workers),u.shape[1]))
        def resampled(value):
            return value[wi[:,:,None],vi].mean((1,2))
        ub,gb=resampled(u),resampled(g0)
        for key in sorted(keys):
            value=arrays[key]
            sample=resampled(value)
            gain=100*(u.mean()-value.mean())/u.mean()
            gain0=100*(g0.mean()-value.mean())/g0.mean()
            gain_ci=np.percentile(100*(ub-sample)/ub,[2.5,97.5]).tolist()
            g0_ci=np.percentile(100*(gb-sample)/gb,[2.5,97.5]).tolist()
            rows.append({'geometry':geometry,'batch':batch,'length':length,'mode':mode,'tag':key[-1],
                         'mean_visit_median_us':float(value.mean()),'worker_means_us':value.mean(1).tolist(),
                         'worker_visit_medians_us':value.tolist(),
                         'gain_vs_upstream_percent':float(gain),'gain_vs_upstream_CI95':gain_ci,
                         'gain_vs_anchor_percent':float(gain0),'gain_vs_anchor_CI95':g0_ci,
                         'worker_gains_vs_upstream_percent':(100*(u.mean(1)-value.mean(1))/u.mean(1)).tolist(),
                         'worker_gains_vs_anchor_percent':(100*(g0.mean(1)-value.mean(1))/g0.mean(1)).tolist(),
                         'exploratory_practical_win_vs_U':gain_ci[0]>3,
                         'exploratory_practical_win_vs_G0':g0_ci[0]>3,
                         'exploratory_practical_loss_vs_U':gain_ci[1]<-3})
    result={'status':'OK','raw_samples':sample_count,'physical_worker_cells':physical_cells,
            'physical_conditions':physical_cells//len(workers),'logical_conditions':len(rows),
            'workload_groups':len(grouped),'workers':environments,'sources':sources,'rows':rows,
            'estimator':'mean of worker/visit medians of 31 samples; paired worker/visit bootstrap 4096',
            'CI_scope':'exploratory pointwise intervals, not multiple-comparison-adjusted or held-out selector evidence',
            'main_units':'microseconds per single-call CUDA Graph',
            'no_selector_trained':True}
    stream=io.StringIO()
    fields=['geometry','batch','length','mode','tag','mean_visit_median_us','gain_vs_upstream_percent',
            'gain_vs_anchor_percent','gain_vs_upstream_CI95','gain_vs_anchor_CI95']
    writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field:row[field] for field in fields})
    encoded_csv=stream.getvalue().encode()
    if write:
        c.write_once(c.OUT/'stage_g2/timing_analysis.json',result)
        c.write_once(c.OUT/'stage_g2/all_timings.csv',encoded_csv)
        print('TIMING ANALYSIS OK:',sample_count,'raw samples;',len(rows),'logical conditions;',len(grouped),'workloads')
    return result,encoded_csv


def counter_analysis(write=True):
    from experiments.tma_reduction_layout.phase11.c2.analyze import source_blocks,hardware_rows,FAMILIES
    protocol=c.read(c.OUT/'stage_g2/preregistration/protocol.json')
    compiled=c.read(c.OUT/'stage_g2/compile_gate.json')
    bytag={row['tag']:row for row in compiled['records']}
    result={'status':'OK','reports':[],'dynamic':[],'hardware':[],'NCU_duration_used_for_timing':False}
    for case in protocol['geometries']:
        attempt=3 if case['family']=='C02' else 2
        root=c.OUT/'stage_g2/profiles'/case['id']/f'attempt_{attempt:03}/export'
        raw=c.read(root/'result.json')
        assert raw['status']=='OK',case['id']
        assert raw['protocol_SHA256']==c.sha(c.encode(protocol))
        assert raw['payload_SHA256']==c.sha((c.OUT/'stage_g2/compiled_payload.zip').read_bytes())
        profile_protocol=c.read(c.OUT/f'stage_g2/preregistration/profile_protocol_v{attempt}.json')
        assert raw['profile_protocol_SHA256']==c.sha(c.encode(profile_protocol))
        for mode in ('software','hardware'):
            folder=root/mode
            job=c.read(folder/'job.json')
            collect=c.read(folder/'collect.json')
            marker=re.findall(r'PROFILE_BINDING (\{[^\n]+\})',collect['output'])
            assert marker,case['id']
            import json
            binding=json.loads(marker[-1])
            assert not binding['Triton_imported'] and binding['compile_calls']==0 and binding['driver_return_codes']==[0]
            assert len(binding['records'])==len(job['metadata'])
            report=folder/'report.ncu-rep'
            result['reports'].append({'path':report.relative_to(c.ROOT).as_posix(),'SHA256':c.sha(report.read_bytes()),
                                      'mode':mode,'launches':len(binding['records']),'environment':binding['environment']})
            imported=hardware_rows(c.read(folder/'import.json')['output'])
            assert len(imported)==len(binding['records'])
            pcs=source_blocks(c.read(folder/'source.json')['output']) if mode=='software' else None
            if pcs is not None:
                assert len(pcs)==len(binding['records'])
            for index,row in enumerate(binding['records']):
                tag=row['tag']
                assert row['correctness']['pass']
                assert row['binding']['cubin_SHA256']==row['binding']['loaded_SHA256']==bytag[tag]['cubin_SHA256']
                common={'tag':tag,'geometry':case['id'],'grid':row['grid'],
                        'GPU_UUID':binding['environment']['gpu_uuid'],'resources':row['binding']['attributes']}
                if mode=='software':
                    counts={family:sum(value['count'] for value in pcs[index].values() if family in value['families']) for family in FAMILIES}
                    counts['global_load']=sum(value['count'] for value in pcs[index].values() if re.search(r'\bLDG(?:\.|\s)',value['instruction']))
                    result['dynamic'].append({**common,'counts':counts,'per_PC':pcs[index],
                        'scope':'single CTA; SG context2048/splits4 => 8 KV iterations, not one PV operation'})
                else:
                    metrics=imported[index]
                    for name in protocol['profiles']['metrics']:
                        assert name in metrics,(tag,name)
                    result['hardware'].append({**common,'metrics':metrics,
                        'CTAs':int(np.prod(row['grid'])),'batch':8,'length':binding['length']})
    if write:
        c.write_once(c.OUT/'stage_g2/counter_analysis.json',result)
        print('COUNTERS OK:',len(result['reports']),'reports;',len(result['dynamic']),'single-CTA and',len(result['hardware']),'full-grid launches')
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['timing','counters'])
    args=parser.parse_args()
    (timing_analysis if args.action=='timing' else counter_analysis)()


if __name__=='__main__':
    main()
