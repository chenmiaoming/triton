"""All-layout paired training analysis, counter re-import and bounded scientific claims."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse,csv,io,json,re
import numpy as np
from . import common as c
from .g3b_prepare import OUT,PROTOCOL_FILE
from .g3_analyze import imported
from experiments.tma_reduction_layout.phase11.c2.analyze import hardware_rows,source_blocks,FAMILIES


def save_derived(path,value):
    blob=c.encode(value) if not isinstance(value,bytes) else value
    if path.exists():assert path.read_bytes()==blob,path
    else:c.write_once(path,blob)


def timing():
    p=c.read(OUT/'preregistration'/PROTOCOL_FILE);c.verify_sources(p)
    records=c.read(OUT/'cpu/attempt_002/export/result.json')['records'];bytag={r['tag']:r for r in records}
    paired=defaultdict(dict);sample_count=0;physical=0;env=[];sources=[]
    for i in range(1,4):
        path=OUT/f'timing/invocation_{i}/raw.json';raw=c.read(path)
        assert raw['status']=='OK' and not raw['Triton_imported'] and raw['driver_return_codes']==[0]
        assert raw['protocol_SHA256']==c.sha(c.encode(p))
        assert raw['payload_SHA256']==c.sha((OUT/'compiled_payload.zip').read_bytes())
        assert all(x['pass'] for x in raw['checks'])
        env.append(raw['environment']);sources.append({'path':str(path.relative_to(c.ROOT)),'SHA256':c.sha(path.read_bytes())})
        for cell in raw['cells']:
            assert len(cell['visits'])==5
            for visit in cell['visits']:
                assert len(visit['samples_us'])==31 and all(np.isfinite(t) and t>0 for t in visit['samples_us'])
                assert sorted(visit['node_types'])==([0,7,7] if cell['mode']=='merge' else [0,0,2,2,2,7,7])
                sample_count+=len(visit['samples_us'])
            values=[float(np.median(v['samples_us'])) for v in cell['visits']]
            for tag in [cell['tag']]+cell['aliases']:
                paired[(cell['geometry'],cell['mode'],cell['M'],tag)][i]=values
            physical+=1
        for tag,binding in raw['bindings'].items():assert binding['cubin_SHA256']==binding['loaded_SHA256']==bytag[tag]['cubin_SHA256']
    arrays={k:np.asarray([v[i] for i in range(1,4)]) for k,v in paired.items()}
    rows=[]
    for work in sorted({k[:3] for k in arrays}):
        u=arrays[(*work,work[0]+':upstream')];g0=arrays[(*work,work[0]+':gluon_anchor')]
        rng=np.random.default_rng(20261022+int(c.sha(c.encode(work))[:8],16))
        wi=rng.integers(0,3,(4096,3));vi=rng.integers(0,5,(4096,3,5))
        def resample(x):return x[wi[:,:,None],vi].mean((1,2))
        ub,gb=resample(u),resample(g0)
        for key in sorted(k for k in arrays if k[:3]==work):
            v=arrays[key];vb=resample(v)
            rows.append({'geometry':work[0],'mode':work[1],'M':work[2],'tag':key[-1],
                'time_us':float(v.mean()),'worker_visit_medians_us':v.tolist(),
                'gain_vs_U_percent':float(100*(u.mean()-v.mean())/u.mean()),
                'gain_vs_G0_percent':float(100*(g0.mean()-v.mean())/g0.mean()),
                'gain_vs_U_CI95':np.percentile(100*(ub-vb)/ub,[2.5,97.5]).tolist(),
                'gain_vs_G0_CI95':np.percentile(100*(gb-vb)/gb,[2.5,97.5]).tolist()})
    result={'status':'OK','raw_samples':sample_count,'physical_worker_cells':physical,'logical_conditions':len(rows),
            'workloads':9,'workers':env,'sources':sources,'rows':rows,
            'estimator':'mean of worker/visit medians; paired worker/visit bootstrap 4096; three workers, five visits, 31 samples',
            'scope':'exploratory pointwise CI; no multiplicity correction; best-in-sample is not a frozen selector'}
    save_derived(OUT/'timing_analysis.json',result)
    stream=io.StringIO();fields=list(rows[0]);writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    save_derived(OUT/'all_timings.csv',stream.getvalue().encode())
    print('G3b timing PASS',sample_count,'samples;',len(rows),'logical conditions')


def counters():
    rows=[];reports=[];jobs=[]
    for directory,pfile in [('profiles','profile_protocol.json'),('profiles_extension','profile_extension_protocol.json')]:
        p=c.read(OUT/'preregistration'/pfile);c.verify_sources(p)
        for group in p['groups']:
            root=OUT/directory/group['case']['id']/'attempt_001/export';result=c.read(root/'result.json')
            assert result['status']=='OK' and result['protocol_SHA256']==c.sha(c.encode(p))
            assert result['payload_SHA256']==c.sha((OUT/'compiled_payload.zip').read_bytes())
            for r in result['reports']:
                folder=root/Path(r['path']).parent
                assert c.sha((folder/'report.ncu-rep').read_bytes())==r['SHA256']
                binds=re.findall(r'PROFILE_BINDING (\{[^\n]+\})',c.read(folder/'collect.json')['output']);assert binds
                b=json.loads(binds[-1]);assert not b['Triton_imported'] and b['compile_calls']==0 and b['driver_return_codes']==[0]
                item=b['records'][0];assert item['correctness']['pass']
                assert item['binding']['loaded_SHA256']==item['binding']['cubin_SHA256']
                mt=hardware_rows(c.read(folder/'import.json')['output']);assert len(mt)==1
                row={**r,'geometry':group['case']['id'],'resources':item['binding']['attributes'],
                     'CTAs':int(np.prod(item['grid'])),'metrics':mt[0],'environment':b['environment']}
                if r['mode']=='software':
                    pc=source_blocks(c.read(folder/'source.json')['output']);assert len(pc)==1
                    row['per_PC']=pc[0];row['counts']={f:sum(x['count'] for x in pc[0].values() if f in x['families']) for f in FAMILIES}
                rows.append(row);jobs.append((folder,r['mode']))
                reports.append({'path':str((folder/'report.ncu-rep').relative_to(c.ROOT)),'SHA256':r['SHA256'],'mode':r['mode']})
    # G3a's helper writes CPU imports under its own OUT; pass a local implementation instead.
    import subprocess
    def one(args):
        folder,mode=args;target=OUT/'offline_imports'/(folder.relative_to(OUT).as_posix().replace('/','_')+'.json')
        cmds=[['/usr/local/cuda/bin/ncu','--import',str(folder/'report.ncu-rep'),'--page','raw','--csv','--print-units','base','--print-fp']]
        if mode=='software':cmds.append(['/usr/local/cuda/bin/ncu','--import',str(folder/'report.ncu-rep'),'--page','source','--print-source','sass','--csv'])
        if target.exists():value=c.read(target)
        else:
            outputs=[]
            for argv in cmds:
                p=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
                assert p.returncode==0,p.stdout[-2000:];outputs.append({'argv':argv,'returncode':p.returncode,'output':p.stdout})
            value={'CPU_only':True,'report_SHA256':c.sha((folder/'report.ncu-rep').read_bytes()),'commands':outputs};c.write_once(target,value)
        assert hardware_rows(value['commands'][0]['output'])==hardware_rows(c.read(folder/'import.json')['output'])
        if mode=='software':assert source_blocks(value['commands'][1]['output'])==source_blocks(c.read(folder/'source.json')['output'])
        return {'path':str(target.relative_to(c.ROOT)),'values_identical':True}
    with ThreadPoolExecutor(max_workers=4) as pool:imports=list(pool.map(one,jobs))
    save_derived(OUT/'counter_analysis.json',{'status':'OK','rows':rows,'reports':reports,'offline_imports':imports})
    print('G3b counters PASS',len(rows),'reports; all locally re-imported')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['timing','counters']);a=p.parse_args()
    (timing if a.action=='timing' else counters)()
