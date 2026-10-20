"""Matched static plans, full-kernel contrasts and retained-PC provenance."""
from collections import defaultdict
import math
import re
import statistics as st
import subprocess
from experiments.tma_reduction_layout.phase11.m1 import common as c
from experiments.tma_reduction_layout.phase11.m1 import pipeline
from experiments.tma_reduction_layout.phase11.c3.static import plan


def provenance(bundle,cohort,tag):
    gate=bundle['gates'][cohort];cg=bundle['compiles'][cohort]
    counter=bundle['counters'][cohort];origin=counter['profile_aliases'].get(tag,tag)
    dynamic=next(r for r in counter['single_pass'] if r['tag']==origin)
    artifact=c.ROOT/cg['artifact_directory']/tag.replace(':','/')/'kernel.cubin'
    target=c.OUT/'disassembly'/cohort/(tag.replace(':','_')+'.json')
    argv=['/usr/local/cuda/bin/nvdisasm','-g','-c',str(artifact)]
    if target.exists():raw=c.read(target)
    else:
        p=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
        raw={'argv':argv,'returncode':p.returncode,'output':p.stdout,'CPU_only':True,
             'CUBIN_SHA256':c.sha(artifact.read_bytes())}
        c.write_once(target,raw)
    assert raw['returncode']==0 and raw['CUBIN_SHA256']==gate['metadata'][tag]['cubin_SHA256']
    mapping={};location=None
    for line in raw['output'].splitlines():
        loc=re.search(r'//## File "([^"]+)", line (\d+)',line)
        if loc:location=(loc[1],int(loc[2]))
        ins=re.search(r'/\*([0-9a-fA-F]+)\*/\s+(.*?);',line)
        if ins:mapping[int(ins[1],16)]={'instruction':ins[2].strip(),'location':location}
    by_source=defaultdict(lambda:defaultdict(int));pcs=[];unmapped=0
    for pc,row in dynamic['per_PC'].items():
        entry=mapping.get(int(pc,16));location=entry['location'] if entry else None
        filename,line=(location if location else ('unknown',0))
        stage='unattributed'
        if filename.endswith('/phase11/kernels.py'):
            stage={44:'ML',45:'local_conversion',46:'MR',48:'MO',50:'global_store'}.get(line,'prefix_or_other')
        elif filename.endswith('/phase11/c2c/kernels.py'):
            stage={22:'joint_ML_local_MR',28:'MO',30:'global_store',26:'control_gather'}.get(line,'prefix_or_other')
        if entry is None:unmapped+=1
        for family in row['families']:by_source[stage][family]+=row['count']
        pcs.append({'relative_PC':pc,'source_location':location,'source_stage':stage,
            'instruction':row['instruction'],'count':row['count'],'families':row['families'],
            'mapped_to_original_CUBIN':entry is not None})
    return {'cohort':cohort,'tag':tag,'disassembly':str(target.relative_to(c.ROOT)),
        'disassembly_SHA256':c.sha(target.read_bytes()),'dynamic_by_source_stage':dict(by_source),
        'per_PC':pcs,'unmapped_PC_sites':unmapped,
        'scope':'Debug-line provenance only. A combined source line is not separated into ML/local/MR. No per-stage latency attribution.'}


def run():
    bundle=pipeline.load();diag=c.read(c.OUT/'pipeline.json');pairs=[];pc_records=[]
    for cohort,gate in bundle['gates'].items():
        meta=[r for r in gate['metadata'].values() if r['family']=='ONE_FULL']
        groups=defaultdict(list)
        for r in meta:groups[r['M'],r['N'],r['num_warps'],r['vec']].append(r)
        for (m,n,w,v),items in groups.items():
            if len(items)!=2:continue
            nf=next((r for r in items if r['placement']=='n_first'),None)
            wi=next((r for r in items if r['placement']=='within'),None)
            if not nf or not wi:continue
            a,b=(next(r['case'] for r in bundle['full'] if r['cohort']==cohort and r['tag']==x['tag']) for x in (nf,wi))
            pa,pb=plan(a),plan(b)
            controls=[x for x in bundle['controls'] if x['cohort']==cohort and x['geometry_vector']==[m,n,w]]
            rows=[x for x in bundle['full'] if x['cohort']==cohort and x['tag'] in (nf['tag'],wi['tag'])]
            contrasts=[]
            for r in rows:
                if r['tag']!=nf['tag']:continue
                s=next((x for x in rows if x['tag']==wi['tag'] and x['worker']==r['worker'] and
                    x['GPU_UUID']==r['GPU_UUID'] and x['q']==r['q']),None)
                if not s:continue
                d=next((x for x in controls if x['worker']==r['worker'] and x['GPU_UUID']==r['GPU_UUID'] and x['q']==r['q']),None)
                preds={}
                for model in ('C34_resource_only','C34_full_auto'):
                    dr=[x for x in diag['records'] if x['model']==model and x['cohort']==cohort and
                        x['worker']==r['worker'] and x['GPU_UUID']==r['GPU_UUID'] and x['actual_q']==r['q']]
                    ar=next(x for x in dr if x['tag']==nf['tag']);br=next(x for x in dr if x['tag']==wi['tag'])
                    preds[model]={key:br['predictions'][key]['tile_ns']-ar['predictions'][key]['tile_ns']
                        if ar['predictions'][key] and br['predictions'][key] else None for key in pipeline.VARIANTS}
                contrasts.append({'q':r['q'],'worker':r['worker'],'GPU_UUID':r['GPU_UUID'],
                    'natural':r['q_condition']=='natural' and s['q_condition']=='natural',
                    'N_first_FULL_ns':r['target_ns'],'within_FULL_ns':s['target_ns'],
                    'control_ns':d['target_ns'] if d else None,
                    'within_minus_N_first_ns':s['target_ns']-r['target_ns'],
                    'within_gain_percent':100*(1-s['target_ns']/r['target_ns']),
                    'predicted_within_minus_N_first_ns':preds})
            static={}
            for key in ('warp_load_instructions','ideal_wavefronts','wavefronts','bank_conflicts'):
                static['ML_'+key]=[pa['ML'][key],pb['ML'][key]]
            for key in ('local_max_scalar_equivalents','shuffle_warp_instructions','shared_load_wavefronts','barrier_warp_instructions'):
                static['MR_'+key]=[pa['MR'][key],pb['MR'][key]]
            resources=[gate['resources'][x['tag']] for x in (nf,wi)]
            ct=bundle['counters'][cohort]
            dynamic=[];hardware=[]
            for x in (nf,wi):
                origin=ct['profile_aliases'].get(x['tag'],x['tag'])
                dynamic.append(next(r for r in ct['single_pass'] if r['tag']==origin)['counts'])
                hardware.append([{**r,'metrics_per_CTA':{k:value/r['grid'] for k,value in r['metrics'].items()
                    if k.endswith('.sum')}} for r in ct['hardware'] if r['tag']==origin])
            pair={'cohort':cohort,'geometry':[m,n,w],'vec':v,'N_first':nf['tag'],'within':wi['tag'],
                'static_subgraph_counts':static,'p':[pa['p'],pb['p']],'u':[pa['u'],pb['u']],'w':[pa['w'],pb['w']],
                'coarse_MO_flags':[pa['output'],pb['output']],
                'same_static_ML':all(static['ML_'+k][0]==static['ML_'+k][1] for k in ('warp_load_instructions','ideal_wavefronts','wavefronts','bank_conflicts')),
                'same_static_local_max_demand':static['MR_local_max_scalar_equivalents'][0]==static['MR_local_max_scalar_equivalents'][1],
                'n_regs':[r['attributes']['num_regs'] for r in resources],
                'natural_q':[r['reservations']['natural']['capacity'] for r in resources],
                'local_bytes':[r['attributes']['local_bytes'] for r in resources],
                'full_dynamic_counts':dynamic,'full_hardware_counters':hardware,
                'same_worker_same_q_contrasts':contrasts,
                'causal_scope':'Matched static demands are screening criteria; code dependencies, lowering, output conversion and physical register allocation can differ.'}
            pairs.append(pair)
            if [m,n,w] in c.protocol()['primary_diagnostic_geometries'] and v==2:
                for x in (nf,wi):pc_records.append(provenance(bundle,cohort,x['tag']))
    result={'status':'OK','pairs':pairs,'PC_provenance':pc_records,
        'GPU_allocations':0,'NCU_new_collections':0,'no_raw_artifacts_modified':True,
        'scope':'Existing FULL kernels and retained software/hardware counters only; no synthetic loop extrapolation.'}
    c.derived(c.OUT/'mechanisms.json',result)
    print('M1 mechanisms complete',len(pairs),'layout pairs;',len(pc_records),'retained-CUBIN source maps',flush=True)
    return result


if __name__=='__main__':run()
