"""Matched FULL contrasts, machine-code changes and dynamic source provenance."""
import ast
from collections import Counter, defaultdict
import csv
import io
import math
import re
import statistics as st
import subprocess
from experiments.tma_reduction_layout.phase11.m2 import common as c
from experiments.tma_reduction_layout.phase11.m2.static import output_plan
from experiments.tma_reduction_layout.phase11.c3.static import candidates
from experiments.tma_reduction_layout.phase11.m1.pipeline import contexts, response


def describe(values):
    n=len(values)
    return {'median':st.median(values),'min':min(values),'max':max(values),
        'values':values,'mean':st.mean(values),'worker_t_interval':
            [st.mean(values)-4.303*st.stdev(values)/math.sqrt(n),
             st.mean(values)+4.303*st.stdev(values)/math.sqrt(n)] if n==3 else None}


def unique_services(rows):
    result={}
    for r in sorted(rows,key=lambda x:(x['q_condition']!='natural',x['q_condition'])):
        key=(r['tag'],r['q'],r['invocation'],r['GPU_UUID'])
        if key in result:
            assert math.isclose(result[key]['tile_slope_ns'],r['tile_slope_ns'],abs_tol=1e-7)
            result[key]['aliases'].append(r['q_condition'])
        else:result[key]={**r,'aliases':[r['q_condition']]}
    return list(result.values())


def source_lines():
    source=(c.SOURCE/'kernels.py').read_text()
    result={}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call):
            name=ast.unparse(node.value.func)
            if name in ('smem.load','raw.to','gl.max','gl.convert_layout'):
                result[node.lineno]={'smem.load':'ML','raw.to':'local_conversion',
                    'gl.max':'MR','gl.convert_layout':'MO'}[name]
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='gl.store':
            result[node.lineno]='global_store'
    return result


def disassembly(meta,compile_gate):
    path=c.artifact_folder(meta,compile_gate)/'kernel.cubin'
    dest=c.OUT/'disassembly'/(meta['tag'].replace(':','_')+'.json')
    if dest.exists():raw=c.read(dest)
    else:
        argv=['/usr/local/cuda/bin/nvdisasm','-g','-c',str(path)]
        proc=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        raw={'argv':argv,'returncode':proc.returncode,'output':proc.stdout,
             'CUBIN_SHA256':c.sha(path.read_bytes()),'CPU_only':True}
        c.write_once(dest,raw)
    assert raw['returncode']==0 and raw['CUBIN_SHA256']==meta['cubin_SHA256']
    mapping={};location=None
    for line in raw['output'].splitlines():
        loc=re.search(r'//## File "([^"]+)", line (\d+)',line)
        if loc:location=[loc[1],int(loc[2])]
        ins=re.search(r'/\*([0-9a-fA-F]+)\*/\s+(.*?);',line)
        if ins:mapping[int(ins[1],16)]={'instruction':ins[2].strip(),'location':location}
    return dest,mapping


def code_audit(gate,counter):
    result=[]
    cg=c.read(c.OUT/'compile_gate.json')
    linenos=source_lines()
    for tag,meta in gate['metadata'].items():
        origin=counter['profile_aliases'].get(tag,tag)
        dynamic=next(r for r in counter['single_pass'] if r['tag']==origin)
        path,mapping=disassembly(meta,cg)
        counts=defaultdict(Counter)
        ops=defaultdict(Counter)
        pcs=[]
        for pc,row in dynamic['per_PC'].items():
            entry=mapping.get(int(pc,16))
            assert entry is not None,(tag,pc)
            loc=entry['location'];stage='unattributed'
            if loc:
                name,line=loc
                if name.endswith('/m2/kernels.py'):stage=linenos.get(line,'prefix_or_other')
                elif name.endswith('/phase11/kernels.py'):
                    stage={44:'ML',45:'local_conversion',46:'MR',48:'MO',50:'global_store'}.get(line,'prefix_or_other')
                elif name.endswith('/c2c/kernels.py'):
                    stage={22:'joint_ML_local_MR',28:'MO',30:'global_store',26:'control_gather'}.get(line,'prefix_or_other')
            for family in row['families']:counts[stage][family]+=row['count']
            instruction=row['instruction']
            # Retain full operands/predicates when checking upstream equality;
            # register allocation alone may change these, so report both forms.
            ops[stage][instruction]+=row['count']
            pcs.append({**row,'relative_PC':pc,'location':loc,'stage':stage})
        expected=cg['static_audit'][tag]
        result.append({'tag':tag,'output_policy':meta['output_policy'],
            'disassembly':str(path.relative_to(c.ROOT)),'CUBIN_SHA256':meta['cubin_SHA256'],
            'dynamic_by_source':{k:dict(v) for k,v in counts.items()},
            'dynamic_instruction_by_source':{k:dict(v) for k,v in ops.items()},
            'whole_counts':dynamic['counts'],'whole_opcode_counts':dynamic['opcode_counts'],
            'per_PC':pcs,'static_audit':expected,
            'attributes':gate['resources'][tag]['attributes'],
            'q':gate['resources'][tag]['reservations']['natural']['capacity']})
    return result


def main():
    p=c.protocol();gate=c.read(c.OUT/'artifact_gate.json')
    analysis=c.read(c.OUT/'analysis.json');counter=c.read(c.OUT/'counter_gate.json')
    rows=unique_services(analysis['oneshot'])
    index={(r['tag'],r['q'],r['invocation'],r['GPU_UUID']):r for r in rows}
    contexts_=contexts()
    pairs=[];bridges=[];model_failures=[]
    for m,n,w,vec in p['pair_geometries_vectors']:
        prefix=f'M{m}_N{n}_w{w}:vec{vec}'
        tags={(place,policy):prefix+':'+place+':'+policy+':ONE_FULL'
            for place in ('n_first','within') for policy in ('legacy_fixed','fixed','native')}
        cohort='D1' if (m,n,w)==(16,256,8) else 'development'
        geo=f'{m}x{n}w{w}'
        for row in rows:
            if row['tag']!=tags['n_first','fixed']:continue
            key=(row['q'],row['invocation'],row['GPU_UUID'])
            found={k:index.get((tag,*key)) for k,tag in tags.items()}
            if not all(found.values()):continue
            v={k:r['tile_slope_ns'] for k,r in found.items()}
            fixed=v['within','fixed']-v['n_first','fixed']
            native=v['within','native']-v['n_first','native']
            legacy=v['within','legacy_fixed']-v['n_first','legacy_fixed']
            controls=[r for r in rows if r['family']=='ONE_TRANSFER' and r['geometry']==[m,n,w]
                      and (r['q'],r['invocation'],r['GPU_UUID'])==key]
            pairs.append({'geometry':[m,n,w],'vec':vec,'q':key[0],'worker':key[1],
                'GPU_UUID':key[2],'natural':all('natural' in r['aliases'] for r in found.values()),
                'fixed_gap_ns':fixed,'native_gap_ns':native,'legacy_gap_ns':legacy,
                'interaction_ns':fixed-native,'source_bridge_gap_ns':fixed-legacy,
                'N_first_fixed_ns':v['n_first','fixed'],'within_fixed_ns':v['within','fixed'],
                'N_first_native_ns':v['n_first','native'],'within_native_ns':v['within','native'],
                'control_ns':controls[0]['tile_slope_ns'] if controls else None})
            for model in ('C34_resource_only','C34_full_auto'):
                ctx=contexts_[model,cohort,geo]
                cases={place:next(x for x in candidates(m,n,w) if x['id']==prefix+':'+place)
                       for place in ('n_first','within')}
                a=response(ctx['model'],cases['n_first'],key[0],ctx['tables'])
                b=response(ctx['model'],cases['within'],key[0],ctx['tables'])
                model_failures.append({'model':model,'geometry':[m,n,w],'vec':vec,
                    'q':key[0],'worker':key[1],'GPU_UUID':key[2],
                    'predicted_fixed_gap_ns':b['tile_ns']-a['tile_ns'],
                    'measured_matched_fixed_gap_ns':fixed,'measured_legacy_gap_ns':legacy,
                    'measured_native_gap_ns':native,
                    'scope':'Frozen fixed-output model; native is an intervention, not in-domain prediction.'})
        for place in ('n_first','within'):
            for r in rows:
                if r['tag']!=tags[place,'legacy_fixed']:continue
                other=index.get((tags[place,'fixed'],r['q'],r['invocation'],r['GPU_UUID']))
                if other:
                    bridges.append({'geometry':[m,n,w],'vec':vec,'placement':place,'q':r['q'],
                        'worker':r['invocation'],'GPU_UUID':r['GPU_UUID'],
                        'old_ns':r['tile_slope_ns'],'new_fixed_ns':other['tile_slope_ns'],
                        'new_minus_old_ns':other['tile_slope_ns']-r['tile_slope_ns'],
                        'relative_change_percent':100*(other['tile_slope_ns']/r['tile_slope_ns']-1)})
    grouped=defaultdict(list)
    for r in pairs:grouped[tuple(r['geometry']),r['vec'],r['q']].append(r)
    summary=[]
    for (geometry,vec,q),values in sorted(grouped.items()):
        assert len(values)==3
        summary.append({'geometry':list(geometry),'vec':vec,'q':q,
            'natural':all(r['natural'] for r in values),
            **{k:describe([r[k] for r in values]) for k in
               ('fixed_gap_ns','native_gap_ns','legacy_gap_ns','interaction_ns','source_bridge_gap_ns',
                'N_first_fixed_ns','within_fixed_ns','N_first_native_ns','within_native_ns')}})
    codes=code_audit(gate,counter)
    code_index={r['tag']:r for r in codes}
    comparisons=[]
    for meta in gate['metadata'].values():
        if meta['output_policy']!='fixed':continue
        native=next(r for r in gate['metadata'].values() if r.get('layout_key')==meta['layout_key'] and r['output_policy']=='native')
        a,b=code_index[meta['tag']],code_index[native['tag']]
        case=next(r for r in candidates(meta['M'],meta['N'],meta['num_warps']) if r['id']==meta['layout_key'])
        plan=output_plan(case)
        stages=('ML','local_conversion','MR','joint_ML_local_MR')
        actual=a['dynamic_by_source'].get('MO',{})
        observed='shared' if actual.get('shared_load',0) or actual.get('shared_store',0) else 'shuffle' if actual.get('shuffle',0) else 'register_or_unattributed'
        hardware=[]
        for tag in (meta['tag'],native['tag']):
            origin=counter['profile_aliases'].get(tag,tag)
            hardware += [{**r,'original_tag':tag,'metrics_per_CTA':{k:v/r['grid'] for k,v in r['metrics'].items() if k.endswith('.sum')}}
                         for r in counter['hardware'] if r['tag']==origin]
        comparisons.append({'layout_key':meta['layout_key'],'fixed_tag':meta['tag'],'native_tag':native['tag'],
            'predicted_minimum_conversion_reach':plan['predicted_fixed_conversion_reach'],
            'source_observed_fixed_conversion':observed,
            'native_MO_attributed_counts':b['dynamic_by_source'].get('MO',{}),
            'fixed_MO_attributed_counts':actual,
            'same_ML_MR_family_counts':all(a['dynamic_by_source'].get(s,{})==b['dynamic_by_source'].get(s,{}) for s in stages),
            'same_ML_MR_instructions_and_operands':all(a['dynamic_instruction_by_source'].get(s,{})==b['dynamic_instruction_by_source'].get(s,{}) for s in stages),
            'same_natural_q':a['q']==b['q'],'same_n_regs':a['attributes']['num_regs']==b['attributes']['num_regs'],
            'fixed_whole_counts':a['whole_counts'],'native_whole_counts':b['whole_counts'],
            'hardware':hardware,
            'scope':'Source attribution incomplete; equality of family counts does not prove equal scheduling/dependencies.'})
    result={'status':'OK','pairs':pairs,'summary':summary,'source_bridges':bridges,
        'frozen_model_comparisons':model_failures,'code':codes,'code_comparisons':comparisons,
        'selector_refit':False,'new_blind_claim':False,'NCU_duration_used_for_timing':False}
    c.derived(c.OUT/'diagnostics.json',result)
    for name,records in [('paired_interactions',pairs),('source_bridges',bridges),('frozen_model_comparisons',model_failures)]:
        buffer=io.StringIO()
        head=list(records[0])
        writer=csv.DictWriter(buffer,head,lineterminator='\n')
        writer.writeheader();writer.writerows(records)
        c.derived(c.OUT/'tables'/(name+'.csv'),buffer.getvalue().encode())
    print('M2 diagnostics complete',len(pairs),'same-worker interactions;',len(comparisons),'code-policy contrasts',flush=True)
    return result


if __name__=='__main__':main()
