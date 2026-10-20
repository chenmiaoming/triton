"""Matched FULL contrasts, machine-code changes and dynamic source provenance."""
import ast
from collections import Counter, defaultdict
import csv
import io
import math
import re
import statistics as st
import subprocess
from experiments.tma_reduction_layout.phase11.m3 import common as c
from experiments.tma_reduction_layout.phase11.m3.static import output_plan
from experiments.tma_reduction_layout.phase11.c3.static import candidates
from experiments.tma_reduction_layout.phase11.m1.pipeline import contexts, response
from experiments.tma_reduction_layout.phase11.m2.diagnose import source_lines as m2_source_lines


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
                if name.endswith('/m3/kernels.py'):stage=linenos.get(line,'prefix_or_other')
                elif name.endswith('/m2/kernels.py'):stage=m2_source_lines().get(line,'prefix_or_other')
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
    p=c.protocol();gate=c.read(c.OUT/'artifact_gate.json');counter=c.read(c.OUT/'counter_gate.json')
    analysis=c.read(c.OUT/'analysis.json');pred=c.read(c.OUT/'preregistration/static_predictions.json')
    codes=code_audit(gate,counter);code={r['tag']:r for r in codes}
    index=pred['layouts'];rows=[]
    for expectation in index:
        key=expectation['layout_key'];fixed=key+':fixed:ONE_FULL';native=key+':native:ONE_FULL'
        a,b=code[fixed],code[native]
        mo=a['dynamic_by_source'].get('MO',{})
        observed='shared' if mo.get('shared_load',0) or mo.get('shared_store',0) else 'shuffle' if mo.get('shuffle',0) else 'register_or_unattributed'
        hardware={}
        for tag in (fixed,native):
            origin=counter['profile_aliases'].get(tag,tag)
            h=next(r for r in counter['hardware'] if r['tag']==origin and r['q']=='natural')
            hardware[tag]={'sectors_per_CTA':h['metrics']['l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum']/h['grid'],
                'shared_wavefronts_per_CTA':h['metrics']['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum']/h['grid'],
                'store_instructions':{k:v for k,v in code[tag]['whole_opcode_counts'].items() if k.startswith('STG')},
                'raw_metrics':h['metrics'],'GPU_UUID':h['GPU_UUID']}
        expected_fixed=expectation['predicted_fixed_store_sectors_per_CTA']
        expected_native=expectation['predicted_native_store_sectors_per_CTA']
        rows.append({'layout_key':key,'prediction':expectation,'observed_MO':observed,
            'conversion_prediction_match':observed==expectation['predicted_fixed_lowering'],
            'fixed_MO_counts':mo,'native_MO_counts':b['dynamic_by_source'].get('MO',{}),
            'hardware':hardware,
            'fixed_sector_prediction_match':math.isclose(hardware[fixed]['sectors_per_CTA'],expected_fixed),
            'native_sector_prediction_match':math.isclose(hardware[native]['sectors_per_CTA'],expected_native),
            'fixed_registers':a['attributes']['num_regs'],'native_registers':b['attributes']['num_regs'],
            'fixed_q':a['q'],'native_q':b['q']})
    probes=[]
    for expectation in pred['broadcast']:
        broadcast=code[expectation['tag']];native=code[expectation['anchor_native']]
        observed=broadcast['whole_counts']['shared_load']
        a=native['whole_counts'];b=broadcast['whole_counts']
        probes.append({'tag':expectation['tag'],'expectation':expectation,
            'native_LDS':a['shared_load'],'broadcast_LDS':observed,
            'prediction_match':observed==expectation['predicted_broadcast_whole_LDS'],
            'same_MAX_count':a['max']==b['max'],'same_SHFL_count':a['shuffle']==b['shuffle'],
            'same_ML_load_count':native['dynamic_by_source'].get('ML',{}).get('shared_load',0)==broadcast['dynamic_by_source'].get('ML',{}).get('shared_load',0),
            'native_counts':a,'broadcast_counts':b,
            'native_regs':native['attributes']['num_regs'],'broadcast_regs':broadcast['attributes']['num_regs'],
            'profile_only':True,'formal_timing_comparison':False})
    services=unique_services(analysis['oneshot'])
    lookup={(r['tag'],r['q'],r['invocation'],r['GPU_UUID']):r for r in services}
    paired=[]
    for m,n,w,v in p['pair_geometries_vectors']:
        prefix=f'M{m}_N{n}_w{w}:vec{v}'
        for a in services:
            if a['tag']!=prefix+':n_first:fixed:ONE_FULL':continue
            key=(a['q'],a['invocation'],a['GPU_UUID'])
            tags=[prefix+':'+place+':'+policy+':ONE_FULL' for place in ('n_first','within') for policy in ('fixed','native')]
            records=[lookup.get((tag,*key)) for tag in tags]
            if not all(records):continue
            nf,nn,wf,wn=[r['tile_slope_ns'] for r in records]
            paired.append({'geometry':[m,n,w],'vec':v,'q':key[0],'worker':key[1],'GPU_UUID':key[2],
                'natural':all('natural' in r['aliases'] for r in records),
                'fixed_gap_ns':wf-nf,'native_gap_ns':wn-nn,'interaction_ns':(wf-nf)-(wn-nn),
                'N_first_fixed_ns':nf,'within_fixed_ns':wf,'N_first_native_ns':nn,'within_native_ns':wn})
    grouped=defaultdict(list)
    for r in paired:grouped[tuple(r['geometry']),r['vec'],r['q']].append(r)
    summary=[]
    for (geometry,v,q),values in sorted(grouped.items()):
        assert len(values)==3
        summary.append({'geometry':list(geometry),'vec':v,'q':q,'natural':all(r['natural'] for r in values),
            **{k:describe([r[k] for r in values]) for k in ('fixed_gap_ns','native_gap_ns','interaction_ns',
            'N_first_fixed_ns','within_fixed_ns','N_first_native_ns','within_native_ns')}})
    result={'status':'OK','layout_tests':rows,'consumer_tests':probes,'code':codes,
        'pairs':paired,'summary':summary,'selector_refit':False,'new_blind_claim':False,
        'conversion_matches':sum(r['conversion_prediction_match'] for r in rows),
        'sector_matches':sum(r['fixed_sector_prediction_match']+r['native_sector_prediction_match'] for r in rows),
        'consumer_matches':sum(r['prediction_match'] for r in probes)}
    c.derived(c.OUT/'diagnostics.json',result)
    print('M3 diagnostics: conversion',result['conversion_matches'],'/',len(rows),
        'sector',result['sector_matches'],'/',2*len(rows),'consumer',result['consumer_matches'],'/',len(probes),flush=True)
    return result


if __name__=='__main__':main()
