"""Paired FULL responses, source-level communication audits and consumer tables."""
import ast
from collections import Counter, defaultdict
import math
import re
import statistics as st
import subprocess
from . import common as c
from .static import load_plan, plan
from experiments.tma_reduction_layout.phase11.c3.static import candidates, upstream


def describe(values):
    values=list(values)
    mean=st.mean(values)
    radius=4.303*st.stdev(values)/math.sqrt(3) if len(values)==3 else 0
    return {'mean':mean,'median':st.median(values),'min':min(values),'max':max(values),
            'values':values,'paired_worker_t95':[mean-radius,mean+radius]}


def source_classes(path):
    result={}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node,ast.Assign):
            calls={ast.unparse(child.func) for child in ast.walk(node.value) if isinstance(child,ast.Call)}
            if 'smem.load' in calls:kind='ML'
            elif 'gl.max' in calls:kind='MR'
            elif 'gl.convert_layout' in calls:kind='MO'
            elif 'raw.to' in calls:kind='widening'
            elif 'smem.gather' in calls:kind='control_gather'
            else:kind='other'
            result[node.lineno]=kind
        elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='gl.store':
            result[node.lineno]='global_store'
    return result


def code_audit():
    gate=c.read(c.OUT/'compile_gate.json')
    resource=c.read(c.OUT/'artifact_gate.json')
    counter=c.read(c.OUT/'counter_gate.json')
    dynamic={row['tag']:row for row in counter['single_pass']}
    classes={}
    standard_path=c.OUT/'compiler_reference/python/triton/language/standard.py'
    if not standard_path.exists():
        c.write_once(standard_path,subprocess.check_output(['git','show',c.PIN+':python/triton/language/standard.py'],cwd=c.ROOT))
    standard_tree=ast.parse(standard_path.read_text())
    max_function=next(node for node in standard_tree.body if isinstance(node,ast.FunctionDef) and node.name=='max')
    reduce_lines={node.lineno for node in ast.walk(max_function) if isinstance(node,ast.Return) and
                  isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='core.reduce'}
    audits=[]
    for tag,meta in gate['metadata'].items():
        row=dynamic[counter['profile_aliases'].get(tag,tag)]
        folder=c.artifact_folder(meta,gate)
        if (folder/'kernel.lines.sass').exists():
            text=(folder/'kernel.lines.sass').read_text()
        else:
            target=c.OUT/'disassembly'/(tag.replace(':','_')+'.json')
            if target.exists():
                value=c.read(target)
            else:
                argv=['/usr/local/cuda/bin/nvdisasm','-g','-c',str(folder/'kernel.cubin')]
                process=subprocess.run(argv,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
                value={'argv':argv,'CPU_only':True,'returncode':process.returncode,'output':process.stdout,
                       'CUBIN_SHA256':meta['cubin_SHA256']}
                c.write_once(target,value)
            assert value['returncode']==0 and value['CUBIN_SHA256']==meta['cubin_SHA256']
            text=value['output']
        locations={};location=None
        for line in text.splitlines():
            match=re.search(r'//## File "([^"]+)", line (\d+)',line)
            if match:location=(match[1],int(match[2]))
            instruction=re.search(r'/\*([0-9a-fA-F]+)\*/\s+(.*?);',line)
            if instruction:locations[int(instruction[1],16)]=(location,instruction[2].strip())
        groups=defaultdict(Counter);opcodes=defaultdict(Counter);pcs=[]
        for pc,value in row['per_PC'].items():
            location,instruction=locations[int(pc,16)]
            tokens=value['instruction'].split()
            opcode=tokens[1] if tokens[0].startswith('@') else tokens[0]
            tokens2=instruction.split()
            assert opcode==(tokens2[1] if tokens2[0].startswith('@') else tokens2[0]),(tag,pc)
            stage='unattributed'
            if location:
                name,line=location
                if name.endswith('/python/triton/language/standard.py') and line in reduce_lines:
                    stage='MR'
                marker='/experiments/'
                if marker in name:
                    relative='experiments/'+name.split(marker,1)[1]
                    path=c.ROOT/relative
                    if path.exists():
                        if relative not in classes:classes[relative]=source_classes(path)
                        stage=classes[relative].get(line,'prefix_or_other')
            if value['count']:
                for family in value['families']:groups[stage][family]+=value['count']
                opcodes[stage][opcode]+=value['count']
            pcs.append({'PC':pc,'instruction':value['instruction'],'count':value['count'],
                        'location':location,'source_group':stage})
        audits.append({'tag':tag,'output_policy':meta['output_policy'],
                       'source_family_counts':{k:dict(v) for k,v in groups.items()},
                       'source_opcode_counts':{k:dict(v) for k,v in opcodes.items()},
                       'whole_counts':row['counts'],'whole_opcode_counts':row['opcode_counts'],
                       'attributes':resource['resources'][tag]['attributes'],
                       'q_capacity':resource['resources'][tag]['reservations']['natural']['capacity'],
                       'PTX_max':gate['static_audit'][tag]['max_PTX_sites'],'PC_source':pcs})
    return audits


def main():
    p=c.protocol();gate=c.read(c.OUT/'artifact_gate.json')
    metadata=gate['metadata']
    services=c.read(c.OUT/'analysis.json')['oneshot']
    rows={}
    for row in services:
        key=(row['tag'],row['q'],row['invocation'])
        if key in rows:
            assert math.isclose(rows[key]['tile_slope_ns'],row['tile_slope_ns'],abs_tol=1e-7)
        else:rows[key]=row
    hardware=c.read(c.OUT/'counter_gate.json')['hardware']
    counters=[]
    for row in hardware:
        counters.append({**row,'per_CTA':{name:value/row['grid'] for name,value in row['metrics'].items()
                           if name.endswith('.sum') and any(x in name for x in ('wavefronts','bank_conflicts','sectors'))}})
    codes=code_audit()
    if c.STAGE=='m4a':
        contrasts=[];load_deltas=[]
        for m,n,w,_ in p['pair_geometries_vectors']:
            for sw in (128,64,32,0):
                tags=[f'M{m}_N{n}_w{w}:vec2:{placement}:sw{sw}:native:ONE_FULL' for placement in ('n_first','within')]
                if not all(tag in metadata for tag in tags):continue
                qs=set(row['q'] for row in rows.values() if row['tag']==tags[0])&set(row['q'] for row in rows.values() if row['tag']==tags[1])
                for q in sorted(qs):
                    contrasts.append({'geometry':[m,n,w],'swizzle':sw,'q_capacity':q,
                        'within_minus_NF_ns':describe(rows[tags[1],q,i]['tile_slope_ns']-rows[tags[0],q,i]['tile_slope_ns'] for i in (1,2,3))})
            for placement in ('n_first','within'):
                original=f'M{m}_N{n}_w{w}:vec2:{placement}:sw128:native:ONE_FULL'
                for sw in (64,32,0):
                    tag=f'M{m}_N{n}_w{w}:vec2:{placement}:sw{sw}:native:ONE_FULL'
                    if tag not in metadata:continue
                    qs=set(row['q'] for row in rows.values() if row['tag']==tag)&set(row['q'] for row in rows.values() if row['tag']==original)
                    code={r['tag']:r for r in codes}
                    for q in sorted(qs):
                        load_deltas.append({'tag':tag,'anchor':original,'q_capacity':q,
                            'service_delta_ns':describe(rows[tag,q,i]['tile_slope_ns']-rows[original,q,i]['tile_slope_ns'] for i in (1,2,3)),
                            'MR_source_opcode_counts_equal':bool(code[tag]['source_opcode_counts'].get('MR')) and code[tag]['source_opcode_counts'].get('MR')==code[original]['source_opcode_counts'].get('MR'),
                            'registers_equal':code[tag]['attributes']['num_regs']==code[original]['attributes']['num_regs'],
                            'qualification':'Shared swizzle also changes TMA transaction grouping; check matching transfer control, not an isolated latency coefficient.'})
        old=[]
        for stage in ('m2','m3'):
            value=c.read(c.BASE/f'results/phase11/stage_{stage}/analysis.json')
            old.extend({'source_stage':stage,**row} for row in value['oneshot']
                       if row['geometry'] in ([32,64,8],[32,128,4],[16,256,8]) and ':vec2:' in row['tag'])
        result={'stage':'M4a','within_NF_contrasts':contrasts,'shared_layout_deltas':load_deltas,
                'historical_q_curves_reused':old,'historical_offset_used':False}
    else:
        scores=[];store_checks=[]
        dynamic={row['tag']:row for row in c.read(c.OUT/'counter_gate.json')['single_pass']}
        for m,n,w,_ in p['pair_geometries_vectors']:
            for consumer in ('native','broadcast','centered'):
                eligible=[(tag,meta) for tag,meta in metadata.items() if (meta['M'],meta['N'],meta['num_warps'],meta['output_policy'])==(m,n,w,consumer)]
                default=next(tag for tag,meta in eligible if meta['layout_key']==upstream(m,n,w))
                vec1=next(tag for tag,meta in eligible if meta['vec']==1 and meta['placement']=='n_first')
                for tag,meta in eligible:
                    def natural(t):
                        return {row['invocation']:row for row in services if row['tag']==t and row['q_condition']=='natural'}
                    current,base,one=natural(tag),natural(default),natural(vec1)
                    scores.append({'geometry':[m,n,w],'consumer':consumer,'tag':tag,
                        'q_capacity':gate['resources'][tag]['reservations']['natural']['capacity'],
                        'service_ns':describe(current[i]['tile_slope_ns'] for i in (1,2,3)),
                        'raw_128MiB_us':describe(current[i]['full_us'] for i in (1,2,3)),
                        'gain_vs_default_percent':describe(100*(base[i]['tile_slope_ns']-current[i]['tile_slope_ns'])/base[i]['tile_slope_ns'] for i in (1,2,3)),
                        'gain_vs_fixed_vec1_percent':describe(100*(one[i]['tile_slope_ns']-current[i]['tile_slope_ns'])/one[i]['tile_slope_ns'] for i in (1,2,3))})
                    case=next(case for case in candidates(m,n,w) if case['id']==meta['layout_key'])
                    pred=plan(case,consumer)
                    alias=c.read(c.OUT/'counter_gate.json')['profile_aliases'].get(tag,tag)
                    hw=next(row for row in counters if row['tag']==alias and row['q']=='natural')
                    actual=hw['metrics']['l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum']/hw['grid']
                    store_checks.append({'tag':tag,'predicted':pred['store']['predicted_sector_requests_per_CTA'],
                                         'measured':actual,'matched':math.isclose(actual,pred['store']['predicted_sector_requests_per_CTA'],rel_tol=.001,abs_tol=.05)})
        result={'stage':'M4b','scores':scores,'forward_store_predictions':store_checks,
                'no_selector_fit':True,'different_consumer_output_bytes_not_combined':True}
    result.update(status='OK',source_code_audits=codes,hardware=counters,
                  stall_scope='whole-kernel warp-state rates; correlation only, not independent phase nanoseconds')
    c.derived(c.OUT/'diagnostics.json',result)
    print(c.STAGE,'diagnostics OK;',len(codes),'source/PC audits')


if __name__=='__main__':main()
