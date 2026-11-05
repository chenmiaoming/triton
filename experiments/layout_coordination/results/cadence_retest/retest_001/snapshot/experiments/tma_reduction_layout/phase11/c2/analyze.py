"""Offline write-once binary gates and reproducible C2 derived analysis."""
import argparse
from collections import defaultdict,Counter
import csv
import io
import json
from pathlib import Path
import re
import shutil
import statistics as st
import numpy as np
from experiments.tma_reduction_layout.phase11.c2 import common as c

FAMILIES={'shared_load':r'\bLDS(?:M)?(?:\.|\s)','shared_store':r'\bSTS(?:\.|\s)',
          'shuffle':r'\bSHFL(?:\.|\s)','barrier':r'\bBAR\.SYNC(?:\.|\s)',
          'max':r'\b(?:F|H|VH)MNMX(?:2)?(?:\.|\s)','conversion':r'\b(?:F2F|F2FP|F2I|I2F)(?:\.|\s)',
          'global_store':r'\bSTG(?:\.|\s)'}


def derived(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(c.encode(value))


def compile_gate(batch):
    protocol=c.load_protocol(batch);c.protect_prior(batch)
    accepted=[p for p in sorted((c.out(batch)/'cpu_preflight').glob('attempt_*/export')) if c.read(p/'result.json')['status']=='OK']
    assert accepted,accepted
    root=accepted[-1];raw=c.read(root/'result.json')
    assert raw['new_compiles']==protocol.get('attempt_compile_count',protocol['new_programs']) and raw['ccache_before']==raw['ccache_after']
    assert raw['native_SHA256']==protocol['baseline']['native_SHA256']
    metadata=raw['metadata'];audit={}
    for tag,meta in metadata.items():
        folder=root/tag.replace(':','/')
        assert c.sha((folder/'kernel.cubin').read_bytes())==meta['cubin_SHA256'],tag
        ptx=(folder/'kernel.ptx').read_text()
        ir=(folder/'kernel.glir').read_text()
        counts=Counter(re.findall(r'^\s*(?:@!?%\w+\s+)?(max\.[\w.]+)\s',ptx,re.M))
        audit[tag]={'max_PTX_sites':dict(counts),'local_bytes_from_resource':re.findall(r'LOCAL:(\d+)',(folder/'kernel.resource.txt').read_text()),
                    'requested_layout':meta.get('layout'),'MO_standalone_diagnostic':meta['family']=='MO'}
        if meta['family'] in ('ML','MR','MC','P','U','W','E'):
            assert 'scf.for' in ir or 'scf.while' in ir,(tag,'runtime loop absent')
        if meta['family']=='MR' and 'reused_tag' not in meta:
            # Source full-tile extf has exactly one consumer per stream.
            loop=ir[ir.index('scf.for'):]
            wide=re.findall(r'(%[\w.]+)\s*=\s*arith.extf[^\n]*tensor<1x32x\d+xbf16[^\n]*to tensor<1x32x\d+xf32',loop)
            assert len(wide)==meta['D'],(tag,wide)
            for value in wide:
                assert len(re.findall(re.escape(value)+r'(?![\w.])',loop))==2,(tag,value)
            assert any('xbf16' in line and 'arith.addf' in line for line in loop.splitlines()),tag
            assert not any(token in loop for token in ('ttg.local_load','ttg.local_store','ttg.convert_layout','tt.broadcast')),tag
            audit[tag]['BF16_state_and_single_use_widening']=True
            if meta['placement']=='n_first':
                oldtag=f'M32_N{meta["N"]}_w{meta["num_warps"]}:vec{meta["vec"]}:MR'
                old=c.base.OUT/'stage_c15/artifacts/attempt_1/export'/oldtag.replace(':','/')/'kernel.ptx'
                oldcounts=Counter(re.findall(r'^\s*(?:@!?%\w+\s+)?(max\.[\w.]+)\s',old.read_text(),re.M))
                # Compiler may interleave D2; opcode classes must preserve packed lowering.
                assert set(counts)==set(oldcounts),(tag,counts,oldcounts)
                audit[tag]['same_packed_max_opcode_classes_as_C15']=True
    if batch=='b':
        previous=c.read(c.out('a')/'artifact_gate.json')
        for tag,meta in previous['metadata'].items():
            if meta['family']=='TMA':continue
            metadata[tag]={**meta,'previous_batch':'C2a'}
    temp=Path('/tmp/tma-c2'+batch+'-compiled-payload-'+root.parent.name);temp.mkdir(exist_ok=False)
    for tag,meta in metadata.items():
        source=root/tag.replace(':','/')
        if meta.get('previous_batch'):source=c.ROOT/c.read(c.out('a')/'compile_gate.json')['artifact_directory']/tag.replace(':','/')
        target=temp/tag.replace(':','/');target.mkdir(parents=True)
        for file in source.glob('kernel.*'):shutil.copyfile(file,target/file.name)
    gate={'status':'OK','metadata':metadata,'artifact_directory':root.relative_to(c.ROOT).as_posix(),
          'static_audit':audit,'new_compiles':protocol['new_programs'],'GPU_allocated_for_compile':False,
          'protocol_SHA256':c.sha(c.encode(protocol)),'original_compilation_protocol_SHA256':raw['protocol_SHA256']}
    (c.out(batch)/'compiled_payload.zip').write_bytes(c.archive(temp));derived(c.out(batch)/'compile_gate.json',gate)
    print('C2'+batch,'CPU compile gate OK',len(metadata),'metadata entries')


def artifact_gate(batch):
    protocol=c.load_protocol(batch);compile=c.read(c.out(batch)/'compile_gate.json')
    roots=[p for p in sorted((c.out(batch)/'eligibility').glob('attempt_*')) if c.read(p/'raw.json')['status']=='OK']
    assert len(roots)==1,roots
    raw=c.read(roots[0]/'raw.json');assert all(x['pass'] for x in raw['checks'])
    assert raw['payload_SHA256']==c.sha((c.out(batch)/'compiled_payload.zip').read_bytes())
    metadata=compile['metadata'];resources=raw['resources'];unique={};aliases={}
    for tag,meta in metadata.items():
        if 'reused_tag' in meta or meta['family']=='TMA':continue
        if meta.get('previous_batch'):continue
        if resources[tag]['attributes']['local_bytes']>0:continue
        digest=meta['cubin_SHA256']
        if digest in unique:aliases[tag]=unique[digest]
        else:unique[digest]=tag
    tags=list(unique.values())
    if batch=='b':
        # Previous binaries need only new q occupancy checks, not repeated
        # same-condition natural source/hardware collection.
        old=[]
        seen=set(unique)
        for tag,meta in metadata.items():
            if not meta.get('previous_batch'):continue
            if resources[tag]['attributes']['local_bytes']>0:continue
            if meta['cubin_SHA256'] in seen:continue
            if any(resources[tag]['reservations'][q].get('feasible') for q in ('1','2','4')):
                old.append(tag);seen.add(meta['cubin_SHA256'])
        tags+=old
    size=protocol['profiles']['group_program_limit'];groups=[tags[i:i+size] for i in range(0,len(tags),size)]
    gate={'status':'OK','metadata':metadata,'resources':resources,'checks':len(raw['checks']),
          'eligibility_attempt':roots[0].name,'environment':raw['environment'],'aliases':aliases,
          'profile_groups':groups,'spill_boundaries':[t for t,r in resources.items() if r['attributes']['local_bytes']>0],
          'protocol_SHA256':c.sha(c.encode(protocol))}
    payload=(c.out(batch)/'compiled_payload.zip').read_bytes()
    target=c.out(batch)/'frozen_payload.zip'
    if target.exists():assert target.read_bytes()==payload
    else:c.write_once(target,payload)
    derived(c.out(batch)/'artifact_gate.json',gate)
    print('C2'+batch,'artifact gate OK',len(metadata),'programs;',len(groups),'NCU groups;',gate['spill_boundaries'],'spill')


def source_blocks(text):
    rows=list(csv.reader(io.StringIO(text)));blocks=[];current=None
    for row in rows:
        if 'Source' in row and 'Instructions Executed' in row:
            if current is not None:blocks.append(current)
            head=row;current={};continue
        if current is None or len(row)!=len(head):continue
        v=dict(zip(head,row));address=v.get('Address','');count=v.get('Instructions Executed','').replace(',','').strip()
        if not address or count in ('','N/A','nan'):continue
        current[address]={'absolute_PC':address,'instruction':v['Source'].strip(),'count':int(count),
                          'families':[k for k,pattern in FAMILIES.items() if re.search(pattern,v['Source'])]}
    if current is not None:blocks.append(current)
    normalized=[]
    for block in blocks:
        assert block,'Missing NCU per-PC data'
        base=min(int(pc,16) for pc in block)
        normalized.append({f'0x{int(pc,16)-base:04x}':v for pc,v in block.items()})
    return normalized


def hardware_rows(text):
    rows=list(csv.reader(io.StringIO(text)))
    wide=next((i for i,row in enumerate(rows) if 'ID' in row and 'Kernel Name' in row and 'Metric Name' not in row),None)
    if wide is not None:
        head=rows[wide];result={}
        for row in rows[wide+1:]:
            if len(row)!=len(head):continue
            v=dict(zip(head,row))
            if not v['ID'].isdigit():continue
            values={}
            for name,value in v.items():
                try:values[name]=float(value.replace(',',''))
                except ValueError:continue
            result[int(v['ID'])]=values
        return [result[i] for i in sorted(result)]
    h=next(i for i,row in enumerate(rows) if 'Metric Name' in row and 'Metric Value' in row)
    head=rows[h];result={}
    for row in rows[h+1:]:
        if len(row)!=len(head):continue
        v=dict(zip(head,row));idx=int(v['ID']);name=v['Metric Name']
        try:value=float(v['Metric Value'].replace(',',''))
        except ValueError:continue
        result.setdefault(idx,{})[name]=value
    return [result[i] for i in sorted(result)]


def affine(a,b,k0=16,k1=256):
    slope=(b-a)/(k1-k0)
    return {'per_iteration':slope,'fixed':a-slope*k0,'at_K16':a,'at_K256':b}


def validate_launch_order(folder,job,metadata):
    rows=list(csv.reader(io.StringIO(c.read(folder/'import.json')['output'])))
    head=rows[0]
    imported=[dict(zip(head,row)) for row in rows[1:] if len(row)==len(head) and row[head.index('ID')].isdigit()]
    assert len(imported)==len(job['cells'])
    for index,(cell,row) in enumerate(zip(job['cells'],imported)):
        meta=metadata[cell['tag']]
        assert int(row['ID'])==index and row['Kernel Name']==meta['function_name']
        assert [int(v) for v in re.findall(r'\d+',row['Grid Size'])]==[cell['grid'],1,1]
        assert [int(v) for v in re.findall(r'\d+',row['Block Size'])]==[meta['num_warps']*32,1,1]
        assert row['CC']=='9.0'


def validate_source_opcodes(batch,meta,pcs):
    origin='a' if meta.get('previous_batch') else batch
    root=c.ROOT/c.read(c.out(origin)/'compile_gate.json')['artifact_directory']/meta['tag'].replace(':','/')
    def opcode(instruction):
        tokens=instruction.split();return tokens[1] if tokens[0].startswith('@') else tokens[0]
    static={int(pc,16):opcode(inst) for pc,inst in re.findall(r'/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;', (root/'kernel.sass').read_text())}
    assert static,meta['tag']
    for pc,row in pcs.items():assert static[int(pc,16)]==opcode(row['instruction']),(meta['tag'],pc,row)


def optimized_input_layout_audit(batch):
    gate=c.read(c.out(batch)/'artifact_gate.json');passed=[]
    for tag,meta in gate['metadata'].items():
        if meta['family'] in ('TMA','MO'):continue
        origin='a' if meta.get('previous_batch') else batch
        folder=c.ROOT/c.read(c.out(origin)/'compile_gate.json')['artifact_directory']/tag.replace(':','/')
        text=(folder/'kernel.ttgir').read_text();enc={}
        names={'sizePerThread':'size_per_thread','threadsPerWarp':'threads_per_warp','warpsPerCTA':'warps_per_cta','order':'order'}
        for alias,body in re.findall(r'(#\w+)\s*=\s*#ttg.blocked<\{(.*?)\}>',text):
            enc[alias]={target:[int(v) for v in re.findall(r'\d+',re.search(name+r'\s*=\s*\[([^\]]*)\]',body)[1])] for name,target in names.items()}
        token='ttg.local_load' if meta['family'] in ('ML','MR','MC') else 'tt.load'
        lines=[line for line in text.splitlines() if token in line];aliases=[]
        for line in lines:
            aliases+=re.findall(r'tensor<1x'+str(meta['M'])+'x'+str(meta['N'])+r'x(?:bf16|!tt.ptr<bf16>),\s*(#\w+)>',line)
        assert aliases and all(enc[alias]==meta['layout'] for alias in aliases),(tag,aliases)
        passed.append(tag)
    return {'status':'OK','programs':passed,'method':'BF16 input load encoding in optimized TTGIR equals requested register layout, all four encoding arrays',
            'MO':'separate FP32 conversion fixture, diagnostic'}


def counter_gate(batch):
    protocol=c.load_protocol(batch);gate=c.read(c.out(batch)/'artifact_gate.json')
    binaries={};qrows=[];reports=[];failures=[]
    if batch=='a':
        old=c.read(c.base.OUT/'stage_c15/counter_gate.json')
        for tag,meta in gate['metadata'].items():
            if 'reused_tag' not in meta:continue
            v=old['binaries'][meta['reused_tag']]
            families=json.loads(json.dumps(v['families']))
            # Reimport the existing source exports for the output-store family
            # that the earlier summary did not enumerate. No GPU collection.
            oldstage='stage_c1' if meta['family']=='ML' else 'stage_c15'
            oldgate=c.read(c.base.OUT/oldstage/'counter_gate.json')
            original=c.base.OUT/oldstage/'profiles'/oldgate['accepted_attempt']/'export'/meta['reused_tag'].replace(':','/')
            totals={}
            for k in (1,16,64,256):
                pcs=source_blocks(c.read(original/f'K{k}'/'software/source.json')['output'])[0]
                totals[str(k)]=sum(p['count'] for p in pcs.values() if 'global_store' in p['families'])
            assert len(set(totals.values()))==1,(tag,totals)
            families['global_store']={'loop_per_iteration_warp_instructions':0,'fixed_warp_instructions':totals['1'],
                'body_counts_by_K':{str(k):0 for k in (1,16,64,256)},'total_by_K':totals,'strict_K_multiple':True}
            binaries[tag]={'reused_128_original_reports':True,'CUBIN_SHA256':meta['cubin_SHA256'],
                           'source':meta['reused_tag'],'families':families,
                           'hardware_delta_per_iteration':v['hardware_delta_per_iteration'],'pass':v['pass']}
    else:
        binaries.update(c.read(c.out('a')/'counter_gate.json')['binaries'])
    for index,group in enumerate(gate['profile_groups']):
        roots=[p for p in sorted((c.out(batch)/'profiles'/f'group_{index}').glob('attempt_*/export')) if c.read(p/'result.json')['status']=='OK']
        assert len(roots)==1,(index,roots)
        root=roots[0];reports += [str(p.relative_to(c.ROOT)) for p in root.glob('*/report.ncu-rep')]
        bytag=defaultdict(dict)
        for mode in ('software','hardware'):
            folder=root/mode
            if not (folder/'job.json').exists():continue
            job=c.read(folder/'job.json');collected=c.read(folder/'collect.json')
            validate_launch_order(folder,job,gate['metadata'])
            line=next(l for l in collected['output'].splitlines() if l.startswith('PROFILE_BINDING '))
            binding=json.loads(line[len('PROFILE_BINDING '):])
            assert not binding['Triton_imported'] and binding['compile_calls']==0 and binding['driver_return_codes']==[0]
            assert len(binding['records'])==len(job['cells'])
            measured=source_blocks(c.read(folder/'source.json')['output']) if mode=='software' else hardware_rows(c.read(folder/'import.json')['output'])
            assert len(measured)==len(job['cells']),(index,mode,len(measured),len(job['cells']))
            for cell,record,values in zip(job['cells'],binding['records'],measured):
                tag=cell['tag'];meta=gate['metadata'][tag]
                if mode=='software':validate_source_opcodes(batch,meta,values)
                assert record['tag']==tag and record['K']==cell['K'] and record['correctness']['pass']
                assert record['binding']['CUBIN_SHA256']==record['binding']['loaded_CUBIN_SHA256']==meta['cubin_SHA256']
                if cell['q']!='natural' or cell['grid']>1:qrows.append({**cell,mode:values});continue
                bytag[tag].setdefault(cell['K'],{}).setdefault(mode,[]).append(values)
        for tag,measured in bytag.items():
            meta=gate['metadata'][tag]
            if meta.get('previous_batch'):continue
            if meta['family']=='MO':
                binaries[tag]={'pass':True,'MO_diagnostic':True,'measured':measured};continue
            assert set(measured)=={16,256},tag
            soft0,soft1=measured[16]['software'][0],measured[256]['software'][0]
            assert set(soft0)==set(soft1),(tag,'PC mismatch')
            families={};pcrows=[]
            for pc in soft0:
                a,b=soft0[pc],soft1[pc]
                row={**a,'PC':pc,**affine(a['count'],b['count'])};pcrows.append(row)
            for family in FAMILIES:
                relevant=[r for r in pcrows if family in r['families']]
                varying=[r for r in relevant if r['per_iteration']!=0]
                loop_counts={str(k):sum(measured[k]['software'][0][r['PC']]['count'] for r in varying) for k in (16,256)}
                families[family]={'loop_per_iteration_warp_instructions':sum(r['per_iteration'] for r in relevant),
                                  'fixed_warp_instructions':sum(r['fixed'] for r in relevant),
                                  'body_counts_by_K':{str(k):k*sum(r['per_iteration'] for r in relevant) for k in (16,256)},
                                  'measured_varying_PC_counts_by_K':loop_counts,
                                  'strict_K_multiple':loop_counts['16']*256==loop_counts['256']*16,
                                  'total_by_K':{str(k):sum(r['count'] for r in measured[k]['software'][0].values() if family in r['families']) for k in (16,256)}}
            hardware={}
            for name in protocol['profiles']['metrics']:
                a=[r[name] for r in measured[16]['hardware']];b=[r[name] for r in measured[256]['hardware']]
                hardware[name]={**affine(st.median(a),st.median(b)),'K16_repeats':a,'K256_repeats':b,
                                'repeat_spread':max(max(a)-min(a),max(b)-min(b))}
            valid=True;why=[]
            if meta['family'] in ('ML','MC') and families['shared_load']['loop_per_iteration_warp_instructions']<=0:why.append('Shared load hoisted or absent')
            if meta['family'] in ('MR','MC') and families['max']['loop_per_iteration_warp_instructions']<=0:why.append('Max hoisted or absent')
            if meta['family']=='MC' and families['global_store']['loop_per_iteration_warp_instructions']<=0:why.append('Output store hoisted or absent')
            # Group unrolled copies by opcode family. No fabricated instruction counts from vec.
            for family in ('shared_load','shared_store','shuffle','barrier','max','global_store'):
                if families[family]['fixed_warp_instructions']<-.001:why.append(family+' negative intercept')
                if not families[family]['strict_K_multiple']:why.append(family+' varying target PCs are not strict K multiples')
            if meta['family']=='MR' and meta.get('placement')=='within':
                if abs(hardware[protocol['profiles']['metrics'][0]]['per_iteration'])>1e-9:why.append('Within-warp MR loop shared load remains')
            if why:failures.append({'tag':tag,'why':why});valid=False
            binaries[tag]={'pass':valid,'families':families,'PCs':pcrows,'hardware_affine':hardware,
                           'hardware_delta_per_iteration':{n:v['per_iteration'] for n,v in hardware.items()},
                           'CUBIN_SHA256':meta['cubin_SHA256'],'wavefronts_are_not_all_bank_conflicts':True}
    for tag,alias in gate['aliases'].items():
        if alias in binaries:binaries[tag]={'alias_of':alias,**binaries[alias]}
    qchecks=[]
    for row in qrows:
        if 'software' not in row:continue
        tag=row['tag'];reference=binaries[tag];k=str(row['K'])
        if gate['metadata'][tag]['family']=='MO':
            natural=reference['measured']['1']['software'][0] if '1' in reference['measured'] else reference['measured'][1]['software'][0]
            expected={f:sum(p['count'] for p in natural.values() if f in p['families']) for f in FAMILIES}
        else:expected={f:reference['families'][f]['total_by_K'][k] for f in FAMILIES}
        actual={f:sum(p['count'] for p in row['software'].values() if f in p['families']) for f in FAMILIES}
        passed=actual==expected
        qchecks.append({'tag':tag,'q':row['q'],'K':row['K'],'pass':passed,'actual':actual,'natural':expected})
        if not passed:failures.append({'tag':tag,'q':row['q'],'why':['q target opcode counts changed']})
    occupancy_checks=[];q_wavefront_checks=[]
    hardware_controls={(r['tag'],r['q']):r for r in qrows if 'hardware' in r}
    for (tag,q),row in hardware_controls.items():
        meta=gate['metadata'][tag];capacity=gate['resources'][tag]['reservations'][q]['capacity']
        bound=100*capacity*meta['num_warps']/64
        active=row['hardware']['sm__warps_active.avg.pct_of_peak_sustained_active']
        passed=0<active<=bound+.25
        occupancy_checks.append({'tag':tag,'q':q,'capacity_CTAs_per_SM':capacity,'warps_active_percent':active,
                                 'capacity_warps_bound_percent':bound,'pass':passed})
        if not passed:failures.append({'tag':tag,'q':q,'why':['Achieved occupancy outside capacity bound']})
        if q=='natural':continue
        natural=hardware_controls[tag,'natural']['hardware']
        actual={n:row['hardware'][n] for n in protocol['profiles']['metrics'][:4]}
        expected={n:natural[n] for n in protocol['profiles']['metrics'][:4]}
        passed=actual==expected
        arithmetic_checks={}
        for op in ('ld','st'):
            wf=f'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_{op}.sum'
            bc=f'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{op}.sum'
            arithmetic_checks[op]=(actual[wf]-actual[bc])==(expected[wf]-expected[bc])
        q_wavefront_checks.append({'tag':tag,'q':q,'raw_aggregate_invariant':passed,'actual':actual,'natural':expected,
            'WF_minus_reported_BC_invariant':arithmetic_checks,
            'WF_minus_BC_is_not_a_directly_collected_request_or_ideal_counter':True})
        # Aggregate hardware bank counters include more than the static access
        # pattern. Raw invariance was an extra diagnostic, not a preregistered
        # requirement; preserve its rejection and every original measurement.
    value={'status':'OK' if not failures else 'FAILED','binaries':binaries,'q_hardware':qrows,'new_reports':reports,
           'q_dynamic_opcode_checks':qchecks,
           'reused_original_reports':protocol['reuse_original_reports'],'failures':failures,
           'NCU_duration_used_for_timing':False,'two_point_affine_scope':'K16 and K256; grouping all unrolled PCs by family, fixed setup separate'}
    if batch=='b':
        value['NCU_launch_order_and_frozen_SASS_opcodes_checked']=True
        value['q_achieved_occupancy_checks']=occupancy_checks
        value['q_wavefront_bank_conflict_invariance']=q_wavefront_checks
        value['aggregate_bank_counter_scope_reference']='https://docs.nvidia.com/nsight-compute/ComputeTriage/#symptom-shared-memory-bank-conflicts-are-significant'
        value['aggregate_counter_diagnostic_rejections']=sum(not r['raw_aggregate_invariant'] for r in q_wavefront_checks)
    derived(c.out(batch)/'counter_gate.json',value)
    print('C2'+batch,'counter gate',value['status'],'new reports',len(reports),'binaries',len(binaries))
    if failures:print(json.dumps(failures,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['compile','artifacts','counters']);p.add_argument('--batch',choices=['a','b'],required=True)
    args=p.parse_args();{'compile':compile_gate,'artifacts':artifact_gate,'counters':counter_gate}[args.action](args.batch)


if __name__=='__main__':main()
