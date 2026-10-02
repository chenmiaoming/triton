"""Offline independent Stage A audit and adversarial regression probes.

Never import a compiler/runtime or observe timings. Source JSON is mixed,
but selection is tested against recursive dictionaries denying nonstructural
keys. Artifact probes modify only in-memory text or temporary synthetic archives.
"""
import ast
import copy
import hashlib
import itertools
import json
import math
import random
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.tma_reduction_layout.phase4 import preregister as p
from experiments.tma_reduction_layout.phase4 import artifact_gate as gate
from experiments.tma_reduction_layout.phase4 import analysis_contract as analysis
from experiments.tma_reduction_layout.gluon import artifact_checks as ac

CHECKS=[]

def check(name,condition):
    if not condition:raise AssertionError(name)
    CHECKS.append(name)
    print('PASS:',name)


def rejected(operation):
    try:operation()
    except (ValueError,KeyError,FileNotFoundError,AssertionError):return True
    return False


ALLOWED={'configs','M','N','num_warps','candidates','default','8','4','2','1',
         'is_legal','observed','ttgir','blocked_encoding','module_attributes',
         'shared_encoding','family','rank','num_ctas','threads_per_warp',*p.FIELDS}

def allowed(key):
    return key in ALLOWED or bool(re.fullmatch(r'M\d+_N\d+_w\d+',str(key)))

class StructuralOnly(dict):
    """Deny indexing/get/iteration of every key outside explicit whitelist."""
    def __getitem__(self,key):
        if not allowed(key):raise AssertionError('Forbidden selection access: '+str(key))
        return super().__getitem__(key)
    def get(self,key,default=None):
        if not allowed(key):raise AssertionError('Forbidden selection access: '+str(key))
        return super().get(key,default)
    def __iter__(self):
        for key in super().__iter__():
            if not allowed(key):raise AssertionError('Forbidden selection iteration: '+str(key))
            yield key
    def keys(self):return list(iter(self))
    def items(self):return [(k,self[k]) for k in self]
    def values(self):return [self[k] for k in self]


def guarded(value):
    if isinstance(value,dict):return StructuralOnly({k:guarded(v) for k,v in value.items()})
    if isinstance(value,list):return [guarded(v) for v in value]
    return value


def poison(value,seed,delete=False):
    """Mutate/delete ALL nonwhitelisted fields, including resources/artifact SHA."""
    rng=random.Random(seed)
    def walk(node):
        if isinstance(node,dict):
            result={}
            for key,item in node.items():
                if not allowed(key):
                    if not delete:result[key]=rng.choice([1e300,-1e300,None,False,{'winner':'POISON','time_us':-999999}])
                else:result[key]=walk(item)
            # Inject typical forbidden keys even when absent from archival schema.
            mapping=all(re.fullmatch(r'M\d+_N\d+_w\d+',str(k)) for k in node) or set(node)=={'default','8','4','2','1'}
            if not delete and not mapping:
                result.update({k:rng.choice([1e300,-1e300,'POISON']) for k in ('winner','timing','slope','speedup','regret','marginal_ns','time_us')})
            return result
        if isinstance(node,list):return [walk(x) for x in node]
        return node
    return walk(value)


def independent_pool(source,pool):
    # Independent rules directly from the source; no selector/coverage helpers.
    expected_domain={(m,n,w) for m in (32,64,128) for n in (16,32,64,128,256) for w in (4,8)}
    check('entire 30-configuration Cartesian domain exactly once',
          {(c['M'],c['N'],c['num_warps']) for c in source['configs'].values()}==expected_domain
          and len(pool['transitions'])==len(source['configs'])==30
          and len({r['config_id'] for r in pool['transitions']})==30)
    counters={k:Counter() for k in ('M','N','num_warps','origin','lanePart_M_transitions','warpPart_M_values','sizePerThread_transitions')}
    reasons=Counter();included=0
    for row in pool['transitions']:
        cfg=source['configs'][row['config_id']];m,n,w=cfg['M'],cfg['N'],cfg['num_warps']
        actual=[];layouts=[]
        for name,key in (('default','default'),('cand4','4')):
            c=cfg['candidates'][key]
            legal=True if key=='default' else 4<=min(8,max(m*n//(w*32),1)) and n%4==0
            assert c['is_legal']==legal==row[name]['legal']
            if not legal:
                actual.append('CAND4_ILLEGAL' if key=='4' else 'DEFAULT_ILLEGAL');layouts.append(None);continue
            t=c['observed']['ttgir'];layout={f:t['blocked_encoding'][f] for f in p.FIELDS}
            assert row[name]['layout']==layout
            factors=[layout[k] for k in p.FIELDS[:3]]
            valid=all(len(v)==3 and all(type(x) is int and x>0 and x&(x-1)==0 for x in v) for v in factors)
            valid=valid and sorted(layout['order'])==[0,1,2] and math.prod(factors[1])==32 and math.prod(factors[2])==w
            valid=valid and all(v[0]==1 for v in factors) and t['shared_encoding']['family']=='nvmma_shared' and t['shared_encoding']['rank']==3
            assert valid and t['module_attributes']=={'num_ctas':1,'num_warps':w,'threads_per_warp':32}
            assert row[name]['vec']==factors[0][2] and row[name]['lanePart_M']==factors[1][1] and row[name]['warpPart_M']==factors[2][1]
            layouts.append(layout)
        d,c=layouts
        if d and c:
            if d==c:actual.append('IDENTICAL_LAYOUT')
            if d['warpsPerCTA'][1]!=c['warpsPerCTA'][1]:actual.append('WARP_PART_CHANGED')
            if d['threadsPerWarp'][1]<=c['threadsPerWarp'][1]:actual.append('LANE_PART_NOT_REDUCED')
        assert set(actual)==set(row['reason_codes']) and row['included_in_structural_pool']==(not actual)
        assert row['logical_shape']==[1,m,n] and row['reduction_axis']==1 and row['num_warps']==w
        origin='ANCHOR_EXISTING' if row['config_id'] in ('M32_N64_w8','M32_N128_w4') else 'NEW_GENERALIZATION'
        assert row['origin']==origin
        reasons.update(actual)
        if not actual:
            included+=1
            for k,value in [('M',m),('N',n),('num_warps',w),('origin',origin),
                            ('lanePart_M_transitions',f"{d['threadsPerWarp'][1]}->{c['threadsPerWarp'][1]}"),
                            ('warpPart_M_values',d['warpsPerCTA'][1]),
                            ('sizePerThread_transitions',f"{d['sizePerThread']}->{c['sizePerThread']}")]:counters[k][str(value)]+=1
    check('independent legality, all layouts, A-H membership, exclusion reasons and anchors',included==pool['included_structural_pool']==20 and pool['excluded_structural_pool']==10)
    check('independent coverage and nonexclusive reason counts',all(dict(v)==pool['coverage'][k] for k,v in counters.items()) and all(reasons[r]==pool['reason_counts_nonexclusive'][r] for r in p.REASONS))
    check('150 source combinations; six N256 exclusions are structural',pool['source_combinations']==sum(len(c['candidates']) for c in source['configs'].values())==150 and len(pool['coverage']['N256_audit'])==6 and all(set(r['reason_codes'])=={'WARP_PART_CHANGED','LANE_PART_NOT_REDUCED'} for r in pool['coverage']['N256_audit']))


def structural_probes(source,pool):
    baseline=p.dump(pool)
    check('deny all performance/resource-key access',p.dump(p.create_pool(guarded(source)))==baseline)
    for seed in (0,73,20261002):
        check(f'extreme randomized poison seed {seed}: bit-identical complete pool',p.dump(p.create_pool(guarded(poison(source,seed))))==baseline)
    check('deletion of every nonstructural key: bit-identical complete pool',p.dump(p.create_pool(guarded(poison(source,0,True))))==baseline)
    rows,_=p.structural_projection(source,p.kernel_contract())
    i=next(i for i,r in enumerate(rows) if r['config_id']=='M32_N64_w8')
    for reason,mutate in (
        ('DEFAULT_ILLEGAL',lambda r:r['default'].update(legal=False)),
        ('CAND4_ILLEGAL',lambda r:r['cand4'].update(legal=False)),
        ('REDUCTION_AXIS_MISMATCH',lambda r:r.update(reduction_axis=2)),
        ('WARP_COUNT_MISMATCH',lambda r:r['cand4'].update(compiled_num_warps=4)),
        ('UNSUPPORTED_LAYOUT',lambda r:r['cand4']['layout'].update(order=[1,1,2])),
        ('UNSUPPORTED_LAYOUT',lambda r:r['cand4']['layout'].update(threadsPerWarp=[])),
        ('IDENTICAL_LAYOUT',lambda r:r.update(cand4=copy.deepcopy(r['default']))),
        ('WARP_PART_CHANGED',lambda r:r['cand4']['layout'].update(warpsPerCTA=[1,4,2])),
        ('LANE_PART_NOT_REDUCED',lambda r:r['cand4']['layout'].update(threadsPerWarp=[1,4,8]))):
        changed=copy.deepcopy(rows);mutate(changed[i]);selected=p.select(changed)[i]
        check('structural negative probe: '+reason,not selected['included_in_structural_pool'] and reason in selected['reason_codes'])
    incomplete=copy.deepcopy(source);incomplete['configs'].pop('M128_N256_w8')
    check('missing source transition fails closed',rejected(lambda:p.create_pool(incomplete)))
    missing=copy.deepcopy(source);missing['configs']['M32_N64_w8']['candidates']['4']['observed'].pop('ttgir')
    row=next(r for r in p.create_pool(missing)['transitions'] if r['config_id']=='M32_N64_w8')
    check('missing legal layout recorded as unsupported, not silently admitted',not row['included_in_structural_pool'] and 'UNSUPPORTED_LAYOUT' in row['reason_codes'])


def archive_probes():
    source=(p.BASE/'gluon/kernel_repeated.py').read_text()
    for cfg in p.ANCHORS:
        single_audits,repeated_audits={},{}
        shape=[1,32,64 if cfg==p.ANCHORS[0] else 128];w=8 if shape[2]==64 else 4
        for candidate in ('default','4'):
            def load(root):return {ext:(root/f'{candidate}.{ext}').read_text() for ext in ('ptx','ttgir','sass','resource.txt')}
            canon=load(p.BASE/'results/phase3/fixed_binary_artifacts/canonical'/cfg)
            single=load(p.BASE/'results/phase3/gluon_reproduction/artifacts'/cfg)
            bundle=load(p.BASE/'results/phase3/gluon_repeated/artifacts'/cfg)
            a=gate.reproduction(canon,single,shape,w)
            b=gate.repeated(canon,bundle,shape,w,source)
            expected='PIPELINED_OPCODE_EQUIVALENT' if cfg==p.ANCHORS[1] and candidate=='default' else 'EXACT_SEQUENCE_EQUIVALENT'
            single_audits[candidate]=a;repeated_audits[candidate]=b
            check(f'archival full single and repeated body {cfg}/{candidate}',a['structure_pass'] and a['classification']=='EXACT_SEQUENCE_EQUIVALENT' and b['structure_pass'] and b['canonical_local_load_match']==('DIFFERENT_ENCODING' if candidate=='4' else 'EXACT') and b['classification']==expected and gate.body_fingerprint(canon['ptx'])==ac.canonical_fingerprint(p.BASE,cfg,candidate))
        check(f'historical {cfg} body/isolation tier without config-name exceptions',gate.body_gate_tier(single_audits,repeated_audits)==('PRIMARY' if cfg==p.ANCHORS[0] else 'SECONDARY'))
    # Primary/default memory fixtures: every mutation affects only a Python string.
    cfg=p.ANCHORS[0];candidate='default';shape=[1,32,64];w=8
    canon=load(p.BASE/'results/phase3/fixed_binary_artifacts/canonical'/cfg)
    single=load(p.BASE/'results/phase3/gluon_reproduction/artifacts'/cfg)
    bundle=load(p.BASE/'results/phase3/gluon_repeated/artifacts'/cfg)
    loads=ac.initial_load_signature(single['ptx']);changed=copy.deepcopy(single)
    lines=changed['ptx'].splitlines();target=loads[-1];lines[target['line']-1]=lines[target['line']-1].replace(target['opcode'],'ld.shared.b16');changed['ptx']='\n'.join(lines)
    check('late LocalLoad width mismatch rejected',not gate.reproduction(canon,changed,shape,w)['localload_match'])
    loop=next(e for e in ac.ptx_backedges(bundle['ptx']) if e['kind']=='compiler_loop')
    def insert(text,at,extra):
        lines=text.splitlines();lines.insert(at,extra);return '\n'.join(lines)
    changed=copy.deepcopy(bundle)
    original_lines=bundle['ttgir'].splitlines()
    load_line=next(line for line in original_lines if 'ttg.local_load' in line)
    changed['ttgir']=changed['ttgir'].replace(load_line+'\n','',1)
    for_header=next(line for line in changed['ttgir'].splitlines() if 'scf.for' in line)
    changed['ttgir']=changed['ttgir'].replace(for_header,for_header+'\n'+load_line,1)
    check('TTGIR LocalLoad moved into runtime loop fails',not gate.repeated(canon,changed,shape,w,source)['checks']['one_preloop_localload_no_tile_reload'])
    changed=copy.deepcopy(bundle)
    load=ac.initial_load_signature(changed['ptx'])[0]
    changed['ptx']=changed['ptx'].replace(load['opcode'],'ld.shared.v4.b16',1)
    changed_audit=gate.repeated(canon,changed,shape,w,source)
    check('different but wrong pre-loop payload remains MISMATCH',changed_audit['canonical_local_load_match']=='MISMATCH' and not changed_audit['structure_pass'])
    # Replace one v4.b32 load with two v4.b16 loads: same complete byte payload.
    changed=copy.deepcopy(bundle)
    replacement='ld.shared.v4.b16 {%rs9, %rs10, %rs11, %rs12}, [%r20];\nld.shared.v4.b16 {%rs13, %rs14, %rs15, %rs16}, [%r20+8];'
    lines=changed['ptx'].splitlines();lines[load['line']-1]=replacement;changed['ptx']='\n'.join(lines)
    changed_audit=gate.repeated(canon,changed,shape,w,source)
    check('one-time pre-loop encoding change alone does not force EXCLUDE',changed_audit['canonical_local_load_match']=='DIFFERENT_ENCODING' and changed_audit['structure_pass'])
    pool=p.create_pool(p.read_json(p.SWEEP));expected_pool=p.dump(pool)
    altered=p.read_json(p.SWEEP)
    altered['configs'][cfg]['candidates']['4']['observed']['ptx']={'initial_localload_opcodes':['ld.shared.v4.b16']}
    check('repeated pre-loop encoding metadata does not alter structural membership',p.dump(p.create_pool(altered))==expected_pool)
    runtime_archive={'texts':{'ptx':changed['ptx']},'hashes':{'cubin':'SYNTHETIC_CUBIN','ptx':'SYNTHETIC_PTX'}}
    records=[{'R':r,'B_RUN':b,'cubin_sha256':'SYNTHETIC_CUBIN','ptx_sha256':'SYNTHETIC_PTX','pre_loop_localload_opcodes':['ld.shared.v4.b16','ld.shared.v4.b16']} for r in (0,1) for b in (16384,32768,65536)]
    check('different pre-loop encoding admitted with R0 and one planned binary',gate.runtime_binary_contract(records,runtime_archive)['fixed_binary_across_R_and_B'])
    check('missing R0 runtime plan fails',rejected(lambda:gate.runtime_binary_contract([record for record in records if record['R']==1],runtime_archive)))
    for field in ('cubin_sha256','ptx_sha256','pre_loop_localload_opcodes'):
        altered=copy.deepcopy(records);altered[-1][field]=['ld.shared.v2.b32'] if field=='pre_loop_localload_opcodes' else 'CHANGED'
        check('R-dependent '+field+' fails',rejected(lambda:gate.runtime_binary_contract(altered,runtime_archive)))
    for name,extra,contract in (
        ('extra whole-loop shared load','ld.shared.b32 %r999, [%r998];','one_preloop_localload_no_tile_reload'),
        ('float accumulator','add.f32 %f999, %f998, %f997;','no_accumulator_or_global_effect'),
        ('loop global store','st.global.b32 [%rd998], %r999;','no_accumulator_or_global_effect')):
        changed=copy.deepcopy(bundle);changed['ptx']=insert(bundle['ptx'],loop['start'],extra)
        check(name+' rejected',not gate.repeated(canon,changed,shape,w,source)['checks'][contract])
    changed=copy.deepcopy(bundle);changed['resource.txt']=changed['resource.txt'].replace('LOCAL:0','LOCAL:4')
    check('actual spill rejected',not gate.repeated(canon,changed,shape,w,source)['checks']['zero_spills'])
    check('runtime specialization rejected',not gate.repeated(canon,bundle,shape,w,source.replace('do_not_specialize','removed_specialization_flag'))['checks']['runtime_R_unspecialized'])
    changed=copy.deepcopy(bundle);changed['ptx']=insert(bundle['ptx'],loop['end']+1,'ld.shared.b32 %r999, [%r998];')
    check('post-loop terminal shared reload rejected',not gate.repeated(canon,changed,shape,w,source)['checks']['terminal_exchanges_inside'])
    changed=copy.deepcopy(bundle)
    pairs=gate.repeated(canon,bundle,shape,w,source)['copy_pairs']
    changed['ptx']=changed['ptx'].replace(pairs[1][0]+', '+pairs[1][1],pairs[1][0]+', '+pairs[0][1],1)
    check('duplicate tied-copy source rejected',not gate.repeated(canon,changed,shape,w,source)['checks']['input_copies_one_to_one'])
    for op in ('MOV','IMAD.MOV.U32'):
        changed=copy.deepcopy(bundle);audit=ac.inspect_sass(changed['sass']);address=int(audit['loop_start_addr'],16)
        changed['sass']=re.sub(r'(/\*'+f'{address:04x}'+r'\*/\s+).*?;',lambda m:m[1]+op+' R99, R98;',changed['sass'],count=1)
        check('explicit SASS '+op+' rejected',not gate.repeated(canon,changed,shape,w,source)['checks']['no_explicit_sass_mov_observed'])
    check('multiset equality is secondary; complete extra opcode mismatches',gate.equivalence(['max.f32','bar.sync 0'],['bar.sync 0','max.f32'])=='PIPELINED_OPCODE_EQUIVALENT' and gate.equivalence(['max.f32'],['max.f32','bar.sync 0'])=='REDUCTION_FINGERPRINT_MISMATCH')
    check('shuffle hex immediates remain distinct',ac.normalize('shfl.sync.bfly.b32 %r1, %r2, 0x1, 31, -1;')!=ac.normalize('shfl.sync.bfly.b32 %r1, %r2, 0x2, 31, -1;'))
    check('historical missing actual CUBIN cannot close archive',rejected(lambda:gate.exact_archive({'paths':{},'SHA256':{}})))
    synthetic_archive_probes()


def analysis_probes():
    check('deterministic one-based average ranks for exact ties',analysis.average_ranks([10,10,30,40])==[1.5,1.5,3.0,4.0] and analysis.average_ranks([40,20,20,10])==[4.0,2.5,2.5,1.0])
    check('Spearman is Pearson of average ranks, not untied d-squared formula',math.isclose(analysis.spearman([10,10,30,40],[40,20,20,10]),-5/6,abs_tol=1e-15))
    check('continuous means ranked without sign discretization',analysis.spearman([.1,.2,.3,.4],[2,4,6,8])==1.0)
    check('constant or fewer-than-three association is undefined',analysis.spearman([1,1,1],[1,2,3]) is None and analysis.spearman([1,2],[1,2]) is None)
    cases={name:(i,i*2) for i,name in enumerate(('a','b','c','d','e'))}
    sensitivity=analysis.leave_one_out_spearman(cases)
    check('all PRIMARY leave-one-out cases retained; min/max/median reported',sensitivity=={'rho_minus_i':{name:1.0 for name in cases},'min':1.0,'max':1.0,'median':1.0,'defined_count':5,'undefined_count':0} and len(cases)==5)
    small=analysis.leave_one_out_spearman({'a':(1,2),'b':(2,3),'c':(3,4)})
    check('leave-one-out undefined cases remain reported',small['defined_count']==0 and small['undefined_count']==3 and small['min'] is small['max'] is small['median'] is None)
    interval=analysis.sign_resolution([1.0,2.0,3.0])
    check('SIGN_UNRESOLVED uses three-pair sample SD and df2 Student-t band',interval['category']=='SIGN_UNRESOLVED' and interval['mean']==2 and interval['sample_SD']==1 and math.isclose(interval['half_width'],4.302652729911275/math.sqrt(3)) and interval['interval'][0]<0<interval['interval'][1])
    check('sign band reproducible under paired-value permutations',all(analysis.sign_resolution(values)==interval for values in itertools.permutations([1.0,2.0,3.0])))
    check('resolved positive/negative and exact-zero categories',analysis.sign_resolution([1,1,1])['category']=='POSITIVE' and analysis.sign_resolution([-1,-1,-1])['category']=='NEGATIVE' and analysis.sign_resolution([0,0,0])['category']=='SIGN_UNRESOLVED')
    check('sign category never zeros continuous metric',interval['mean']==2)
    check('invalid replication count, categorical values and nonfinite inputs rejected',rejected(lambda:analysis.sign_resolution([1,2])) and rejected(lambda:analysis.spearman(['POSITIVE']*3,[1,2,3])) and rejected(lambda:analysis.sign_resolution([1,2,float('nan')])))


def synthetic_archive_probes():
    # Deliberately synthetic bytes test schema/binding only: not a real GPU binary.
    with tempfile.TemporaryDirectory(prefix='phase4-schema-test-') as folder:
        folder=Path(folder);source=folder/'kernel.py'
        source.write_text('def kernel(x):\n    y = x.to(gl.float32)\n    return gl.max(y, axis=1)\n')
        sha=lambda b:hashlib.sha256(b).hexdigest()
        resources={'num_regs':32,'local_bytes':0,'stack_bytes':0,'static_smem_bytes':1024}
        source_sha=sha(source.read_bytes())
        stage={'file_id':1,'source_lines':[2,3],'source_path':str(source),'source_sha256':source_sha,'kernel_function':'kernel','ptx_source_path':'/build/kernel.py'}
        metadata={'logical_shape':[1,32,64],'num_warps':8,'dynamic_shared_bytes':4104,'resources':resources,'reduction_stage':stage}
        source_manifest={'files':[{'path':str(source),'SHA256':source_sha,'ptx_source_path':'/build/kernel.py'}]}
        blobs={key:b'SYNTHETIC-NOT-A-COMPILER-ARTIFACT' for key in gate.ARTIFACT_KEYS}
        blobs.update({'cubin':b'\x7fELF_SYNTHETIC_SCHEMA_PROBE','resource.txt':b'REG:32 LOCAL:0 STACK:0 SHARED:1024','ptx':b'.file 1 "/build/kernel.py"\n'})
        blobs['cubin.sha256']=sha(blobs['cubin']).encode()
        blobs['compile_metadata.json']=json.dumps(metadata).encode();blobs['source_manifest.json']=json.dumps(source_manifest).encode()
        env={'gpu_uuid':'SYNTHETIC','driver':'SYNTHETIC','CUDA':'SYNTHETIC','toolchain':'SYNTHETIC'}
        provenance={'modal_image':'SYNTHETIC','build_identity':'SYNTHETIC','toolchain':'SYNTHETIC','source_manifest_sha256':sha(blobs['source_manifest.json']),
                    'export_SHA256':{key:sha(blobs[key]) for key in ('ttgir','ptx','sass','resource.txt','cubin','cubin.sha256','compile_metadata.json')}}
        occupancy={'query_api':'cudaOccupancyMaxActiveBlocksPerMultiprocessor','queried_cubin_sha256':sha(blobs['cubin']),'resource_sha256':sha(blobs['resource.txt']),
                   'num_warps':8,'block_threads':256,'dynamic_shared_bytes':4104,'blocks_per_sm':4,'active_warps_per_sm':32,'resources':resources,'gpu_uuid':'SYNTHETIC','build_identity':'SYNTHETIC'}
        for key,data in [('environment.json',env),('build_provenance.json',provenance),('occupancy.json',occupancy)]:blobs[key]=json.dumps(data).encode()
        def write(current):
            for key,data in current.items():(folder/key).write_bytes(data)
            return {'paths':{k:str(folder/k) for k in current},'SHA256':{k:sha(v) for k,v in current.items()}}
        check('synthetic archive schema accepts internally bound records (not binary authenticity)',gate.exact_archive(write(blobs))['resources']==resources)
        changed_stage=dict(stage,source_lines=[3])
        check('source-stage matching window cannot omit convert',rejected(lambda:gate.source_stage({'reduction_stage':changed_stage},source_manifest,blobs['ptx'].decode())))
        check('PTX source file mapping mismatch rejected',rejected(lambda:gate.source_stage(metadata,source_manifest,'.file 1 "/wrong/kernel.py"')))
        cases=[('wrong queried binary','occupancy.json','queried_cubin_sha256','0'*64),
               ('wrong queried resource','occupancy.json','resources',dict(resources,num_regs=64)),
               ('wrong occupancy threads','occupancy.json','block_threads',128),
               ('wrong occupancy launch shared memory','occupancy.json','dynamic_shared_bytes',8200),
               ('wrong GPU provenance','occupancy.json','gpu_uuid','OTHER'),
               ('wrong export provenance','build_provenance.json','export_SHA256',{}),
               ('wrong source provenance','build_provenance.json','source_manifest_sha256','0'*64)]
        for name,key,field,value in cases:
            changed=dict(blobs);obj=json.loads(changed[key]);obj[field]=value;changed[key]=json.dumps(obj).encode()
            check('archive rejects '+name,rejected(lambda:gate.exact_archive(write(changed))))
        changed=dict(blobs);changed['cubin']=b'HASH-ONLY-NOT-ELF';changed['cubin.sha256']=sha(changed['cubin']).encode()
        check('hash-only CUBIN rejected',rejected(lambda:gate.exact_archive(write(changed))))
        source.write_text(source.read_text()+'# mutated\n')
        check('mutated source bytes rejected',rejected(lambda:gate.exact_archive(write(blobs))))


def validate_schedule(pool, schedule, policy):
    """Validate semantic Cartesian domain, identities, rotations and sample counts.

    Expected dimensions come from the scientific protocol and structural pool;
    observed cardinalities come independently from serialized entries/orders.
    """
    sampling=policy['future_sampling']
    included={row['config_id'] for row in pool['transitions'] if row['included_in_structural_pool']}
    candidates=('default','4')
    b_values=sampling['B_RUN'];r_values=sampling['R']
    assert len(b_values)==len(set(b_values)) and len(r_values)==len(set(r_values))
    n_cases=len(included);n_candidates=len(candidates);n_B=len(b_values)
    n_invocations=sampling['invocations'];n_rounds=sampling['rounds'];samples_per_round=sampling['samples_per_round']
    assert sampling['samples_per_condition']==n_rounds*samples_per_round
    master=schedule['master_conditions'];expected_master={};ids=[]
    for cfg in sorted(included):
        for harness in ('canonical','repeated'):
            for candidate in candidates:
                for r in ([None] if harness=='canonical' else r_values):
                    for b in b_values:
                        identifier=f'{cfg}:{harness}:{candidate}:R{r}:B{b}'
                        ids.append(identifier)
                        expected_master[identifier]={'config_id':cfg,'harness':harness,'candidate':candidate,'R':r,'B_RUN':b}
    assert master==expected_master, 'Missing, duplicated or altered case/candidate/R/B/harness entry'
    assert schedule['execution']=='FUTURE_ONLY_NO_SAMPLES'
    expected_per_harness={harness:n_cases*n_candidates*n_B*len(rs)
                          for harness,rs in [('canonical',[None]),('repeated',r_values)]}
    conditions_per_invocation=sum(expected_per_harness.values())
    assert len(master)==conditions_per_invocation
    assert len(schedule['invocations'])==n_invocations
    visits=Counter();sample_counts=Counter()
    for i,invocation in enumerate(schedule['invocations'],1):
        assert invocation['invocation']==i and len(invocation['rounds'])==n_rounds
        for j,round_ in enumerate(invocation['rounds'],1):
            shift=((i-1)*97+(j-1)*37)%conditions_per_invocation
            expected=ids[shift:]+ids[:shift]
            if i%2==0:expected.reverse()
            assert round_=={'round':j,'shift':shift,'order':expected,'samples_per_visit':samples_per_round}, 'Missing/duplicated entry or altered frozen round order'
            for identifier in round_['order']:
                visits[(i,identifier)]+=1
                sample_counts[(i,identifier)]+=round_['samples_per_visit']
    assert len(visits)==conditions_per_invocation*n_invocations
    assert all(count==n_rounds for count in visits.values())
    assert all(count==n_rounds*samples_per_round for count in sample_counts.values())
    accounting=policy['schedule_cardinality']
    assert accounting['master_case_count']==n_cases
    assert set(accounting['by_harness'])==set(expected_per_harness)
    assert accounting['invocations']==n_invocations
    assert accounting['rounds_per_invocation']==[n_rounds]*n_invocations
    assert accounting['samples_per_visit']==[samples_per_round]
    assert accounting['stored_candidate_paired_scheduling_units']==0
    for harness,rs in [('canonical',[None]),('repeated',r_values)]:
        counts=accounting['by_harness'][harness]
        observed=[entry for entry in master.values() if entry['harness']==harness]
        actual_conditions=len(observed);expected_conditions=expected_per_harness[harness]
        assert actual_conditions==expected_conditions
        observed_visits=sum(count for (i,key),count in visits.items() if master[key]['harness']==harness)
        observed_samples=sum(count for (i,key),count in sample_counts.items() if master[key]['harness']==harness)
        assert counts['cases']==n_cases and counts['candidates']==n_candidates
        assert counts['R_values']==sorted(rs,key=str) and counts['B_RUN_values']==sorted(b_values)
        assert counts['measurement_conditions_per_invocation']==actual_conditions
        assert counts['candidate_level_invocation_conditions']==actual_conditions*n_invocations
        assert counts['round_order_visits']==observed_visits==expected_conditions*n_invocations*n_rounds
        assert counts['planned_scalar_timing_samples']==observed_samples==expected_conditions*n_invocations*n_rounds*samples_per_round
    assert accounting['total_measurement_conditions_per_invocation']==conditions_per_invocation
    assert accounting['total_candidate_level_invocation_conditions']==conditions_per_invocation*n_invocations
    assert accounting['total_round_order_visits']==sum(visits.values())
    assert accounting['total_planned_scalar_timing_samples']==sum(sample_counts.values())
    return accounting


def audit_schedule(pool,schedule,policy):
    validate_schedule(pool,schedule,policy)
    check('dimension-derived complete canonical/repeated measurement domains and identities',True)
    check('invocation/round visits and scalar samples independently reconcile',True)
    check('all original complete rotations retained with no paired scheduling units',True)
    identifier=next(key for key,value in schedule['master_conditions'].items() if value['harness']=='repeated')
    changed=copy.deepcopy(schedule);changed['master_conditions'].pop(identifier)
    check('schedule deletion of one (case,candidate,R,B) entry fails',rejected(lambda:validate_schedule(pool,changed,policy)))
    changed=copy.deepcopy(schedule);changed['master_conditions'][identifier+':duplicate']=copy.deepcopy(changed['master_conditions'][identifier])
    check('schedule duplication of one (case,candidate,R,B) entry fails',rejected(lambda:validate_schedule(pool,changed,policy)))
    for field,value in [('config_id','MISSING_CASE'),('candidate','MISSING_CANDIDATE'),('R',99),('B_RUN',99)]:
        changed=copy.deepcopy(schedule);changed['master_conditions'][identifier][field]=value
        check('schedule altered '+field+' semantic identity fails',rejected(lambda:validate_schedule(pool,changed,policy)))
    changed=copy.deepcopy(schedule);changed['invocations'][0]['rounds'][0]['order'].remove(identifier)
    check('round-order missing measurement visit fails',rejected(lambda:validate_schedule(pool,changed,policy)))
    changed=copy.deepcopy(schedule);changed['invocations'][0]['rounds'][0]['order'].append(identifier)
    check('round-order duplicate measurement visit fails',rejected(lambda:validate_schedule(pool,changed,policy)))
    changed=copy.deepcopy(schedule);changed['invocations'][0]['rounds'][0]['samples_per_visit']+=1
    check('sample count cannot be confused with schedule entries',rejected(lambda:validate_schedule(pool,changed,policy)))


def main():
    outputs={path.name for path in p.OUTPUT.iterdir() if path.is_file()}
    check('output whitelist; no timing outputs',outputs==set(p.OUTPUT_NAMES) and not any(path.is_dir() for path in p.OUTPUT.iterdir()))
    pool=p.read_json(p.OUTPUT/'structural_pool.json');source=p.read_json(p.SWEEP)
    bindings=p.read_json(p.OUTPUT/'source_bindings.json')
    for entry in bindings['files']:
        path=ROOT/entry['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['SHA256']
        if entry['git_head']:
            original=subprocess.check_output(['git','show',f"{entry['git_head']}:{entry['path']}"],cwd=ROOT)
            assert original==path.read_bytes() and entry['git_head']==p.BASELINE
    check('all source/code/protocol SHAs and archived Git bindings unchanged',bindings['source_git_head']==p.BASELINE and len({e['path'] for e in bindings['files']})==len(bindings['files']))
    regenerated=p.generate()
    check('all nine outputs independently replay byte-for-byte',all((p.OUTPUT/name).read_text()==text for name,text in regenerated.items()))
    independent_pool(source,pool);structural_probes(source,pool);archive_probes();analysis_probes()
    policy=p.read_json(p.OUTPUT/'protocol.json');sampling=policy['future_sampling']
    audit_schedule(pool,p.read_json(p.OUTPUT/'rotation_schedule.json'),policy)
    check('frozen future B/R, sample counts and invocations',all(sampling[k]==v for k,v in {'B_DESC':65536,'B_RUN':[16384,32768,65536],'R':[0,1],'rounds':10,'samples_per_round':10,'samples_per_condition':100,'invocations':3}.items()))
    t=math.sqrt(2*.95**2/(1-.95**2))
    check('uncertainty band coefficient is exact df=2 t quantile',math.isclose(t,4.302652729911275,abs_tol=1e-12))
    check('all nine signed counterexample categories retained',set(policy['counterexamples'])=={f'{a}/{b}' for a,b in itertools.product(('POSITIVE','SIGN_UNRESOLVED','NEGATIVE'),repeat=2)})
    check('machine sign label and interpretability/LOO policy frozen',policy['sign_resolution']['machine_label']=='SIGN_UNRESOLVED' and 'n<5' in policy['association']['minimum_interpretability'] and 'rho_minus_i' in policy['association']['leave_one_out'] and 'no sign-category discretization' in policy['association']['primary'])
    for name in ('structural_pool.json','exclusions.json','rotation_schedule.json'):
        path=p.OUTPUT/name
        expected=p.ORIGINAL_STAGE_A_SHA256[name]
        check('original Stage A artifact preserved byte-for-byte: '+name,hashlib.sha256(path.read_bytes()).hexdigest()==expected and bindings['original_stage_a_preserved_SHA256'][name]==expected)
    inventory=p.read_json(p.OUTPUT/'canonical_eligibility.json')
    check('all 30 archival records; 20 structural cases remain pending and zero fresh eligibility',len(inventory['entries'])==30 and inventory['pre_timing_eligible_count']==0 and all(not e['pre_timing_eligible'] for e in inventory['entries']))
    # All baseline paths remain untouched, including original samples/compiler artifacts.
    changed=subprocess.check_output(['git','diff','--name-only',p.BASELINE],cwd=ROOT,text=True).splitlines()
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
    prefixes=('experiments/tma_reduction_layout/phase4/','experiments/tma_reduction_layout/results/phase4/preregistration/')
    check('baseline unchanged outside authorized Phase 4 additions',all(path.startswith(prefixes) for path in changed+untracked))
    prohibited={'triton','torch','modal','cuda','cupy','pycuda'}
    for name in ('preregister.py','artifact_gate.py','analysis_contract.py','validate_preregistration.py'):
        tree=ast.parse((p.BASE/'phase4'/name).read_text())
        imported=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):imported.extend(alias.name.split('.')[0] for alias in node.names)
            elif isinstance(node,ast.ImportFrom):imported.append((node.module or '').split('.')[0])
        assert not set(imported)&prohibited
    check('Stage A modules import no GPU/runtime/compiler/Modal package',True)
    print(f'PHASE 4 STAGE A VALIDATOR: {len(CHECKS)}/{len(CHECKS)} PASS; performance-field mutation changes cohort: NO')


if __name__=='__main__':main()
