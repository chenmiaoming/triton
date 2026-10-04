"""Independent byte/source/body/resource admission for the 2x2 experiment."""
from collections import Counter
import copy
import json
from pathlib import Path
import re
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.tma_reduction_layout.phase7 import common as c, contracts as ct
from experiments.tma_reduction_layout.phase4 import artifact_gate as ag
from experiments.tma_reduction_layout.gluon import artifact_checks as ac
from experiments.tma_reduction_layout.source_provenance import compute_manifest_digest

DEST=ct.GATE
DERIVED={"gate_results.json","launch_stage_c.json","launch_stage_d.json","summary.md","validation.json","raw_manifest.json"}


def raw_inventory():
    return {p.relative_to(DEST).as_posix():c.sha(p.read_bytes()) for p in sorted(DEST.rglob('*')) if p.is_file() and p.name not in DERIVED}


def sources():
    prov,env,bindings,dispatch=[c.read(DEST/n) for n in ('local_source_provenance.json','environment.json','source_bindings.json','modal_dispatch.json')]
    manifest=prov['source_manifest']
    c.require(compute_manifest_digest(manifest)==prov['source_manifest_sha256']==env['source_verification']['remote_source_subset_sha256']==bindings['uploaded_source_subset_sha256'],'Complete source manifest closure')
    c.require(c.sha((DEST/'uploaded_source.zip').read_bytes())==bindings['uploaded_source_archive_sha256'],'Original source ZIP')
    with zipfile.ZipFile(DEST/'uploaded_source.zip') as z:
        c.require(set(z.namelist())==set(manifest) and len(z.namelist())==len(set(z.namelist())),'Complete unique source snapshot')
        for name,digest in manifest.items():c.require(c.sha(z.read(name))==digest,'Source bytes: '+name)
    for name in ('pool.json','protocol.json'):
        c.require((DEST/('frozen_'+name)).read_bytes()==(ct.DEST/name).read_bytes(),'Frozen preregistration: '+name)
    c.require(c.sha((DEST/'raw_export.zip').read_bytes())==dispatch['raw_export_zip_sha256'],'Original compiler return ZIP')
    with zipfile.ZipFile(DEST/'raw_export.zip') as z:
        for name in z.namelist():c.require(z.read(name)==(DEST/name).read_bytes(),'Original compiler export: '+name)
    c.require('H100' in env['gpu_name'] and env['compute_capability']==[9,0] and env['no_performance_observation'] is True,'Strict H100, no timing')
    old_dispatch=c.read(c.PRIOR/'stage_d_gate/modal_dispatch.json')
    old_env=c.read(c.PRIOR/'stage_d_gate/environment.json')
    c.require(dispatch['core_image_id']==old_dispatch['core_image_id'],'Exact completed native image reused')
    for name in ('actual_compiler_ptxas','built_triton_native_extension'):
        c.require(env['toolchain'][name]['sha256']==old_env['toolchain'][name]['sha256'],'Actual compiler/native identity reused')
    return manifest,env,dispatch


def bundle(case,h,k,manifest,env):
    path=DEST/h/case['config_id']/k
    attempt=c.read(path/'attempt.json')
    actual={p.name:c.sha(p.read_bytes()) for p in sorted(path.iterdir()) if p.is_file() and p.name!='attempt.json'}
    c.require(attempt['artifact_SHA256']==actual,'Every original attempt file SHA')
    c.require((attempt['case_id'],attempt['harness'],attempt['candidate'],attempt['attempted'])==(case['config_id'],h,k,True),'Actual attempt identity')
    if attempt['status']!='EXPORTED':
        c.require(attempt['error']==(path/'error.txt').read_text(),'Complete failure retained')
        reasons=[attempt['failure_kind']] if attempt['status']=='FAILED_DETERMINISTIC' else ['PENDING']
        return {'complete':False,'reasons':reasons,'status':attempt['status']}
    meta,occupancy,smoke=[c.read(path/n) for n in ('metadata.json','occupancy.json','smoke.json')]
    cubin=(path/'kernel.cubin').read_bytes();digest=c.sha(cubin)
    c.require(cubin.startswith(b'\x7fELF') and digest==meta['cubin_sha256']==occupancy['cubin_sha256']==occupancy['queried_cubin_sha256'],'Actual CUBIN/occupancy identity')
    c.require((path/'kernel.cubin.sha256').read_text().strip()==digest,'CUBIN companion SHA')
    ptx,ir=[(path/('kernel.'+n)).read_text() for n in ('ptx','ttgir')]
    c.require(ct.abi(ptx)==ct.expected_abi(h),'Complete supported PTX ABI')
    stage=meta['reduction_stage'];sp=stage['source_path']
    stage_manifest={'files':[{'path':sp,'SHA256':manifest[sp],'ptx_source_path':stage['ptx_source_path']}]}
    ag.source_stage(meta,stage_manifest,ptx)
    layout=ag.ttgir_contract(ir,case['logical_shape'],case['num_warps'])
    c.require(layout['blocked']==case['default' if k=='default' else 'cand4']['layout'],'Actual frozen register layout')
    resources=ac.parse_resource((path/'kernel.resource.txt').read_text())
    c.require(resources==meta['resources'] and resources['num_regs']==occupancy['num_regs'] and
              resources['local_bytes']+resources['stack_bytes']==occupancy['local_bytes'],'Actual resource observations')
    c.require(occupancy['resource_sha256']==actual['kernel.resource.txt'] and
              occupancy['dynamic_smem_bytes']==meta['dynamic_smem_bytes'] and occupancy['num_warps']==case['num_warps'],'Exact dynamic SMEM/warp occupancy')
    c.require(occupancy['checked_cuda_calls'] and all(x['return_code']==0 for x in occupancy['checked_cuda_calls']),'Checked driver resource calls')
    c.require(smoke['passed'] is True and smoke['max_abs_diff']==0 and smoke['timed'] is False and
              smoke['launched_cubin_sha256']==smoke['after_cubin_sha256']==digest and smoke['B_DESC']==65536 and
              smoke['B_RUN']==smoke['initialized_prefix_tiles']==4 and smoke['full_descriptor_allocation'] is True and
              smoke['input_allocation_shape']==[65536,case['M'],case['N']] and
              smoke['input_allocation_bytes']==65536*case['M']*case['N']*2 and smoke['compile_calls']==1,'Full-descriptor non-timed correctness and exact launch SHA')
    c.require(not any(key in smoke for key in ('samples_us','elapsed_time','latency','throughput')),'No smoke performance observations')
    # Entire source-bound convert/max stage, including terminal exchanges.
    # Native output has no convert_layout. Its terminal result load can be
    # delayed into the store's debug location; include the entire interval to
    # first global store, as the frozen Phase6 minimal-output auditor does.
    # Canonical output must end before its explicit layout conversion.
    native_output=h.endswith('_native')
    body=ag.body_instructions(ptx,stage,minimal_output=native_output)
    fp=ac.fingerprint(body)
    c.require(fp,'Full nonempty reduction fingerprint')
    device='ttng.tensormap_create' in ir
    if device:
        c.require('ttng.tensormap_fenceproxy_acquire' in ir and 'ttg.global_scratch_alloc' in ir,'Lowered device descriptor publication/scratch present')
    c.require(device==(h=='canonical' or h.startswith('device_')),'Descriptor intervention present in actual IR')
    if h=='canonical' or h.endswith('_canonical'):
        store_lines=[line for line in ir.splitlines() if 'tt.store' in line]
        c.require(len(store_lines)==1,'One complete FP32 output store')
        m=re.search(r'tensor<'+str(case['N'])+r'x!tt.ptr<f32>, #(\w+)>',store_lines[0])
        c.require(m is not None,'Canonical output shape/layout')
        definition=re.search(r'^#'+m[1]+r' = #ttg.blocked<\{(.*?)\}>',ir,re.M)
        c.require(definition is not None,'Canonical blocked output encoding')
        for name,want in {'sizePerThread':[1],'threadsPerWarp':[32],'warpsPerCTA':[case['num_warps']],'order':[0]}.items():
            found=re.search(name+r' = \[([^]]+)\]',definition[1])
            c.require(found is not None and list(map(int,found[1].split(',')))==want,'Actual canonical output '+name)
    else:
        c.require('ttg.convert_layout' not in ir,'Inherited output layout has no extra conversion')
    reasons=[]
    if resources['local_bytes'] or resources['stack_bytes']:reasons.append('SPILL')
    if meta['dynamic_smem_bytes']>env['device_limits']['max_shared_memory_per_block_optin'] or occupancy['blocks_per_sm_actual_dynamic_smem']<=0:reasons.append('RESOURCE_UNSUPPORTED')
    binding={'archive_path':(path/'kernel.cubin').relative_to(c.ROOT).as_posix(),'archive_sha256':digest,
             'ptx_sha256':actual['kernel.ptx'],'metadata':meta,'ABI':ct.abi(ptx),'shared_layout':layout['shared'],'occupancy':occupancy,
             'scratch_bytes_per_cta':128 if device else 0}
    return {'complete':True,'reasons':reasons,'layout':layout,'fingerprint':fp,
            'body_operand_SHA256':c.sha(c.encode(body)),
            'body_boundary':'FIRST_GLOBAL_STORE_NO_OUTPUT_CONVERSION' if native_output else 'SOURCE_BOUND_BEFORE_OUTPUT_CONVERSION',
            'original_collector_fingerprint':c.read(path/'observed_structure.json')['reduction_fingerprint'],
            'initial_load_opcodes':[i['opcode'] for i in ac.initial_load_signature(ptx)],
            'resources':resources,'occupancy':occupancy,'binding':binding,'device_descriptor':device,
            'output_conversions':len(re.findall(r'ttg\.convert_layout',ir)),
            'all_PTX_SHA256':actual['kernel.ptx'],'all_SASS_SHA256':actual['kernel.sass']}


def derive():
    c.inventory();manifest,env,dispatch=sources()
    pool=c.read(ct.DEST/'pool.json');cases=ct.population(pool)
    attempts=c.read(DEST/'attempts.json')
    expected={(r['config_id'],h,k) for r in cases for h in c.HARNESSES for k in c.CANDIDATES}
    c.require(len(attempts)==len(expected)==290 and {(r['case_id'],r['harness'],r['candidate']) for r in attempts}==expected,'All 290 frozen binary attempts')
    for attempt in attempts:c.require(attempt==c.read(DEST/attempt['harness']/attempt['case_id']/attempt['candidate']/'attempt.json'),'Complete original attempt ledger')
    rows=[]
    for case in cases:
        bundles={h+':'+k:bundle(case,h,k,manifest,env) for h in c.HARNESSES for k in c.CANDIDATES}
        reasons=sorted({reason for b in bundles.values() for reason in b['reasons']})
        tiers=[]
        if all(b['complete'] for b in bundles.values()):
            for h in c.VARIANTS:
                for k in c.CANDIDATES:
                    reference,other=bundles['canonical:'+k],bundles[h+':'+k]
                    tier=ag.equivalence(reference['fingerprint'],other['fingerprint']);tiers.append(tier)
                    if tier=='REDUCTION_FINGERPRINT_MISMATCH':reasons.append(tier)
                    if reference['layout']['shared']!=other['layout']['shared'] or reference['initial_load_opcodes']!=other['initial_load_opcodes']:reasons.append('SHARED_OR_INITIAL_LOAD_DRIFT')
            for h in c.HARNESSES:
                a,b=[bundles[h+':'+k]['occupancy']['blocks_per_sm_actual_dynamic_smem'] for k in c.CANDIDATES]
                if a!=b:reasons.append('CANDIDATE_RESIDENCY_MISMATCH')
        classification='EXCLUDE_FROM_TIMING' if reasons else 'PRIMARY' if tiers and all(t=='EXACT_SEQUENCE_EQUIVALENT' for t in tiers) else 'SECONDARY'
        row={**case,'case_id':case['config_id'],'final_class':classification,'pre_timing_eligible':not reasons,
             'reason_codes':sorted(set(reasons)),'bundles':bundles,'reduction_equivalence_tiers':tiers}
        if row['pre_timing_eligible']:
            row['cross_cell_residency_matched']=len({b['occupancy']['blocks_per_sm_actual_dynamic_smem'] for b in bundles.values()})==1
            row['compiler_response_registers']={key:b['resources']['num_regs'] for key,b in bundles.items()}
        rows.append(row)
    launches={}
    for stage,origin in (('stage_c','OUTCOME_INFORMED_PRIMARY_DIAGNOSTIC'),('stage_d','UNMEASURED_LARGE_M_WARP_EXTENSION')):
        included={row['case_id']:row for row in rows if row['origin']==origin and row['pre_timing_eligible']}
        case_map={cfg:{k:r[k] for k in ('case_id','M','N','num_warps','origin','final_class')} for cfg,r in included.items()}
        binaries={cfg+':'+key:b['binding'] for cfg,row in included.items() for key,b in row['bundles'].items()}
        counts=dict(Counter(row['final_class'] for row in rows if row['origin']==origin))
        launches[stage]={'cases':case_map,'binaries':binaries,'schedule':ct.schedule(case_map) if case_map else None,'counts':counts,
                        'ALL_ELIGIBLE_coverage_before_timing':'TESTABLE' if len(included)>=5 else 'INCONCLUSIVE_BY_COVERAGE',
                        'PRIMARY_coverage_before_timing':'TESTABLE' if counts.get('PRIMARY',0)>=5 else 'INCONCLUSIVE_BY_COVERAGE',
                        'protocol_SHA256':c.sha((ct.DEST/'protocol.json').read_bytes())}
    return json.loads(c.encode({'cases':rows,'counts':dict(Counter(r['final_class'] for r in rows)),'attempts':290,'performance_observations':0,
                               'native_core_image_reused':dispatch['core_image_id'],'no_native_rebuild':True})),json.loads(c.encode(launches))


def report(result,launches):
    rows=[[r['case_id'],r['origin'],r['final_class'],', '.join(r['reason_codes']) or 'ALL_GATES_PASS'] for r in result['cases']]
    return '\n'.join(['# Phase7 StageB — Exact-artifact admission','',c.table(['Case','Role','Class','Reasons'],rows),'',
        'All 290 attempts/partial exports/source/ABIs/correctness/resources retained. No timing. Exact Phase6 core image reused: '+result['native_core_image_reused']+'.','',
        c.table(['Future stage','Eligible','PRIMARY','PRIMARY coverage'],[[stage,len(x['cases']),x['counts'].get('PRIMARY',0),x['PRIMARY_coverage_before_timing']] for stage,x in launches.items()]),'',
        'PRIMARY refers only to complete source-bound reduction opcode sequence. All operand-sensitive body and full-PTX/SASS SHA observations retained; complete machine dataflow/scheduling equivalence is not inferred. Cross-cell register/residency changes are compiler responses, not hidden or artificially normalized.',''])


def main():
    result,launches=derive()
    if '--validate' in sys.argv:
        c.require(c.read(DEST/'gate_results.json')==result,'Independent complete artifact admission')
        for stage,launch in launches.items():c.require(c.read(DEST/('launch_'+stage+'.json'))==launch,'Frozen launch closure '+stage)
        c.require(c.read(DEST/'raw_manifest.json')['files']==raw_inventory(),'All original artifact bytes unchanged')
        c.require((DEST/'summary.md').read_text()==report(result,launches),'Artifact summary closure')
        probes=[]
        for name,mutate in (('class',lambda x:x['cases'][0].update(final_class='CORRUPT')),('count',lambda x:x.update(attempts=0))):
            bad=copy.deepcopy(result);mutate(bad);c.require(bad!=result,'Artifact corruption rejected');probes.append(name)
        valid={'status':'PASS','counts':result['counts'],'attempts':290,'corruption_probes':probes,'no_native_rebuild':True}
        c.write(DEST/'validation.json',valid);print(valid)
    else:
        c.write(DEST/'gate_results.json',result)
        for stage,launch in launches.items():c.write(DEST/('launch_'+stage+'.json'),launch)
        c.write(DEST/'raw_manifest.json',{'files':raw_inventory(),'before_any_new_timing':True})
        (DEST/'summary.md').write_text(report(result,launches));print(result['counts'],{k:len(v['cases']) for k,v in launches.items()})


if __name__=='__main__':main()
