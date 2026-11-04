"""Independent CPU-only N2/N3 audit, with optional same-case confirmation.

This module reads frozen artifacts and recomputes derived summaries. It never
imports Triton, compiles a kernel, starts a profiler, or launches GPU work.
N1's accepted validators and reports are left unchanged.
"""
import argparse
import ast
from collections import Counter
import copy
from datetime import datetime
import json
from pathlib import Path
import re
import sys

from . import common as c
from .elf import elf
from . import validate as n1


def require(condition,message):
    if not condition:
        raise AssertionError(message)


def protocol(path):
    path=n1._path(path);value=c.read(path)
    require(value['compiler']==n1.COMPILER and value['native_SHA256']==c.NATIVE,'Compiler/native pin changed')
    expected_backend=c.read(c.OUT/'stage_n1/revision_008/preregistration/protocol.json')['backend_SHA256']
    require(value['backend_SHA256']==expected_backend,'Backend pin changed')
    sources=n1._zip_members(path.parent/'source_snapshot.zip')
    require(set(sources)==set(value['source_SHA256']),'Frozen source ZIP member set changed')
    for name,digest in value['source_SHA256'].items():
        require(c.sha(sources[name])==digest,'Frozen source changed: '+name)
    for name,digest in value.get('protected_SHA256',{}).items():
        require(c.sha((c.ROOT/name).read_bytes())==digest,'Protected historical artifact changed: '+name)
    if 'payload_path' in value:
        require(c.sha(n1._path(value['payload_path']).read_bytes())==value['payload_SHA256'],'Frozen payload changed')
    return value,sources


def attempt(folder,p):
    folder=Path(folder);dispatch=c.read(folder/'dispatch.json')
    require(dispatch['protocol_SHA256']==c.sha(c.encode(p)),'Dispatch protocol changed: '+str(folder))
    members=n1._zip_members(folder/'raw_export.zip')
    for name,blob in members.items():
        require((folder/'export'/name).read_bytes()==blob,'Extracted raw evidence changed: '+name)
    if (folder/'raw.json').exists():
        require((folder/'raw.json').read_bytes()==members['result.json'],'Raw result copy changed')
    result=json.loads(members['result.json'])
    require(result.get('protocol_SHA256')==dispatch['protocol_SHA256'],'Worker protocol mismatch')
    return result,members


def resource(blob):
    values={key:int(value) for key,value in n1._resources(blob).items()}
    require(values.get('REG',0)>0,'Missing physical register allocation')
    return values


def artifact(meta,members,kind='candidate',upstream=None):
    prefix=meta['artifact_directory'].rstrip('/')+'/'
    for name,digest in meta['artifact_SHA256'].items():
        require(c.sha(members[prefix+name])==digest,'Artifact SHA changed: '+meta['tag']+'/'+name)
    require(c.sha(members[prefix+'kernel.cubin'])==meta['cubin_SHA256'],'CUBIN SHA changed')
    disk=json.loads(members[prefix+'metadata.json'])
    for key,value in disk.items():
        if kind=='imported' and key in {'tag','artifact_directory'}:
            continue
        if key=='specialized_runtime_values' and kind=='candidate':
            require(value=={} or value==meta[key],'Unexpected exporter specialization contract')
            require(upstream is not None and meta[key]==upstream[key],
                    'Candidate runtime specialization copy differs from U')
            continue
        require(meta.get(key)==value,'Raw metadata/index disagreement: '+meta['tag']+'/'+key)
    values=resource(members[prefix+'kernel.resource.txt'])
    require(meta.get('resources',values)==values,'Resource index differs from cuobjdump')
    require(meta['global_scratch_size']==0,'Unexpected global scratch')
    forbidden=re.compile(r'\b(?:tt\.(?:reduce|scan|dot|atomic_\w+|descriptor_(?:load|store|gather|scatter))|ttng\.\w*(?:tma|mma)\w*|ttg\.warp_group_dot)\b')
    irs=[]
    for extension in ('ttir','glir','ttgir'):
        name=prefix+'kernel.'+extension
        if name not in members:
            continue
        text=members[name].decode();irs.append(text)
        require(not forbidden.search(text),'Reduction/TMA/MMA operation in '+meta['tag'])
        for line in text.splitlines():
            if 'inline_asm' in line:
                require(not re.search(r'%(?:laneid|warpid|tid\.[xyz]|ntid\.[xyz])\b',line),
                        'Source inline assembly introduces hardware-ID scalars')
    require(irs and prefix+'kernel.ttgir' in members,'Missing canonical target IR')
    return values


def host_menu(source_blob,case,plan,module_name):
    """Execute only plain host function definitions from the frozen source.

    No module-level statements or decorated JIT functions are executed. The
    packing host helpers import only the existing CPU layout parser.
    """
    tree=ast.parse(source_blob)
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and not node.decorator_list]
    module=ast.Module(body=functions,type_ignores=[])
    namespace=dict(copy=copy,re=re,__package__=__package__,__name__=module_name)
    exec(compile(module,module_name,'exec'),namespace)
    return namespace['variants'](copy.deepcopy(case),copy.deepcopy(plan))


def validate_compile(root=c.OUT):
    root=Path(root);stage=root/'stage_n2_compile_r2'
    p,sources=protocol(stage/'preregistration/protocol.json')
    folder=stage/'compile/attempt_001/invocation_-01'
    result,members=attempt(folder,p)
    require(result['status']=='OK' and result['GPU_allocated'] is False and
            result['GPU_kernel_launches']==0 and result['native_rebuilds']==0,'Candidate compile was not CPU-only')
    require(result['native_SHA256']==c.NATIVE and result['backend_SHA256']==p['backend_SHA256'],
            'Candidate worker compiler pin mismatch')
    require(result['compilation_target']==dict(backend='cuda',capability=90,warp_size=32),'Wrong compilation target')
    require(result['payload_SHA256']==p['payload_SHA256'],'Compile baseline payload changed')
    baseline_path=n1._path(p['payload_path']);baseline_members=n1._zip_members(baseline_path)
    baseline=json.loads(baseline_members['result.json'])
    require(p['payload_SHA256']==c.sha((root/'stage_n1/baseline_payload.zip').read_bytes()),
            'Candidate compilation does not inherit accepted N1 payload')
    baseline_programs={meta['tag']:meta for meta in baseline['programs']}
    programs={meta['tag']:meta for meta in result['programs']}
    records={row['metadata']['tag']:row for row in result['records'] if row.get('metadata')}
    require(len(programs)==len(result['programs'])==len(records)==138,'Expected 138 unique logical programs')
    require(all(row['status']=='OK' for row in result['records']),'Unaccepted failed candidate')
    require(result['counts']==dict(cases=14,logical_programs=138,variants=110,failed_records=0),
            'Candidate coverage counts changed')
    require(len(p['cases'])==14 and len({x['family'] for x in p['cases']})==6,'Candidate geometry scope changed')
    menus={};counts=Counter()
    base=str(c.SOURCE.relative_to(c.ROOT))+'/'
    files={'N01':'quant_adapter.py','N02':'quant_adapter.py','N03':'rotary_adapter.py',
           'N04':'pack_gate_gluon.py','N05':'pack_gate_gluon.py','N06':'cache_adapter.py'}
    for case in p['cases']:
        case_id=case['id'];up=baseline_programs[case_id+':U']
        menu=host_menu(sources[base+files[case['family']]].decode(),case,result['plans'][case_id],files[case['family']])
        require(len({row['name'] for row in menu})==len(menu),'Duplicate declared candidate names')
        menus[case_id]={row['name']:row for row in menu}
        expected={case_id+':U',case_id+':G0'}|{case_id+':GL_'+name for name in menus[case_id]}
        require({tag for tag,meta in programs.items() if meta['geometry']==case_id}==expected,
                'Candidate menu is incomplete or changed: '+case_id)
        for tag in expected:
            meta=programs[tag];row=records[tag]
            require(row['metadata']==meta and row['case']==case_id and row['variant']==meta['variant'],
                    'Compile record/index mismatch')
            require(meta['case']==case and meta['num_warps']==case['W'],'Candidate changed fixed case/warps')
            if meta['variant'] in {'U','G0'}:
                original=baseline_programs[tag]
                expected_meta={**original,'case':case,'variant':meta['variant'],
                    'baseline_artifact_directory':original['artifact_directory'],
                    'artifact_directory':'baseline/'+original['artifact_directory'],
                    'imported_baseline':True,'resources':meta['resources']}
                require(meta==expected_meta,'Imported baseline rebasing changed executable metadata')
                for filename in original['artifact_SHA256']:
                    require(members[meta['artifact_directory']+'/'+filename]==
                            baseline_members[original['artifact_directory']+'/'+filename],
                            'Imported baseline artifact bytes changed')
                artifact(meta,members,'imported')
            else:
                name=meta['variant'].removeprefix('GL_')
                require(meta['choice']==row['choice']==menus[case_id][name],'Candidate intervention differs from declared menu')
                require(meta['imported_baseline'] is False,'Candidate marked imported')
                for key in ('runtime_argument_names','specialized_runtime_values','num_warps'):
                    require(meta[key]==up[key],'Candidate changed U runtime contract: '+key)
                for key,value in up['JIT_signature'].items():
                    require(meta['JIT_signature'][key]==value,'Candidate changed original JIT signature')
                extras=set(meta['JIT_signature'])-set(up['JIT_signature'])
                require(extras and all(meta['JIT_signature'][key]=='constexpr' for key in extras),
                        'Appended candidate parameters are not layout constexprs')
                for key,value in up['JIT_constants'].items():
                    require(meta['JIT_constants'][key]==value,'Candidate changed non-layout constexpr')
                for key,value in up['JIT_attributes'].items():
                    require(meta['JIT_attributes'][key]==value,'Candidate changed actual U JIT attributes')
                artifact(meta,members,'candidate',up)
                audit=row['effective_layout_audit'];text=members[meta['artifact_directory']+'/kernel.ttgir'].decode()
                require(audit['status']=='OK' and audit['ttgir_SHA256']==c.sha(text.encode()),
                        'Effective layout audit is not bound to target IR')
                counts[case_id]+=1
    require(sum(counts.values())==110,'Expected all 110 logical candidates, including aliases')
    for name in set(files.values())|{'quant_gluon.py','rotary_gluon.py','cache_gluon.py'}:
        n1._no_reduction_source(sources[base+name].decode())
    # A later N01 layout-expression refactor has its own CPU bridge to the
    # original U. The accepted N1 binaries themselves are never rewritten.
    bridges=[n1._accepted(dict(case=case_id,g0_revision=103,g0_attempt=1),root/'stage_n1')
             for case_id in ('N01_m4096_n128','N01_m11008_n128')]
    return dict(status='OK',logical_programs=138,candidates=110,baselines=28,
        candidate_counts=dict(counts),native_rebuilds=0,GPU_allocated=False,
        compile_protocol_SHA256=c.sha(c.encode(p)),compile_result_SHA256=c.sha(members['result.json']),
        source_bridge=bridges),result,members


def _baseline_hash_pairs(p,raw):
    by={}
    for check in raw['checks']:
        hashes={row['name']:row['actual_SHA256'] for row in check['details'] if 'name' in row}
        for tag in [check['tag']]+check.get('aliases',[]):
            by[(tag,check['pattern'])]=hashes
    patterns=['finite'] if int(raw['invocation']) else p['correctness']['patterns']
    for case in p['cases']:
        for pattern in patterns:
            require(by[(case['id']+':U',pattern)]==by[(case['id']+':G0',pattern)],
                    'Same-input U/G0 outputs differ bitwise: '+case['id'])


def validate_measurement(stage,expected_programs,root=c.OUT):
    root=Path(root);target=root/stage;p,_=protocol(target/'preregistration/protocol.json')
    require(n1._programs(p)==expected_programs,'Formal protocol changed compiled program index')
    require(p['timing'].get('sample_order')=='interleaved_rounds','Expected frozen interleaved rounds')
    require((p['timing']['workers'],p['timing']['visits'],p['timing']['samples'])==(3,10,31),
            'Formal worker/visit/sample counts changed')
    require(p['timing']['bootstrap_replicates']==4096 and p['timing']['practical_threshold_percent']==3,
            'Frozen statistical estimator changed')
    correctness,_=attempt(target/'check/attempt_001/invocation_000',p)
    rows=[n1.validate_runtime(p,correctness)];_baseline_hash_pairs(p,correctness)
    timing=[]
    for invocation in range(1,4):
        raw,_=attempt(target/'timing/attempt_001'/f'invocation_{invocation:03}',p)
        require(raw['invocation']==invocation,'Timing worker invocation changed')
        rows.append(n1.validate_runtime(p,raw));_baseline_hash_pairs(p,raw);timing.append(raw)
        for tag,binding in raw['bindings'].items():
            meta=expected_programs[tag]
            require(binding['attributes']['n_regs']==meta['resources']['REG'],'Runtime physical register mismatch')
    analysis=c.read(target/'analysis/analysis.json')
    summary=n1.validate_timing_analysis(p,timing,analysis)
    from .analyze import csv_bytes
    require((target/'analysis/timing.csv').read_bytes()==csv_bytes(analysis),'Derived timing CSV changed')
    return dict(status='OK',runtime=rows,analysis=summary,
                baseline_equivalence_is_observed_not_forced=True,
                analysis_SHA256=c.sha((target/'analysis/analysis.json').read_bytes())),p,analysis


def _profile_binding(binding,selection,mode):
    meta=selection['metadata']
    require(binding['mode']==mode and binding['warmups']==8,'Profile mode/warmup mismatch')
    require(binding['compile_calls']==0 and binding['Triton_imported'] is False and
            binding['profiler_attached'] is True and binding['formal_timing'] is False,
            'Profiler execution imported/compiler/formal-timing contract changed')
    require(binding['driver_return_codes']==[0] and len(binding['records'])==1,'Profile contains CUDA errors/multiple kernels')
    env=binding['environment']
    require(env['cc']==[9,0] and 'H100' in env['name'] and env['gpu_uuid'].startswith('GPU-'),
            'Missing actual H100 profile identity')
    actual=binding['records'][0]
    n1._check_correctness(actual['correctness'])
    require(actual['tag']==meta['tag'],'Profile bound another program')
    require(actual['binding']['cubin_SHA256']==actual['binding']['loaded_SHA256']==meta['cubin_SHA256'],
            'NCU loaded CUBIN mismatch')
    require(actual['runtime_arguments_unchanged_from_full_fixture'] and
            actual['inplace_restored_before_each_launch'],'Profile changed runtime inputs/in-place state')
    require(actual['binding']['attributes']['n_regs']==meta['resources']['REG'],'Profile register resource mismatch')
    if mode=='software':
        require(actual['grid']==[1,1,1] and actual['software_active_masks'],'Software profile is not unchanged-argument CTA0')
        for name,mask in actual['software_active_masks'].items():
            require(mask['shape']==actual['fixture']['views'][name]['shape'] and mask['elements']>0,
                    'Software fixture resized its original runtime tensor')
    else:
        require(actual['grid']==actual['original_grid'],'Hardware profile changed full grid')
    return actual


def validate_profiles(compiled,n2_protocol,root=c.OUT,attempts=None):
    from . import profile_analyze as pa
    from . import profile_prepare as pp
    root=Path(root);stage=root/'stage_n3';p,sources=protocol(stage/'preregistration/protocol.json')
    require(p['formal_timing'] is False and p['selection_uses_candidate_timing'] is False,
            'NCU role/selection contract changed')
    require(p['maximum_concurrent_GPU_groups']==3,'Profiler concurrency bound changed')
    require(len(p['profile_groups'])==7,'Expected seven non-reduction subtypes')
    compile_protocol=c.read(root/'stage_n2_compile_r2/preregistration/protocol.json')
    require(datetime.fromisoformat(p['frozen_at_UTC'])>=datetime.fromisoformat(compile_protocol['frozen_at_UTC']),
            'Profile protocol predates its compiled payload')
    # Selection was frozen before any candidate timing dispatch. Checking
    # protocol hashes alone cannot prove wall-clock order; use the preserved
    # local dispatch file timestamps as a second, explicitly local audit.
    timing_dispatches=list((root/'stage_n2/timing').glob('attempt_*/invocation_*/dispatch.json'))
    freeze_time=datetime.fromisoformat(p['frozen_at_UTC']).timestamp()
    require(not timing_dispatches or all(path.stat().st_mtime>=freeze_time for path in timing_dispatches),
            'N3 selection was not frozen before local candidate timing dispatch')
    payload=stage/'compiled_payload.zip'
    require(c.sha(payload.read_bytes())==p['payload_SHA256']==n2_protocol['payload_SHA256'],
            'N3 does not use the exact complete CPU-compiled candidate payload')
    selector_source=str(Path(pp.__file__).relative_to(c.ROOT))
    require(c.sha(Path(pp.__file__).read_bytes())==p['source_SHA256'][selector_source],
            'Static selector replay source changed since freezing')
    groups,aliases,declaration=pp.select(payload,include_cold_pack=p['selectors']['optional_cold_pack'])
    require(groups==p['profile_groups'] and aliases==p['alias_proofs'] and declaration==p['selectors'],
            'NCU static choices differ from declared mechanism contrasts')
    require(c.read(stage/'preregistration/selectors.json')==declaration,'Frozen static selector manifest changed')
    expected=sum(len(group['records'])*len(group['modes']) for group in groups)
    require(expected==60,'Expected 60 physical NCU reports')
    metric_provenance=p['metric_provenance']
    require(c.sha(n1._path(metric_provenance['path']).read_bytes())==metric_provenance['SHA256'],
            'NCU metric provenance changed')
    require(len(p['metrics'])==len(set(p['metrics']))==26,'NCU required metric set changed')
    by={meta['tag']:meta for meta in compiled['programs']}
    attempts=attempts or {};report_count=0;replay_count=0
    settings={'software':('kernel','all'),'warm_application':('application','none'),'cold_isolated':('kernel','all')}
    for group in groups:
        require(group['logical_variants'][:2]==['U','G0'] and len(group['logical_variants'])==4,
                'Static profile group lacks independent U/G0 + two GL contrasts')
        number=int(attempts.get(group['id'],1));folder=stage/'profiles'/group['id']/f'attempt_{number:03}'
        raw,members=attempt(folder,p)
        require(raw['status']=='OK' and raw['formal_timing'] is False and raw['native_rebuilds']==0,
                'NCU group failed or rebuilt compiler')
        require(raw['group']==group['id'] and raw['geometry']==group['case']['id'],'Profile group lineage changed')
        selected={record['metadata']['tag']:record for record in group['records']}
        require(all(record['metadata']==by[tag] for tag,record in selected.items()),'Profile selected another compiled binary')
        target={(tag,mode) for tag in selected for mode in group['modes']}
        observed=[(report['tag'],report['mode']) for report in raw['reports']]
        require(len(observed)==len(set(observed))==len(target) and set(observed)==target,
                'NCU report target coverage changed')
        for report in raw['reports']:
            record=selected[report['tag']];mode=report['mode'];prefix=str(Path(report['path']).parent)+'/'
            require(c.sha(members[report['path']])==report['SHA256'],'Raw NCU report SHA changed')
            require(c.sha(members[report['binding_path']])==report['binding_SHA256'],'Profile binding SHA changed')
            job=json.loads(members[prefix+'job.json'])
            require(job==dict(case=group['case'],mode=mode,metadata=[record['metadata']],warmups=8),
                    'Profile job changed selected full arguments')
            collect=json.loads(members[prefix+'collect.json'])
            require(collect['returncode']==0 and not collect['timed_out'],'NCU subprocess failed')
            argv=collect['argv'];replay,cache=settings[mode]
            declared_settings=json.loads(members[prefix+'settings.json'])
            require(declared_settings['mode']==mode and declared_settings['replay']==replay and
                    declared_settings['cache']==cache and declared_settings['warmups']==8 and
                    declared_settings['one_binary_per_report'] is True,'Profile settings manifest changed')
            for flag,value in [('--replay-mode',replay),('--cache-control',cache),
                               ('--clock-control','none'),('--profile-from-start','off')]:
                require(argv[argv.index(flag)+1]==value,'NCU cache/replay/profile setting changed')
            if mode=='software':
                require(argv[argv.index('--section')+1]=='InstructionStats','Software mode did not collect InstructionStats')
            else:
                require(argv[argv.index('--metrics')+1].split(',')==p['metrics'],'Hardware metric command changed')
            bindings=json.loads(members[prefix+'all_replay_bindings.json'])
            emitted=[json.loads(value) for value in re.findall(r'PROFILE_BINDING (\{[^\n]+\})',collect['output'])]
            require(bindings==emitted and len(bindings)==report['emitted_replay_bindings'] and bindings,
                    'Not all application-replay executable bindings were retained')
            require(json.loads(members[prefix+'binding.json'])==bindings[-1],'Last replay binding changed')
            for binding in bindings:
                _profile_binding(binding,record,mode)
            for filename in ['import.json']+(['source.json'] if mode=='software' else []):
                command=json.loads(members[prefix+filename])
                require(command['returncode']==0 and not command['timed_out'],'NCU raw/source import failed')
            replay_count+=len(bindings);report_count+=1
    require(report_count==60,'Incomplete NCU physical reports')
    expected_analysis=pa.analyze(p,stage/'profiles',payload,attempts)
    summary=c.read(stage/'analysis/summary.json')
    for key,value in expected_analysis.items():
        require(summary.get(key)==value,'NCU derived summary changed: '+key)
    require(c.sha(n1._path(summary['analysis_source']['path']).read_bytes())==summary['analysis_source']['SHA256'],
            'NCU analysis source changed')
    require((stage/'analysis/summary.csv').read_bytes()==pa.csv_bytes(summary,p['metrics']),'NCU summary CSV changed')
    require((stage/'analysis/metrics.csv').read_bytes()==pa.metric_csv_bytes(summary),'NCU metric CSV changed')
    require((stage/'analysis/REPORT.md').read_bytes()==pa.render(summary,p).encode(),'NCU report changed')
    return dict(status='OK',physical_reports=report_count,emitted_replay_bindings=replay_count,
                logical_reports=summary['logical_report_count'],
                baseline_software_crosschecks=summary['baseline_software_crosschecks'],
                source_selection_uses_timing=False,compile_calls=0)


def validate_confirmation(stage,n2_protocol,n2_analysis,root=c.OUT):
    root=Path(root);target=root/stage;p,_=protocol(target/'preregistration/protocol.json')
    decision=c.read(target/'selection/decision.json')
    require(all(row['classification']=='practically_equivalent' for row in n2_analysis['baseline_equivalence']),
            'Confirmation started before N2 baseline controls resolved')
    require(decision['inputs']['analysis_SHA256']==c.sha((root/'stage_n2/analysis/analysis.json').read_bytes()),
            'Confirmation selected from another exploratory analysis')
    require(decision['inputs']['protocol_SHA256']==c.sha((root/'stage_n2/preregistration/protocol.json').read_bytes()),
            'Confirmation exploratory protocol changed')
    by=n1._programs(n2_protocol);selected={};expected_choices=[]
    for row in n2_analysis['cases']:
        case=row['case'];candidates=[item for item in n2_analysis['rows']
            if item['case']==case and item['variant'].startswith('GL_')]
        best=min(candidates,key=lambda item:(item['time_us'],item['tag']))
        require(row['best_observed_tag']==best['tag'],'Exploratory best-observed summary inconsistent')
        for tag in [case+':U',case+':G0',best['tag']]:
            selected[tag]=by[tag]
        expected_choices.append(dict(case=case,candidate=best['tag'],
                                     exploratory_gain_vs_U=row['best_observed_gain_vs_U_percent']))
    require(decision['choices']==expected_choices,'Confirmation dropped null cases or changed frozen choice')
    require(len(selected)==42 and n1._programs(p)==selected,'Confirmation changed selected CUBIN/ABI index')
    payload=n1._zip_members(n1._path(p['payload_path']))
    require(json.loads(payload['result.json'])['selection']==decision,'Confirmation payload/decision mismatch')
    parent=n1._zip_members(n1._path(n2_protocol['payload_path']))
    for meta in selected.values():
        for filename in meta['artifact_SHA256']:
            name=meta['artifact_directory']+'/'+filename
            require(payload[name]==parent[name],'Confirmation rebuilt or altered selected binary')
    original_calls={c.read(path)['call_id'] for path in (root/'stage_n2/timing').glob('attempt_*/invocation_*/dispatch.json')}
    confirmation_calls={c.read(path)['call_id'] for path in (target/'timing').glob('attempt_*/invocation_*/dispatch.json')}
    require(len(confirmation_calls)==3 and not original_calls&confirmation_calls,
            'Confirmation did not use three new timing invocations')
    measured,_,_=validate_measurement(stage,selected,root)
    return dict(status='OK',logical_programs=42,all_cases=14,measurement=measured,
                scope='Independent same-case fixed-choice confirmation; not unseen geometry/model generalization.')


def run(root=c.OUT,n2_only=False,confirmation=None,profile_attempts=None):
    audit=n1.Audit();root=Path(root)
    compiled=audit.check('N2_CPU_compile',lambda:validate_compile(root))
    results={}
    if compiled:
        results['compile']=compiled[0]
        measured=audit.check('N2_frozen_runtime_and_analysis',lambda:
            validate_measurement('stage_n2',{meta['tag']:meta for meta in compiled[1]['programs']},root))
        if measured:
            results['N2']=measured[0]
            if not n2_only:
                results['N3']=audit.check('N3_profiles',lambda:
                    validate_profiles(compiled[1],measured[1],root,profile_attempts))
            if confirmation:
                results['confirmation']=audit.check('independent_confirmation',lambda:
                    validate_confirmation(confirmation,measured[1],measured[2],root))
    require(not any(name=='triton' or name.startswith('triton.') for name in sys.modules),
            'Validator imported Triton')
    return dict(status='FAILED' if audit.errors else 'OK',CPU_only=True,GPU_allocations=0,
        compiler_invocations=0,checks_passed=len(audit.checks),errors=audit.errors,
        compiler=n1.COMPILER,native_SHA256=c.NATIVE,results=results,
        validator_source_SHA256=c.sha(Path(__file__).read_bytes()),
        scope='Artifact, lineage, coverage, execution and estimator validation; favorable scientific outcomes are not required.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=c.OUT)
    parser.add_argument('--n2',action='store_true',help='Validate N2 without requiring N3 completion')
    parser.add_argument('--confirmation',help='Also validate a frozen same-case confirmation stage, e.g. stage_n4')
    parser.add_argument('--profile-attempt-map',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();attempts=c.read(args.profile_attempt_map) if args.profile_attempt_map else None
    report=run(args.root,args.n2,args.confirmation,attempts)
    output=args.output or args.root/('stage_n2' if args.n2 else 'stage_n3')/'validation_final.json'
    c.write_once(output,report)
    print(json.dumps(dict(status=report['status'],checks_passed=report['checks_passed'],errors=report['errors']),indent=2))
    if report['errors']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
