"""Final read-only CPU aggregation for N1 through the bounded N5 follow-ups.

Scientific outcomes are retained: this audit checks provenance, execution,
coverage and numerical reproducibility, not whether a candidate is faster.
"""
import argparse
import ast
import copy
import importlib
import json
from pathlib import Path
import re
import sys

from . import common as c
from . import validate as n1
from . import final_validate as suite
from .elf import elf


require=suite.require


def profile_paths(value):
    value=copy.deepcopy(value)
    for row in value.get('rows',[]):
        for key in ('report_path','import_path','binding_path','source_path'):
            if key in row:
                row[key]=str(n1._path(row[key]).resolve(strict=True))
    for key in ('reports','NCU_versions'):
        for row in value.get(key,[]):
            if 'path' in row:
                row['path']=str(n1._path(row['path']).resolve(strict=True))
    return value


def profile_raw(stage,expected_count):
    """Audit every emitted replay, not merely the final binding per report."""
    root=c.OUT/stage;p,_=suite.protocol(root/'preregistration/protocol.json')
    reports=replays=0
    for group in p['profile_groups']:
        result,members=suite.attempt(root/'profiles'/group['id']/'attempt_001',p)
        require(result['status']=='OK' and result['native_rebuilds']==0 and result['formal_timing'] is False,
                'Supplement profile worker failed/rebuilt native')
        selected={record['metadata']['tag']:record for record in group['records']}
        expected={(tag,mode) for tag in selected for mode in group['modes']}
        observed=[(row['tag'],row['mode']) for row in result['reports']]
        require(len(observed)==len(set(observed))==len(expected) and set(observed)==expected,
                'Supplement profile coverage mismatch')
        for report in result['reports']:
            mode=report['mode'];selection=selected[report['tag']]
            prefix=str(Path(report['path']).parent)+'/'
            require(c.sha(members[report['path']])==report['SHA256'],'Supplement NCU report changed')
            require(c.sha(members[report['binding_path']])==report['binding_SHA256'],'Supplement binding changed')
            command=json.loads(members[prefix+'collect.json'])
            require(command['returncode']==0 and not command['timed_out'],'Supplement NCU collection failed')
            argv=command['argv'];settings=p['ncu_settings'][mode]
            for flag,value in [('--replay-mode',settings['replay']),('--cache-control',settings['cache']),
                               ('--profile-from-start','off'),('--clock-control','none')]:
                require(argv[argv.index(flag)+1]==value,'Supplement NCU setting changed')
            require(mode in {'software','warm_application'},'Unexpected supplemental profile mode')
            if mode=='software':
                require(argv[argv.index('--section')+1]=='InstructionStats','Missing supplemental software section')
            else:
                require(argv[argv.index('--metrics')+1].split(',')==p['metrics'],'Supplement hardware metrics changed')
            bindings=json.loads(members[prefix+'all_replay_bindings.json'])
            emitted=[json.loads(value) for value in re.findall(r'PROFILE_BINDING (\{[^\n]+\})',command['output'])]
            require(bindings==emitted and bindings and len(bindings)==report['emitted_replay_bindings'],
                    'Supplement omitted replay bindings')
            require(json.loads(members[prefix+'binding.json'])==bindings[-1],'Supplement last binding changed')
            for binding in bindings:
                suite._profile_binding(binding,selection,mode)
            job=json.loads(members[prefix+'job.json'])
            require(job==dict(case=group['case'],mode=mode,metadata=[selection['metadata']],warmups=8),
                    'Supplement profile job changed original runtime arguments')
            for filename in ['import.json']+(['source.json'] if mode=='software' else []):
                imported=json.loads(members[prefix+filename])
                require(imported['returncode']==0 and not imported['timed_out'],'Supplement raw/source export failed')
            reports+=1;replays+=len(bindings)
    require(reports==expected_count,'Supplement physical report count mismatch')
    return p,dict(status='OK',physical_reports=reports,replay_bindings=replays)


def validate_n3b():
    from . import profile_analyze as original
    from . import profile_supplement_analyze as wrapper
    from . import profile_supplement_prepare as prepare
    stage=c.OUT/'stage_n3b';p,raw=profile_raw('stage_n3b',16)
    require(p['evidence_role']=='post_timing_mechanism_diagnostic' and p['confirmatory'] is False
            and p['selection_uses_candidate_timing'] is True and p['formal_timing_or_selector_changes'] is False,
            'Post-timing N3b role was upgraded to confirmatory/static')
    payload=stage/'compiled_payload.zip'
    require(c.sha(payload.read_bytes())==p['payload_SHA256'],'N3b payload changed')
    groups,proofs=prepare.select(payload)
    require(groups==p['profile_groups'] and proofs==p['alias_proofs'],'N3b prespecified contrast changed')
    for record in p['selection_evidence'].values():
        require(c.sha(n1._path(record['path']).read_bytes())==record['SHA256'],'N3b triggering timing evidence changed')
        source=c.read(n1._path(record['path']))
        require(source['protocol_SHA256']==record['protocol_SHA256'] and source['payload_SHA256']==record['payload_SHA256'],
                'N3b trigger protocol/payload mismatch')
        for row in record['rows']:
            actual=next(x for x in source['rows'] if x['case']==row['case'] and x['variant']==row['variant'])
            require(all(actual[key]==value for key,value in row.items()),'N3b altered exact timing intervals')
    for boundary in p['N4_outcome_boundaries']:
        row=next(x for x in p['selection_evidence']['N4_independent_fixed_choice']['rows'] if x['case']==boundary['case'])
        passed=row['gain_vs_U_CI95'][0]>3 and row['gain_vs_G0_CI95'][0]>3
        require(boundary['strict_improvement_against_both_controls'] is passed
                and boundary['exact_CI95_U']==row['gain_vs_U_CI95']
                and boundary['exact_CI95_G0']==row['gain_vs_G0_CI95'],'N3b rounded/upgraded N4 result')
    expected=original.analyze(p,stage/'profiles',payload)
    summary=c.read(stage/'analysis/summary.json')
    canonical=profile_paths(summary)
    for key,value in profile_paths(expected).items():
        if key!='scope':
            require(canonical.get(key)==value,'N3b numerical analysis changed: '+key)
    for key,value in dict(evidence_role=p['evidence_role'],confirmatory=False,
        selection_uses_candidate_timing=True,formal_timing_or_selector_changes=False,
        selection_evidence=p['selection_evidence'],N4_outcome_boundaries=p['N4_outcome_boundaries']).items():
        require(summary[key]==value,'N3b wrapper boundary changed: '+key)
    run_node=next(node for node in ast.parse(Path(wrapper.__file__).read_text()).body
                  if isinstance(node,ast.FunctionDef) and node.name=='run')
    scope=next(keyword.value.value for node in ast.walk(run_node) if isinstance(node,ast.Call)
               for keyword in node.keywords if keyword.arg=='scope')
    require(summary['scope']==scope,'N3b wrapper scope changed')
    for source in summary['analysis_sources']:
        require(c.sha(n1._path(source['path']).read_bytes())==source['SHA256'],'N3b numerical/wrapper source changed')
    require((stage/'analysis/summary.csv').read_bytes()==original.csv_bytes(summary,p['metrics']),'N3b summary CSV changed')
    require((stage/'analysis/metrics.csv').read_bytes()==original.metric_csv_bytes(expected),'N3b metric CSV changed')
    require((stage/'analysis/REPORT.md').read_bytes()==wrapper.render(summary,p).encode(),'N3b report lost post-timing boundary')
    return dict(**raw,evidence_role=p['evidence_role'],confirmatory=False,
                preserved_N4_boundaries=p['N4_outcome_boundaries'])


def validate_n5_binding_timing():
    stage=c.OUT/'stage_n5/revision_002';p,sources=suite.protocol(stage/'preregistration/protocol.json')
    result,members=suite.attempt(stage/'binding/attempt_001',p)
    require(result['status']=='OK' and result['formal_timing'] is False and result['timing_events_created']==0,
            'N5 binding was used as formal timing')
    require(result['native_rebuilds']==0 and result['compiler_intervention'] is False
            and result['native_SHA256']==c.NATIVE and result['backend_SHA256']==p['backend_SHA256'],
            'N5 compiler/native contract changed')
    require(result['counts']==dict(cases=3,programs=16,full_correctness_checks=48,passing_checks=48),
            'N5 binding/check coverage changed')
    require(len(p['protected_SHA256'])==1113,'N5 protected compiler-evidence set changed')
    reference_path=n1._path(p['candidate_reference_path'])
    require(c.sha(reference_path.read_bytes())==p['candidate_reference_SHA256'],'N5 candidate definition origin changed')
    reference=c.read(reference_path);by={meta['tag']:meta for meta in result['programs']}
    require(len(by)==len(result['programs'])==16,'N5 logical program IDs changed')
    source=str(c.SOURCE.relative_to(c.ROOT))+'/quant_adapter.py'
    for case in p['cases']:
        case_id=case['id'];row=next(x for x in result['records'] if x['case']==case_id)
        require(row['status']=='OK' and row['allocated_sections_equal'] and row['ABI_and_launch_equal']
                and row['resources_equal'],'N5 actual U/G0 gate failed')
        u,g=by[case_id+':U'],by[case_id+':G0']
        require(row['upstream']==u and row['metadata']==g,'N5 baseline record/index mismatch')
        prefix=lambda meta:meta['artifact_directory']+'/'
        require(elf(members[prefix(u)+'kernel.cubin'])==elf(members[prefix(g)+'kernel.cubin']),
                'N5 U/G0 allocated ELF differs')
        require(u['resources']==g['resources'],'N5 U/G0 compiler resources differ')
        for key in n1.RUNTIME_KEYS:
            require(u[key]==g[key],'N5 U/G0 ABI/launch differs: '+key)
        reference_id='N01_m4096_n128' if case['family']=='N01' else 'N02_k3584_p448_g128'
        menu=suite.host_menu(sources[source].decode(),case,reference['plans'][reference_id],'frozen_n5_quant_adapter')
        if case['followup_role']=='footprint':
            menu=[choice for choice in menu if choice['name']==case['selected_variant']]
        require(menu==p['candidate_definitions'][case_id],'N5 candidate definitions were retuned')
        expected={case_id+':U',case_id+':G0'}|{case_id+':GL_'+choice['name'] for choice in menu}
        require(set(row['compiled_programs'])==expected,'N5 candidate/alias coverage changed')
        for tag in expected:
            meta=by[tag];suite.artifact(meta,members,kind='actual')
            require(meta['case']==case and meta['compiler_options']==dict(num_warps=case['W'],num_stages=case['stages']),
                    'N5 program changed fixed case/launch')
            require(meta['fixture']['views'],'N5 program lacks actual Tensor fixture metadata')
            checks=[check for check in result['checks'] if check['tag']==tag]
            require(len(checks)==3 and {x['pattern'] for x in checks}=={'finite','structured','edge'},
                    'N5 missing correctness pattern')
            for check in checks:
                n1._check_correctness(check)
            if meta['variant'].startswith('GL_'):
                choice=next(choice for choice in menu if 'GL_'+choice['name']==meta['variant'])
                require(meta['choice']==choice,'N5 actual GL changed frozen intervention')
                require(meta['runtime_argument_names']==u['runtime_argument_names']
                        and meta['specialized_runtime_values']==u['specialized_runtime_values'],
                        'N5 candidate changed original runtime specialization')
    timing_payload=stage/'binding/attempt_001/accepted_timing_payload.zip'
    selected=n1._zip_members(timing_payload);index=json.loads(selected['result.json'])
    timing_cases={case['id'] for case in p['cases'] if case['followup_role']=='footprint'}
    timing_programs={tag:meta for tag,meta in by.items() if meta['geometry'] in timing_cases}
    require(len(timing_programs)==6 and index['programs']==[meta for meta in result['programs']
            if meta['geometry'] in timing_cases],'N5 timing payload changed accepted binary selection')
    require(index['source_binding_protocol_SHA256']==result['protocol_SHA256'],'N5 timing payload origin mismatch')
    for meta in timing_programs.values():
        for filename in meta['artifact_SHA256']:
            name=meta['artifact_directory']+'/'+filename
            require(selected[name]==members[name],'N5 timing payload changed compiler artifact bytes')
    measured,_,analysis=suite.validate_measurement('stage_n5_timing',timing_programs)
    require(measured['analysis']['raw_timing_samples']==5580,'N5 formal raw sample coverage changed')
    return dict(status='OK',actual_JIT_programs=16,full_correctness_checks=48,
        protected_prior_compiler_files=1113,footprint_measurement=measured,
        observed_candidate_gains=[dict(case=row['case'],variant=row['variant'],
            gain_vs_U_percent=row['gain_vs_U_percent'],CI95=row['gain_vs_U_CI95'])
            for row in analysis['rows'] if row['variant'].startswith('GL_')],
        scientific_outcome_required=False)


def validate_n5_profiles():
    module=importlib.import_module('.n5_profile_analyze',__package__)
    stage=c.OUT/'stage_n5_profiles';p,raw=profile_raw('stage_n5_profiles',6)
    require(all(group['modes']==['warm_application'] for group in p['profile_groups']),
            'N5 profiling introduced an undeclared mode')
    require(p['InstructionStats_collected'] is False and p['maximum_concurrent_GPU_groups']==1,
            'N5 introduced software counts or changed declared sequential collection')
    payload=stage/'compiled_payload.zip'
    expected=module.analyze(p,stage/'profiles',payload)
    summary=c.read(stage/'analysis/summary.json')
    actual_canonical=profile_paths(summary)
    for key,value in profile_paths(expected).items():
        require(actual_canonical.get(key)==value,'N5 profile numerical analysis changed: '+key)
    # The helper supplies pure formatting functions; only derived paths are
    # canonicalized above. Raw metric names, values, units and outcomes stay exact.
    require((stage/'analysis/summary.csv').read_bytes()==module.csv_bytes(expected,p['metrics']),'N5 profile summary CSV changed')
    require((stage/'analysis/metrics_long.csv').read_bytes()==module.metric_csv_bytes(expected),'N5 profile metric CSV changed')
    require((stage/'analysis/REPORT.md').read_bytes()==module.render(summary,p).encode(),'N5 profile report changed')
    combined=importlib.import_module('.n5_report',__package__)
    report_path=c.OUT/'stage_n5/REPORT.md'
    require(report_path.read_bytes()==combined.render(combined.load_inputs(c.OUT)).encode(),
            'N5 combined report differs from frozen numeric inputs')
    return dict(**raw,software_instruction_stats_collected=False,
                combined_report_SHA256=c.sha(report_path.read_bytes()))


def run(include_n5_profiles=True):
    audit=n1.Audit();results={}
    manifest=c.read(c.OUT/'stage_n1/accepted_manifest.json')
    def composed(function):
        value=function()
        require(value['status']=='OK' and not value['errors'],'Composed stage validator failed: '+str(value.get('errors')))
        return value
    first=audit.check('N1',lambda:composed(lambda:
        n1.validate_stage_n1(accepted=manifest,require_complete=True,require_final=True)))
    if first:
        results['N1']=first
    second=audit.check('N2_N3_N4',lambda:composed(lambda:suite.run(confirmation='stage_n4')))
    if second:
        results['N2_N3_N4']=second
    results['N3b']=audit.check('N3b',validate_n3b)
    results['N5_binding_timing']=audit.check('N5_binding_timing',validate_n5_binding_timing)
    if include_n5_profiles:
        results['N5_profiles']=audit.check('N5_profiles',validate_n5_profiles)
    require(not any(name=='triton' or name.startswith('triton.') for name in sys.modules),
            'Final CPU validator imported Triton')
    return dict(status='FAILED' if audit.errors else ('OK' if include_n5_profiles else 'PREFLIGHT_OK'),
        CPU_only=True,GPU_allocations=0,compiler_invocations=0,checks_passed=len(audit.checks),
        errors=audit.errors,results=results,pending=[] if include_n5_profiles else ['N5_profiles'],
        validator_source_SHA256=c.sha(Path(__file__).read_bytes()),
        scope='N1 baseline, N2 exploration, N3 static profiles, N4 same-case confirmation, N3b post-timing diagnostic, and N5 bounded footprint/stride follow-up. No favorable scientific result is required.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=c.OUT/'FINAL_VALIDATION.json')
    args=parser.parse_args();report=run()
    c.write_once(args.output,report)
    print(json.dumps(dict(status=report['status'],checks_passed=report['checks_passed'],errors=report['errors']),indent=2))
    if report['status']!='OK':
        raise SystemExit(1)


if __name__=='__main__':
    main()
