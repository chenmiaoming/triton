"""Independent CPU provenance/completeness validation of cadence remeasurement.

No timing direction, practical equivalence, or favorable model outcome is a
validation requirement. The eligibility gate and all old artifacts stay intact.
"""
import argparse
from collections import Counter
import gzip
import importlib.util
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
import zipfile

from experiments.tma_reduction_layout.phase11.operator_gpu import common as c
from experiments.layout_coordination.clock_control.same_program_validate import (
    file_sha, safe_relative, telemetry_audit)

ROOT = c.ROOT
COUNTS = {'nonreduction': (16, 134, 150), 'operators': (64, 439, 505),
          'legacy': (216, 405, 594), 'anchors': (57, 417, 474)}


def require(value, message):
    if not value:
        raise ValueError(message)


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def reference(path):
    return dict(path=str(Path(path).resolve()), SHA256=file_sha(path))


def expected(protocol, suite):
    """Derive every original physical condition and alias from frozen inputs."""
    p = protocol['suites'][suite]
    contexts, bindings = {}, {}
    if suite == 'nonreduction':
        from .nonreduction_audit import physical_groups
        for source, parent in p['parent_protocols'].items():
            programs = p['programs_by_source'][source]
            require(programs == parent['programs'], 'Nonreduction original metadata changed')
            for case in parent['cases']:
                metas = [m for m in programs if m['geometry'] == case['id']]
                by_tag = {m['tag']: m for m in metas}
                fixture = next(m['fixture'] for m in metas if m['variant'] in ('U','G0') and 'fixture' in m)
                copies = len({fixture['views'][name]['group'] for name in fixture['inplace']})
                items = {}
                for group in physical_groups(metas):
                    tag = group['representative_tag']
                    meta = by_tag[tag]
                    items[tag] = dict(role=meta['variant'] if meta['variant'] in ('U','G0') else 'GL',
                        aliases=group['aliases'], nodes=dict(kernels=1,memsets=0,copies=copies))
                    bindings[tag] = dict(meta=meta,
                        artifact=source+'/'+meta['artifact_directory']+'/kernel.cubin')
                contexts[case['id']] = items
    elif suite == 'operators':
        from .operators import inference_contexts, training_contexts, representatives
        compiled = p['compiled']
        records = compiled['records']
        kernels = {m['tag']: SimpleNamespace(meta=m) for m in records}
        for group, work, mode, context in inference_contexts(p['parent_protocol']):
            metas = [m for m in records if m['group'] == group['id'] and not m['stage2']]
            contexts[context['id']] = {row['tag']: dict(role=row['role'],aliases=row['aliases'],
                nodes=dict(kernels=2 if mode=='decode' else 1,memsets=0,copies=0))
                for row in representatives(metas,kernels,mode)}
        for case,mode,m,context in training_contexts(p['parent_protocol']):
            metas = [row for row in records if row['geometry']==case['id'] and row['role']=='merge']
            contexts[context['id']] = {row['tag']: dict(role=row['role'],aliases=row['aliases'],
                nodes=dict(kernels=2 if mode=='pipeline' else 1,
                           memsets=3 if mode=='pipeline' else 0,copies=0))
                for row in representatives(metas,kernels,mode)}
        bindings = {m['tag']: dict(meta=m,artifact=m['artifact_directory']+'/kernel.cubin') for m in records}
    elif suite == 'legacy':
        from .legacy import groups
        for (m,n,w,vector,consumer,q,grid), rows in groups(p):
            context_id=f'M3|M{m}|N{n}|W{w}|{vector}|{consumer}|q{q}|G{grid}'
            contexts[context_id] = {row['cell_id']: dict(
                role='default' if ':n_first:' in row['tag'] else 'candidate',
                aliases=row['aliases'], nodes=dict(kernels=1,memsets=0,copies=0)) for row in rows}
        bindings={tag:dict(meta=meta,resource=p['gate']['resources'][tag],
            artifact=tag.replace(':','/')+'/kernel.cubin') for tag,meta in p['gate']['metadata'].items()}
    else:
        from .anchors_audit import groups,selected_cells,default_tag
        for panel_name,panel in p['panels'].items():
            for (m,n,w,grid),rows in groups(panel):
                context_id=f'{panel_name}|M{m}|N{n}|W{w}|qnatural|G{grid}'
                items={}
                for row in rows:
                    meta=panel['gate']['metadata'][row['tag']]
                    role=('control' if meta['family']=='ONE_TRANSFER' else
                          'default' if row['tag']==default_tag(meta) else 'candidate')
                    items[panel_name+'|'+row['cell_id']]=dict(role=role,
                        aliases=[panel_name+'|'+a for a in row['aliases']],
                        nodes=dict(kernels=1,memsets=0,copies=0))
                contexts[context_id]=items
            for tag in {row['tag'] for row in selected_cells(panel)}:
                bindings[panel_name+'|'+tag]=dict(meta=panel['gate']['metadata'][tag],
                    resource=panel['gate']['resources'][tag],
                    artifact=panel_name+'/'+tag.replace(':','/')+'/kernel.cubin')
    original_count=sum(len(items) for items in contexts.values())
    total_count=0
    for items in contexts.values():
        bases=[tag for tag,row in items.items() if row['role']=='U']
        if not bases:bases=[tag for tag,row in items.items() if row['role']=='default']
        total_count+=len(items)+len(bases)
    require((len(contexts),original_count,total_count)==COUNTS[suite], 'Expected matrix changed: '+suite)
    return contexts,bindings


def preflight(protocol_path, root, suite):
    root=Path(root).resolve();protocol_path=Path(protocol_path).resolve()
    p=c.read(protocol_path)
    require(suite in COUNTS and suite in p['suites'], 'Unknown suite')
    timing=p['timing']
    require((timing['workers'],timing['visits'],timing['samples'])==(3,10,31), 'Sampling matrix changed')
    require(timing['primes']==8 and timing['warmups']==3, 'Priming/warmup changed')
    require(timing['host_gap_seconds']==.001 and timing['telemetry_period_s']==.05, 'Cadence/telemetry changed')
    require(p.get('profile')=='chenmiaoming', 'Profile changed')
    require(p.get('GPU')=='H100!' and p.get('clock_control')=='unlocked', 'Device/clock contract changed')
    require(p.get('GPU_configuration_writes') is False, 'GPU-setting mutation in protocol')
    require(p.get('compiler')=='20d9d98fead78c2f65e1b3ba5a7881d86a836dad', 'Compiler changed')
    require(p.get('native_SHA256')==c.NATIVE, 'Native compiler changed')
    require(p.get('kernel_compiles',0)==p.get('compiler_calls',0)==p.get('native_rebuilds',0)==0, 'Compilation requested')
    for name,digest in p['source_SHA256'].items():
        name_path=safe_relative(name)
        require(file_sha(root/p['snapshot']/name_path)==digest, 'Snapshot source mismatch: '+name)
        require(file_sha(ROOT/name_path)==digest, 'Current source mismatch: '+name)
    protected_path=root/safe_relative(p['protected_inventory_path'])
    require(file_sha(protected_path)==p['protected_inventory_SHA256'], 'Protected inventory changed')
    protected=c.read(protected_path)
    require(len(protected)==p['protected_file_count']>0, 'Protected-file coverage changed')
    for name,digest in protected.items():
        require(file_sha(ROOT/safe_relative(name))==digest, 'Old protected file changed: '+name)
    payload_path=root/safe_relative(p['suites'][suite]['payload_path'])
    require(file_sha(payload_path)==p['suites'][suite]['payload_SHA256'], 'Suite payload changed')
    require(file_sha(root/safe_relative(p['audit_path']))==p['audit_SHA256'], 'Historical impact audit changed')
    freeze=c.read(root/'preregistration/freeze.json')
    require(freeze['status']=='OK' and freeze['CPU_only'] is True and freeze['GPU_dispatches']==0, 'Invalid CPU freeze')
    require(freeze['protocol_SHA256']==c.sha(c.encode(p)), 'Frozen protocol mismatch')
    cpu_path=root/safe_relative(p['cpu_validation_path'])
    require(file_sha(cpu_path)==p['cpu_validation_SHA256'], 'CPU behavioral validation changed')
    cpu=c.read(cpu_path)
    require(cpu['status']=='OK' and cpu['GPU_dispatches']==0 and cpu['CUDA_libraries_loaded'] is False, 'CPU behavioral gate failed')
    engine_name='experiments/layout_coordination/cadence_retest/engine.py'
    require(cpu['engine_source_SHA256']==p['source_SHA256'][engine_name], 'CPU tests used a different capture engine')
    require(len(cpu['tests'])==5 and all(row['status']=='OK' for row in cpu['tests']), 'CPU behavior tests incomplete')
    for name,digest in cpu['reviewed_sources'].items():
        relative=str(Path(name).resolve().relative_to(ROOT))
        require(p['source_SHA256'][relative]==digest, 'CPU reviewed source differs from snapshot')
    contexts,bindings=expected(p,suite)
    with zipfile.ZipFile(payload_path) as archive:
        names=archive.namelist()
        require(len(names)==len(set(names)), 'Duplicate payload members')
        for name in names:safe_relative(name)
        for row in bindings.values():
            require(c.sha(archive.read(row['artifact']))==row['meta']['cubin_SHA256'], 'Payload CUBIN changed: '+row['artifact'])
        # A combined ZIP may change container metadata; every original member
        # must nevertheless remain byte-identical under its explicit namespace.
        if suite in ('nonreduction','anchors'):
            sources=(p['suites'][suite]['original_payloads'] if suite=='nonreduction' else
                {name:dict(path=panel['source']['payload_path'],SHA256=panel['payload_SHA256'])
                 for name,panel in p['suites'][suite]['panels'].items()})
            for namespace,source in sources.items():
                old_path=ROOT/safe_relative(source['path'])
                require(file_sha(old_path)==source['SHA256'], 'Original payload lineage changed')
                with zipfile.ZipFile(old_path) as prior:
                    for name in prior.namelist():
                        if not name.endswith('/'):
                            require(archive.read(namespace+'/'+name)==prior.read(name), 'Namespaced original artifact changed: '+name)
        else:
            source=p['suites'][suite]['original_payload']
            require(file_sha(ROOT/safe_relative(source['path']))==source['SHA256']==p['suites'][suite]['payload_SHA256'], 'Byte-exact original payload not reused')
    if suite=='nonreduction':
        for source,stage in [('n2','stage_n2'),('n5','stage_n5_timing')]:
            old=ROOT/'experiments/tma_reduction_layout/results/phase11/nonreduction'/stage/'preregistration/protocol.json'
            require(p['suites'][suite]['parent_protocols'][source]==c.read(old), 'Nonreduction parent protocol changed')
            proof=p['suites'][suite]['eligibility_by_source'][source]
            proof_path=ROOT/safe_relative(proof['path'])
            require(file_sha(proof_path)==proof['SHA256'], 'Prior three-pattern proof changed')
            prior=c.read(proof_path)
            require(prior['status']=='OK' and prior['protocol_SHA256']==c.sha(c.encode(p['suites'][suite]['parent_protocols'][source])), 'Prior correctness gate failed')
            patterns={}
            for check in prior['checks']:
                require(check['pass'] is True and all(check['guards']), 'Prior correctness/guards failed')
                for tag in [check['tag'],*check.get('aliases',[])]:
                    patterns.setdefault(tag,set()).add(check['pattern'])
            require(all(patterns.get(m['tag'])=={'finite','structured','edge'}
                for m in p['suites'][suite]['programs_by_source'][source]), 'Prior pattern coverage incomplete')
    elif suite=='operators':
        require(p['suites'][suite]['compiled']['status']=='OK', 'Old compile gate failed')
        require(p['suites'][suite]['compiled']['payload_SHA256']==p['suites'][suite]['payload_SHA256'], 'Old G8 payload not reused exactly')
    for source in p['suites'][suite].get('old_raw_refs',[]):
        require(file_sha(ROOT/safe_relative(source['path']))==source['SHA256'], 'Historical timing reference changed')
    result=dict(status='OK',protocol=reference(protocol_path),suite=suite,
        payload=reference(payload_path),source_files=len(p['source_SHA256']),
        protected_files=len(protected),expected_contexts=len(contexts),
        expected_original_cells=COUNTS[suite][1],expected_cells_with_shams=COUNTS[suite][2],
        outcome_requirement='None; provenance, correctness and observation completeness only.')
    return p,result,contexts,bindings


def _exports(suite_root,phase):
    return sorted((Path(suite_root)/phase/'attempt_001').glob('invocation_*/export/result.json'))


def _worker(path,p,suite,contexts,binding_specs,invocation):
    raw=c.read(path);export=path.parent;folder=export.parent
    require(raw['status']=='OK' and raw['suite']==suite and raw['invocation']==invocation, 'Worker failed/wrong identity')
    require(raw['formal_timing']==(invocation>0), 'Formal/check mode mismatch')
    require(raw['protocol_SHA256']==c.sha(c.encode(p)), 'Worker protocol mismatch')
    require(raw['payload_SHA256']==p['suites'][suite]['payload_SHA256'], 'Worker payload mismatch')
    require(raw['clock_control']=='unlocked' and raw['profiler_attached'] is False, 'Clock/profiler mode changed')
    require(raw['compile_calls']==raw['native_rebuilds']==0 and raw['Triton_imported'] is False, 'Compiler imported or executed')
    require(raw['driver_return_codes']==[0] and not raw['cleanup_errors'] and not raw['partial_errors'], 'CUDA/cleanup/partial error')
    environment=raw['environment']
    require(environment['cc']==[9,0] and environment['SM_count']==132, 'Wrong GPU architecture')
    require('H100' in environment['name'] and 'H200' not in environment['name'], 'Wrong GPU product')
    require(environment['readonly']['read_only'] is True, 'No read-only environment proof')
    require(environment['readonly']['nvml']['requested_uuid']==environment['gpu_uuid'], 'NVML/CUDA UUID mismatch')
    require(c.read(folder/'raw.json')==raw, 'Raw/export result mismatch')
    dispatch=c.read(folder/'dispatch.json')
    require(dispatch['protocol_SHA256']==raw['protocol_SHA256'] and dispatch['payload_SHA256']==raw['payload_SHA256'], 'Dispatch byte provenance mismatch')
    require(dispatch['suite']==suite and dispatch['invocation']==invocation and dispatch['attempt']==1,
        'Dispatch suite/worker/attempt mismatch')
    require(dispatch['profile']=='chenmiaoming' and bool(dispatch['call_id']), 'Dispatch account/call missing')
    require(dispatch['action']==('timing' if invocation else 'check'), 'Dispatch action mismatch')
    if invocation:
        gate=folder.parents[2]/'check/attempt_001/validation.json'
        require(dispatch['eligibility_gate_SHA256']==file_sha(gate), 'Timing did not pin the accepted eligibility gate')
    else:
        require(dispatch['eligibility_gate_SHA256'] is None, 'Unexpected check gate dependency')
    manifest=c.read(export/'export_manifest.json')
    require(manifest['status']=='OK' and manifest['result_status']==raw['status'], 'Export manifest/result mismatch')
    actual_files={str(f.relative_to(export)) for f in export.rglob('*')
        if f.is_file() and f.name!='export_manifest.json' and '__pycache__' not in f.parts}
    require(set(manifest['files'])==actual_files, 'Export manifest coverage mismatch')
    with zipfile.ZipFile(folder/'raw_export.zip') as archive:
        require(json.loads(archive.read('result.json'))==raw, 'Archive/result mismatch')
        require(len(archive.namelist())==len(set(archive.namelist())), 'Duplicate export archive members')
        require(set(archive.namelist())==actual_files|{'export_manifest.json'}, 'Archive manifest coverage mismatch')
        require(archive.read('export_manifest.json')==(export/'export_manifest.json').read_bytes(), 'Archive/export manifest differs')
        for name,digest in manifest['files'].items():
            relative=safe_relative(name)
            require(file_sha(export/relative)==digest and c.sha(archive.read(name))==digest, 'Archive/export bytes changed: '+name)
    require(set(raw['bindings'])==set(binding_specs), 'Missing/extra kernel binding')
    for tag,row in raw['bindings'].items():
        spec=binding_specs[tag];meta=spec['meta']
        digest=row.get('cubin_SHA256',row.get('CUBIN_SHA256'))
        loaded=row.get('loaded_SHA256',row.get('loaded_CUBIN_SHA256'))
        require(digest==loaded==meta['cubin_SHA256'], 'Loaded CUBIN changed: '+tag)
        artifact=export/'program_files'/spec['artifact']
        require(file_sha(artifact)==digest, 'Exported CUBIN changed: '+tag)
        if 'resource' in spec:
            for key in ('ABI','attributes','reservations'):
                require(row[key]==spec['resource'][key], 'Legacy resource/ABI changed: '+tag+'/'+key)
        else:
            text=artifact.with_suffix('.ptx').read_text()
            match=re.search(r'\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid',text,re.S)
            require(bool(match),'Missing frozen PTX ABI')
            abi=[re.search(r'\.param\s+\.(\w+)',line)[1] for line in match[1].splitlines() if '.param' in line]
            require(row['ABI']==abi, 'Pointer ABI changed: '+tag)
            require(row['attributes']['occupancy_capacity_q']>0, 'Invalid occupancy capacity')
            if meta.get('resources'):
                require(row['attributes']['n_regs']==meta['resources']['REG'], 'Register allocation changed')
                require(row['attributes']['local_bytes']==meta['resources']['LOCAL'], 'Local allocation changed')
    shards={};raw_count=0
    for entry in raw['shards']:
        relative=safe_relative(entry['path']);shard_path=export/relative
        require(file_sha(shard_path)==entry['SHA256'], 'Shard changed')
        shard=c.read(shard_path);context=shard['context'];cid=context['id']
        require(cid not in shards and cid in contexts, 'Missing/duplicate/extra context: '+cid)
        require(context==entry['context'], 'Shard context mismatch')
        originals=contexts[cid]
        items={tag:dict(**row,sham_of=None) for tag,row in originals.items()}
        bases=[tag for tag,row in originals.items() if row['role']=='U']
        if not bases:bases=[tag for tag,row in originals.items() if row['role']=='default']
        for tag in bases:
            items[tag+':repeat']=dict(role='U_repeat' if originals[tag]['role']=='U' else 'default_repeat',
                aliases=[],sham_of=tag,nodes=originals[tag]['nodes'])
        cells={row['tag']:row for row in shard['cells']}
        require(set(cells)==set(items) and len(cells)==len(shard['cells']), 'Incomplete/duplicate physical cells')
        require(entry['physical_cells']==len(cells), 'Shard physical count mismatch')
        require(entry['logical_cells']==sum(1+len(x['aliases']) for x in cells.values()), 'Shard logical count mismatch')
        by_visit={}
        for tag,cell in cells.items():
            item=items[tag]
            require(cell['context']==context and cell['role']==item['role'] and cell['sham_of']==item['sham_of'], 'Cell identity changed')
            require(set(cell['aliases'])==set(item['aliases']) and len(cell['aliases'])==len(item['aliases']), 'Alias lineage changed')
            require(len(cell['visits'])==(10 if invocation else 0), 'Wrong visit count')
            require(sorted(v['visit'] for v in cell['visits'])==(list(range(10)) if invocation else []), 'Missing/duplicate visit')
            for visit in cell['visits']:
                for field in ('samples_us','host_gaps_ns','launch_ns','completion_ns'):
                    require(len(visit[field])==31 and all(numeric(x) for x in visit[field]), 'Invalid samples: '+field)
                require(min(visit['samples_us'])>0, 'Nonpositive target time')
                require(min(visit['host_gaps_ns'])>=1_000_000, 'Host gap below 1ms')
                require(all(visit['start_ns']<=a<b<=visit['end_ns'] for a,b in zip(visit['launch_ns'],visit['completion_ns'])), 'Bad observation window')
                by_visit[visit['visit'],tag]=visit
                raw_count+=31
        require(len(shard['orders'])==(10 if invocation else 0), 'Missing round orders')
        for order in shard['orders']:
            require(len(order['round_orders'])==31, 'Wrong round count')
            for index,tags in enumerate(order['round_orders']):
                require(len(tags)==len(items) and set(tags)==set(items), 'Nonuniform observation round')
                observed=sorted((by_visit[order['visit'],tag]['launch_ns'][index],tag) for tag in items)
                require([tag for _,tag in observed]==tags, 'Recorded order differs from launch timestamps')
        require(entry['raw_samples']==(len(cells)*310 if invocation else 0), 'Shard raw count mismatch')
        shards[cid]=shard
    require(set(shards)==set(contexts) and len(shards)==COUNTS[suite][0], 'Incomplete context matrix')
    checks={(row['context']['id'],row['tag'],row['phase']):row for row in raw['checks']}
    require(len(checks)==len(raw['checks'])==COUNTS[suite][1]*3, 'Correctness coverage changed')
    proofs={(row['context']['id'],row['tag']):row for row in raw['graph_proofs']}
    require(len(proofs)==len(raw['graph_proofs'])==COUNTS[suite][1], 'Graph proof coverage changed')
    for cid,items in contexts.items():
        for tag,item in items.items():
            for phase in ('before','graph_before','graph_after'):
                check=checks[cid,tag,phase]
                require(check['pass'] is True and check['role']==item['role'], 'Correctness/graph oracle failed')
                snapshots=check.get('immutable_restore_snapshots',[])
                require(len(snapshots)==item['nodes']['copies'], 'Missing immutable GPU restore proof')
                for snapshot in snapshots:
                    require(snapshot['pass_'] is True and snapshot['actual_SHA256']==snapshot['initial_SHA256'], 'GPU restoration source changed')
                    fixture=check['context']['expected_fixture']
                    require(snapshot['initial_SHA256']==fixture['initial_storage_SHA256'][snapshot['group']], 'GPU restore source differs from original input')
            proof=proofs[cid,tag];nodes=item['nodes']
            require(proof['expected_nodes_per_semantic_iteration']==nodes and proof['primes']==8, 'Semantic graph plan changed')
            desired=[0]*(9*nodes['kernels'])+[7,7]+[2]*(9*nodes['memsets'])+[1]*(9*nodes['copies'])
            require(sorted(proof['node_types'])==sorted(desired), 'Actual graph nodes changed')
            is_base=item['role']=='U' or (item['role']=='default' and not any(x['role']=='U' for x in items.values()))
            replays=2 if not invocation else 342+(310 if is_base else 0)
            require(proof['replays']==replays, 'Graph replay coverage changed')
    telemetry_meta=raw['telemetry'];telemetry_path=export/safe_relative(telemetry_meta['path'])
    require(file_sha(telemetry_path)==telemetry_meta['SHA256'], 'Compressed telemetry changed')
    with gzip.open(telemetry_path,'rt') as stream:telemetry=[json.loads(line) for line in stream]
    require(len(telemetry)==telemetry_meta['rows'], 'Telemetry row count mismatch')
    telemetry_result=telemetry_audit({'telemetry':telemetry})
    if suite=='legacy':
        require(raw['readonly_input_check']['pass'] is True, 'Legacy read-only input changed')
        require(len(raw['execution_bindings'])==405, 'Legacy execution binding coverage changed')
        if invocation==0:
            eligibility=raw['complete_eligibility']
            require(len(eligibility)==len(p['suites'][suite]['gate']['metadata'])*len(p['suites'][suite]['parent_protocol']['correctness']['patterns']), 'Legacy pattern eligibility incomplete')
            require(all(row['pass'] for row in eligibility), 'Legacy eligibility pattern failed')
    if suite=='anchors':
        require(raw['D1_new_blind_claim'] is False and raw['selector_refit'] is False, 'Historical blindness/model freeze altered')
        require(len(raw['execution_bindings'])==417, 'Anchor execution binding coverage changed')
        require(len(raw['readonly_input_checks'])==2 and all(x['pass'] for x in raw['readonly_input_checks']), 'Anchor read-only inputs changed')
        if invocation==0:
            eligibility=raw['complete_eligibility']
            require(set(eligibility)=={'c2d','d1'}, 'Anchor eligibility panels incomplete')
            require(sum(len(x) for x in eligibility.values())==417 and all(x['pass'] for rows in eligibility.values() for x in rows), 'Anchor pattern eligibility incomplete/failed')
    return dict(invocation=invocation,gpu_uuid=environment['gpu_uuid'],contexts=len(shards),
        original_cells=COUNTS[suite][1],physical_cells_with_shams=COUNTS[suite][2],
        raw_samples=raw_count,checks=len(checks),telemetry=telemetry_result)


def validate(protocol_path,root,suite,*,eligibility_only=False,preregistration_only=False):
    root=Path(root).resolve();protocol_path=Path(protocol_path).resolve()
    p,registration,contexts,binding_specs=preflight(protocol_path,root,suite)
    if preregistration_only:return dict(status='OK',scope='CPU preregistration only',preregistration=registration)
    suite_root=root/suite
    check_paths=_exports(suite_root,'check')
    require(len(check_paths)==1 and check_paths[0].parent.parent.name=='invocation_000', 'One completed eligibility worker required')
    check=_worker(check_paths[0],p,suite,contexts,binding_specs,0)
    if eligibility_only:
        return dict(status='OK',scope='GPU exact-graph eligibility; no formal timing score',
            protocol_SHA256=c.sha(c.encode(p)),suite=suite,preregistration=registration,worker=check)
    gate_path=suite_root/'check/attempt_001/validation.json'
    gate=c.read(gate_path)
    require(gate['status']=='OK' and gate['protocol_SHA256']==c.sha(c.encode(p)) and gate['suite']==suite, 'Eligibility gate missing/changed')
    paths=_exports(suite_root,'timing')
    require(len(paths)==3,'Three completed formal workers required')
    workers=[]
    for invocation,path in enumerate(paths,1):
        require(path.parent.parent.name==f'invocation_{invocation:03}', 'Unexpected formal worker index')
        workers.append(_worker(path,p,suite,contexts,binding_specs,invocation))
    require(len({w['gpu_uuid'] for w in workers})==3,'Formal workers must have distinct GPU UUIDs')
    require(sum(w['raw_samples'] for w in workers)==COUNTS[suite][2]*310*3,'Incomplete raw sample matrix')
    from . import analyze
    result=analyze.analyze(p,suite,suite_root)
    outputs={'analysis.json':c.encode(result),'timing.csv':analyze.csv_bytes(result),
             'REPORT.md':analyze.render(result).encode()}
    for name,blob in outputs.items():
        require((suite_root/'analysis'/name).read_bytes()==blob,'Derived analysis does not reproduce: '+name)
    return dict(status='OK',scope='Complete fixed-cadence remeasurement, provenance and derived-output audit',
        protocol_SHA256=c.sha(c.encode(p)),suite=suite,preregistration=registration,
        eligibility_gate=reference(gate_path),check_worker=check,workers=workers,
        raw_samples=sum(w['raw_samples'] for w in workers),analysis_recomputed=True,
        derived_SHA256={name:c.sha(blob) for name,blob in outputs.items()},
        outcome_requirement='None: uncertain, adverse, or changed conclusions are preserved.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--suite',choices=tuple(COUNTS),required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--eligibility-only',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=validate(args.protocol,args.root,args.suite,
        eligibility_only=args.eligibility_only,preregistration_only=args.preflight)
    if args.output:c.write_once(args.output,result)
    print(json.dumps(dict(status=result['status'],suite=args.suite,
        preflight=args.preflight,eligibility_only=args.eligibility_only,
        raw_samples=result.get('raw_samples'),output=str(args.output) if args.output else None),sort_keys=True))


if __name__=='__main__':main()
