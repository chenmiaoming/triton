"""Offline completeness and provenance validator for replacement G8 evidence.

Scientific directions are observations, never validation requirements.  This
module imports reports and frozen metadata only; it does not launch a kernel,
allocate a GPU, or invoke a compiler.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import zipfile

from . import common as c
from .g5_analyze import elf
from .g8_analyze import (EXPECTED_PANELS, collect, expected_conditions, gate_path,
                         payload_path, physical_identity, protocol_path)
from .layout_parsing import encodings, plan, tensor_encoding
from .offline_import import hardware_rows, source_blocks

OUT = c.OUT/'stage_g8'


def exact_export(folder):
    """Check extracted files against the unmodified returned archive."""
    with zipfile.ZipFile(folder/'raw_export.zip') as archive:
        names = [name for name in archive.namelist() if not name.endswith('/')]
        assert len(names) == len(set(names)), folder
        assert all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in names)
        extracted = {str(path.relative_to(folder/'export')) for path in (folder/'export').rglob('*') if path.is_file()}
        assert set(names) == extracted, folder
        for name in names:
            assert archive.read(name) == (folder/'export'/name).read_bytes(), name


def frozen_inputs():
    protocols = sorted((OUT/'preregistration').glob('protocol*.json'))
    assert protocols
    for path in protocols:
        protocol = c.read(path)
        suffix = path.stem.removeprefix('protocol')
        with zipfile.ZipFile(path.with_name('source_snapshot'+suffix+'.zip')) as archive:
            for name, digest in protocol['worker_source_SHA256'].items():
                assert c.sha(archive.read(name)) == digest, name
        for name, digest in {**protocol['protected_SHA256'], **protocol.get('prior_attempt_SHA256', {})}.items():
            assert c.sha((c.ROOT/name).read_bytes()) == digest, name
    protocol = c.read(protocol_path()); c.verify_sources(protocol)
    assert protocol['compiler'] == '20d9d98fead78c2f65e1b3ba5a7881d86a836dad'
    assert protocol['native_SHA256'] == c.NATIVE and not protocol['compiler_intervention']
    assert protocol['requested_gpu'] == 'H100!:1' and protocol['profile'] == 'chenmiaoming'
    assert protocol['timing']['no_result_dependent_repeats']
    assert c.sha((OUT/'preregistration/reference_payload.zip').read_bytes()) == protocol['reference_payload_SHA256']
    gate = c.read(gate_path()); payload = payload_path().read_bytes()
    assert gate['status'] == 'OK' and gate['protocol_SHA256'] == c.sha(c.encode(protocol))
    assert gate['payload_SHA256'] == c.sha(payload) and gate['native_rebuilds'] == 0
    root = c.ROOT/gate['CPU_export_directory']
    exact_export(root.parent)
    raw = c.read(root/'result.json')
    assert raw['status'] == 'OK' and not raw['GPU_allocated'] and not raw['compiler_intervention']
    assert raw['native_SHA256'] == c.NATIVE and raw['ccache_before'] == raw['ccache_after']
    assert raw['protocol_SHA256'] == c.sha(c.encode(protocol))
    assert raw['records'] == gate['records']
    metas = {row['tag']:row for row in gate['records']}
    assert len(metas) == len(gate['records'])
    expected = {g['id']+':'+v['name'] for g in protocol['inference_groups'] for v in g['variants']}
    expected |= {g['id']+':stage2' for g in protocol['inference_groups'] if g['case']['family'] == 'C02'}
    expected |= {g['id']+':'+v['name'] for g in protocol['training_cases'] for v in g['variants']}
    expected |= {g['id']+':dx' for g in protocol['training_cases']}
    assert set(metas) == expected
    cases = {g['id']:g['case'] for g in protocol['inference_groups']}
    cases.update({g['id']:g for g in protocol['training_cases']})
    audit = {row['tag']:row for row in gate['effective_layout_audit']}
    assert set(audit) == {tag for tag, meta in metas.items() if meta['role'] not in ['dx', 'stage2']}
    defaults = 0
    with zipfile.ZipFile(payload_path()) as archive:
        expected_files = set()
        for tag, meta in metas.items():
            folder = meta['artifact_directory']
            for name, digest in meta['artifact_SHA256'].items():
                key = folder+'/'+name; expected_files.add(key)
                assert c.sha(archive.read(key)) == digest, (tag, key)
                assert archive.read(key) == (root/key).read_bytes(), (tag, key)
            assert c.sha(archive.read(folder+'/kernel.cubin')) == meta['cubin_SHA256']
            if meta['role'] in ['dx', 'stage2']:
                continue
            case = cases[meta['group']]; ir = archive.read(folder+'/kernel.ttgir').decode()
            if case['family'] == 'C02':
                actual = plan(ir, case)
                assert actual['V'] == meta['requested_input'] and actual['K'] == meta['requested_K']
                assert actual['output'] == meta['requested_output']
            else:
                defs = encodings(ir)
                load = next(s for s in ir.splitlines() if 'tt.load ' in s and f'tensor<{case["BM"]}x{case["BN"]}x' in s)
                store = next(s for s in ir.splitlines() if 'tt.store ' in s and f'tensor<{case["BN"]}x' in s)
                actual = dict(input=tensor_encoding(load, [case['BM'],case['BN']], defs),
                              output=tensor_encoding(store, [case['BN']], defs))
                expected_out = {'kind':'slice','dim':0,'parent':meta['requested_input']} if meta['native_output'] else meta['requested_output']
                assert actual['input'] == meta['requested_input'] and actual['output'] == expected_out
            assert audit[tag]['requested_plan_preserved'] and audit[tag]['effective_plan'] == actual
            if meta['variant']['name'].endswith('gluon_anchor'):
                prefix = meta['variant']['name'].removesuffix('gluon_anchor')
                reference = metas[meta['group']+':'+prefix+'upstream']
                assert elf(archive.read(folder+'/kernel.cubin')) == elf(archive.read(reference['artifact_directory']+'/kernel.cubin')), tag
                for key in ['runtime_argument_names','num_warps','dynamic_smem_bytes','global_scratch_size']:
                    assert meta[key] == reference[key], (tag, key)
                defaults += 1
        packaged = {name for name in archive.namelist() if not name.endswith('/')}
        extras = packaged-expected_files
        assert all(Path(name).name == 'metadata.json' and Path(name).parent.as_posix() in
                   {meta['artifact_directory'] for meta in metas.values()} for name in extras)
        for name in extras:
            assert archive.read(name) == (root/name).read_bytes()
        assert expected_files <= packaged
    assert len(expected_conditions(protocol)) == 517
    return protocol, gate, metas, dict(logical_programs=len(metas), source_aligned_default_pairs=defaults,
        new_unique_Gluon_programs=raw['new_unique_Gluon_programs'], reused_reference_programs=raw['reused_reference_programs'],
        effective_layouts_checked=len(audit), native_rebuilds=0, protected_files_unchanged=len(protocol['protected_SHA256']))


def runtime_checks(protocol, gate, metas):
    payload_sha = gate['payload_SHA256']; checks = 0; logical_checks = 0; environments = []
    for invocation in range(protocol['timing']['workers']+1):
        path = OUT/('correctness/attempt_001/raw.json' if not invocation else f'timing/invocation_{invocation}/raw.json')
        raw = c.read(path)
        assert raw['status'] == 'OK' and raw['invocation'] == invocation
        assert raw['protocol_SHA256'] == c.sha(c.encode(protocol)) and raw['payload_SHA256'] == payload_sha
        for domain in ['inference','training']:
            part = raw[domain]
            assert part['status'] == 'OK' and part['invocation'] == invocation
            assert part['protocol_SHA256'] == raw['protocol_SHA256'] and part['payload_SHA256'] == payload_sha
            assert part['driver_return_codes'] == [0] and not part['Triton_imported'] and part.get('compile_calls',0) == 0
            assert part['environment']['cc'] == [9,0] and 'H100' in part['environment']['name']
            environments.append(dict(invocation=invocation, domain=domain, **part['environment']))
            family_metas = {tag:meta for tag,meta in metas.items() if (meta['family'] == 'C01') == (domain == 'training')}
            assert set(part['bindings']) == set(family_metas)
            for tag, binding in part['bindings'].items():
                assert binding['cubin_SHA256'] == binding['loaded_SHA256'] == metas[tag]['cubin_SHA256']
                if binding['guards'] == 0:
                    # The training runner loads every logical tag, then
                    # launches each identical CUBIN/ABI representative once.
                    alias_rows = [row for row in part['checks'] if tag in row.get('aliases',[])]
                    assert domain == 'training' and alias_rows, tag
                    for row in alias_rows:
                        assert row['pass'] and part['bindings'][row['tag']]['guards'] > 0
                        assert physical_identity(metas[tag]) == physical_identity(metas[row['tag']])
                else:
                    assert binding['guards'] > 0, tag
                if metas[tag].get('variant',{}).get('name','').endswith('gluon_anchor'):
                    prefix = metas[tag]['variant']['name'].removesuffix('gluon_anchor')
                    reference = part['bindings'][metas[tag]['group']+':'+prefix+'upstream']
                    assert binding['ABI'] == reference['ABI'] and binding['attributes'] == reference['attributes']
            patterns = ['finite'] if invocation else protocol['correctness' if domain == 'inference' else 'training_correctness']['patterns']
            if domain == 'inference':
                expected = {(g['id'],w['batch'],w['length'],pattern,g['id']+':'+variant['name'])
                            for g in protocol['inference_groups'] for w in g['workloads'] for pattern in patterns for variant in g['variants']}
                observed = [(row['group'],row['batch'],row['length'],row['pattern'],row['tag']) for row in part['checks']]
            else:
                expected = {(g['id'],mode,m,pattern,g['id']+':'+variant['name'])
                            for g in protocol['training_cases'] for mode,m in [('merge',128)]+[('pipeline',m) for m in g['pipeline_M']]
                            for pattern in patterns for variant in g['variants']}
                observed = []
                for row in part['checks']:
                    aliases = [row['tag']]+row.get('aliases',[])
                    assert len(aliases) == len(set(aliases))
                    for tag in aliases:
                        assert physical_identity(metas[tag]) == physical_identity(metas[row['tag']])
                        observed.append((metas[tag]['geometry'],row['mode'],row['M'],row['pattern'],tag))
            assert len(observed) == len(set(observed)) == len(expected), (invocation,domain)
            assert set(observed) == expected, (invocation,domain,expected-set(observed),set(observed)-expected)
            assert all(row['pass'] for row in part['checks'])
            for row in part['checks']:
                if domain == 'inference':
                    assert row['details'] and all(item['pass'] and item['canary_pass'] for item in row['details'])
                else:
                    assert row['outputs'] and all(item['pass'] for item in row['outputs'])
                    assert row['buffers'] and all(item['canaries_and_inputs_pass'] for item in row['buffers'])
            checks += len(part['checks']); logical_checks += len(observed)
        if not invocation:
            assert not raw['inference']['cells']
            assert all(not cell['visits'] for cell in raw['training']['cells'])
    expected, arrays, representatives, timing = collect(protocol, gate, payload_sha)
    assert dict(Counter(row['panel'] for row in expected.values())) == EXPECTED_PANELS
    analysis = c.read(OUT/'timing_analysis.json')
    assert analysis['status'] == 'OK' and analysis['logical_conditions'] == len(expected) == len(analysis['rows']) == 517
    assert analysis['protocol_SHA256'] == c.sha(c.encode(protocol)) and analysis['payload_SHA256'] == payload_sha
    assert analysis['analysis_source_SHA256'] == c.sha((c.SOURCE/'g8_analyze.py').read_bytes())
    assert analysis['raw_samples'] == timing['raw_samples'] and analysis['worker_counts'] == timing['worker_counts']
    assert analysis['raw_sources'] == timing['raw_sources']
    assert analysis['logical_conditions_by_panel'] == EXPECTED_PANELS
    assert len(analysis['contrasts']) == 50
    seen_rows = set()
    for row in analysis['rows']:
        if row['domain'] == 'inference':
            key = ('inference',row['panel'],row['group'],row['batch'],row['length'],row['mode'],row['tag'])
        else:
            key = ('training','g3b',row['group'],row['mode'],row['M'],row['tag'])
        assert key in expected and key not in seen_rows; seen_rows.add(key)
        assert all(row[name] == value for name,value in expected[key].items())
        values = arrays[key]
        assert row['worker_visit_medians_us'] == values.tolist() and row['time_us'] == float(values.mean())
        assert row['physical_representative_tags'] == [representatives[key][i] for i in range(1,4)]
        for baseline,tag_name in [('U','upstream_tag'),('G0','anchor_tag')]:
            reference = arrays[key[:-1]+(row[tag_name],)]
            gain = float(100*(reference.mean()-values.mean())/reference.mean())
            assert row['gain_vs_'+baseline+'_percent'] == gain
            low,high = row['gain_vs_'+baseline+'_CI95']; assert low <= high
    assert seen_rows == set(expected)
    with (OUT/'all_timings.csv').open(newline='') as stream:
        tabular = list(csv.DictReader(stream))
    assert len(tabular) == len(analysis['rows'])
    for actual,row in zip(tabular,analysis['rows']):
        encoded = {name:json.dumps(value,separators=(',',':')) if isinstance(value,(dict,list))
                   else '' if value is None else str(value) for name,value in row.items()}
        assert actual == encoded
    history = c.read(OUT/'historical_gain_directions.json')
    assert history['status'] == 'OK' and history['matched_conditions'] == len(history['rows']) == 517
    for source in history['historical_sources']:
        assert c.sha((c.ROOT/source['path']).read_bytes()) == source['SHA256']
    return dict(full_output_checks=checks, logical_full_output_checks=logical_checks,
                logical_timing_conditions=len(expected), logical_conditions_by_panel=EXPECTED_PANELS,
                raw_samples=timing['raw_samples'], physical_worker_cells=timing['physical_worker_cells'],
                worker_counts=timing['worker_counts'], environments=environments)


def profile_checks(protocol, gate, metas):
    from .g8_counters import EXTRA_FAMILIES, FAMILIES, opcode, static_pc_map
    original = c.read(OUT/'profile_protocol.json')
    collections = []
    for filename,directory,role,count in [
        ('profile_protocol.json','profiles','preregistered_remeasurement',167),
        ('profile_protocol_supplement.json','profiles_supplement','post_timing_exploratory_diagnostic',6)]:
        path = OUT/filename; profile = c.read(path); c.verify_sources(profile)
        assert profile['parent_protocol_SHA256'] == c.sha(c.encode(protocol))
        assert profile['compile_gate_SHA256'] == c.sha(gate_path().read_bytes())
        assert profile['payload_SHA256'] == gate['payload_SHA256']
        if role == 'post_timing_exploratory_diagnostic':
            assert profile['original_profile_protocol_SHA256'] == c.sha(c.encode(original))
            assert profile['evidence_role'] == role and not profile['confirmatory']
            assert not profile['formal_timing_or_selector_changes']
            for name,digest in {**profile['post_timing_source_SHA256'],**profile['selection_evidence_SHA256']}.items():
                assert c.sha((c.ROOT/name).read_bytes()) == digest,name
        planned = sum(len(group['records'])*len(group['modes']) for group in profile['profile_groups'])
        assert planned == count
        collections.append(dict(path=path,directory=OUT/directory,evidence_role=role,
                                SHA256=c.sha(c.encode(profile)),protocol=profile,report_count=count))
    counter = c.read(OUT/'counter_analysis.json')
    assert counter['status'] == 'OK' and counter['CPU_only'] and counter['GPU_allocations'] == 0
    assert counter['protocol_SHA256'] == c.sha(c.encode(original)) and counter['payload_SHA256'] == gate['payload_SHA256']
    summaries = [dict(path=str(item['path'].relative_to(c.ROOT)),directory=str(item['directory'].relative_to(c.ROOT)),
                      evidence_role=item['evidence_role'],SHA256=item['SHA256'],report_count=item['report_count']) for item in collections]
    assert counter['collection_protocols'] == summaries
    counter_rows = {(row['group'],row['tag'],row['mode']):row for row in counter['rows']}
    assert len(counter_rows) == len(counter['rows']) == counter['report_count'] == 173
    report_provenance = {(row['group'],row['tag'],row['mode']):row for row in counter['reports']}
    assert len(report_provenance) == len(counter['reports']) == len(counter_rows)
    imports = {row['report_SHA256']:row for row in counter['local_reimports']}
    assert len(counter['local_reimports']) == len(counter_rows)
    observed = set(); selected_aliases = []; alias_groups_seen = set(); software_count = 0
    cases = {group['id']:group['case'] for group in protocol['inference_groups']}
    cases.update({case['id']:case for case in protocol['training_cases']})
    requested = set()
    for selection in protocol['profile_selectors']:
        for name in selection['names']:
            tag = selection['group']+':'+name
            if tag not in metas:
                # Some case layouts are declared identical before compilation
                # (for example vec2:within at BN=256).  The builder selects
                # their canonical metadata directly, rather than inventing a
                # second compiled program or a second hardware observation.
                variants = cases[selection['group']]['layouts']
                equivalent = next(row for row in variants if name in row.get('aliases',[]))
                tag = selection['group']+':'+equivalent['name']
                assert metas[tag]['requested_input'] == {'kind':'blocked','fields':equivalent['layout']}
            for mode in selection['modes']: requested.add((tag,mode))
    supplement = collections[1]['protocol']
    requested |= {(selection['group']+':'+name,mode) for selection in supplement['profile_selectors']
                  for name in selection['names'] for mode in selection['modes']}
    covered = set()
    for collection,group in [(item,group) for item in collections for group in item['protocol']['profile_groups']]:
        profile = collection['protocol']
        target = collection['directory']/group['id']/'attempt_001'; exact_export(target)
        root = target/'export'; raw = c.read(root/'result.json')
        assert raw['status'] == 'OK' and raw['group'] == group['id'] and not raw['formal_timing']
        assert raw['protocol_SHA256'] == c.sha(c.encode(profile)) and raw['payload_SHA256'] == gate['payload_SHA256']
        selection = {row['metadata']['tag']:row for row in group['records']}
        expected = {(tag,mode) for tag in selection for mode in group['modes']}
        actual = [(row['tag'],row['mode']) for row in raw['reports']]
        assert len(actual) == len(set(actual)) == len(expected) and set(actual) == expected
        for report in raw['reports']:
            key = (group['id'],report['tag'],report['mode']); assert key not in observed; observed.add(key)
            meta = metas[report['tag']]; chosen = selection[report['tag']]
            assert chosen['metadata'] == meta and report['aliases'] == chosen['aliases']
            folder = root/Path(report['path']).parent
            assert c.sha((root/report['path']).read_bytes()) == report['SHA256']
            binding_path = root/report['binding_path']
            assert c.sha(binding_path.read_bytes()) == report['binding_SHA256']
            binding = c.read(binding_path)
            assert not binding['Triton_imported'] and binding['compile_calls'] == 0 and binding['driver_return_codes'] == [0]
            assert binding['environment']['cc'] == [9,0] and 'H100' in binding['environment']['name']
            assert len(binding['records']) == 1
            record = binding['records'][0]
            assert record['tag'] == report['tag'] and record['correctness']['pass']
            assert record['binding']['loaded_SHA256'] == record['binding']['cubin_SHA256'] == meta['cubin_SHA256']
            for filename in ['collect.json','import.json']+(['source.json'] if report['mode'] == 'software' else []):
                assert c.read(folder/filename)['returncode'] == 0
            collected = c.read(folder/'collect.json')
            assert 'PROFILE_BINDING ' in collected['output']
            flags = collected['argv']
            replay,cache = ('application','none') if report['mode'] == 'warm_application' else ('kernel','all')
            for flag,value in [('--replay-mode',replay),('--cache-control',cache),('--clock-control','none'),('--profile-from-start','off')]:
                assert flags[flags.index(flag)+1] == value
            if report['mode'] == 'software':
                assert flags[flags.index('--section')+1] == 'InstructionStats'
            else:
                assert flags[flags.index('--metrics')+1] == ','.join(profile['profile_metrics'])
            job = c.read(folder/'job.json')
            assert job['metadata'] == [meta] and job['case'] == group['case'] and job['mode'] == report['mode']
            covered.add((meta['tag'],report['mode']))
            if collection['evidence_role'] == 'post_timing_exploratory_diagnostic':
                assert group['id'] == 'diagnostic_'+meta['group']
                covered.add((group['id']+':'+meta['variant']['name'],report['mode']))
            row = counter_rows[key]; assert row['SHA256'] == report['SHA256'] and row['binding'] == binding
            assert row['evidence_role'] == collection['evidence_role']
            assert row['collection_protocol_SHA256'] == collection['SHA256']
            assert row['collection_protocol_path'] == str(collection['path'].relative_to(c.ROOT))
            provenance = report_provenance[key]
            assert provenance['SHA256'] == report['SHA256'] and provenance['evidence_role'] == collection['evidence_role']
            assert provenance['collection_protocol_SHA256'] == collection['SHA256']
            imported = imports[report['SHA256']]; assert imported['values_identical']
            import_path = c.ROOT/imported['path']; assert c.sha(import_path.read_bytes()) == imported['SHA256']
            local = c.read(import_path)
            assert local['CPU_only'] and local['GPU_allocations'] == 0 and local['report_SHA256'] == report['SHA256']
            assert all(cmd['returncode'] == 0 for cmd in local['commands'])
            assert all('--import' in cmd['argv'] for cmd in local['commands'])
            values = hardware_rows(c.read(folder/'import.json')['output'])
            assert len(values) == 1 and values == hardware_rows(local['commands'][0]['output']) and row['metrics'] == values[0]
            ctas = 1
            for extent in record['grid']: ctas *= int(extent)
            assert ctas > 0 and row['CTAs'] == ctas
            assert row['metrics_per_CTA'] == {name:value/ctas for name,value in values[0].items() if name.endswith('.sum')}
            if report['mode'] == 'software':
                software_count += 1
                assert imported['source_identical'] and row['all_executed_PC_sites_mapped']
                assert source_blocks(c.read(folder/'source.json')['output']) == source_blocks(local['commands'][1]['output'])
                source = row['source_correlation']; source_path = c.ROOT/source['path']
                assert c.sha(source_path.read_bytes()) == source['SHA256']
                disassembly = c.read(source_path)
                assert disassembly['CPU_only'] and disassembly['CUBIN_SHA256'] == meta['cubin_SHA256'] and disassembly['returncode'] == 0
                assert '-gi' in disassembly['argv'] and '-c' in disassembly['argv']
                static = static_pc_map(disassembly['output'],meta['function_name'])
                dynamic = source_blocks(c.read(folder/'source.json')['output']); assert len(dynamic) == 1
                assert set(row['per_PC']) == set(dynamic[0]) and row['per_PC']
                counts = {name:0 for name in [*FAMILIES,*EXTRA_FAMILIES]}; opcodes = Counter()
                lines = defaultdict(lambda:dict(dynamic_instructions=0,families=Counter(),opcodes=Counter()))
                for pc,item in dynamic[0].items():
                    assert pc in static and opcode(item['instruction']) == opcode(static[pc]['instruction'])
                    families = list(item['families'])+[name for name,pattern in EXTRA_FAMILIES.items() if re.search(pattern,item['instruction'])]
                    assert row['per_PC'][pc] == {**item,'families':families,'source':static[pc]}
                    instruction = opcode(item['instruction']); opcodes[instruction] += item['count']
                    summary = lines[static[pc]['file']+':'+str(static[pc]['line'])]
                    summary['dynamic_instructions'] += item['count']; summary['opcodes'][instruction] += item['count']
                    for family in families:
                        counts[family] += item['count']; summary['families'][family] += item['count']
                assert row['opcode_counts'] == dict(opcodes) and row['per_source_line'] == dict(lines)
                assert row['counts'] == {'CTAs':ctas,**counts}
                assert row['counts_per_CTA'] == {name:value/ctas for name,value in counts.items()}
            for audit in row['alias_audit']:
                assert audit['CUBIN_SHA256'] == meta['cubin_SHA256']
                assert all(audit[name] for name in ['full_CUBIN_equal','allocated_sections_equal',
                           'runtime_names_and_parameter_types_equal','launch_resources_equal'])
                assert set(audit['requested_modes']) <= set(group['modes'])
                if key[:2] not in alias_groups_seen:
                    selected_aliases.append(audit)
                if report['mode'] in audit['requested_modes']:
                    covered.add((audit['alias'],report['mode']))
            alias_groups_seen.add(key[:2])
    assert observed == set(counter_rows)
    assert requested <= covered, requested-covered
    assert selected_aliases == counter['alias_audit'] or Counter(c.encode(row) for row in selected_aliases) == Counter(c.encode(row) for row in counter['alias_audit'])
    return dict(NCU_reports=len(observed),preregistered_NCU_reports=167,post_timing_exploratory_NCU_reports=6,
                software_reports_with_current_CUBIN_PC_mapping=software_count,
                local_report_reimports=len(counter['local_reimports']), profiled_aliases=len(counter['alias_audit']))


def supplemental_actual_abi(gate, metas):
    """Check the separately frozen exact batch64 specialization audit."""
    from .g8_actualabi import AUDIT, program
    from .g8_actualabi_verify import verify
    verification = verify()
    assert c.read(AUDIT/'verification.json') == verification
    protocol = c.read(AUDIT/'preregistration/protocol.json'); c.verify_sources(protocol)
    with zipfile.ZipFile(AUDIT/'preregistration/source_snapshot.zip') as archive:
        for name,digest in protocol['worker_source_SHA256'].items():
            assert c.sha(archive.read(name)) == digest
    for name,digest in protocol['protected_SHA256'].items():
        assert c.sha((c.ROOT/name).read_bytes()) == digest,name
    assert protocol['parent_compile_gate_SHA256'] == c.sha(c.encode(gate))
    exact_export(AUDIT/'cpu/attempt_001')
    root = AUDIT/'cpu/attempt_001/export'; raw = c.read(root/'result.json')
    analysis = c.read(AUDIT/'analysis.json')
    assert raw['status'] == analysis['status'] == 'OK'
    assert raw['GPU_kernel_launches'] == raw['native_rebuilds'] == 0 and not raw['GPU_allocated']
    assert not raw['compiler_intervention'] and raw['native_SHA256'] == c.NATIVE
    assert raw['ccache_before'] == raw['ccache_after']
    assert analysis['CPU_only'] and analysis['GPU_allocations'] == analysis['GPU_kernel_launches'] == 0
    assert analysis['reuse_G8_evidence_supported'] and analysis['compiled_Gluon_programs'] == 4
    assert raw['protocol_SHA256'] == analysis['protocol_SHA256'] == c.sha(c.encode(protocol))
    assert analysis['raw_result_SHA256'] == c.sha((root/'result.json').read_bytes())
    assert len(raw['records']) == len(protocol['jobs']) == len(analysis['records']) == 4
    original_root = c.ROOT/gate['CPU_export_directory']
    checked = set()
    for job,record,row in zip(protocol['jobs'],raw['records'],analysis['records']):
        tag = job['tag']; assert record['tag'] == row['tag'] == tag and tag not in checked; checked.add(tag)
        assert job['expected_G8_metadata'] == metas[tag] and record['status'] == 'OK'
        assert record['metadata']['JIT_attributes'] == job['exact_attributes']
        assert record['metadata']['JIT_signature'] == metas[tag]['JIT_signature']
        assert record['metadata']['JIT_constants'] == metas[tag]['JIT_constants']
        actual = program(root,record['metadata']); generic = program(original_root,metas[tag])
        fields = {name:actual[name] == generic[name] for name in actual if name != 'CUBIN_SHA256'}
        assert all(fields.values()) and row['checks'] == fields and row['all_execution_fields_equal']
        assert row['exact_batch64'] == actual and row['generic_G8'] == generic
        assert row['full_CUBIN_identical'] == (actual['CUBIN_SHA256'] == generic['CUBIN_SHA256'])
        if job['variant']['name'].endswith('gluon_anchor'):
            frozen = job['G7_actual_G0']; prior = program(c.ROOT/frozen['real_JIT_export_root'],frozen['real_JIT_metadata'])
            assert all(actual[name] == prior[name] for name in fields)
            assert row['G7_real_JIT_G0_execution_fields_equal']
    assert len(protocol['reference_pairs']) == len(analysis['upstream_reference_pairs']) == 2
    for pair,row in zip(protocol['reference_pairs'],analysis['upstream_reference_pairs']):
        generic = program(original_root,pair['G8_upstream']); real = pair['G7_actual_U']
        actual = program(c.ROOT/real['real_JIT_export_root'],real['real_JIT_metadata'])
        assert pair['G8_upstream'] == metas[pair['group']+':actual:upstream']
        assert all(actual[name] == generic[name] for name in actual if name != 'CUBIN_SHA256')
        assert row['all_execution_fields_equal'] and row['generic_G8'] == generic and row['G7_real_JIT_actual'] == actual
    return dict(supplemental_exact_batch64_Gluon_programs=4,exact_batch64_upstream_reference_pairs=2,
                actual_attribute_execution_equivalence=True,actual_attribute_terminal_failure_recovered=True)


def validate(write=False, preflight=False):
    protocol, gate, metas, result = frozen_inputs()
    if preflight:
        print(json.dumps(dict(status='OK',stage='G8_CPU_preflight',**result),indent=2)); return
    result.update(runtime_checks(protocol,gate,metas))
    result.update(profile_checks(protocol,gate,metas))
    result.update(supplemental_actual_abi(gate,metas))
    for name in ['IMPACT_AUDIT.md','impact_audit.json','REPORT.md','REPRODUCE.md','all_timings.csv']:
        assert (OUT/name).is_file() and (OUT/name).stat().st_size > 0, name
    inventory = {str(path.relative_to(OUT)):c.sha(path.read_bytes()) for path in sorted(OUT.rglob('*'))
                 if path.is_file() and path.name != 'final_validation.json'}
    result = dict(status='OK',stage='G8',**result,CPU_only_validator=True,GPU_allocations=0,
        source_SHA256={str(path.relative_to(c.ROOT)):c.sha(path.read_bytes()) for path in sorted(c.SOURCE.glob('g8_*.py'))},
        REPORT_SHA256=c.sha((OUT/'REPORT.md').read_bytes()),evidence_files=len(inventory),
        evidence_inventory_SHA256=c.sha(c.encode(inventory)))
    final = OUT/'final_validation.json'
    if write: c.write_once(final,result)
    if final.exists(): assert c.read(final) == result
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--write',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    args = parser.parse_args(); assert not (args.write and args.preflight)
    validate(args.write,args.preflight)
