"""CPU-only final provenance, raw-data, ABI, counter and freeze validator."""
import ast
import csv
import io
import re
import zipfile
from . import common as c
from .analyze import timing_analysis, counter_analysis
from .bind_sources import functions
from .layout_parsing import plan
from .make_controlled import RemoveInterventions
from .offline_import import one as validate_offline_import
from .production_cases import configuration
from .validate_g1 import evaluate as validate_g1


ROOT = c.OUT / 'stage_g2'


def manifest(path):
    value = c.read(path)
    for name, digest in value['files_SHA256'].items():
        assert c.sha((c.ROOT / name).read_bytes()) == digest, name
    return len(value['files_SHA256'])


def source_parity():
    bindings = c.read(c.SOURCE / 'upstream/bindings.json')
    pairs = [('C06', 'vllm', ('cdiv_fn', 'find_seq_idx', 'reduce_segments')),
             ('C02', 'sglang', ('tanh', '_fwd_kernel_stage1'))]
    for family, module, names in pairs:
        original = (c.SOURCE / 'upstream' / (module + '.py.txt')).read_text()
        assert c.sha(original.encode()) == bindings['source_SHA256'][module]
        extracted = (c.SOURCE / (module + '_triton.py')).read_text()
        direct_names = names if family == 'C02' else ('reduce_segments',)
        assert functions(original, direct_names) == functions(extracted, direct_names)
        if family == 'C06':
            helper = (c.SOURCE / 'upstream/vllm_helpers.py.txt').read_text()
            assert c.sha(helper.encode()) == bindings['helper_SHA256']
            assert functions(helper, names[:2]) == functions(extracted, names[:2])
        else:
            assert functions(original, ('_fwd_kernel_stage2',)) == functions(extracted, ('_fwd_kernel_stage2',))
        def normalized(text):
            return {node.name: ast.dump(RemoveInterventions().visit(node), include_attributes=False)
                    for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
        controlled = (c.SOURCE / (module + '_controlled.py')).read_text()
        assert normalized(functions(extracted, names)) == normalized(functions(controlled, names)), family


def snapshots():
    names = [('baseline_protocol.json', 'baseline_source'),
             ('controlled_protocol.json', 'controlled_source'),
             ('protocol.json', 'source'),
             ('profile_protocol.json', 'profile_source'),
             ('profile_protocol_v2.json', 'profile_source_v2'),
             ('profile_protocol_v3.json', 'profile_source_v3')]
    for name, folder in names:
        protocol = c.read(ROOT / 'preregistration' / name)
        for filename, digest in protocol['worker_source_SHA256'].items():
            assert c.sha((ROOT / 'preregistration' / folder / filename).read_bytes()) == digest, (name, filename)
    c.verify_sources(c.read(ROOT / 'preregistration/protocol.json'))
    c.verify_sources(c.read(ROOT / 'preregistration/profile_protocol_v3.json'))
    main_hash = c.sha((ROOT / 'preregistration/protocol.json').read_bytes())
    previous = None
    for version in ('', '_v2', '_v3'):
        path = ROOT / ('preregistration/profile_protocol' + version + '.json')
        p = c.read(path)
        assert p['main_protocol_SHA256'] == main_hash
        assert not p['selection_changed_by_timing'] and not p.get('timing_repeated', False)
        if previous:
            assert p['previous_profile_protocol_SHA256'] == previous
        previous = c.sha(path.read_bytes())
    for name in ('ncu_cli_local_check.json', 'ncu_cli_image_check.json'):
        check = c.read(ROOT / 'preregistration' / name)
        assert check['returncode'] == 0 and check['CPU_only']
        required = c.read(ROOT / 'preregistration/protocol.json')['profiles']['metrics']
        assert all(metric in check['output'] for metric in required)


def artifacts():
    protocol = c.read(ROOT / 'preregistration/protocol.json')
    gate = c.read(ROOT / 'compile_gate.json')
    origin = c.ROOT / gate['artifact_directory']
    bycase = {case['id']: case for case in protocol['geometries']}
    assert len(gate['records']) == 56 and gate['unique_CUBINs'] == 50
    assert len({row['cubin_SHA256'] for row in gate['records']}) == 50
    for stage, expected in (('baseline_cpu', 9), ('controlled_cpu', 47), ('bind', 56)):
        raw = c.read(ROOT / stage / 'attempt_001/export/result.json')
        assert raw['status'] == 'OK' and len(raw['records']) == expected
        assert raw['native_rebuilds'] == 0 and raw['native_SHA256'] == c.NATIVE
        assert raw['ccache_before'] == raw['ccache_after']
        if 'native_SHA256_after' in raw:
            assert raw['native_SHA256_after'] == c.NATIVE
        for row in raw['records']:
            meta = row.get('metadata', row)
            directory = meta['tag'].replace(':', '/') if stage == 'baseline_cpu' else meta['artifact_directory']
            folder = ROOT / stage / 'attempt_001/export' / directory
            for name, digest in meta['artifact_SHA256'].items():
                assert c.sha((folder / name).read_bytes()) == digest, (stage, meta['tag'], name)
    with zipfile.ZipFile(ROOT / 'compiled_payload.zip') as payload:
        assert payload.testzip() is None
        for row in gate['records']:
            folder = origin / row['artifact_directory']
            for name, digest in row['artifact_SHA256'].items():
                assert c.sha((folder / name).read_bytes()) == digest
                assert c.sha(payload.read(row['artifact_directory'] + '/' + name)) == digest
            module = 'vllm' if row['family'] == 'C06' else 'sglang'
            suffix = '_triton.py' if row['tag'].endswith(':upstream') or row['stage2'] else '_controlled.py'
            tree = ast.parse((c.SOURCE / (module + suffix)).read_text())
            fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == row['function_name'])
            expected = [arg.arg for index, arg in enumerate(fn.args.args) if str((index,)) not in row['JIT_constants']]
            assert row['runtime_argument_names'] == expected, row['tag']
            ptx = (folder / 'kernel.ptx').read_text()
            entry = re.search(r'\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid', ptx, re.S)
            abi = re.findall(r'\.param\s+\.(\w+)', entry[1])
            assert len(abi) == len(expected) + 2
            constants, pointers, scalars, _ = configuration(bycase[row['geometry']], row['stage2'])
            assert set(expected) <= set(constants) | set(pointers) | set(scalars)
            audit = gate['static_audit'][row['tag']]
            assert audit['ABI'] == abi and audit['local_bytes'] == [0]
            if row['stage2']:
                continue
            actual = plan((folder / 'kernel.ttgir').read_text(), bycase[row['geometry']])
            assert actual == audit['encodings']
            baseline = gate['plans'][row['geometry']]
            if not row['tag'].endswith(':upstream'):
                assert actual['V'] == row['intervention_layout']
                for key in (('Q', 'K', 'output') if row['family'] == 'C02' else ('statistics', 'output')):
                    assert actual[key] == baseline[key], (row['tag'], key)
    return gate


def raw_timing(protocol, gate):
    payload_hash = c.sha((ROOT / 'compiled_payload.zip').read_bytes())
    bytag = {row['tag']: row for row in gate['records']}
    expected = {(geometry, work['batch'], work['length'], work['mode'], tag)
                for geometry, works in protocol['workloads'].items() for work in works
                for tag, row in bytag.items() if row['geometry'] == geometry and not row['stage2']}
    actual_workers = []
    uuids = []
    for worker in range(1, 4):
        raw = c.read(ROOT / f'timing/invocation_{worker}/raw.json')
        assert raw['status'] == 'OK' and raw['formal_timing'] and not raw['profiler_attached']
        assert not raw['Triton_imported'] and raw['driver_return_codes'] == [0]
        assert raw['payload_SHA256'] == payload_hash
        assert raw['environment']['cc'] == [9, 0]
        uuids.append(raw['environment']['gpu_uuid'])
        present = set()
        for cell in raw['cells']:
            assert cell['complete_output_check']['pass']
            for alias in [cell['tag']] + cell['aliases']:
                present.add((cell['geometry'], cell['batch'], cell['length'], cell['mode'], alias))
                assert bytag[alias]['cubin_SHA256'] == bytag[cell['tag']]['cubin_SHA256']
            for visit in cell['visits']:
                assert len(visit['samples_us']) == 31
        assert present == expected
        for tag, binding in raw['bindings'].items():
            assert binding['loaded_SHA256'] == binding['cubin_SHA256'] == bytag[tag]['cubin_SHA256']
            assert binding['attributes']['local_bytes'] == 0
        actual_workers.append(len(raw['cells']))
    assert len(set(uuids)) == 3 and actual_workers == [261] * 3
    recomputed, csv_bytes = timing_analysis(write=False)
    assert c.encode(recomputed) == (ROOT / 'timing_analysis.json').read_bytes()
    assert csv_bytes == (ROOT / 'all_timings.csv').read_bytes()
    assert recomputed['raw_samples'] == 121365 and recomputed['logical_conditions'] == 294
    return recomputed


def counters(gate):
    values = counter_analysis(write=False)
    assert c.encode(values) == (ROOT / 'counter_analysis.json').read_bytes()
    assert len(values['reports']) == 12 and len(values['dynamic']) == len(values['hardware']) == 30
    bytag = {row['tag']: row for row in gate['records']}
    origin = c.ROOT / gate['artifact_directory']
    for report in values['reports']:
        path = c.ROOT / report['path']
        job = c.read(path.parent / 'job.json')
        reader = list(csv.reader(io.StringIO(c.read(path.parent / 'import.json')['output'])))
        header = next(row for row in reader if 'ID' in row and 'Kernel Name' in row)
        launches = [dict(zip(header, row)) for row in reader if len(row) == len(header) and row[header.index('ID')].isdigit()]
        assert len(launches) == len(job['metadata'])
        for index, (launch, meta) in enumerate(zip(launches, job['metadata'])):
            assert launch['ID'] == str(index) and launch['Kernel Name'] == meta['function_name']
            assert [int(v) for v in re.findall(r'\d+', launch['Block Size'])] == [128, 1, 1]
            grid = [1, 1, 1] if report['mode'] == 'software' else [8, 32, 4] if meta['family'] == 'C02' else [8, 32, 1]
            assert [int(v) for v in re.findall(r'\d+', launch['Grid Size'])] == grid
            assert launch['CC'] == '9.0'
        validate_offline_import(report)
    for row in values['dynamic']:
        sass = (origin / bytag[row['tag']]['artifact_directory'] / 'kernel.sass').read_text()
        def opcode(instruction):
            tokens = instruction.split()
            return tokens[1] if tokens[0].startswith('@') else tokens[0]
        static = {int(pc, 16): opcode(inst) for pc, inst in re.findall(r'/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;', sass)}
        assert static
        for pc, instruction in row['per_PC'].items():
            assert static[int(pc, 16)] == opcode(instruction['instruction']), (row['tag'], pc)
    for case in c.read(ROOT / 'preregistration/protocol.json')['geometries']:
        first = ROOT / 'profiles' / case['id'] / 'attempt_001/export'
        failed = c.read(first / 'result.json')
        assert failed['status'] == 'FAILED' and failed['reports'] == []
        assert c.read(first / 'metric_query.json')['returncode'] != 0
        second = c.read(ROOT / 'profiles' / case['id'] / 'attempt_002/export/result.json')
        if case['family'] == 'C02':
            assert second['status'] == 'FAILED' and not second['reports']
            assert 'assignment destination is read-only' in second['error']
        else:
            assert second['status'] == 'OK'
    offline = c.read(ROOT / 'offline_import_validation.json')
    assert offline['status'] == 'OK' and offline['GPU_allocations'] == 0 and len(offline['reports']) == 12
    return values


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    prior_count = manifest(c.PREP / 'FREEZE.json') + manifest(c.OUT / 'stage_g1/FREEZE.json')
    assert validate_g1()['post_audit_passes'] == 234
    protocol = c.read(ROOT / 'preregistration/protocol.json')
    assert protocol['G1_freeze_SHA256'] == c.sha((c.OUT / 'stage_g1/FREEZE.json').read_bytes())
    source_parity()
    snapshots()
    gate = artifacts()
    raw = c.read(ROOT / 'correctness/attempt_001/raw.json')
    assert raw['status'] == 'OK' and len(raw['checks']) == 159 and all(row['pass'] for row in raw['checks'])
    assert not raw['Triton_imported'] and raw['driver_return_codes'] == [0]
    assert raw['protocol_SHA256'] == c.sha(c.encode(protocol))
    assert raw['payload_SHA256'] == c.sha((ROOT / 'compiled_payload.zip').read_bytes())
    timing = raw_timing(protocol, gate)
    counts = counters(gate)
    for filename in ('REPORT.md', 'MATRIX.md', 'SOURCE_BINDING_SCOPE.md', 'figures/layout_gains.png', 'figures/layout_gains.pdf'):
        assert (ROOT / filename).stat().st_size > 0
    result = {'status': 'OK', 'CPU_only_validator': True, 'prior_files_verified': prior_count,
              'semantic_fixture_audited_checks': 234, 'production_checks': 159,
              'logical_programs': 56, 'unique_CUBINs': 50, 'physical_conditions': 261,
              'logical_timing_conditions': 294, 'raw_samples': timing['raw_samples'],
              'NCU_reports': len(counts['reports']), 'NCU_local_reimports': 12,
              'timing_analysis_recomputed_exactly': True, 'source_AST_parity': True,
              'actual_ABI_and_NCU_PC_binding_verified': True, 'native_rebuilds': 0,
              'failed_attempts_preserved': True, 'full_lowering_bridge_proven': False,
              'whole_serving_package_measured': False, 'selector_trained': False}
    if args.freeze:
        c.write_once(ROOT / 'validation.json', result)
        files = [path for path in c.SOURCE.rglob('*') if path.is_file() and '__pycache__' not in path.parts]
        files += [path for path in ROOT.rglob('*') if path.is_file() and path.name != 'FREEZE.json']
        frozen = {path.relative_to(c.ROOT).as_posix(): c.sha(path.read_bytes()) for path in sorted(files)}
        c.write_once(ROOT / 'FREEZE.json', {'stage': 'G2', 'status': 'OK', 'files_SHA256': frozen,
                     'protocol_SHA256': c.sha(c.encode(protocol)), 'source_reproduction_scope_preserved': True})
    else:
        assert c.read(ROOT / 'validation.json') == result
    count = manifest(ROOT / 'FREEZE.json')
    print('G2 VALIDATOR OK:', count, 'frozen files;', prior_count, 'prior files unchanged;',
          '56 programs / 50 CUBINs; 159 checks; 121365 samples; 12 NCU reports/reimports')


if __name__ == '__main__':
    main()
