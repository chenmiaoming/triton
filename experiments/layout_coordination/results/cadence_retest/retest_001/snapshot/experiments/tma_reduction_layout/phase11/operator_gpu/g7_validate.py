"""Offline structural validation of frozen binding and complete GPU evidence."""
import argparse
import json
import zipfile
from pathlib import Path
from . import common as c
from .g5_analyze import elf
from .g7_prepare import OUT


def validate(write=False):
    protocols = {}
    for suffix in ['', '_v2']:
        p = c.read(OUT/f'preregistration/protocol{suffix}.json'); protocols[suffix] = p
        with zipfile.ZipFile(OUT/f'preregistration/source_snapshot{suffix}.zip') as archive:
            for name,digest in p['worker_source_SHA256'].items():
                assert c.sha(archive.read(name)) == digest
        for name,digest in {**p['protected_SHA256'], **p.get('prior_attempt_SHA256', {})}.items():
            assert c.sha((c.ROOT/name).read_bytes()) == digest, name
    p = protocols['_v2']; c.verify_sources(p)
    gate = c.read(OUT/'compile_gate_v2.json'); payload = (OUT/'compiled_payload.zip').read_bytes()
    assert gate['status'] == 'OK' and gate['protocol_SHA256'] == c.sha(c.encode(p))
    assert gate['payload_SHA256'] == c.sha(payload) and gate['compiled_new_programs'] == 0
    metas = {r['tag']:r for r in gate['records']}; assert len(metas) == len(gate['records']) == 48
    expected_artifact_tags = {g['id']+':'+role for g in p['inference_groups'] for role in ['U','G0']}
    expected_artifact_tags |= {g['id']+':stage2' for g in p['inference_groups'] if g['case']['family'] == 'C02'}
    expected_artifact_tags |= {g['id']+':'+role for g in p['training_cases'] for role in ['U','G0','dx']}
    assert set(metas) == expected_artifact_tags
    assert {r['form_id'] for r in gate['records'] if r['pair_role'] in ['U','G0']} == {
        item['case']['id'] for item in p['binding_items']}
    with zipfile.ZipFile(OUT/'compiled_payload.zip') as archive:
        for meta in metas.values():
            for name,digest in meta['artifact_SHA256'].items():
                assert c.sha(archive.read(meta['artifact_directory']+'/'+name)) == digest
        for group in p['inference_groups'] + p['training_cases']:
            a,b = [metas[group['id']+':'+role] for role in ['U','G0']]
            assert elf(archive.read(a['artifact_directory']+'/kernel.cubin')) == elf(archive.read(b['artifact_directory']+'/kernel.cubin'))
            assert all(a[k] == b[k] for k in ['num_warps','dynamic_smem_bytes','runtime_argument_names','global_scratch_size'])

    raw_bindings = {}
    for i,suffix in [(1,''),(2,'_v2')]:
        target = OUT/'binding'/f'attempt_{i:03}'
        with zipfile.ZipFile(target/'raw_export.zip') as archive:
            names = {n for n in archive.namelist() if not n.endswith('/')}
            assert names == {str(f.relative_to(target/'export')) for f in (target/'export').rglob('*') if f.is_file()}
            for name in names: assert archive.read(name) == (target/'export'/name).read_bytes()
        raw = c.read(target/'export/returned_result.json')
        assert raw['protocol_SHA256'] == c.sha(c.encode(protocols[suffix])) and raw['payload_SHA256'] == c.sha(payload)
        assert raw['ccache_before'] == raw['ccache_after']
        assert raw['native_SHA256'] == raw['native_SHA256_after'] == c.NATIVE
        assert raw['backend_SHA256'] == raw['backend_SHA256_after'] == p['backend_SHA256']
        assert raw['environment']['cc'] == [9,0] and 'H100' in raw['environment']['name']
        assert len(raw['records']) == (26 if i == 1 else 8)
        assert raw['experiment_kernel_launches'] == 0 and raw['timing_samples'] == 0 and raw['native_rebuilds'] == 0
        for row in raw['records']:
            if 'real_JIT_metadata' in row: raw_bindings[(i,row['tag'])] = row
    binding = c.read(OUT/'binding_gate.json'); assert binding['status'] == 'OK' and len(binding['records']) == 26
    assert binding['protocol_SHA256'] == c.sha(c.encode(p))
    expected_binding_tags = {g['id']+':'+s for g in p['binding_groups']+p['training_cases'] for s in ['U','G0']}
    assert len({r['tag'] for r in binding['records']}) == 26
    assert {r['tag'] for r in binding['records']} == expected_binding_tags
    for row in binding['records']:
        raw = raw_bindings[(row['attempt'],row['tag'])]; actual = raw['real_JIT_metadata']; frozen = metas[row['tag']]
        assert actual == row['real_JIT_metadata']
        assert all(actual[k] == frozen[k] for k in row['checks']) and all(row['checks'].values())
        aa = {k:v for k,v in actual['JIT_attributes'].items() if v}
        bb = {k:v for k,v in frozen['JIT_attributes'].items() if v}
        if aa != bb:
            assert frozen['family'] == 'C06' and raw['batch'] == 64
            assert aa == {**bb,'(5,)':[['tt.divisibility',16]]} and raw['batch'] % 16 == 0
        for name,digest in actual['artifact_SHA256'].items():
            assert c.sha((c.ROOT/row['real_JIT_export_root']/actual['artifact_directory']/name).read_bytes()) == digest

    samples = 0; oracle_checks = 0; pair_checks = 0; partial_checks = 0
    expected_conditions = sum(2 if g['case']['family'] == 'C02' else 1 for g in p['inference_groups'])
    expected_conditions += sum(1+len(g['pipeline_M']) for g in p['training_cases'])
    assert expected_conditions == 32
    for invocation in range(4):
        path = OUT/('correctness/attempt_001/raw.json' if invocation == 0 else f'timing/invocation_{invocation}/raw.json')
        raw = c.read(path)
        assert raw['status'] == 'OK' and raw['invocation'] == invocation
        assert raw['protocol_SHA256'] == c.sha(c.encode(p)) and raw['payload_SHA256'] == c.sha(payload)
        for family in ['inference','training']:
            part = raw[family]
            assert part['status'] == 'OK' and part['driver_return_codes'] == [0] and not part['Triton_imported']
            assert part['compile_calls'] == 0 and part['protocol_SHA256'] == c.sha(c.encode(p))
            assert part['payload_SHA256'] == c.sha(payload)
            assert part['environment']['cc'] == [9,0] and 'H100' in part['environment']['name']
            assert all(x['pass'] for x in part['checks']); oracle_checks += len(part['checks'])
            patterns = ['finite'] if invocation else p['correctness' if family == 'inference' else 'training_correctness']['patterns']
            if family == 'inference':
                expected_checks = {(g['id'],w['batch'],w['length'],pattern,g['id']+':'+role)
                    for g in p['inference_groups'] for w in g['workloads'] for pattern in patterns for role in ['U','G0']}
                observed_checks = [(r['group'],r['batch'],r['length'],r['pattern'],r['tag']) for r in part['checks']]
                expected_pairs = {x[:4] for x in expected_checks}
                observed_pairs = [(r['group'],r['batch'],r['length'],r['pattern']) for r in part['bitwise_comparisons']]
            else:
                conditions = [(g,mode,m) for g in p['training_cases']
                    for mode,m in [('merge',128)]+[('pipeline',m) for m in g['pipeline_M']]]
                expected_checks = {(g['id'],mode,m,pattern,g['id']+':'+role)
                    for g,mode,m in conditions for pattern in patterns for role in ['U','G0']}
                observed_checks = [(r['geometry'],r['mode'],r['M'],r['pattern'],r['tag']) for r in part['checks']]
                expected_pairs = {x[:4] for x in expected_checks}
                observed_pairs = [(r['geometry'],r['mode'],r['M'],r['pattern']) for r in part['bitwise_comparisons']]
                expected_partials = {(g['id'],'same_partials_merge',m,pattern) for g in p['training_cases']
                    for m in g['pipeline_M'] for pattern in patterns} if invocation == 0 else set()
                observed_partials = [(r['geometry'],r['mode'],r['M'],r['pattern']) for r in part['same_partial_comparisons']]
                assert len(observed_partials) == len(set(observed_partials)) == len(expected_partials)
                assert set(observed_partials) == expected_partials
            assert len(observed_checks) == len(set(observed_checks)) == len(expected_checks)
            assert set(observed_checks) == expected_checks
            assert len(observed_pairs) == len(set(observed_pairs)) == len(expected_pairs)
            assert set(observed_pairs) == expected_pairs
            for tag,b in part['bindings'].items():
                assert b['loaded_SHA256'] == b['cubin_SHA256'] == metas[tag]['cubin_SHA256']
                assert b['guards'] > 0
            groups = p['inference_groups'] if family == 'inference' else p['training_cases']
            for group in groups:
                a,b = [part['bindings'][group['id']+':'+role] for role in ['U','G0']]
                assert a['ABI'] == b['ABI'] and a['attributes'] == b['attributes']
            comparisons = part['bitwise_comparisons']
            for comp in comparisons:
                if family == 'inference' or comp['required_bitwise_equal']:
                    assert comp['bitwise_equal']; pair_checks += 1
            if family == 'training':
                for comp in part['same_partial_comparisons']:
                    assert comp['bitwise_equal'] and comp['partials_and_lock_unchanged']
                    assert all(x['pass'] for x in comp['checks']); partial_checks += 1
            if invocation == 0: continue
            expected_cells = (46 if family == 'inference' else 18)
            assert len(part['cells']) == expected_cells
            for cell in part['cells']:
                assert not cell['aliases'] and cell['complete_output_check']['pass']
                assert len(cell['visits']) == p['timing']['visits']
                for visit in cell['visits']:
                    assert len(visit['samples_us']) == p['timing']['samples']
                    assert all(0 < x < 1.e6 for x in visit['samples_us'])
                    expected_nodes = [0,0,7,7,2,2,2] if cell['mode'] == 'pipeline' else [0,0,7,7] if cell['mode'] == 'decode' else [0,7,7]
                    assert sorted(visit['node_types']) == sorted(expected_nodes)
                    samples += len(visit['samples_us'])
    assert oracle_checks == 300 and partial_checks == 18 and pair_checks == 114
    assert samples == 32*2*3*5*31 == 29760
    analysis = c.read(OUT/'timing_analysis.json')
    # Statistical results are reported, never required to say "equivalent".
    assert analysis['raw_samples'] == samples and len(analysis['rows']) == 32
    assert (OUT/'REPORT.md').exists() and (OUT/'REPRODUCE.md').exists()
    inventory = {str(path.relative_to(OUT)):c.sha(path.read_bytes()) for path in sorted(OUT.rglob('*'))
                 if path.is_file() and path.name != 'final_validation.json'}
    result = dict(status='OK', stage='G7', actual_tensor_JIT_bindings=26,
        completed_missing_SGLang_bindings=8, reused_real_JIT_bindings=18,
        full_output_oracle_checks=oracle_checks, required_bitwise_pairs=pair_checks,
        same_partial_bitwise_checks=partial_checks, timing_raw_samples=samples, paired_timing_conditions=32,
        native_rebuilds=0, prior_files_unchanged=len(p['protected_SHA256']),
        source_SHA256={str(path.relative_to(c.ROOT)):c.sha(path.read_bytes()) for path in sorted(c.SOURCE.glob('g7_*.py'))},
        REPORT_SHA256=c.sha((OUT/'REPORT.md').read_bytes()), evidence_files=len(inventory),
        evidence_inventory_SHA256=c.sha(c.encode(inventory)))
    final = OUT/'final_validation.json'
    if write: c.write_once(final,result)
    if final.exists(): assert c.read(final) == result
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--write',action='store_true')
    validate(parser.parse_args().write)
