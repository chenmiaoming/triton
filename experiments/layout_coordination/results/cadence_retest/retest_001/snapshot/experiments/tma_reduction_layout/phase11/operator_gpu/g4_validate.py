"""Validate complete new evidence and all protected prior artifacts, without GPU work."""
from pathlib import Path
import argparse
import json
from . import common as c
from .g4_prepare import out, protocol_path, gate_path, payload_path


def validate(stage, write=False):
    target = out(stage); p = c.read(protocol_path(stage)); c.verify_sources(p)
    for name, digest in {**p['old_evidence_SHA256'], **p.get('prior_attempt_SHA256', {})}.items(): assert c.sha((c.ROOT / name).read_bytes()) == digest, name
    gate = c.read(gate_path(stage))
    assert gate['status'] == 'OK' and gate['protocol_SHA256'] == c.sha(c.encode(p))
    assert gate['payload_SHA256'] == c.sha(payload_path(stage).read_bytes())
    cpu = c.read(c.ROOT / gate['CPU_export_directory'] / 'result.json')
    assert cpu['status'] == 'OK' and not cpu['GPU_allocated'] and cpu['ccache_before'] == cpu['ccache_after']
    assert cpu['native_SHA256'] == c.NATIVE and cpu['native_rebuilds'] == 0
    for row in cpu['records']:
        root = c.ROOT / gate['CPU_export_directory'] / row['artifact_directory']
        for name, digest in row['artifact_SHA256'].items(): assert c.sha((root / name).read_bytes()) == digest
    bind = next(x for path in (target / 'bind').glob('attempt_*/raw.json') if (x := c.read(path))['status'] == 'OK' and x['protocol_SHA256'] == c.sha(c.encode(p)))
    expected_jit = sum(r['provenance'] == 'CPU_actual_ABI' for r in gate['records'])
    assert bind['status'] == 'OK' and len(bind['records']) == expected_jit and all(x['match'] for x in bind['records'])
    assert bind['ccache_before'] == bind['ccache_after'] and bind['native_rebuilds'] == 0
    raw = c.read(target / 'correctness/attempt_001/raw.json')
    assert raw['status'] == 'OK' and not raw['Triton_imported'] and all(x['pass'] for x in raw['checks'])
    assert raw['protocol_SHA256'] == c.sha(c.encode(p)) and raw['payload_SHA256'] == gate['payload_SHA256']
    expected_checks = sum(len(g['variants']) * (2 if stage == 'g4b' else 1) * len(g['workloads']) * 3 for g in p['groups'])
    assert len(raw['checks']) == expected_checks, (len(raw['checks']), expected_checks)
    bytag = {r['tag']: r for r in gate['records']}
    expected_patterns = {(g['id'], m['tag'], w['batch'], w['length'], pattern)
                         for g in p['groups'] for w in g['workloads'] for pattern in p['correctness']['patterns']
                         for m in gate['records'] if m['group'] == g['id'] and not m['stage2']}
    assert {(x['group'], x['tag'], x['batch'], x['length'], x['pattern']) for x in raw['checks']} == expected_patterns
    logical_expected = set()
    for group in p['groups']:
        for work in group['workloads']:
            for mode in (['stage1', 'decode'] if group['case']['family'] == 'C02' else ['merge']):
                for meta in gate['records']:
                    if meta['group'] == group['id'] and not meta['stage2']:
                        logical_expected.add((group['id'], work['batch'], work['length'], mode, meta['tag']))
    samples = 0
    for i in range(1, 4):
        timed = c.read(target / f'timing/invocation_{i}/raw.json')
        assert timed['status'] == 'OK' and timed['driver_return_codes'] == [0] and not timed['Triton_imported']
        assert timed['protocol_SHA256'] == c.sha(c.encode(p)) and timed['payload_SHA256'] == gate['payload_SHA256']
        assert all(x['pass'] for x in timed['checks'])
        observed = set()
        for cell in timed['cells']:
            assert cell['complete_output_check']['pass'] and len(cell['visits']) == p['timing']['visits']
            for visit in cell['visits']:
                assert len(visit['samples_us']) == p['timing']['samples']
                assert all(x > 0 and x < 1.e6 for x in visit['samples_us'])
                assert sorted(visit['node_types']) == [0] * (2 if cell['mode'] == 'decode' else 1) + [7, 7]
                samples += len(visit['samples_us'])
            for tag in [cell['tag']] + cell['aliases']:
                observed.add((cell['group'], cell['batch'], cell['length'], cell['mode'], tag))
                assert timed['bindings'][tag]['loaded_SHA256'] == timed['bindings'][tag]['cubin_SHA256'] == bytag[tag]['cubin_SHA256']
        assert observed == logical_expected, (len(observed), len(logical_expected))
    analysis = c.read(target / 'timing_analysis.json')
    assert analysis['raw_samples'] == samples and analysis['logical_conditions'] == len(logical_expected)
    assert len(analysis['rows']) == len(logical_expected)
    actual_plans = c.read(target/'effective_layout_audit.json')
    assert actual_plans['status'] == 'OK' and all(x['requested_plan_matches'] for x in actual_plans['rows'])
    assert len(actual_plans['rows']) == sum(r['provenance'] == 'CPU_actual_ABI' and not r['stage2'] for r in gate['records'])
    if stage == 'g4b':
        old = {r['tag']: r for r in c.read(c.OUT/'stage_g2/compile_gate.json')['records']}
        for group in p['groups']:
            if group['case']['family'] != 'C02': continue
            actual = bytag[group['id']+':actual:stage2']; before = old[group['case']['id']+':stage2']
            assert actual['cubin_SHA256'] == before['cubin_SHA256']
            assert actual['runtime_argument_names'] == before['runtime_argument_names']
    profiles = 0
    if stage == 'g4a':
        from .g4_runtime import identity
        counters = c.read(target / 'counter_analysis.json'); profiles = len(counters['rows'])
        expected_profiles = 2 * sum(len({identity(r) for r in gate['records'] if r['group'] == g['id'] and not r['stage2']}) for g in p['groups'])
        assert profiles == expected_profiles == len(counters['local_reimports'])
        for row in counters['rows']: assert c.sha((c.ROOT / row['report_path']).read_bytes()) == row['SHA256']
        assert all(x['values_identical'] for x in counters['local_reimports'])
        attribution = c.read(target/'source_attribution.json')
        assert attribution['status'] == 'OK' and len(attribution['rows']) == 8
        assert all(x['all_executed_PC_sites_mapped'] for x in attribution['rows'])
        for imported in counters['local_reimports']:
            assert c.sha((c.ROOT/imported['path']).read_bytes()) == imported['SHA256']
    assert (target / 'REPORT.md').is_file() and (target / 'code_audit_v2.json').is_file()
    inventory = {str(path.relative_to(target)):c.sha(path.read_bytes()) for path in sorted(target.rglob('*'))
                 if path.is_file() and path.name != 'final_validation.json'}
    result = {'status': 'OK', 'stage': stage, 'real_tensor_JIT_matches': expected_jit,
              'logical_correctness_checks': expected_checks, 'timing_raw_samples': samples,
              'timing_logical_conditions': len(logical_expected), 'NCU_reports_and_local_reimports': profiles,
              'prior_files_unchanged': len(p['old_evidence_SHA256']), 'native_rebuilds': 0,
              'protocol_SHA256': c.sha(c.encode(p)), 'REPORT_SHA256': c.sha((target / 'REPORT.md').read_bytes()),
              'validator_SHA256': c.sha(Path(__file__).read_bytes()),
              'evidence_files':len(inventory),'evidence_inventory_SHA256':c.sha(c.encode(inventory))}
    final = target / 'final_validation.json'
    if write: c.write_once(final, result)
    if final.exists(): assert c.read(final) == result
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=['g4a', 'g4b', 'all'], default='all')
    parser.add_argument('--write', action='store_true'); args = parser.parse_args()
    for stage in (['g4a', 'g4b'] if args.stage == 'all' else [args.stage]): validate(stage, args.write)
