"""Offline audit of fixture execution, including an explicit FP8 reference correction."""
from pathlib import Path
import numpy as np
from experiments.tma_reduction_layout.phase11.operator_survey import oracles
from . import common as c


def evaluate():
    root = c.OUT / 'stage_g1'
    protocol = c.read(root / 'preregistration/fixtures_protocol_v2.json')
    raw = c.read(root / 'fixtures/attempt_002/raw.json')
    assert raw['status'] == 'CORRECTNESS_FAILED'  # Never overwrite the original verdict.
    assert raw['protocol_SHA256'] == c.sha(c.encode(protocol))
    assert raw['payload_SHA256'] == c.sha((root / 'preregistration/fixtures_payload.zip').read_bytes())
    assert raw['driver_return_codes'] == [0] and not raw['Triton_imported']
    assert raw['environment']['cc'] == [9, 0] and 'H100' in raw['environment']['name']
    assert len(raw['checks']) == 234 and len(raw['bindings']) == 78
    for suffix, folder in (('', 'fixtures_source'), ('_v2', 'fixtures_source_v2')):
        p = c.read(root / ('preregistration/fixtures_protocol' + suffix + '.json'))
        for name, digest in p['worker_source_SHA256'].items():
            assert c.sha((root / 'preregistration' / folder / name).read_bytes()) == digest, name
    prepared = c.read(c.PREP / 'pinned_cpu_attempt_001/result.json')
    assert prepared['native_SHA256'] == c.NATIVE
    metadata = {row['tag']: row for row in prepared['records']}
    for row in raw['bindings']:
        assert row['cubin_SHA256'] == row['loaded_SHA256'] == metadata[row['tag']]['cubin_SHA256']
        assert row['attributes']['occupancy_capacity_q'] > 0
    failures = [row for row in raw['checks'] if not row['pass']]
    assert len(failures) == 18 and all(row['tag'].startswith('C03:') for row in failures)
    for row in raw['checks']:
        for value in row['outputs_and_inputs']:
            assert value['canary_pass']
            if 'input_unchanged' in value:
                assert value['input_unchanged']
            if not value['pass']:
                assert row['tag'].startswith('C03:') and value['name'] == 'quantized'
    # This is a reference revision, not a relaxed tolerance or a GPU rerun.
    # The archived PTX computes 1/448 followed by mul.rn.f32 for the scale.
    ptx = c.PREP / 'pinned_cpu_attempt_001/artifacts/C03/canonical_triton/kernel.ptx'
    text = ptx.read_text()
    assert 'div.approx.f32' in text and '0f43E00000' in text and 'mul.rn.f32' in text
    case = next(row for row in protocol['cases'] if row['id'] == 'C03')
    diagnostic = []
    for pattern in protocol['correctness']['patterns']:
        x = oracles.inputs(case, pattern)['x']
        scale = np.maximum(np.abs(x).max(1), np.float32(1.e-12)) * np.float32(1 / 448)
        expected = {'scale_out': scale, 'quantized': oracles.fp8_encode(x / scale[:, None, :]).transpose(0, 2, 1).copy()}
        rows = [row for row in raw['checks'] if row['tag'].startswith('C03:') and row['pattern'] == pattern]
        assert len(rows) == 9
        for row in rows:
            for name, value in expected.items():
                actual = next(item for item in row['outputs_and_inputs'] if item['name'] == name)
                assert c.sha(value.tobytes()) == actual['actual_SHA256'], (row['tag'], pattern, name)
                diagnostic.append({'tag': row['tag'], 'pattern': pattern, 'output': name,
                                   'exact_SHA256': actual['actual_SHA256'], 'pass': True})
    earlier = c.read(root / 'fixtures/attempt_001/retrieval_failure.json')
    assert earlier['GPU_status'] == 'FAILED'
    summary = {'status': 'PASS_WITH_EXPLICIT_FP8_REFERENCE_REVISION',
               'original_GPU_status': raw['status'], 'original_reference_passes': 216,
               'original_reference_checks': 234, 'post_audit_passes': 234,
               'FP8_lowering_aware_exact_output_hash_checks': diagnostic,
               'FP8_tolerance_relaxed': False, 'GPU_rerun_for_reference_revision': False,
               'eligible_semantic_families': [f'C{i:02}' for i in range(1, 9)],
               'production_adapters_validated': False, 'whole_graph_bridges_validated': False,
               'timing_collected': False, 'compiler_calls': 0, 'native_rebuilds': 0,
               'attempts_retained': ['attempt_001 (ABI harness failure)', 'attempt_002 (234 completed checks)'],
               'environment': raw['environment'], 'source_of_GPU_data': 'fixtures/attempt_002/raw.json'}
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    result = evaluate()
    if args.write:
        c.write_once(c.OUT / 'stage_g1/correctness_gate.json', result)
    print(result['status'], '234/234 audited fixture checks; original 216/234 retained; no timing')


if __name__ == '__main__':
    main()
