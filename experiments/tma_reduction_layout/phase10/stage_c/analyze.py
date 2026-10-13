"""Independent timing contracts and descriptive effects; no Modal/CUDA imports."""
import argparse
from collections import Counter
import math
import statistics
import zipfile

from experiments.tma_reduction_layout.phase10.stage_c import common as c


def expected_order(identities, invocation, round_number):
    pairs = sorted((name, fraction) for name in identities for fraction in (1, 4, 16))
    # The preregistered master sorts the string case:gF, hence g16 before g4.
    pairs.sort(key=lambda pair: pair[0] + ':g' + str(pair[1]))
    offset = ((invocation - 1) * 17 + (round_number - 1) * 13) % len(pairs)
    pairs = pairs[offset:] + pairs[:offset]
    variants = ('baseline', 'patch') if (invocation + round_number - 1) % 2 == 0 else ('patch', 'baseline')
    return [(name + ':' + v, f) for name, f in pairs for v in variants]


def validate_raw(raw, target, invocation, metadata, identities):
    protocol = c.protocol()
    assert raw['status'] == 'OK' and raw['target'] == target and raw['invocation'] == invocation
    assert raw['protocol_SHA256'] == c.sha(c.encode(protocol))
    assert raw['worker_source_SHA256'] == protocol['worker_source_SHA256']
    assert raw['compilation'] == {'triton_imported': False, 'subprocess_calls': 0, 'compiler_calls': 0}
    assert raw['cuda_return_codes'] == [0]
    assert raw['environment']['cc'] == protocol['targets'][target]['cc']
    assert protocol['targets'][target]['name'] in raw['environment']['gpu_name']
    uuid = raw['environment']['gpu_uuid']
    assert uuid
    assert set(raw['binaries']) == set(metadata) == set(raw['allocations'])
    for tag, binding in raw['binaries'].items():
        assert binding['cubin_SHA256'] == metadata[tag]['cubin_SHA256']
        assert binding['launches'] == binding['guards'] == 159
        expected_abi = ['descriptor128'] + ['u32'] * 3 + ['u64'] * 6 if metadata[tag]['path'] == 'host' else ['u64'] * 4
        assert binding['ABI'] == expected_abi
        alloc = raw['allocations'][tag]
        assert alloc['scratch_bytes'] >= max(1, metadata[tag]['global_scratch_size'] * metadata[tag]['grid'])
        assert alloc['output_elements'] == metadata[tag]['grid'] * c.output_per_cta(metadata[tag])
        other = tag.rsplit(':', 1)[0] + (':patch' if tag.endswith(':baseline') else ':baseline')
        assert raw['allocations'][tag] == raw['allocations'][other]

    def check(record, tag, fraction):
        meta = metadata[tag]
        assert record['tag'] == tag and record['fraction'] == fraction
        assert record['grid'] == meta['grid'] // fraction and record['gpu_uuid'] == uuid
        assert record['binary']['cubin_SHA256'] == meta['cubin_SHA256']
        for key in ('module', 'function', 'ABI'):
            assert record['binary'][key] == raw['binaries'][tag][key]
        alloc = raw['allocations'][tag]
        vals = record['actual_parameters']
        expected = ([
            vals[0], 1, 8192, 8192, 8192 * 8192, 8192, 1, alloc['output_pointer'], alloc['scratch_pointer'], 0
        ] if meta['path'] == 'host' else [alloc['input_pointer'], alloc['output_pointer'], alloc['scratch_pointer'], 0])
        assert vals == expected
        if meta['path'] == 'host':
            assert vals[0] > 0 and vals[0] % 64 == 0

    first = expected_order(identities, invocation, 1)
    assert len(raw['warmups']) == len(first)
    warm = {}
    for record, (tag, fraction) in zip(raw['warmups'], first):
        check(record, tag, fraction)
        assert record['count'] == 3 and record['correctness_max_abs_diff'] == 0
        assert record['checked_output_elements'] == record['grid'] * c.output_per_cta(metadata[tag])
        warm[tag, fraction] = record['actual_parameters']
    assert len(raw['visits']) == len(first) * 10
    samples = {}
    for round_number in range(1, 11):
        order = expected_order(identities, invocation, round_number)
        for j, (tag, fraction) in enumerate(order):
            visit = raw['visits'][(round_number - 1) * len(order) + j]
            check(visit, tag, fraction)
            assert visit['round'] == round_number and visit['actual_parameters'] == warm[tag, fraction]
            assert len(visit['samples_us']) == len(visit['launch_SHA256']) == 5
            assert all(d == metadata[tag]['cubin_SHA256'] for d in visit['launch_SHA256'])
            assert all(math.isfinite(x) and x > 0 for x in visit['samples_us'])
            samples.setdefault((tag, fraction), []).extend(visit['samples_us'])
    count = len(metadata) * 3 * 50
    assert sum(map(len, samples.values())) == count
    assert raw['cuda_call_counts']['cuLaunchKernel'] == len(metadata) * 159
    assert raw['cuda_call_counts']['cuEventRecord'] == 2 * count
    assert raw['cuda_call_counts']['cuEventElapsedTime'] == count
    assert raw['cuda_call_counts']['cuEventSynchronize'] == count
    assert all(len(values) == 50 for values in samples.values())
    return {tag: (sorted(values)[24] + sorted(values)[25]) / 2 for tag, values in samples.items()}


def effect(values):
    mean = statistics.mean(values)
    half = 4.302652729911275 * statistics.stdev(values) / math.sqrt(3)
    lo, hi = mean - half, mean + half
    status = ('BENEFIT' if lo > 3 else
              'REGRESSION' if hi < -3 else 'WITHIN_PRACTICAL_BAND' if lo >= -3 and hi <= 3 else 'UNRESOLVED')
    return {'invocation_effects_percent': values, 'mean_percent': mean, 'band': [lo, hi], 'status': status}


def run():
    protocol = c.protocol()
    gate = c.read(c.OUT / 'artifact_gate.json')
    rows, totals = [], {}
    cases = {row['case_id']: row for row in protocol['cases']}
    for target, target_gate in gate.items():
        root = c.OUT / 'timing' / target
        metadata = c.read(root / 'metadata.json')
        with zipfile.ZipFile(root / 'payload.zip') as z:
            assert len(z.namelist()) == len(metadata) * 2
            for tag, meta in metadata.items():
                assert c.sha(z.read(tag.replace(':', '/') + '/kernel.cubin')) == meta['cubin_SHA256']
        medians, uuids, calls = {}, [], []
        for invocation in (1, 2, 3):
            raw = c.read(root / f'raw_invocation_{invocation}.json')
            assert raw['payload_SHA256'] == c.sha((root / 'payload.zip').read_bytes())
            assert c.read(root / ('attempt_' + raw['call_id'] + '.json')) == raw
            medians[invocation] = validate_raw(raw, target, invocation, metadata, target_gate['eligible'])
            uuids.append(raw['environment']['gpu_uuid'])
            calls.append(raw['call_id'])
        assert len(set(calls)) == 3
        observations = {row['case_id']: row for row in target_gate['observations']}
        for identity in target_gate['eligible']:
            for fraction in (1, 4, 16):
                a, b = identity + ':baseline', identity + ':patch'
                values = [
                    100 * (medians[i][a, fraction] - medians[i][b, fraction]) / medians[i][a, fraction]
                    for i in (1, 2, 3)
                ]
                rows.append({
                    'target': target, 'case_id': identity, 'cohort': cases[identity]['cohort'], 'fraction': fraction,
                    'grid': c.grid(cases[identity],
                                   fraction), 'baseline_medians_us': [medians[i][a, fraction] for i in (1, 2, 3)],
                    'patch_medians_us': [medians[i][b, fraction]
                                         for i in (1, 2, 3)], **effect(values), 'physical_UUIDs': uuids, 'same_CUBIN':
                    observations[identity]['same_CUBIN'], 'resource_stratum': observations[identity]['resource_stratum']
                })
        totals[target] = {
            'eligible_pairs':
            len(target_gate['eligible']), 'unavailable_pairs':
            len(target_gate['unavailable']), 'samples':
            len(target_gate['eligible']) * 900, 'UUIDs':
            uuids, 'primary_counts':
            dict(Counter(row['status'] for row in rows if row['target'] == target and row['fraction'] == 1))
        }
    result = {
        'rows': rows, 'targets': totals, 'total_samples': sum(v['samples'] for v in totals.values()), 'unavailable':
        {t: g['unavailable']
         for t, g in gate.items()}, 'patch_adjusted': False, 'stop_after_stage_c': True
    }
    text = '# Phase 10 Stage C — Frozen actual-patch validation\n\n'
    text += ('The exact Stage B patch is compared with the same upstream and assembler, on strict '
             'H100, B200 and RTX PRO 6000 workers. Eight unseen tile/warp triples have the preregistered '
             'BF16/max, FP16/min and FP32/sum cells on host/device paths; anchors and controls remain separate. '
             'The matrix is a declared subset, not complete dtype/op factorial coverage.\n\n')
    text += '| Target | Eligible pairs | Unavailable | Samples | Primary decisions |\n| --- | --- | --- | --- | --- |\n'
    for target, data in totals.items():
        text += f'| {target} | {data["eligible_pairs"]} | {data["unavailable_pairs"]} | {data["samples"]} | {data["primary_counts"]} |\n'
    text += '\n| Target | Primary case | Cohort | Effect % | Process band % | Decision | Same CUBIN | Resource stratum |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n'
    for row in rows:
        if row['fraction'] == 1:
            text += f'| {row["target"]} | {row["case_id"]} | {row["cohort"]} | {row["mean_percent"]:.3f} | [{row["band"][0]:.3f}, {row["band"][1]:.3f}] | {row["status"]} | {row["same_CUBIN"]} | {row["resource_stratum"]} |\n'
    text += ('\nAll secondary grid effects and individual invocation medians remain in results.json. '
             'Bands describe three processes and have no familywise significance guarantee. '
             'An unavailable or unresolved result is not safety evidence. Identical-CUBIN timing variation '
             'cannot be attributed to a compiler code change. Native counts alone do not identify '
             'bank-conflict, latency or historical regression causality. No policy adjustment, outcome '
             'rerun, PR creation/update or later stage occurred.\n')
    return result, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    result, text = run()
    if args.validate:
        assert (c.OUT / 'results.json').read_bytes() == c.encode(result)
        assert (c.OUT / 'summary.md').read_text() == text
    else:
        c.write_once(c.OUT / 'results.json', result)
        c.write_once(c.OUT / 'summary.md', text.encode())
    print('Timing analysis PASS', result['total_samples'], 'samples')


if __name__ == '__main__':
    main()
