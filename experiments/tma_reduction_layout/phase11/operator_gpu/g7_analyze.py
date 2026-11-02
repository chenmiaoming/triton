"""Offline paired timing and output-byte audit of the frozen G7 U/G0 runs."""
import csv
import io
from collections import defaultdict
from pathlib import Path
import numpy as np
from . import common as c
from .g7_prepare import OUT


def save(path, value):
    blob = value if isinstance(value, bytes) else c.encode(value)
    if path.exists():
        assert path.read_bytes() == blob, path
    else:
        c.write_once(path, blob)


def protocol_path():
    revised = OUT/'preregistration/protocol_v2.json'
    return revised if revised.exists() else OUT/'preregistration/protocol.json'


def expected_conditions(protocol):
    result = {}
    actual_groups = {g['id'] for g in protocol['binding_groups']}
    for group in protocol['inference_groups']:
        for work in group['workloads']:
            modes = ['stage1', 'decode'] if group['case']['family'] == 'C02' else ['merge']
            for mode in modes:
                key = ('inference', group['id'], work['batch'], work['length'], mode)
                assert key not in result
                result[key] = dict(domain='inference', group=group['id'], geometry=group['case']['id'],
                    batch=work['batch'], length=work['length'], M=None, kernel_M=None, mode=mode,
                    panel='actual_inference' if group['id'] in actual_groups else 'default_inference')
    for case in protocol['training_cases']:
        for mode, m in [('merge', 128)] + [('pipeline', m) for m in case['pipeline_M']]:
            key = ('training', case['id'], m, mode)
            assert key not in result
            result[key] = dict(domain='training', group=case['id'], geometry=case['id'],
                batch=None, length=None, M=m, kernel_M=case['P'] if mode == 'merge' else min(case['P'], m),
                mode=mode, panel='training_'+mode)
    assert len(result) == 32, len(result)
    return result


def validate_run(raw, protocol, invocation, payload_sha):
    assert raw['status'] == 'OK' and raw['invocation'] == invocation
    assert raw['protocol_SHA256'] == c.sha(c.encode(protocol))
    assert raw['payload_SHA256'] == payload_sha
    for domain in ['inference', 'training']:
        value = raw[domain]
        assert value['status'] == 'OK' and value['invocation'] == invocation
        assert value['protocol_SHA256'] == raw['protocol_SHA256']
        assert value['payload_SHA256'] == payload_sha
        assert value['compile_calls'] == 0 and not value['Triton_imported']
        assert value['driver_return_codes'] == [0]
        assert all(check['pass'] for check in value['checks'])
        assert value['environment']['cc'] == [9, 0] and 'H100' in value['environment']['name']


def bitwise_summary(runs):
    totals = {name: dict(comparisons=0, bitwise_equal=0, bitwise_different=0,
                         required_bitwise_equal=required)
              for name, required in [('inference', True), ('training_fixed_input_merge', True),
                                     ('training_independent_pipeline', False),
                                     ('training_same_partials_merge', True)]}
    by_run = []
    for label, raw in runs:
        counts = defaultdict(lambda: dict(comparisons=0, bitwise_equal=0, bitwise_different=0))
        groups = [('inference', value) for value in raw['inference']['bitwise_comparisons']]
        groups += [('training_fixed_input_merge' if value['mode'] == 'merge'
                    else 'training_independent_pipeline', value)
                   for value in raw['training']['bitwise_comparisons']]
        groups += [('training_same_partials_merge', value)
                   for value in raw['training']['same_partial_comparisons']]
        for category, value in groups:
            equal = bool(value['bitwise_equal'])
            if totals[category]['required_bitwise_equal']:
                assert equal, (label, category, value)
            if category == 'training_same_partials_merge':
                assert value['partials_and_lock_unchanged'] and value['dX_executions'] == 0
                assert all(check['pass'] for check in value['checks'])
            for dest in [totals[category], counts[category]]:
                dest['comparisons'] += 1
                dest['bitwise_equal' if equal else 'bitwise_different'] += 1
        by_run.append({'run': label, 'categories': dict(counts)})
    return dict(categories=totals, by_run=by_run,
                scope='Independent whole backward reruns may have different partial accumulation order; '
                      'fixed-input and same-produced-partial merge comparisons require byte equality.')


def main():
    ppath = protocol_path(); protocol = c.read(ppath); c.verify_sources(protocol)
    timing = protocol['timing']; assert timing['no_result_dependent_repeats']
    workers, visits, samples = [timing[name] for name in ['workers', 'visits', 'samples']]
    assert (workers, visits, samples) == (3, 5, 31)
    expected = expected_conditions(protocol)
    payload_sha = c.sha((OUT/'compiled_payload.zip').read_bytes())
    observations = defaultdict(dict); raw_sources = []; environments = []; all_runs = []
    raw_samples = 0
    for invocation in range(1, workers+1):
        path = OUT/f'timing/invocation_{invocation}/raw.json'; raw = c.read(path)
        validate_run(raw, protocol, invocation, payload_sha)
        raw_sources.append(dict(path=str(path.relative_to(c.ROOT)), SHA256=c.sha(path.read_bytes())))
        environments.append(dict(invocation=invocation, call_id=raw['call_id'],
                                 inference=raw['inference']['environment'], training=raw['training']['environment']))
        all_runs.append((f'timing/invocation_{invocation}', raw))
        seen = set()
        for domain in ['inference', 'training']:
            for cell in raw[domain]['cells']:
                if domain == 'inference':
                    key = (domain, cell['group'], cell['batch'], cell['length'], cell['mode'])
                else:
                    key = (domain, cell['geometry'], cell['M'], cell['mode'])
                    assert cell['kernel_M'] == expected[key]['kernel_M']
                assert key in expected and cell['complete_output_check']['pass']
                assert not cell.get('aliases'), cell
                role = cell['tag'].rsplit(':', 1)[1]; assert role in ['U', 'G0']
                fullkey = (key, role); assert fullkey not in seen; seen.add(fullkey)
                by_visit = {value['visit']: value for value in cell['visits']}
                assert len(cell['visits']) == len(by_visit) == visits
                assert set(by_visit) == set(range(visits))
                medians = []
                for visit in range(visits):
                    value = by_visit[visit]
                    values = np.asarray(value['samples_us'], dtype=np.float64)
                    assert values.shape == (samples,) and np.all(np.isfinite(values)) and np.all(values > 0)
                    nodes = [0, 0, 2, 2, 2, 7, 7] if domain == 'training' and cell['mode'] == 'pipeline' else (
                        [0, 0, 7, 7] if cell['mode'] == 'decode' else [0, 7, 7])
                    assert sorted(value['node_types']) == nodes, (key, value['node_types'])
                    medians.append(float(np.median(values))); raw_samples += values.size
                observations[fullkey][invocation] = medians
        assert seen == {(key, role) for key in expected for role in ['U', 'G0']}
    correctness = OUT/'correctness/attempt_001/raw.json'
    correct = c.read(correctness); validate_run(correct, protocol, 0, payload_sha)
    all_runs.insert(0, ('correctness/attempt_001', correct))
    correct_source = dict(path=str(correctness.relative_to(c.ROOT)), SHA256=c.sha(correctness.read_bytes()))
    rows = []
    for key in sorted(expected):
        arrays = {role: np.asarray([observations[(key, role)][i] for i in range(1, workers+1)])
                  for role in ['U', 'G0']}
        u, g = arrays['U'], arrays['G0']
        # Worker resampling and visit resampling use the same indices for U/G0.
        seed = timing['seed'] + int(c.sha(c.encode(key))[:8], 16)
        rng = np.random.default_rng(seed); reps = timing['bootstrap_replicates']
        wi = rng.integers(0, workers, (reps, workers))
        vi = rng.integers(0, visits, (reps, workers, visits))
        ub = u[wi[:, :, None], vi].mean((1, 2)); gb = g[wi[:, :, None], vi].mean((1, 2))
        differences = 100*(gb/ub-1)
        ci = np.percentile(differences, [2.5, 97.5]).tolist()
        band = timing['practical_threshold_percent']
        equivalent = ci[0] >= -band and ci[1] <= band
        changes = 100*(g.mean(1)/u.mean(1)-1)
        workers_detail = [dict(invocation=i+1, U_visit_medians_us=u[i].tolist(),
                               G0_visit_medians_us=g[i].tolist(), U_mean_visit_median_us=float(u[i].mean()),
                               G0_mean_visit_median_us=float(g[i].mean()), relative_change_percent=float(changes[i]))
                          for i in range(workers)]
        rows.append({**expected[key], 'U_time_us': float(u.mean()), 'G0_time_us': float(g.mean()),
                     'G0_minus_U_us': float(g.mean()-u.mean()),
                     'G0_over_U_change_percent': float(100*(g.mean()/u.mean()-1)),
                     'G0_over_U_change_CI95_percent': ci,
                     'practical_equivalence_band_percent': [-band, band],
                     'timing_classification': 'practically_equivalent' if equivalent else 'uncertain',
                     'worker_changes_percent': changes.tolist(), 'worker_visit_medians': workers_detail,
                     'bootstrap_seed': seed})
    expected_samples = len(expected)*2*workers*visits*samples
    assert raw_samples == expected_samples
    summary = bitwise_summary(all_runs)
    result = dict(status='OK', protocol_path=str(ppath.relative_to(c.ROOT)),
                  protocol_SHA256=c.sha(c.encode(protocol)), payload_SHA256=payload_sha,
                  logical_conditions=len(rows), raw_samples=raw_samples,
                  classification_counts={name: sum(r['timing_classification'] == name for r in rows)
                                         for name in ['practically_equivalent', 'uncertain']},
                  rows=rows, raw_sources=raw_sources, correctness_source=correct_source,
                  environments=environments, bitwise_summary=summary,
                  estimator='mean of 3 workers x 5 visit medians; paired worker/visit hierarchical bootstrap 4096',
                  relative_change_definition='100*(G0/U-1): negative means G0 faster; positive means G0 slower',
                  bootstrap_master_seed=timing['seed'],
                  CI_scope='Pointwise 95% intervals, no multiplicity adjustment. A CI wholly inside the '
                           'preregistered [-3%,+3%] band is practical timing equivalence; all others stay uncertain. '
                           'This is not a new layout-performance claim or permission for result-dependent reruns.')
    save(OUT/'timing_analysis.json', result)
    stream = io.StringIO()
    columns = ['domain', 'panel', 'group', 'geometry', 'batch', 'length', 'M', 'kernel_M', 'mode',
               'U_time_us', 'G0_time_us', 'G0_minus_U_us', 'G0_over_U_change_percent',
               'CI95_low_percent', 'CI95_high_percent', 'timing_classification',
               'worker_1_change_percent', 'worker_2_change_percent', 'worker_3_change_percent']
    writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
    for row in rows:
        value = {name: row[name] for name in columns if name in row}
        value.update(CI95_low_percent=row['G0_over_U_change_CI95_percent'][0],
                     CI95_high_percent=row['G0_over_U_change_CI95_percent'][1])
        value.update({f'worker_{i+1}_change_percent': change for i, change in enumerate(row['worker_changes_percent'])})
        writer.writerow(value)
    save(OUT/'timing.csv', stream.getvalue().encode())
    print('G7 timing analysis:', len(rows), 'conditions;', raw_samples, 'raw samples;', result['classification_counts'])


if __name__ == '__main__':
    main()
