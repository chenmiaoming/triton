"""CPU-only paired analysis of immutable non-reduction timing evidence.

This follows the frozen G8 estimator: arithmetic mean of worker/visit sample
medians, with the same worker and visit bootstrap draws for every layout of a
case.  Raw timing evidence and compiler artifacts are never rewritten.
"""
import argparse
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path

import numpy as np

from . import common as c


def programs(protocol):
    """Return one metadata record for each expected logical program tag."""
    value = protocol['programs']
    if isinstance(value, dict):
        result = {tag: dict(row, tag=tag) for tag, row in value.items()}
    else:
        result = {row['tag']: dict(row) for row in value}
        assert len(result) == len(value), 'Duplicate program tags in protocol'
    cases = {row['id']: row for row in protocol['cases']}
    assert result
    for tag, row in result.items():
        assert row['geometry'] in cases, (tag, row['geometry'])
        assert isinstance(row['variant'], str), (tag, row)
    for case_id in cases:
        variants = [row['variant'] for row in result.values() if row['geometry'] == case_id]
        assert variants.count('U') == variants.count('G0') == 1, (case_id, variants)
    return result


def collect(protocol, raw_records):
    """Validate exact logical coverage, including measured-program aliases."""
    expected = programs(protocol)
    timing = protocol['timing']
    workers, visits, samples = [int(timing[key]) for key in ('workers', 'visits', 'samples')]
    assert len(raw_records) == workers, (len(raw_records), workers)
    protocol_sha = c.sha(c.encode(protocol))
    seen_workers = set()
    indexed, representatives, visit_ids = {}, {}, {}
    physical_sample_count = 0
    physical_cell_count = 0
    environments, checks = [], []
    nodes = Counter()
    for raw in raw_records:
        assert raw['status'] == 'OK', raw.get('status')
        assert raw['protocol_SHA256'] == protocol_sha
        assert raw['payload_SHA256'] == protocol['payload_SHA256']
        assert raw['phase'] == protocol['phase']
        worker = str(raw['invocation'])
        assert worker not in seen_workers, ('Duplicate worker', worker)
        seen_workers.add(worker)
        environments.append(dict(invocation=worker, environment=raw['environment']))
        checks.append(dict(invocation=worker, count=len(raw['checks'])))
        assert isinstance(raw['bindings'], dict)
        assert set(raw['bindings']) == set(expected), ('Binding coverage', worker)
        logical = {}
        common_ids = None
        for cell in raw['cells']:
            tag = cell['tag']
            assert tag in expected, ('Unknown representative tag', tag)
            assert cell['case'] == expected[tag]['geometry']
            assert cell['pattern'] == 'finite'
            entries = cell['visits']
            assert len(entries) == visits, (worker, tag, len(entries), visits)
            by_visit = {str(entry['visit']): entry for entry in entries}
            assert len(by_visit) == visits, ('Duplicate visit', worker, tag)
            ids = sorted(by_visit, key=lambda x: (len(x), x))
            if common_ids is None:
                common_ids = ids
            assert ids == common_ids, ('Unpaired visits', worker, tag)
            values = []
            for visit in ids:
                entry = by_visit[visit]
                vector = np.asarray(entry['samples_us'], dtype=float)
                assert vector.shape == (samples,), (worker, tag, visit, vector.shape)
                assert np.all(np.isfinite(vector)) and np.all(vector > 0), (worker, tag, visit)
                values.append(float(np.median(vector)))
                physical_sample_count += len(vector)
                node_types = entry.get('node_types', [])
                nodes.update(str(value) for value in node_types)
            tags = [tag] + list(cell.get('aliases', []))
            assert len(tags) == len(set(tags)), ('Duplicate aliases', worker, tag)
            for alias in tags:
                assert alias in expected, ('Unknown alias', alias)
                assert expected[alias]['geometry'] == cell['case'], (alias, cell['case'])
                assert alias not in logical, ('Logical program measured twice', worker, alias)
                logical[alias] = values
                representatives.setdefault(alias, {})[worker] = tag
            physical_cell_count += 1
        assert set(logical) == set(expected), {
            'worker': worker, 'missing': sorted(set(expected)-set(logical)),
            'unexpected': sorted(set(logical)-set(expected))}
        indexed[worker] = logical
        visit_ids[worker] = common_ids
        # U and G0 are deliberately separate physical timing observations.
        for case in protocol['cases']:
            tags = {row['variant']: tag for tag, row in expected.items()
                    if row['geometry'] == case['id'] and row['variant'] in ('U', 'G0')}
            assert representatives[tags['U']][worker] != representatives[tags['G0']][worker], (
                'U/G0 timing must be independently measured', case['id'], worker)
    worker_ids = sorted(indexed, key=lambda x: (len(x), x))
    arrays = {tag: np.asarray([indexed[worker][tag] for worker in worker_ids]) for tag in expected}
    assert all(value.shape == (workers, visits) for value in arrays.values())
    audit = dict(workers=worker_ids, visits_by_worker=visit_ids,
        logical_programs=len(expected), physical_cells_all_workers=physical_cell_count,
        raw_sample_count=physical_sample_count,
        logical_equivalent_sample_count=len(expected)*workers*visits*samples,
        correctness_check_counts=checks, environments=environments,
        graph_node_type_observations=dict(sorted(nodes.items())),
        coverage='Exact protocol program tags on every worker; aliases inherit the same observed visit medians.',
        baseline_measurement='U and G0 occupy separate physical timing cells on every worker.')
    return expected, arrays, representatives, audit


def direction(ci, threshold):
    if ci[0] > threshold:
        return 'practical_improvement'
    if ci[1] < -threshold:
        return 'practical_regression'
    if ci[0] >= -threshold and ci[1] <= threshold:
        return 'practically_equivalent'
    return 'inconclusive'


def analyze(protocol, raw_records):
    """Pure analysis API; the caller supplies already loaded immutable JSON."""
    expected, arrays, representatives, audit = collect(protocol, raw_records)
    timing = protocol['timing']
    workers, visits = int(timing['workers']), int(timing['visits'])
    reps = int(timing['bootstrap_replicates'])
    assert reps == 4096, 'Use the preregistered G8 bootstrap replicate count'
    threshold = float(timing['practical_threshold_percent'])
    master_seed = int(timing['seed'])
    by_case = defaultdict(list)
    for tag, spec in expected.items():
        by_case[spec['geometry']].append(tag)
    rows, baseline_rows, case_rows = [], [], []
    for case in protocol['cases']:
        case_id = case['id']
        tags = sorted(by_case[case_id])
        baselines = {expected[tag]['variant']: tag for tag in tags
                     if expected[tag]['variant'] in ('U', 'G0')}
        seed = master_seed + int(c.sha(c.encode(case_id))[:8], 16)
        rng = np.random.default_rng(seed)
        wi = rng.integers(0, workers, (reps, workers))
        vi = rng.integers(0, visits, (reps, workers, visits))
        def resample(value):
            return value[wi[:, :, None], vi].mean((1, 2))
        boot = {tag: resample(arrays[tag]) for tag in tags}
        case_output = []
        for tag in tags:
            value = arrays[tag]
            row = dict(case=case_id, family=case['family'], tag=tag,
                variant=expected[tag]['variant'], time_us=float(value.mean()),
                worker_visit_medians_us=value.tolist(), worker_means_us=value.mean(1).tolist(),
                physical_representative_tags=[representatives[tag][w] for w in audit['workers']],
                bootstrap_seed=seed)
            for name in ('U', 'G0'):
                base_tag = baselines[name]
                base = arrays[base_tag]
                gain = float(100*(base.mean()-value.mean())/base.mean())
                ci = np.percentile(100*(boot[base_tag]-boot[tag])/boot[base_tag], [2.5, 97.5]).tolist()
                row['gain_vs_'+name+'_percent'] = gain
                row['gain_vs_'+name+'_CI95'] = ci
                row['worker_gains_vs_'+name+'_percent'] = (100*(base.mean(1)-value.mean(1))/base.mean(1)).tolist()
                row['direction_vs_'+name] = direction(ci, threshold)
                row['reference_'+name+'_tag'] = base_tag
                row['shared_physical_observation_with_'+name] = all(
                    representatives[tag][w] == representatives[base_tag][w] for w in audit['workers'])
            rows.append(row)
            case_output.append(row)
            if row['variant'] == 'G0':
                baseline_rows.append(dict(case=case_id, family=case['family'],
                    U_time_us=float(arrays[baselines['U']].mean()), G0_time_us=row['time_us'],
                    gain_percent=row['gain_vs_U_percent'], CI95=row['gain_vs_U_CI95'],
                    classification=row['direction_vs_U'], independent_physical_measurements=True,
                    bootstrap_seed=seed))
        candidates = [row for row in case_output if row['variant'] not in ('U', 'G0')]
        best = min(candidates, key=lambda row: (row['time_us'], row['tag'])) if candidates else None
        baseline = next(row for row in case_output if row['variant'] == 'G0')
        case_rows.append(dict(case=case_id, family=case['family'],
            baseline_equivalence=baseline['direction_vs_U'], candidate_count=len(candidates),
            best_observed_tag=best['tag'] if best else None,
            best_observed_variant=best['variant'] if best else None,
            best_observed_gain_vs_U_percent=best['gain_vs_U_percent'] if best else None,
            best_observed_gain_vs_G0_percent=best['gain_vs_G0_percent'] if best else None,
            practical_improvement_candidates_vs_U=sum(row['direction_vs_U']=='practical_improvement' for row in candidates),
            practical_regression_candidates_vs_U=sum(row['direction_vs_U']=='practical_regression' for row in candidates),
            claim_scope=case.get('evidence_scope', 'Standalone pinned source kernel, fixed case and configuration.')))
    families = []
    for family in sorted({row['family'] for row in case_rows}):
        group = [row for row in case_rows if row['family']==family]
        measured = [row for row in group if row['best_observed_tag'] is not None]
        families.append(dict(family=family, cases=len(group),
            baseline_equivalent_cases=sum(row['baseline_equivalence']=='practically_equivalent' for row in group),
            candidate_logical_programs=sum(row['candidate_count'] for row in group),
            cases_with_practical_candidate_improvement=sum(row['practical_improvement_candidates_vs_U']>0 for row in group),
            best_observed_gain_vs_U_mean_percent=float(np.mean([row['best_observed_gain_vs_U_percent'] for row in measured])) if measured else None,
            scope='Descriptive equal-case summary of standalone kernels; post hoc best observed candidates are exploratory.'))
    return dict(status='OK', phase=protocol['phase'], protocol_SHA256=c.sha(c.encode(protocol)),
        payload_SHA256=protocol['payload_SHA256'], audit=audit, rows=rows,
        baseline_equivalence=baseline_rows, cases=case_rows, families=families,
        estimator='Arithmetic mean of worker x visit medians of 31 raw samples; paired worker/visit hierarchical bootstrap with shared U/G0/GL draws, 4096 replicates.',
        bootstrap_master_seed=master_seed, practical_threshold_percent=threshold,
        uncertainty='Pointwise percentile 95% intervals; no within-visit raw-sample resampling and no multiplicity correction. Best-observed candidate summaries are exploratory, not held-out confirmation.',
        interpretation='Gain is 100*(mean_reference-mean_candidate)/mean_reference. Timing equivalence is a classification, never an asserted outcome. No framework or complete-model performance claim.')


def csv_bytes(result):
    fields = ['case', 'family', 'tag', 'variant', 'time_us',
        'gain_vs_U_percent', 'gain_vs_U_CI95', 'direction_vs_U',
        'gain_vs_G0_percent', 'gain_vs_G0_CI95', 'direction_vs_G0',
        'worker_means_us', 'worker_gains_vs_U_percent', 'worker_gains_vs_G0_percent',
        'physical_representative_tags', 'bootstrap_seed']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for row in result['rows']:
        writer.writerow({key: json.dumps(row[key], separators=(',', ':')) if isinstance(row[key], (list, dict))
                         else row[key] for key in fields})
    return stream.getvalue().encode()


def run(protocol_path, raw_paths, output):
    protocol_path, output = Path(protocol_path), Path(output)
    raw_paths = [Path(path) for path in raw_paths]
    protocol = c.read(protocol_path)
    result = analyze(protocol, [c.read(path) for path in raw_paths])
    result['inputs'] = dict(protocol=dict(path=str(protocol_path), SHA256=c.sha(protocol_path.read_bytes())),
        raw=[dict(path=str(path), SHA256=c.sha(path.read_bytes())) for path in raw_paths],
        analysis_source=dict(path=str(Path(__file__)), SHA256=c.sha(Path(__file__).read_bytes())))
    c.write_once(output/'analysis.json', result)
    c.write_once(output/'timing.csv', csv_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--raw', type=Path, action='append', default=[])
    parser.add_argument('--timing-dir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = list(args.raw)
    if args.timing_dir:
        paths.extend(sorted(args.timing_dir.glob('invocation_*/raw.json')))
    assert paths and len(paths)==len(set(paths)), 'Supply each immutable raw worker JSON once'
    result = run(args.protocol, paths, args.output)
    print(result['phase'], result['status'], 'logical programs', result['audit']['logical_programs'],
          'raw samples', result['audit']['raw_sample_count'])


if __name__ == '__main__':
    main()
