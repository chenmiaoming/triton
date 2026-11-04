"""Complete paired analysis of the source-aligned replacement experiment.

Physical replay aliases share observations, but every preregistered logical
condition is represented once.  Old/new summaries compare within-run gains;
they never mix an old U duration with a newly measured candidate duration.
"""
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import numpy as np
from . import common as c

OUT = c.OUT/'stage_g8'
EXPECTED_PANELS = {'g2': 294, 'g3b': 99, 'g4a': 40, 'g4b': 84}


def save(path, value):
    blob = value if isinstance(value, bytes) else c.encode(value)
    if path.exists():
        assert path.read_bytes() == blob, path
    else:
        c.write_once(path, blob)


def protocol_path():
    revised = OUT/'preregistration/protocol_v2.json'
    return revised if revised.exists() else OUT/'preregistration/protocol.json'


def gate_path():
    revised = OUT/'compile_gate_v2.json'
    return revised if revised.exists() else OUT/'compile_gate.json'


def payload_path():
    revised = OUT/'compiled_payload_v2.zip'
    return revised if revised.exists() else OUT/'compiled_payload.zip'


def expected_conditions(protocol):
    result = {}
    for group in protocol['inference_groups']:
        assert group['panel'] in ['g2', 'g4a', 'g4b']
        for work in group['workloads']:
            modes = ['stage1', 'decode'] if group['case']['family'] == 'C02' else ['merge']
            for mode in modes:
                for variant in group['variants']:
                    tag = group['id']+':'+variant['name']
                    key = ('inference', group['panel'], group['id'], work['batch'], work['length'], mode, tag)
                    assert key not in result, key
                    result[key] = dict(domain='inference', panel=group['panel'], group=group['id'],
                        geometry=group['case']['id'], batch=work['batch'], length=work['length'],
                        mode=mode, M=None, kernel_M=None, tag=tag, variant=variant['name'])
    for case in protocol['training_cases']:
        for mode, m in [('merge', 128)]+[('pipeline', m) for m in case['pipeline_M']]:
            for variant in case['variants']:
                tag = case['id']+':'+variant['name']
                key = ('training', 'g3b', case['id'], mode, m, tag)
                assert key not in result, key
                result[key] = dict(domain='training', panel='g3b', group=case['id'], geometry=case['id'],
                    batch=None, length=None, mode=mode, M=m,
                    kernel_M=case['P'] if mode == 'merge' else min(case['P'], m),
                    tag=tag, variant=variant['name'])
    counts = dict(Counter(row['panel'] for row in result.values()))
    assert counts == EXPECTED_PANELS and len(result) == 517, counts
    return result


def physical_identity(meta):
    return (meta['cubin_SHA256'], tuple(meta['runtime_argument_names']),
            c.encode(meta.get('specialized_runtime_values', {})), meta['num_warps'],
            meta['dynamic_smem_bytes'], meta.get('global_scratch_size', 0))


def baseline_tags(row):
    prefix = row['group']+':'
    label = 'default'
    if row['panel'] == 'g4b':
        label = row['variant'].split(':', 1)[0]
        assert label in ['actual', 'B4'], row
        prefix += label+':'
    return prefix+'upstream', prefix+'gluon_anchor', label


def _cell_key(domain, cell, tag, groups):
    if domain == 'inference':
        group = groups[cell['group']]
        return (domain, group['panel'], cell['group'], cell['batch'], cell['length'], cell['mode'], tag)
    return (domain, 'g3b', cell['geometry'], cell['mode'], cell['M'], tag)


def _nodes(domain, mode):
    if domain == 'training' and mode == 'pipeline':
        return [0, 0, 2, 2, 2, 7, 7]
    return [0, 0, 7, 7] if mode == 'decode' else [0, 7, 7]


def collect(protocol, gate, payload_sha):
    expected = expected_conditions(protocol)
    metas = {row['tag']: row for row in gate['records']}
    assert len(metas) == len(gate['records'])
    groups = {g['id']: g for g in protocol['inference_groups']}
    timing = protocol['timing']; workers, visits, samples = [timing[k] for k in ['workers', 'visits', 'samples']]
    assert (workers, visits, samples) == (3, 5, 31)
    observed = defaultdict(dict); representatives = defaultdict(dict)
    total = 0; physical_cells = 0; environments = []; sources = []; worker_counts = []
    for invocation in range(1, workers+1):
        path = OUT/f'timing/invocation_{invocation}/raw.json'; raw = c.read(path)
        assert raw['status'] == 'OK' and raw['invocation'] == invocation
        assert raw['protocol_SHA256'] == c.sha(c.encode(protocol)) and raw['payload_SHA256'] == payload_sha
        sources.append(dict(path=str(path.relative_to(c.ROOT)), SHA256=c.sha(path.read_bytes())))
        seen = set(); worker_samples = 0; worker_physical = 0
        for domain in ['inference', 'training']:
            part = raw[domain]
            assert part['status'] == 'OK' and part['invocation'] == invocation
            assert part['protocol_SHA256'] == raw['protocol_SHA256'] and part['payload_SHA256'] == payload_sha
            # The original G3b runner does not emit a compile_calls field;
            # its process has no Triton imports and uses only driver loading.
            assert part.get('compile_calls', 0) == 0 and not part['Triton_imported'] and part['driver_return_codes'] == [0]
            assert all(check['pass'] for check in part['checks'])
            assert part['environment']['cc'] == [9, 0] and 'H100' in part['environment']['name']
            environments.append(dict(invocation=invocation, domain=domain, environment=part['environment']))
            for tag, binding in part['bindings'].items():
                assert tag in metas
                assert binding['loaded_SHA256'] == binding['cubin_SHA256'] == metas[tag]['cubin_SHA256']
            for cell in part['cells']:
                check = cell.get('complete_output_check')
                if check is not None:
                    assert check['pass']
                else:
                    # The original training runner stores this result in its
                    # checks list instead of repeating it inside each cell.
                    matches = [x for x in part['checks'] if cell['tag'] in [x['tag']]+x.get('aliases', [])
                               and x['mode'] == cell['mode'] and x['M'] == cell['M']
                               and x['pattern'] == cell['pattern']]
                    assert domain == 'training' and len(matches) == 1 and matches[0]['pass'], cell
                aliases = [cell['tag']]+cell.get('aliases', [])
                assert len(aliases) == len(set(aliases))
                for tag in aliases:
                    assert tag in metas and tag in part['bindings']
                    assert physical_identity(metas[tag]) == physical_identity(metas[cell['tag']]), (cell['tag'], tag)
                by_visit = {v['visit']: v for v in cell['visits']}
                assert len(cell['visits']) == len(by_visit) == visits and set(by_visit) == set(range(visits))
                medians = []
                for visit in range(visits):
                    value = by_visit[visit]; values = np.asarray(value['samples_us'], dtype=np.float64)
                    assert values.shape == (samples,) and np.all(np.isfinite(values)) and np.all(values > 0)
                    assert sorted(value['node_types']) == _nodes(domain, cell['mode'])
                    medians.append(float(np.median(values)))
                    total += samples; worker_samples += samples
                physical_cells += 1; worker_physical += 1
                for tag in aliases:
                    key = _cell_key(domain, cell, tag, groups)
                    assert key in expected and key not in seen, key
                    if domain == 'training' and 'kernel_M' in cell:
                        assert cell['kernel_M'] == expected[key]['kernel_M']
                    seen.add(key); observed[key][invocation] = medians
                    representatives[key][invocation] = cell['tag']
        assert seen == set(expected), (invocation, set(expected)-seen, seen-set(expected))
        worker_counts.append(dict(invocation=invocation, physical_cells=worker_physical,
                                  physical_samples=worker_samples, logical_conditions=len(seen)))
    assert set(observed) == set(expected)
    arrays = {}
    for key, values in observed.items():
        assert set(values) == set(range(1, workers+1)), key
        arrays[key] = np.asarray([values[i] for i in range(1, workers+1)], dtype=np.float64)
        assert arrays[key].shape == (workers, visits)
    assert total == physical_cells*visits*samples
    return expected, arrays, representatives, dict(raw_samples=total, physical_worker_cells=physical_cells,
        worker_counts=worker_counts, raw_sources=sources, environments=environments)


def _point_direction(value):
    return 'positive' if value > 1.e-12 else 'negative' if value < -1.e-12 else 'zero'


def _practical_direction(interval, threshold):
    if interval[0] > threshold:
        return 'gain_above_threshold'
    if interval[1] < -threshold:
        return 'loss_below_threshold'
    return 'not_resolved_beyond_threshold'


def historical_directions(rows, threshold):
    """Descriptive lineage of internally normalized gains, not cross-run timing."""
    sources = []; old = {}; matched = []; missing = []
    for panel in EXPECTED_PANELS:
        path = c.OUT/f'stage_{panel}/timing_analysis.json'
        assert path.exists(), path
        data = c.read(path); assert data['status'] == 'OK'
        sources.append(dict(panel=panel, path=str(path.relative_to(c.ROOT)), SHA256=c.sha(path.read_bytes())))
        lookup = {}
        for row in data['rows']:
            if panel == 'g3b':
                key = (row['geometry'], row['mode'], row['M'], row['tag'])
            elif panel == 'g2':
                key = (row['geometry'], row['batch'], row['length'], row['mode'], row['tag'])
            else:
                key = (row['group'], row['batch'], row['length'], row['mode'], row['tag'])
            assert key not in lookup; lookup[key] = row
        old[panel] = lookup
    for row in rows:
        panel = row['panel']; geometry = row['geometry']; variant = row['variant']
        if panel == 'g3b':
            legacy_tag = geometry+':'+variant
            key = (geometry, row['mode'], row['M'], legacy_tag)
        elif panel == 'g2':
            legacy_tag = geometry+':'+variant
            key = (geometry, row['batch'], row['length'], row['mode'], legacy_tag)
        else:
            group = row['group'][len(panel)+1:]
            if panel == 'g4b' and variant.startswith('B4:'):
                variant = 'old_B4:'+variant[len('B4:'):]
            legacy_tag = group+':'+variant
            key = (group, row['batch'], row['length'], row['mode'], legacy_tag)
        if key not in old[panel]:
            missing.append(dict(panel=panel, tag=row['tag'], legacy_key=list(key)))
            continue
        previous = old[panel][key]
        comparisons = {}
        for baseline in ['U', 'G0']:
            old_name = ('upstream' if baseline == 'U' else 'anchor') if panel == 'g2' else baseline
            gain = previous['gain_vs_'+old_name+'_percent']; ci = previous['gain_vs_'+old_name+'_CI95']
            new_gain = row['gain_vs_'+baseline+'_percent']; new_ci = row['gain_vs_'+baseline+'_CI95']
            comparisons[baseline] = dict(old_internal_gain_percent=gain, old_internal_gain_CI95=ci,
                new_internal_gain_percent=new_gain, new_internal_gain_CI95=new_ci,
                old_point_direction=_point_direction(gain), new_point_direction=_point_direction(new_gain),
                point_direction_retained=_point_direction(gain) == _point_direction(new_gain),
                old_practical_direction=_practical_direction(ci, threshold),
                new_practical_direction=_practical_direction(new_ci, threshold))
        matched.append(dict(panel=panel, tag=row['tag'], legacy_tag=legacy_tag,
            group=row['group'], batch=row['batch'], length=row['length'], mode=row['mode'], M=row['M'],
            comparisons=comparisons))
    assert not missing and len(matched) == 517, missing
    return dict(status='OK', rows=matched, historical_sources=sources, matched_conditions=len(matched),
        scope='Each gain uses its own contemporaneous U/G0 on the same worker/visit. '
              'Old G0 was not code-aligned; this is descriptive direction lineage, not a causal '
              'difference of historical and new absolute durations or a performance model refit.')


def main():
    path = protocol_path(); protocol = c.read(path); c.verify_sources(protocol)
    timing = protocol['timing']; assert timing['no_result_dependent_repeats']
    gate = c.read(gate_path()); payload_sha = c.sha(payload_path().read_bytes())
    assert gate['status'] == 'OK' and gate['protocol_SHA256'] == c.sha(c.encode(protocol))
    assert gate['payload_SHA256'] == payload_sha
    expected, arrays, representatives, audit = collect(protocol, gate, payload_sha)
    by_work = defaultdict(list)
    for key in expected:
        by_work[key[:-1]].append(key)
    rows = []; contrasts = []; threshold = timing['practical_threshold_percent']
    for work in sorted(by_work):
        keys = sorted(by_work[work]); named = {expected[key]['tag']: arrays[key] for key in keys}
        seed = timing['seed']+int(c.sha(c.encode(work))[:8], 16)
        rng = np.random.default_rng(seed); reps = timing['bootstrap_replicates']
        assert reps == 4096
        wi = rng.integers(0, 3, (reps, 3)); vi = rng.integers(0, 5, (reps, 3, 5))
        def resample(value):
            return value[wi[:, :, None], vi].mean((1, 2))
        for key in keys:
            spec = expected[key]; u_tag, g_tag, label = baseline_tags(spec)
            assert u_tag in named and g_tag in named
            value = arrays[key]; u, g = named[u_tag], named[g_tag]
            vb, ub, gb = resample(value), resample(u), resample(g)
            row = dict(spec, comparison_label=label, upstream_tag=u_tag, anchor_tag=g_tag,
                time_us=float(value.mean()), worker_visit_medians_us=value.tolist(),
                worker_means_us=value.mean(1).tolist(),
                physical_representative_tags=[representatives[key][i] for i in range(1, 4)],
                gain_vs_U_percent=float(100*(u.mean()-value.mean())/u.mean()),
                gain_vs_G0_percent=float(100*(g.mean()-value.mean())/g.mean()),
                gain_vs_U_CI95=np.percentile(100*(ub-vb)/ub, [2.5, 97.5]).tolist(),
                gain_vs_G0_CI95=np.percentile(100*(gb-vb)/gb, [2.5, 97.5]).tolist(),
                worker_gains_vs_U_percent=(100*(u.mean(1)-value.mean(1))/u.mean(1)).tolist(),
                worker_gains_vs_G0_percent=(100*(g.mean(1)-value.mean(1))/g.mean(1)).tolist(),
                bootstrap_seed=seed)
            for base in ['U', 'G0']:
                row['exploratory_practical_direction_vs_'+base] = _practical_direction(row['gain_vs_'+base+'_CI95'], threshold)
            rows.append(row)
        spec = expected[keys[0]]; prefix = spec['group']+':'
        if spec['panel'] == 'g4a':
            anchor, pv, qk, together = [named[prefix+n] for n in ['gluon_anchor', 'pv_only', 'qk_only', 'coordinated']]
            interaction = together-pv-qk+anchor
            contrasts.append(dict(panel='g4a', group=spec['group'], batch=spec['batch'], length=spec['length'], mode=spec['mode'],
                contrast='coordinated - pv_only - qk_only + anchor', interaction_us=float(interaction.mean()),
                interaction_CI95_us=np.percentile(resample(interaction), [2.5, 97.5]).tolist(), bootstrap_seed=seed))
        elif spec['panel'] == 'g4b':
            for name in ['upstream', 'gluon_anchor', 'target']:
                actual = named[prefix+'actual:'+name]; b4 = named[prefix+'B4:'+name]
                ab, bb = resample(actual), resample(b4)
                contrasts.append(dict(panel='g4b', group=spec['group'], batch=spec['batch'], length=spec['length'], mode=spec['mode'],
                    contrast=name+' actual-vs-B4', actual_minus_B4_us=float(actual.mean()-b4.mean()),
                    gain_percent=float(100*(b4.mean()-actual.mean())/b4.mean()),
                    gain_CI95=np.percentile(100*(bb-ab)/bb, [2.5, 97.5]).tolist(), bootstrap_seed=seed))
    assert len(rows) == 517 and dict(Counter(r['panel'] for r in rows)) == EXPECTED_PANELS
    assert len(contrasts) == 50
    history = historical_directions(rows, threshold)
    result = dict(status='OK', protocol_path=str(path.relative_to(c.ROOT)), protocol_SHA256=c.sha(c.encode(protocol)),
        payload_SHA256=payload_sha, compile_gate_SHA256=c.sha(gate_path().read_bytes()),
        analysis_source_SHA256=c.sha(Path(__file__).read_bytes()), logical_conditions=len(rows),
        logical_conditions_by_panel=EXPECTED_PANELS, logical_equivalent_sample_count=517*3*5*31,
        rows=rows, contrasts=contrasts, **audit,
        estimator='Mean of three worker x five visit medians; shared paired worker/visit hierarchical bootstrap, 4096 replicates.',
        gain_definition='100*(reference-candidate)/reference; U/G0 references belong to the same panel, workload and G4b actual/B4 label.',
        CI_scope='Exploratory pointwise 95% intervals; no multiplicity correction or selector fitting. '
                 'Physical aliases are expanded logically but their samples are counted once. '
                 'No statistical direction is a validator success requirement.')
    save(OUT/'timing_analysis.json', result)
    save(OUT/'historical_gain_directions.json', history)
    stream = io.StringIO()
    columns = list(rows[0]); writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    save(OUT/'all_timings.csv', stream.getvalue().encode())
    print('G8 timing analysis:', len(rows), 'logical conditions;', audit['raw_samples'], 'physical raw samples;', len(contrasts), 'contrasts')


if __name__ == '__main__':
    main()
