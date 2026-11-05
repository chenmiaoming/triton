"""Paired analysis of complete fixed-cadence, frozen-binary replacement matrices.

All contexts use the same worker/visit bootstrap draws. Exact aliases share
observations; U, G0 and same-program shams remain distinct measurements.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from experiments.tma_reduction_layout.phase11.operator_gpu import common as c


EXPECTED = {'nonreduction': 150, 'operators': 505, 'legacy': 594, 'anchors': 474}
SOURCE = Path(__file__).resolve()


def require(value, message):
    if not value:
        raise ValueError(message)


encode = c.encode


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not a.size:
        return dict(count=0, min=None, p05=None, median=None, mean=None, p95=None, max=None)
    return dict(count=int(a.size), min=float(a.min()), p05=float(np.percentile(a, 5)),
        median=float(np.median(a)), mean=float(a.mean()), p95=float(np.percentile(a, 95)), max=float(a.max()))


def estimate(point, boot=None, threshold=None):
    if point is None or boot is None:
        return dict(estimate=point, CI95=None, classification='incomplete')
    require(np.all(np.isfinite(boot)), 'Nonfinite bootstrap output')
    low, high = map(float, np.percentile(boot, [2.5, 97.5]))
    result = dict(estimate=float(point), CI95=[low, high])
    if threshold is not None:
        result['classification'] = ('practically_equivalent' if low >= -threshold and high <= threshold
            else 'practical_improvement' if low > threshold
            else 'practical_regression' if high < -threshold else 'inconclusive')
    return result


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), SHA256=c.sha(path.read_bytes()))


def result_paths(run_root, action):
    base = Path(run_root)/action
    candidates = list(base.glob('**/export/result.json'))
    if not candidates:
        candidates = list(base.glob('**/raw.json'))
    result = {}
    for path in sorted(candidates):
        value = c.read(path)
        if 'invocation' not in value or 'suite' not in value:
            continue
        invocation = int(value['invocation'])
        require(invocation not in result, 'Duplicate invocation result: '+str(path))
        result[invocation] = (path, value)
    return result


def logical_role(suite, tag, physical_role):
    if physical_role.endswith('_repeat'):
        return physical_role
    if suite == 'nonreduction':
        return 'U' if tag.endswith(':U') else 'G0' if tag.endswith(':G0') else 'GL'
    if suite == 'operators':
        name = tag.split(':')[-1]
        return 'U' if name == 'upstream' else 'G0' if name == 'gluon_anchor' else 'GL'
    if suite == 'legacy' and ':control:' in tag:
        return 'control'
    return physical_role


def baseline_tags(suite, tag, logical, physical):
    if suite == 'operators':
        parts = tag.split(':')
        prefix = parts[0]
        if len(parts) > 1 and parts[1] in ('actual', 'B4'):
            prefix += ':'+parts[1]
        return prefix+':upstream', prefix+':gluon_anchor'
    if suite == 'nonreduction':
        bases = {role: [name for name, entry in logical.items() if entry['role'] == role]
                 for role in ('U', 'G0')}
        return (bases['U'][0] if len(bases['U']) == 1 else None,
                bases['G0'][0] if len(bases['G0']) == 1 else None)
    bases = [entry['tag'] for entry in physical.values() if entry['role'] == 'default']
    return (bases[0] if len(bases) == 1 else None), None


def _telemetry(path, raw):
    entry = raw.get('telemetry')
    if not entry:
        return dict(available=False)
    source = path.parent/entry['path']
    require(c.sha(source.read_bytes()) == entry['SHA256'], 'Telemetry SHA mismatch')
    fields = ('sm_clock_mhz', 'graphics_clock_mhz', 'memory_clock_mhz',
        'power_usage_mw', 'power_limit_mw', 'temperature_c',
        'gpu_utilization_percent', 'pstate', 'clock_event_reasons')
    values = {name: [] for name in fields}
    codes = {name: Counter() for name in fields}
    contexts = Counter()
    count = 0
    with gzip.open(source, 'rt') as handle:
        for line in handle:
            row = json.loads(line)
            count += 1
            contexts[row.get('context', {}).get('phase', 'unknown')] += 1
            for name in fields:
                field = row.get(name, {})
                codes[name][str(field.get('rc'))] += 1
                if field.get('rc') == 0 and isinstance(field.get('value'), (int, float)):
                    values[name].append(field['value'])
    require(count == entry['rows'], 'Telemetry row count mismatch')
    return dict(available=True, source=reference(source), rows=count,
        phases=dict(contexts), return_codes={name: dict(counts) for name, counts in codes.items()},
        metrics={name: summary(a) for name, a in values.items()},
        observed_clock_event_masks=dict(Counter(map(str, values['clock_event_reasons']))),
        scope='Sampled nominal clocks; hardware power/utilization averaging may cross observation boundaries. These are not per-target fixed-clock or causal DVFS measurements.')


def collect(protocol, suite, run_root):
    timing = protocol['timing']
    workers, visits, samples = [int(timing[k]) for k in ('workers', 'visits', 'samples')]
    require((workers, visits, samples) == (3, 10, 31), 'Unexpected frozen sampling matrix')
    require(suite in EXPECTED and suite in protocol['suites'], 'Unknown suite')
    protocol_sha = c.sha(c.encode(protocol))
    found = result_paths(run_root, 'timing')
    expected_invocations = set(range(1, workers+1))
    require(set(found) <= expected_invocations, 'Unexpected formal invocation')
    cells, contexts, checks, environments, inputs, telemetry = {}, {}, [], [], [], []
    problems = []
    counts = {}
    raw_sample_count = 0
    for invocation in range(1, workers+1):
        if invocation not in found:
            problems.append(dict(invocation=invocation, reason='missing_worker'))
            continue
        path, raw = found[invocation]
        require(raw['suite'] == suite and raw['protocol_SHA256'] == protocol_sha, 'Worker suite/protocol mismatch')
        require(raw['payload_SHA256'] == protocol['suites'][suite]['payload_SHA256'], 'Worker payload mismatch')
        inputs.append(reference(path))
        if raw['status'] != 'OK':
            problems.append(dict(invocation=invocation, reason='failed_worker', error=raw.get('error')))
        if raw.get('compile_calls', 0) or raw.get('native_rebuilds', 0) or raw.get('Triton_imported'):
            problems.append(dict(invocation=invocation, reason='unexpected_compilation'))
        if raw.get('driver_return_codes') != [0] or raw.get('cleanup_errors'):
            problems.append(dict(invocation=invocation, reason='execution_or_cleanup_error'))
        environments.append(dict(invocation=invocation, environment=raw.get('environment')))
        checks.extend(dict(invocation=invocation, **check) for check in raw.get('checks', []))
        telemetry.append(dict(invocation=invocation, **_telemetry(path, raw)))
        seen = set()
        worker_count = 0
        for shard_ref in raw.get('shards', []):
            shard_path = path.parent/shard_ref['path']
            require(c.sha(shard_path.read_bytes()) == shard_ref['SHA256'], 'Shard SHA mismatch')
            shard = c.read(shard_path)
            context = shard['context']
            context_id = context['id']
            if context_id in contexts:
                require(c.encode(contexts[context_id]) == c.encode(context), 'Context differs between workers')
            contexts[context_id] = context
            for cell in shard['cells']:
                key = (context_id, cell['tag'])
                require(key not in seen, 'Duplicate worker/cell')
                seen.add(key)
                metadata = dict(tag=cell['tag'], aliases=cell.get('aliases', []),
                    role=cell['role'], sham_of=cell.get('sham_of'))
                if key not in cells:
                    cells[key] = dict(**metadata, array=np.full((workers, visits), np.nan),
                        sample_summaries={}, gap_summaries={}, host_gap_below_requested=0)
                item = cells[key]
                require(all(item[name] == value for name, value in metadata.items()), 'Cell aliases/roles differ')
                ids = [v['visit'] for v in cell['visits']]
                if sorted(ids) != list(range(visits)):
                    problems.append(dict(invocation=invocation, context=context_id,
                        tag=cell['tag'], reason='incomplete_visits'))
                require(len(ids) == len(set(ids)), 'Duplicate visits')
                all_samples, gaps = [], []
                for visit in cell['visits']:
                    vi = int(visit['visit'])
                    require(0 <= vi < visits, 'Invalid visit identifier')
                    values = np.asarray(visit['samples_us'], dtype=np.float64)
                    require(values.ndim == 1 and np.all(np.isfinite(values)) and np.all(values > 0), 'Invalid kernel timing')
                    if len(values) != samples:
                        problems.append(dict(invocation=invocation, context=context_id,
                            tag=cell['tag'], visit=vi, reason='incomplete_samples'))
                    else:
                        item['array'][invocation-1, vi] = np.median(values)
                    all_samples.extend(values.tolist())
                    gap = np.asarray(visit['host_gaps_ns'], dtype=np.float64)
                    require(gap.shape == values.shape and np.all(np.isfinite(gap)) and np.all(gap >= 0), 'Invalid host gaps')
                    gaps.extend((gap/1000).tolist())
                    item['host_gap_below_requested'] += int(np.sum(gap < float(timing['host_gap_seconds'])*1e9))
                    raw_sample_count += len(values)
                item['sample_summaries'][str(invocation)] = summary(all_samples)
                item['gap_summaries'][str(invocation)] = summary(gaps)
                worker_count += 1
        counts[str(invocation)] = worker_count
    spec = protocol['suites'][suite]
    expected_count = int(spec.get('expected_physical_cells', spec.get('expected', {}).get('physical', EXPECTED[suite])))
    for invocation in range(1, workers+1):
        if counts.get(str(invocation), 0) != expected_count:
            problems.append(dict(invocation=invocation, reason='incomplete_matrix',
                observed=counts.get(str(invocation), 0), expected=expected_count))
    failed_checks = [dict(invocation=x['invocation'], context=x.get('context'), tag=x.get('tag'), phase=x.get('phase'))
        for x in checks if x.get('pass') is not True]
    if failed_checks:
        problems.append(dict(reason='failed_correctness_checks', checks=failed_checks))
    uuids = [x['environment'].get('gpu_uuid') for x in environments if x.get('environment')]
    if len(uuids) == workers and len(set(uuids)) != workers:
        problems.append(dict(reason='nonunique_GPU_UUIDs'))
    eligibility = result_paths(run_root, 'check')
    eligibility_rows = []
    for invocation, (path, raw) in eligibility.items():
        inputs.append(reference(path))
        eligibility_rows.append(dict(invocation=invocation, status=raw['status'],
            checks=len(raw.get('checks', [])), source=reference(path)))
    if not eligibility_rows or any(x['status'] != 'OK' for x in eligibility_rows):
        problems.append(dict(reason='missing_or_failed_eligibility'))
    return cells, contexts, dict(status='OK' if not problems else 'INCOMPLETE',
        problems=problems, workers=workers, visits=visits, samples=samples,
        physical_cells_per_worker=counts, expected_physical_cells_per_worker=expected_count,
        distinct_contexts=len(contexts), raw_sample_count=raw_sample_count,
        complete_correctness_checks=len(checks), eligibility=eligibility_rows,
        environments=environments, telemetry=telemetry, inputs=inputs)


def old_key(suite, context, tag):
    if suite == 'operators':
        return (context['domain'], context['group'], context.get('batch'), context.get('length'),
                context['mode'], context.get('M'), tag)
    if suite == 'nonreduction':
        return (context.get('geometry', context.get('case')), tag)
    if suite == 'anchors' and tag is not None and tag.split('|', 1)[0] in ('c2d', 'd1'):
        tag = tag.split('|', 1)[1]
    return (tag,)


def _old_paths(suite):
    nonred = c.ROOT/'experiments/tma_reduction_layout/results/phase11/nonreduction'
    if suite == 'nonreduction':
        return [(stage, path) for stage in ('stage_n2', 'stage_n4', 'stage_n5_timing')
                for path in sorted((nonred/stage/'timing/attempt_001').glob('invocation_*/export/result.json'))]
    if suite == 'operators':
        return [('stage_g8', path) for path in sorted((c.OUT/'stage_g8/timing').glob('invocation_*/raw.json'))]
    stages = ['stage_m3'] if suite == 'legacy' else ['stage_c2d', 'stage_d1']
    base = c.ROOT/'experiments/tma_reduction_layout/results/phase11'
    return [(stage, path) for stage in stages for path in sorted((base/stage/'timing').glob('invocation_*/raw.json'))]


def historical(suite, refs=None):
    observed = defaultdict(lambda: defaultdict(list))
    inputs = []
    paths = _old_paths(suite)
    if refs is not None:
        paths = []
        for item in refs:
            path = c.ROOT/item['path']
            require(c.sha(path.read_bytes()) == item['SHA256'], 'Historical source SHA mismatch')
            stage = next(part for part in path.parts if part.startswith('stage_'))
            paths.append((stage, path))
    for stage, path in paths:
        raw = c.read(path)
        require(raw['status'] == 'OK', 'Historical timing is not accepted: '+str(path))
        inputs.append(dict(stage=stage, **reference(path)))
        parts = [('inference', raw['inference']), ('training', raw['training'])] if suite == 'operators' else [(None, raw)]
        for domain, part in parts:
            for cell in part['cells']:
                medians = [float(np.median(v['samples_us'])) for v in cell['visits']]
                if suite == 'operators':
                    context = dict(domain=domain, group=cell.get('group', cell.get('geometry')),
                        batch=cell.get('batch'), length=cell.get('length'), mode=cell['mode'], M=cell.get('M'))
                    tags = [cell['tag'], *cell.get('aliases', [])]
                elif suite == 'nonreduction':
                    context = dict(geometry=cell['case'])
                    tags = [cell['tag'], *cell.get('aliases', [])]
                else:
                    context = {}
                    tags = [cell['cell_id'], *cell.get('aliases', [])]
                for tag in tags:
                    observed[old_key(suite, context, tag)][stage].extend(medians)
    return {key: {stage: float(np.mean(values)) for stage, values in stages.items()}
            for key, stages in observed.items()}, inputs


def _gain(candidate, baseline, boots, threshold):
    if candidate is None or baseline is None:
        return estimate(None)
    if candidate not in boots or baseline not in boots:
        return estimate(None)
    cp, cb = boots[candidate]
    bp, bb = boots[baseline]
    return estimate(100*(bp-cp)/bp, 100*(bb-cb)/bb, threshold)


def _anchor_services(contexts, cells, wi, vi, threshold, old=None, validated=True):
    old = old or {}
    grouped = defaultdict(list)
    for (context_id, tag), item in cells.items():
        context = contexts[context_id]
        grid = context.get('grid')
        if not isinstance(grid, (int, float)):
            continue
        plain = tag.split('|', 1)[1] if tag.split('|', 1)[0] in ('c2d', 'd1') else tag
        original = plain.split('|q', 1)[0]
        geometry = context.get('geometry', original.split(':')[0])
        grouped[(geometry, original, item['role'])].append((float(grid), item['array'], context_id, tag))
    rows = []
    for (geometry, tag, role), entries in sorted(grouped.items()):
        if len({x[0] for x in entries}) != 3 or any(not np.all(np.isfinite(x[1])) for x in entries):
            continue
        entries.sort(key=lambda x: x[0])
        x = np.asarray([row[0] for row in entries])
        centered = x-x.mean()
        weights = centered/np.sum(centered**2)
        slope = sum(weight*row[1] for weight, row in zip(weights, entries))
        service = slope*132000
        boot = service[wi[:, :, None], vi].mean((1, 2))
        point = float(service.mean())
        history = {}
        previous = [old.get(old_key('anchors', contexts[row[2]], row[3]), {}) for row in entries]
        if previous:
            for stage in set.intersection(*(set(value) for value in previous)):
                history[stage] = float(np.dot(weights, [value[stage] for value in previous])*132000)
        rows.append(dict(geometry=geometry, tag=tag, role=role,
            grids=x.astype(int).tolist(), context_ids=[row[2] for row in entries],
            service_ns_per_tile_per_SM=estimate(point, boot),
            worker_visit_services_ns=service.tolist(),
            old_service_ns_per_tile_per_SM=history, _boot=boot))
    by_geometry = defaultdict(list)
    for row in rows:
        by_geometry[row['geometry']].append(row)
    for geometry, group in by_geometry.items():
        defaults = [r for r in group if r['role'] == 'default']
        repeats = [r for r in group if r['role'] == 'default_repeat']
        control = estimate(None)
        if len(defaults) == len(repeats) == 1:
            default, repeat = defaults[0], repeats[0]
            value = default['service_ns_per_tile_per_SM']['estimate']
            if value > 0 and np.all(default['_boot'] > 0):
                control = estimate(100*(value-repeat['service_ns_per_tile_per_SM']['estimate'])/value,
                    100*(default['_boot']-repeat['_boot'])/default['_boot'], threshold)
        for row in group:
            row['default_service_sham_equivalence'] = control
            if len(defaults) == 1 and row['role'] != 'control':
                default = defaults[0]
                a, b = row['service_ns_per_tile_per_SM']['estimate'], default['service_ns_per_tile_per_SM']['estimate']
                if b > 0 and np.all(default['_boot'] > 0):
                    row['gain_vs_default_percent'] = estimate(100*(b-a)/b,
                        100*(default['_boot']-row['_boot'])/default['_boot'], threshold)
                    row['default_tag'] = default['tag']
                    old_gains = {}
                    for stage, value in row['old_service_ns_per_tile_per_SM'].items():
                        old_default = default['old_service_ns_per_tile_per_SM'].get(stage)
                        if old_default is not None and old_default > 0:
                            old_gains[stage] = 100*(old_default-value)/old_default
                    row['old_gain_vs_default_percent'] = old_gains
                    row['new_minus_old_gain_pp'] = {stage:row['gain_vs_default_percent']['estimate']-value
                        for stage,value in old_gains.items()}
            gain = row.get('gain_vs_default_percent', {})
            row['strict_practical_improvement'] = bool(validated and row['role'] not in ('default', 'control', 'default_repeat')
                and control.get('classification') == 'practically_equivalent'
                and gain.get('CI95') is not None and gain['CI95'][0] > threshold)
        for row in group:
            row.pop('_boot', None)
    return rows


def analyze(protocol, suite, run_root):
    run_root = Path(run_root).resolve()
    cells, contexts, audit = collect(protocol, suite, run_root)
    timing = protocol['timing']
    reps = int(timing['bootstrap_replicates'])
    require(reps == 8192, 'Unexpected bootstrap count')
    threshold = float(timing['practical_threshold_percent'])
    require(threshold == 3, 'Unexpected practical threshold')
    rng = np.random.default_rng(int(timing['seed']))
    wi = rng.integers(0, 3, (reps, 3))
    vi = rng.integers(0, 10, (reps, 3, 10))
    draw_sha = c.sha(wi.tobytes()+vi.tobytes())
    old_refs = protocol['suites'][suite].get('old_raw_refs')
    old, old_inputs = historical(suite, refs=old_refs) if old_refs is not None else historical(suite)
    selected = {}
    if suite == 'nonreduction':
        selected_ref = protocol['suites'][suite].get('old_frozen_selection')
        path = (c.ROOT/selected_ref['path']) if selected_ref else c.ROOT/'experiments/tma_reduction_layout/results/phase11/nonreduction/stage_n4/selection/decision.json'
        if selected_ref:
            require(c.sha(path.read_bytes()) == selected_ref['SHA256'], 'Frozen N4 selection SHA mismatch')
        selected = {x['case']: x['candidate'] for x in c.read(path)['choices']}
        old_inputs.append(dict(stage='stage_n4_selection', **reference(path)))
    rows, controls, shams, context_summaries, physical_rows = [], [], [], [], []
    for context_id, context in sorted(contexts.items()):
        physical = {tag: item for (cid, tag), item in cells.items() if cid == context_id}
        logical, boots = {}, {}
        for tag, item in physical.items():
            array = item['array']
            complete = bool(np.all(np.isfinite(array)))
            if complete:
                boots[tag] = (float(array.mean()), array[wi[:, :, None], vi].mean((1, 2)))
            worker_variation = []
            for worker in range(3):
                values = array[worker]
                valid = values[np.isfinite(values)]
                worker_variation.append(dict(invocation=worker+1, visit_medians_us=values.tolist() if np.all(np.isfinite(values)) else None,
                    summary=summary(valid), spread_percent=float(100*(valid.max()/valid.min()-1)) if valid.size else None))
            physical_rows.append(dict(context_id=context_id, tag=tag, aliases=item['aliases'], role=item['role'],
                complete=complete, worker_visit_medians_us=array.tolist() if complete else None,
                time_us=estimate(*boots[tag]) if complete else estimate(None),
                worker_variation=worker_variation, raw_sample_summaries=item['sample_summaries'],
                host_gap_us=item['gap_summaries'], host_gap_below_requested=item['host_gap_below_requested']))
            for name in [tag, *item['aliases']]:
                require(name not in logical, 'Duplicate logical alias in context')
                logical[name] = dict(physical=tag, role=logical_role(suite, name, item['role']))
        control_cache = {}
        for tag, entry in sorted(logical.items()):
            role = entry['role']
            representative = entry['physical']
            u, g0 = baseline_tags(suite, tag, logical, physical)
            up = logical[u]['physical'] if u in logical else None
            gp = logical[g0]['physical'] if g0 in logical else None
            gain_u = _gain(representative, up, boots, threshold)
            gain_g0 = _gain(representative, gp, boots, threshold) if suite in ('operators', 'nonreduction') else None
            baseline_pair = (u, g0)
            if baseline_pair not in control_cache:
                matching = [(name, x) for name, x in physical.items()
                            if x.get('sham_of') == up and x['role'].endswith('_repeat')]
                sham_tag = matching[0][0] if len(matching) == 1 else None
                sham_gain = _gain(sham_tag, up, boots, threshold)
                baseline_gain = _gain(gp, up, boots, threshold) if g0 else None
                if suite in ('operators', 'nonreduction') and (up is None or gp is None):
                    state = 'inconclusive_missing_U_G0'
                elif up is None:
                    state = 'no_paired_reference'
                elif sham_gain.get('classification') != 'practically_equivalent':
                    state = 'inconclusive_sham'
                elif g0 and baseline_gain.get('classification') != 'practically_equivalent':
                    state = 'inconclusive_U_G0'
                else:
                    state = 'validated'
                control = dict(context_id=context_id, U_tag=u if suite in ('operators','nonreduction') else None,
                    G0_tag=g0, G0_physical=gp, reference_tag=u, reference_physical=up, sham_tag=sham_tag,
                    status=state, U_G0_equivalence=baseline_gain, sham_equivalence=sham_gain)
                control_cache[baseline_pair] = control
                if up is not None:
                    controls.append(control)
                    if sham_tag:
                        shams.append(dict(context_id=context_id, reference_tag=u,
                            reference_physical=up, sham_tag=sham_tag, result=sham_gain))
            control = control_cache[baseline_pair]
            point, boot = boots.get(representative, (None, None))
            historical_values = old.get(old_key(suite, context, tag), {})
            old_comparisons = []
            old_u = old.get(old_key(suite, context, u), {}) if u else {}
            old_g = old.get(old_key(suite, context, g0), {}) if g0 else {}
            for stage, value in historical_values.items():
                item = dict(stage=stage, old_time_us=value,
                    new_vs_old_time_change_percent=100*(point/value-1) if point is not None else None)
                if role != 'control' and stage in old_u:
                    item['old_gain_vs_reference_percent'] = 100*(old_u[stage]-value)/old_u[stage]
                    item['new_minus_old_gain_pp'] = (gain_u['estimate']-item['old_gain_vs_reference_percent']) if gain_u['estimate'] is not None else None
                if stage in old_g:
                    item['old_gain_vs_G0_percent'] = 100*(old_g[stage]-value)/old_g[stage]
                old_comparisons.append(item)
            strict = (audit['status'] == 'OK' and role in ('GL', 'candidate') and control['status'] == 'validated'
                and gain_u.get('CI95') is not None and gain_u['CI95'][0] > threshold
                and (gain_g0 is None or (gain_g0.get('CI95') is not None and gain_g0['CI95'][0] > threshold)))
            if role == 'control':
                strict = False
            row = dict(context_id=context_id, context=context, tag=tag,
                representative_tag=representative, role=role,
                independent_observation=(tag == representative),
                control_status=control['status'] if role != 'control' else 'not_a_layout_comparison',
                time_us=estimate(point, boot), gain_vs_U_percent=gain_u if suite in ('operators','nonreduction') else None,
                gain_vs_G0_percent=gain_g0,
                gain_vs_reference_percent=gain_u if suite not in ('operators','nonreduction') and role != 'control' else None,
                reference_tag=u if role != 'control' else None, G0_tag=g0, strict_practical_improvement=bool(strict),
                old_comparisons=old_comparisons,
                predetermined_N4_choice=selected.get(context.get('geometry', context.get('case'))) == tag)
            rows.append(row)
        candidates = [r for r in rows if r['context_id'] == context_id
            and r['role'] in ('GL','candidate') and r['time_us']['estimate'] is not None]
        best = min(candidates, key=lambda r: r['time_us']['estimate']) if candidates else None
        context_summaries.append(dict(context_id=context_id, context=context,
            logical_rows=sum(r['context_id'] == context_id for r in rows), physical_rows=len(physical),
            descriptive_minimum_candidate_tag=best['tag'] if best else None,
            minimum_candidate_time_us=best['time_us'] if best else None,
            minimum_candidate_gain_percent=(best['gain_vs_U_percent'] or best['gain_vs_reference_percent']) if best else None,
            control_status=best['control_status'] if best else None,
            strict_improvements=[r['tag'] for r in candidates if r['strict_practical_improvement']]))
    services = _anchor_services(contexts, cells, wi, vi, threshold, old=old,
        validated=audit['status'] == 'OK') if suite == 'anchors' else []
    drift = {}
    for role in sorted({row['role'] for row in physical_rows}):
        spreads = [worker['spread_percent'] for row in physical_rows if row['role'] == role
                   for worker in row['worker_variation'] if worker['spread_percent'] is not None]
        drift[role] = dict(worker_cells=len(spreads), spread_percent=summary(spreads),
            spread_above_3_percent=sum(value > 3 for value in spreads),
            spread_above_10_percent=sum(value > 10 for value in spreads))
    gap_stats = [value for row in physical_rows for value in row['host_gap_us'].values() if value['count']]
    cadence = dict(requested_minimum_host_gap_us=float(timing['host_gap_seconds'])*1e6,
        observations_below_requested=sum(row['host_gap_below_requested'] for row in physical_rows),
        cell_worker_median_gap_us=summary([value['median'] for value in gap_stats]),
        minimum_observed_gap_us=min((value['min'] for value in gap_stats), default=None),
        maximum_observed_gap_us=max((value['max'] for value in gap_stats), default=None),
        boundary='Host gaps exclude GPU-side primes, which execute after submission and before target events. Long host gaps remain in data; no sample is filtered.')
    source_inputs = dict(analysis_source=reference(SOURCE), run_root=str(run_root),
        canonical_protocol_SHA256=c.sha(c.encode(protocol)), historical_raw=old_inputs,
        current_raw=audit.pop('inputs'))
    return dict(status=audit['status'], suite=suite, protocol_SHA256=c.sha(c.encode(protocol)),
        payload_SHA256=protocol['suites'][suite]['payload_SHA256'], inputs=source_inputs,
        audit=audit, bootstrap=dict(replicates=reps, seed=int(timing['seed']), draw_SHA256=draw_sha,
            scheme='Shared worker/visit draws across every context and comparison; percentile pointwise CI95, without multiplicity correction.'),
        physical_rows=physical_rows, rows=rows, controls=controls, shams=shams,
        context_summaries=context_summaries, anchors_service_rows=services,
        baseline_and_candidate_visit_drift=drift, cadence=cadence,
        summary=dict(logical_rows=len(rows), physical_rows=len(physical_rows),
            U_G0_pairs=sum(x['G0_tag'] is not None for x in controls),
            U_G0_practically_equivalent=sum(x['U_G0_equivalence'] is not None and
                x['U_G0_equivalence'].get('classification') == 'practically_equivalent' for x in controls),
            sham_comparisons=len(shams), sham_practically_equivalent=sum(
                x['result'].get('classification') == 'practically_equivalent' for x in shams),
            physical_sham_rows=sum(row['role'].endswith('_repeat') for row in physical_rows),
            physical_U_G0_pairs=len({(row['context_id'],row['reference_physical'],row['G0_physical'])
                for row in controls if row['G0_physical'] is not None}),
            strict_practical_improvements=sum(r['strict_practical_improvement'] for r in rows),
            candidate_rows=sum(r['role'] in ('GL','candidate') for r in rows)),
        interpretation=[
            'The new estimand is a single target semantic iteration after eight GPU-side primes, with input restoration outside its events and a fixed minimum host gap. It is not the historical unrestricted-cadence estimand or an idle-first-use measurement.',
            'U/G0 and same-program sham failures remain inconclusive controls; no sample/worker is dropped and no favorable equivalence is required by analysis.',
            'Exact aliases reuse a physical observation and never count as additional independent data. Each suite has three GPU allocations, not raw_sample_count independent GPUs.',
            'Every logical candidate is reported. Context minima are descriptive post-hoc summaries, not a pre-registered selector or a multiplicity-controlled ranking.',
            'Old/new comparisons are separate-cohort descriptive means and within-run gains. They are not paired causal differences, do not subtract a fixed idle penalty, and do not relabel old NCU reports.',
            'No model is refit and seen D1 geometries are not called new blind tests. Legacy and anchor defaults are explicit Gluon references, not invented source-aligned Triton U/G0 baselines.'])


def cell_text(value):
    if not value or value.get('estimate') is None:
        return 'unavailable'
    interval = value.get('CI95')
    text = f"{value['estimate']:.3f}"
    return text+f" [{interval[0]:.3f},{interval[1]:.3f}]" if interval is not None else text


def render(result):
    lines = [f"# Fixed-cadence replacement: {result['suite']}", '',
        f"Data status: **{result['status']}**. Three workers × ten visits × 31 samples per physical condition; {result['audit']['raw_sample_count']:,} target observations.",
        'Each graph runs eight untimed semantic iterations, then exactly one timed iteration. Original pipeline zeroes remain inside each iteration; any input restoration precedes the target events. Between observations the requested minimum host gap is 1 ms; actual gaps remain recorded.', '',
        f"U/G0 equivalence: {result['summary']['U_G0_practically_equivalent']}/{result['summary']['U_G0_pairs']}. Same-program sham equivalence: {result['summary']['sham_practically_equivalent']}/{result['summary']['sham_comparisons']}.",
        f"Those are logical workload comparisons; exact aliases can share data. Distinct physical U/G0 pairs: {result['summary']['physical_U_G0_pairs']}; distinct physical sham labels per worker: {result['summary']['physical_sham_rows']}. They are not additional independent GPUs.",
        'Equivalence requires the full paired CI95 to be inside ±3%. A candidate is marked strictly improved only when its fresh U and G0 lower bounds both exceed +3%, and the relevant controls pass. Legacy/anchor comparisons instead use their explicitly named Gluon reference.', '',
        'All logical rows, physical aliases, visit medians, host-gap summaries, old/new descriptive comparisons and control decisions are retained in `analysis.json` and `timing.csv`. No unfavorable or inconclusive control is removed.', '',
        f"Recorded host gaps below the requested minimum: {result['cadence']['observations_below_requested']}. Observed minimum/maximum: {result['cadence']['minimum_observed_gap_us']} / {result['cadence']['maximum_observed_gap_us']} µs. GPU-side primes are outside these host-gap durations and outside target timing.", '',
        '## Control results', '', '| Context | U/reference | G0 vs U gain % [CI95] | Same-program repeat gain % [CI95] | Status |', '|---|---|---:|---:|---|']
    for row in result['controls']:
        lines.append(f"| {row['context_id']} | {row['reference_tag']} | {cell_text(row['U_G0_equivalence'])} | {cell_text(row['sham_equivalence'])} | {row['status']} |")
    lines += ['', '## Every context', '',
        'The minimum candidate below is selected retrospectively for a compact inventory, not used as a deployed model. Full candidate and losing-layout data remain in the CSV/JSON.', '',
        '| Context | Physical rows including sham | Descriptive minimum candidate | Target time µs [CI95] | Gain vs paired reference % [CI95] | Control |',
        '|---|---:|---|---:|---:|---|']
    for row in result['context_summaries']:
        lines.append(f"| {row['context_id']} | {row['physical_rows']} | {row['descriptive_minimum_candidate_tag'] or '—'} | {cell_text(row['minimum_candidate_time_us'])} | {cell_text(row['minimum_candidate_gain_percent'])} | {row['control_status'] or 'no layout comparison'} |")
    selected = [r for r in result['rows'] if r['predetermined_N4_choice']]
    if selected:
        lines += ['', '## Previously frozen N4 choices', '',
            'These candidates were fixed by the earlier N4 protocol. They are not reselected using the new matrix.', '',
            '| Case | Frozen candidate | Gain vs U % [CI95] | Gain vs G0 % [CI95] | Strict dual-control >3% |',
            '|---|---|---:|---:|---|']
        for row in selected:
            lines.append(f"| {row['context_id']} | {row['tag']} | {cell_text(row['gain_vs_U_percent'])} | {cell_text(row['gain_vs_G0_percent'])} | {row['strict_practical_improvement']} |")
    if result['anchors_service_rows']:
        lines += ['', '## Anchor service slopes', '',
            'OLS is applied to the same worker/visit medians across three grids. The slope in µs/CTA is multiplied by 132 SM × 1000 to obtain ns/tile/SM. Shared bootstrap draws preserve paired grid/default differences; negative or uncertain slopes are not truncated.', '',
            '| Geometry | Program | Role | Service ns/tile/SM [CI95] | Gain vs default % [CI95] |', '|---|---|---|---:|---:|']
        for row in result['anchors_service_rows']:
            lines.append(f"| {row['geometry']} | {row['tag']} | {row['role']} | {cell_text(row['service_ns_per_tile_per_SM'])} | {cell_text(row.get('gain_vs_default_percent'))} |")
    lines += ['', '## Interpretation boundaries', '']
    lines.extend('- '+text for text in result['interpretation'])
    lines += ['', 'Telemetry retains return codes and compressed raw samples. Its nominal clocks and hardware averaging windows do not establish per-kernel fixed frequency or identify a causal DVFS mechanism.', '']
    if result['audit']['problems']:
        lines += ['Data completeness/correctness problems are retained in `audit.problems`; this report is not a completed validated replacement.', '']
    return '\n'.join(lines)


def csv_bytes(result):
    stream = io.StringIO(newline='')
    fields = ['context_id','tag','representative_tag','role','control_status','time_us','time_CI95',
        'gain_vs_U_percent','gain_vs_G0_percent','gain_vs_reference_percent',
        'strict_practical_improvement','predetermined_N4_choice','old_comparisons']
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for row in result['rows']:
        output = {name: row.get(name) for name in fields}
        output['time_us'] = row['time_us']['estimate']
        output['time_CI95'] = row['time_us']['CI95']
        writer.writerow({name: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                         for name, value in output.items()})
    return stream.getvalue().encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--suite', choices=sorted(EXPECTED), required=True)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = c.read(args.protocol)
    result = analyze(protocol, args.suite, args.run_root)
    outputs = {'analysis.json': encode(result), 'timing.csv': csv_bytes(result),
               'REPORT.md': render(result).encode()}
    require(not any((args.output/name).exists() for name in outputs), 'Refusing to overwrite analysis outputs')
    for name, blob in outputs.items():
        c.write_once(args.output/name, blob)
    print(json.dumps(dict(status=result['status'], suite=args.suite,
        raw_samples=result['audit']['raw_sample_count'], output=str(args.output)), sort_keys=True))


if __name__ == '__main__':
    main()
