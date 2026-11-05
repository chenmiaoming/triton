"""Calibration tables bound to immutable raw C2 evidence; never read D1 timing."""
from collections import defaultdict
import statistics as st
from experiments.tma_reduction_layout.phase11.c32 import common as c
from experiments.tma_reduction_layout.phase11.c31.data import development as old_development
from experiments.tma_reduction_layout.phase11.c3.static import candidates


def case_for(meta):
    return next(row for row in candidates(meta['M'], meta['N'], meta['num_warps'])
                if row['id'] == meta['layout_id'])


def load():
    a = c.read(c.prior.OUT/'analysis.json')
    gate = c.read(c.prior.OUT/'artifact_gate.json')
    _, old_workers, _ = old_development()
    steady = [{**row, 'target_ns': row['service_ns']} for row in old_workers]
    full, control, scaffold = [], [], []
    for row in a['services']:
        meta = gate['metadata'][row['tag']]
        if meta['kind'] == 'composition' and meta['family'] == 'MC':
            case = case_for(meta)
            steady.append({**row, 'case': case, 'layout': case['id'],
                'geometry': f'{case["BM"]}x{case["BN"]}w{case["W"]}',
                'worker': row['invocation'], 'target_ns': row['service_ns']})
    for row in a['oneshot']:
        meta = gate['metadata'][row['tag']]
        m, n, w = row['geometry']
        item = {**row, 'geometry_key': f'{m}x{n}w{w}', 'worker': row['invocation'],
                'geometry_vector': [m, n, w],
                'target_ns': row['tile_slope_ns'],
                'grid_slope_nonlinearity_ns': 1000*row['grid_fit']['max_abs_residual']/max(row['grid_fit']['xs']),
                'cohort': 'C2c'}
        if meta['family'] in ('ONE_FULL', 'ORIGINAL_FULL'):
            case = case_for(meta)
            full.append({**item, 'case': case, 'layout': case['id'], 'geometry': item['geometry_key']})
        elif meta['family'] in ('ONE_TRANSFER', 'ORIGINAL_TRANSFER'):
            control.append(item)
        elif meta['family'] == 'SCAFFOLD':
            scaffold.append(item)
    # Alias conditions are the same physical experiment, not independent data.
    def unique(rows, has_case):
        seen = set()
        result = []
        for row in sorted(rows, key=lambda x: x['q_condition'] != 'natural'):
            identity = (row['layout'] if has_case else row['geometry_key'], row['q'],
                        row['cohort'], row['worker'], row['GPU_UUID'])
            if identity not in seen:
                seen.add(identity)
                result.append(row)
        return result
    return {'steady': unique(steady, True), 'full': unique(full, True),
            'control': unique(control, False), 'scaffold': unique(scaffold, False),
            'local_services': [row for row in a['services'] if row['kind'] == 'local'],
            'gate': gate, 'raw_analysis_SHA256': c.sha((c.prior.OUT/'analysis.json').read_bytes())}


def medians(rows, kind='case'):
    groups = defaultdict(list)
    for row in rows:
        identity = row['layout'] if kind == 'case' else row['geometry_key']
        groups[identity, row['q'], row['cohort']].append(row)
    return [{**items[0], 'target_ns': st.median(row['target_ns'] for row in items),
             'worker_range_ns': [min(row['target_ns'] for row in items), max(row['target_ns'] for row in items)],
             'workers': len(items)} for items in groups.values()]


def local_tables(bundle):
    groups = defaultdict(list)
    seen = set()
    for row in bundle['local_services']:
        meta = bundle['gate']['metadata'][row['tag']]
        key = (row['tag'], row['q'], row['invocation'])
        if key in seen:
            continue
        seen.add(key)
        groups[meta['R'], meta['p'], meta['columns'], meta['variant'], row['q']].append(row['service_ns'])
    return [{'R': r, 'p': p, 'columns': columns, 'variant': variant, 'q': q,
             'point_ns': st.median(values), 'low_ns': min(values), 'high_ns': max(values),
             'workers': len(values)} for (r, p, columns, variant, q), values in sorted(groups.items())]


def paired(rows, predictions):
    from experiments.tma_reduction_layout.phase11.c3.static import upstream
    index = {(row['layout'], row['q_condition'], row['cohort'], row['worker'], row['GPU_UUID']): (row, value)
             for row, value in zip(rows, predictions)}
    result = []
    for row, value in zip(rows, predictions):
        case = row['case']
        default = upstream(case['BM'], case['BN'], case['W'])
        if default == row['layout']:
            continue
        key = (default, row['q_condition'], row['cohort'], row['worker'], row['GPU_UUID'])
        if key not in index:
            # A natural alias can have been deduplicated under another label.
            possible = [(other, predicted) for other, predicted in zip(rows, predictions)
                        if other['layout'] == default and other['cohort'] == row['cohort'] and
                        other['worker'] == row['worker'] and other['GPU_UUID'] == row['GPU_UUID'] and
                        other['q'] == row['q']]
            if not possible:
                continue
            d, dp = possible[0]
        else:
            d, dp = index[key]
        observed = row['target_ns']-d['target_ns']
        predicted = value-dp
        result.append({'geometry': row['geometry'], 'layout': row['layout'], 'default': default,
            'q_condition': row['q_condition'], 'candidate_q': row['q'], 'default_q': d['q'],
            'cohort': row['cohort'], 'worker': row['worker'], 'GPU_UUID': row['GPU_UUID'],
            'observed_delta_ns': observed, 'predicted_delta_ns': predicted,
            'delta_residual_ns': observed-predicted,
            'default_residual_ns': d['target_ns']-dp,
            'observed_default_ns': d['target_ns'], 'predicted_default_ns': dp})
    return result
