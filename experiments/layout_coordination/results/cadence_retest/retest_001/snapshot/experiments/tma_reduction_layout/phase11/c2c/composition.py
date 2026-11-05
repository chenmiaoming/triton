"""Test additive composition rather than assuming a common physical base."""
from collections import defaultdict
import statistics as st
import numpy as np
from experiments.tma_reduction_layout.phase11.c2c import common as c


def best_base(transfer, steady, full):
    h, s, f = map(lambda x: np.asarray(x, float), (transfer, steady, full))
    breaks = sorted({0., *np.maximum(h-s, 0.).tolist()})
    trials = [0.]
    for low, high in zip(breaks, breaks[1:]+[float('inf')]):
        active = s+low+1e-8 > h
        if active.any():
            candidate = float(np.mean(f[active]-s[active]))
            trials.append(max(low, min(high, candidate)))
        trials.append(low)
    return min(trials, key=lambda b: float(np.mean((np.maximum(h, b+s)-f)**2)))


def run():
    a = c.read(c.OUT/'analysis.json')
    gate = c.read(c.OUT/'artifact_gate.json')
    metadata = gate['metadata']
    mc = {(metadata[row['tag']].get('layout_id'), row['q_condition'], row['invocation']): row
          for row in a['services'] if row['family'] == 'MC'}
    controls = {(tuple(row['geometry']), row['q_condition'], row['invocation']): row
                for row in a['oneshot'] if row['family'] in ('ONE_TRANSFER', 'ORIGINAL_TRANSFER')}
    pairs = []
    for row in a['oneshot']:
        if row['family'] not in ('ONE_FULL', 'ORIGINAL_FULL'):
            continue
        meta = metadata[row['tag']]
        key = (meta['layout_id'], row['q_condition'], row['invocation'])
        if key not in mc:
            continue
        steady = mc[key]
        control = controls[tuple(row['geometry']), row['q_condition'], row['invocation']]
        same_q = row['q'] == steady['q'] == control['q']
        pairs.append({'layout': meta['layout_id'], 'geometry': row['geometry'],
            'q_condition': row['q_condition'], 'worker': row['invocation'], 'GPU_UUID': row['GPU_UUID'],
            'full_q': row['q'], 'MC_q': steady['q'], 'control_q': control['q'], 'same_q': same_q,
            'full_tile_slope_ns': row['tile_slope_ns'], 'MC_steady_ns': steady['service_ns'],
            'control_tile_slope_ns': control['tile_slope_ns'],
            'zero_base_prediction_ns': max(control['tile_slope_ns'], steady['service_ns']),
            'full_minus_MC_ns': row['tile_slope_ns']-steady['service_ns'],
            'full_grid_us': row['full_us'], 'global_intercept_us': row['global_intercept_us']})
    groups = defaultdict(list)
    for row in pairs:
        if row['same_q']:
            groups[tuple(row['geometry']), row['q_condition'], row['worker']].append(row)
    fitted = []
    for (geometry, q, worker), rows in groups.items():
        h = [row['control_tile_slope_ns'] for row in rows]
        s = [row['MC_steady_ns'] for row in rows]
        f = [row['full_tile_slope_ns'] for row in rows]
        b = best_base(h, s, f)
        predicted = np.maximum(h, b+np.asarray(s))
        errors = 100*(predicted/np.asarray(f)-1)
        fitted.append({'geometry': list(geometry), 'q_condition': q, 'worker': worker,
            'best_effective_base_ns': b, 'layouts': len(rows),
            'max_abs_error_percent': float(np.max(np.abs(errors))),
            'RMSE_ns': float(np.sqrt(np.mean((predicted-np.asarray(f))**2))),
            'nonnegative_pure_transfer_compatible': b <= min(h),
            'all_points_within_3_percent': bool(np.max(np.abs(errors)) <= 3.),
            'not_an_independently_identified_physical_base': True,
            'predictions': [{'layout': row['layout'], 'actual_ns': actual, 'predicted_ns': float(pred),
                             'error_percent': float(error)} for row, actual, pred, error in zip(rows, f, predicted, errors)]})
    result = {'pairs': pairs, 'best_common_base_diagnostics': fitted,
        'physical_base_identified': False,
        'interpretation': 'Same-q common-base test is deliberately generous: an in-sample base per geometry/worker. Failure rejects this simple decomposition. Success alone still cannot isolate a physical base or pure transfer.',
        'global_intercepts_used_in_selector': False, 'historical_26_6_used': False}
    c.derived(c.OUT/'composition.json', result)
    print('C2c composition:', len(pairs), 'paired full/MC/control rows;',
          sum(not row['all_points_within_3_percent'] for row in fitted), '/', len(fitted),
          'same-q common-base groups fail 3% closure; no isolated physical base claimed')
    return result


if __name__ == '__main__':
    run()
