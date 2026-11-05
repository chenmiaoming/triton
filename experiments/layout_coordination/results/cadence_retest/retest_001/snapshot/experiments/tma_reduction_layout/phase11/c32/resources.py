"""Calibrated register intervals, partitioned by actual program family."""
from math import ceil, floor, sqrt
import numpy as np
from experiments.tma_reduction_layout.phase11.c32 import common as c
from experiments.tma_reduction_layout.phase11.c32.data import case_for
from experiments.tma_reduction_layout.phase11.c32.features import resource_features, static
from experiments.tma_reduction_layout.phase11.c31.fit import nnls


def census():
    rows = []
    for stage in ('stage_c2a', 'stage_c2b', 'stage_c2c'):
        gate = c.read(c.BASE/'results/phase11'/stage/'artifact_gate.json')
        for tag, meta in gate['metadata'].items():
            binding = gate['resources'][tag]
            family = 'other_subgraph'
            case = None
            if meta.get('layout_id'):
                case = case_for(meta)
                if meta.get('resource_only') or meta['family'] in ('ONE_FULL', 'ORIGINAL_FULL'):
                    family = 'full_single_pass'
                elif meta['family'] == 'MC':
                    family = 'resident_loop'
            if meta.get('kind') == 'local' or meta['family'] in ('P', 'U', 'W', 'E'):
                family = 'local_fixture'
            rows.append({'stage': stage, 'tag': tag, 'case': case, 'family': family,
                'cubin_SHA256': meta['cubin_SHA256'],
                'n_regs': binding['attributes']['num_regs'],
                'local_bytes': binding['attributes']['local_bytes'],
                'static_shared_bytes': binding['attributes']['static_smem_bytes'],
                'dynamic_shared_bytes': meta['dynamic_smem_bytes'],
                'natural_q': binding['reservations']['natural']['capacity'],
                'geometry': f'{case["BM"]}x{case["BN"]}w{case["W"]}' if case else None})
    unique = {}
    for row in rows:
        unique.setdefault((row['cubin_SHA256'], row['family']), row)
    return rows, list(unique.values())


def fit_model(rows):
    fs = [resource_features(row['case']) for row in rows]
    names = sorted(fs[0])
    x = np.array([[f[key] for key in names] for f in fs])
    y = np.array([row['n_regs'] for row in rows])
    beta, diagnostic = nnls(x, y, c.protocol()['resource_ridge'])
    scale = np.maximum(np.linalg.norm(x, axis=0), 1.)
    z = x/scale
    inverse = np.linalg.pinv(z.T@z+1e-3*np.eye(len(names)))
    lev = np.einsum('ij,jk,ik->i', z, inverse, z)
    return {'feature_names': names, 'coefficients': beta.tolist(), 'diagnostic': diagnostic,
            'scale': scale.tolist(), 'inverse_gram': inverse.tolist(),
            'median_training_leverage': float(np.median(lev)),
            'R_support': [min(static(row['case'])['elements_per_thread'] for row in rows),
                          max(static(row['case'])['elements_per_thread'] for row in rows)],
            'W_support': sorted({row['case']['W'] for row in rows})}


def point(model, case):
    f = resource_features(case)
    return float(np.dot(model['coefficients'], [f[key] for key in model['feature_names']]))


def calibrate(rows, family, exclude_geometry=None):
    excluded = set() if exclude_geometry is None else ({exclude_geometry} if isinstance(exclude_geometry, str) else set(exclude_geometry))
    training = [row for row in rows if row['family'] == family and
                row['geometry'] not in excluded and not row['local_bytes']]
    assert training, family
    model = fit_model(training)
    errors = []
    for geometry in sorted({row['geometry'] for row in training}):
        keep = [row for row in training if row['geometry'] != geometry]
        if not keep:
            continue
        fold = fit_model(keep)
        for row in training:
            if row['geometry'] == geometry:
                predicted = point(fold, row['case'])
                errors.append({'geometry': geometry, 'layout': row['case']['id'],
                    'observed': row['n_regs'], 'predicted': predicted,
                    'residual': row['n_regs']-predicted})
    residuals = [row['residual'] for row in errors]
    from experiments.tma_reduction_layout.phase11.c3.static import upstream
    by_geometry = {}
    for row in training:
        case = row['case']
        if case['id'] == upstream(case['BM'], case['BN'], case['W']):
            by_geometry[row['geometry']] = row
    paired_errors = []
    for row in errors:
        default = by_geometry.get(row['geometry'])
        if default is None or row['layout'] == default['case']['id']:
            continue
        d = next(item for item in errors if item['geometry'] == row['geometry'] and item['layout'] == default['case']['id'])
        paired_errors.append({'geometry': row['geometry'], 'layout': row['layout'],
            'default': default['case']['id'], 'default_residual': d['residual'],
            'delta_residual': row['residual']-d['residual']})
    model.update(family=family, observations=len(training), held_geometry_errors=errors,
                 residual_interval=[min(residuals), max(residuals)],
                 paired_register_errors=paired_errors,
                 local_storage_or_spill_rows=[row for row in rows if row['family'] == family and row['local_bytes']],
                 empirical_interval_not_formal_RA_bound=True)
    model['training_geometries'] = sorted({row['geometry'] for row in training})
    model['excluded_geometries'] = sorted(excluded)
    return model


def q_for(case, registers, shared_upper=False):
    p = static(case)
    shared = p['resources']['shared_bytes_interval'][int(shared_upper)]
    block_regs = ceil(max(8, min(255, registers))*32/256)*256*case['W']
    return max(1, min(32, 64//case['W'], 65536//block_regs,
                      233472//(ceil(shared/256)*256)))


def estimate(case, model):
    f = resource_features(case)
    x = np.array([f[key] for key in model['feature_names']])/np.array(model['scale'])
    leverage = float(x@np.array(model['inverse_gram'])@x)
    inflation = sqrt(max(1., leverage/max(model['median_training_leverage'], 1e-12)))
    center = point(model, case)
    margin = c.protocol()['uncertainty']['resource_margin_registers']*inflation
    low = max(8, floor(center+model['residual_interval'][0]-margin))
    high = max(low, ceil(center+model['residual_interval'][1]+margin))
    p = static(case)
    shared_low, shared_high = p['resources']['shared_bytes_interval']

    def capacity(registers, shared):
        register_block = ceil(registers*32/256)*256*case['W']
        return max(0, min(32, 64//case['W'], 65536//max(register_block, 1),
                          233472//(ceil(shared/256)*256)))

    qlow = capacity(min(high, 255), shared_high)
    qhigh = capacity(low, shared_low)
    qpoint = capacity(max(8, min(255, round(center))), shared_low)
    return {'n_regs_point': center, 'n_regs_interval': [low, high],
        'q_interval': [max(1, qlow), max(1, qhigh)], 'q_point': max(1, qpoint),
        'shared_bytes_interval': [shared_low, shared_high],
        'shared_legal': p['resources']['shared_legal'],
        'spill_uncertainty': high > 255, 'leverage': leverage, 'leverage_inflation': inflation,
        'new_warp_count': case['W'] not in model['W_support'],
        'R_outside_support': not model['R_support'][0] <= p['elements_per_thread'] <= model['R_support'][1],
        'interval_scope': 'Held-geometry empirical register residuals plus 8-register leverage margin; not a proof of no spill.'}
