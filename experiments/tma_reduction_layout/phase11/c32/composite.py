"""Calibrated full-service envelopes; fitted corrections are not physical bases."""
import numpy as np
from experiments.tma_reduction_layout.phase11.c32 import regression as reg
from experiments.tma_reduction_layout.phase11.c32.features import features, transfer_features
from experiments.tma_reduction_layout.phase11.c31.fit import nnls


def control_features(geometry, q):
    f = transfer_features(geometry)
    for name in ('CTA', 'input_128B', 'N_64col_segments', 'warps'):
        f[name+'__inverse_q'] = f[name]/q
    return f


def fit_control(rows, ridge):
    fs = [control_features(row['geometry'], row['q']) for row in rows]
    ys = [row['target_ns'] for row in rows]
    bulk_names = ['input_128B', 'output_128B']
    latency_names = ['CTA__inverse_q', 'input_128B__inverse_q',
                     'N_64col_segments__inverse_q', 'warps__inverse_q']
    names = bulk_names+latency_names
    x = np.array([[f[name] for name in names] for f in fs])
    scale = np.maximum(np.linalg.norm(x, axis=0), 1.)
    z = x/scale
    y = np.array(ys)
    aa, bb = z[:, :2], z[:, 2:]
    a, _ = nnls(aa, y, ridge)
    b, _ = nnls(bb, y, ridge)
    best = None
    # Effective bulk/finite-residency control branches. These are response
    # parameters; this fitting does not identify pure physical TMA latency.
    for wa, wb in ((1., .5), (.5, 1.), (1., 1.)):
        ca, cb = a*wa, b*wb
        step = 1/(2*max(np.linalg.norm(aa, 2)**2, np.linalg.norm(bb, 2)**2)+2*ridge+1e-9)
        for _ in range(500):
            pa, pb = aa@ca, bb@cb
            active = pa >= pb
            error = np.maximum(pa, pb)-y
            ca = np.maximum(0., ca-step*(2*aa.T@(error*active)+2*ridge*ca))
            cb = np.maximum(0., cb-step*(2*bb.T@(error*(~active))+2*ridge*cb))
        loss = float(np.mean((np.maximum(aa@ca, bb@cb)-y)**2))
        if best is None or loss < best[0]:
            best = loss, ca.copy(), cb.copy()
    loss, ca, cb = best
    return {'kind': 'bulk_or_finite_residency_control',
            'bulk': {'feature_names': bulk_names, 'coefficients': (ca/scale[:2]).tolist()},
            'latency': {'feature_names': latency_names, 'coefficients': (cb/scale[2:]).tolist()},
            'training_RMSE_ns': loss**.5,
            'interpretation': 'Operational transfer-control envelope, including row/output/wait work; not a pure physical decomposition.'}


def control_point(model, geometry, q):
    f = control_features(geometry, q)
    return max(0., reg.predict(model['bulk'], f), reg.predict(model['latency'], f))


def mc_point(model, case, q, recipe, tables, mode='full'):
    return max(0., reg.predict(model, features(case, q, recipe, tables, mode)))


def base_features(case, q, scaffold):
    m, n, w = case['BM'], case['BN'], case['W']
    return {'measured_scaffold_proxy': max(0., scaffold), 'N_64col_segments_per_q': n/64/q,
            'warps_per_q': w/q, 'input_128B_per_q': 2*m*n/128/q}


def full_features(case, q, mc, control, recipe, tables, mode):
    f = features(case, q, recipe, tables, mode)
    return {**f, 'combined_MC_prediction': mc,
            'transfer_control_prediction': control,
            'transfer_control__inverse_q': control/q}


def fit_floor(fs, ys, floors, ridge):
    model = reg.fit(fs, ys, ridge)
    names = model['feature_names']
    a = np.array([[f[name] for name in names] for f in fs])
    y = np.array(ys)
    floor = np.array(floors)
    beta = np.array(model['coefficients'])
    best_loss = float(np.mean((np.maximum(floor, a@beta)-y)**2))
    for _ in range(20):
        active = (y > floor) | (a@beta > floor)
        if not active.any():
            break
        proposal, _ = nnls(a[active], y[active], ridge)
        accepted = False
        for weight in (1., .5, .25, .125, .0625):
            trial = weight*proposal+(1-weight)*beta
            loss = float(np.mean((np.maximum(floor, a@trial)-y)**2))
            if loss < best_loss-1e-9:
                beta, best_loss, accepted = trial, loss, True
                break
        if not accepted:
            break
    model['coefficients'] = beta.tolist()
    model['diagnostic']['floor_envelope_RMSE_ns'] = best_loss**.5
    return model


def train(mc_model, full_rows, control_rows, scaffold_rows, recipe, tables,
          composition, mode='full'):
    control = fit_control(control_rows, recipe['ridge'])
    scaffold = reg.fit([control_features(row['geometry'], row['q']) for row in scaffold_rows],
                       [row['target_ns'] for row in scaffold_rows], recipe['ridge'])
    mc = [mc_point(mc_model, row['case'], row['q'], recipe, tables, mode) for row in full_rows]
    controls = [control_point(control, row['geometry_vector'], row['q']) for row in full_rows]
    fixtures = [max(0., reg.predict(scaffold, control_features(row['geometry_vector'], row['q']))) for row in full_rows]
    y = [row['target_ns'] for row in full_rows]
    result = {'kind': composition, 'MC': mc_model, 'control': control, 'scaffold': scaffold,
              'recipe': recipe, 'mode': mode, 'historical_offset_used': False}
    if composition == 'shared_tile_base':
        fs = [base_features(row['case'], row['q'], fixture) for row, fixture in zip(full_rows, fixtures)]
        names = sorted(fs[0])
        a = np.array([[f[name] for name in names] for f in fs])
        beta = np.zeros(len(names))
        for _ in range(20):
            current = np.maximum(controls, np.array(mc)+a@beta)
            active = (np.array(y) > np.array(controls)) | (np.array(mc)+a@beta > controls)
            if not active.any():
                break
            proposal, _ = nnls(a[active], np.array(y)[active]-np.array(mc)[active], recipe['ridge'])
            if np.mean((np.maximum(controls, np.array(mc)+a@proposal)-y)**2) >= np.mean((current-y)**2)-1e-9:
                break
            beta = proposal
        result['base'] = {'feature_names': names, 'coefficients': beta.tolist(),
                          'interpretation': 'Shared effective tile correction; not an independently isolated physical B.'}
    elif composition == 'shared_full_service':
        fs = [full_features(row['case'], row['q'], s, h, recipe, tables, mode)
              for row, s, h in zip(full_rows, mc, controls)]
        result['full'] = fit_floor(fs, y, controls, recipe['ridge'])
        result['interpretation'] = 'Static full-service calibration with MC and independent controls; no unique physical B/transfer decomposition claimed.'
    return result


def predict(model, case, q, tables, disable_envelope=False):
    recipe, mode = model['recipe'], model['mode']
    mc = mc_point(model['MC'], case, q, recipe, tables, mode)
    geometry = [case['BM'], case['BN'], case['W']]
    control = control_point(model['control'], geometry, q)
    scaffold = max(0., reg.predict(model['scaffold'], control_features(geometry, q)))
    compute = mc
    if model['kind'] == 'shared_tile_base':
        compute += max(0., reg.predict(model['base'], base_features(case, q, scaffold)))
    elif model['kind'] == 'shared_full_service':
        compute = max(0., reg.predict(model['full'], full_features(case, q, mc, control, recipe, tables, mode)))
    return {'tile_ns': compute if disable_envelope else max(control, compute),
            'MC_ns': mc, 'control_ns': control, 'compute_path_ns': compute,
            'scaffold_proxy_ns': scaffold, 'q': q}
