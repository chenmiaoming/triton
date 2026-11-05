"""Static H100 selector: no compiler, device, runtime grid or measured winner."""
from math import sqrt
import numpy as np
from experiments.tma_reduction_layout.phase11.c32 import common as c
from experiments.tma_reduction_layout.phase11.c32 import resources, regression as reg, composite
from experiments.tma_reduction_layout.phase11.c32.features import static, features
from experiments.tma_reduction_layout.phase11.c3.static import candidates, upstream


def difference_leverage(theta, candidate, default, qc, qd):
    model = theta['model']
    fit = model['full'] if model['kind'] == 'shared_full_service' else model['MC']

    def vector(case, q):
        f = features(case, q, model['recipe'], theta['fixed_R_tables'], model['mode'])
        if model['kind'] == 'shared_full_service':
            pred = composite.predict(model, case, q, theta['fixed_R_tables'])
            f = composite.full_features(case, q, pred['MC_ns'], pred['control_ns'],
                                        model['recipe'], theta['fixed_R_tables'], model['mode'])
        return np.array([f[name] for name in fit['feature_names']])/np.array(fit['scale'])

    # Common geometry demands cancel in the candidate-minus-default vector.
    difference = vector(candidate, qc)-vector(default, qd)
    h = max(0., float(difference@np.array(fit['inverse_gram'])@difference))
    return {'value': h, 'inflation': sqrt(max(1., h/max(fit['median_training_leverage'], 1e-12)))}


def joint_q(theta, candidate, default, ce, de):
    """Shared register-error scenarios, never independent q interval extrema."""
    model = theta['resource_model']
    paired = model['paired_register_errors']
    paired = [{'default_residual': 0., 'delta_residual': 0.}]+paired
    common_margin = c.protocol()['uncertainty']['resource_margin_registers']*max(
        ce['leverage_inflation'], de['leverage_inflation'])
    positive = [max(0., row['delta_residual'])**2 for row in paired]
    difference_margin = max(2., float(np.sqrt(np.mean(positive))))*max(
        ce['leverage_inflation'], de['leverage_inflation'])
    result = {}
    for row in paired:
        for shift in (-common_margin, 0., common_margin):
            for shared_upper in (False, True):
                dr = de['n_regs_point']+row['default_residual']+shift
                cr = ce['n_regs_point']+row['default_residual']+row['delta_residual']+shift+difference_margin
                qc = resources.q_for(candidate, cr, shared_upper)
                qd = resources.q_for(default, dr, shared_upper)
                result.setdefault((qc, qd), {'candidate_q': qc, 'default_q': qd,
                    'source': 'joint held-geometry register residual, common allocation margin and shared-memory bound'})
    result.setdefault((ce['q_point'], de['q_point']), {'candidate_q': ce['q_point'], 'default_q': de['q_point'],
        'source': 'nominal shared resource prediction'})
    return list(result.values()), {'common_register_margin': common_margin,
                                  'candidate_delta_register_margin': difference_margin}


def predict(geometry, theta=None, disable_envelope=False):
    theta = theta or c.read(c.OUT/'theta_H100.json')
    m, n, w = map(int, geometry)
    pool = candidates(m, n, w)
    default_id = upstream(m, n, w)
    default = next(case for case in pool if case['id'] == default_id)
    resource = {case['id']: resources.estimate(case, theta['resource_model']) for case in pool}
    de = resource[default_id]
    table = theta['fixed_R_tables']
    model = theta['model']
    point = {case['id']: composite.predict(model, case, resource[case['id']]['q_point'], table,
                                         disable_envelope) for case in pool}
    residuals = [{'delta_residual_ns': 0., 'default_residual_ns': 0., 'geometry': 'nominal'}]
    residuals += theta['paired_cost_residuals']
    records = []
    for case in pool:
        identity = case['id']
        ce = resource[identity]
        plan = static(case)
        r = plan['elements_per_thread']
        support = theta['MC_R_support']
        extrapolation = max(1., r/max(support), min(support)/r)
        estimate = point[identity]
        if identity == default_id:
            records.append({'layout': identity, 'case': case, 'plan_parameters':
                {name: plan[name] for name in ('p', 'u', 'w', 'elements_per_thread', 'columns_per_thread')},
                'resource': ce, 'point': estimate, 'nominal_improvement_percent': 0.,
                'paired_delta_interval_ns': [0., 0.], 'worst_improvement_percent': 0.,
                'robust_switch': False, 'reason': 'upstream_default', 'performance_R_extrapolation': extrapolation})
            continue
        scenarios, reg_margin = joint_q(theta, case, default, ce, de)
        lows, highs, ratios, margins = [], [], [], []
        qrecords = []
        for scenario in scenarios:
            qc, qd = scenario['candidate_q'], scenario['default_q']
            pc = composite.predict(model, case, qc, table, disable_envelope)
            pd = composite.predict(model, default, qd, table, disable_envelope)
            lev = difference_leverage(theta, case, default, qc, qd)
            margin = theta['one_sided_additional_margin_ns']*lev['inflation']*extrapolation
            delta = pc['tile_ns']-pd['tile_ns']
            uppers, lowers, rr = [], [], []
            for error in residuals:
                low = delta+error['delta_residual_ns']
                upper = low+margin
                # Preserve the joint default/difference residual, including its
                # sign; this is not independent marginal interval subtraction.
                denominator = pd['tile_ns']+error['default_residual_ns']
                if denominator <= 0.:
                    rr.append(float('inf'))
                else:
                    rr.append(upper/denominator)
                lowers.append(low)
                uppers.append(upper)
            lows.extend(lowers)
            highs.extend(uppers)
            ratios.extend(rr)
            margins.append(margin)
            qrecords.append({**scenario, 'candidate_service_ns': pc['tile_ns'],
                'default_service_ns': pd['tile_ns'], 'predicted_delta_ns': delta,
                'one_sided_margin_ns': margin, 'paired_leverage': lev,
                'worst_delta_ns': max(uppers),
                'worst_improvement_percent': -100*max(rr) if all(np.isfinite(rr)) else None})
        worst = max(ratios)
        robust = worst < -c.protocol()['threshold']
        reason = 'robust_paired_improvement' if robust else 'no_robust_steady_improvement'
        if not ce['shared_legal']:
            robust, reason = False, 'static_shared_illegal'
        elif ce['spill_uncertainty']:
            robust, reason = False, 'calibrated_register_interval_exceeds_hardware_limit'
        elif ce['R_outside_support']:
            robust, reason = False, 'resource_R_extrapolation'
        elif ce['new_warp_count']:
            robust, reason = False, 'unseen_warp_count_resource_response'
        elif extrapolation > 1:
            robust, reason = False, 'performance_R_extrapolation_requires_validation'
        records.append({'layout': identity, 'case': case, 'plan_parameters':
            {name: plan[name] for name in ('p', 'u', 'w', 'elements_per_thread', 'columns_per_thread')},
            'resource': ce, 'point': estimate,
            'nominal_improvement_percent': 100*(1-estimate['tile_ns']/point[default_id]['tile_ns']),
            'paired_delta_interval_ns': [min(lows), max(highs)],
            'worst_improvement_percent': -100*worst if np.isfinite(worst) else None,
            'margin_range_ns': [min(margins), max(margins)], 'register_pair_margin': reg_margin,
            'joint_q_scenarios': qrecords, 'robust_switch': bool(robust), 'reason': reason,
            'performance_R_extrapolation': extrapolation})
    winners = [row for row in records if row['robust_switch']]
    # Equal transfer-floor predictions do not resolve a ranking. Among safe
    # candidates with the same nominal service, prefer predicted on-chip
    # headroom, rather than ranking sub-nanosecond uncertainty-margin changes.
    best_safe = min((row['point']['tile_ns'] for row in winners), default=0.)
    tied = [row for row in winners if row['point']['tile_ns'] <= best_safe+1e-8]
    chosen = min(tied, key=lambda row: (row['point']['MC_ns'], -row['worst_improvement_percent'], row['layout'])) if tied else None
    nominal = min(records, key=lambda row: (row['point']['tile_ns'], row['point']['MC_ns'], row['layout']))
    best = nominal['point']['tile_ns']
    comparable = [row['layout'] for row in records if row['point']['tile_ns'] <= best/(1-c.protocol()['threshold'])]
    return {'geometry': [m, n, w], 'R': m*n//(32*w), 'version': 'c3_v2', 'upstream_default': default_id,
        'selected': chosen['layout'] if chosen else default_id,
        'nominal_best': nominal['layout'], 'nominal_comparable_set_3_percent': comparable,
        'changed': chosen is not None,
        'decision_reason': 'robust_paired_improvement' if chosen else 'retain_default; consult per-candidate limits',
        'threshold': c.protocol()['threshold'], 'candidates': records,
        'inputs': ['BM', 'BN', 'W', 'dtype', 'reduce_axis', 'register_layout', 'shared_layout'],
        'uses_runtime_grid': False, 'uses_observed_candidate_n_regs': False,
        'D1_compiled': False, 'D1_timed': False}


if __name__ == '__main__':
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument('geometry', nargs=3, type=int)
    print(json.dumps(predict(parser.parse_args().geometry), indent=2, allow_nan=False))
