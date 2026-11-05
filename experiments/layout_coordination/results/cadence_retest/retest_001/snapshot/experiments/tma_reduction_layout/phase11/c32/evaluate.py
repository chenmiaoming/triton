"""Development choices and ablations; measured outcomes never enter select()."""
from collections import defaultdict
import copy
import statistics as st
from experiments.tma_reduction_layout.phase11.c32 import common as c
from experiments.tma_reduction_layout.phase11.c32 import composite, resources, data
from experiments.tma_reduction_layout.phase11.c32.calibrate import Calibration, rmse_pairs
from experiments.tma_reduction_layout.phase11.c32.model import predict


def observed(bundle, geometry):
    key = f'{geometry[0]}x{geometry[1]}w{geometry[2]}'
    grouped = defaultdict(list)
    for row in bundle['full']:
        if row['geometry'] == key and row['q_condition'] == 'natural':
            grouped[row['layout']].append(row['target_ns'])
    return {name: st.median(values) for name, values in grouped.items()}


def score(decision, values):
    default = decision['upstream_default']
    selected = decision['selected']
    oracle = min(values, key=values.get)
    return {'geometry': decision['geometry'], 'upstream_default': default, 'selected': selected,
        'measured_layouts': len(values), 'measured_default_ns': values[default],
        'measured_selected_ns': values.get(selected), 'measured_best': oracle,
        'measured_best_ns': values[oracle],
        'measured_selected_gain_percent': 100*(1-values[selected]/values[default]) if selected in values else None,
        'measured_oracle_gain_percent': 100*(1-values[oracle]/values[default]),
        'selected_regret_percent': 100*(values[selected]/values[oracle]-1) if selected in values else None,
        'point_selected': decision['nominal_best'], 'changed': decision['changed'],
        'scope': 'Development outcome evaluation after static selection; not held-out efficacy.'}


def ablation_theta(calibration, theta, mode):
    result = copy.deepcopy(theta)
    recipe, kind = theta['selected_recipe'], theta['selected_composition']
    residuals, errors, conditional = [], [], []
    # Keep the full-model recipe fixed to isolate feature removal. Each held
    # geometry is still excluded when refitting ablation coefficients.
    for held in calibration.geometries:
        model = calibration.model({held}, recipe, kind, mode)
        resource = calibration.resource({held})
        rows = [row for row in calibration.bundle['full'] if row['geometry'] == held and row['q_condition'] == 'natural']
        pairs, points = calibration.predict_rows(model, resource, rows)
        residuals.extend(pairs)
        errors.extend(points)
        conditional.extend(calibration.predict_rows(model, resource, rows, controlled=True)[0])
    result['model'] = calibration.model(set(), recipe, kind, mode)
    result['paired_outer_residuals'] = residuals
    result['paired_cost_residuals'] = conditional
    result['point_outer_errors'] = errors
    result['paired_delta_residual_interval_ns'] = [min(row['delta_residual_ns'] for row in residuals), max(row['delta_residual_ns'] for row in residuals)]
    result['paired_cost_residual_interval_ns'] = [min(row['delta_residual_ns'] for row in conditional), max(row['delta_residual_ns'] for row in conditional)]
    by_geometry = defaultdict(list)
    for row in conditional:
        by_geometry[row['geometry']].append(max(0., row['delta_residual_ns'])**2)
    result['one_sided_additional_margin_ns'] = max(5., st.mean(st.mean(values) for values in by_geometry.values())**.5)
    result['nested_geometry_balanced_paired_RMSE_ns'] = rmse_pairs(residuals)
    result['validation_scope'] = 'Held-geometry ablation refits; recipe inherited from full development selection, not independently nested.'
    return result


def run():
    theta = c.read(c.OUT/'theta_H100.json')
    calibration = Calibration()
    geometries = [list(map(int, geometry.replace('x', ' ').replace('w', ' ').split())) for geometry in calibration.geometries]
    decisions = [predict(geometry, theta) for geometry in geometries]
    values = {tuple(geometry): observed(calibration.bundle, geometry) for geometry in geometries}
    evaluations = [score(decision, values[tuple(decision['geometry'])]) for decision in decisions]
    modes = {}
    for mode in ('Load-blind', 'Comm-blind', 'Fixed-R-blind'):
        ablated = ablation_theta(calibration, theta, mode)
        choices = [predict(geometry, ablated) for geometry in geometries]
        modes[mode] = {'theta': ablated, 'decisions': choices,
            'evaluation': [score(choice, values[tuple(choice['geometry'])]) for choice in choices]}
    # A diagnostic bypass, not a separately validated alternative selector.
    envelope_blind = [predict(geometry, theta, disable_envelope=True) for geometry in geometries]
    modes['Envelope-blind'] = {'decisions': envelope_blind,
        'evaluation': [score(choice, values[tuple(choice['geometry'])]) for choice in envelope_blind],
        'scope': 'Diagnostic bypass of the control floor with the same coefficients/residuals; no independent efficacy claim.'}
    baselines = []
    for decision in decisions:
        values_here = values[tuple(decision['geometry'])]
        available = [row for row in decision['candidates'] if row['layout'] in values_here]
        for name in ('upstream_default', 'vec1', 'vec2', 'vec4', 'vec8', 'fanout_only'):
            if name == 'upstream_default':
                selected = decision['upstream_default']
            elif name == 'fanout_only':
                row = min(available, key=lambda row: (row['plan_parameters']['u']*row['plan_parameters']['w'],
                                                      row['case']['placement'] != 'n_first', row['layout']))
                selected = row['layout']
            else:
                rows = [row for row in available if row['case']['register_layout']['size_per_thread'][2] == int(name[-1]) and row['case']['placement'] == 'n_first']
                if not rows:
                    continue
                selected = rows[0]['layout']
            baselines.append({'baseline': name, **score({**decision, 'selected': selected,
                'changed': selected != decision['upstream_default']}, values_here)})
    old = c.read(c.BASE/'results/phase11/stage_c31/c3_v1/development.json')
    available = {row['layout']: row for decision in decisions for row in decision['candidates']}
    historical = []
    for row in old['absolute_FULL_diagnostics']:
        now = available[row['layout']]['point']['tile_ns']
        actual = row['old_FULL_tile_SM_ns']
        historical.append({'layout': row['layout'], 'historical_observed_tile_SM_ns': actual,
            'c31_prediction_ns': row['predicted_bottleneck_ns'], 'c32_prediction_ns': now,
            'c31_error_percent': row['absolute_error_percent'],
            'c32_error_percent': 100*(now/actual-1),
            'scope': 'Same eight historical FULL points, evaluation only. Different phase/cohort; no fitted offset and no grid feature.'})
    c.derived(c.OUT/'development.json', {'decisions': decisions, 'evaluation': evaluations,
        'baselines': baselines, 'ablations': modes,
        'historical_FULL_diagnostics': historical,
        'primary_metric': 'True loop-free natural-q full-program grid slope, ns/tile/SM; grid only normalizes measurements.',
        'warning': 'Only eight development geometries. No D1 efficacy has been measured.'})
    print('C3.2 development changes', sum(row['changed'] for row in evaluations), '/', len(evaluations), flush=True)


if __name__ == '__main__':
    run()
