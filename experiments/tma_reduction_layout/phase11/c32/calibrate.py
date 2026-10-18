"""Nested development validation; no D1 compiler or timing dependency."""
from collections import defaultdict
from math import sqrt
import numpy as np
from experiments.tma_reduction_layout.phase11.c32 import common as c
from experiments.tma_reduction_layout.phase11.c32 import data, resources, regression as reg, composite
from experiments.tma_reduction_layout.phase11.c32.features import features


def geometry(row):
    return row.get('geometry_key', row.get('geometry'))


def rmse_pairs(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['geometry']].append(row['delta_residual_ns']**2)
    return float(np.sqrt(np.mean([np.mean(values) for values in grouped.values()]))) if grouped else float('inf')


class Calibration:
    def __init__(self):
        self.bundle = data.load()
        self.tables = data.local_tables(self.bundle)
        self.steady = data.medians(self.bundle['steady'])
        self.full = data.medians(self.bundle['full'])
        self.control = data.medians(self.bundle['control'], 'geometry')
        self.scaffold = data.medians(self.bundle['scaffold'], 'geometry')
        self.geometries = sorted({row['geometry'] for row in self.full})
        self.all_resources, self.resource_rows = resources.census()
        self.mc_cache, self.model_cache, self.resource_cache = {}, {}, {}
        self.recipe_search, self.composition_search = {}, {}

    def mc(self, excluded, recipe, mode='full'):
        key = tuple(sorted(excluded)), tuple(sorted(recipe.items())), mode
        if key not in self.mc_cache:
            rows = [row for row in self.steady if row['geometry'] not in excluded]
            self.mc_cache[key] = reg.fit([features(row['case'], row['q'], recipe, self.tables, mode) for row in rows],
                                         [row['target_ns'] for row in rows], recipe['ridge'])
            self.mc_cache[key]['training_geometries'] = sorted({row['geometry'] for row in rows})
            self.mc_cache[key]['training_layouts'] = sorted({row['layout'] for row in rows})
        return self.mc_cache[key]

    def resource(self, excluded):
        key = tuple(sorted(excluded))
        if key not in self.resource_cache:
            self.resource_cache[key] = resources.calibrate(self.resource_rows, 'full_single_pass', excluded)
        return self.resource_cache[key]

    def model(self, excluded, recipe, kind, mode='full'):
        key = tuple(sorted(excluded)), tuple(sorted(recipe.items())), kind, mode
        if key not in self.model_cache:
            self.model_cache[key] = composite.train(self.mc(excluded, recipe, mode),
                [row for row in self.full if row['geometry'] not in excluded],
                [row for row in self.control if geometry(row) not in excluded],
                [row for row in self.scaffold if geometry(row) not in excluded],
                recipe, self.tables, kind, mode)
            self.model_cache[key]['full_training_geometries'] = sorted({row['geometry'] for row in self.full if row['geometry'] not in excluded})
            self.model_cache[key]['excluded_geometries'] = sorted(excluded)
        return self.model_cache[key]

    def recipe(self, excluded):
        key = tuple(sorted(excluded))
        if key not in self.recipe_search:
            options = []
            for recipe in c.protocol()['recipes']:
                residuals = []
                for held in self.geometries:
                    if held in excluded:
                        continue
                    model = self.mc(set(excluded)|{held}, recipe)
                    rows = [row for row in self.bundle['steady'] if row['geometry'] == held]
                    predictions = [max(0., reg.predict(model, features(row['case'], row['q'], recipe, self.tables))) for row in rows]
                    residuals.extend(data.paired(rows, predictions))
                options.append({'recipe': recipe, 'geometry_balanced_paired_RMSE_ns': rmse_pairs(residuals)})
            self.recipe_search[key] = options
        return min(self.recipe_search[key], key=lambda row: (row['geometry_balanced_paired_RMSE_ns'], -row['recipe']['ridge']))['recipe']

    def predict_rows(self, model, resource, rows, controlled=False):
        predictions = []
        records = []
        for row in rows:
            q = row['q'] if controlled else resources.q_for(row['case'], resources.point(resource, row['case']))
            predicted = composite.predict(model, row['case'], q, self.tables)
            predictions.append(predicted['tile_ns'])
            records.append({'layout': row['layout'], 'geometry': row['geometry'], 'worker': row['worker'],
                'GPU_UUID': row['GPU_UUID'], 'observed_ns': row['target_ns'],
                'predicted_ns': predicted['tile_ns'], 'observed_q': row['q'], 'predicted_q': q,
                'error_ns': row['target_ns']-predicted['tile_ns'],
                'relative_error_percent': 100*(predicted['tile_ns']/row['target_ns']-1)})
        return data.paired(rows, predictions), records

    def composition(self, excluded, recipe):
        key = tuple(sorted(excluded)), tuple(sorted(recipe.items()))
        if key not in self.composition_search:
            options = []
            for kind in c.protocol()['composition_models']:
                errors = []
                for held in self.geometries:
                    if held in excluded:
                        continue
                    omit = set(excluded)|{held}
                    model = self.model(omit, recipe, kind)
                    resource = self.resource(omit)
                    rows = [row for row in self.bundle['full'] if row['geometry'] == held and row['q_condition'] == 'natural']
                    paired, _ = self.predict_rows(model, resource, rows)
                    errors.extend(paired)
                options.append({'kind': kind, 'geometry_balanced_paired_RMSE_ns': rmse_pairs(errors)})
            self.composition_search[key] = options
        return min(self.composition_search[key], key=lambda row: row['geometry_balanced_paired_RMSE_ns'])['kind']


def run():
    calibration = Calibration()
    folds, residuals, point_errors, conditional = [], [], [], []
    for held in calibration.geometries:
        excluded = {held}
        recipe = calibration.recipe(excluded)
        kind = calibration.composition(excluded, recipe)
        model = calibration.model(excluded, recipe, kind)
        resource = calibration.resource(excluded)
        rows = [row for row in calibration.bundle['full'] if row['geometry'] == held and row['q_condition'] == 'natural']
        errors, points = calibration.predict_rows(model, resource, rows)
        residuals.extend(errors)
        point_errors.extend(points)
        conditional.extend(calibration.predict_rows(model, resource, rows, controlled=True)[0])
        folds.append({'held': held, 'recipe': recipe, 'composition': kind,
                      'model': model, 'resource_model': resource,
                      'inner_recipe_search': calibration.recipe_search[tuple(sorted(excluded))],
                      'inner_composition_search': calibration.composition_search[(tuple(sorted(excluded)), tuple(sorted(recipe.items())))]})
        print('C3.2 outer held geometry', held, recipe, kind, 'paired', len(errors), flush=True)
    recipe = calibration.recipe(set())
    kind = calibration.composition(set(), recipe)
    model = calibration.model(set(), recipe, kind)
    resource = calibration.resource(set())
    by_geometry = defaultdict(list)
    for row in conditional:
        by_geometry[row['geometry']].append(max(0., row['delta_residual_ns'])**2)
    margin = max(5., sqrt(float(np.mean([np.mean(values) for values in by_geometry.values()]))))
    theta = {'version': 'c3_v2', 'stage': 'C3.2', 'target': 'H100 sm90',
        'model': model, 'resource_model': resource, 'fixed_R_tables': calibration.tables,
        'selected_recipe': recipe, 'selected_composition': kind,
        'recipe_search': calibration.recipe_search[()],
        'composition_search': calibration.composition_search[((), tuple(sorted(recipe.items())))],
        'outer_folds': folds, 'paired_outer_residuals': residuals, 'point_outer_errors': point_errors,
        'paired_cost_residuals': conditional,
        'paired_cost_residual_interval_ns': [min(row['delta_residual_ns'] for row in conditional), max(row['delta_residual_ns'] for row in conditional)],
        'uncertainty_decomposition': 'Static-q FULL errors are reported as validation. Conditional-at-observed-q paired FULL cost residuals combine with joint resource scenarios, avoiding duplicate resource-error accounting. Observed q is calibration data, never a selector input.',
        'paired_delta_residual_interval_ns': [min(row['delta_residual_ns'] for row in residuals), max(row['delta_residual_ns'] for row in residuals)],
        'one_sided_additional_margin_ns': margin,
        'uncertainty_scope': 'Signed nested held-geometry paired residuals plus finite/extrapolation engineering margin. No reflection, no independent guarantee.',
        'nested_geometry_balanced_paired_RMSE_ns': rmse_pairs(residuals),
        'full_timing_geometries': calibration.geometries,
        'MC_R_support': sorted({row['case']['BM']*row['case']['BN']//(32*row['case']['W']) for row in calibration.steady}),
        'decision_tie_policy': 'Among robust >3% candidates with equal nominal full service (1e-8 ns numerical tolerance), choose lower predicted MC service as on-chip headroom; never use measured winners.',
        'physical_base_identified': False, 'historical_26_6_used': False,
        'selector_uses_runtime_grid': False, 'D1_compiled': False, 'D1_timed': False,
        'raw_C2c_analysis_SHA256': calibration.bundle['raw_analysis_SHA256']}
    census = {'rows': calibration.all_resources, 'unique': calibration.resource_rows,
              'full_single_pass_model': resource,
              'resident_loop_model': resources.calibrate(calibration.resource_rows, 'resident_loop')}
    c.derived(c.OUT/'theta_H100.json', theta)
    c.derived(c.OUT/'resource_calibration.json', census)
    c.derived(c.OUT/'calibration_cells.json', {'steady': calibration.steady, 'full': calibration.full,
        'control': calibration.control, 'scaffold': calibration.scaffold})
    print('C3.2 calibrated', recipe, kind, 'nested paired RMSE', theta['nested_geometry_balanced_paired_RMSE_ns'], flush=True)


if __name__ == '__main__':
    run()
