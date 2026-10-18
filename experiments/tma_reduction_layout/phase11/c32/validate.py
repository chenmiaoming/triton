"""Audit evidence, held-geometry arithmetic and deterministic pre-D1 choices."""
import argparse
import ast
import subprocess
import statistics as st
from pathlib import Path
from experiments.tma_reduction_layout.phase11.c32 import common as c
from experiments.tma_reduction_layout.phase11.c32 import data, resources, composite, regression as reg
from experiments.tma_reduction_layout.phase11.c32.model import predict
from experiments.tma_reduction_layout.phase11.c32.freeze import audit


def close(a, b):
    assert abs(a-b) <= 1e-7*max(1., abs(a), abs(b)), (a, b)


def run(pre_freeze=False, committed=False):
    c.prior.protect_prior()
    evidence = c.read(c.prior.OUT/'validation.json')
    assert evidence['status'] == 'OK' and not evidence['D1_compiled'] and not evidence['D1_timed']
    protocol = c.protocol()
    assert not protocol['global_grid_feature'] and not protocol['historical_26_6_used']
    assert not protocol['uncertainty']['reflection'] and protocol['uncertainty']['one_sided']
    assert c.sha((c.SOURCE/'PLAN.md').read_bytes()) == protocol['plan_SHA256']
    binding = c.read(c.prior.OUT/'preregistration/C32_method_binding.json')
    assert c.sha(c.encode(protocol)) == binding['C32_protocol_SHA256']
    theta = c.read(c.OUT/'theta_H100.json')
    assert theta['physical_base_identified'] is False and not theta['historical_26_6_used']
    assert not theta['selector_uses_runtime_grid'] and not theta['D1_compiled'] and not theta['D1_timed']
    assert theta['selected_recipe'] in protocol['recipes']
    assert theta['selected_composition'] in protocol['composition_models']
    assert theta['model']['kind'] == theta['selected_composition']
    bundle = data.load()
    assert bundle['raw_analysis_SHA256'] == theta['raw_C2c_analysis_SHA256']
    protected = {tuple(row) for row in protocol['protected_D1']}
    for row in bundle['steady']+bundle['full']:
        assert (row['case']['BM'], row['case']['BN'], row['case']['W']) not in protected
    assert len(theta['outer_folds']) == len(theta['full_timing_geometries']) == 8
    residuals, conditional = [], []
    for fold in theta['outer_folds']:
        held = fold['held']
        assert held in fold['model']['excluded_geometries']
        assert held not in fold['model']['MC']['training_geometries']
        assert held not in fold['model']['full_training_geometries']
        assert held not in fold['resource_model']['training_geometries']
        assert held in fold['resource_model']['excluded_geometries']
        rows = [row for row in bundle['full'] if row['geometry'] == held and row['q_condition'] == 'natural']
        predictions = []
        controlled = []
        for row in rows:
            q = resources.q_for(row['case'], resources.point(fold['resource_model'], row['case']))
            predictions.append(composite.predict(fold['model'], row['case'], q, theta['fixed_R_tables'])['tile_ns'])
            controlled.append(composite.predict(fold['model'], row['case'], row['q'], theta['fixed_R_tables'])['tile_ns'])
        residuals.extend(data.paired(rows, predictions))
        conditional.extend(data.paired(rows, controlled))
    assert residuals == theta['paired_outer_residuals']
    assert conditional == theta['paired_cost_residuals']
    close(min(row['delta_residual_ns'] for row in conditional), theta['paired_cost_residual_interval_ns'][0])
    close(max(row['delta_residual_ns'] for row in conditional), theta['paired_cost_residual_interval_ns'][1])
    delta = [row['delta_residual_ns'] for row in residuals]
    close(min(delta), theta['paired_delta_residual_interval_ns'][0])
    close(max(delta), theta['paired_delta_residual_interval_ns'][1])
    from collections import defaultdict
    positive = defaultdict(list)
    for row in conditional:
        positive[row['geometry']].append(max(0., row['delta_residual_ns'])**2)
    expected_margin = max(5., st.mean(st.mean(values) for values in positive.values())**.5)
    close(expected_margin, theta['one_sided_additional_margin_ns'])
    model = theta['model']
    for part in ('MC', 'full', 'base', 'scaffold'):
        if part in model:
            assert all(value >= -1e-9 for value in model[part]['coefficients'])
            assert not any('grid' in name.lower() or '26.6' in name or 'P_shape' in name
                           for name in model[part]['feature_names'])
    resource = c.read(c.OUT/'resource_calibration.json')
    recounted, unique = resources.census()
    assert recounted == resource['rows'] and unique == resource['unique']
    assert theta['resource_model'] == resource['full_single_pass_model']
    assert resource['full_single_pass_model']['R_support'][1] >= 256
    big = [row for row in unique if row['family'] == 'full_single_pass' and row['case']['BM']*row['case']['BN']//(32*row['case']['W']) >= 128]
    assert big and all(not row['local_bytes'] for row in big)
    plans = c.read(c.OUT/'D1_predictions.json')
    history = c.read(c.OUT/'history_audit.json')
    assert history == audit()
    assert plans['count'] == len(plans['cases']) == len(protected) == 11
    assert plans['theta_SHA256'] == c.sha((c.OUT/'theta_H100.json').read_bytes())
    assert {tuple(row['geometry']) for row in plans['cases']} == protected
    assert sum(row['stratum'] == 'empirically_unseen_core' for row in plans['cases']) == 6
    assert sum(row['history_category'] == 'strict_zero_mention' and row['stratum'] == 'empirically_unseen_core' for row in plans['cases']) == 5
    decisions = 0
    for row in plans['cases']:
        assert row['prediction'] == predict(row['geometry'], theta)
        assert row['history_category'] != 'executed_or_inspected'
        for candidate in row['prediction']['candidates']:
            ce = candidate['resource']
            assert ce['interval_scope'].startswith('Held-geometry')
            if candidate['robust_switch']:
                assert candidate['worst_improvement_percent'] > 3.
                assert not ce['spill_uncertainty'] and ce['shared_legal']
                assert candidate['performance_R_extrapolation'] == 1.
        decisions += 1
    development = c.read(c.OUT/'development.json')
    for decision in development['decisions']:
        assert predict(decision['geometry'], theta) == decision
    from experiments.tma_reduction_layout.phase11.c32.evaluate import score, observed
    for decision, result in zip(development['decisions'], development['evaluation']):
        assert score(decision, observed(bundle, decision['geometry'])) == result
    # Pure selector modules cannot invoke GPU/compiler/network code. The freeze
    # tool uses subprocess only for the read-only git parent binding.
    for name in ('model.py', 'features.py', 'composite.py', 'resources.py', 'regression.py'):
        tree = ast.parse((c.SOURCE/name).read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert not any(value and value.split('.')[0] in ('triton', 'torch', 'modal', 'ctypes', 'subprocess') for value in imports), (name, imports)
    assert not list(c.OUT.rglob('*.cubin')) and not list(c.OUT.rglob('*.ptx')) and not list(c.OUT.rglob('*.ncu-rep'))
    if not pre_freeze:
        frozen = c.read(c.OUT/'freeze.json')
        for name, digest in frozen['files'].items():
            assert c.sha((c.ROOT/name).read_bytes()) == digest, name
        assert not frozen['GPU_allocations_in_C32'] and not frozen['D1_compiled'] and not frozen['D1_timed']
        assert frozen['C2c_validation_SHA256'] == c.sha((c.prior.OUT/'validation.json').read_bytes())
        relative = str((c.prior.OUT/'validation.json').relative_to(c.ROOT))
        recorded = subprocess.check_output(['git', 'show', frozen['C2c_commit']+':'+relative], cwd=c.ROOT)
        assert c.sha(recorded) == frozen['C2c_validation_SHA256']
    if committed:
        diff = subprocess.check_output(['git', 'diff', 'HEAD', '--', str(c.SOURCE.relative_to(c.ROOT)), str(c.OUT.relative_to(c.ROOT))], cwd=c.ROOT)
        assert not diff, 'Committed C3.2 changed'
    result = {'stage': 'C3.2', 'status': 'OK', 'pre_freeze': pre_freeze,
        'outer_held_geometries': 8, 'paired_residuals_recomputed': len(residuals),
        'D1_predictions_recomputed': decisions, 'D1_protected_geometries': 11,
        'resource_records': len(recounted), 'unique_resource_CUBIN_families': len(unique),
        'signed_residuals_reflected': False, 'runtime_grid_selector_feature': False,
        'native_rebuilds': 0, 'C32_GPU_allocations': 0, 'D1_compiled': False, 'D1_timed': False,
        'scope': 'Evidence, split, numerical recomputation and frozen choices PASS; D1 efficacy and physical component identification remain untested.'}
    if not pre_freeze:
        c.derived(c.OUT/'validation.json', result)
    print('C3.2 validator OK;', len(residuals), 'outer paired residuals;', decisions,
          'prepared' if pre_freeze else 'frozen', 'predictions; D1 untouched')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre-freeze', action='store_true')
    parser.add_argument('--committed', action='store_true')
    args = parser.parse_args()
    run(args.pre_freeze, args.committed)
