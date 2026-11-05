"""Freeze all empirically unseen core shapes and legacy boundaries, without GPUs."""
from collections import Counter
import subprocess
from experiments.tma_reduction_layout.phase11.c32 import common as c
from experiments.tma_reduction_layout.phase11.c32.model import predict


def audit():
    original = c.read(c.prior.OUT/'preregistration/history_audit.json')
    reserved = c.read(c.prior.OUT/'preregistration/D1_reservation.json')
    gate = c.read(c.prior.OUT/'artifact_gate.json')
    now = {(row['M'], row['N'], row['num_warps']) for row in gate['metadata'].values()
           if row['kind'] != 'local'}
    protected = {tuple(geometry) for geometry in reserved['reserved_geometries']}
    assert not now & protected
    rows = []
    for old in original['rows']:
        geometry = old['BM'], old['BN'], old['W']
        row = {**old, 'pre_C3_category': old['category'],
            'C2c_compiled_or_measured': geometry in now,
            'post_C2c_category': 'executed_or_inspected' if geometry in now else old['category'],
            'D1_reserved': geometry in protected}
        rows.append(row)
    blind = []
    for row in rows:
        if not row['D1_reserved']:
            continue
        assert row['post_C2c_category'] != 'executed_or_inspected'
        blind.append({'geometry': [row['BM'], row['BN'], row['W']], 'R': row['R'],
            'history_category': row['post_C2c_category'],
            'stratum': 'empirically_unseen_core' if row['core_domain'] else 'legacy_boundary',
            'prior_static_rule_prediction': row.get('prior_static_rule_prediction', False),
            'pre_C3_evidence': row['evidence'],
            'claim': 'No recorded timing/compiler inspection; zero mention refers to the historical pre-C3 cutoff.'})
    retrospective = [{'geometry': [row['BM'], row['BN'], row['W']], 'R': row['R'],
                      'history_category': row['post_C2c_category'], 'stratum': 'retrospective_core_coverage'}
                     for row in rows if row['core_domain'] and not row['D1_reserved'] and
                     row['post_C2c_category'] == 'executed_or_inspected']
    return {'historical_cutoff': original['cutoff'], 'rows': rows, 'blind': blind,
        'retrospective_core_panel': retrospective,
        'blind_strata': dict(Counter(row['stratum'] for row in blind)),
        'blind_history_categories': dict(Counter(row['history_category'] for row in blind)),
        'selection': 'All six recorded empirically unseen core geometries plus every legacy boundary; no score-based replacement.',
        'coverage_gap': reserved['coverage_limit'],
        'strict_zero_mention_statistics': 'Evaluate the pre-C3 strict subset separately, then all unseen core, then boundary; overlap is explicit.',
        'retrospective_statistics': 'Previously exposed core coverage is diagnostic and cannot increase the blind sample size.',
        'source_history_audit_SHA256': c.sha(c.encode(original)),
        'D1_reservation_SHA256': c.sha(c.encode(reserved))}


def predictions(value, theta):
    return [{**row, 'prediction': predict(row['geometry'], theta)} for row in value['blind']]


def prepare():
    theta = c.read(c.OUT/'theta_H100.json')
    value = audit()
    rows = predictions(value, theta)
    retrospective = [{**row, 'prediction': predict(row['geometry'], theta)} for row in value['retrospective_core_panel']]
    c.derived(c.OUT/'history_audit.json', value)
    c.derived(c.OUT/'D1_predictions.json', {'version': 'c3_v2', 'stage': 'C3.2',
        'cases': rows, 'count': len(rows), 'no_D1_compilation_or_timing': True,
        'theta_SHA256': c.sha((c.OUT/'theta_H100.json').read_bytes()),
        'historical_categories_are_not_new_results': True})
    c.derived(c.OUT/'retrospective_coverage_predictions.json', {'cases': retrospective,
        'counts': {'BM': dict(Counter(row['geometry'][0] for row in retrospective)),
                   'R': dict(Counter(row['R'] for row in retrospective)),
                   'W': dict(Counter(row['geometry'][2] for row in retrospective))},
        'outcomes_measured_in_this_stage': False, 'blind': False})
    print('C3.2 predictions prepared:', len(rows), 'blind geometries;', len(retrospective), 'retrospective diagnostics')


def freeze():
    paths = sorted(c.SOURCE.glob('*.py'))+[c.SOURCE/'PLAN.md']
    paths += [c.BASE/'phase11/c3/static.py', c.BASE/'phase11/c31/fit.py']
    paths += [c.OUT/name for name in ('theta_H100.json', 'resource_calibration.json', 'calibration_cells.json',
                                    'development.json', 'history_audit.json', 'D1_predictions.json',
                                    'retrospective_coverage_predictions.json', 'REPORT.md')]
    paths += sorted((c.OUT/'figures').glob('*.png'))
    artifacts = {str(path.relative_to(c.ROOT)): c.sha(path.read_bytes()) for path in paths}
    c.write_once(c.OUT/'freeze.json', {'stage': 'C3.2', 'version': 'c3_v2',
        'files': artifacts, 'C2c_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c.ROOT, text=True).strip(),
        'C2c_validation_SHA256': c.sha((c.prior.OUT/'validation.json').read_bytes()),
        'C2c_main_protocol_SHA256': c.sha(c.encode(c.prior.protocol())),
        'method_preregistration_SHA256': c.sha(c.encode(c.protocol())),
        'native_rebuilds': 0, 'GPU_allocations_in_C32': 0,
        'D1_compiled': False, 'D1_timed': False, 'no_PR_created': True})
    print('C3.2 code, parameters and all candidate predictions frozen before D1')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'freeze'])
    prepare() if parser.parse_args().action == 'prepare' else freeze()
