"""C3.2 derived-only analysis; no GPU/compiler execution entrypoint."""
from pathlib import Path
from experiments.tma_reduction_layout.phase11.c2c import common as prior

ROOT, BASE = prior.ROOT, prior.BASE
OUT = BASE/'results/phase11/stage_c32'
SOURCE = Path(__file__).resolve().parent
read, encode, sha = prior.read, prior.encode, prior.sha
derived, write_once = prior.derived, prior.write_once


def protocol():
    return read(OUT/'preregistration/protocol.json')


def preregister():
    import subprocess
    protected = read(prior.OUT/'preregistration/D1_reservation.json')
    value = {'stage': 'C3.2', 'date': '2026-10-17',
        'pre_C2c_GPU_calibration_values': True,
        'parent_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'plan_SHA256': sha((SOURCE/'PLAN.md').read_bytes()),
        'protected_D1': protected['reserved_geometries'],
        'split_SHA256': sha(encode(protected)), 'threshold': .03,
        'recipes': [{'local_q_scaling': scaling, 'ridge': ridge}
                    for scaling in ('direct', 'warp_normalized') for ridge in (0., .001, .01)],
        'composition_models': ['zero_base', 'shared_tile_base', 'shared_full_service'],
        'resource_ridge': .001,
        'uncertainty': {'reflection': False, 'one_sided': True,
            'outer_validation': 'leave complete geometry out; nested recipe selection',
            'upper_margin': 'positive held-out paired residual plus max(5ns, geometry-level RMS positive residual) times leverage inflation; finite-data engineering margin, not a guaranteed confidence bound',
            'resource_margin_registers': 8,
            'leverage': 'sqrt(max(1,h_new / median_training_leverage)); report outside-support flags'},
        'global_grid_feature': False, 'historical_26_6_used': False,
        'D1_compiled': False, 'D1_timed': False, 'native_rebuilds': 0,
        'stop': 'After validator, separate commit and push; before D1 compilation/timing.'}
    write_once(OUT/'preregistration/protocol.json', value)
    print('C3.2 method frozen before C2c GPU values;', len(value['protected_D1']), 'holdouts')


if __name__ == '__main__':
    preregister()
