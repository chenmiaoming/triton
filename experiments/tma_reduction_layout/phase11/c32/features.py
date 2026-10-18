"""Static operation demand and independently measured fixed-R net contrasts."""
from functools import lru_cache
from math import log2
import json
import numpy as np
from experiments.tma_reduction_layout.phase11.c3.static import plan

_local_cache = {}
_local_index = {}


@lru_cache(maxsize=512)
def cached_plan(encoded):
    return plan(json.loads(encoded))


def static(case):
    return cached_plan(json.dumps(case, sort_keys=True))


def local_query(tables, r, p, q, variant):
    key = (id(tables), r, p, q, variant)
    if key in _local_cache:
        return _local_cache[key]
    if id(tables) not in _local_index:
        _local_index[id(tables)] = {
            (rr, pp, vv): sorted([row for row in tables if row['R'] == rr and row['p'] == pp and row['variant'] == vv],
                                  key=lambda row: 1/row['q'])
            for rr, pp, vv in {(row['R'], row['p'], row['variant']) for row in tables}}
    index = _local_index[id(tables)]
    sizes = sorted({rr for rr, _, _ in index})
    near = min(sizes, key=lambda x: abs(log2(x/r)))
    pp = min(max(1, p), near)
    pp = min({pp for rr, pp, _ in index if rr == near}, key=lambda x: abs(log2(x/pp)))
    rows = index[near, pp, variant]
    assert rows, (r, p, variant)
    point = float(np.interp(1/q, [1/row['q'] for row in rows], [row['point_ns'] for row in rows]))
    # Out-of-support scaling is labelled by predict(), never called measured.
    _local_cache[key] = point*r/near
    return _local_cache[key]


def features(case, q, recipe, tables, mode='full'):
    p = static(case)
    w, columns, r = case['W'], p['columns_per_thread'], p['elements_per_thread']
    qq = q*w/4 if recipe['local_q_scaling'] == 'warp_normalized' else q
    native = local_query(tables, r, p['p'], qq, 'native')
    fixture = local_query(tables, r, p['p'], qq, 'fixture')
    wide = local_query(tables, r, p['p'], qq, 'wide')
    fixture_wide = local_query(tables, r, p['p'], qq, 'fixture_wide')
    result = {
        'CTA_warps': float(w), 'ML_wavefronts': float(p['ML']['wavefronts']),
        'ML_warp_instructions': float(p['ML']['warp_load_instructions']),
        'local_max_equivalents': float(p['MR']['local_max_scalar_equivalents']),
        'local_tree_columns': w*columns*log2(p['p']),
        'widened_partial_columns': float(w*columns),
        'MR_shuffle': float(p['MR']['shuffle_warp_instructions']),
        'MR_wavefronts': float(p['MR']['shared_load_wavefronts']),
        'MR_barriers': float(p['MR']['barrier_warp_instructions']),
        'MR_warp_shuffle_columns': w*columns*log2(p['u']),
        'MR_cross_warp_columns': w*columns*log2(p['w']),
        'MR_lane_fanout_columns': float(w*columns*(p['u']-1)),
        'MO_thread_redistribution': float(w*p['output']['requires_thread_redistribution']),
        'MO_warp_redistribution': float(w*p['output']['warp_redistribution_possible']),
        'fixed_R_native_net': (native-fixture)*w/4,
        'fixed_R_wide_net': (wide-fixture_wide)*w/4,
        'fixed_R_fixture': fixture*w/4,
    }
    for name in ('CTA_warps', 'ML_warp_instructions', 'local_tree_columns',
                 'MR_shuffle', 'MR_barriers', 'widened_partial_columns', 'MR_warp_shuffle_columns',
                 'MO_thread_redistribution', 'MO_warp_redistribution'):
        result[name+'__inverse_q'] = result[name]/q
    if mode == 'Load-blind':
        result = {key: value for key, value in result.items() if not key.startswith('ML_')}
    elif mode == 'Comm-blind':
        result = {key: value for key, value in result.items() if not key.startswith('MR_')}
    elif mode == 'Fixed-R-blind':
        result = {key: value for key, value in result.items() if not key.startswith('fixed_R_')}
    return result


def transfer_features(geometry):
    m, n, w = geometry
    # N/64 is a dimensionless column-extent proxy. It is not a count of
    # executed TMA instructions, transactions, or DRAM requests.
    return {'CTA': 1., 'input_128B': 2*m*n/128, 'output_128B': 4*n/128,
            'N_64col_segments': n/64, 'warps': float(w)}


def resource_features(case):
    p = static(case)
    return {'fixed': 1., 'BF16_packed_elements': p['elements_per_thread']/2,
            'partial_columns': float(p['columns_per_thread']),
            'local_tree_depth': log2(p['p']), 'warps': float(case['W']),
            'shuffle_depth': log2(p['u']*p['w']),
            'shared_conversion_rounds': float(p['MR']['shared_conversion_rounds']),
            'output_redistribution': float(p['output']['requires_thread_redistribution'])}
