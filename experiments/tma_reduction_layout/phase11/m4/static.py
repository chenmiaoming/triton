"""Restricted, compile-time M4 operation plans; no timings or artifacts imported."""
from collections import defaultdict
from experiments.tma_reduction_layout.phase11.c3 import static as input_plan
from experiments.tma_reduction_layout.phase11.m2.static import output_plan
from experiments.tma_reduction_layout.phase11.m3.static import plan as conversion_plan


GEOMETRIES = [(32, 64, 8), (32, 128, 4), (16, 256, 8)]
CONSUMERS = ['fixed', 'native', 'broadcast', 'centered']


def shared_offset(row, col, bn, swizzle):
    if swizzle == 0:
        return row * bn + col
    assert swizzle in (32, 64, 128)
    core = swizzle // 2
    # Pinned getVec=8, perPhase=128/swizzle, maxPhase=swizzle/16.
    xor = 8 * ((row % 8 // (128 // swizzle)) % (swizzle // 16))
    return (row // 8) * 8 * bn + (col // core) * (8 * core) + (row % 8) * core + ((col % core) ^ xor)


def load_plan(case, swizzle):
    m, n, w = case['BM'], case['BN'], case['W']
    ll = case['register_layout']
    vec = ll['size_per_thread'][2]
    ideal = actual = instructions = 0
    for rm, rn, warp, coordinates in input_plan.coordinates(m, n, w, ll):
        instructions += 1
        group = min(32, 128 // (2 * vec))
        for start in range(0, 32, group):
            banks = defaultdict(set)
            for row, col in coordinates[start:start + group]:
                for offset in range(vec):
                    word = shared_offset(row, col + offset, n, swizzle) // 2
                    banks[word % 32].add(word)
            ideal += 1
            actual += max(map(len, banks.values()))
    return {'warp_load_instructions_if_vec_preserved': instructions,
            'vector_bytes_per_lane_assumed': 2 * vec, 'ideal_wavefronts': ideal,
            'wavefronts': actual, 'bank_conflicts': actual - ideal,
            'qualification': 'Address prediction assumes emitted load grouping preserves logical vec; verify actual SASS/NCU.'}


def dense_stores(case):
    m, n, w = case['BM'], case['BN'], case['W']
    ll = case['register_layout']
    vec = ll['size_per_thread'][2]
    requests = []
    demands = defaultdict(set)
    for rm, rn, warp, coordinates in input_plan.coordinates(m, n, w, ll):
        for offset in range(0, vec, 4):
            sectors = set()
            for lane, (row, col) in enumerate(coordinates):
                for element in range(offset, min(offset + 4, vec)):
                    sectors.add((row * n + col + element) // 8)
                    demands[col + element].add((warp, lane))
            requests.append({'repeat_M': rm, 'repeat_N': rn, 'warp': warp,
                             'vector_chunk': offset // 4, 'sectors': sorted(sectors)})
    return requests, {str(col): [list(thread) for thread in sorted(threads)] for col, threads in demands.items()}


def plan(case, consumer='native', swizzle=128):
    assert consumer in CONSUMERS
    old = output_plan(case)
    cvt = conversion_plan(case)
    requests, all_demands = dense_stores(case)
    if consumer == 'fixed':
        demands = {str(col): [thread[:2]] for col, thread in old['fixed_owners'].items()}
    elif consumer == 'native':
        demands = {str(col): [min(choices)[:2]] for col, choices in old['native_owners'].items()}
    else:
        demands = all_demands
    dense = consumer in ('broadcast', 'centered')
    sectors = sum(len(row['sectors']) for row in requests) if dense else cvt[
        'predicted_fixed_store_sectors_per_CTA' if consumer == 'fixed' else 'predicted_native_store_sectors_per_CTA']
    fanin = case['register_layout']['warps_per_cta'][1]
    required_warps = {col: sorted({thread[0] for thread in threads}) for col, threads in demands.items()}
    return {'id': case['id'], 'geometry': [case['BM'], case['BN'], case['W']],
            'consumer': consumer, 'shared_swizzle_bytes': swizzle,
            'register_layout': case['register_layout'], 'ML': load_plan(case, swizzle),
            'MR_structural_plan': old['ML_MR_plan']['MR'],
            'output_conversion': cvt if consumer == 'fixed' else {'predicted_fixed_lowering': 'none'},
            'store': {'output_elements': case['BM'] * case['BN'] if dense else case['BN'],
                      'output_dtype': 'fp32', 'max_bits_per_instruction_assumed': 128,
                      'predicted_sector_requests_per_CTA': sectors,
                      'dense_warp_instruction_groups': requests if dense else cvt['store_requests']},
            'consumer_demand': {'threads_by_column': demands, 'warps_by_column': required_warps,
                                'all_reduction_replicas_needed': dense,
                                'cross_warp_return_can_be_pruned': not dense and fanin > 1,
                                'pruning_scope': 'Eligibility/required owners, not a count of emitted return LDS.'},
            'local_liveness': {'wide_input_uses': 2 if consumer == 'centered' else 1,
                               'single_use_BF16_max_fold_pattern_preserved': consumer != 'centered',
                               'qualification': 'Centered consumer can change max lowering, live values and registers; inspect rather than assume packing/q.'},
            'scope': 'Restricted contiguous-N rank3 BF16 Blocked/Slice layouts; structural predictions, no time cost or selector.'}
