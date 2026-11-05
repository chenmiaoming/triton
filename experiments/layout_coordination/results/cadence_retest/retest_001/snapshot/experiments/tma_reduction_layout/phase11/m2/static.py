"""Restricted output ownership plan, independent of measured timings."""
from experiments.tma_reduction_layout.phase11.c3 import static as base


def output_plan(case):
    """Classify communication reach by enumerating the blocked linear bases.

    Reduction-result SliceLayouts retain zero (replication) M bases. A fixed
    destination can consume any replica. This predicts reach, not latency,
    shared swizzle/vector grouping, or the number of emitted instructions.
    """
    n, w, ll = case['BN'], case['W'], case['register_layout']
    vec = ll['size_per_thread'][2]
    tn, wn = ll['threads_per_warp'][2], ll['warps_per_cta'][2]
    reps = n//(vec*tn*wn)
    owners = {col: [] for col in range(n)}
    for warp in range(w):
        for lane in range(32):
            for repeat in range(reps):
                for v in range(vec):
                    col = (warp % wn)*tn*vec + (lane % tn)*vec + repeat*tn*wn*vec + v
                    owners[col].append([warp, lane, repeat*vec+v])
    fixed = {col: [(col % (32*w))//32, col % 32, col//(32*w)] for col in range(n)}
    same_thread = all(any(a[:2] == fixed[col][:2] for a in choices) for col, choices in owners.items())
    same_warp = all(any(a[0] == fixed[col][0] for a in choices) for col, choices in owners.items())
    reach = 'register' if same_thread else 'shuffle' if same_warp else 'shared'
    return {'predicted_fixed_conversion_reach': reach,
            'native_conversion_reach': 'none',
            'source_layout': 'Slice(0,Slice(1,Blocked))',
            'destination_layout': {'size_per_thread':[1], 'threads_per_warp':[32],
                                   'warps_per_cta':[w], 'order':[0]},
            'native_owners': owners, 'fixed_owners': fixed,
            'native_redundancy_per_value': len(owners[0]),
            'native_unique_values': len(owners),
            'native_store_unique_elements': n,
            'conversion_scope': 'minimum communication reach; backend may use shared even for a warp-local mapping',
            'ML_MR_plan': base.compact(base.plan(case))}
