"""Bounded compile-only pilot; these are semantic fixtures, not model traces."""
from copy import deepcopy
from collections import Counter


def cases():
    rows = [
        dict(id='C01', kernel='column_reduce', M=128, N=4096, BM=32, BN=128, W=4,
             MODE='dwdb', signature=dict(a='*fp32', b='*fp32', weights='*fp32', out='*fp32', aux='*fp32')),
        dict(id='C02', kernel='weighted_pv', M=64, N=128, BM=32, BN=128, W=4,
             signature=dict(prob='*fp32', values='*bf16', out='*fp32')),
        dict(id='C03', kernel='column_quant', M=32, N=128, BM=32, BN=128, W=4,
             signature=dict(x='*bf16', quantized='*fp8e4nv', scale_out='*fp32')),
        dict(id='C04', kernel='rms_backward', M=19, N=128, BM=16, BN=128, W=4,
             signature=dict(x='*bf16', dy='*bf16', gamma='*bf16', rstd='*fp32', dx='*bf16', partial_dw='*fp32')),
        dict(id='C05', kernel='column_reduce', M=257, N=1024, BM=32, BN=128, W=4,
             MODE='bias', signature=dict(a='*bf16', b='*bf16', weights='*fp32', out='*fp32', aux='*fp32')),
        dict(id='C06', kernel='segment_merge', M=13, N=128, BM=16, BN=128, W=4,
             signature=dict(partial='*fp32', maxima='*fp32', sums='*fp32', out='*fp32')),
        dict(id='C07', kernel='recurrent_gdn', M=128, N=128, BM=8, BN=128, W=1,
             STEPS=4, signature=dict(state='*fp32', queries='*bf16', keys='*bf16', values='*bf16',
                                      gates='*fp32', beta='*fp32', out='*bf16', updated='*fp32')),
        dict(id='C08', kernel='column_reduce', M=193, N=768, BM=32, BN=128, W=4,
             MODE='pool_mean', signature=dict(a='*bf16', b='*bf16', weights='*fp32', out='*fp32', aux='*fp32')),
    ]
    for row in rows:
        row['scope'] = 'semantic_fixture'
        row['production_adapter_verified'] = False
        row['groups_for_correctness'] = 2
    # A max-pooling specialization also checks the masked max consumer path.
    extra = deepcopy(rows[-1])
    extra.update(id='C08_max', parent='C08', MODE='pool_max')
    return rows + [extra]


def constants(case):
    return {k: case[k] for k in ('M', 'N', 'BM', 'BN', 'MODE', 'STEPS') if k in case}


def layouts(bm, bn, warps):
    """Enumerate legal 2D N-first/within variants, retaining logical aliases."""
    result = []
    for vec in (1, 2, 4, 8):
        for placement in ('n_first', 'within'):
            if placement == 'within':
                if bn % (vec * warps):
                    continue
                tn = bn // (vec * warps)
                if not 1 <= tn <= 32:
                    continue
                tm, wm, wn = 32 // tn, 1, warps
            else:
                tn = min(32, bn // vec)
                tm = 32 // tn
                wn = min(warps, bn // (vec * tn))
                wm = warps // wn
            if tm * wm > bm or bm % (tm * wm):
                continue
            layout = dict(size_per_thread=[1, vec], threads_per_warp=[tm, tn],
                          warps_per_cta=[wm, wn], order=[1, 0])
            name = f'vec{vec}:{placement}'
            duplicate = next((x for x in result if x['layout'] == layout), None)
            if duplicate is not None:
                duplicate['aliases'].append(name)
            else:
                result.append(dict(name=name, layout=layout, aliases=[]))
    return result


def ownership(bm, bn, warps, layout):
    """CPU coverage check for these restricted nonredundant blocked layouts."""
    sm, sn = layout['size_per_thread']
    tm, tn = layout['threads_per_warp']
    wm, wn = layout['warps_per_cta']
    if sm != 1 or tm * tn != 32 or wm * wn != warps or layout['order'] != [1, 0]:
        raise ValueError('Unsupported layout in preparation coverage check')
    owners = {}
    for rbase in range(0, bm, tm * wm):
        for cbase in range(0, bn, sn * tn * wn):
            for warp in range(warps):
                for lane in range(32):
                    row = rbase + lane // tn + (warp // wn) * tm
                    first = cbase + (lane % tn) * sn + (warp % wn) * sn * tn
                    for i in range(sn):
                        col = first + i
                        if row < bm and col < bn:
                            if (row, col) in owners:
                                raise ValueError('Duplicated ownership')
                            owners[row, col] = (warp, lane)
    if len(owners) != bm * bn:
        raise ValueError('Incomplete ownership')
    counts = Counter(owners.values())
    return {'logical_elements': len(owners), 'physical_threads': 32 * warps,
            'min_elements_per_thread': min(counts.values()),
            'max_elements_per_thread': max(counts.values())}


def launch_grid(case, groups):
    if case['kernel'] in ('rms_backward', 'recurrent_gdn'):
        return [groups, (case['M'] + case['BM'] - 1) // case['BM'], 1]
    return [groups, (case['N'] + case['BN'] - 1) // case['BN'], 1]
