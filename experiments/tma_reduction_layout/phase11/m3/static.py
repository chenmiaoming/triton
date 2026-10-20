"""Restricted pinned lowering and per-instruction store-sector hypotheses."""
import math
from experiments.tma_reduction_layout.phase11.m2.static import output_plan


def plan(case):
    old=output_plan(case)
    vec=case['register_layout']['size_per_thread'][2]
    tn=case['register_layout']['threads_per_warp'][2]
    k=int(math.log2(vec))
    reachable=old['predicted_fixed_conversion_reach']
    permutation={};mixed=None;cycles=[]
    if reachable=='shuffle':
        # Restricted contiguous-N bases. Source lane bits selected by target
        # warp coordinates (column bits >=5) are projected to zero, matching
        # ownerLaneMask in getWarpLayoutConvertDecomposition. Common zero
        # columns become fixed points; remaining zeros pair in dimension order.
        source=[('r',i,1<<i) for i in range(k)]
        source += [('l',i,(vec<<i) if i<int(math.log2(tn)) and (vec<<i)<32 else 0) for i in range(5)]
        target=[('r',i,0) for i in range(k)]+[('l',i,1<<i) for i in range(5)]
        s={r[:2]:r[2] for r in source};t={r[:2]:r[2] for r in target}
        lookup={value:loc for loc,value in t.items() if value}
        src_zeros=[];dst_zeros=[]
        for loc,value in s.items():
            if value:permutation[loc]=lookup[value]
            elif t[loc]==0:permutation[loc]=loc
            else:src_zeros.append(loc)
            if value!=0 and t[loc]==0:dst_zeros.append(loc)
        assert len(src_zeros)==len(dst_zeros)
        permutation.update(zip(src_zeros,dst_zeros))
        visited=set();mixed=0
        for loc in s:
            if loc in visited:continue
            cycle=[];curr=loc
            while curr not in visited:
                visited.add(curr);cycle.append(list(curr))
                nxt=permutation[curr]
                mixed += curr[0]=='l' and nxt[0]=='r'
                curr=nxt
            cycles.append(cycle)
        assert len(visited)==len(s)
    predicted='register' if reachable=='register' else 'shuffle' if reachable=='shuffle' and mixed<2 else 'shared'
    # Canonical replica selection: lowest linear owner. FP32 vector stores
    # have at most four elements/instruction (128 bits). Count 32-byte sectors
    # per warp/instruction, retaining repeated requests across instructions.
    owners=old['native_owners']
    stores={}
    for col,choices in owners.items():
        warp,lane,reg=min(choices)
        stores.setdefault((warp,reg//4),[]).append((lane,col))
    requests=[]
    for (warp,chunk),values in sorted(stores.items()):
        sectors=sorted({col//8 for lane,col in values})
        requests.append({'warp':warp,'register_chunk':chunk,'sectors':sectors})
    return {'layout_key':case['id'],'minimum_reach':reachable,
        'restricted_mixed_transpositions':mixed,'permutation_cycles':cycles,
        'predicted_fixed_lowering':predicted,
        'predicted_native_store_sectors_per_CTA':sum(len(r['sectors']) for r in requests),
        'predicted_fixed_store_sectors_per_CTA':case['BN']//8,
        'store_requests':requests,
        'assumptions':'selected contiguous-N Blocked/Slice encodings; 128-bit max store, aligned FP32 output; not a general lowering predictor'}
