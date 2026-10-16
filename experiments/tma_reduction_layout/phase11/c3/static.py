"""Executable restricted LinearLayout/address and reduction analysis.

No artifact/time/counter imports. Rank-3 batch-one BF16, N-first blocked
bases and 128B nvmma core tiles only. Unsupported encodings fail closed.
"""
from collections import defaultdict
from math import ceil, log2


def layout(m,n,w,v,placement='n_first'):
    if placement == 'within':
        if n % (v*w):
            return None
        tn = n//(v*w)
        if tn>32 or tn<1:
            return None
        tm,wn,wm=32//tn,w,1
    else:
        tn=min(32,n//v)
        tm=32//tn
        wn=min(w,n//(v*tn))
        wm=w//wn
    if tm*wm>m or m%(tm*wm):
        return None
    return {'size_per_thread':[1,1,v],'threads_per_warp':[1,tm,tn],
            'warps_per_cta':[1,wm,wn],'order':[2,1,0]}


def candidates(m,n,w):
    result=[]
    for v in (1,2,4,8):
        for pl in ('n_first','within'):
            ll=layout(m,n,w,v,pl)
            if ll is None or any(x['register_layout']==ll for x in result):
                continue
            result.append({'BM':m,'BN':n,'W':w,'dtype':'bfloat16',
                'reduce_axis':1,'register_layout':ll,'placement':pl,
                'shared_layout':{'swizzle_byte_width':128,'element_bitwidth':16,
                                 'rank':3,'transposed':False},
                'id':f'M{m}_N{n}_w{w}:vec{v}:{pl}'})
    return result


def upstream(m,n,w):
    # Coalesce.cpp::pickDescriptorLoadStoreLayout, pinned 20d9d98.
    v=min(max(m*n//(32*w),1),8)
    return next(x['id'] for x in candidates(m,n,w)
                if x['register_layout']==layout(m,n,w,v))


def coordinates(m,n,w,ll):
    v=ll['size_per_thread'][2]
    u,tn=ll['threads_per_warp'][1:]
    wm,wn=ll['warps_per_cta'][1:]
    # Register bases: contiguous N bits then repeated M bits.
    for rm in range(m//(u*wm)):
        for rn in range(n//(v*tn*wn)):
            for warp in range(w):
                yield rm,rn,warp,[(lane//tn+(warp//wn)*u+rm*u*wm,
                               (lane%tn)*v+(warp%wn)*tn*v+rn*v*tn*wn)
                              for lane in range(32)]


def shared_offset(m,n,bn):
    # Invert getCoreMatrixLinearLayout (8 rows x 64 BF16 columns), then
    # extend the column basis before the row basis, as buildNvmmaSharedLL.
    return (m//8)*8*bn+(n//64)*512+(m%8)*64+((n%64)^((m%8)*8))


def access(m,n,w,ll):
    v=ll['size_per_thread'][2]
    ideal=actual=instructions=0
    transactions=[]
    for rm,rn,warp,coords in coordinates(m,n,w,ll):
        instructions+=1
        group=min(32,128//(2*v))
        for start in range(0,32,group):
            banks=defaultdict(set)
            addresses=[]
            for row,col in coords[start:start+group]:
                words={shared_offset(row,col+i,n)//2 for i in range(v)}
                for word in words:
                    banks[word%32].add(word)
                addresses.append([2*shared_offset(row,col+i,n) for i in range(v)])
            # Per 128-byte hardware subtransaction; same-word reads broadcast.
            waves=max(map(len,banks.values()))
            ideal+=1
            actual+=waves
            transactions.append({'repeat_M':rm,'repeat_N':rn,'warp':warp,'lane_begin':start,
                'lane_count':group,'byte_addresses':addresses,
                'max_distinct_words_per_bank':waves})
    return {'vector_bytes_per_lane':2*v,'warp_load_instructions':instructions,
            'ideal_wavefronts':ideal,'wavefronts':actual,
            'bank_conflicts':actual-ideal,'transactions':transactions}


def plan(case):
    m,n,w=case['BM'],case['BN'],case['W']
    ll=case['register_layout']
    assert case['dtype']=='bfloat16' and case['reduce_axis']==1
    assert case['shared_layout']=={'swizzle_byte_width':128,'element_bitwidth':16,
                                  'rank':3,'transposed':False}
    assert ll['order']==[2,1,0] and ll['size_per_thread'][:2]==[1,1]
    assert ll==layout(m,n,w,ll['size_per_thread'][2],case['placement'])
    v=ll['size_per_thread'][2]
    u=ll['threads_per_warp'][1]
    wm=ll['warps_per_cta'][1]
    p=m//(u*wm)
    columns=n//(ll['threads_per_warp'][2]*ll['warps_per_cta'][2])
    # getInterWarpReductionLayout swaps warp-axis bases into zero lane bases.
    # A return conversion is needed if insufficient lane broadcasts exist.
    comm_rounds=0 if wm==1 else (1 if wm<=u else 2)
    mr_wf=comm_rounds*w*columns
    bars=2*comm_rounds*w*ceil(columns/4)
    shfl=w*columns*int(log2(u*wm))
    # Packed max is possible only for a nontrivial local tree; backend
    # instruction packing is an interval, not claimed equal to PTX sites.
    local_scalar=w*columns*(p-1)
    # ptxas can also fuse a two-input tree into VHMNMX (three inputs), so
    # two BF16 lanes and ternary fusion permit up to four scalar equivalents.
    local_packed_min=w*ceil(columns*(p-1)/4)
    output_pairs=[]
    for rn in range(columns//v):
        for warp in range(w):
            for lane in range(32):
                col=(lane%ll['threads_per_warp'][2])*v+(warp%ll['warps_per_cta'][2])*v*ll['threads_per_warp'][2]+rn*v*ll['threads_per_warp'][2]*ll['warps_per_cta'][2]
                output_pairs += [(col+i,warp*32+lane,rn*v+i) for i in range(v)]
    output_owners={col:{thread for c,thread,_ in output_pairs if c==col}
                   for col in range(n)}
    same_thread=all((col%(32*w)) in output_owners[col] for col in range(n))
    data=access(m,n,w,ll) if m>=8 and n>=64 else None
    # Static liveness range: raw BF16 and widened values may overlap partially;
    # tree temporaries, addresses, descriptor and outputs are bounded separately.
    elements=m*n//(32*w)
    reg_low=max(8,ceil(elements/4)+ceil(columns/2)+8)
    reg_high=2*elements+2*columns+48
    # Single pass may share a scratch lifetime with the input; report both.
    tile_bytes=2*m*n
    scratch=0 if wm==1 else max(4*n*wm,4*n)
    smem_low=tile_bytes+8
    smem_high=tile_bytes+scratch+1024
    def capacity(reg,smem):
        rounded=ceil(reg*32/256)*256*w
        return max(0,min(32,64//w,65536//rounded,233472//(ceil(smem/256)*256)))
    qlo=capacity(min(reg_high,255),smem_high)
    qhi=capacity(reg_low,smem_low)
    return {'id':case['id'],'p':p,'u':u,'w':wm,'e_vector':v,
        'elements_per_thread':elements,'columns_per_thread':columns,
        'ML':data,'MR':{'shuffle_warp_instructions':shfl,
            'barrier_warp_instructions':bars,'shared_load_wavefronts':mr_wf,
            'shared_conversion_rounds':comm_rounds,
            'local_max_scalar_equivalents':local_scalar,
            'local_max_dynamic_interval':[local_packed_min,local_scalar],
            'local_PTX_hypotheses':['max.bf16','max.bf16x2'] if p>1 else [],
            'cross_thread_max_f32':shfl,
            'scratch_upper_bytes':scratch},
        'output':{'native':'SliceLayout(axis=1,parent=register_layout)',
                  'fixed':'Blocked([1],[32],[W],[0])',
                  'requires_thread_redistribution':not same_thread,
                  'warp_redistribution_possible':wm>1 or v>1},
        'resources':{'registers_per_thread_interval':[reg_low,reg_high],
            'shared_bytes_interval':[smem_low,smem_high],
            'q_interval':[qlo,qhi],
            'spill_possible':reg_high>255,'definite_register_infeasible':reg_low>255,
            'static_shape_legal':m>=8 and n>=64,
            'shared_legal':smem_high<=232448,
            'assumption':'single buffer, nonpipelined; unknown RA increases uncertainty'}}


def compact(p):
    return {**p,'ML':{k:v for k,v in p['ML'].items() if k!='transactions'} if p['ML'] else None}
