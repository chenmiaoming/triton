"""SGLang's original NVIDIA HND cache scatter, with unchanged launch geometry."""
from . import common as c


def cases():
    return [dict(id=f'N06_h{h}_d{d}',family='N06',W=4,stages=3,T=2048,H=h,D=d,block_size=16,
                 caller_options='W/stages omitted upstream; pinned backend defaults') for h,d in [(8,64),(32,128)]]


def upstream(case):
    from .sources.cache_upstream import reshape_and_cache_flash
    return reshape_and_cache_flash


def fixture(case,pattern='finite'):
    import torch
    torch.set_num_threads(2);g=torch.Generator().manual_seed(20261105+case['D'])
    t,h,d=case['T'],case['H'],case['D'];b=case['block_size'];blocks=(t+b-1)//b+1
    shape=(t,h,d)
    if pattern=='structured':
        key=((torch.arange(t*h*d).reshape(shape)%31)-15).to(torch.bfloat16)/16
        value=-key
    else:
        key=(torch.randn(shape,generator=g)*.125).bfloat16();value=(torch.randn(shape,generator=g)*.125).bfloat16()
    physical=torch.full((blocks,h,b,d),-7.,dtype=torch.bfloat16)
    kc=physical.permute(0,2,1,3);vc=physical.clone().permute(0,2,1,3)
    slots=torch.randperm(t,generator=g).int()
    if pattern=='edge':slots[::17]=-1
    ek=kc.clone();ev=vc.clone();valid=slots>=0
    idx=slots[valid].long();ek[idx//b,idx%b]=key[valid];ev[idx//b,idx%b]=value[valid]
    tensors=dict(key_ptr=key,value_ptr=value,key_cache_ptr=kc,value_cache_ptr=vc,
                 slot_mapping_ptr=slots,k_scale_ptr=key,v_scale_ptr=key)
    scalars=dict(swa_slot_mapping_ptr=None,block_stride=kc.stride(0),value_block_stride=vc.stride(0),
                 key_stride=key.stride(0),value_stride=value.stride(0),num_heads=h,head_size=d,block_size=b)
    const=dict(cache_token_stride=kc.stride(1),cache_head_stride=kc.stride(2),
               value_cache_token_stride=vc.stride(1),value_cache_head_stride=vc.stride(2),
               HEAD_BLOCK=4,BLOCK_D=d,HAS_SWA=False,USE_SCALE=False)
    return dict(tensors=tensors,scalars=scalars,constants=const,grid=(t,(h+3)//4,1),
                outputs=['key_cache_ptr','value_cache_ptr'],expected=dict(key_cache_ptr=ek,value_cache_ptr=ev))


def derive_plan(ir,case):
    from ..operator_gpu.layout_parsing import encodings,tensor_encoding
    defs=encodings(ir);lines=[x for x in ir.splitlines() if 'tt.load' in x and f'tensor<4x{case["D"]}x' in x]
    return dict(input=tensor_encoding(lines[0],[4,case['D']],defs))


def gluon(case):
    from .cache_gluon import reshape_and_cache_flash
    return reshape_and_cache_flash


def layout_constants(case,plan,variant='G0'):
    from triton.experimental.gluon import language as gl
    from ..operator_gpu.layout_parsing import instantiate
    layout=plan['input'] if variant=='G0' else variant['layout']
    return dict(L=instantiate(layout,gl))


def variants(case,plan):
    from copy import deepcopy
    rows=[]
    for e,u in [(1,1),(2,1),(4,1),(8,1),(1,4),(2,4),(4,2),(8,2)]:
        layout=dict(kind='blocked',fields=dict(size_per_thread=[1,e],threads_per_warp=[u,32//u],
                    warps_per_cta=[1,4],order=[1,0]))
        rows.append(dict(name=f'e{e}_heads{u}',layout=layout))
    return rows
