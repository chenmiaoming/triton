"""Pinned, unreduced rotary kernels and CPU-only fixtures.

Upstream imports stay lazy so the frozen driver process can construct fixtures
without importing Triton. The two source functions are exact AST extractions;
their complete original files and provenance are retained beside them.
"""
from functools import lru_cache
import copy
import re
import importlib.util
from pathlib import Path


SOURCES = Path(__file__).resolve().parent / 'sources'


def cases():
    result = []
    for dim in (64, 128):
        result.append(dict(id=f'N03_FA_D{dim}', family='N03', operator='flash_rope',
            W=4, stages=3, B=4, S=2048, H=32, D=dim, rotary_dim=dim,
            BLOCK_H=2, BLOCK_M=8, layout_rank=3,
            caller_configuration='Original wrapper omits num_warps/num_stages; '
                                 'verify pinned compiler defaults are 4/3.',
            inplace=False, interleaved=True, varlen=False))
    for dim in (64, 128):
        result.append(dict(id=f'N03_MROPE_D{dim}', family='N03', operator='mrope',
            W=4, stages=3, T=2048, QH=32, KH=8, D=dim, rotary_dim=dim,
            sections=[dim//8, 3*dim//16, 3*dim//16], layout_rank=2,
            caller_configuration='Original CUDA wrapper omits num_warps/num_stages; '
                                 'verify pinned compiler defaults are 4/3.',
            inplace=True, interleaved=False, neox=True))
    return result


@lru_cache(None)
def _module(name):
    spec = importlib.util.spec_from_file_location('nonreduction_upstream_'+name,
                                                SOURCES/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def upstream(case):
    if case['operator'] == 'flash_rope':
        return _module('rotary_flash_kernel').rotary_kernel
    assert case['operator'] == 'mrope'
    return _module('rotary_mrope_kernel')._triton_mrope_forward_fused


def _input(shape, pattern, seed):
    import torch
    assert pattern in ('finite', 'structured', 'edge')
    if pattern == 'finite':
        generator = torch.Generator(device='cpu').manual_seed(seed)
        value = torch.randn(shape, generator=generator, dtype=torch.float32)*.125
    else:
        import math
        value = torch.arange(math.prod(shape), dtype=torch.int64).reshape(shape)
        if pattern == 'structured':
            value = ((value % 29)-14).to(torch.float32)/16
        else:
            # Finite, sign-changing and exactly representable boundary inputs.
            levels = torch.tensor([0., -0., 1/128, -1/128, .5, -.5, 2., -2.])
            value = levels[(value % levels.numel()).long()]
    return value.to(torch.bfloat16)


def _trig(length, half_dim, pattern):
    import torch
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    frequency = torch.arange(half_dim, dtype=torch.float32)[None, :]+1
    angle = pos*frequency/257
    cos, sin = torch.cos(angle).to(torch.bfloat16), torch.sin(angle).to(torch.bfloat16)
    if pattern == 'edge':
        # Includes exact zero/one coefficients without changing the source path.
        cos[::7] = 1
        sin[::7] = 0
    return cos, sin


def fixture(case, pattern='finite'):
    import torch
    if case['operator'] == 'flash_rope':
        b, s, h, d = [case[k] for k in ('B','S','H','D')]
        x = _input((b,s,h,d),pattern,20261105+d)
        cos, sin = _trig(s+64,d//2,pattern)
        wide = x.to(torch.float32)
        cb = cos[:s][None,:,None,:].float()
        sb = sin[:s][None,:,None,:].float()
        first, second = wide[...,0::2], wide[...,1::2]
        expected = torch.empty_like(x)
        expected[...,0::2] = (first*cb-second*sb).to(x.dtype)
        expected[...,1::2] = (first*sb+second*cb).to(x.dtype)
        out = torch.full_like(x, -13.)
        scalars = dict(seqlen=s,nheads=h,seqlen_ro=cos.shape[0],
            stride_out_batch=out.stride(0),stride_out_seqlen=out.stride(1),
            stride_out_nheads=out.stride(2),stride_out_headdim=out.stride(3),
            stride_x_batch=x.stride(0),stride_x_seqlen=x.stride(1),
            stride_x_nheads=x.stride(2),stride_x_headdim=x.stride(3),SEQLEN_OFFSETS=0)
        constants = dict(CU_SEQLENS=None,ROTARY_DIM=d,
            IS_SEQLEN_OFFSETS_TENSOR=False,IS_VARLEN=False,INTERLEAVED=True,
            CONJUGATE=False,BLOCK_H=case['BLOCK_H'],BLOCK_M=case['BLOCK_M'])
        return dict(tensors=dict(OUT=out,X=x,COS=cos,SIN=sin),scalars=scalars,
            constants=constants,grid=((h+1)//2,(s+case['BLOCK_M']-1)//case['BLOCK_M'],b),
            outputs=['OUT'],expected={'OUT':expected},inplace=[],
            tolerances={'OUT':dict(rtol=.008,atol=.001,bitwise=False)})
    assert case['operator'] == 'mrope'
    t, qh, kh, d = [case[k] for k in ('T','QH','KH','D')]
    q = _input((t,qh*d),pattern,20261205+d)
    k = _input((t,kh*d),pattern,20261305+d)
    cos, sin = _trig(8192,d//2,pattern)
    cache = torch.cat((cos,sin),dim=1).contiguous()
    sequence = torch.arange(t,dtype=torch.int64)
    positions = torch.stack(((sequence*17+3)%cache.shape[0],
                             (sequence*7+5)%cache.shape[0],
                             (sequence*11+7)%cache.shape[0])).contiguous()
    section_t, section_h, section_w = case['sections']
    assert section_t+section_h+section_w == d//2
    index = torch.arange(d//2)
    axes = torch.where(index<section_t,0,torch.where(index<section_t+section_h,1,2))
    selected_pos = positions[axes[None,:],sequence[:,None]]
    cb = cache[selected_pos,index[None,:]][:,None,:]
    sb = cache[selected_pos,index[None,:]+d//2][:,None,:]
    def reference(value,heads):
        blocks = value.reshape(t,heads,d)
        first, second = blocks[...,:d//2],blocks[...,d//2:]
        # Mirror the original BF16 source operations and their intermediate
        # rounding; the source explicitly converts inputs to sin_row.dtype.
        output = torch.empty_like(blocks)
        output[...,:d//2] = first*cb-second*sb
        output[...,d//2:] = second*cb+first*sb
        return output.reshape_as(value)
    constants = dict(n_qh=qh,n_kh=kh,hd=d,rd=d,pad_n_qh=qh,pad_n_kh=kh,
        pad_hd=d,mrope_section_t=section_t,mrope_section_h=section_h,
        mrope_section_w=section_w,is_interleaved=False,is_interleaved_glm=False,
        is_neox_style=True,axis_map_ptr=None)
    return dict(tensors=dict(q_ptr=q,k_ptr=k,cos_sin_cache_ptr=cache,
                            positions_ptr=positions),
        scalars=dict(q_stride=q.stride(0),k_stride=k.stride(0),positions_stride=positions.stride(0)),
        constants=constants,grid=(t,1,1),outputs=['q_ptr','k_ptr'],
        expected={'q_ptr':reference(q,qh),'k_ptr':reference(k,kh)},
        inplace=['q_ptr','k_ptr'],tolerances={name:dict(rtol=.008,atol=.001,bitwise=False)
                                          for name in ['q_ptr','k_ptr']})


def gluon(case):
    from . import rotary_gluon
    return (rotary_gluon.rotary_kernel if case['operator']=='flash_rope'
            else rotary_gluon._triton_mrope_forward_fused)


def _encoding(line, shape, definitions):
    from ..operator_gpu.layout_parsing import resolve
    # Match the element type immediately after the exact rank/shape. A loose
    # prefix would mistake tensor<32x32x...> for a one-dimensional 32-vector.
    element=r'(?:!tt\.ptr<[^>]+>|bf16|[ifu]\d+)'
    pattern=(r'tensor<'+'x'.join(map(str,shape))+'x'+element+
             r',\s*(#ttg\.slice<\{[^}]+\}>|#\w+)>')
    match=re.search(pattern,line)
    assert match,(shape,line)
    return resolve(match[1],definitions)


def _has_shape(line,shape):
    return re.search(r'tensor<'+'x'.join(map(str,shape))+
                     r'x(?:!tt\.ptr<|bf16\b|[ifu]\d+\b)',line) is not None


def derive_plan(ttgir,case):
    from ..operator_gpu.layout_parsing import encodings
    definitions=encodings(ttgir)
    lines=ttgir.splitlines()
    loads=[line for line in lines if 'tt.load ' in line]
    stores=[line for line in lines if 'tt.store ' in line]
    result=dict(family='N03',operator=case['operator'],layouts=copy.deepcopy(definitions),
        loads=loads,stores=stores,
        conversions=[line for line in lines if 'convert_layout' in line],
        reshapes=[line for line in lines if 'tt.reshape' in line or 'tt.split' in line or 'tt.join' in line])
    if case['operator']=='flash_rope':
        shape=[case['BLOCK_H'],case['BLOCK_M'],case['D']]
        trig_shape=[case['BLOCK_M'],case['D']//2]
        x=next(line for line in loads if _has_shape(line,shape))
        trig=[line for line in loads if _has_shape(line,trig_shape)]
        reshape=next(line for line in lines if 'tt.reshape' in line)
        split=next(line for line in lines if 'tt.split' in line)
        result.update(input=_encoding(x,shape,definitions),
            trig=_encoding(trig[0],trig_shape,definitions),
            compute=_encoding(reshape,shape,definitions),
            pair=_encoding(split,[*shape[:-1],shape[-1]//2],definitions),
            output=_encoding(stores[0],shape,definitions),
            trig_convert_before_widen=any('convert_layout' in line and 'bf16' in line
                                         and 'tensor<'+'x'.join(map(str,trig_shape))+'x' in line for line in lines))
        assert all(_encoding(line,trig_shape,definitions)==result['trig'] for line in trig)
    else:
        qshape=[case['QH'],case['D']//2];kshape=[case['KH'],case['D']//2];trigshape=[case['D']//2]
        def typed(rows,shape):return [line for line in rows if _has_shape(line,shape)]
        qloads=typed(loads,qshape);kloads=typed(loads,kshape);tloads=typed(loads,trigshape)
        qstores=typed(stores,qshape);kstores=typed(stores,kshape)
        result.update(Q=_encoding(qloads[0],qshape,definitions),K=_encoding(kloads[0],kshape,definitions),
            Q_trig=_encoding(tloads[0],trigshape,definitions),
            K_trig=_encoding(tloads[1],trigshape,definitions),
            Q_output=_encoding(qstores[0],qshape,definitions),K_output=_encoding(kstores[0],kshape,definitions),
            trig_load_encodings=[_encoding(line,trigshape,definitions) for line in tloads])
        assert all(_encoding(line,qshape,definitions)==result['Q'] for line in qloads)
        assert all(_encoding(line,kshape,definitions)==result['K'] for line in kloads)
        assert len(tloads)==12
        assert result['Q_trig']==dict(kind='slice',dim=0,parent=result['Q'])
        assert result['K_trig']==dict(kind='slice',dim=0,parent=result['K'])
    return result


def _make_layout(shape,vec,warps):
    rank=len(shape)
    size=[1]*rank;size[-1]=vec
    threads=[1]*rank;remaining=32
    for axis in range(rank-1,-1,-1):
        threads[axis]=min(remaining,max(1,shape[axis]//size[axis]));remaining//=threads[axis]
    threads[0]*=remaining
    warp=[1]*rank;remaining=warps
    for axis in range(rank-1,-1,-1):
        warp[axis]=min(remaining,max(1,shape[axis]//(size[axis]*threads[axis])));remaining//=warp[axis]
    warp[0]*=remaining
    return dict(kind='blocked',fields=dict(size_per_thread=size,threads_per_warp=threads,
        warps_per_cta=warp,order=list(range(rank-1,-1,-1))))


def variants(case,plan):
    result=[]
    if case['operator']=='flash_rope':
        shape=[case['BLOCK_H'],case['BLOCK_M'],case['D']]
        for vec in (1,2,4,8):
            layout=_make_layout(shape,vec,case['W'])
            result.append(dict(name=f'input_vec{vec}',input=layout))
            if vec>=2:
                result.append(dict(name=f'paired_vec{vec}',input=layout,compute=layout))
    else:
        for vec in (1,2,4,8):
            q=_make_layout([case['QH'],case['D']//2],vec,case['W'])
            k=_make_layout([case['KH'],case['D']//2],vec,case['W'])
            result.append(dict(name=f'Q_vec{vec}',Q=q))
            if vec in (2,4):
                result.append(dict(name=f'K_vec{vec}',K=k))
                result.append(dict(name=f'QK_vec{vec}',Q=q,K=k))
    return result


def layout_constants(case,plan,variant='G0'):
    from triton.experimental.gluon import language as gl
    from ..operator_gpu.layout_parsing import instantiate
    choice={} if variant=='G0' else (variant if isinstance(variant,dict) else
        next(item for item in variants(case,plan) if item['name']==variant))
    if case['operator']=='flash_rope':
        fields=dict(LX=choice.get('input',plan['input']),LC=plan['trig'],
            LWORK=choice.get('compute',plan['compute']),LPAIR=plan['pair'],LOUT=plan['output'])
        return {**{name:instantiate(spec,gl) for name,spec in fields.items()},
                'TRIG_CONVERT_BEFORE_WIDEN':plan['trig_convert_before_widen']}
    fields=dict(LQ=choice.get('Q',plan['Q']),LK=choice.get('K',plan['K']),
                LTQ=plan['Q_trig'],LTK=plan['K_trig'],
                LQOUT=plan['Q_output'],LKOUT=plan['K_output'])
    return {name:instantiate(spec,gl) for name,spec in fields.items()}
