# SPDX-License-Identifier: Apache-2.0 AND MIT
"""Explicit-layout renderings of pinned standalone pack and gate kernels.

No compiler hook or pass alteration is used.  G0 is accepted only after the
root experiment verifies allocated machine-code sections, ABI and resources.
"""
import copy
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from .sources.gate_original import s_softplus, s_constraints, NUM_REG


@gluon.jit
def _divide_rn(numerator, denominator):
    return gl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;", "=f,f,f", [numerator, denominator],
        dtype=gl.float32, is_pure=True, pack=1,
    )


@gluon.jit
def _load_tile(base, head, tokens, columns, valid,
               head_stride: gl.constexpr, token_stride: gl.constexpr):
    offsets = head * head_stride + tokens[:, None] * token_stride + columns[None, :]
    return gl.load(base + offsets, mask=valid, other=0.0).to(gl.float32)


@gluon.jit
def _quantize(values, scale):
    scaled = _divide_rn(values, scale)
    scaled = gl.maximum(gl.minimum(scaled, 448.0), -448.0)
    return scaled.to(gl.float8e4nv)


@gluon.jit
def _pack_qkv(
    q, k, v, q_fp8, k_fp8, v_fp8, scales, sequence,
    head_stride: gl.constexpr, token_stride: gl.constexpr,
    padded_queries: gl.constexpr, padded_keys: gl.constexpr,
    heads: gl.constexpr, block_tokens: gl.constexpr,
    L: gl.constexpr, LV: gl.constexpr, LVC: gl.constexpr, LVO: gl.constexpr,
):
    head = gl.program_id(0)
    block = gl.program_id(1)
    tokens = block * block_tokens + gl.arange(0, block_tokens, layout=gl.SliceLayout(1, L))
    # U's layout-rematerialized transposed-store token range stays at entry.
    tokens_out = block * block_tokens + gl.arange(0, block_tokens, layout=gl.SliceLayout(0, LVO))
    valid = tokens[:, None] < sequence
    q_scale = gl.load(scales + 0 * heads + head)
    k_scale = gl.load(scales + 1 * heads + head)
    v_scale = gl.load(scales + 2 * heads + head)

    # U rematerializes the feature range at its first use, after these two
    # leading offset terms.  Keep that ordering to align register assignment.
    hsd_leading = head * padded_queries * 128 + tokens[:, None] * 128
    columns = gl.arange(0, 128, layout=gl.SliceLayout(0, L))
    hsd_offsets = hsd_leading + columns[None, :]
    k_offsets = head * padded_keys * 128 + tokens[:, None] * 128 + columns[None, :]
    in_queries = tokens[:, None] < padded_queries
    in_keys = tokens[:, None] < padded_keys
    q_tile = _load_tile(q, head, tokens, columns, valid, head_stride, token_stride)
    gl.store(q_fp8 + hsd_offsets, _quantize(q_tile, q_scale), mask=in_queries)
    k_tile = _load_tile(k, head, tokens, columns, valid, head_stride, token_stride)
    gl.store(k_fp8 + k_offsets, _quantize(k_tile, k_scale), mask=in_keys)

    if LV == L:
        tokens_v = tokens
        columns_v = columns
    else:
        tokens_v = block * block_tokens + gl.arange(0, block_tokens, layout=gl.SliceLayout(1, LV))
        columns_v = gl.arange(0, 128, layout=gl.SliceLayout(0, LV))
    source_tokens = (
        tokens_v // 16 * 16 + 8 * ((tokens_v % 4) // 2)
        + 2 * ((tokens_v % 16) // 4) + tokens_v % 2
    )
    valid_source = source_tokens[:, None] < sequence
    columns_out = gl.arange(0, 128, layout=gl.SliceLayout(1, LVO))
    hds_offsets = head * 128 * padded_keys + columns_out[:, None] * padded_keys + tokens_out[None, :]
    in_keys_transposed = tokens_out[None, :] < padded_keys

    offsets = head * head_stride + source_tokens[:, None] * token_stride + columns_v[None, :]
    v_raw = gl.load(v + offsets, mask=valid_source, other=0.0)
    # Crucial G0 boundary: U converts BF16 ownership BEFORE FP32 widening,
    # RN division and FP8 quantization.  Do not move this after quantization.
    v_tile = gl.convert_layout(v_raw, LVC).to(gl.float32)
    v_output = v_fp8 + hds_offsets
    v_transposed = gl.permute(_quantize(v_tile, v_scale), (1, 0))
    gl.store(v_output, gl.convert_layout(v_transposed, LVO), mask=in_keys_transposed)


@gluon.jit
def softplus_nv(x):
    return gl.inline_asm_elementwise(asm=s_softplus, constraints=s_constraints,
        pack=NUM_REG, args=[x], dtype=gl.float32, is_pure=True)


@gluon.jit
def softplus(x):
    return softplus_nv(x)


@gluon.jit
def exp(x):
    return gl.exp(x.to(gl.float32))


@gluon.jit(do_not_specialize=['T'])
def kda_gate_fwd_kernel(
    g, A_log, dt_bias, beta, yg, yb, lower_bound, T,
    H: gl.constexpr, D: gl.constexpr, BT: gl.constexpr, BD: gl.constexpr,
    HAS_A: gl.constexpr, HAS_BIAS: gl.constexpr, HAS_BETA: gl.constexpr,
    USE_LOWER_BOUND: gl.constexpr,
    LI: gl.constexpr, LB: gl.constexpr, LC: gl.constexpr, LO: gl.constexpr,
):
    i_t = gl.program_id(0).to(gl.int64)
    i_h = gl.program_id(1).to(gl.int64)
    b_A = gl.load(A_log + i_h).to(gl.float32) if HAS_A else 1.0
    o_t = i_t * BT + gl.arange(0, BT, layout=gl.SliceLayout(1, LI))
    o_d = gl.arange(0, BD, layout=gl.SliceLayout(0, LI))
    m_t = o_t < T
    m_g = m_t[:, None] & (o_d[None, :] < D)
    p_g = g + i_h * D + o_t[:, None] * (H * D) + o_d[None, :]
    # Match U's rematerialized output addresses before the input load.  Input
    # and FP32 output have different coalesced vector widths in this kernel.
    o_t_out = i_t * BT + gl.arange(0, BT, layout=gl.SliceLayout(1, LO))
    o_d_out = gl.arange(0, BD, layout=gl.SliceLayout(0, LO))
    m_g_out = (o_t_out[:, None] < T) & (o_d_out[None, :] < D)
    p_yg = yg + i_h * D + o_t_out[:, None] * (H * D) + o_d_out[None, :]
    # Actual U moves BF16 ownership before widening, just as N04 does.
    b_g = gl.convert_layout(gl.load(p_g, mask=m_g, other=0.0), LC).to(gl.float32)
    if HAS_BIAS:
        o_b = i_h * D + gl.arange(0, BD, layout=LB)
        b_bias = gl.load(dt_bias + o_b, mask=o_b < H * D, other=0.0).to(gl.float32)
        b_bias = gl.convert_layout(b_bias, gl.SliceLayout(0, LC))
        b_g = b_g + b_bias[None, :]
    if not USE_LOWER_BOUND:
        b_yg = -(exp(b_A) if HAS_A else b_A) * softplus(b_g)
    else:
        b_yg = lower_bound * gl.sigmoid((exp(b_A) if HAS_A else b_A) * b_g)
    gl.store(p_yg, gl.convert_layout(b_yg, LO).to(p_yg.dtype.element_ty), mask=m_g_out)
    # The public wrapper used by this experiment sets HAS_BETA=False.
    # Keep the original branch explicit rather than silently deleting semantics.
    if HAS_BETA:
        p_b = beta + i_h + o_t * H
        p_yb = yb + i_h + o_t * H
        b_yb = gl.sigmoid(gl.load(p_b, mask=m_t, other=0.0).to(gl.float32))
        gl.store(p_yb, b_yb.to(p_yb.dtype.element_ty), mask=m_t)


def _encoding(line, shape, definitions):
    from ..operator_gpu.layout_parsing import tensor_encoding
    return tensor_encoding(line, shape, definitions)


def _pack_plan(plan):
    if 'pack' in plan:
        return copy.deepcopy(plan['pack'])
    definitions = plan['layouts']
    loads = [line for line in plan['loads'] if 'tensor<64x128x' in line]
    stores = plan['stores']
    assert len(loads) == len(stores) == 3
    qi,ki,vi = [_encoding(line,[64,128],definitions) for line in loads]
    qo,ko = [_encoding(line,[64,128],definitions) for line in stores[:2]]
    vo = _encoding(stores[2],[128,64],definitions)
    assert qi == ki == qo == ko
    inverse = copy.deepcopy(vo)
    assert inverse['kind'] == 'blocked'
    inverse['fields'] = {k:[values[1],values[0]] if k != 'order' else [1-x for x in values]
                         for k,values in vo['fields'].items()}
    return dict(L=qi,LV=vi,LVC=inverse,LVO=vo)


def _gate_plan(case,plan):
    definitions = plan['layouts']
    shape = [case['BT'],1<<(case['D']-1).bit_length()]
    LI = plan['input'];LO = plan['output'];LB = plan['bias']
    asm = [line for line in plan.get('inline_asm',[]) if f'tensor<{shape[0]}x{shape[1]}x' in line]
    assert len(asm) == 1, 'Expected the exact pinned softplus tensor operation in U IR'
    LC = _encoding(asm[0],shape,definitions)
    return dict(LI=LI,LB=LB,LC=LC,LO=LO)


def _layout(shape,warps,axis,vec):
    other = 1-axis
    assert shape[axis] % vec == 0
    tp = [1,1];tp[axis]=min(32,shape[axis]//vec);tp[other]=32//tp[axis]
    wp = [1,1];wp[axis]=min(warps,max(1,shape[axis]//(vec*tp[axis])));wp[other]=warps//wp[axis]
    size=[1,1];size[axis]=vec
    return dict(kind='blocked',fields=dict(size_per_thread=size,threads_per_warp=tp,
        warps_per_cta=wp,order=[axis,other]))


def variants(case,plan):
    result=[];seen=set()
    if case['family']=='N04':
        original=_pack_plan(plan)
        choices=[(f'v_input_vec{v}',dict(LV=_layout([64,128],case['W'],1,v))) for v in [1,2,4,8]]
        choices += [('v_compute_input',dict(LVC=original['LV']))]
        choices += [(f'v_compute_token_vec{v}',dict(LVC=_layout([64,128],case['W'],0,v))) for v in [4,8,32]]
    else:
        original=_gate_plan(case,plan)
        shape=[case['BT'],case['D']]
        choices=[(f'feature_vec{v}',dict(LI=_layout(shape,case['W'],1,v),LC=_layout(shape,case['W'],1,v))) for v in [1,2,4,8]]
        choices += [(f'token_vec{v}',dict(LI=_layout(shape,case['W'],0,v),LC=_layout(shape,case['W'],0,v))) for v in [1,2,4,8] if shape[0]>=v]
    original_key=json_key(original)
    for name,change in choices:
        value={**copy.deepcopy(original),**change};key=json_key(value)
        if key==original_key or key in seen:continue
        seen.add(key);result.append(dict(name=name,layouts=value,intervention=change))
    assert len(result)<=8
    return result


def json_key(value):
    import json
    return json.dumps(value,sort_keys=True,separators=(',',':'))


def layout_constants(case,plan,variant='G0'):
    from ..operator_gpu.layout_parsing import instantiate
    value=_pack_plan(plan) if case['family']=='N04' else _gate_plan(case,plan)
    if isinstance(variant,dict):
        value=copy.deepcopy(variant['layouts'])
    elif variant!='G0':
        value=next(row['layouts'] for row in variants(case,plan) if row['name']==variant)
    return {name:instantiate(spec,gl) for name,spec in value.items()}
