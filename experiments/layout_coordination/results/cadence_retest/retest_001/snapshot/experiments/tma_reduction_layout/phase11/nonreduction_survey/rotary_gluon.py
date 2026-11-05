"""Explicit-layout renderings of the frozen rotary source specializations.

These are provisional until the ordinary compiler produces exactly U's
allocated ELF sections, ABI and resources. No pass is disabled or replaced.
"""
import copy
import re
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def rotary_kernel(
    OUT, X, COS, SIN, CU_SEQLENS, SEQLEN_OFFSETS,
    seqlen, nheads, seqlen_ro,
    stride_out_batch, stride_out_seqlen, stride_out_nheads, stride_out_headdim,
    stride_x_batch, stride_x_seqlen, stride_x_nheads, stride_x_headdim,
    ROTARY_DIM: gl.constexpr, IS_SEQLEN_OFFSETS_TENSOR: gl.constexpr,
    IS_VARLEN: gl.constexpr, INTERLEAVED: gl.constexpr, CONJUGATE: gl.constexpr,
    BLOCK_H: gl.constexpr, BLOCK_M: gl.constexpr,
    LX: gl.constexpr, LC: gl.constexpr, LWORK: gl.constexpr, LPAIR: gl.constexpr,
    LOUT: gl.constexpr,
    TRIG_CONVERT_BEFORE_WIDEN: gl.constexpr = False,
):
    gl.static_assert(not IS_VARLEN and INTERLEAVED and not CONJUGATE)
    gl.static_assert(not IS_SEQLEN_OFFSETS_TENSOR)
    BLOCK_K: gl.constexpr = triton.next_power_of_2(ROTARY_DIM)
    ROTARY_DIM_HALF = ROTARY_DIM // 2
    pid_head = gl.program_id(0)
    pid_m = gl.program_id(1)
    pid_batch = gl.program_id(2)
    X = X + pid_batch * stride_x_batch
    OUT = OUT + pid_batch * stride_out_batch
    if pid_m * BLOCK_M >= seqlen:
        return

    # Preserve the source's separate trig and input pointer expressions;
    # rematerialize their ranges in the actual layouts read from U.
    rh = pid_head * BLOCK_H + gl.arange(0, BLOCK_H,
        layout=gl.SliceLayout(1, gl.SliceLayout(2, LX)))
    rm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M,
        layout=gl.SliceLayout(0, gl.SliceLayout(2, LX)))
    rm_cs = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LC))
    rm_cs = rm_cs + SEQLEN_OFFSETS
    rk_half = gl.arange(0, BLOCK_K//2, layout=gl.SliceLayout(0, LC))
    COS = COS + (rm_cs[:,None]*ROTARY_DIM_HALF + rk_half[None,:])
    SIN = SIN + (rm_cs[:,None]*ROTARY_DIM_HALF + rk_half[None,:])
    mask_cs = (rm_cs[:,None] < seqlen_ro) & (rk_half[None,:] < ROTARY_DIM_HALF)
    if TRIG_CONVERT_BEFORE_WIDEN:
        cos = gl.convert_layout(gl.load(COS,mask=mask_cs,other=1.0),
                                gl.SliceLayout(0,LPAIR)).to(gl.float32)
        sin = gl.convert_layout(gl.load(SIN,mask=mask_cs,other=0.0),
                                gl.SliceLayout(0,LPAIR)).to(gl.float32)
    else:
        cos = gl.load(COS, mask=mask_cs, other=1.0).to(gl.float32)
        sin = gl.load(SIN, mask=mask_cs, other=0.0).to(gl.float32)

    rk = gl.arange(0, BLOCK_K,
        layout=gl.SliceLayout(0, gl.SliceLayout(1, LX)))
    X = X + (rh[:,None,None]*stride_x_nheads
             + rm[None,:,None]*stride_x_seqlen
             + rk[None,None,:]*stride_x_headdim)
    rh_out = pid_head*BLOCK_H+gl.arange(0,BLOCK_H,
        layout=gl.SliceLayout(1,gl.SliceLayout(2,LOUT)))
    rm_out = pid_m*BLOCK_M+gl.arange(0,BLOCK_M,
        layout=gl.SliceLayout(0,gl.SliceLayout(2,LOUT)))
    rk_out = gl.arange(0,BLOCK_K,
        layout=gl.SliceLayout(0,gl.SliceLayout(1,LOUT)))
    out_ptr = OUT+(rh_out[:,None,None]*stride_out_nheads
                  +rm_out[None,:,None]*stride_out_seqlen
                  +rk_out[None,None,:]*stride_out_headdim)
    mask = ((rh[:,None,None] < nheads) & (rm[None,:,None] < seqlen)
            & (rk[None,None,:] < ROTARY_DIM))
    out_mask = ((rh_out[:,None,None]<nheads)&(rm_out[None,:,None]<seqlen)
                &(rk_out[None,None,:]<ROTARY_DIM))
    x = gl.load(X,mask=mask,other=0.0).to(gl.float32)
    x = gl.convert_layout(x,LWORK)
    x0,x1 = gl.split(gl.reshape(x,[BLOCK_H,BLOCK_M,BLOCK_K//2,2]))
    cos = gl.convert_layout(cos,gl.SliceLayout(0,x0.type.layout))
    sin = gl.convert_layout(sin,gl.SliceLayout(0,x0.type.layout))
    o0 = x0*cos[None,:,:] - x1*sin[None,:,:]
    o1 = x0*sin[None,:,:] + x1*cos[None,:,:]
    o = gl.reshape(gl.join(o0,o1),[BLOCK_H,BLOCK_M,BLOCK_K])
    o = gl.convert_layout(o.to(OUT.dtype.element_ty),LOUT)
    gl.store(out_ptr,o,mask=out_mask)


@gluon.jit
def _triton_mrope_forward_fused(
    q_ptr,k_ptr,cos_sin_cache_ptr,positions_ptr,q_stride,k_stride,positions_stride,
    n_qh:gl.constexpr,n_kh:gl.constexpr,hd:gl.constexpr,rd:gl.constexpr,
    pad_n_qh:gl.constexpr,pad_n_kh:gl.constexpr,pad_hd:gl.constexpr,
    mrope_section_t:gl.constexpr,mrope_section_h:gl.constexpr,mrope_section_w:gl.constexpr,
    is_interleaved:gl.constexpr,is_interleaved_glm:gl.constexpr,
    is_neox_style:gl.constexpr,axis_map_ptr,
    LQ:gl.constexpr,LK:gl.constexpr,LTQ:gl.constexpr,LTK:gl.constexpr,
    LQOUT:gl.constexpr,LKOUT:gl.constexpr,
):
    gl.static_assert(is_neox_style and not is_interleaved and not is_interleaved_glm)
    pid=gl.program_id(0)
    q_ptr=q_ptr+pid*q_stride
    k_ptr=k_ptr+pid*k_stride
    half_rd=rd//2
    t=gl.load(positions_ptr+0*positions_stride+pid)
    h=gl.load(positions_ptr+1*positions_stride+pid)
    w=gl.load(positions_ptr+2*positions_stride+pid)
    t_cos=cos_sin_cache_ptr+t*rd
    h_cos=cos_sin_cache_ptr+h*rd
    w_cos=cos_sin_cache_ptr+w*rd
    t_sin=t_cos+half_rd
    h_sin=h_cos+half_rd
    w_sin=w_cos+half_rd
    cos_offsets=gl.arange(0,pad_hd//2,layout=LTQ)
    cos_offsets_k=gl.arange(0,pad_hd//2,layout=LTK)
    t_end=mrope_section_t
    h_end=t_end+mrope_section_h
    t_mask=cos_offsets<mrope_section_t
    h_mask=(t_end<=cos_offsets)&(cos_offsets<h_end)
    w_mask=(h_end<=cos_offsets)&(cos_offsets<half_rd)
    t_mask_k=cos_offsets_k<mrope_section_t
    h_mask_k=(t_end<=cos_offsets_k)&(cos_offsets_k<h_end)
    w_mask_k=(h_end<=cos_offsets_k)&(cos_offsets_k<half_rd)
    t_cos_row=gl.load(t_cos+cos_offsets,mask=t_mask,other=0)
    t_cos_row_k=gl.load(t_cos+cos_offsets_k,mask=t_mask_k,other=0)
    t_sin_row=gl.load(t_sin+cos_offsets,mask=t_mask,other=0)
    t_sin_row_k=gl.load(t_sin+cos_offsets_k,mask=t_mask_k,other=0)
    h_cos_row=gl.load(h_cos+cos_offsets,mask=h_mask,other=0)
    h_cos_row_k=gl.load(h_cos+cos_offsets_k,mask=h_mask_k,other=0)
    h_sin_row=gl.load(h_sin+cos_offsets,mask=h_mask,other=0)
    h_sin_row_k=gl.load(h_sin+cos_offsets_k,mask=h_mask_k,other=0)
    w_cos_row=gl.load(w_cos+cos_offsets,mask=w_mask,other=0)
    w_cos_row_k=gl.load(w_cos+cos_offsets_k,mask=w_mask_k,other=0)
    w_sin_row=gl.load(w_sin+cos_offsets,mask=w_mask,other=0)
    w_sin_row_k=gl.load(w_sin+cos_offsets_k,mask=w_mask_k,other=0)
    cos_row=t_cos_row+h_cos_row
    cos_row_k=t_cos_row_k+h_cos_row_k
    cos_row=cos_row+w_cos_row
    cos_row_k=cos_row_k+w_cos_row_k
    sin_row=t_sin_row+h_sin_row
    sin_row_k=t_sin_row_k+h_sin_row_k
    sin_row=sin_row+w_sin_row
    sin_row_k=sin_row_k+w_sin_row_k

    qh=gl.arange(0,pad_n_qh,layout=gl.SliceLayout(1,LQ))
    qd=gl.arange(0,pad_hd//2,layout=gl.SliceLayout(0,LQ))
    kh=gl.arange(0,pad_n_kh,layout=gl.SliceLayout(1,LK))
    kd=gl.arange(0,pad_hd//2,layout=gl.SliceLayout(0,LK))
    fhq=qh[:,None]*hd+qd[None,:]
    fhk=kh[:,None]*hd+kd[None,:]
    fqm=(qh[:,None]<n_qh)&(qd[None,:]<rd//2)
    fkm=(kh[:,None]<n_kh)&(kd[None,:]<rd//2)
    q1=gl.load(q_ptr+fhq,mask=fqm,other=0).to(sin_row.dtype)
    k1=gl.load(k_ptr+fhk,mask=fkm,other=0).to(sin_row.dtype)
    shq=fhq+rd//2
    shk=fhk+rd//2
    q2=gl.load(q_ptr+shq,mask=fqm,other=0).to(sin_row.dtype)
    k2=gl.load(k_ptr+shk,mask=fkm,other=0).to(sin_row.dtype)
    qc=gl.convert_layout(cos_row,gl.SliceLayout(0,LQ))
    qs=gl.convert_layout(sin_row,gl.SliceLayout(0,LQ))
    kc=gl.convert_layout(cos_row_k,gl.SliceLayout(0,LK))
    ks=gl.convert_layout(sin_row_k,gl.SliceLayout(0,LK))
    qh_out=gl.arange(0,pad_n_qh,layout=gl.SliceLayout(1,LQOUT))
    qd_out=gl.arange(0,pad_hd//2,layout=gl.SliceLayout(0,LQOUT))
    kh_out=gl.arange(0,pad_n_kh,layout=gl.SliceLayout(1,LKOUT))
    kd_out=gl.arange(0,pad_hd//2,layout=gl.SliceLayout(0,LKOUT))
    qfirst=qh_out[:,None]*hd+qd_out[None,:]
    kfirst=kh_out[:,None]*hd+kd_out[None,:]
    qsecond=qfirst+rd//2
    ksecond=kfirst+rd//2
    qm=(qh_out[:,None]<n_qh)&(qd_out[None,:]<rd//2)
    km=(kh_out[:,None]<n_kh)&(kd_out[None,:]<rd//2)
    gl.store(q_ptr+qfirst,gl.convert_layout((q1*qc[None,:]-q2*qs[None,:]).to(q_ptr.dtype.element_ty),LQOUT),mask=qm)
    gl.store(q_ptr+qsecond,gl.convert_layout((q2*qc[None,:]+q1*qs[None,:]).to(q_ptr.dtype.element_ty),LQOUT),mask=qm)
    gl.store(k_ptr+kfirst,gl.convert_layout((k1*kc[None,:]-k2*ks[None,:]).to(k_ptr.dtype.element_ty),LKOUT),mask=km)
    gl.store(k_ptr+ksecond,gl.convert_layout((k2*kc[None,:]+k1*ks[None,:]).to(k_ptr.dtype.element_ty),LKOUT),mask=km)

