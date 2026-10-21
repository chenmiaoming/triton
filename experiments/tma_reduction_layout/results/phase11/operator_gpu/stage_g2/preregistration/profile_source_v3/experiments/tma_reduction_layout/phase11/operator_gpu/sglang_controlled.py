# SPDX-License-Identifier: Apache-2.0
# Controlled layout derivative of the frozen upstream body.
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from types import SimpleNamespace
float8_info = SimpleNamespace(min=-448., max=448.)

@gluon.jit
def tanh(x):
    return 2 * gl.sigmoid(2 * x) - 1

@gluon.jit
def _fwd_kernel_stage1(Q, K_Buffer, V_Buffer, sm_scale_withk, kv_indptr, kv_indices, Att_Out, Att_Lse, num_kv_splits, stride_qbs, stride_qh, stride_buf_kbs, stride_buf_kh, stride_buf_vbs, stride_buf_vh, stride_buf_kpage, stride_buf_ktok, stride_buf_vpage, stride_buf_vtok, stride_mid_ob, stride_mid_oh, stride_mid_os, kv_group_num: gl.constexpr, BLOCK_DMODEL: gl.constexpr, BLOCK_DV: gl.constexpr, BLOCK_N: gl.constexpr, MIN_BLOCK_KV: gl.constexpr, logit_cap: gl.constexpr, Lk: gl.constexpr, Lv: gl.constexpr, xai_temperature_len: gl.constexpr, PAGE_SIZE: gl.constexpr, SCORE_MOD: gl.constexpr=None, Aux0=None, aux0_stride_t=0, aux0_stride_h=0, aux0_len=0, uneven_batch=None, ROTATE_BATCH: gl.constexpr=False, L: gl.constexpr=None, LK: gl.constexpr=None, LQ: gl.constexpr=None, LOUT: gl.constexpr=None):
    cur_batch = gl.program_id(0)
    cur_head = gl.program_id(1)
    split_kv_id = gl.program_id(2)
    if ROTATE_BATCH:
        if gl.load(uneven_batch) != 0:
            cur_batch = (cur_batch + cur_head + split_kv_id) % gl.num_programs(0)
    cur_batch = cur_batch.to(gl.int64)
    cur_kv_head = cur_head // kv_group_num
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=LQ)
    offs_dv = gl.arange(0, BLOCK_DV, layout=gl.SliceLayout(0, L))
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_kv_start_idx = gl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = gl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = gl.load(num_kv_splits + cur_batch)
    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / gl.log2(float(xai_temperature_len))
        _qtemp = gl.log2(offs_qidx.to(gl.float32)) * xai_temperature_scale
        xai_temperature_reg = gl.where(offs_qidx > xai_temperature_len, _qtemp, 1.0)
    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    kv_len_per_split = gl.cdiv(gl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = gl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    e_max = -float('inf')
    e_sum = 0.0
    acc = gl.zeros([BLOCK_DV], dtype=gl.float32, layout=gl.SliceLayout(0, L))
    if split_kv_end > split_kv_start:
        q = gl.load(Q + off_q, mask=mask_d, other=0.0)
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, LK))
            kv_loc = gl.load(kv_indices + cur_batch_kv_start_idx + offs_n, mask=offs_n < split_kv_end, other=0)
            if PAGE_SIZE == 1:
                offs_buf_k = kv_loc[:, None] * stride_buf_kbs + cur_kv_head * stride_buf_kh + gl.convert_layout(offs_d, gl.SliceLayout(0, LK))[None, :]
            else:
                page_id = kv_loc // PAGE_SIZE
                tok_in_p = kv_loc % PAGE_SIZE
                offs_buf_k = page_id[:, None] * stride_buf_kpage + tok_in_p[:, None] * stride_buf_ktok + cur_kv_head * stride_buf_kh + gl.convert_layout(offs_d, gl.SliceLayout(0, LK))[None, :]
            k = gl.load(K_Buffer + offs_buf_k, mask=(offs_n[:, None] < split_kv_end) & gl.convert_layout(mask_d, gl.SliceLayout(0, LK))[None, :], other=0.0)
            qk = gl.sum((gl.convert_layout(q, gl.SliceLayout(0, LK))[None, :] * k).to(gl.float32), 1)
            qk *= sm_scale_withk
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)
            if xai_temperature_len > 0:
                qk *= xai_temperature_reg
            if SCORE_MOD is not None:
                qk = SCORE_MOD(qk, cur_batch_seq_len - 1, offs_n, cur_batch, cur_head, offs_n < split_kv_end, Aux0, aux0_stride_t, aux0_stride_h, aux0_len)
            qk = gl.where(offs_n < split_kv_end, qk, float('-inf'))
            if PAGE_SIZE == 1:
                offs_buf_v = gl.convert_layout(kv_loc, gl.SliceLayout(1, L))[:, None] * stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_dv[None, :]
            else:
                offs_buf_v = gl.convert_layout(page_id, gl.SliceLayout(1, L))[:, None] * stride_buf_vpage + gl.convert_layout(tok_in_p, gl.SliceLayout(1, L))[:, None] * stride_buf_vtok + cur_kv_head * stride_buf_vh + offs_dv[None, :]
            v = gl.load(V_Buffer + offs_buf_v, mask=(gl.convert_layout(offs_n, gl.SliceLayout(1, L))[:, None] < split_kv_end) & mask_dv[None, :], other=0.0)
            n_e_max = gl.maximum(gl.max(qk, 0), e_max)
            re_scale = gl.exp(e_max - n_e_max)
            p = gl.exp(qk - n_e_max)
            acc *= re_scale
            acc += gl.sum(gl.convert_layout(p, gl.SliceLayout(1, L))[:, None] * v, 0)
            e_sum = e_sum * re_scale + gl.sum(p, 0)
            e_max = n_e_max
        offs_mid_o = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os + offs_dv
        gl.store(gl.convert_layout(Att_Out + offs_mid_o, LOUT), gl.convert_layout(acc / e_sum, LOUT), mask=gl.convert_layout(mask_dv, LOUT))
        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh + split_kv_id * stride_mid_os) // Lv
        gl.store(Att_Lse + offs_mid_o_1, e_max + gl.log(e_sum))
