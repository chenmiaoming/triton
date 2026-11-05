# SPDX-License-Identifier: Apache-2.0
"""Source-level rendering of U's explicit intermediate layouts and loop invariants.

Uses the ordinary Gluon pipeline.  The default layout is accepted as a baseline
only after comparison with the frozen upstream executable; alternate layouts
remain independent candidates and require an effective-layout audit.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def tanh(x):
    return 2 * gl.sigmoid(2 * x) - 1


@gluon.jit
def _fwd_kernel_stage1(
    Q, K_Buffer, V_Buffer, sm_scale_withk, kv_indptr, kv_indices, Att_Out,
    Att_Lse, num_kv_splits, stride_qbs, stride_qh, stride_buf_kbs,
    stride_buf_kh, stride_buf_vbs, stride_buf_vh, stride_buf_kpage,
    stride_buf_ktok, stride_buf_vpage, stride_buf_vtok, stride_mid_ob,
    stride_mid_oh, stride_mid_os, kv_group_num: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr, BLOCK_DV: gl.constexpr, BLOCK_N: gl.constexpr,
    MIN_BLOCK_KV: gl.constexpr, logit_cap: gl.constexpr, Lk: gl.constexpr,
    Lv: gl.constexpr, xai_temperature_len: gl.constexpr, PAGE_SIZE: gl.constexpr,
    SCORE_MOD: gl.constexpr = None, Aux0=None, aux0_stride_t=0,
    aux0_stride_h=0, aux0_len=0, uneven_batch=None,
    ROTATE_BATCH: gl.constexpr = False, L: gl.constexpr = None,
    LK: gl.constexpr = None, LQ: gl.constexpr = None,
    LOUT: gl.constexpr = None,
):
    cur_batch = gl.program_id(0)
    cur_head = gl.program_id(1)
    split_kv_id = gl.program_id(2)
    if ROTATE_BATCH:
        if gl.load(uneven_batch) != 0:
            cur_batch = (cur_batch + cur_head + split_kv_id) % gl.num_programs(0)
    cur_batch = cur_batch.to(gl.int64)
    cur_kv_head = cur_head // kv_group_num

    # RMLC in U rematerializes indices/masks in each consumer's layout.
    # With the same logical range extent, U's rematerialized row range keeps
    # the original entry-block placement rather than being created in the if.
    if BLOCK_N == BLOCK_DMODEL:
        rows_k = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, LK))
    offs_d_k = gl.arange(0, BLOCK_DMODEL, layout=gl.SliceLayout(0, LK))
    offs_d_q = gl.arange(0, BLOCK_DMODEL, layout=LQ)
    offs_d_v = gl.arange(0, BLOCK_DV, layout=gl.SliceLayout(0, L))
    offs_d_out = gl.arange(0, BLOCK_DV, layout=LOUT)
    mask_d_k = offs_d_k < Lk
    mask_d_q = offs_d_q < Lk
    mask_d_v = offs_d_v < Lv
    mask_d_out = offs_d_out < Lv
    cur_batch_kv_start_idx = gl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = gl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx
    kv_splits = gl.load(num_kv_splits + cur_batch)
    if xai_temperature_len > 0:
        offs_qidx = cur_batch_seq_len - 1
        xai_temperature_scale = 1.0 / gl.log2(float(xai_temperature_len))
        qtemp = gl.log2(offs_qidx.to(gl.float32)) * xai_temperature_scale
        xai_temperature_reg = gl.where(offs_qidx > xai_temperature_len, qtemp, 1.0)

    off_q_base = cur_batch * stride_qbs + cur_head * stride_qh
    offs_d_q_wide = offs_d_q.to(gl.int64)
    offs_d_out_wide = offs_d_out.to(gl.int64)
    off_q = off_q_base + offs_d_q_wide
    kv_len_per_split = gl.cdiv(gl.cdiv(cur_batch_seq_len, kv_splits), MIN_BLOCK_KV) * MIN_BLOCK_KV
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = gl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)
    e_max = -float('inf')
    e_sum = 0.0
    acc = gl.zeros([BLOCK_DV], dtype=gl.float32, layout=gl.SliceLayout(0, L))

    if split_kv_end > split_kv_start:
        q = gl.load(Q + off_q, mask=mask_d_q, other=0.0)

        # Spell LICM's loop-invariant tensor values before the loop.  In
        # particular the query conversion is paid once per nonempty split.
        if BLOCK_N != BLOCK_DMODEL:
            rows_k = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, LK))
        end_k = gl.full([BLOCK_N], split_kv_end, gl.int32, gl.SliceLayout(1, LK))
        kv_ptrs = gl.full([BLOCK_N], kv_indices + cur_batch_kv_start_idx,
                          kv_indices.dtype, gl.SliceLayout(1, LK))
        if PAGE_SIZE == 1:
            k_stride = gl.full([BLOCK_N, 1], stride_buf_kbs, gl.int32, LK)
            v_stride = gl.full([BLOCK_N, 1], stride_buf_vbs, gl.int32, L)
        else:
            k_page_stride = gl.full([BLOCK_N, 1], stride_buf_kpage, gl.int32, LK)
            k_tok_stride = gl.full([BLOCK_N, 1], stride_buf_ktok, gl.int32, LK)
            v_page_stride = gl.full([BLOCK_N, 1], stride_buf_vpage, gl.int32, L)
            v_tok_stride = gl.full([BLOCK_N, 1], stride_buf_vtok, gl.int32, L)
        k_head = gl.full([BLOCK_N, 1], cur_kv_head * stride_buf_kh, gl.int32, LK)
        k_columns = offs_d_k[None, :].broadcast_to(BLOCK_N, BLOCK_DMODEL)
        k_end = gl.full([BLOCK_N, 1], split_kv_end, gl.int32, LK)
        k_mask_columns = mask_d_k[None, :].broadcast_to(BLOCK_N, BLOCK_DMODEL)
        k_base = gl.full([BLOCK_N, BLOCK_DMODEL], K_Buffer, K_Buffer.dtype, LK)
        if LQ != gl.SliceLayout(0, LK):
            q_k = gl.convert_layout(q, gl.SliceLayout(0, LK))
        else:
            q_k = q
        q_matrix = q_k[None, :].broadcast_to(BLOCK_N, BLOCK_DMODEL)
        q_scale = gl.full([BLOCK_N], sm_scale_withk, gl.float32, gl.SliceLayout(1, LK))
        v_head = gl.full([BLOCK_N, 1], cur_kv_head * stride_buf_vh, gl.int32, L)
        v_columns = offs_d_v[None, :].broadcast_to(BLOCK_N, BLOCK_DV)
        v_end = gl.full([BLOCK_N, 1], split_kv_end, gl.int32, L)
        v_mask_columns = mask_d_v[None, :].broadcast_to(BLOCK_N, BLOCK_DV)
        v_base = gl.full([BLOCK_N, BLOCK_DV], V_Buffer, V_Buffer.dtype, L)

        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + rows_k
            row_mask = offs_n < end_k
            kv_loc = gl.load(kv_ptrs + offs_n, mask=row_mask, other=0)
            if PAGE_SIZE == 1:
                k_rows = kv_loc[:, None] * k_stride + k_head
            else:
                page_id = kv_loc // PAGE_SIZE
                tok_in_p = kv_loc % PAGE_SIZE
                k_rows = page_id[:, None] * k_page_stride + tok_in_p[:, None] * k_tok_stride + k_head
            offs_buf_k = k_rows.broadcast_to(BLOCK_N, BLOCK_DMODEL) + k_columns
            k_mask_rows = (offs_n[:, None] < k_end).broadcast_to(BLOCK_N, BLOCK_DMODEL)
            k = gl.load(k_base + offs_buf_k, mask=k_mask_rows & k_mask_columns, other=0.0)
            qk = gl.sum((q_matrix * k).to(gl.float32), 1)
            qk *= q_scale
            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)
            if xai_temperature_len > 0:
                qk *= xai_temperature_reg
            if SCORE_MOD is not None:
                qk = SCORE_MOD(qk, cur_batch_seq_len - 1, offs_n, cur_batch,
                              cur_head, row_mask, Aux0, aux0_stride_t,
                              aux0_stride_h, aux0_len)
            qk = gl.where(row_mask, qk, float('-inf'))

            if gl.SliceLayout(1, LK) != gl.SliceLayout(1, L):
                kv_loc_v = gl.convert_layout(kv_loc, gl.SliceLayout(1, L))
                offs_n_v = gl.convert_layout(offs_n, gl.SliceLayout(1, L))
            else:
                kv_loc_v = kv_loc
                offs_n_v = offs_n
            if PAGE_SIZE == 1:
                v_rows = kv_loc_v[:, None] * v_stride + v_head
            else:
                page_v = kv_loc_v // PAGE_SIZE
                tok_v = kv_loc_v % PAGE_SIZE
                v_rows = page_v[:, None] * v_page_stride + tok_v[:, None] * v_tok_stride + v_head
            offs_buf_v = v_rows.broadcast_to(BLOCK_N, BLOCK_DV) + v_columns
            v_mask_rows = (offs_n_v[:, None] < v_end).broadcast_to(BLOCK_N, BLOCK_DV)
            v = gl.load(v_base + offs_buf_v, mask=v_mask_rows & v_mask_columns, other=0.0)

            n_e_max = gl.maximum(gl.max(qk, 0), e_max)
            re_scale = gl.exp(e_max - n_e_max)
            p = gl.exp(qk - n_e_max)
            acc *= re_scale
            if gl.SliceLayout(1, LK) != gl.SliceLayout(1, L):
                p_v = gl.convert_layout(p, gl.SliceLayout(1, L))
            else:
                p_v = p
            acc += gl.sum(p_v[:, None] * v, 0)
            e_sum = e_sum * re_scale + gl.sum(p, 0)
            e_max = n_e_max

        # Recompute addresses/mask in LOUT; only actual output values move.
        offs_mid_o = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh
                      + split_kv_id * stride_mid_os + offs_d_out_wide)
        output_ptrs = Att_Out + offs_mid_o
        output = acc / e_sum
        if gl.SliceLayout(0, L) != LOUT:
            output = gl.convert_layout(output, LOUT)
        gl.store(output_ptrs, output, mask=mask_d_out)
        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh
                        + split_kv_id * stride_mid_os) // Lv
        gl.store(Att_Lse + offs_mid_o_1, e_max + gl.log(e_sum))
