# SPDX-License-Identifier: Apache-2.0
# Controlled layout derivative of the frozen upstream body.
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from types import SimpleNamespace
float8_info = SimpleNamespace(min=-448., max=448.)

@gluon.jit
def cdiv_fn(x, y):
    """Ceiling division.  Kept as a helper to keep kernel bodies terse."""
    return (x + y - 1) // y

@gluon.jit
def find_seq_idx(query_start_len_ptr, target_idx, num_seqs, BLOCK_Q: gl.constexpr, use_q_block_mode: gl.constexpr):
    """Binary search over the cumulative query-length prefix.

    When ``use_q_block_mode`` is True, the prefix values are reshaped
    into units of ``BLOCK_Q`` plus one entry per boundary — matching
    the q-block grid laid out by the attention kernels.  When False
    we search the plain cumulative-length prefix (used by
    ``reduce_segments`` which iterates over raw query tokens).
    """
    left: gl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = gl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1

@gluon.jit
def reduce_segments(output_ptr, segm_output_ptr, segm_max_ptr, segm_expsum_ptr, seq_lens_ptr, num_seqs, num_query_heads: gl.constexpr, out_scale_inv, output_stride_0: gl.int64, output_stride_1: gl.int64, block_table_stride: gl.int64, TILE_SIZE: gl.constexpr, HEAD_SIZE: gl.constexpr, HEAD_SIZE_PADDED: gl.constexpr, query_start_len_ptr, BLOCK_Q: gl.constexpr, NUM_SEGMENTS_PER_SEQ: gl.constexpr, USE_FP8: gl.constexpr, FP8_MIN: gl.constexpr=float8_info.min, FP8_MAX: gl.constexpr=float8_info.max, L: gl.constexpr=None, LSTAT: gl.constexpr=None, LOUT: gl.constexpr=None):
    query_token_idx = gl.program_id(0)
    query_head_idx = gl.program_id(1)
    if query_token_idx >= gl.load(query_start_len_ptr + num_seqs):
        return
    seq_idx = find_seq_idx(query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False)
    seq_len = gl.load(seq_lens_ptr + seq_idx)
    if seq_len <= 0:
        return
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = gl.arange(0, NUM_SEGMENTS_PER_SEQ, layout=LSTAT) < gl.full([NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=gl.int32, layout=LSTAT)
    dim_mask = gl.where(gl.arange(0, HEAD_SIZE_PADDED, layout=gl.SliceLayout(0, L)) < HEAD_SIZE, 1, 0).to(gl.int1)
    segm_offset = query_token_idx.to(gl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ) + query_head_idx * NUM_SEGMENTS_PER_SEQ + gl.arange(0, NUM_SEGMENTS_PER_SEQ, layout=LSTAT)
    segm_max = gl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float('-inf'))
    segm_expsum = gl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    valid_segment = segm_mask & (segm_expsum > 0) & (segm_max != float('-inf'))
    segm_max = gl.where(valid_segment, segm_max, float('-inf'))
    overall_max = gl.max(segm_max)
    segment_scale = gl.where(valid_segment, gl.exp(segm_max - overall_max), 0.0)
    segm_expsum = segm_expsum * segment_scale
    overall_expsum = gl.sum(segm_expsum)
    segm_output_offset = query_token_idx.to(gl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED) + gl.arange(0, NUM_SEGMENTS_PER_SEQ, layout=gl.SliceLayout(1, L))[:, None] * HEAD_SIZE_PADDED + gl.arange(0, HEAD_SIZE_PADDED, layout=gl.SliceLayout(0, L))[None, :]
    segm_output = gl.load(segm_output_ptr + segm_output_offset, mask=gl.convert_layout(segm_mask, gl.SliceLayout(1, L))[:, None] & dim_mask[None, :], other=0.0)
    segm_output *= gl.convert_layout(segment_scale, gl.SliceLayout(1, L))[:, None]
    acc_sum = gl.sum(segm_output, axis=0)
    acc = gl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)
    if USE_FP8:
        acc = acc * gl.load(out_scale_inv)
        acc = gl.clamp(acc, FP8_MIN, FP8_MAX)
    output_offset = query_token_idx * output_stride_0 + query_head_idx * output_stride_1 + gl.arange(0, HEAD_SIZE_PADDED, layout=gl.SliceLayout(0, L))
    gl.store(gl.convert_layout(output_ptr + output_offset, LOUT), gl.convert_layout(acc, LOUT), mask=gl.convert_layout(dim_mask, LOUT))
