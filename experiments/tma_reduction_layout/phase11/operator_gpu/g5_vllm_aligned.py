"""Literal Gluon rendering of U's two statistics paths and narrow output conversion."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from .vllm_controlled import find_seq_idx, cdiv_fn


@gluon.jit
def reduce_segments(output_ptr, segm_output_ptr, segm_max_ptr, segm_expsum_ptr, seq_lens_ptr,
                    num_seqs, num_query_heads: gl.constexpr, out_scale_inv,
                    output_stride_0: gl.int64, output_stride_1: gl.int64, block_table_stride: gl.int64,
                    TILE_SIZE: gl.constexpr, HEAD_SIZE: gl.constexpr, HEAD_SIZE_PADDED: gl.constexpr,
                    query_start_len_ptr, BLOCK_Q: gl.constexpr, NUM_SEGMENTS_PER_SEQ: gl.constexpr,
                    USE_FP8: gl.constexpr, FP8_MIN: gl.constexpr, FP8_MAX: gl.constexpr,
                    L: gl.constexpr, LSTAT: gl.constexpr, LOUT: gl.constexpr):
    query_token_idx=gl.program_id(0)
    query_head_idx=gl.program_id(1)
    if query_token_idx>=gl.load(query_start_len_ptr+num_seqs):
        return
    seq_idx=find_seq_idx(query_start_len_ptr,query_token_idx,num_seqs,BLOCK_Q,False)
    seq_len=gl.load(seq_lens_ptr+seq_idx)
    if seq_len<=0:
        return
    tiles_per_segment=cdiv_fn(seq_len,NUM_SEGMENTS_PER_SEQ*TILE_SIZE)
    active=cdiv_fn(seq_len,tiles_per_segment*TILE_SIZE)
    rows=gl.arange(0,NUM_SEGMENTS_PER_SEQ,layout=LSTAT)
    rows_local=gl.arange(0,NUM_SEGMENTS_PER_SEQ,layout=gl.SliceLayout(1,L))
    mask=rows<active
    mask_local=rows_local<active
    base=query_token_idx.to(gl.int64)*(num_query_heads*NUM_SEGMENTS_PER_SEQ)+query_head_idx*NUM_SEGMENTS_PER_SEQ
    m=gl.load(segm_max_ptr+base+rows,mask=mask,other=float('-inf'))
    m_local=gl.load(segm_max_ptr+base+rows_local,mask=mask_local,other=float('-inf'))
    z=gl.load(segm_expsum_ptr+base+rows,mask=mask,other=0.)
    z_local=gl.load(segm_expsum_ptr+base+rows_local,mask=mask_local,other=0.)
    valid=mask&(z>0)&(m!=float('-inf'))
    valid_local=mask_local&(z_local>0)&(m_local!=float('-inf'))
    m=gl.where(valid,m,float('-inf'))
    m_local=gl.where(valid_local,m_local,float('-inf'))
    maximum=gl.max(m,0)
    scale=gl.where(valid,gl.exp(m-maximum),0.)
    scale_local=gl.where(valid_local,gl.exp(m_local-maximum),0.)
    denominator=gl.sum(z*scale,0)
    cols=gl.arange(0,HEAD_SIZE_PADDED,layout=gl.SliceLayout(0,L))
    ptr=query_token_idx.to(gl.int64)*(num_query_heads*NUM_SEGMENTS_PER_SEQ*HEAD_SIZE_PADDED)+query_head_idx*(NUM_SEGMENTS_PER_SEQ*HEAD_SIZE_PADDED)+rows_local[:,None]*HEAD_SIZE_PADDED+cols[None,:]
    x=gl.load(segm_output_ptr+ptr,mask=mask_local[:,None]&(cols[None,:]<HEAD_SIZE),other=0.)
    acc=gl.sum(x*scale_local[:,None],0)
    acc=gl.where(denominator==0.,0.,acc/denominator)
    if USE_FP8:
        acc=acc*gl.load(out_scale_inv)
        acc=gl.clamp(acc,FP8_MIN,FP8_MAX)
    # Match U: narrowing precedes data movement; addresses and mask are rematerialized.
    output=gl.convert_layout(acc.to(gl.bfloat16),LOUT)
    cols_out=gl.arange(0,HEAD_SIZE_PADDED,layout=LOUT)
    offs=query_token_idx*output_stride_0+query_head_idx*output_stride_1+cols_out
    gl.store(output_ptr+offs,output,mask=cols_out<HEAD_SIZE)
