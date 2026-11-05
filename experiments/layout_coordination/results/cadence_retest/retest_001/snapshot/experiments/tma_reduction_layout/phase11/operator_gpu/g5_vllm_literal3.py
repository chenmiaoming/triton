"""Third literal Gluon rendering of U's two statistics paths and narrow output conversion."""
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
    act_num_segments=cdiv_fn(seq_len,tiles_per_segment*TILE_SIZE)
    rows_local=gl.arange(0,NUM_SEGMENTS_PER_SEQ,layout=gl.SliceLayout(1,L))
    rows_stat=gl.arange(0,NUM_SEGMENTS_PER_SEQ,layout=LSTAT)
    act_local=gl.full([NUM_SEGMENTS_PER_SEQ],act_num_segments,dtype=gl.int32,layout=gl.SliceLayout(1,L))
    act_stat=gl.full([NUM_SEGMENTS_PER_SEQ],act_num_segments,dtype=gl.int32,layout=LSTAT)
    mask_local=rows_local<act_local
    mask_stat=rows_stat<act_stat
    cols_local=gl.arange(0,HEAD_SIZE_PADDED,layout=gl.SliceLayout(0,L))
    cols_out=gl.arange(0,HEAD_SIZE_PADDED,layout=LOUT)
    dim_mask_local=gl.where(cols_local<HEAD_SIZE,1,0).to(gl.int1)
    dim_mask_out=gl.where(cols_out<HEAD_SIZE,1,0).to(gl.int1)
    segm_offset_local=query_token_idx.to(gl.int64)*(num_query_heads*NUM_SEGMENTS_PER_SEQ)+query_head_idx*NUM_SEGMENTS_PER_SEQ+rows_local
    segm_offset_stat=query_token_idx.to(gl.int64)*(num_query_heads*NUM_SEGMENTS_PER_SEQ)+query_head_idx*NUM_SEGMENTS_PER_SEQ+rows_stat
    m_local=gl.load(segm_max_ptr+segm_offset_local,mask=mask_local,other=float('-inf'))
    m_stat=gl.load(segm_max_ptr+segm_offset_stat,mask=mask_stat,other=float('-inf'))
    z_local=gl.load(segm_expsum_ptr+segm_offset_local,mask=mask_local,other=0.)
    z_stat=gl.load(segm_expsum_ptr+segm_offset_stat,mask=mask_stat,other=0.)
    valid_local_0=z_local>0
    valid_stat_0=z_stat>0
    valid_local_1=mask_local&valid_local_0
    valid_stat_1=mask_stat&valid_stat_0
    valid_local_2=m_local!=float('-inf')
    valid_stat_2=m_stat!=float('-inf')
    valid_local=valid_local_1&valid_local_2
    valid_stat=valid_stat_1&valid_stat_2
    m_local=gl.where(valid_local,m_local,float('-inf'))
    m_stat=gl.where(valid_stat,m_stat,float('-inf'))
    overall_max=gl.max(m_stat)
    diff_local=m_local-overall_max
    diff_stat=m_stat-overall_max
    exp_local=gl.exp(diff_local)
    exp_stat=gl.exp(diff_stat)
    scale_local=gl.where(valid_local,exp_local,0.)
    scale_stat=gl.where(valid_stat,exp_stat,0.)
    overall_expsum=gl.sum(z_stat*scale_stat)
    segm_output_offset=query_token_idx.to(gl.int64)*(num_query_heads*NUM_SEGMENTS_PER_SEQ*HEAD_SIZE_PADDED)+query_head_idx*(NUM_SEGMENTS_PER_SEQ*HEAD_SIZE_PADDED)+rows_local[:,None]*HEAD_SIZE_PADDED+cols_local[None,:]
    segm_output=gl.load(segm_output_ptr+segm_output_offset,mask=mask_local[:,None]&dim_mask_local[None,:],other=0.)
    segm_output*=scale_local[:,None]
    acc_sum=gl.sum(segm_output,axis=0)
    acc=gl.where(overall_expsum==0.,0.,acc_sum/overall_expsum)
    if USE_FP8:
        acc=acc*gl.load(out_scale_inv)
        acc=gl.clamp(acc,FP8_MIN,FP8_MAX)
    output_offset=query_token_idx*output_stride_0+query_head_idx*output_stride_1+cols_out
    output_ptrs=output_ptr+output_offset
    output=gl.convert_layout(acc.to(gl.bfloat16),LOUT)
    gl.store(output_ptrs,output,mask=dim_mask_out)
