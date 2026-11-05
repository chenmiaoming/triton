"""Layout parameterization of the unchanged public cache-scatter body."""
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl


@g.jit
def reshape_and_cache_flash(
    key_ptr,value_ptr,key_cache_ptr,value_cache_ptr,slot_mapping_ptr,swa_slot_mapping_ptr,
    k_scale_ptr,v_scale_ptr,block_stride,value_block_stride,
    cache_token_stride:gl.constexpr,cache_head_stride:gl.constexpr,
    value_cache_token_stride:gl.constexpr,value_cache_head_stride:gl.constexpr,
    key_stride,value_stride,num_heads,head_size,block_size,
    HEAD_BLOCK:gl.constexpr,BLOCK_D:gl.constexpr,HAS_SWA:gl.constexpr,USE_SCALE:gl.constexpr,L:gl.constexpr,
):
    token_idx=gl.program_id(0);head_block_idx=gl.program_id(1)
    slot_idx=gl.load(slot_mapping_ptr+token_idx)
    if HAS_SWA:slot_idx=gl.load(swa_slot_mapping_ptr+slot_idx)
    if slot_idx<0:return
    block_idx=slot_idx//block_size;block_offset=slot_idx%block_size
    head_idx=head_block_idx*HEAD_BLOCK+gl.arange(0,HEAD_BLOCK,layout=gl.SliceLayout(1,L))
    head_mask=head_idx<num_heads
    dim_idx=gl.arange(0,BLOCK_D,layout=gl.SliceLayout(0,L))
    offs=head_idx[:,None]*head_size+dim_idx[None,:]
    mask=head_mask[:,None]&(dim_idx[None,:]<head_size)
    src_key=token_idx*key_stride+offs;src_value=token_idx*value_stride+offs
    k=gl.load(key_ptr+src_key,mask=mask);v=gl.load(value_ptr+src_value,mask=mask)
    if USE_SCALE:
        k_scale=gl.load(k_scale_ptr);v_scale=gl.load(v_scale_ptr)
        k=k/k_scale;v=v/v_scale
    tgt_k=block_idx*block_stride+block_offset*cache_token_stride+head_idx[:,None]*cache_head_stride+dim_idx[None,:]
    tgt_v=block_idx*value_block_stride+block_offset*value_cache_token_stride+head_idx[:,None]*value_cache_head_stride+dim_idx[None,:]
    gl.store(key_cache_ptr+tgt_k,k,mask=mask);gl.store(value_cache_ptr+tgt_v,v,mask=mask)
