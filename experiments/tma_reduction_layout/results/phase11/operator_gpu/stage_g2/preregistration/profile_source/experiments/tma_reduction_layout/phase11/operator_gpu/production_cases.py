"""Bounded source-level reproductions of two verified non-TensorCore paths."""
import math
from experiments.tma_reduction_layout.phase11.operator_survey.cases import layouts


def geometries():
    rows = []
    for segments, dim in ((16, 128), (32, 128), (32, 256)):
        rows.append(dict(id=f'C06_s{segments}_d{dim}', family='C06', BM=segments, BN=dim,
                         W=4, num_stages=3, segments=segments, H=32))
    for dim in (64, 128, 256):
        rows.append(dict(id=f'C02_n64_d{dim}', family='C02', BM=64, BN=dim,
                         W=4, num_stages=2, segments=4, H=32))
    for row in rows:
        row['layouts'] = layouts(row['BM'], row['BN'], row['W'])
    return rows


def configuration(case, stage2=False):
    h, dim, splits = case['H'], case['BN'], case['segments']
    if case['family'] == 'C06':
        constants = dict(num_query_heads=h, TILE_SIZE=32, HEAD_SIZE=dim, HEAD_SIZE_PADDED=dim,
                         BLOCK_Q=16, NUM_SEGMENTS_PER_SEQ=splits, USE_FP8=False, FP8_MIN=-448., FP8_MAX=448.)
        pointers = dict(output_ptr='*bf16', segm_output_ptr='*fp32', segm_max_ptr='*fp32',
                        segm_expsum_ptr='*fp32', seq_lens_ptr='*i32', query_start_len_ptr='*i32')
        values = dict(num_seqs=4, out_scale_inv=1., output_stride_0=h*dim, output_stride_1=dim,
                      block_table_stride=256)
        types = {key: 'i64' for key in ('output_stride_0', 'output_stride_1', 'block_table_stride')}
        types.update(num_seqs='i32', out_scale_inv='fp32')
        return constants, pointers, values, types
    if stage2:
        constants = dict(MAX_KV_SPLITS=splits, MIN_BLOCK_KV=32, BLOCK_DV=dim, Lv=dim,
                         HAS_SINK=False, USE_PDL=False, FORCED_KV_SPLITS=0, sink_ptr=None)
        pointers = dict(Mid_O='*fp32', Mid_O_1='*fp32', O='*bf16', kv_indptr='*i32', num_kv_splits='*i32')
        values = dict(v_scale=1., stride_mid_ob=h*splits*dim, stride_mid_oh=splits*dim,
                      stride_mid_os=dim, stride_obs=h*dim, stride_oh=dim)
    else:
        constants = dict(kv_group_num=1, BLOCK_DMODEL=dim, BLOCK_DV=dim, BLOCK_N=64,
                         MIN_BLOCK_KV=32, logit_cap=0., Lk=dim, Lv=dim,
                         xai_temperature_len=-1, PAGE_SIZE=1, SCORE_MOD=None,
                         Aux0=None, aux0_stride_t=0, aux0_stride_h=0, aux0_len=0,
                         uneven_batch=None, ROTATE_BATCH=False)
        pointers = dict(Q='*bf16', K_Buffer='*bf16', V_Buffer='*bf16', kv_indptr='*i32',
                        kv_indices='*i32', Att_Out='*fp32', Att_Lse='*fp32', num_kv_splits='*i32')
        values = dict(sm_scale_withk=1/math.sqrt(dim), stride_qbs=h*dim, stride_qh=dim,
                      stride_buf_kbs=h*dim, stride_buf_kh=dim, stride_buf_vbs=h*dim, stride_buf_vh=dim,
                      stride_buf_kpage=h*dim, stride_buf_ktok=h*dim, stride_buf_vpage=h*dim,
                      stride_buf_vtok=h*dim, stride_mid_ob=h*splits*dim,
                      stride_mid_oh=splits*dim, stride_mid_os=dim)
    types = {key: ('fp32' if isinstance(value, float) else 'i32') for key, value in values.items()}
    return constants, pointers, values, types
