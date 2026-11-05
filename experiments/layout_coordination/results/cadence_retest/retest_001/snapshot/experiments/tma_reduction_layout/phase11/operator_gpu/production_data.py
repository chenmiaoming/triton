"""Synthetic serving-sized inputs and independent references for pinned kernel bodies."""
import math
import numpy as np
from experiments.tma_reduction_layout.phase11.operator_survey.oracles import bf16_round
from .production_cases import configuration


def make_data(case, batch=4, length=129, pattern='finite', with_reference=True, padded=False):
    h, dim, splits = case['H'], case['BN'], case['segments']
    rng = np.random.default_rng(20261021 + dim + splits)
    def random(shape, bf16=False):
        value = (rng.standard_normal(shape) * .125).astype(np.float32)
        if pattern == 'quarter':
            value = ((np.arange(value.size).reshape(shape) % 17 - 8) / 8.).astype(np.float32)
        return bf16_round(value) if bf16 else value
    constants, types, scalars, scalar_types = configuration(case)
    arrays, expected, masks = {}, {}, {}
    if case['family'] == 'C06':
        slots = batch + (2 if padded else 0)
        arrays.update(segm_output_ptr=random((batch, h, splits, dim)),
                      segm_max_ptr=rng.uniform(-5., 5., (batch, h, splits)).astype(np.float32),
                      segm_expsum_ptr=rng.uniform(.25, 2., (batch, h, splits)).astype(np.float32),
                      seq_lens_ptr=np.full(batch, length, np.int32),
                      query_start_len_ptr=np.arange(batch+1, dtype=np.int32))
        arrays['segm_output_ptr'] *= arrays['segm_expsum_ptr'][..., None]
        if pattern == 'masked':
            arrays['seq_lens_ptr'][0] = 0
            if batch >= 4:
                arrays['seq_lens_ptr'][1:4] = [17, length, length*2]
                arrays['query_start_len_ptr'][:5] = [0, 1, 3, 3, 4]
            arrays['segm_max_ptr'][:, 0, :] = -np.inf
            arrays['segm_expsum_ptr'][:, 0, :] = 1.
        output_shapes = {'output_ptr': (slots, h, dim)}
        scalars['num_seqs'] = batch
        if with_reference:
            value = np.zeros((slots, h, dim), np.float64)
            mask = np.zeros((slots, h, dim), bool)
            for token in range(batch):
                seq = np.searchsorted(arrays['query_start_len_ptr'], token, side='right') - 1
                seq_len = int(arrays['seq_lens_ptr'][seq])
                if seq_len <= 0:
                    continue
                tiles = (seq_len + splits*32-1)//(splits*32)
                active = (seq_len + tiles*32-1)//(tiles*32)
                maximum = arrays['segm_max_ptr'][token].astype(np.float64)
                sums = arrays['segm_expsum_ptr'][token].astype(np.float64)
                valid = (np.arange(splits)[None, :] < active) & (sums > 0) & np.isfinite(maximum)
                maximum = np.where(valid, maximum, -np.inf)
                with np.errstate(invalid='ignore'):
                    scale = np.where(valid, np.exp(maximum - maximum.max(1, keepdims=True)), 0.)
                den = (sums * scale).sum(1)
                num = (arrays['segm_output_ptr'][token].astype(np.float64) * scale[..., None]).sum(1)
                np.divide(num, den[:, None], out=value[token], where=den[:, None] > 0)
                mask[token] = True
            expected['output_ptr'], masks['output_ptr'] = value, mask
        grid = (slots, h, 1)
    else:
        arrays.update(Q=random((batch, h, dim), True), K_Buffer=random((batch*length, h, dim), True),
                      V_Buffer=random((batch*length, h, dim), True),
                      kv_indptr=np.arange(batch+1, dtype=np.int32)*length,
                      kv_indices=rng.permutation(batch*length).astype(np.int32),
                      num_kv_splits=np.full(batch, splits, np.int32))
        if pattern == 'masked' and batch > 1:
            arrays['num_kv_splits'][::2] = max(1, splits-1)
        output_shapes = {'Att_Out': (batch, h, splits, dim), 'Att_Lse': (batch, h, splits), 'O': (batch, h, dim)}
        types.update(O='*bf16')
        if with_reference:
            mid = np.zeros(output_shapes['Att_Out'], np.float64)
            lse = np.zeros(output_shapes['Att_Lse'], np.float64)
            valid = np.zeros(output_shapes['Att_Lse'], bool)
            full = np.zeros(output_shapes['O'], np.float64)
            for b in range(batch):
                indices = arrays['kv_indices'][b*length:(b+1)*length]
                q = arrays['Q'][b]
                k = arrays['K_Buffer'][indices]
                v = arrays['V_Buffer'][indices].astype(np.float64)
                # Preserve upstream's BF16 product followed by an FP32 reduction.
                scores = bf16_round(q[None, :, :] * k).astype(np.float64).sum(2) / math.sqrt(dim)
                maximum = scores.max(0)
                probabilities = np.exp(scores-maximum[None, :])
                full[b] = (probabilities[..., None] * v).sum(0) / probabilities.sum(0)[:, None]
                count = int(arrays['num_kv_splits'][b])
                per = ((length+count-1)//count+31)//32*32
                for s in range(splits):
                    lo, hi = s*per, min((s+1)*per, length)
                    if hi <= lo:
                        continue
                    m = scores[lo:hi].max(0)
                    p = np.exp(scores[lo:hi]-m[None, :])
                    total = p.sum(0)
                    mid[b, :, s, :] = (p[..., None]*v[lo:hi]).sum(0)/total[:, None]
                    lse[b, :, s] = m+np.log(total)
                    valid[b, :, s] = True
            expected.update(Att_Out=mid, Att_Lse=lse, O=full)
            masks.update(Att_Out=np.broadcast_to(valid[..., None], mid.shape), Att_Lse=valid,
                         O=np.ones(full.shape, bool))
        grid = (batch, h, splits)
    return {'arrays': arrays, 'types': types, 'output_shapes': output_shapes, 'expected': expected,
            'masks': masks, 'scalars': scalars, 'grid': grid, 'batch': batch, 'length': length}
