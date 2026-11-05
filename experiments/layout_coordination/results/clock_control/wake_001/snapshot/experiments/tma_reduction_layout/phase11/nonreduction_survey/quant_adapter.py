"""CPU-only fixtures and lazy compilation adapters for N01/N02.

The original kernels are exact AST-preserving extractions from pinned public
sources.  Importing this module, including constructing fixtures, imports no
Triton code and performs no GPU work.
"""
import copy
from pathlib import Path


def cases():
    result = []
    for rows, cols in [(4096, 128), (11008, 128)]:
        result.append(dict(id=f'N01_m{rows}_n{cols}', family='N01', W=4,
            stages=3, rows=rows, cols=cols, BM=128, BN=4,
            caller_defaults='Original caller omits warp/stage options; pinned CUDA defaults are 4/3.',
            source_function='triton_scale_swizzle',
            evidence_scope='Optional torchao Triton scale-rearrangement backend; no scale calculation.'))
    for rows, packed_cols in [(3584, 448), (1024, 128)]:
        result.append(dict(id=f'N02_k{rows}_p{packed_cols}_g128', family='N02',
            W=4, stages=3, rows=rows, packed_cols=packed_cols,
            cols=packed_cols * 8, group_size=128, BM=32, BN=32,
            caller_defaults='Original caller omits warp/stage options; pinned CUDA defaults are 4/3.',
            source_function='awq_dequantize_kernel',
            evidence_scope='Optional VLLM_USE_TRITON_AWQ dequantization backend; not AWQ GEMM.'))
    return result


def upstream(case):
    from .sources import quant_original
    return getattr(quant_original, case['source_function'])


def provenance():
    import json
    return json.loads((Path(__file__).parent/'sources/quant_provenance.json').read_text())


def fixture(case, pattern='finite'):
    import torch
    assert pattern in ['finite', 'structured', 'edge'], pattern
    generator = torch.Generator(device='cpu').manual_seed(20261105 + case['rows'] + case['cols'])
    if case['family'] == 'N01':
        rows, cols = case['rows'], case['cols']
        values = torch.randint(0, 256, (rows, cols), generator=generator, dtype=torch.uint8)
        if pattern != 'finite':
            values = ((torch.arange(rows)[:, None]*17 + torch.arange(cols)[None, :]*31) % 256).to(torch.uint8)
            if pattern == 'edge':
                values[::7, ::3] = 255
                values[::11, ::5] = 0
        row_stride = case.get('row_stride', cols)
        col_stride = case.get('col_stride', 1)
        from math import gcd
        assert row_stride > 0 and col_stride > 0
        divisor = gcd(row_stride, col_stride)
        assert not (col_stride//divisor < rows and row_stride//divisor < cols), 'Overlapping input views are excluded.'
        storage = torch.full(((rows-1)*row_stride+(cols-1)*col_stride+1,), 0x6B, dtype=torch.uint8)
        source = torch.as_strided(storage, (rows, cols), (row_stride, col_stride))
        source.copy_(values)
        row_blocks, col_blocks = (rows+127)//128, (cols+3)//4
        padded_rows, padded_cols = row_blocks*128, col_blocks*4
        padded = torch.zeros((padded_rows, padded_cols), dtype=torch.uint8)
        padded[:rows, :cols] = source
        # Independent high-level permutation reference from the documented
        # block format, rather than duplicating the kernel's address equation.
        expected = (padded.reshape(row_blocks, 128, col_blocks, 4)
            .permute(0, 2, 1, 3).contiguous().reshape(-1, 4, 32, 4)
            .transpose(1, 2).contiguous().reshape(padded_rows, padded_cols))
        output = torch.full_like(expected, 0xCD)
        return dict(tensors=dict(scale_ptr=source, output_ptr=output),
            scalars=dict(scale_rows=rows, scale_cols=cols,
                input_row_stride=row_stride, input_col_stride=col_stride,
                output_block_stride=512*col_blocks),
            constants=dict(BLOCK_ROWS=128, BLOCK_COLS=4),
            grid=(row_blocks, col_blocks, 1), outputs=['output_ptr'],
            expected=dict(output_ptr=expected), inplace=[],
            tolerances=dict(output_ptr=dict(rtol=0., atol=0., bitwise=True)))
    assert case['family'] == 'N02'
    rows, packed_cols, group = case['rows'], case['packed_cols'], case['group_size']
    assert rows % group == 0 and group >= 32 and group % 32 == 0
    cols = packed_cols*8
    qweight = torch.randint(0, 2**32, (rows, packed_cols), generator=generator, dtype=torch.int64).to(torch.int32)
    zeros = torch.randint(0, 2**32, (rows//group, packed_cols), generator=generator, dtype=torch.int64).to(torch.int32)
    scales = (torch.rand((rows//group, cols), generator=generator)*0.75+0.125).to(torch.float16)
    if pattern != 'finite':
        # Populate every physical nibble; the logical reverse-AWQ order is
        # checked separately by the unpacking reference below.
        index = torch.arange(rows*packed_cols, dtype=torch.int64).reshape(rows, packed_cols)
        qweight = sum((((index+n*3) % 16) << (4*n)) for n in range(8)).to(torch.int32)
        index = torch.arange((rows//group)*packed_cols, dtype=torch.int64).reshape(rows//group, packed_cols)
        zeros = sum((((index+n*5) % 16) << (4*n)) for n in range(8)).to(torch.int32)
        values = torch.tensor([0., 2.**-14, 0.125, 0.5, 1., 2., 32., 1024.], dtype=torch.float16)
        scales = values[(torch.arange(scales.numel()).reshape(scales.shape) % values.numel())]
        if pattern == 'edge':
            qweight[::13] = -1
            qweight[::17] = 0
            zeros[::3] = 0
    reverse_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], dtype=torch.int64)
    def unpack(values):
        return ((values.to(torch.int64)[..., None] >> (4*reverse_order)) & 15).reshape(values.shape[0], -1)
    decoded = unpack(qweight)
    decoded_zero = unpack(zeros).repeat_interleave(group, dim=0)
    # Integer differences are exactly representable.  Multiplication in FP64
    # followed by one FP16 conversion specifies the FP16 product rounding.
    expected = ((decoded-decoded_zero).to(torch.float64)
        * scales.to(torch.float64).repeat_interleave(group, dim=0)).to(torch.float16)
    return dict(tensors=dict(qweight_ptr=qweight, scales_ptr=scales,
            zeros_ptr=zeros, result_ptr=torch.full((rows, cols), -123., dtype=torch.float16)),
        scalars=dict(group_size=group, num_cols=packed_cols, num_rows=rows),
        constants=dict(BLOCK_SIZE_X=32, BLOCK_SIZE_Y=32),
        grid=((packed_cols+31)//32, (rows+31)//32, 1), outputs=['result_ptr'],
        expected=dict(result_ptr=expected), inplace=[],
        tolerances=dict(result_ptr=dict(rtol=0., atol=0., bitwise=True)))


def derive_plan(ttgir, case):
    from ..operator_gpu.layout_parsing import encodings, tensor_encoding
    import ast
    import re
    definitions = encodings(ttgir)
    lines = ttgir.splitlines()
    linear_definitions = dict(re.findall(r'(#\w+)\s*=\s*#ttg\.linear<\{([^\n]+)\}>',ttgir))
    def encoding(line, shape):
        try:
            return tensor_encoding(line, shape, definitions)
        except (AssertionError,KeyError):
            match = re.search(r',\s*(#\w+)>\s*(?:loc|$)',line)
            assert match and match[1] in linear_definitions,line
            body = linear_definitions[match[1]]
            fields = {}
            for original,key in [('register','reg_bases'),('lane','lane_bases'),
                                 ('warp','warp_bases'),('block','block_bases')]:
                found = re.search(original+r'\s*=\s*(\[(?:[^\[\]]|\[[^\[\]]*\])*\])',body)
                assert found,(original,body)
                fields[key] = ast.literal_eval(found[1])
            fields['shape'] = list(shape)
            return dict(kind='linear',fields=fields)
    result = dict(family=case['family'], layouts=copy.deepcopy(definitions))
    if case['family'] == 'N01':
        load = next(line for line in lines if 'tt.load' in line and 'tensor<128x4x' in line)
        store = next(line for line in lines if 'tt.store' in line)
        result.update(input=encoding(load, [128, 4]),
                      output=encoding(store, [512]))
    else:
        load = lambda shape: next(line for line in lines if 'tt.load' in line
            and 'tensor<'+'x'.join(map(str, shape))+'x' in line)
        store = next(line for line in lines if 'tt.store' in line)
        expanded = next(line for line in lines if 'arith.subi' in line
            and 'tensor<32x256xi32' in line)
        result.update(qweight=tensor_encoding(load([32, 32]), [32, 32], definitions),
            zeros=tensor_encoding(load([1, 32]), [1, 32], definitions),
            scales=tensor_encoding(load([1, 256]), [1, 256], definitions),
            expanded=tensor_encoding(expanded, [32, 256], definitions),
            output=tensor_encoding(store, [32, 256], definitions))
        result['loads'] = [line for line in lines if 'tt.load' in line]
        result['stores'] = [line for line in lines if 'tt.store' in line]
        result['conversions'] = [line for line in lines if 'convert_layout' in line]
    return result


def gluon(case):
    from . import quant_gluon
    return getattr(quant_gluon, case['source_function'])


def layout_constants(case, plan, variant='G0'):
    from .quant_gluon import layout_constants as constants
    return constants(case, plan, variant)


def variants(case, plan):
    if case['family'] == 'N01':
        choices = [([1,1],[8,4],[4,1],[1,0]),
            ([1,2],[16,2],[4,1],[1,0]),
            ([4,1],[8,4],[4,1],[0,1]),
            ([2,2],[16,2],[4,1],[1,0])]
        result = []
        for spt,tpw,wpc,order in choices:
            spec = dict(kind='blocked',fields=dict(size_per_thread=spt,
                threads_per_warp=tpw,warps_per_cta=wpc,order=order))
            if spec == plan['input']:continue
            name='load_s'+str(spt[0])+'x'+str(spt[1])+'_t'+str(tpw[0])+'x'+str(tpw[1])+'_o'+''.join(map(str,order))
            result.append(dict(name=name,plan_override=dict(input=spec)))
        # Output-friendly ownership is a permutation of the same four logical
        # bytes per hardware thread.  These are layout bases, not source-level
        # lane/warp IDs.  Both a full and a partial permutation are tested,
        # together with matched input ownership to separate conversion costs.
        output_maps = {
            'fully_coalesced':dict(reg_bases=[[1],[2]],
                lane_bases=[[128],[256],[4],[8],[16]],warp_bases=[[32],[64]],
                block_bases=[],shape=[512]),
            'partially_coalesced':dict(reg_bases=[[1],[2]],
                lane_bases=[[4],[8],[16],[32],[128]],warp_bases=[[64],[256]],
                block_bases=[],shape=[512]),
        }
        for name,fields in output_maps.items():
            output=dict(kind='linear',fields=fields)
            result.append(dict(name='output_'+name,plan_override=dict(output=output)))
            projected={key:[[value[0]//4,value[0]%4] for value in fields[key]]
                for key in ['reg_bases','lane_bases','warp_bases','block_bases']}
            projected['shape']=[128,4]
            result.append(dict(name='coordinated_'+name,
                plan_override=dict(input=dict(kind='linear',fields=projected),output=copy.deepcopy(output))))
        assert len(result)==8
        return result
    def blocked(spt, tpw, wpc=(4,1), order=(1,0)):
        return dict(kind='blocked',fields=dict(size_per_thread=list(spt),
            threads_per_warp=list(tpw),warps_per_cta=list(wpc),order=list(order)))
    result = []
    for width in (1,2):
        spec = copy.deepcopy(plan['qweight'])
        spec['fields']['size_per_thread'] = [1,width]
        result.append(dict(name=f'weight_load_vec{width}',plan_override=dict(qweight=spec)))
    for width,tpw in [(1,(1,32)),(2,(2,16))]:
        packed = blocked((1,width),tpw)
        expanded = blocked((1,width*8),tpw)
        result.append(dict(name=f'packed_compute_vec{width}',
            plan_override=dict(qweight=packed,zeros=copy.deepcopy(packed),expanded=expanded)))
    for width in (2,4):
        result.append(dict(name=f'output_vec{width}',
            plan_override=dict(output=blocked((1,width),(1,32)))))
    result.append(dict(name='output_lane_rows2',
        plan_override=dict(output=blocked((1,8),(2,16)))))
    result.append(dict(name='scale_load_output_encoding',
        plan_override=dict(scales=copy.deepcopy(plan['output']))))
    assert len(result)==8
    return result
