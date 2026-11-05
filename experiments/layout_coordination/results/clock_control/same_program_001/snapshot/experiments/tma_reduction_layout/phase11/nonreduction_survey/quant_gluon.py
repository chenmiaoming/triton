"""Layout-explicit translations of the pinned scale-swizzle/AWQ kernels.

The source functions and their original license notices are retained under
sources/quant_*.  G0 is admissible only after the full native-code/ABI gate.
Index expressions use AutoLayout until an explicit memory-boundary encoding;
values have concrete layouts.  No compiler pipeline is modified.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def triton_scale_swizzle(
    scale_ptr, scale_rows, scale_cols, output_ptr, input_row_stride,
    input_col_stride, output_block_stride, BLOCK_ROWS: gl.constexpr,
    BLOCK_COLS: gl.constexpr, LINPUT: gl.constexpr, LOUT: gl.constexpr,
):
    pid_row = gl.program_id(0)
    pid_col = gl.program_id(1)
    rows = gl.arange(0, BLOCK_ROWS)[:, None]
    cols = gl.arange(0, BLOCK_COLS)[None, :]
    start_row = pid_row * BLOCK_ROWS
    start_col = pid_col * BLOCK_COLS
    global_rows = start_row + rows
    global_cols = start_col + cols
    mask = (global_rows < scale_rows) & (global_cols < scale_cols)
    input_pointer = gl.set_auto_layout(
        scale_ptr + global_rows*input_row_stride + global_cols*input_col_stride,
        LINPUT)
    input_scales = gl.load(input_pointer, mask=gl.set_auto_layout(mask, LINPUT), other=0.0)
    # The input and output have independent ownership.  Rematerialize the
    # logical destination index in LOUT instead of constraining the same Auto
    # rows/columns to both layouts.  flat//BLOCK_COLS and flat%BLOCK_COLS are
    # exactly the original row/column pair, with the same padded store domain.
    flat = gl.arange(0, BLOCK_ROWS*BLOCK_COLS, layout=LOUT)
    output_rows = flat//BLOCK_COLS
    output_cols = flat%BLOCK_COLS
    dest_indices_flat = (output_rows%32)*16 + (output_rows//32)*4 + output_cols
    scales_flat = gl.reshape(input_scales, (BLOCK_ROWS*BLOCK_COLS,))
    LOCAL_NUMEL = BLOCK_ROWS*BLOCK_COLS
    block_offset = pid_col*LOCAL_NUMEL + pid_row*output_block_stride
    output_pointer = output_ptr + block_offset + dest_indices_flat
    gl.store(output_pointer, gl.convert_layout(scales_flat, LOUT))



@gluon.jit
def _interleave(a, b):
    # Exact join/reshape semantics of triton.language.standard.interleave.
    joined = gl.join(a, b)
    return gl.reshape(joined, joined.shape[:-2] + [2*joined.shape[-2]])


@gluon.jit
def awq_dequantize_kernel(
    qweight_ptr, scales_ptr, zeros_ptr, group_size, result_ptr, num_cols,
    num_rows, BLOCK_SIZE_X: gl.constexpr, BLOCK_SIZE_Y: gl.constexpr,
    LQ: gl.constexpr, LZ: gl.constexpr, LS: gl.constexpr,
    LEXP: gl.constexpr, LOUT: gl.constexpr,
):
    pid_x = gl.program_id(0)
    pid_y = gl.program_id(1)
    # U rematerializes shared logical indices in each memory encoding.  Make
    # these block tensors explicit, preserving its operation order as well as
    # its address equations; otherwise AutoLayout can change backend schedule.
    row_origin = pid_y*BLOCK_SIZE_Y
    rows_q = gl.arange(0, BLOCK_SIZE_Y, layout=gl.SliceLayout(1, LQ))
    rows_o = gl.arange(0, BLOCK_SIZE_Y, layout=gl.SliceLayout(1, LOUT))
    cols_q = gl.arange(0, BLOCK_SIZE_X, layout=gl.SliceLayout(0, LQ))
    row_origin_q = gl.full((BLOCK_SIZE_Y,), row_origin, gl.int32, gl.SliceLayout(1, LQ))
    row_origin_o = gl.full((BLOCK_SIZE_Y,), row_origin, gl.int32, gl.SliceLayout(1, LOUT))
    offsets_y = row_origin_q+rows_q
    result_offsets_y = row_origin_o+rows_o
    col_origin = pid_x*BLOCK_SIZE_X
    offsets_x = gl.full((BLOCK_SIZE_X,), col_origin, gl.int32, gl.SliceLayout(0, LQ))+cols_q
    offsets_y_2d = offsets_y[:, None]
    result_offsets_y_2d = result_offsets_y[:, None]
    offsets_row = gl.full((BLOCK_SIZE_Y, 1), num_cols, gl.int32, LQ)*offsets_y_2d
    offsets_x_2d = offsets_x[None, :]
    offsets = offsets_row+offsets_x_2d
    num_rows_q = gl.full((BLOCK_SIZE_Y,), num_rows, gl.int32, gl.SliceLayout(1, LQ))
    num_rows_o = gl.full((BLOCK_SIZE_Y,), num_rows, gl.int32, gl.SliceLayout(1, LOUT))
    masks_y = offsets_y < num_rows_q
    result_masks_y = result_offsets_y < num_rows_o
    masks_x = offsets_x < num_cols
    masks_y_2d = masks_y[:, None]
    result_masks_y_2d = result_masks_y[:, None]
    masks_x_2d = masks_x[None, :]
    masks = masks_y_2d & masks_x_2d
    result_col_origin = pid_x*BLOCK_SIZE_X*8
    cols_s = gl.arange(0, BLOCK_SIZE_X*8, layout=gl.SliceLayout(0, LS))
    cols_o = gl.arange(0, BLOCK_SIZE_X*8, layout=gl.SliceLayout(0, LOUT))
    col_origin_s = gl.full((BLOCK_SIZE_X*8,), result_col_origin, gl.int32, gl.SliceLayout(0, LS))
    col_origin_o = gl.full((BLOCK_SIZE_X*8,), result_col_origin, gl.int32, gl.SliceLayout(0, LOUT))
    scale_offsets_x = col_origin_s+cols_s
    result_offsets_x = col_origin_o+cols_o
    result_num_cols = num_cols*8
    result_offsets_row = gl.full((BLOCK_SIZE_Y, 1), result_num_cols, gl.int32, LOUT)*result_offsets_y_2d
    scale_offsets_x_2d = scale_offsets_x[None, :]
    result_offsets_x_2d = result_offsets_x[None, :]
    result_offsets = result_offsets_row+result_offsets_x_2d
    scale_num_cols = gl.full((BLOCK_SIZE_X*8,), result_num_cols, gl.int32, gl.SliceLayout(0, LS))
    output_num_cols = gl.full((BLOCK_SIZE_X*8,), result_num_cols, gl.int32, gl.SliceLayout(0, LOUT))
    scale_masks_x = scale_offsets_x < scale_num_cols
    result_masks_x = result_offsets_x < output_num_cols
    scale_masks_x_2d = scale_masks_x[None, :]
    result_masks_x_2d = result_masks_x[None, :]
    result_masks = result_masks_y_2d & result_masks_x_2d
    iweights = gl.load(qweight_ptr + offsets, masks, 0.0)
    iweights = _interleave(iweights, iweights)
    iweights = _interleave(iweights, iweights)
    iweights = _interleave(iweights, iweights)
    iweights = gl.convert_layout(iweights, LEXP)
    reverse_awq_order_tensor = (
        (gl.arange(0, 2)*4)[None, :] + gl.arange(0, 4)[:, None]
    ).reshape(8)
    shifts = reverse_awq_order_tensor*4
    shifts = shifts[None, :].broadcast_to((BLOCK_SIZE_Y*BLOCK_SIZE_X, 8))
    shifts = gl.reshape(shifts, (BLOCK_SIZE_Y, BLOCK_SIZE_X*8))
    shifts = gl.set_auto_layout(shifts, LEXP)
    iweights = (iweights >> shifts) & 0xF
    # The optimized U program scalarizes the singleton row index.  Preserve
    # that block-uniform scalar instead of creating a rank-one Gluon splat.
    zero_offsets_y = pid_y*BLOCK_SIZE_Y//group_size
    zero_offsets_x = pid_x*BLOCK_SIZE_X + gl.arange(0, BLOCK_SIZE_X)
    zero_offsets = num_cols*zero_offsets_y + zero_offsets_x[None, :]
    zero_masks_y = zero_offsets_y < num_rows//group_size
    zero_masks_x = zero_offsets_x < num_cols
    zero_masks = zero_masks_y & zero_masks_x[None, :]
    zeros = gl.load(gl.set_auto_layout(zeros_ptr + zero_offsets, LZ),
        gl.set_auto_layout(zero_masks, LZ), 0.0)
    zeros = _interleave(zeros, zeros)
    zeros = _interleave(zeros, zeros)
    zeros = _interleave(zeros, zeros)
    zeros = zeros.broadcast_to((BLOCK_SIZE_Y, BLOCK_SIZE_X*8))
    zeros = gl.convert_layout(zeros, LEXP)
    zeros = (zeros >> shifts) & 0xF
    scale_offsets_y = pid_y*BLOCK_SIZE_Y//group_size
    scale_offsets = result_num_cols*scale_offsets_y + scale_offsets_x_2d
    scale_masks_y = scale_offsets_y < num_rows//group_size
    scale_masks = scale_masks_y & scale_masks_x_2d
    scales = gl.load(scales_ptr + scale_offsets, scale_masks, 0.0)
    # U narrows the integer difference before redistributing it to the output
    # encoding, and redistributes the scale row before its row broadcast.
    # Keeping those boundaries is part of the G0 code-equivalence contract.
    scales = gl.convert_layout(scales, LOUT)
    scales = scales.broadcast_to((BLOCK_SIZE_Y, BLOCK_SIZE_X*8))
    integer_difference = iweights - zeros
    difference = integer_difference.to(scales.dtype)
    difference = gl.convert_layout(difference, LOUT)
    iweights = difference*scales
    iweights = iweights.to(result_ptr.type.element_ty)
    gl.store(result_ptr + result_offsets, iweights, result_masks)


def layout_constants(case, plan, variant='G0'):
    from ..operator_gpu.layout_parsing import instantiate
    if isinstance(variant, str):
        assert variant == 'G0', variant
        current = plan
    else:
        current = {**plan, **variant.get('plan_override', {})}
    names = {'LINPUT':'input', 'LOUT':'output'} if case['family']=='N01' else {
        'LQ':'qweight', 'LZ':'zeros', 'LS':'scales', 'LEXP':'expanded', 'LOUT':'output'}
    def concrete(spec):
        if spec['kind']=='linear':
            return gl.DistributedLinearLayout(**spec['fields'])
        return instantiate(spec, gl)
    return {name:concrete(current[key]) for name,key in names.items()}



def variants(case, plan):
    from .quant_adapter import variants as choices
    return choices(case, plan)
