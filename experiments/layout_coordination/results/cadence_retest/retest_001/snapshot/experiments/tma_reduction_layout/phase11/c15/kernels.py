"""Eight new MR programs. ML reuses C1 CUBINs byte for byte."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma


@gluon.jit(do_not_specialize=['K'])
def micro(desc, out, K, BM: gl.constexpr, BN: gl.constexpr, W: gl.constexpr,
          FAMILY: gl.constexpr, register_layout: gl.constexpr, shared_layout: gl.constexpr):
    gl.static_assert(FAMILY == 'MR')
    pid = gl.program_id(0)
    pm, pn = pid // (8192 // BN), pid % (8192 // BN)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    smem = gl.allocate_shared_memory(gl.bfloat16, [1, BM, BN], shared_layout)
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_load(desc, [0, pm * BM, pn * BN], bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    state = smem.load(register_layout)
    matrix_layout: gl.constexpr = gl.SliceLayout(0, register_layout)
    rows = gl.arange(0, BM, layout=gl.SliceLayout(1, matrix_layout))
    cols = gl.arange(0, BN, layout=gl.SliceLayout(0, matrix_layout))
    step = gl.expand_dims((((rows[:, None] * 13 + cols[None, :] * 7) % 7 + 1).to(gl.float32) * .25).to(gl.bfloat16), 0)
    threshold = gl.full([1, BM, BN], 8, gl.bfloat16, register_layout)
    wrap = gl.full([1, BM, BN], 16, gl.bfloat16, register_layout)
    result_layout: gl.constexpr = gl.SliceLayout(1, register_layout)
    acc = gl.full([1, BN], 0, gl.float32, result_layout)
    for i in range(K):
        # Every operand and result of private stepping is BF16. The widened
        # updated state is used only by max, never by the private recurrence.
        state = gl.where(state >= threshold, state - wrap, state + step)
        gl.static_assert(state.dtype == gl.bfloat16)
        x_fp32 = state.to(gl.float32)
        y = gl.max(x_fp32, axis=1)
        acc = acc + y
    output_cols = gl.arange(0, BN, layout=gl.SliceLayout(0, result_layout))
    gl.store(out + pid * BN + output_cols[None, :], acc)
