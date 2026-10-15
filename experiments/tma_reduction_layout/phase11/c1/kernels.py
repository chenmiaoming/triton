"""Sixteen legal Gluon programs: eight layouts times ML/MR, runtime K.

No hardware IDs, inline assembly, compiler edits, or reduction-result feedback.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma


@gluon.jit(do_not_specialize=['K'])
def micro(desc, out, K, BM: gl.constexpr, BN: gl.constexpr, W: gl.constexpr,
          FAMILY: gl.constexpr, register_layout: gl.constexpr,
          shared_layout: gl.constexpr):
    pid = gl.program_id(0)
    pm, pn = pid // (8192 // BN), pid % (8192 // BN)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    matrix_layout: gl.constexpr = gl.SliceLayout(0, register_layout)
    rows = gl.arange(0, BM, layout=gl.SliceLayout(1, matrix_layout))
    cols = gl.arange(0, BN, layout=gl.SliceLayout(0, matrix_layout))
    if FAMILY == 'ML':
        # Both slots are filled once, outside the measured repetition loop.
        # Runtime memdesc indexing prevents a single invariant load from
        # representing all K iterations. NCU must confirm every load PC.
        smem = gl.allocate_shared_memory(gl.bfloat16, [2, 1, BM, BN], shared_layout)
        mbarrier.expect(bar, 2 * desc.block_type.nbytes)
        tma.async_load(desc, [0, pm * BM, pn * BN], bar, smem.index(0))
        tma.async_load(desc, [0, ((pm + 128) % 256) * BM, pn * BN], bar, smem.index(1))
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)
        acc = gl.full([1, BM, BN], 0, gl.float32, register_layout)
        for i in range(K):
            value = smem.index(i & 1).load(register_layout)
            acc = acc + value.to(gl.float32)
        offsets = pid * BM * BN + rows[:, None] * BN + cols[None, :]
        gl.store(out + gl.expand_dims(offsets, 0), acc)
    else:
        smem = gl.allocate_shared_memory(gl.bfloat16, [1, BM, BN], shared_layout)
        mbarrier.expect(bar, desc.block_type.nbytes)
        tma.async_load(desc, [0, pm * BM, pn * BN], bar, smem)
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)
        state = smem.load(register_layout)
        # Coordinate tensors follow the requested layout. Each element updates
        # privately; neither y nor acc ever feeds the input tensor or a smem store.
        step = gl.expand_dims(((rows[:, None] * 13 + cols[None, :] * 7) % 7 + 1).to(gl.float32) * 0.25, 0)
        result_layout: gl.constexpr = gl.SliceLayout(1, register_layout)
        acc = gl.full([1, BN], 0, gl.float32, result_layout)
        for i in range(K):
            y = gl.max(state.to(gl.float32), axis=1)
            acc = acc + y
            value = state.to(gl.float32)
            state = gl.where(value >= 8.0, value - 16.0, value + step).to(gl.bfloat16)
        # Preserve the native reduction result layout, with no reverse
        # conversion/broadcast to the input layout and no explicit smem writes.
        output_cols = gl.arange(0, BN, layout=gl.SliceLayout(0, result_layout))
        gl.store(out + pid * BN + output_cols[None, :], acc)
