"""Non-TMA diagnostic: explicit direct and cp.async paths, valid block programs."""
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp


@triton.jit
def canonical(x, out, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    pm, pn = pid // (8192 // BN), pid % (8192 // BN)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    offsets = tl.expand_dims(rows[:, None] * 8192 + cols[None, :], 0)
    values = tl.load(x + offsets).to(tl.float32)
    result = tl.max(values, axis=1)
    tl.store(out + pm * 8192 + cols, tl.reshape(result, [BN]))


@gluon.jit
def offsets(BM: gl.constexpr, BN: gl.constexpr, pm, pn, layout: gl.constexpr):
    matrix_layout: gl.constexpr = gl.SliceLayout(0, layout)
    rows = pm * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, matrix_layout))
    cols = pn * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, matrix_layout))
    return gl.expand_dims(rows[:, None] * 8192 + cols[None, :], 0)


@gluon.jit
def explicit(x, out, BM: gl.constexpr, BN: gl.constexpr, W: gl.constexpr,
             USE_SHARED: gl.constexpr, register_layout: gl.constexpr,
             copy_layout: gl.constexpr, shared_layout: gl.constexpr):
    pid = gl.program_id(0)
    pm, pn = pid // (8192 // BN), pid % (8192 // BN)
    if USE_SHARED:
        # The writer layout/shared layout are fixed across all consumer layouts.
        ptrs = x + offsets(BM, BN, pm, pn, copy_layout)
        smem = gl.allocate_shared_memory(gl.bfloat16, [1, BM, BN], shared_layout)
        cp.async_copy_global_to_shared(smem, ptrs)
        cp.commit_group()
        cp.wait_group(0)
        raw = smem.load(register_layout)
    else:
        raw = gl.load(x + offsets(BM, BN, pm, pn, register_layout))
    result = gl.max(raw.to(gl.float32), axis=1)
    output_layout: gl.constexpr = gl.BlockedLayout([1], [32], [W], [0])
    values = gl.convert_layout(gl.reshape(result, [BN]), output_layout)
    cols = pn * BN + gl.arange(0, BN, layout=output_layout)
    gl.store(out + pm * 8192 + cols, values)
