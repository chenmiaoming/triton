"""Outcome-blind, shape-general Stage B kernels; no timing entry points."""
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier


def make_canonical():
    # Fresh JIT object per candidate, as in the frozen fixed-binary pilot.
    @triton.jit
    def canonical_kernel(a_ptr, out_ptr, stride_b, stride_m,
                         B_DESC: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
        pid = tl.program_id(0)
        desc = tl.make_tensor_descriptor(a_ptr, shape=[B_DESC, M, N],
                                         strides=[stride_b, stride_m, 1], block_shape=[1, M, N])
        x = desc.load([pid, 0, 0])
        x_fp32 = x.to(tl.float32)
        y = tl.max(x_fp32, axis=1)
        offs_n = tl.arange(0, N)
        tl.store(out_ptr + pid * N + offs_n, tl.reshape(y, [N]))
    return canonical_kernel


@gluon.jit
def single_kernel(in_desc, out_ptr, register_layout: gl.constexpr,
                  shared_layout: gl.constexpr, B_DESC: gl.constexpr,
                  M: gl.constexpr, N: gl.constexpr):
    pid = gl.program_id(0)
    smem = gl.allocate_shared_memory(gl.bfloat16, [1, M, N], shared_layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, in_desc.block_type.nbytes)
    tma.async_load(in_desc, [pid, 0, 0], bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    x = smem.load(register_layout)
    x_fp32 = x.to(gl.float32)
    y = gl.max(x_fp32, axis=1)
    offs_0 = gl.arange(0, 1, layout=gl.SliceLayout(1, gl.SliceLayout(1, register_layout)))[:, None]
    offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, gl.SliceLayout(1, register_layout)))[None, :]
    gl.store(out_ptr + pid * N + offs_0 * N + offs_n, y)


@gluon.jit(do_not_specialize=["num_reductions"])
def repeated_kernel(in_desc, out_ptr, num_reductions: gl.int32,
                    register_layout: gl.constexpr, shared_layout: gl.constexpr,
                    B_DESC: gl.constexpr, M: gl.constexpr, N: gl.constexpr,
                    X_CONSTRAINTS: gl.constexpr, R_CONSTRAINTS: gl.constexpr):
    pid = gl.program_id(0)
    smem = gl.allocate_shared_memory(gl.bfloat16, [1, M, N], shared_layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, in_desc.block_type.nbytes)
    tma.async_load(in_desc, [pid, 0, 0], bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    x = smem.load(register_layout)
    for _ in range(0, num_reductions):
        x_iter = gl.inline_asm("", X_CONSTRAINTS, [x], x.type, is_pure=False)
        x_fp32 = x_iter.to(gl.float32)
        y = gl.max(x_fp32, axis=1)
        gl.inline_asm("", R_CONSTRAINTS, [y], (), is_pure=False)
    gl.store(out_ptr + pid, gl.to_tensor(0.0))
