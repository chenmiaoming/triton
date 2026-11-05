"""Explicit-layout Gluon and canonical Triton kernels for identical workloads."""
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier


@triton.jit
def canonical(arg, out, BM: tl.constexpr, BN: tl.constexpr, HOST: tl.constexpr):
    pid = tl.program_id(0)
    pm = pid // (8192 // BN)
    pn = pid % (8192 // BN)
    if HOST:
        desc = arg
    else:
        desc = tl.make_tensor_descriptor(arg, [1,8192,8192], [8192*8192,8192,1], [1,BM,BN])
    raw = desc.load([0,pm*BM,pn*BN])
    x = raw.to(tl.float32)
    y = tl.max(x, axis=1)
    offsets = pm*8192 + pn*BN + tl.arange(0,BN)
    tl.store(out+offsets, tl.reshape(y,[BN]))


@gluon.jit
def explicit(arg, out, BM: gl.constexpr, BN: gl.constexpr, W: gl.constexpr,
             HOST: gl.constexpr, DTYPE: gl.constexpr,
             register_layout: gl.constexpr, shared_layout: gl.constexpr):
    pid = gl.program_id(0)
    pm = pid // (8192 // BN)
    pn = pid % (8192 // BN)
    if HOST:
        desc = arg
    else:
        desc = tma.make_tensor_descriptor(arg, [1,8192,8192], [8192*8192,8192,1],
                                          [1,BM,BN], shared_layout)
    smem = gl.allocate_shared_memory(DTYPE, [1,BM,BN], shared_layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_load(desc, [0,pm*BM,pn*BN], bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    raw = smem.load(register_layout)
    x = raw.to(gl.float32)
    y = gl.max(x, axis=1)
    output_layout: gl.constexpr = gl.BlockedLayout([1],[32],[W],[0])
    values = gl.convert_layout(gl.reshape(y,[BN]), output_layout)
    offsets = pm*8192 + pn*BN + gl.arange(0,BN,layout=output_layout)
    gl.store(out+offsets, values)
