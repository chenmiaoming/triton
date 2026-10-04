"""Valid block programs: two descriptor paths by two output layouts."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier


@gluon.jit
def host_kernel(in_desc, out_ptr, register_layout: gl.constexpr, shared_layout: gl.constexpr,
                B_DESC: gl.constexpr, M: gl.constexpr, N: gl.constexpr,
                W: gl.constexpr, CANONICAL_STORE: gl.constexpr):
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
    if CANONICAL_STORE:
        store_layout: gl.constexpr = gl.BlockedLayout([1], [32], [W], [0])
        values = gl.convert_layout(gl.reshape(y, [N]), store_layout)
        offs_n = gl.arange(0, N, layout=store_layout)
        gl.store(out_ptr + pid * N + offs_n, values)
    else:
        offs_0 = gl.arange(0, 1, layout=gl.SliceLayout(1, gl.SliceLayout(1, register_layout)))[:, None]
        offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, gl.SliceLayout(1, register_layout)))[None, :]
        gl.store(out_ptr + pid * N + offs_0 * N + offs_n, y)


@gluon.jit
def device_kernel(a_ptr, out_ptr, stride_b, stride_m,
                  register_layout: gl.constexpr, shared_layout: gl.constexpr,
                  B_DESC: gl.constexpr, M: gl.constexpr, N: gl.constexpr,
                  W: gl.constexpr, CANONICAL_STORE: gl.constexpr):
    pid = gl.program_id(0)
    desc = tma.make_tensor_descriptor(a_ptr, [B_DESC, M, N], [stride_b, stride_m, 1],
                                      [1, M, N], shared_layout)
    smem = gl.allocate_shared_memory(gl.bfloat16, [1, M, N], shared_layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_load(desc, [pid, 0, 0], bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    x = smem.load(register_layout)
    x_fp32 = x.to(gl.float32)
    y = gl.max(x_fp32, axis=1)
    if CANONICAL_STORE:
        store_layout: gl.constexpr = gl.BlockedLayout([1], [32], [W], [0])
        values = gl.convert_layout(gl.reshape(y, [N]), store_layout)
        offs_n = gl.arange(0, N, layout=store_layout)
        gl.store(out_ptr + pid * N + offs_n, values)
    else:
        offs_0 = gl.arange(0, 1, layout=gl.SliceLayout(1, gl.SliceLayout(1, register_layout)))[:, None]
        offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, gl.SliceLayout(1, register_layout)))[None, :]
        gl.store(out_ptr + pid * N + offs_0 * N + offs_n, y)
