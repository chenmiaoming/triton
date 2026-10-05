"""A runtime block-uniform descriptor choice with a shared reduction/store."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier
from experiments.tma_reduction_layout.phase7.kernels import host_kernel, device_kernel


@gluon.jit(do_not_specialize=["mode"])
def switch_kernel(in_desc, a_ptr, out_ptr, stride_b, stride_m, mode,
                  register_layout: gl.constexpr, shared_layout: gl.constexpr,
                  B_DESC: gl.constexpr, M: gl.constexpr, N: gl.constexpr, W: gl.constexpr):
    pid = gl.program_id(0)
    smem = gl.allocate_shared_memory(gl.bfloat16, [1, M, N], shared_layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, 2 * M * N)
    if mode != 0:
        desc = tma.make_tensor_descriptor(a_ptr, [B_DESC, M, N], [stride_b, stride_m, 1],
                                          [1, M, N], shared_layout)
        tma.async_load(desc, [pid, 0, 0], bar, smem)
    else:
        tma.async_load(in_desc, [pid, 0, 0], bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    x = smem.load(register_layout)
    x_fp32 = x.to(gl.float32)
    y = gl.max(x_fp32, axis=1)
    store_layout: gl.constexpr = gl.BlockedLayout([1], [32], [W], [0])
    values = gl.convert_layout(gl.reshape(y, [N]), store_layout)
    offs_n = gl.arange(0, N, layout=store_layout)
    gl.store(out_ptr + pid * N + offs_n, values)
