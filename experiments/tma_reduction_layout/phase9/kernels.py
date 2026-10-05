"""Valid block programs for architecture reproduction and load/store controls."""
import triton
import triton.language as tl
from experiments.tma_reduction_layout.phase4.kernels_stage_b import make_canonical


def make_kernel(harness):
    if harness == "reduction":
        return make_canonical()
    if harness == "load_copy":
        @triton.jit
        def load_copy_kernel(a_ptr, out_ptr, stride_b, stride_m,
                             B_DESC: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
            pid = tl.program_id(0)
            desc = tl.make_tensor_descriptor(a_ptr, shape=[B_DESC, M, N],
                strides=[stride_b, stride_m, 1], block_shape=[1, M, N])
            x = desc.load([pid, 0, 0])
            mi = tl.arange(0, M)[None, :, None]
            ni = tl.arange(0, N)[None, None, :]
            tl.store(out_ptr + pid*stride_b + mi*stride_m + ni, x)
        return load_copy_kernel
    if harness == "store_copy":
        @triton.jit
        def store_copy_kernel(a_ptr, out_ptr, stride_b, stride_m,
                              B_DESC: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
            pid = tl.program_id(0)
            mi = tl.arange(0, M)[None, :, None]
            ni = tl.arange(0, N)[None, None, :]
            x = tl.load(a_ptr + pid*stride_b + mi*stride_m + ni)
            desc = tl.make_tensor_descriptor(out_ptr, shape=[B_DESC, M, N],
                strides=[stride_b, stride_m, 1], block_shape=[1, M, N])
            desc.store([pid, 0, 0], x)
        return store_copy_kernel
    raise ValueError(harness)
