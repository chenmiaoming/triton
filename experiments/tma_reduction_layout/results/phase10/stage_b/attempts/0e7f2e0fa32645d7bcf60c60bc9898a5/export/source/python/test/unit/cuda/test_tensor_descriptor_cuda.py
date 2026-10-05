import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import requires_tma
from triton.tools.tensor_descriptor import TensorDescriptor


@requires_tma
def test_specialization_after_host_tensordesc():

    @triton.jit
    def kernel(a, b):
        pass

    device = "cuda"
    A = torch.randn(1024, device=device)
    desc = TensorDescriptor.from_tensor(A, [128])
    h = kernel.warmup(desc, 16, grid=(1, ))
    assert "%a: !tt.tensordesc<128xf32>" in h.asm["ttir"]
    assert "%b: i32 {tt.divisibility = 16 : i32}" in h.asm["ttir"]


@requires_tma
@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("block_n,num_warps", [(64, 8), (128, 4), (128, 8)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("op", ["max", "min", "sum"])
def test_tensor_descriptor_partial_reduction_layout(host, block_n, num_warps, dtype, op, with_allocator):

    @triton.jit
    def kernel(arg, out, BN: tl.constexpr, HOST: tl.constexpr, OP: tl.constexpr):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        if HOST:
            desc = arg
        else:
            desc = tl.make_tensor_descriptor(arg, shape=[1, 128, 4 * BN],
                                             strides=[128 * 4 * BN, 4 * BN, 1], block_shape=[1, 32, BN])
        x = desc.load([0, pm * 32, pn * BN]).to(tl.float32)
        if OP == "max":
            y = tl.max(x, axis=1)
        elif OP == "min":
            y = tl.min(x, axis=1)
        else:
            y = tl.sum(x, axis=1)
        tl.store(out + pm * (4 * BN) + pn * BN + tl.arange(0, BN), tl.reshape(y, [BN]))

    # Small multiples of 1/4 are exact in every input dtype and in f32 sums,
    # making the numerical oracle independent of the reduction tree.
    values = (torch.arange(128 * 4 * block_n, device="cuda") % 97 - 48).float() / 4
    x = values.to(dtype).reshape(1, 128, 4 * block_n)
    out = torch.empty((4, 4 * block_n), dtype=torch.float32, device="cuda")
    arg = TensorDescriptor.from_tensor(x, [1, 32, block_n]) if host else x
    kernel[(4, 4)](arg, out, block_n, host, op, num_warps=num_warps)
    tiles = x.float().reshape(4, 32, 4 * block_n)
    reference = {"max": torch.amax, "min": torch.amin, "sum": torch.sum}[op](tiles, dim=1)
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
