#!/usr/bin/env python3
"""
Gluon Structural Reproduction Kernel Definitions for TMA Reduction Layouts.

Defines:
- Canonical distributed layouts using gl.BlockedLayout.
- Canonical NVMMA shared layout using gl.NVMMASharedLayout.
- TMA load via mbarrier and async_load.
- Explicit shared -> register load using smem.load(register_layout).
- Reduction along reduction axis (axis=1, M=32).
- Minimal observable output store.
"""

from typing import Dict, Any
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
)


def get_canonical_gluon_layouts() -> Dict[str, Dict[str, Any]]:
    """Return the exact pre-registered canonical Gluon layouts for all 4 cases."""
    return {
        "M32_N64_w8": {
            "default": gl.BlockedLayout(
                size_per_thread=[1, 1, 8],
                threads_per_warp=[1, 4, 8],
                warps_per_cta=[1, 8, 1],
                order=[2, 1, 0],
            ),
            "4": gl.BlockedLayout(
                size_per_thread=[1, 1, 4],
                threads_per_warp=[1, 2, 16],
                warps_per_cta=[1, 8, 1],
                order=[2, 1, 0],
            ),
        },
        "M32_N128_w4": {
            "default": gl.BlockedLayout(
                size_per_thread=[1, 1, 8],
                threads_per_warp=[1, 2, 16],
                warps_per_cta=[1, 4, 1],
                order=[2, 1, 0],
            ),
            "4": gl.BlockedLayout(
                size_per_thread=[1, 1, 4],
                threads_per_warp=[1, 1, 32],
                warps_per_cta=[1, 4, 1],
                order=[2, 1, 0],
            ),
        },
    }


def get_canonical_gluon_shared_layout() -> gl.NVMMASharedLayout:
    """Return the canonical NVMMA shared layout."""
    return gl.NVMMASharedLayout(
        swizzle_byte_width=128,
        element_bitwidth=16,
        rank=3,
        transposed=False,
    )


@gluon.jit
def gluon_canonical_reduction_kernel(
    in_desc, out_ptr,
    register_layout: gl.constexpr,
    shared_layout: gl.constexpr,
    B_DESC: gl.constexpr, M: gl.constexpr, N: gl.constexpr
):
    pid = gl.program_id(0)

    # 1. Allocate shared memory for TMA input tile and mbarrier
    smem = gl.allocate_shared_memory(
        gl.bfloat16,
        [1, M, N],
        shared_layout,
    )
    bar = gl.allocate_shared_memory(
        gl.int64,
        [1],
        mbarrier.MBarrierLayout(),
    )

    # 2. TMA asynchronous load from global memory to shared memory
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, in_desc.block_type.nbytes)
    tma.async_load(
        in_desc,
        [pid, 0, 0],
        bar,
        smem,
    )
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    # 3. Explicit shared -> register load directly into requested distributed layout
    x = smem.load(register_layout)

    # 4. Canonical reduction: convert to float32, then max along axis 1 (M=32)
    x_f32 = x.to(gl.float32)
    r = gl.max(x_f32, axis=1)

    # 5. Minimal observable output store (excluded from reduction-core comparison)
    # r has shape [1, N] and layout gl.SliceLayout(1, register_layout)
    offs_0 = gl.arange(0, 1, layout=gl.SliceLayout(1, gl.SliceLayout(1, register_layout)))[:, None]
    offs_n = gl.arange(0, N, layout=gl.SliceLayout(0, gl.SliceLayout(1, register_layout)))[None, :]
    gl.store(out_ptr + pid * N + offs_0 * N + offs_n, r)
