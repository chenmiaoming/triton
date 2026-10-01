#!/usr/bin/env python3
"""
Gluon Repeated-Reduction Kernel Definitions for TMA Reduction Layouts.
Phase 3 Step D.

Defines:
- Canonical distributed layouts using gl.BlockedLayout.
- Canonical NVMMA shared layout using gl.NVMMASharedLayout.
- TMA load via mbarrier and async_load outside the runtime loop.
- Explicit shared -> register load using smem.load(register_layout) outside the runtime loop.
- Runtime loop over R (num_reductions), with do_not_specialize=["num_reductions"].
- Zero-instruction compiler barriers (Prototype X: input opaque barrier, Prototype Y: result sink).
- Minimal observable scalar store outside the runtime loop.
"""

from typing import Dict, Any, Tuple
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
)
from experiments.tma_reduction_layout.gluon.kernel import (
    get_canonical_gluon_layouts,
    get_canonical_gluon_shared_layout,
)


def get_barrier_constraints(cfg_name: str, cand: str) -> Tuple[str, Tuple[str, ...]]:
    """
    Return (input_tied_constraints, result_sink_constraints) for the specialization.
    Uses explicitly numbered ties for input tensor x to guarantee zero machine instructions.
    """
    # Number of elements per thread for x (bfloat16) and r (float32)
    elem_counts = {
        ("M32_N64_w8", "default"): (8, 8),
        ("M32_N64_w8", "4"): (8, 4),
        ("M32_N128_w4", "default"): (32, 8),
        ("M32_N128_w4", "4"): (32, 4),
    }
    num_x, num_r = elem_counts[(cfg_name, cand)]
    # Input barrier: num_x outputs ("=h"), num_x inputs tied to outputs ("0", "1", ...)
    x_constraints = ",".join([f"=h" for _ in range(num_x)] + [f"{i}" for i in range(num_x)])
    # Result sink: consumes r ("f")
    r_constraints = ("f",)
    return x_constraints, r_constraints


@gluon.jit(do_not_specialize=["num_reductions"])
def gluon_repeated_reduction_kernel(
    in_desc, out_ptr,
    num_reductions: gl.int32,  # R, runtime repetition count
    register_layout: gl.constexpr,
    shared_layout: gl.constexpr,
    B_DESC: gl.constexpr, M: gl.constexpr, N: gl.constexpr,
    X_CONSTRAINTS: gl.constexpr,
    R_CONSTRAINTS: gl.constexpr,
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

    # 2. TMA asynchronous load (ONCE outside loop)
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

    # 3. Explicit shared -> register load directly into requested distributed layout (ONCE outside loop)
    x = smem.load(register_layout)

    # 4. Runtime R loop
    for _ in range(0, num_reductions):
        # Prototype X: Input opaque barrier (anti-LICM)
        # Empty assembly with tied operands creates a new SSA value with side-effects but 0 machine instructions
        x_iter = gl.inline_asm("", X_CONSTRAINTS, [x], x.type, is_pure=False)

        # Canonical reduction: convert to float32, then max along axis 1 (M=32)
        x_f32 = x_iter.to(gl.float32)
        r = gl.max(x_f32, axis=1)

        # Prototype Y: Result sink (anti-DCE & anti-sink)
        # Consumes r with side-effects but emits 0 machine instructions
        gl.inline_asm("", R_CONSTRAINTS, [r], (), is_pure=False)

    # 5. Minimal observable scalar store outside the runtime loop (candidate-symmetric, R-independent)
    gl.store(out_ptr + pid, gl.to_tensor(0.0))
