"""Loop-free, block-valid FULL output-policy interventions."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma


@gluon.jit
def full_output(desc, out, BM: gl.constexpr, BN: gl.constexpr, W: gl.constexpr,
                L: gl.constexpr, S: gl.constexpr, OUTPUT: gl.constexpr):
    pid = gl.program_id(0)
    pm = pid//(8192//BN)
    pn = pid % (8192//BN)
    smem = gl.allocate_shared_memory(gl.bfloat16, [1, BM, BN], S)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_load(desc, [0, pm*BM, pn*BN], bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    raw = smem.load(L)
    wide = raw.to(gl.float32)
    reduced = gl.max(wide, axis=1)
    flat = gl.reshape(reduced, [BN])
    if OUTPUT == 'fixed':
        output_layout: gl.constexpr = gl.BlockedLayout([1], [32], [W], [0])
        values = gl.convert_layout(flat, output_layout)
        cols = gl.arange(0, BN, layout=output_layout)
        gl.store(out+pid*BN+cols, values)
    else:
        # Two logical unit/reduced dimensions are removed. No hardware ID or
        # lane-varying source scalar; redundant stores are masked by lowering.
        native_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, L))
        cols = gl.arange(0, BN, layout=native_layout)
        gl.store(out+pid*BN+cols, flat)
