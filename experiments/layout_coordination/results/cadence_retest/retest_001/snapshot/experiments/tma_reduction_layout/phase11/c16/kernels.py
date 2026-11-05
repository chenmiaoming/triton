"""Two transfer controls: original TMA tile and output size, first-row output."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma


@gluon.jit
def transfer(desc, out, BM: gl.constexpr, BN: gl.constexpr, W: gl.constexpr,
             shared_layout: gl.constexpr):
    pid = gl.program_id(0)
    pm, pn = pid // (8192 // BN), pid % (8192 // BN)
    smem = gl.allocate_shared_memory(gl.bfloat16, [1, BM, BN], shared_layout)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_load(desc, [0, pm * BM, pn * BN], bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    # Gather row zero from the original descriptor, without splitting swizzle.
    row_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 1, 32], [1, 1, W], [2, 1, 0])
    indices = gl.full([1, 1, BN], 0, gl.int32, row_layout)
    row = smem.gather(indices, axis=1)
    output_layout: gl.constexpr = gl.BlockedLayout([1], [32], [W], [0])
    offsets = pid * BN + gl.arange(0, BN, layout=output_layout)
    values = gl.convert_layout(gl.reshape(row.to(gl.float32), [BN]), output_layout)
    gl.store(out + offsets, values)
