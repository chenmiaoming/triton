"""Dense diagnostic and genuine centered consumers as valid Gluon block programs."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma


@gluon.jit
def dense_output(desc, out, BM: gl.constexpr, BN: gl.constexpr, W: gl.constexpr,
                 L: gl.constexpr, S: gl.constexpr, OUTPUT: gl.constexpr):
    pid = gl.program_id(0)
    pm, pn = pid//(8192//BN), pid % (8192//BN)
    smem = gl.allocate_shared_memory(gl.bfloat16, [1,BM,BN], S)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar,count=1)
    mbarrier.expect(bar,desc.block_type.nbytes)
    tma.async_load(desc,[0,pm*BM,pn*BN],bar,smem)
    mbarrier.wait(bar,phase=0)
    mbarrier.invalidate(bar)
    raw = smem.load(L)
    wide = raw.to(gl.float32)
    if OUTPUT == 'copy':
        values = wide
    else:
        reduced = gl.max(wide,axis=1)
        if OUTPUT == 'centered':
            # The same widened input has two real consumers. Do not replace
            # this with BF16 max just to retain the old packed-max pattern.
            values = wide-reduced[:,None,:]
        else:
            row_layout: gl.constexpr = gl.SliceLayout(0,gl.SliceLayout(2,L))
            rows = gl.arange(0,BM,layout=row_layout)
            values = reduced[:,None,:]+rows[None,:,None].to(gl.float32)
    row_layout: gl.constexpr = gl.SliceLayout(0,gl.SliceLayout(2,L))
    col_layout: gl.constexpr = gl.SliceLayout(0,gl.SliceLayout(1,L))
    rows = gl.arange(0,BM,layout=row_layout)
    cols = gl.arange(0,BN,layout=col_layout)
    offsets = rows[None,:,None]*BN+cols[None,None,:]
    gl.store(out+pid*(BM*BN)+offsets,values)
