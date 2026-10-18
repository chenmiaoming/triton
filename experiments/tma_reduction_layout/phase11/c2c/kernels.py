"""Valid block programs for composition and fixed-R matched local contrasts."""
from triton.experimental import gluon
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier, tma
from experiments.tma_reduction_layout.phase11.c2.kernels import advance


@gluon.jit
def one_pass(desc, out, BM: gl.constexpr, BN: gl.constexpr, W: gl.constexpr,
             REDUCE: gl.constexpr, L: gl.constexpr, S: gl.constexpr):
    pid = gl.program_id(0)
    pm, pn = pid//(8192//BN), pid % (8192//BN)
    smem = gl.allocate_shared_memory(gl.bfloat16, [1, BM, BN], S)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_load(desc, [0, pm*BM, pn*BN], bar, smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    if REDUCE:
        y = gl.max(smem.load(L).to(gl.float32), axis=1)
    else:
        row_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 1, 32], [1, 1, W], [2, 1, 0])
        indices = gl.full([1, 1, BN], 0, gl.int32, row_layout)
        y = smem.gather(indices, axis=1).to(gl.float32)
    output_layout: gl.constexpr = gl.BlockedLayout([1], [32], [W], [0])
    values = gl.convert_layout(gl.reshape(y, [BN]), output_layout)
    cols = gl.arange(0, BN, layout=output_layout)
    gl.store(out+pid*BN+cols, values)


@gluon.jit(do_not_specialize=['K'])
def streaming(desc, out, K, BM: gl.constexpr, BN: gl.constexpr,
              W: gl.constexpr, REDUCE: gl.constexpr,
              L: gl.constexpr, S: gl.constexpr):
    pid = gl.program_id(0)
    pm, pn = pid // (8192 // BN), pid % (8192 // BN)
    smem = gl.allocate_shared_memory(gl.bfloat16, [1, BM, BN], S)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    row_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 1, 32], [1, 1, W], [2, 1, 0])
    indices = gl.full([1, 1, BN], 0, gl.int32, row_layout)
    output_layout: gl.constexpr = gl.BlockedLayout([1], [32], [W], [0])
    cols = gl.arange(0, BN, layout=output_layout)
    for i in range(K):
        # Fresh origins per iteration; at a full grid every iteration covers
        # the original matrix. No lane/warp-varying source scalars.
        row = ((pm + i) % (8192 // BM)) * BM
        mbarrier.expect(bar, desc.block_type.nbytes)
        tma.async_load(desc, [0, row, pn * BN], bar, smem)
        mbarrier.wait(bar, phase=i & 1)
        if REDUCE:
            x = smem.load(L)
            y = gl.max(x.to(gl.float32), axis=1)
        else:
            y = smem.gather(indices, axis=1).to(gl.float32)
        result = gl.convert_layout(gl.reshape(y, [BN]), output_layout)
        gl.store(out + (pid*K+i)*BN + cols, result)
        if i + 1 < K:
            # Required read-before-overwrite protection for a reused shared
            # slot; count it explicitly as repeated-transfer fixture work.
            gl.barrier()
    mbarrier.invalidate(bar)


@gluon.jit(do_not_specialize=['K'])
def resident_row(desc, out, K, BM: gl.constexpr, BN: gl.constexpr,
                 W: gl.constexpr, S: gl.constexpr):
    pid = gl.program_id(0)
    pm, pn = pid // (8192 // BN), pid % (8192 // BN)
    smem = gl.allocate_shared_memory(gl.bfloat16, [2, 1, BM, BN], S)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    if K > 1:
        mbarrier.expect(bar, 2*desc.block_type.nbytes)
    else:
        mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_load(desc, [0, pm*BM, pn*BN], bar, smem.index(0))
    if K > 1:
        tma.async_load(desc, [0, ((pm+(8192//BM)//2) % (8192//BM))*BM, pn*BN], bar, smem.index(1))
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    row_layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 1, 32], [1, 1, W], [2, 1, 0])
    indices = gl.full([1, 1, BN], 0, gl.int32, row_layout)
    output_layout: gl.constexpr = gl.BlockedLayout([1], [32], [W], [0])
    cols = gl.arange(0, BN, layout=output_layout)
    for i in range(K):
        row = smem.index(i & 1).gather(indices, axis=1)
        result = gl.convert_layout(gl.reshape(row.to(gl.float32), [BN]), output_layout)
        gl.store(out+(pid*K+i)*BN+cols, result)


@gluon.jit(do_not_specialize=['K'])
def resident_mc(desc, out, K, BM: gl.constexpr, BN: gl.constexpr,
                W: gl.constexpr, L: gl.constexpr, S: gl.constexpr):
    pid = gl.program_id(0)
    pm, pn = pid // (8192//BN), pid % (8192//BN)
    smem = gl.allocate_shared_memory(gl.bfloat16, [2, 1, BM, BN], S)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    if K > 1:
        mbarrier.expect(bar, 2*desc.block_type.nbytes)
    else:
        mbarrier.expect(bar, desc.block_type.nbytes)
    tma.async_load(desc, [0, pm*BM, pn*BN], bar, smem.index(0))
    if K > 1:
        tma.async_load(desc, [0, ((pm+(8192//BM)//2) % (8192//BM))*BM, pn*BN], bar, smem.index(1))
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    output_layout: gl.constexpr = gl.BlockedLayout([1], [32], [W], [0])
    cols = gl.arange(0, BN, layout=output_layout)
    for i in range(K):
        raw = smem.index(i & 1).load(L)
        y = gl.max(raw.to(gl.float32), axis=1)
        result = gl.convert_layout(gl.reshape(y, [BN]), output_layout)
        gl.store(out+(pid*K+i)*BN+cols, result)


@gluon.jit(do_not_specialize=['K'])
def row_scaffold(arg, out, K, BM: gl.constexpr, BN: gl.constexpr, W: gl.constexpr):
    pid = gl.program_id(0)
    pm, pn = pid // (8192//BN), pid % (8192//BN)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [W], [0])
    cols = gl.arange(0, BN, layout=layout)
    for i in range(K):
        row = ((pm+i) % (8192//BM))*BM
        x = gl.load(arg+row*8192+pn*BN+cols)
        gl.store(out+(pid*K+i)*BN+cols, x.to(gl.float32))


@g.jit(do_not_specialize=['K'])
def fixed_local(arg, out, K, R: gl.constexpr, P: gl.constexpr,
                C: gl.constexpr, VARIANT: gl.constexpr, L: gl.constexpr):
    # Explicit logical thread axis is a tensor dimension, not a hardware ID.
    # All source scalars remain block-uniform; each owner has exactly R values.
    pid = gl.program_id(0)
    owner = gl.arange(0, 128, layout=gl.SliceLayout(1, gl.SliceLayout(2, L)))
    rows = gl.arange(0, P, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    cols = gl.arange(0, C, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    offsets = owner[:, None, None]*R + rows[None, :, None]*C + cols[None, None, :]
    x = gl.load(arg+(pid % 1024)*128*R+offsets)
    step = ((((offsets*13) % 7)+1).to(gl.float32)*.25).to(gl.bfloat16)
    reduced_layout: gl.constexpr = gl.SliceLayout(1, L)
    consumer_layout: gl.constexpr = gl.SliceLayout(1, reduced_layout)
    checksum = gl.full([128], 0, gl.float32, consumer_layout)
    index_layout: gl.constexpr = L
    indices = gl.full([128, 1, C], 0, gl.int32, index_layout)
    for i in range(K):
        x = advance(x, step, L)
        if VARIANT == 'native':
            # Single-use conversion directly into max permits BF16 lowering.
            partial = gl.max(x.to(gl.float32), axis=1)
        elif VARIANT == 'wide':
            # Exact power-of-two transform preserves results on bounded BF16
            # inputs but blocks convert->max recognition. Its extra operations
            # are measured by the corresponding fixture_wide control.
            partial = gl.max(x.to(gl.float32)*2., axis=1)*.5
        elif VARIANT == 'fixture':
            partial = gl.reshape(gl.gather(x, indices, axis=1).to(gl.float32), [128, C])
        else:
            partial = gl.reshape(gl.gather(x, indices, axis=1).to(gl.float32)*2., [128, C])*.5
        # Identical consumer and observable size for all four variants at a
        # given (R,p,columns). No cross-owner reduction or reverse broadcast.
        partial = gl.convert_layout(partial, reduced_layout)
        checksum = checksum + gl.sum(partial, axis=1)
    checksum = gl.convert_layout(checksum, gl.SliceLayout(1, gl.SliceLayout(2, L)))
    gl.store(out+pid*128*(R+1)+owner, checksum)
    # Observe every final state so fixture variants cannot delete updates of
    # rows not selected by gather. This store is outside the repeated slope.
    gl.store(out+pid*128*(R+1)+128+offsets, x.to(gl.float32))
