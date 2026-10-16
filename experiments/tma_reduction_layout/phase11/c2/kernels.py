"""Legal Gluon calibration subgraphs; no assembly or hardware-ID scalars."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import mbarrier,tma


@gluon.jit
def advance(state,step,L:gl.constexpr):
    threshold=gl.full(state.shape,8,gl.bfloat16,L)
    wrap=gl.full(state.shape,16,gl.bfloat16,L)
    updated=gl.where(state>=threshold,state-wrap,state+step)
    gl.static_assert(updated.dtype==gl.bfloat16)
    return updated


@gluon.jit(do_not_specialize=['K'])
def micro(desc,out,K,BM:gl.constexpr,BN:gl.constexpr,W:gl.constexpr,
          FAMILY:gl.constexpr,D:gl.constexpr,L:gl.constexpr,S:gl.constexpr):
    pid=gl.program_id(0)
    pm,pn=pid//(8192//BN),pid%(8192//BN)
    bar=gl.allocate_shared_memory(gl.int64,[1],mbarrier.MBarrierLayout())
    mbarrier.init(bar,count=1)
    matrix:gl.constexpr=gl.SliceLayout(0,L)
    rows=gl.arange(0,BM,layout=gl.SliceLayout(1,matrix))
    cols=gl.arange(0,BN,layout=gl.SliceLayout(0,matrix))
    if FAMILY=='ML' or FAMILY=='MC':
        smem=gl.allocate_shared_memory(gl.bfloat16,[2,1,BM,BN],S)
        if FAMILY=='MC':
            if K>1:
                mbarrier.expect(bar,2*desc.block_type.nbytes)
            else:
                mbarrier.expect(bar,desc.block_type.nbytes)
        else:
            mbarrier.expect(bar,2*desc.block_type.nbytes)
        tma.async_load(desc,[0,pm*BM,pn*BN],bar,smem.index(0))
        if FAMILY=='ML':
            tma.async_load(desc,[0,((pm+128)%256)*BM,pn*BN],bar,smem.index(1))
        else:
            if K>1:
                tma.async_load(desc,[0,((pm+128)%256)*BM,pn*BN],bar,smem.index(1))
        mbarrier.wait(bar,phase=0)
        mbarrier.invalidate(bar)
        if FAMILY=='ML':
            acc0=gl.full([1,BM,BN],0,gl.float32,L)
            if D==2:
                acc1=gl.full([1,BM,BN],0,gl.float32,L)
            for i in range(K):
                raw0=smem.index(i&1).load(L)
                acc0=acc0+raw0.to(gl.float32)
                if D==2:
                    raw1=smem.index((i+1)&1).load(L)
                    acc1=acc1+raw1.to(gl.float32)
            offset=pid*D*BM*BN+rows[:,None]*BN+cols[None,:]
            gl.store(out+gl.expand_dims(offset,0),acc0)
            if D==2:
                gl.store(out+gl.expand_dims(offset+BM*BN,0),acc1)
        else:
            output_layout:gl.constexpr=gl.BlockedLayout([1],[32],[W],[0])
            outcols=gl.arange(0,BN,layout=output_layout)
            for i in range(K):
                raw=smem.index(i&1).load(L)
                y=gl.max(raw.to(gl.float32),axis=1)
                values=gl.convert_layout(gl.reshape(y,[BN]),output_layout)
                # Distinct observable output per iteration, no checksum or
                # reduction-result feedback. K1 is the original computation.
                gl.store(out+(pid*K+i)*BN+outcols,values)
    else:
        smem=gl.allocate_shared_memory(gl.bfloat16,[D,1,BM,BN],S)
        mbarrier.expect(bar,D*desc.block_type.nbytes)
        tma.async_load(desc,[0,pm*BM,pn*BN],bar,smem.index(0))
        if D==2:
            tma.async_load(desc,[0,((pm+128)%256)*BM,pn*BN],bar,smem.index(1))
        mbarrier.wait(bar,phase=0)
        mbarrier.invalidate(bar)
        state0=smem.index(0).load(L)
        if D==2:
            state1=smem.index(1).load(L)
        if D==1 and BN==64 and L.threads_per_warp[1]==32:
            # Algebraically identical step. Avoid the observed undefined PTX
            # half-register in the packed integer construction of this fixture.
            row_step=((((rows*13)%7)+1).to(gl.float32)*.25).to(gl.bfloat16)
            step=gl.expand_dims(row_step[:,None]+gl.full([BM,BN],0,gl.bfloat16,matrix),0)
        else:
            step=gl.expand_dims(((((rows[:,None]*13+cols[None,:]*7)%7)+1).to(gl.float32)*.25).to(gl.bfloat16),0)
        result:gl.constexpr=gl.SliceLayout(1,L)
        acc0=gl.full([1,BN],0,gl.float32,result)
        if D==2:
            acc1=gl.full([1,BN],0,gl.float32,result)
        for i in range(K):
            state0=advance(state0,step,L)
            wide0=state0.to(gl.float32)
            y0=gl.max(wide0,axis=1)
            acc0=acc0+y0
            if D==2:
                state1=advance(state1,step,L)
                wide1=state1.to(gl.float32)
                y1=gl.max(wide1,axis=1)
                acc1=acc1+y1
        outcols=gl.arange(0,BN,layout=gl.SliceLayout(0,result))
        gl.store(out+pid*D*BN+outcols[None,:],acc0)
        if D==2:
            gl.store(out+pid*D*BN+BN+outcols[None,:],acc1)


@gluon.jit
def output_path(arg,out,BN:gl.constexpr,W:gl.constexpr,L:gl.constexpr):
    pid=gl.program_id(0)
    result:gl.constexpr=gl.SliceLayout(1,L)
    native_cols=gl.arange(0,BN,layout=gl.SliceLayout(0,result))
    value=gl.load(arg+pid*BN+native_cols[None,:])
    output_layout:gl.constexpr=gl.BlockedLayout([1],[32],[W],[0])
    values=gl.convert_layout(gl.reshape(value,[BN]),output_layout)
    cols=gl.arange(0,BN,layout=output_layout)
    gl.store(out+pid*BN+cols,values)


@gluon.jit(do_not_specialize=['K'])
def parameter(arg,out,K,BM:gl.constexpr,BN:gl.constexpr,W:gl.constexpr,
              FAMILY:gl.constexpr,D:gl.constexpr,L:gl.constexpr):
    pid=gl.program_id(0)
    full:gl.constexpr=(8192//BM)*(8192//BN)
    source_pid=pid%full
    pm,pn=source_pid//(8192//BN),source_pid%(8192//BN)
    matrix:gl.constexpr=gl.SliceLayout(0,L)
    rows=gl.arange(0,BM,layout=gl.SliceLayout(1,matrix))
    cols=gl.arange(0,BN,layout=gl.SliceLayout(0,matrix))
    offset=pm*BM*8192+pn*BN+rows[:,None]*8192+cols[None,:]
    state0=gl.load(arg+gl.expand_dims(offset,0))
    if D==2:
        pm1=(pm+(8192//BM)//2)%(8192//BM)
        offset1=pm1*BM*8192+pn*BN+rows[:,None]*8192+cols[None,:]
        state1=gl.load(arg+gl.expand_dims(offset1,0))
    # Same mathematical update as MR; factor the identically zero (7*n)%7
    # term before lowering rather than exposing packed integer construction.
    row_step=((((rows*13)%7)+1).to(gl.float32)*.25).to(gl.bfloat16)
    step=gl.expand_dims(row_step[:,None]+gl.full([BM,BN],0,gl.bfloat16,matrix),0)
    if FAMILY=='E':
        result:gl.constexpr=L
        shape:gl.constexpr=[1,BM,BN]
    else:
        result:gl.constexpr=gl.SliceLayout(1,L)
        shape:gl.constexpr=[1,BN]
    acc0=gl.full(shape,0,gl.float32,result)
    if D==2:
        acc1=gl.full(shape,0,gl.float32,result)
    for i in range(K):
        state0=advance(state0,step,L)
        if FAMILY=='E':
            acc0=acc0+state0.to(gl.float32)
        else:
            acc0=acc0+gl.max(state0.to(gl.float32),axis=1)
        if D==2:
            state1=advance(state1,step,L)
            if FAMILY=='E':
                acc1=acc1+state1.to(gl.float32)
            else:
                acc1=acc1+gl.max(state1.to(gl.float32),axis=1)
    if FAMILY=='E':
        offsets=pid*D*BM*BN+rows[:,None]*BN+cols[None,:]
        gl.store(out+gl.expand_dims(offsets,0),acc0)
        if D==2:
            gl.store(out+gl.expand_dims(offsets+BM*BN,0),acc1)
    else:
        outcols=gl.arange(0,BN,layout=gl.SliceLayout(0,result))
        gl.store(out+pid*D*BN+outcols[None,:],acc0)
        if D==2:
            gl.store(out+pid*D*BN+BN+outcols[None,:],acc1)
