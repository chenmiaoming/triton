# SPDX-License-Identifier: MIT
"""Same loop, two FP32 accumulators and two outputs as the pinned tutorial merge."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def merge(DW,DB,FINAL_DW,FINAL_DB,M,N,BLOCK_SIZE_M:gl.constexpr,BLOCK_SIZE_N:gl.constexpr,
          L:gl.constexpr,LOUT:gl.constexpr,NATIVE_OUT:gl.constexpr=False):
    pid=gl.program_id(0)
    cols=pid*BLOCK_SIZE_N+gl.arange(0,BLOCK_SIZE_N,layout=gl.SliceLayout(0,L))
    dw=gl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=gl.float32,layout=L)
    db=gl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=gl.float32,layout=L)
    for i in range(0,M,BLOCK_SIZE_M):
        rows=i+gl.arange(0,BLOCK_SIZE_M,layout=gl.SliceLayout(1,L))
        mask=(rows[:,None]<M)&(cols[None,:]<N)
        offs=rows[:,None]*N+cols[None,:]
        dw+=gl.load(DW+offs,mask=mask,other=0.)
        db+=gl.load(DB+offs,mask=mask,other=0.)
    sum_dw=gl.sum(dw,axis=0)
    sum_db=gl.sum(db,axis=0)
    if NATIVE_OUT:
        gl.store(FINAL_DW+cols,sum_dw,mask=cols<N)
        gl.store(FINAL_DB+cols,sum_db,mask=cols<N)
    else:
        gl.store(gl.convert_layout(FINAL_DW+cols,LOUT),gl.convert_layout(sum_dw,LOUT),
                 mask=gl.convert_layout(cols<N,LOUT))
        gl.store(gl.convert_layout(FINAL_DB+cols,LOUT),gl.convert_layout(sum_db,LOUT),
                 mask=gl.convert_layout(cols<N,LOUT))
