# SPDX-License-Identifier: MIT
"""LayerNorm dw/db merge with U's intermediate layouts expressed in Gluon.

The source spells out the index/pointer broadcasts hoisted by Triton LICM,
rebuilds output indices in LOUT, and narrows each result before converting it.
It requires only the ordinary Gluon compiler pipeline.
"""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def merge(DW, DB, FINAL_DW, FINAL_DB, M, N,
          BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr,
          L: gl.constexpr, LOUT: gl.constexpr, NATIVE_OUT: gl.constexpr = False):
    pid = gl.program_id(0)
    cols = pid * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, L))
    if not NATIVE_OUT:
        cols_out = pid * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=LOUT)
    dw = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=L)
    db = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=L)

    # These values are loop invariant in U's final TTGIR.  Constructing the
    # scalar splats explicitly prevents dependence on a Gluon LICM pass.
    row_indices = gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, L))
    row_limit = gl.full((BLOCK_SIZE_M, 1), M, M.dtype, layout=L)
    cols_row = cols[None, :]
    col_limit = gl.full((1, BLOCK_SIZE_N), N, N.dtype, layout=L)
    col_mask_row = cols_row < col_limit
    col_mask, _ = gl.broadcast(col_mask_row, dw)
    row_stride = gl.full((BLOCK_SIZE_M, 1), N, N.dtype, layout=L)
    cols_block, _ = gl.broadcast(cols_row, dw)
    dw_base = gl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), DW, DW.dtype, layout=L)
    db_base = gl.full((BLOCK_SIZE_M, BLOCK_SIZE_N), DB, DB.dtype, layout=L)

    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + row_indices
        rows_col = rows[:, None]
        row_mask_col = rows_col < row_limit
        row_mask, _ = gl.broadcast(row_mask_col, dw)
        mask = row_mask & col_mask
        row_offsets_col = rows_col * row_stride
        row_offsets, _ = gl.broadcast(row_offsets_col, dw)
        offsets = row_offsets + cols_block
        dw += gl.load(dw_base + offsets, mask=mask, other=0.)
        db += gl.load(db_base + offsets, mask=mask, other=0.)

    sum_dw = gl.sum(dw, axis=0)
    sum_db = gl.sum(db, axis=0)
    if NATIVE_OUT:
        gl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
        gl.store(FINAL_DB + cols, sum_db, mask=cols < N)
    else:
        # Match U's pointer creation / narrowing / conversion order.
        output_mask = cols_out < N
        output_dw = FINAL_DW + cols_out
        value_dw = gl.convert_layout(sum_dw.to(FINAL_DW.dtype.element_ty), LOUT)
        gl.store(output_dw, value_dw, mask=output_mask)
        output_db = FINAL_DB + cols_out
        value_db = gl.convert_layout(sum_db.to(FINAL_DB.dtype.element_ty), LOUT)
        gl.store(output_db, value_db, mask=output_mask)
