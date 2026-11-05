# SPDX-License-Identifier: Apache-2.0
"""Exact target/helper bodies extracted from the pinned SGLang source.

Only the standalone pack pass is imported; no amax/scales pass is included.
"""
import triton
import triton.language as tl

BLOCK_TOKENS = 64

@triton.jit
def _divide_rn(numerator, denominator):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [numerator, denominator],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )

@triton.jit
def _load_tile(
    base,
    head,
    tokens,
    columns,
    valid,
    head_stride: tl.constexpr,
    token_stride: tl.constexpr,
):
    offsets = head * head_stride + tokens[:, None] * token_stride + columns[None, :]
    return tl.load(base + offsets, mask=valid, other=0.0).to(tl.float32)

@triton.jit
def _quantize(values, scale):
    scaled = _divide_rn(values, scale)
    scaled = tl.maximum(tl.minimum(scaled, 448.0), -448.0)
    return scaled.to(tl.float8e4nv)

@triton.jit
def _pack_qkv(
    q,
    k,
    v,
    q_fp8,
    k_fp8,
    v_fp8,
    scales,
    sequence,
    head_stride: tl.constexpr,
    token_stride: tl.constexpr,
    padded_queries: tl.constexpr,
    padded_keys: tl.constexpr,
    heads: tl.constexpr,
    block_tokens: tl.constexpr,
):
    head = tl.program_id(0)
    tokens = tl.program_id(1) * block_tokens + tl.arange(0, block_tokens)
    columns = tl.arange(0, 128)
    valid = tokens[:, None] < sequence

    q_scale = tl.load(scales + 0 * heads + head)
    k_scale = tl.load(scales + 1 * heads + head)
    v_scale = tl.load(scales + 2 * heads + head)

    # Q is padded to the 128-row query tile, K/V to the 32-key tile.
    hsd_offsets = head * padded_queries * 128 + tokens[:, None] * 128 + columns[None, :]
    k_offsets = head * padded_keys * 128 + tokens[:, None] * 128 + columns[None, :]
    in_queries = tokens[:, None] < padded_queries
    in_keys = tokens[:, None] < padded_keys

    q_tile = _load_tile(q, head, tokens, columns, valid, head_stride, token_stride)
    tl.store(q_fp8 + hsd_offsets, _quantize(q_tile, q_scale), mask=in_queries)

    k_tile = _load_tile(k, head, tokens, columns, valid, head_stride, token_stride)
    tl.store(k_fp8 + k_offsets, _quantize(k_tile, k_scale), mask=in_keys)

    # The kernel packs P straight from the score fragment; V^T's stored key
    # order must follow it: stored position p holds key
    # p//16*16 + 8*((p%4)//2) + 2*((p%16)//4) + p%2 (see key_order_for_positions).
    # The permutation is applied on the LOAD side as a row gather: every gathered
    # row is still 256 contiguous bytes, and the transposed store stays contiguous.
    # Permuting the store instead scatters bytes inside 16 B windows and doubled
    # the prepare time.
    source_tokens = (
        tokens // 16 * 16
        + 8 * ((tokens % 4) // 2)
        + 2 * ((tokens % 16) // 4)
        + tokens % 2
    )
    valid_source = source_tokens[:, None] < sequence

    hds_offsets = (
        head * 128 * padded_keys + columns[:, None] * padded_keys + tokens[None, :]
    )
    in_keys_transposed = tokens[None, :] < padded_keys

    v_tile = _load_tile(
        v, head, source_tokens, columns, valid_source, head_stride, token_stride
    )
    tl.store(
        v_fp8 + hds_offsets,
        tl.trans(_quantize(v_tile, v_scale)),
        mask=in_keys_transposed,
    )
