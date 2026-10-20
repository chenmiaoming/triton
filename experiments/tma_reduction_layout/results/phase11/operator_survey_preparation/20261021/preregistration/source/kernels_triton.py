"""Canonical semantic fixtures, not replacements for production-library kernels.

Every function has a same-algorithm Gluon counterpart. CPU compilation does not
establish numerical correctness or a bridge to an upstream library binary.
"""
import triton
import triton.language as tl


@triton.jit
def column_reduce(a, b, weights, out, aux, M: tl.constexpr, N: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, MODE: tl.constexpr):
    group = tl.program_id(0)
    tile = tl.program_id(1)
    rows = tl.arange(0, BM)
    cols = tile * BN + tl.arange(0, BN)
    if MODE == 'pool_max':
        accum = tl.full((BM, BN), -float('inf'), tl.float32)
    else:
        accum = tl.full((BM, BN), 0., tl.float32)
    second = tl.full((BM, BN), 0., tl.float32)
    denominator = tl.full((BM,), 0., tl.float32)
    for start in range(0, M, BM):
        r = start + rows
        mask = (r[:, None] < M) & (cols[None, :] < N)
        offsets = (group * M + r[:, None]) * N + cols[None, :]
        x = tl.load(a + offsets, mask, other=0.).to(tl.float32)
        if MODE == 'dwdb':
            second += tl.load(b + offsets, mask, other=0.).to(tl.float32)
        if MODE == 'pool_mean' or MODE == 'pool_max':
            weight = tl.load(weights + group * M + r, r < M, other=0.)
            if MODE == 'pool_max':
                accum = tl.maximum(accum, tl.where(weight[:, None] > 0, x, -float('inf')))
            else:
                accum += x * weight[:, None]
                denominator += weight
        else:
            accum += x
    if MODE == 'pool_max':
        value = tl.max(accum, 0)
    else:
        value = tl.sum(accum, 0)
    if MODE == 'pool_mean':
        value /= tl.maximum(tl.sum(denominator, 0), 1.e-9)
    tl.store(out + group * N + cols, value, cols < N)
    if MODE == 'dwdb':
        tl.store(aux + group * N + cols, tl.sum(second, 0), cols < N)


@triton.jit
def weighted_pv(prob, values, out, M: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr):
    group, tile = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, BM)
    cols = tile * BN + tl.arange(0, BN)
    accum = tl.full((BM, BN), 0., tl.float32)
    for start in range(0, M, BM):
        r = start + rows
        p = tl.load(prob + group * M + r, r < M, other=0.).to(tl.float32)
        v = tl.load(values + (group * M + r[:, None]) * N + cols[None, :],
                    (r[:, None] < M) & (cols[None, :] < N), other=0.).to(tl.float32)
        accum += p[:, None] * v
    tl.store(out + group * N + cols, tl.sum(accum, 0), cols < N)


@triton.jit
def segment_merge(partial, maxima, sums, out, M: tl.constexpr, N: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr):
    group, tile = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, BM)
    cols = tile * BN + tl.arange(0, BN)
    maximum = tl.load(maxima + group * M + rows, rows < M, other=-float('inf'))
    total = tl.load(sums + group * M + rows, rows < M, other=0.)
    valid = (rows < M) & (total > 0.) & (maximum != -float('inf'))
    maximum = tl.where(valid, maximum, -float('inf'))
    scale = tl.where(valid, tl.exp(maximum - tl.max(maximum, 0)), 0.)
    x = tl.load(partial + (group * M + rows[:, None]) * N + cols[None, :],
                (rows[:, None] < M) & (cols[None, :] < N), other=0.)
    denominator = tl.sum(total * scale, 0)
    numerator = tl.sum(x * scale[:, None], 0)
    y = tl.where(denominator > 0., numerator / denominator, 0.)
    tl.store(out + group * N + cols, y, cols < N)


@triton.jit
def column_quant(x, quantized, scale_out, M: tl.constexpr, N: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr):
    # Deliberately a dynamic FP8 fixture, not MXFP8's E8M0 scale recipe.
    group, tile = tl.program_id(0), tl.program_id(1)
    rows = tl.arange(0, BM)
    cols = tile * BN + tl.arange(0, BN)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    raw = tl.load(x + (group * M + rows[:, None]) * N + cols[None, :], valid, other=0.)
    value = raw.to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value), 0), 1.e-12) / 448.
    normalized = tl.minimum(tl.maximum(value / scale[None, :], -448.), 448.)
    # Actual column-major output, with unchanged format across all variants.
    tl.store(quantized + (group * N + cols[None, :]) * M + rows[:, None],
             normalized.to(tl.float8e4nv), valid)
    tl.store(scale_out + group * N + cols, scale, cols < N)


@triton.jit
def rms_backward(x, dy, gamma, rstd, dx, partial_dw, M: tl.constexpr,
                  N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    # N==BN: hidden-axis statistics are complete, never tile-local estimates.
    group, tile = tl.program_id(0), tl.program_id(1)
    rows = tile * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    offset = (group * M + rows[:, None]) * N + cols[None, :]
    a = tl.load(x + offset, valid, other=0.).to(tl.float32)
    b = tl.load(dy + offset, valid, other=0.).to(tl.float32)
    w = tl.load(gamma + cols, cols < N, other=0.).to(tl.float32)
    inv = tl.load(rstd + group * M + rows, rows < M, other=0.)
    wdy = b * w[None, :]
    dot = tl.sum(wdy * a, 1) / N
    grad = inv[:, None] * wdy - (inv * inv * inv * dot)[:, None] * a
    tl.store(dx + offset, grad, valid)
    dw = tl.sum(b * a * inv[:, None], 0)
    tl.store(partial_dw + (group * tl.cdiv(M, BM) + tile) * N + cols, dw, cols < N)


@triton.jit
def recurrent_gdn(state, queries, keys, values, gates, beta, out, updated,
                   M: tl.constexpr, N: tl.constexpr, BM: tl.constexpr,
                   BN: tl.constexpr, STEPS: tl.constexpr):
    # Physical state [group,V,K], matching the audited layer's V-first path.
    group, tile = tl.program_id(0), tl.program_id(1)
    rows = tile * BM + tl.arange(0, BM)
    cols = tl.arange(0, BN)
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    offsets = (group * M + rows[:, None]) * N + cols[None, :]
    h = tl.load(state + offsets, valid, other=0.).to(tl.float32)
    for step in range(0, STEPS):
        q = tl.load(queries + (group * STEPS + step) * N + cols, cols < N, other=0.).to(tl.float32)
        k = tl.load(keys + (group * STEPS + step) * N + cols, cols < N, other=0.).to(tl.float32)
        v = tl.load(values + (group * STEPS + step) * M + rows, rows < M, other=0.).to(tl.float32)
        decay = tl.exp(tl.load(gates + group * STEPS + step))
        strength = tl.load(beta + group * STEPS + step)
        h *= decay
        delta = strength * (v - tl.sum(h * k[None, :], 1))
        h += delta[:, None] * k[None, :]
        y = tl.sum(h * q[None, :], 1)
        tl.store(out + (group * STEPS + step) * M + rows, y, rows < M)
    tl.store(updated + offsets, h, valid)
