"""Layout interventions for the canonical fixtures; valid block programs only."""
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def column_reduce(a, b, weights, out, aux, M: gl.constexpr, N: gl.constexpr,
                  BM: gl.constexpr, BN: gl.constexpr, MODE: gl.constexpr,
                  L: gl.constexpr):
    group = gl.program_id(0)
    tile = gl.program_id(1)
    rows = gl.arange(0, BM, layout=gl.SliceLayout(1, L))
    cols = tile * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, L))
    if MODE == 'pool_max':
        accum = gl.full((BM, BN), -float('inf'), gl.float32, layout=L)
    else:
        accum = gl.full((BM, BN), 0., gl.float32, layout=L)
    second = gl.full((BM, BN), 0., gl.float32, layout=L)
    denominator = gl.full((BM,), 0., gl.float32, layout=gl.SliceLayout(1, L))
    for start in range(0, M, BM):
        r = start + rows
        mask = (r[:, None] < M) & (cols[None, :] < N)
        offsets = (group * M + r[:, None]) * N + cols[None, :]
        x = gl.load(a + offsets, mask, other=0.).to(gl.float32)
        if MODE == 'dwdb':
            second += gl.load(b + offsets, mask, other=0.).to(gl.float32)
        if MODE == 'pool_mean' or MODE == 'pool_max':
            weight = gl.load(weights + group * M + r, r < M, other=0.)
            if MODE == 'pool_max':
                accum = gl.maximum(accum, gl.where(weight[:, None] > 0, x, -float('inf')))
            else:
                accum += x * weight[:, None]
                denominator += weight
        else:
            accum += x
    if MODE == 'pool_max':
        value = gl.max(accum, 0)
    else:
        value = gl.sum(accum, 0)
    if MODE == 'pool_mean':
        value /= gl.maximum(gl.sum(denominator, 0), 1.e-9)
    gl.store(out + group * N + cols, value, cols < N)
    if MODE == 'dwdb':
        gl.store(aux + group * N + cols, gl.sum(second, 0), cols < N)


@gluon.jit
def weighted_pv(prob, values, out, M: gl.constexpr, N: gl.constexpr,
                BM: gl.constexpr, BN: gl.constexpr, L: gl.constexpr):
    group, tile = gl.program_id(0), gl.program_id(1)
    rows = gl.arange(0, BM, layout=gl.SliceLayout(1, L))
    cols = tile * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, L))
    accum = gl.full((BM, BN), 0., gl.float32, layout=L)
    for start in range(0, M, BM):
        r = start + rows
        p = gl.load(prob + group * M + r, r < M, other=0.).to(gl.float32)
        v = gl.load(values + (group * M + r[:, None]) * N + cols[None, :],
                    (r[:, None] < M) & (cols[None, :] < N), other=0.).to(gl.float32)
        accum += p[:, None] * v
    gl.store(out + group * N + cols, gl.sum(accum, 0), cols < N)


@gluon.jit
def segment_merge(partial, maxima, sums, out, M: gl.constexpr, N: gl.constexpr,
                  BM: gl.constexpr, BN: gl.constexpr, L: gl.constexpr):
    group, tile = gl.program_id(0), gl.program_id(1)
    rows = gl.arange(0, BM, layout=gl.SliceLayout(1, L))
    cols = tile * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, L))
    maximum = gl.load(maxima + group * M + rows, rows < M, other=-float('inf'))
    total = gl.load(sums + group * M + rows, rows < M, other=0.)
    valid = (rows < M) & (total > 0.) & (maximum != -float('inf'))
    maximum = gl.where(valid, maximum, -float('inf'))
    scale = gl.where(valid, gl.exp(maximum - gl.max(maximum, 0)), 0.)
    x = gl.load(partial + (group * M + rows[:, None]) * N + cols[None, :],
                (rows[:, None] < M) & (cols[None, :] < N), other=0.)
    denominator = gl.sum(total * scale, 0)
    numerator = gl.sum(x * scale[:, None], 0)
    y = gl.where(denominator > 0., numerator / denominator, 0.)
    gl.store(out + group * N + cols, y, cols < N)


@gluon.jit
def column_quant(x, quantized, scale_out, M: gl.constexpr, N: gl.constexpr,
                 BM: gl.constexpr, BN: gl.constexpr, L: gl.constexpr):
    group, tile = gl.program_id(0), gl.program_id(1)
    rows = gl.arange(0, BM, layout=gl.SliceLayout(1, L))
    cols = tile * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, L))
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    raw = gl.load(x + (group * M + rows[:, None]) * N + cols[None, :], valid, other=0.)
    value = raw.to(gl.float32)
    scale = gl.maximum(gl.max(gl.abs(value), 0), 1.e-12) / 448.
    normalized = gl.minimum(gl.maximum(value / scale[None, :], -448.), 448.)
    gl.store(quantized + (group * N + cols[None, :]) * M + rows[:, None],
             normalized.to(gl.float8e4nv), valid)
    gl.store(scale_out + group * N + cols, scale, cols < N)


@gluon.jit
def rms_backward(x, dy, gamma, rstd, dx, partial_dw, M: gl.constexpr,
                  N: gl.constexpr, BM: gl.constexpr, BN: gl.constexpr,
                  L: gl.constexpr):
    group, tile = gl.program_id(0), gl.program_id(1)
    rows = tile * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, L))
    cols = gl.arange(0, BN, layout=gl.SliceLayout(0, L))
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    offset = (group * M + rows[:, None]) * N + cols[None, :]
    a = gl.load(x + offset, valid, other=0.).to(gl.float32)
    b = gl.load(dy + offset, valid, other=0.).to(gl.float32)
    w = gl.load(gamma + cols, cols < N, other=0.).to(gl.float32)
    inv = gl.load(rstd + group * M + rows, rows < M, other=0.)
    wdy = b * w[None, :]
    dot = gl.sum(wdy * a, 1) / N
    grad = inv[:, None] * wdy - (inv * inv * inv * dot)[:, None] * a
    gl.store(dx + offset, grad, valid)
    dw = gl.sum(b * a * inv[:, None], 0)
    gl.store(partial_dw + (group * gl.cdiv(M, BM) + tile) * N + cols, dw, cols < N)


@gluon.jit
def recurrent_gdn(state, queries, keys, values, gates, beta, out, updated,
                   M: gl.constexpr, N: gl.constexpr, BM: gl.constexpr,
                   BN: gl.constexpr, STEPS: gl.constexpr, L: gl.constexpr):
    group, tile = gl.program_id(0), gl.program_id(1)
    rows = tile * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, L))
    cols = gl.arange(0, BN, layout=gl.SliceLayout(0, L))
    valid = (rows[:, None] < M) & (cols[None, :] < N)
    offsets = (group * M + rows[:, None]) * N + cols[None, :]
    h = gl.load(state + offsets, valid, other=0.).to(gl.float32)
    for step in range(0, STEPS):
        q = gl.load(queries + (group * STEPS + step) * N + cols, cols < N, other=0.).to(gl.float32)
        k = gl.load(keys + (group * STEPS + step) * N + cols, cols < N, other=0.).to(gl.float32)
        v = gl.load(values + (group * STEPS + step) * M + rows, rows < M, other=0.).to(gl.float32)
        decay = gl.exp(gl.load(gates + group * STEPS + step))
        strength = gl.load(beta + group * STEPS + step)
        h *= decay
        delta = strength * (v - gl.sum(h * k[None, :], 1))
        h += delta[:, None] * k[None, :]
        y = gl.sum(h * q[None, :], 1)
        gl.store(out + (group * STEPS + step) * M + rows, y, rows < M)
    gl.store(updated + offsets, h, valid)
