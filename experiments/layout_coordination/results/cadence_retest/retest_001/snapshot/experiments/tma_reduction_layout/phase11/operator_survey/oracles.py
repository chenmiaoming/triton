"""CPU data and high precision semantic references, independent of Triton/Gluon."""
import numpy as np


def bf16_round(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def fp8_table():
    codes = np.arange(127, dtype=np.int32)
    exponent, fraction = codes >> 3, codes & 7
    return np.where(exponent == 0, fraction * 2. ** -9,
                    (1 + fraction / 8.) * np.exp2(exponent - 7)).astype(np.float32)


def fp8_encode(x):
    """Finite saturated E4M3FN, round to nearest even, including signed zero."""
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite quantization input outside this fixture policy')
    table = fp8_table()
    a = np.minimum(np.abs(x), np.float32(448.))
    high = np.minimum(np.searchsorted(table, a), 126)
    low = np.maximum(high - 1, 0)
    dl, dh = a - table[low], table[high] - a
    choose_high = (dh < dl) | ((dh == dl) & ((high & 1) == 0))
    codes = np.where(choose_high, high, low).astype(np.uint8)
    return codes | (np.signbit(x).astype(np.uint8) << 7)


def inputs(case, pattern='finite', seed=20261021):
    g, m, n = case['groups_for_correctness'], case['M'], case['N']
    rng = np.random.default_rng(seed)
    kernel = case['kernel']
    def matrix(shape, low_precision=True):
        x = (rng.standard_normal(shape) * .125).astype(np.float32)
        if pattern == 'quarter':
            x = ((np.arange(x.size).reshape(shape) % 17 - 8) / 8.).astype(np.float32)
        return bf16_round(x) if low_precision else x
    if kernel == 'column_reduce':
        a = matrix((g, m, n), case['signature']['a'] == '*bf16')
        b = matrix((g, m, n), case['signature']['b'] == '*bf16')
        weights = rng.choice([0., .5, 1.], (g, m)).astype(np.float32)
        if pattern == 'masked':
            weights[0] = 0.
        return dict(a=a, b=b, weights=weights)
    if kernel == 'weighted_pv':
        p = rng.uniform(.1, 1., (g, m)).astype(np.float32)
        p /= p.sum(1, keepdims=True)
        if pattern == 'masked':
            p[0] = 0.
        return dict(prob=p, values=matrix((g, m, n)))
    if kernel == 'segment_merge':
        maxima = rng.uniform(-10., 10., (g, m)).astype(np.float32)
        sums = rng.uniform(.2, 4., (g, m)).astype(np.float32)
        partial = matrix((g, m, n), False) * sums[:, :, None]
        if pattern == 'masked':
            sums[0] = 0.
            maxima[0] = -np.inf
            partial[0] = 0.
        return dict(partial=partial, maxima=maxima, sums=sums)
    if kernel == 'column_quant':
        x = matrix((g, m, n))
        if pattern == 'masked':
            x[0] = 0.
        return dict(x=x)
    if kernel == 'rms_backward':
        x, dy = matrix((g, m, n)), matrix((g, m, n))
        gamma = matrix((n,)) + np.float32(1.)
        gamma = bf16_round(gamma)
        rstd = (1 / np.sqrt((x.astype(np.float64) ** 2).mean(2) + 1.e-5)).astype(np.float32)
        return dict(x=x, dy=dy, gamma=gamma, rstd=rstd)
    if kernel == 'recurrent_gdn':
        t = case['STEPS']
        q, k = matrix((g, t, n)), matrix((g, t, n))
        q = bf16_round(q / np.sqrt(n))
        k = bf16_round(k / np.sqrt(n))
        return dict(state=matrix((g, m, n), False), queries=q, keys=k,
                    values=matrix((g, t, m)), gates=np.full((g, t), -.125, np.float32),
                    beta=np.full((g, t), .5, np.float32))
    raise ValueError(kernel)


def reference(case, data):
    d = {k: v.astype(np.float64) for k, v in data.items()}
    kernel = case['kernel']
    if kernel == 'column_reduce':
        mode = case['MODE']
        if mode == 'pool_max':
            return dict(out=np.where(d['weights'][:, :, None] > 0, d['a'], -np.inf).max(1))
        if mode == 'pool_mean':
            value = (d['a'] * d['weights'][:, :, None]).sum(1)
            return dict(out=value / np.maximum(d['weights'].sum(1), 1.e-9)[:, None])
        result = dict(out=d['a'].sum(1))
        if mode == 'dwdb':
            result['aux'] = d['b'].sum(1)
        return result
    if kernel == 'weighted_pv':
        return dict(out=(d['prob'][:, :, None] * d['values']).sum(1))
    if kernel == 'segment_merge':
        valid = (d['sums'] > 0) & np.isfinite(d['maxima'])
        maxima = np.where(valid, d['maxima'], -np.inf)
        with np.errstate(invalid='ignore'):
            scale = np.where(valid, np.exp(maxima - maxima.max(1, keepdims=True)), 0.)
        numerator = (d['partial'] * scale[:, :, None]).sum(1)
        denominator = (d['sums'] * scale).sum(1)
        value = np.zeros_like(numerator)
        np.divide(numerator, denominator[:, None], out=value, where=denominator[:, None] > 0)
        return dict(out=value)
    if kernel == 'column_quant':
        # Reproduce only the FP32 scale recipe's rounding boundary, then use an
        # independent exhaustive FP8 codebook for the conversion itself.
        x = data['x']
        scale = np.maximum(np.abs(x).max(1), np.float32(1.e-12)) / np.float32(448.)
        normalized = np.clip(x / scale[:, None, :], -448., 448.)
        return dict(quantized=fp8_encode(normalized).transpose(0, 2, 1).copy(), scale_out=scale)
    if kernel == 'rms_backward':
        x, dy, w, inv = d['x'], d['dy'], d['gamma'], d['rstd']
        wdy = dy * w
        dot = (wdy * x).mean(2)
        grad = inv[:, :, None] * wdy - (inv ** 3 * dot)[:, :, None] * x
        partial = dy * x * inv[:, :, None]
        bm = case['BM']
        dw = np.stack([partial[:, i:i+bm].sum(1) for i in range(0, case['M'], bm)], 1)
        return dict(dx=grad, partial_dw=dw)
    if kernel == 'recurrent_gdn':
        state = d['state'].copy()
        outputs = []
        for t in range(case['STEPS']):
            state *= np.exp(d['gates'][:, t, None, None])
            prediction = np.einsum('gmk,gk->gm', state, d['keys'][:, t])
            delta = d['beta'][:, t, None] * (d['values'][:, t] - prediction)
            state += delta[:, :, None] * d['keys'][:, t, None, :]
            outputs.append(np.einsum('gmk,gk->gm', state, d['queries'][:, t]))
        return dict(out=np.stack(outputs, 1), updated=state)
    raise ValueError(kernel)


def validate_cpu_references():
    # Independent CPU torch/autograd checks validate the references, not GPU kernels.
    import torch
    from .cases import cases
    torch.set_num_threads(1)
    rows = cases()
    checks = []
    for case in rows:
        for pattern in ('quarter', 'finite', 'masked'):
            data = inputs(case, pattern)
            answer = reference(case, data)
            if case['kernel'] == 'rms_backward':
                x = torch.tensor(data['x'], dtype=torch.float64, requires_grad=True)
                w = torch.tensor(data['gamma'], dtype=torch.float64, requires_grad=True)
                dy = torch.tensor(data['dy'], dtype=torch.float64)
                y = x * torch.rsqrt((x * x).mean(2, keepdim=True) + 1.e-5) * w
                y.backward(dy)
                np.testing.assert_allclose(x.grad.numpy(), answer['dx'], rtol=2.e-6, atol=2.e-6)
                np.testing.assert_allclose(w.grad.numpy(), answer['partial_dw'].sum((0, 1)), rtol=2.e-6, atol=2.e-6)
            if case['kernel'] == 'recurrent_gdn':
                td = {k: torch.tensor(v, dtype=torch.float64) for k, v in data.items()}
                h = td['state'].clone()
                out = []
                for t in range(case['STEPS']):
                    h = h * td['gates'][:, t].exp()[:, None, None]
                    pred = (h * td['keys'][:, t, None, :]).sum(2)
                    update = td['beta'][:, t, None] * (td['values'][:, t] - pred)
                    h = h + update[:, :, None] * td['keys'][:, t, None, :]
                    out.append((h * td['queries'][:, t, None, :]).sum(2))
                np.testing.assert_allclose(h.numpy(), answer['updated'], rtol=1.e-12, atol=1.e-12)
                np.testing.assert_allclose(torch.stack(out, 1).numpy(), answer['out'], rtol=1.e-12, atol=1.e-12)
            if pattern == 'masked' and case['kernel'] in ('segment_merge', 'weighted_pv'):
                assert np.all(answer['out'][0] == 0.)
            if pattern == 'masked' and case.get('MODE') == 'pool_mean':
                assert np.all(answer['out'][0] == 0.)
            if pattern == 'masked' and case.get('MODE') == 'pool_max':
                assert np.all(np.isneginf(answer['out'][0]))
            checks.append({'case': case['id'], 'pattern': pattern, 'outputs': sorted(answer), 'pass': True})
    table = fp8_table()
    midpoints = (table[1:] + table[:-1]) / 2
    samples = np.concatenate([table, midpoints, -table, -midpoints, np.array([-0., 448., -448.], np.float32)])
    expected = torch.from_numpy(samples).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    np.testing.assert_array_equal(fp8_encode(samples), expected)
    assert not torch.cuda.is_initialized()
    return {'status': 'PASS', 'checks': checks, 'FP8_codebook_checks': len(samples),
            'torch_CUDA_initialized': False, 'GPU_kernel_correctness_verified': False}
