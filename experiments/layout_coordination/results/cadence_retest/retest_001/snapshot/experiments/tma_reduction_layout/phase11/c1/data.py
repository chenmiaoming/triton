"""CPU references; quarter-pattern output is exactly representable in FP32."""
import numpy as np


def bf16_round(value):
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7fff) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xffff0000)).view(np.float32)


def tiles(m, n, pids, slot=0, pattern='quarter'):
    pids = np.asarray(pids, dtype=np.int64)
    pm, pn = pids // (8192 // n), pids % (8192 // n)
    pm = (pm + 128 * slot) % 256
    row = pm[:, None, None] * m + np.arange(m, dtype=np.int64)[None, :, None]
    col = pn[:, None, None] * n + np.arange(n, dtype=np.int64)[None, None, :]
    linear = row * 8192 + col
    if pattern == 'quarter':
        return (((linear % 97) - 48) / 4).astype(np.float32)
    if pattern == 'boundary':
        values = np.asarray([-16, -8, -0., 0., 8, 15.75, .25, -.25], dtype=np.float32)
        return values[linear % len(values)]
    assert pattern == 'random'
    # A deterministic, finite input with non-quarter BF16 values; no RNG API
    # differences between the CPU reference and the GPU input preparation.
    bits = ((linear * 1664525 + 1013904223) & 65535).astype(np.float32)
    return bf16_round((bits - 32768) / np.float32(8192))


def reference(case, k, pids, pattern='quarter'):
    m, n = case['M'], case['N']
    value = tiles(m, n, pids, pattern=pattern)
    if case['family'] == 'ML':
        other = tiles(m, n, pids, slot=1, pattern=pattern)
        result = np.zeros_like(value)
        for i in range(k):
            result = result + (other if i & 1 else value)
        return result
    step = ((np.arange(m)[:, None] * 13 + np.arange(n)[None, :] * 7) % 7 + 1).astype(np.float32) * .25
    result = np.zeros((len(value), n), dtype=np.float32)
    for _ in range(k):
        result = result + value.max(axis=1)
        value = bf16_round(np.where(value >= 8, value - 16, value + step))
    return result


def input_tensor(torch, pattern='quarter'):
    # Input generation is outside event/profiler regions and never compiled
    # into one of the sixteen Gluon programs.
    linear = torch.arange(8192 * 8192, dtype=torch.int64, device='cuda').reshape(1, 8192, 8192)
    if pattern == 'quarter':
        return (((linear % 97) - 48).to(torch.float32) / 4).to(torch.bfloat16)
    if pattern == 'boundary':
        values = torch.tensor([-16, -8, -0., 0., 8, 15.75, .25, -.25], device='cuda')
        return values[linear % 8].to(torch.bfloat16)
    assert pattern == 'random'
    return ((((linear * 1664525 + 1013904223) & 65535).to(torch.float32) - 32768) / 8192).to(torch.bfloat16)


def compare(actual, expected):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    return {'pass': bool(np.array_equal(actual, expected)), 'elements': int(actual.size),
            'max_abs_error': float(np.max(np.abs(actual - expected))),
            'actual_SHA256': __import__('hashlib').sha256(actual.tobytes()).hexdigest(),
            'reference_SHA256': __import__('hashlib').sha256(expected.tobytes()).hexdigest()}
