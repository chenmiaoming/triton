"""Independent exact finite-value oracles. No Triton import."""
import torch
from experiments.tma_reduction_layout.phase10.stage_c import common as c


def input_tensor(dtype):
    # Exact quarter multiples in all three dtypes and all tested f32 sums.
    value = (torch.arange(8192 * 8192, device='cuda', dtype=torch.int32) % 97 - 48).float() / 4
    return value.to(getattr(torch, dtype)).reshape(1, 8192, 8192)


def output_reference(x, case):
    m, n = case['M'], case['N']
    if case['mode'] == 'copy':
        reference = x.reshape(8192 // m, m, 8192 // n, n).permute(0, 2, 1, 3).contiguous().flatten()
    elif case['mode'] == 'innermost':
        reference = x.float().reshape(8192 // m, m, 8192 // n, n).amax(dim=3).permute(0, 2, 1).contiguous().flatten()
    else:
        tiles = x.float().reshape(8192 // m, m, 8192)
        if case['mode'] == 'multi_use':
            reference = tiles.amax(dim=1) + tiles.amin(dim=1)
        else:
            reference = {'max': torch.amax, 'min': torch.amin, 'sum': torch.sum}[case['op']](tiles, dim=1)
        reference = reference.flatten()
    assert reference.numel() == c.grid(case) * c.output_per_cta(case)
    return torch.empty_like(reference), reference
