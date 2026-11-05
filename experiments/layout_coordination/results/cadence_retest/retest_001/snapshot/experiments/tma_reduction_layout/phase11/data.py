"""Frozen input recipes and independent max oracles; no compiler imports."""
import torch


def input_tensor(dtype, pattern='quarter'):
    torch.manual_seed(42)
    if pattern == 'random':
        return torch.randn((1,8192,8192), device='cuda', dtype=torch.float32).to(getattr(torch,dtype))
    x = (torch.arange(8192*8192, device='cuda', dtype=torch.int32)%97-48).float()/4
    x = x.to(getattr(torch,dtype)).reshape(1,8192,8192)
    if pattern == 'boundary':
        tiles = x.reshape(256,32,8192)
        tiles[:,0,0] = float('nan')
        tiles[:,31,1] = float('nan')
        tiles[:,:,2] = 0.0
        tiles[:,1::2,2] = -0.0
        tiles[:,:,3] = -0.0
        tiles[:,:,4] = -float('inf')
        tiles[:,:,5] = -float('inf')
        tiles[:,31,5] = float('inf')
    else:
        assert pattern == 'quarter'
    return x


def reference(x):
    # Pinned tl.max uses maximum(..., PropagateNan.NONE), i.e. maxnum.
    # Pairwise torch.fmax implements that combiner without converting infinities.
    values = x.float().reshape(256,32,8192)
    while values.shape[1]>1:
        values = torch.fmax(values[:,0::2,:],values[:,1::2,:])
    return values[:,0,:]


def compare(out, expected):
    nan_mask = torch.isnan(out)
    nan_equal = torch.equal(nan_mask, torch.isnan(expected))
    valid = ~torch.isnan(expected)
    value_equal = torch.equal(out[valid], expected[valid])
    zeros = out == 0
    result = {'nan_mask_equal':nan_equal, 'non_nan_values_equal':value_equal,
              'actual_nan_count':int(nan_mask.sum().item()), 'expected_nan_count':int(torch.isnan(expected).sum().item()),
              'actual_negative_zero_count':int((zeros & torch.signbit(out)).sum().item()),
              'expected_negative_zero_count':int(((expected == 0) & torch.signbit(expected)).sum().item()),
              'signed_zero_bit_equality_required':False}
    result['pass'] = nan_equal and value_equal
    return result
