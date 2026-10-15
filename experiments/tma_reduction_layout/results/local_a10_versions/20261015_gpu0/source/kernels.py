"""Plain Triton anchor, identical to local_a10.kernels.canonical.

No Gluon, explicit layouts, descriptors, or shared-memory staging are requested.
The compiler chooses every hardware layout.
"""
import triton
import triton.language as tl


@triton.jit
def canonical(x, out, BM: tl.constexpr, BN: tl.constexpr):
    pid = tl.program_id(0)
    pm, pn = pid // (8192 // BN), pid % (8192 // BN)
    rows = pm * BM + tl.arange(0, BM)
    cols = pn * BN + tl.arange(0, BN)
    offsets = tl.expand_dims(rows[:, None] * 8192 + cols[None, :], 0)
    values = tl.load(x + offsets).to(tl.float32)
    result = tl.max(values, axis=1)
    tl.store(out + pm * 8192 + cols, tl.reshape(result, [BN]))
