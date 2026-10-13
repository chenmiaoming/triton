"""Valid block programs: descriptor load, reduction/controls, ordinary stores."""
import triton
import triton.language as tl


@triton.jit
def kernel(arg, out, BM: tl.constexpr, BN: tl.constexpr, HOST: tl.constexpr, OP: tl.constexpr, MODE: tl.constexpr):
    pid = tl.program_id(0)
    pm = pid // (8192 // BN)
    pn = pid % (8192 // BN)
    if HOST:
        desc = arg
    else:
        desc = tl.make_tensor_descriptor(arg, shape=[1, 8192, 8192], strides=[8192 * 8192, 8192, 1],
                                         block_shape=[1, BM, BN])
    raw = desc.load([0, pm * BM, pn * BN])
    if MODE == 'copy':
        offsets = pid * (BM * BN) + tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
        tl.store(out + offsets, tl.reshape(raw, [BM, BN]))
    else:
        x = raw.to(tl.float32)
        if MODE == 'innermost':
            y = tl.max(x, axis=2)
            offsets = pid * BM + tl.arange(0, BM)
            tl.store(out + offsets, tl.reshape(y, [BM]))
        else:
            if MODE == 'multi_use':
                y = tl.max(x, axis=1) + tl.min(x, axis=1)
            elif OP == 'max':
                y = tl.max(x, axis=1)
            elif OP == 'min':
                y = tl.min(x, axis=1)
            else:
                y = tl.sum(x, axis=1)
            offsets = pm * 8192 + pn * BN + tl.arange(0, BN)
            tl.store(out + offsets, tl.reshape(y, [BN]))
