"""Original issue geometry, with valid host and block-uniform device descriptors."""
import triton
import triton.language as tl


@triton.jit
def host_kernel(desc, out, BM:tl.constexpr, BN:tl.constexpr):
    pid = tl.program_id(0)
    pm = pid // (8192 // BN)
    pn = pid % (8192 // BN)
    x = desc.load([0, pm*BM, pn*BN]).to(tl.float32)
    y = tl.max(x, axis=1)
    tl.store(out + pm*8192 + pn*BN + tl.arange(0,BN), tl.reshape(y,[BN]))


@triton.jit
def device_kernel(x_ptr, out, BM:tl.constexpr, BN:tl.constexpr):
    pid = tl.program_id(0)
    pm = pid // (8192 // BN)
    pn = pid % (8192 // BN)
    desc = tl.make_tensor_descriptor(x_ptr,shape=[1,8192,8192],
        strides=[8192*8192,8192,1],block_shape=[1,BM,BN])
    x = desc.load([0,pm*BM,pn*BN]).to(tl.float32)
    y = tl.max(x,axis=1)
    tl.store(out + pm*8192 + pn*BN + tl.arange(0,BN),tl.reshape(y,[BN]))
