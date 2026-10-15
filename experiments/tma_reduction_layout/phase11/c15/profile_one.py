"""One checked driver-only C1 launch in the profiler region."""
import argparse
import ctypes as C
import json
import sys

import numpy as np
from experiments.tma_reduction_layout.phase11.c15 import common as c
from experiments.tma_reduction_layout.phase11.c15.data import reference, compare
from experiments.tma_reduction_layout.phase11.c15.driver import Driver, Loaded


class Pointer:
    def __init__(self, value):
        self.value = value

    def data_ptr(self):
        return self.value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--k', type=int, required=True)
    args = parser.parse_args()
    meta = c.read(args.metadata)[args.tag]
    driver = Driver()
    context = C.c_void_p()
    driver.call('cuCtxCreate_v2', [C.POINTER(C.c_void_p), C.c_uint, C.c_int], C.byref(context), 0, driver.device)
    # BF16 quarter values have exactly representable high 16-bit float bits.
    values = (np.arange(97, dtype=np.float32) - 48) / 4
    pattern = (values.view(np.uint32) >> 16).astype('<u2').tobytes()
    raw = (pattern * ((8192 * 8192) // 97 + 1))[:8192 * 8192 * 2]
    stride = meta['M'] * meta['N'] if meta['family'] == 'ML' else meta['N']
    xptr, outptr, scratchptr = C.c_uint64(), C.c_uint64(), C.c_uint64()
    for ptr, size in ((xptr, len(raw)), (outptr, stride * 4), (scratchptr, max(1, meta['global_scratch_size']))):
        driver.call('cuMemAlloc_v2', [C.POINTER(C.c_uint64), C.c_size_t], C.byref(ptr), size)
    host = C.create_string_buffer(raw)
    driver.call('cuMemcpyHtoD_v2', [C.c_uint64, C.c_void_p, C.c_size_t],
                xptr, C.cast(host, C.c_void_p), len(raw))
    del host, raw
    kernel = Loaded(driver, args.root, meta)
    owners, params, serialized = kernel.parameters(Pointer(xptr.value), Pointer(outptr.value),
                                                   Pointer(scratchptr.value), args.k)
    stream = C.c_void_p(0)
    kernel.guard()
    kernel.launch(1, params, stream)
    driver.call('cuStreamSynchronize', [C.c_void_p], stream)
    kernel.guard()
    driver.call('cuProfilerStart', [])
    kernel.launch(1, params, stream)
    driver.call('cuStreamSynchronize', [C.c_void_p], stream)
    driver.call('cuProfilerStop', [])
    out = np.empty(stride, dtype=np.float32)
    driver.call('cuMemcpyDtoH_v2', [C.c_void_p, C.c_uint64, C.c_size_t],
                C.c_void_p(out.ctypes.data), outptr, out.nbytes)
    shape = (1, meta['M'], meta['N']) if meta['family'] == 'ML' else (1, meta['N'])
    checked = compare(out.reshape(shape), reference(meta, args.k, [0]))
    assert checked['pass'], checked
    assert not any(m == 'triton' or m.startswith('triton.') for m in sys.modules)
    print('PROFILE_BINDING ' + json.dumps({'tag': args.tag, 'K': args.k, 'grid': 1,
          'binding': kernel.binding(), 'correctness': checked, 'parameters': serialized,
          'environment': driver.environment(), 'Triton_imported': False, 'compile_calls': 0}))
    driver.call('cuModuleUnload', [C.c_void_p], kernel.module)
    for ptr in (xptr, outptr, scratchptr):
        driver.call('cuMemFree_v2', [C.c_uint64], ptr)
    driver.call('cuCtxDestroy_v2', [C.c_void_p], context)


if __name__ == '__main__':
    main()
