"""Driver-only application replay; no PyTorch dependency in NCU image."""
import argparse
import ctypes as C
import json
import struct
import sys
import numpy as np
from experiments.tma_reduction_layout.phase11.c16 import common as c
from experiments.tma_reduction_layout.phase11.c16 import runtime as r


class Pointer:
    def __init__(self, address):
        self.address = address

    def data_ptr(self):
        return self.address


def resources(driver, protocol, metadata):
    context = C.c_void_p()
    driver.call('cuCtxCreate_v2', [C.POINTER(C.c_void_p), C.c_uint, C.c_int], C.byref(context), 0, driver.device)
    pattern = b''.join(struct.pack('<H', struct.unpack('<I', struct.pack('<f', v / 4))[0] >> 16) for v in range(-48,49))
    size = 8192 * 8192 * 2
    raw = (pattern * (8192 * 8192 // 97 + 1))[:size]
    scratch_size = max(1, max(m['global_scratch_size'] * m['full_grid'] for m in metadata.values()))
    pointers = []
    for count in (size, 256 * 8192 * 4, scratch_size, protocol['cache_eviction_bytes']):
        ptr = C.c_uint64()
        driver.call('cuMemAlloc_v2', [C.POINTER(C.c_uint64), C.c_size_t], C.byref(ptr), count)
        pointers.append(ptr)
    host = C.create_string_buffer(raw)
    driver.call('cuMemcpyHtoD_v2', [C.c_uint64,C.c_void_p,C.c_size_t], pointers[0], C.cast(host,C.c_void_p), size)
    return context, pointers


def evict(driver, pointers, protocol):
    driver.call('cuMemsetD8_v2', [C.c_uint64,C.c_ubyte,C.c_size_t],
                pointers[3], 0, protocol['cache_eviction_bytes'])


def correctness(driver, meta, out):
    grid, n = meta['full_grid'], meta['N']
    pids = sorted(set([0,1,grid//2,grid-1]))
    actual = []
    for pid in pids:
        host = (C.c_float * n)()
        driver.call('cuMemcpyDtoH_v2', [C.c_void_p,C.c_uint64,C.c_size_t],
                    C.cast(host,C.c_void_p), out.value + pid*n*4, n*4)
        actual.append(list(host))
    checked = r.compare(np.asarray(actual,dtype=np.float32), r.expected(meta,0,pids))
    assert checked['pass'], (meta['tag'],checked)
    return {'pids':pids, **checked}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--metadata', required=True)
    p.add_argument('--protocol', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--cache', choices=['warm', 'cold'], required=True)
    args = p.parse_args()
    protocol = c.read(args.protocol)
    c.verify_sources(protocol)
    meta = c.read(args.metadata)[args.tag]
    assert meta['family'] in ('FULL', 'TMA')
    driver = r.Driver()
    context,pointers = resources(driver,protocol,{args.tag:meta})
    kernel = r.load(driver,args.root,meta)
    owners,values,serialized = kernel.parameters(*[Pointer(p.value) for p in pointers[:3]])
    stream = C.c_void_p(0)
    for _ in range(3):
        kernel.guard()
        kernel.launch(meta['full_grid'],values,stream)
    driver.call('cuStreamSynchronize',[C.c_void_p],stream)
    if args.cache == 'cold':
        evict(driver,pointers,protocol)
    kernel.guard()
    driver.call('cuProfilerStart', [])
    kernel.launch(meta['full_grid'], values, stream)
    driver.call('cuStreamSynchronize', [C.c_void_p], stream)
    driver.call('cuProfilerStop', [])
    checked = correctness(driver,meta,pointers[1])
    assert not any(m == 'triton' or m.startswith('triton.') for m in sys.modules)
    print('PROFILE_BINDING ' + json.dumps({'tag': args.tag, 'cache': args.cache,
          'binding': kernel.binding(), 'parameters': serialized, 'correctness': checked,
          'protocol_SHA256': c.sha(c.encode(protocol)), 'Triton_imported': False,
          'environment': driver.environment()}), flush=True)
    driver.call('cuModuleUnload',[C.c_void_p],kernel.module)
    for ptr in pointers:
        driver.call('cuMemFree_v2',[C.c_uint64],ptr)
    driver.call('cuCtxDestroy_v2',[C.c_void_p],context)


if __name__ == '__main__':
    main()
