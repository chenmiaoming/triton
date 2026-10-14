"""Driver-only archive profiler. One checked launch inside the profile region."""
import argparse
import ctypes as C
import json
import struct
from pathlib import Path
from experiments.tma_reduction_layout.phase11.launch import Driver,Loaded
from experiments.tma_reduction_layout.phase11.common import read


class Pointer:
    def __init__(self,value):
        self.value = value
    def data_ptr(self):
        return self.value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--metadata',required=True)
    parser.add_argument('--tag',required=True)
    args = parser.parse_args()
    meta = read(args.metadata)[args.tag]
    driver = Driver()
    context = C.c_void_p()
    driver.call('cuCtxCreate_v2',[C.POINTER(C.c_void_p),C.c_uint,C.c_int],C.byref(context),0,driver.device)
    if meta['dtype']=='bfloat16':
        pattern = b''.join(struct.pack('<H',struct.unpack('<I',struct.pack('<f',v/4))[0]>>16) for v in range(-48,49))
        elem_bytes = 2
    else:
        pattern = b''.join(struct.pack('<f',v/4) for v in range(-48,49))
        elem_bytes = 4
    count = 8192*8192
    raw = (pattern*(count//97+1))[:count*elem_bytes]
    xptr,outptr,scratchptr = C.c_uint64(),C.c_uint64(),C.c_uint64()
    scratch_bytes = max(1,meta['global_scratch_size']*meta['full_grid'])
    for ptr,size in ((xptr,len(raw)),(outptr,256*8192*4),(scratchptr,scratch_bytes)):
        driver.call('cuMemAlloc_v2',[C.POINTER(C.c_uint64),C.c_size_t],C.byref(ptr),size)
    host = C.create_string_buffer(raw)
    driver.call('cuMemcpyHtoD_v2',[C.c_uint64,C.c_void_p,C.c_size_t],xptr,C.cast(host,C.c_void_p),len(raw))
    del raw,host
    kernel = Loaded(driver,args.root,meta)
    owners,params,values = kernel.parameters(Pointer(xptr.value),Pointer(outptr.value),Pointer(scratchptr.value))
    stream = C.c_void_p(0)
    kernel.guard()
    kernel.launch(meta['full_grid'],params,stream)
    driver.call('cuStreamSynchronize',[C.c_void_p],stream)
    digest = kernel.guard()
    driver.call('cuProfilerStart',[])
    kernel.launch(meta['full_grid'],params,stream)
    driver.call('cuStreamSynchronize',[C.c_void_p],stream)
    driver.call('cuProfilerStop',[])
    out = (C.c_float*128)()
    driver.call('cuMemcpyDtoH_v2',[C.c_void_p,C.c_uint64,C.c_size_t],C.cast(out,C.c_void_p),outptr,C.sizeof(out))
    expected = [max((((r*8192+n)%97)-48)/4 for r in range(32)) for n in range(128)]
    assert list(out)==expected, (meta['tag'],list(out)[:6],expected[:6])
    print('PROFILE_BINDING '+json.dumps({'tag':meta['tag'],'CUBIN_SHA256':digest,'binding':kernel.binding(),
          'full_grid':meta['full_grid'],'parameters':values,'environment':driver.environment(),
          'correctness_prefix128':True,'Triton_imported':False,'compile_calls':0}))
    driver.call('cuModuleUnload',[C.c_void_p],kernel.module)
    for ptr in (xptr,outptr,scratchptr):
        driver.call('cuMemFree_v2',[C.c_uint64],ptr)
    driver.call('cuCtxDestroy_v2',[C.c_void_p],context)


if __name__=='__main__':
    main()
