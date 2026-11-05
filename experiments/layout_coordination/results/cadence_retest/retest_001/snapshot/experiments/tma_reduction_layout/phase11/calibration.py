"""Known-access CUDA/PTX instrument fixtures; never Triton bug reproducers."""
import argparse
import ctypes as C
import json
from pathlib import Path

from experiments.tma_reduction_layout.phase8.launch import Driver

KINDS = ('ordinary_contiguous', 'ordinary_conflict32', 'matrix_x4')
BLOCKS = 128


def ptx(kind, cc):
    assert kind in KINDS
    matrix = kind == 'matrix_x4'
    init = '''
    mov.u32 %i, %lane;
init_loop:
    mad.lo.u32 %addr, %i, 2, %base;
    st.shared.b16 [%addr], 0x3f80;
    add.u32 %i, %i, 32;
    setp.lt.u32 %p, %i, 256;
    @%p bra init_loop;
''' if matrix else '''
    mov.u32 %i, %lane;
init_loop:
    mad.lo.u32 %addr, %i, 4, %base;
    add.u32 %v0, %i, 1;
    st.shared.b32 [%addr], %v0;
    add.u32 %i, %i, 32;
    setp.lt.u32 %p, %i, 1024;
    @%p bra init_loop;
'''
    load = '''
    mad.lo.u32 %addr, %lane, 16, %base;
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%v0, %v1, %v2, %v3}, [%addr];
''' if matrix else f'''
    mad.lo.u32 %addr, %lane, {128 if kind == 'ordinary_conflict32' else 4}, %base;
    ld.shared.b32 %v0, [%addr];
    mov.b32 %v1, %v0;
    mov.b32 %v2, %v0;
    mov.b32 %v3, %v0;
'''
    return f'''.version 9.0
.target sm_{cc}
.address_size 64
.visible .entry {kind}(.param .u64 output) .reqntid 32,1,1 {{
    .shared .align 16 .b8 tile[4096];
    .reg .pred %p;
    .reg .b32 %lane, %bid, %i, %base, %addr, %v0, %v1, %v2, %v3, %offset;
    .reg .b64 %out, %dst, %byte;
    ld.param.u64 %out, [output];
    mov.u32 %lane, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %base, tile;
{init}
    bar.sync 0;
{load}
    mad.lo.u32 %offset, %bid, 32, %lane;
    mul.wide.u32 %byte, %offset, 16;
    add.u64 %dst, %out, %byte;
    st.global.v4.b32 [%dst], {{%v0, %v1, %v2, %v3}};
    ret;
}}
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--kind', choices=KINDS, required=True)
    args = parser.parse_args()
    d = Driver()
    context = C.c_void_p()
    d.call('cuCtxCreate_v2', [C.POINTER(C.c_void_p),C.c_uint,C.c_int], C.byref(context),0,d.device)
    path = Path(args.root) / (args.kind + '.cubin')
    from experiments.tma_reduction_layout.phase11.common import sha
    data = path.read_bytes()
    digest = sha(data)
    buffer = C.create_string_buffer(data)
    module, function = C.c_void_p(), C.c_void_p()
    d.call('cuModuleLoadData', [C.POINTER(C.c_void_p), C.c_void_p], C.byref(module), C.cast(buffer, C.c_void_p))
    d.call('cuModuleGetFunction', [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
           C.byref(function), module, args.kind.encode())
    pointer = C.c_uint64()
    d.call('cuMemAlloc_v2', [C.POINTER(C.c_uint64),C.c_size_t], C.byref(pointer),BLOCKS*32*4*4)
    params = (C.c_void_p * 1)(C.addressof(pointer))
    stream = C.c_void_p(0)
    def launch():
        assert sha(path.read_bytes()) == digest == sha(buffer.raw[:-1])
        d.call('cuLaunchKernel', [C.c_void_p] + [C.c_uint] * 7 + [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p],
               function, BLOCKS, 1, 1, 32, 1, 1, 0, stream, params, None)
        d.call('cuStreamSynchronize', [C.c_void_p], stream)
    launch()
    d.call('cuProfilerStart', [])
    launch()
    d.call('cuProfilerStop', [])
    out = (C.c_uint32 * (BLOCKS*32*4))()
    d.call('cuMemcpyDtoH_v2', [C.c_void_p,C.c_uint64,C.c_size_t], C.cast(out,C.c_void_p),pointer,C.sizeof(out))
    stride = 32 if args.kind == 'ordinary_conflict32' else 1
    assert all(value == (0x3f803f80 if args.kind == 'matrix_x4' else (i//4)%32*stride+1)
               for i,value in enumerate(out)), args.kind
    print('FIXTURE_BINDING ' + json.dumps({'kind':args.kind, 'CUBIN_SHA256':digest,
          'grid':BLOCKS, 'threads':32, 'expected_dynamic_warp_load_instructions':BLOCKS,
          'correctness':True, 'environment':d.environment(), 'purpose':'instrument calibration only'}))
    d.call('cuModuleUnload', [C.c_void_p], module)
    d.call('cuMemFree_v2', [C.c_uint64], pointer)
    d.call('cuCtxDestroy_v2', [C.c_void_p], context)


if __name__ == '__main__':
    main()
