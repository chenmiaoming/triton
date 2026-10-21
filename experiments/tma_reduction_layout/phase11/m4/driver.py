"""Existing frozen-CUBIN driver with metadata-driven, legal unswizzled TMA boxes."""
import ctypes as C
from experiments.tma_reduction_layout.phase11.d1.driver import Loaded as Parent
from experiments.tma_reduction_layout.phase11.d1.driver import Driver, Pointer, Graph, allocate, upload


class Loaded(Parent):
    def descriptor(self, x):
        owner = C.create_string_buffer(191)
        address = (C.addressof(owner)+63)&~63
        spec = self.metadata['tensordesc_meta']
        assert len(spec) == 1
        spec = spec[0]
        swizzle = spec['swizzle']
        assert swizzle in (0,1,2,3)
        assert self.metadata['dtype'] == 'bfloat16'
        rows = self.metadata.get('global_rows',8192)
        shape = (C.c_uint64*3)(8192,rows,1)
        strides = (C.c_uint64*2)(8192*2,8192*rows*2)
        block = spec['block_size']
        assert block[0] == 1 and block[1] == self.metadata['M']
        box = (C.c_uint32*3)(block[2],block[1],1)
        steps = (C.c_uint32*3)(1,1,1)
        self.driver.call('cuTensorMapEncodeTiled',[C.c_void_p,C.c_int,C.c_uint32,C.c_void_p,
            C.POINTER(C.c_uint64),C.POINTER(C.c_uint64),C.POINTER(C.c_uint32),C.POINTER(C.c_uint32),
            C.c_int,C.c_int,C.c_int,C.c_int],C.c_void_p(address),9,3,C.c_void_p(x.data_ptr()),
            shape,strides,box,steps,0,swizzle,2,0)
        return owner,address
