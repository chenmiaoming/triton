"""Stage A checked ABI with metadata-derived tensor-map boxes and dtypes."""
import ctypes as C
from experiments.tma_reduction_layout.phase10.launch import Loaded as BaseLoaded, Driver as Driver


class Loaded(BaseLoaded):

    def descriptor(self, x):
        owner = C.create_string_buffer(128 + 63)
        address = (C.addressof(owner) + 63) & ~63
        specs = self.metadata['tensordesc_meta']
        assert isinstance(specs, list) and len(specs) == 1
        spec = specs[0]
        assert not spec['fp4_padded'] and not spec['is_im2col']
        assert spec['elem_size'] == x.element_size()
        # Same device-to-host dtype mapping as pinned nvidia/backend/driver.py.
        dtype = {8: 10, 9: 8, 10: 9}.get(spec['elem_type'], spec['elem_type'])
        shape = (C.c_uint64 * 3)(8192, 8192, 1)
        size = x.element_size()
        strides = (C.c_uint64 * 2)(8192 * size, 8192 * 8192 * size)
        box = (C.c_uint32 * 3)(*reversed(spec['block_size']))
        elements = (C.c_uint32 * 3)(1, 1, 1)
        self.driver.call('cuTensorMapEncodeTiled', [
            C.c_void_p, C.c_int, C.c_uint32, C.c_void_p,
            C.POINTER(C.c_uint64),
            C.POINTER(C.c_uint64),
            C.POINTER(C.c_uint32),
            C.POINTER(C.c_uint32), C.c_int, C.c_int, C.c_int, C.c_int
        ], C.c_void_p(address), dtype, 3, C.c_void_p(x.data_ptr()), shape, strides, box, elements, 0, spec['swizzle'],
                         2, 0)
        return owner, address

    def launch(self, params, stream, grid):
        assert grid in [self.metadata['grid'] // f for f in (1, 4, 16)]
        self.driver.call('cuLaunchKernel', [
            C.c_void_p, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_void_p,
            C.POINTER(C.c_void_p), C.c_void_p
        ], self.function, grid, 1, 1, self.metadata['num_warps'] * 32, 1, 1, self.metadata['dynamic_smem_bytes'],
                         stream, params, None)
        self.launches += 1
