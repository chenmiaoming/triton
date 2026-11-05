"""C1 descriptor ABI and exact-binary launches. Never imports Triton."""
import ctypes as C
import re
from pathlib import Path
from experiments.tma_reduction_layout.phase11.launch import Loaded as DescriptorLoaded
from experiments.tma_reduction_layout.phase8.launch import Driver
from experiments.tma_reduction_layout.phase11.c1.common import sha


class Loaded(DescriptorLoaded):
    def __init__(self, driver, root, meta):
        self.driver, self.metadata = driver, meta
        self.path = Path(root) / meta['tag'].replace(':', '/') / 'kernel.cubin'
        data = self.path.read_bytes()
        self.digest = sha(data)
        assert data.startswith(b'\x7fELF') and self.digest == meta['cubin_SHA256']
        self.buffer = C.create_string_buffer(data)
        self.module, self.function = C.c_void_p(), C.c_void_p()
        driver.call('cuModuleLoadData', [C.POINTER(C.c_void_p), C.c_void_p],
                    C.byref(self.module), C.cast(self.buffer, C.c_void_p))
        driver.call('cuModuleGetFunction', [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
                    C.byref(self.function), self.module, meta['function_name'].encode())
        if meta['dynamic_smem_bytes'] > 49152:
            driver.call('cuFuncSetAttribute', [C.c_void_p, C.c_int, C.c_int],
                        self.function, 8, meta['dynamic_smem_bytes'])
        ptx = self.path.with_suffix('.ptx').read_text()
        entry = re.search(r'\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid', ptx, re.S)
        assert entry, meta['tag']
        self.abi = ['descriptor128' if '.b8' in p and '[128]' in p else 'u32' if '.u32' in p
                    else 'u64' if '.u64' in p else 'UNKNOWN' for p in entry[1].splitlines() if '.param' in p]
        expected = ['descriptor128'] + ['u32'] * 3 + ['u64'] * 4 + ['u32'] + ['u64'] * 2
        assert self.abi == expected, (meta['tag'], self.abi, expected)
        self.guards = self.launches = 0

    def parameters(self, x, out, scratch, k):
        assert k in (1, 16, 64, 256)
        owner, address = self.descriptor(x)
        values = [C.c_uint32(1), C.c_uint32(8192), C.c_uint32(8192),
                  C.c_uint64(8192 * 8192), C.c_uint64(8192), C.c_uint64(1),
                  C.c_uint64(out.data_ptr()), C.c_uint32(k),
                  C.c_uint64(scratch.data_ptr()), C.c_uint64(0)]
        addresses = [address] + [C.addressof(v) for v in values]
        return (owner, values), (C.c_void_p * len(addresses))(*addresses), [address] + [v.value for v in values]

    def launch(self, grid, params, stream):
        assert 0 < grid <= (8192 // self.metadata['M']) * (8192 // self.metadata['N'])
        self.driver.call('cuLaunchKernel', [C.c_void_p] + [C.c_uint] * 7 +
                         [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p], self.function,
                         grid, 1, 1, self.metadata['num_warps'] * 32, 1, 1,
                         self.metadata['dynamic_smem_bytes'], stream, params, None)
        self.launches += 1
