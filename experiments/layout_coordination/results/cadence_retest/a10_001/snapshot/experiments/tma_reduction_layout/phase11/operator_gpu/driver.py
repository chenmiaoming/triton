"""Driver-only archived pointer-kernel execution; no compiler imports."""
import ctypes as C
from pathlib import Path
import re
from experiments.tma_reduction_layout.phase8.launch import Driver
from .common import sha


class Device:
    def __init__(self):
        self.driver = Driver()
        d = self.driver
        self.context, self.stream = C.c_void_p(), C.c_void_p()
        d.call('cuCtxCreate_v2', [C.POINTER(C.c_void_p), C.c_uint, C.c_int], C.byref(self.context), 0, d.device)
        d.call('cuStreamCreate', [C.POINTER(C.c_void_p), C.c_uint], C.byref(self.stream), 0)
        name = C.create_string_buffer(256)
        d.call('cuDeviceGetName', [C.c_void_p, C.c_int, C.c_int], name, 256, d.device)
        self.environment = {**d.environment(), 'name': name.value.decode(),
                            'cc': [d.attr(75), d.attr(76)], 'SM_count': d.attr(16)}
        assert self.environment['cc'] == [9, 0] and 'H100' in self.environment['name']
        self.allocations = []
        self.kernels = []

    def allocate(self, size):
        pointer = C.c_uint64()
        self.driver.call('cuMemAlloc_v2', [C.POINTER(C.c_uint64), C.c_size_t], C.byref(pointer), max(1, size))
        self.allocations.append(pointer)
        return pointer.value

    def upload(self, address, data):
        owner = C.create_string_buffer(data)
        self.driver.call('cuMemcpyHtoD_v2', [C.c_uint64, C.c_void_p, C.c_size_t], address, owner, len(data))

    def download(self, address, size):
        owner = C.create_string_buffer(size)
        self.driver.call('cuMemcpyDtoH_v2', [C.c_void_p, C.c_uint64, C.c_size_t], owner, address, size)
        return owner.raw

    def synchronize(self):
        self.driver.call('cuStreamSynchronize', [C.c_void_p], self.stream)

    def free_buffers(self):
        self.synchronize()
        for pointer in self.allocations:
            self.driver.call('cuMemFree_v2', [C.c_uint64], pointer)
        self.allocations.clear()

    def close(self):
        self.free_buffers()
        for kernel in self.kernels:
            self.driver.call('cuModuleUnload', [C.c_void_p], kernel.module)
        self.driver.call('cuStreamDestroy_v2', [C.c_void_p], self.stream)
        self.driver.call('cuCtxDestroy_v2', [C.c_void_p], self.context)


class Loaded:
    def __init__(self, device, root, meta):
        self.driver, self.device, self.meta = device.driver, device, meta
        self.path = Path(root) / meta['artifact_directory'] / 'kernel.cubin'
        blob = self.path.read_bytes()
        self.digest = sha(blob)
        assert blob.startswith(b'\x7fELF') and self.digest == meta['cubin_SHA256']
        self.buffer = C.create_string_buffer(blob)
        self.module, self.function = C.c_void_p(), C.c_void_p()
        self.driver.call('cuModuleLoadData', [C.POINTER(C.c_void_p), C.c_void_p], C.byref(self.module), self.buffer)
        self.driver.call('cuModuleGetFunction', [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
                         C.byref(self.function), self.module, meta['function_name'].encode())
        text = self.path.with_suffix('.ptx').read_text()
        match = re.search(r'\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid', text, re.S)
        assert match, meta['tag']
        self.abi = [re.search(r'\.param\s+\.(\w+)', line)[1] for line in match[1].splitlines() if '.param' in line]
        self.attributes = {}
        for key, code in (('n_regs', 4), ('static_smem_bytes', 1), ('local_bytes', 3)):
            value = C.c_int()
            self.driver.call('cuFuncGetAttribute', [C.POINTER(C.c_int), C.c_int, C.c_void_p], C.byref(value), code, self.function)
            self.attributes[key] = value.value
        self.dynamic = meta['dynamic_smem_bytes']
        if self.dynamic > 49152:
            self.driver.call('cuFuncSetAttribute', [C.c_void_p, C.c_int, C.c_int], self.function, 8, self.dynamic)
        capacity = C.c_int()
        self.driver.call('cuOccupancyMaxActiveBlocksPerMultiprocessor',
                         [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t],
                         C.byref(capacity), self.function, meta['num_warps'] * 32, self.dynamic)
        self.attributes['occupancy_capacity_q'] = capacity.value
        self.guards = self.launches = self.graph_launches = self.captures = 0
        device.kernels.append(self)

    def parameters(self, values):
        assert len(values) == len(self.abi), (self.meta['tag'], len(values), self.abi)
        types = {'u64': C.c_uint64, 's64': C.c_int64, 'u32': C.c_uint32, 's32': C.c_int32,
                 'f32': C.c_float, 'f64': C.c_double}
        owners = [types[kind](value) for kind, value in zip(self.abi, values)]
        return owners, (C.c_void_p * len(owners))(*(C.addressof(x) for x in owners))

    def guard(self):
        assert sha(self.path.read_bytes()) == self.digest == sha(self.buffer.raw[:-1])
        self.guards += 1

    def launch(self, grid, params, stream, q='natural'):
        assert q == 'natural'
        grid = (grid, 1, 1) if isinstance(grid, int) else tuple(grid)
        self.driver.call('cuLaunchKernel', [C.c_void_p] + [C.c_uint] * 7 +
                         [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p], self.function,
                         *grid, self.meta['num_warps'] * 32, 1, 1, self.dynamic, stream, params, None)
        self.launches += 1

    def binding(self):
        return {'tag': self.meta['tag'], 'cubin_SHA256': self.digest, 'loaded_SHA256': sha(self.buffer.raw[:-1]),
                'ABI': self.abi, 'attributes': self.attributes, 'guards': self.guards,
                'direct_launches': self.launches, 'graph_launches': self.graph_launches}


class Graph:
    """One-kernel graph, with the same external event nodes as Phase 11 D1."""
    def __init__(self, kernel, grid, params, stream):
        self.kernel, self.driver, self.stream = kernel, kernel.driver, stream
        d = self.driver
        self.first, self.last = d.event(), d.event()
        self.graph, self.executable = C.c_void_p(), C.c_void_p()
        d.call('cuStreamBeginCapture', [C.c_void_p, C.c_int], stream, 0)
        d.call('cuEventRecordWithFlags', [C.c_void_p, C.c_void_p, C.c_uint], self.first, stream, 1)
        kernel.guard()
        kernel.launch(grid, params, stream)
        kernel.captures += 1
        d.call('cuEventRecordWithFlags', [C.c_void_p, C.c_void_p, C.c_uint], self.last, stream, 1)
        d.call('cuStreamEndCapture', [C.c_void_p, C.POINTER(C.c_void_p)], stream, C.byref(self.graph))
        d.call('cuGraphInstantiateWithFlags', [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint64], C.byref(self.executable), self.graph, 0)
        count = C.c_size_t()
        d.call('cuGraphGetNodes', [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)], self.graph, None, C.byref(count))
        nodes = (C.c_void_p * count.value)()
        d.call('cuGraphGetNodes', [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)], self.graph, nodes, C.byref(count))
        self.node_types = []
        for node in nodes:
            value = C.c_int()
            d.call('cuGraphNodeGetType', [C.c_void_p, C.POINTER(C.c_int)], node, C.byref(value))
            self.node_types.append(value.value)
        assert sorted(self.node_types) == [0, 7, 7]

    def run(self):
        self.kernel.guard()
        self.driver.call('cuGraphLaunch', [C.c_void_p, C.c_void_p], self.executable, self.stream)
        self.kernel.graph_launches += 1
        self.driver.call('cuStreamSynchronize', [C.c_void_p], self.stream)
        return self.driver.elapsed_us(self.first, self.last)

    def close(self):
        self.driver.call('cuGraphExecDestroy', [C.c_void_p], self.executable)
        self.driver.call('cuGraphDestroy', [C.c_void_p], self.graph)
        for event in (self.first, self.last):
            self.driver.call('cuEventDestroy_v2', [C.c_void_p], event)
