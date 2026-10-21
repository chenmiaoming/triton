"""Checked CUDA driver calls against bytes read from the archived CUBIN."""
import ctypes as C
import hashlib
import uuid
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ExactCUDA:
    def __init__(self):
        self.lib = C.CDLL("libcuda.so.1")
        self.calls = []
        self.call("cuInit", [C.c_uint], 0)
        self.device = C.c_int()
        self.call("cuDeviceGet", [C.POINTER(C.c_int), C.c_int], C.byref(self.device), 0)

    def call(self, name, types, *args):
        fn = getattr(self.lib, name)
        fn.argtypes, fn.restype = types, C.c_int
        result = fn(*args)
        self.calls.append({"api": name, "return_code": result})
        if result:
            raise RuntimeError(f"{name}: CUDA return code {result}")

    def attr(self, code):
        v = C.c_int()
        self.call("cuDeviceGetAttribute", [C.POINTER(C.c_int), C.c_int, C.c_int],
                  C.byref(v), code, self.device)
        return v.value

    def environment(self):
        raw = (C.c_ubyte * 16)()
        self.call("cuDeviceGetUuid_v2", [C.c_void_p, C.c_int], C.byref(raw), self.device)
        version = C.c_int()
        self.call("cuDriverGetVersion", [C.POINTER(C.c_int)], C.byref(version))
        attrs = {"max_threads_per_sm": 39, "max_threads_per_block": 1,
                 "max_registers_per_block": 12, "max_registers_per_sm": 82,
                 "max_shared_memory_per_sm": 81, "warp_size": 10,
                 "max_shared_memory_per_block_optin": 97}
        self.limits = {key: self.attr(code) for key, code in attrs.items()}
        return {"gpu_uuid": "GPU-" + str(uuid.UUID(bytes=bytes(raw))),
                "driver_API_version": version.value, "device_limits": self.limits,
                "checked_cuda_calls": list(self.calls)}

    def occupancy(self, path, name, dynamic, warps, resource_sha):
        # Read the file after export; never query an equivalent recompile.
        archived = Path(path).read_bytes()
        digest = sha(archived)
        buffer = C.create_string_buffer(archived)
        module, function = C.c_void_p(), C.c_void_p()
        first = len(self.calls)
        self.call("cuModuleLoadData", [C.POINTER(C.c_void_p), C.c_void_p],
                  C.byref(module), C.cast(buffer, C.c_void_p))
        try:
            self.call("cuModuleGetFunction", [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
                      C.byref(function), module, name.encode())
            attrs = {}
            for key, code in {"num_regs": 4, "static_smem_bytes": 1, "local_bytes": 3}.items():
                v = C.c_int()
                self.call("cuFuncGetAttribute", [C.POINTER(C.c_int), C.c_int, C.c_void_p],
                          C.byref(v), code, function)
                attrs[key] = v.value
            # Match Triton's dynamic SMEM opt-in for large allocations.
            if dynamic > 49152:
                self.call("cuFuncSetAttribute", [C.c_void_p, C.c_int, C.c_int], function, 8, dynamic)
            blocks = []
            for size in (dynamic, 0):
                v = C.c_int()
                self.call("cuOccupancyMaxActiveBlocksPerMultiprocessor",
                          [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t],
                          C.byref(v), function, warps * 32, size)
                blocks.append(v.value)
        finally:
            self.call("cuModuleUnload", [C.c_void_p], module)
        if sha(Path(path).read_bytes()) != digest:
            raise RuntimeError("CUBIN changed during occupancy query")
        return {"query_api": "cuOccupancyMaxActiveBlocksPerMultiprocessor",
                "queried_cubin_sha256": digest, "cubin_sha256": digest,
                "load_method": "cuModuleLoadData(buffer_of_archived_file_bytes)",
                "function_name": name, **attrs, "resource_sha256": resource_sha,
                "dynamic_smem_bytes": dynamic, "num_warps": warps, "block_threads": warps * 32,
                "blocks_per_sm_actual_dynamic_smem": blocks[0],
                "blocks_per_sm_zero_dynamic_smem": blocks[1],
                "active_warps_per_sm": blocks[0] * warps, "device_limits": self.limits,
                "checked_cuda_calls": self.calls[first:]}
