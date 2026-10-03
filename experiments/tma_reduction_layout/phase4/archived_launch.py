"""Launch frozen CUBINs through libcuda. No Triton/JIT/compilation imports."""
from collections import Counter
import ctypes as C
from pathlib import Path
import re
from experiments.tma_reduction_layout.phase4.exact_cuda import ExactCUDA, sha


class Driver(ExactCUDA):
    def __init__(self):
        self.counts = Counter()
        self.codes = set()
        self.functions = {}
        super().__init__()

    def call(self, name, types, *args):
        if name not in self.functions:
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = types, C.c_int
            self.functions[name] = fn
        code = self.functions[name](*args)
        self.counts[name] += 1
        self.codes.add(code)
        if name not in {"cuEventRecord", "cuEventSynchronize", "cuEventElapsedTime", "cuLaunchKernel"}:
            self.calls.append({"api": name, "return_code": code})
        if code:
            raise RuntimeError(f"{name}: CUDA return code {code}")

    def tensor_map(self, pointer, m, n, swizzle_bytes):
        owner = C.create_string_buffer(128 + 63)
        address = (C.addressof(owner) + 63) & ~63
        shape = (C.c_uint64 * 3)(n, m, 65536)
        strides = (C.c_uint64 * 2)(n * 2, m * n * 2)
        box = (C.c_uint32 * 3)(min(n, swizzle_bytes // 2), m, 1)
        element_strides = (C.c_uint32 * 3)(1, 1, 1)
        # CUDA host BF16 enum=9, L2_128B=2, OOB zero=0. Same rank/box,
        # swizzle and byte strides as the frozen Triton CUDA host descriptor.
        self.call("cuTensorMapEncodeTiled", [C.c_void_p, C.c_int, C.c_uint32, C.c_void_p,
            C.POINTER(C.c_uint64), C.POINTER(C.c_uint64), C.POINTER(C.c_uint32),
            C.POINTER(C.c_uint32), C.c_int, C.c_int, C.c_int, C.c_int],
            C.c_void_p(address), 9, 3, C.c_void_p(pointer), shape, strides, box,
            element_strides, 0, {32: 1, 64: 2, 128: 3}[swizzle_bytes], 2, 0)
        return owner, address

    def event(self):
        event = C.c_void_p()
        self.call("cuEventCreate", [C.POINTER(C.c_void_p), C.c_uint], C.byref(event), 0)
        return event

    def record(self, event, stream):
        self.call("cuEventRecord", [C.c_void_p, C.c_void_p], event, stream)

    def synchronize_event(self, event):
        self.call("cuEventSynchronize", [C.c_void_p], event)

    def elapsed_us(self, first, last):
        value = C.c_float()
        self.call("cuEventElapsedTime", [C.POINTER(C.c_float), C.c_void_p, C.c_void_p], C.byref(value), first, last)
        return float(value.value) * 1000.0


class Loaded:
    def __init__(self, driver, archive_root, binding, ptx):
        self.driver, self.binding = driver, binding
        self.path = Path(archive_root) / binding["archive_path"]
        data = self.path.read_bytes()
        self.digest = sha(data)
        if self.digest != binding["archive_sha256"] or not data.startswith(b"\x7fELF"):
            raise RuntimeError("ABORT BEFORE TIMING: archived CUBIN SHA mismatch")
        self.buffer = C.create_string_buffer(data)
        self.module, self.function = C.c_void_p(), C.c_void_p()
        driver.call("cuModuleLoadData", [C.POINTER(C.c_void_p), C.c_void_p], C.byref(self.module), C.cast(self.buffer, C.c_void_p))
        driver.call("cuModuleGetFunction", [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
                    C.byref(self.function), self.module, binding["metadata"]["function_name"].encode())
        self.guards = self.launches = 0
        self.harness = binding["metadata"]["harness"]
        entry = re.search(r'\.visible \.entry\s+\w+\((.*?)\)\s*\.reqntid', ptx, re.S)
        if not entry:
            raise RuntimeError("Frozen PTX ABI missing")
        params = [line.strip().rstrip(",") for line in entry[1].splitlines() if ".param" in line]
        types = ["descriptor128" if ".b8" in x and "[128]" in x else "u32" if ".u32" in x else "u64" if ".u64" in x else "UNKNOWN" for x in params]
        expected = ["u64", "u64", "u32", "u32", "u64", "u64"] if self.harness == "canonical" else ["descriptor128", "u32", "u32", "u32", "u64", "u64", "u64", "u64", "u32", "u64", "u64"]
        if types != expected:
            raise RuntimeError(f"Frozen ABI unsupported: {types}")
        self.abi = types

    def parameters(self, tensors, r, descriptor):
        x, out, scratch = tensors
        _, m, n = x.shape
        if self.harness == "canonical":
            values = [C.c_uint64(x.data_ptr()), C.c_uint64(out.data_ptr()), C.c_uint32(m * n),
                      C.c_uint32(n), C.c_uint64(scratch.data_ptr()), C.c_uint64(0)]
            addresses = [C.addressof(v) for v in values]
        else:
            values = [C.c_uint32(65536), C.c_uint32(m), C.c_uint32(n), C.c_uint64(m * n),
                      C.c_uint64(n), C.c_uint64(1), C.c_uint64(out.data_ptr()), C.c_uint32(r),
                      C.c_uint64(0), C.c_uint64(0)]
            addresses = [descriptor[1]] + [C.addressof(v) for v in values]
        return values, (C.c_void_p * len(addresses))(*addresses)

    def guard(self):
        # Do this BEFORE recording the start event, for every launch.
        digest = sha(self.path.read_bytes())
        if digest != self.binding["archive_sha256"] or sha(self.buffer.raw[:-1]) != digest:
            raise RuntimeError("ABORT BEFORE TIMING: runtime/archive CUBIN SHA mismatch")
        self.guards += 1
        return digest

    def launch(self, b_run, params, stream):
        if b_run not in (16384, 32768, 65536):
            raise RuntimeError("Runtime B violates frozen domain")
        meta = self.binding["metadata"]
        self.driver.call("cuLaunchKernel", [C.c_void_p, C.c_uint, C.c_uint, C.c_uint,
            C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p],
            self.function, b_run, 1, 1, meta["num_warps"] * 32, 1, 1,
            meta["dynamic_smem_bytes"], stream, params, None)
        self.launches += 1

    def record_binding(self):
        return {"archive_path": self.binding["archive_path"], "archive_sha256": self.digest,
            "runtime_loaded_cubin_sha256": sha(self.buffer.raw[:-1]),
            "function_name": self.binding["metadata"]["function_name"],
            "module_identity": hex(self.module.value), "function_identity": hex(self.function.value),
            "load_api": "cuModuleLoadData(archived_file_bytes)", "ABI": self.abi,
            "sha_guard_count": self.guards, "launch_count": self.launches}
