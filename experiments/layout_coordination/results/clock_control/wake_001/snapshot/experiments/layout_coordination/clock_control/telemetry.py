"""Read-only NVML telemetry for matched, unlocked-clock GPU measurements.

Importing this module does not load a CUDA/NVML library or initialize a device.
Every queried field retains its NVML return code; an unavailable reading is
``None``, never zero.  Sampling is observational and does not lock clocks.
"""

from __future__ import annotations

import copy
import ctypes
import os
from pathlib import Path
import platform
import threading
import time
from typing import Any


_UINT = ctypes.c_uint
_UINT64 = ctypes.c_ulonglong
_HANDLE = ctypes.c_void_p
_SUCCESS = 0
_INSUFFICIENT_SIZE = 7
_CAP_SYS_ADMIN = 21


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", _UINT), ("memory", _UINT)]


class NVML:
    """A UUID-bound NVML reader.  Only library/init/handle failures raise.

    Call ``close`` after stopping all samplers.  No setting, reset, persistence,
    power-limit, or API-restriction mutation is exposed by this class.
    """

    def __init__(self, uuid: str):
        self.uuid = str(uuid)
        self._lock = threading.RLock()
        self._closed = False
        self._initialized = False
        self._functions: dict[str, Any] = {}
        self._lib = ctypes.CDLL("libnvidia-ml.so.1")
        self._handle = _HANDLE()
        init_name = "nvmlInit_v2" if hasattr(self._lib, "nvmlInit_v2") else "nvmlInit"
        init = self._function(init_name, [])
        if init is None:
            raise RuntimeError("NVML has neither nvmlInit_v2 nor nvmlInit")
        rc = int(init())
        if rc != _SUCCESS:
            raise RuntimeError(f"{init_name}: rc={rc}: {self._error_string(rc)}")
        self._initialized = True
        try:
            get_handle = self._function(
                "nvmlDeviceGetHandleByUUID", [ctypes.c_char_p, ctypes.POINTER(_HANDLE)]
            )
            if get_handle is None:
                raise RuntimeError("nvmlDeviceGetHandleByUUID is unavailable")
            rc = int(get_handle(self.uuid.encode("ascii"), ctypes.byref(self._handle)))
            if rc != _SUCCESS:
                raise RuntimeError(
                    f"nvmlDeviceGetHandleByUUID({self.uuid}): "
                    f"rc={rc}: {self._error_string(rc)}"
                )
        except Exception:
            self.close()
            raise

    def _function(self, name: str, argtypes: list[Any]):
        if name in self._functions:
            return self._functions[name]
        try:
            function = getattr(self._lib, name)
        except AttributeError:
            function = None
        if function is not None:
            function.argtypes = argtypes
            function.restype = ctypes.c_int
        self._functions[name] = function
        return function

    def _error_string(self, rc: int) -> str:
        try:
            function = self._lib.nvmlErrorString
            function.argtypes = [ctypes.c_int]
            function.restype = ctypes.c_char_p
            value = function(rc)
            return value.decode("utf-8", errors="replace") if value else "no error string"
        except Exception as error:
            return f"error string unavailable ({type(error).__name__}: {error})"

    def _query(self, name: str, args: list[Any], argtypes: list[Any], read_value,
               unit: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {"api": name, "rc": None, "value": None}
        if unit is not None:
            result["unit"] = unit
        try:
            with self._lock:
                if self._closed:
                    result["error"] = "NVML reader is closed"
                    return result
                function = self._function(name, argtypes)
                if function is None:
                    result["error"] = "symbol unavailable"
                    return result
                rc = int(function(*args))
                result["rc"] = rc
                if rc == _SUCCESS:
                    result["value"] = read_value()
                else:
                    result["error"] = self._error_string(rc)
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        return result

    def _uint(self, name: str, *extra: int, unit: str | None = None,
              output_type=_UINT) -> dict[str, Any]:
        output = output_type()
        return self._query(
            name, [self._handle, *extra, ctypes.byref(output)],
            [_HANDLE, *([_UINT] * len(extra)), ctypes.POINTER(output_type)],
            lambda: int(output.value), unit,
        )

    def _string(self, name: str, *, device: bool = False) -> dict[str, Any]:
        output = ctypes.create_string_buffer(256)
        args = ([self._handle] if device else []) + [output, len(output)]
        argtypes = ([_HANDLE] if device else []) + [ctypes.POINTER(ctypes.c_char), _UINT]
        return self._query(name, args, argtypes,
                           lambda: output.value.decode("utf-8", errors="replace"))

    def _event_reasons(self, supported: bool = False) -> dict[str, Any]:
        qualifier = "Supported" if supported else "Current"
        names = [f"nvmlDeviceGet{qualifier}ClocksEventReasons",
                 f"nvmlDeviceGet{qualifier}ClocksThrottleReasons"]
        attempts = []
        for name in names:
            result = self._uint(name, output_type=_UINT64)
            attempts.append(dict(result))
            if result["rc"] == _SUCCESS:
                result["hex"] = f"0x{result['value']:016x}"
                break
        if len(attempts) > 1:
            result["attempts"] = attempts
        return result

    def sample(self) -> dict[str, Any]:
        """Read current telemetry; each field includes its own rc/error.

        The row spans multiple driver queries rather than an atomic snapshot.
        Monotonic start/end delimit that interval for later window matching.
        On H100, PowerUsage is a one-second average, and utilization has its own
        driver sampling window (roughly 1/6 to 1 second). Polling every 10 ms
        does not turn either reading into an instantaneous/per-kernel measure.
        """
        started = time.monotonic_ns()
        with self._lock:
            result = {
                "monotonic_ns": started,
                "graphics_clock_mhz": self._uint("nvmlDeviceGetClockInfo", 0, unit="MHz"),
                "sm_clock_mhz": self._uint("nvmlDeviceGetClockInfo", 1, unit="MHz"),
                "memory_clock_mhz": self._uint("nvmlDeviceGetClockInfo", 2, unit="MHz"),
                "power_usage_mw": self._uint("nvmlDeviceGetPowerUsage", unit="mW"),
                "power_limit_mw": self._uint("nvmlDeviceGetPowerManagementLimit", unit="mW"),
                "temperature_c": self._uint("nvmlDeviceGetTemperature", 0, unit="C"),
                "pstate": self._uint("nvmlDeviceGetPerformanceState"),
                "clock_event_reasons": self._event_reasons(),
            }
            utilization = _Utilization()
            utilization_result = self._query(
                "nvmlDeviceGetUtilizationRates", [self._handle, ctypes.byref(utilization)],
                [_HANDLE, ctypes.POINTER(_Utilization)],
                lambda: {"gpu": int(utilization.gpu), "memory": int(utilization.memory)},
            )
            for field, member in [("gpu_utilization_percent", "gpu"),
                                  ("memory_utilization_percent", "memory")]:
                reading = dict(utilization_result)
                reading["unit"] = "%"
                value = utilization_result["value"]
                reading["value"] = value[member] if value is not None else None
                result[field] = reading
        result["sample_end_monotonic_ns"] = time.monotonic_ns()
        result["query_duration_ns"] = result["sample_end_monotonic_ns"] - started
        return result

    def _clock_list(self, name: str, *extra: int) -> dict[str, Any]:
        """Perform NVML's size-then-buffer list query without hiding either rc."""
        count = _UINT(0)
        argtypes = [_HANDLE, *([_UINT] * len(extra)),
                    ctypes.POINTER(_UINT), ctypes.POINTER(_UINT)]
        first = self._query(name, [self._handle, *extra, ctypes.byref(count), None],
                            argtypes, lambda: [])
        if first["rc"] not in (_SUCCESS, _INSUFFICIENT_SIZE):
            return first
        count_query = {"rc": first["rc"], "count": int(count.value)}
        if not count.value and first["rc"] != _SUCCESS:
            first["count_query"] = count_query
            return first
        if not count.value:
            return {"api": name, "rc": _SUCCESS, "value": [], "unit": "MHz",
                    "count_query": count_query}
        if count.value > 4096:
            return {"api": name, "rc": None, "value": None,
                    "error": "reported clock list exceeds safety bound 4096",
                    "count_query": count_query}
        buffer = (_UINT * count.value)()
        capacity = len(buffer)
        result = self._query(name, [self._handle, *extra, ctypes.byref(count), buffer],
                             argtypes, lambda: [int(buffer[i]) for i in range(min(count.value, capacity))],
                             "MHz")
        result["count_query"] = count_query
        result["returned_count"] = int(count.value)
        return result

    def environment(self) -> dict[str, Any]:
        """Read device/driver capabilities and restrictions, without setters."""
        with self._lock:
            result: dict[str, Any] = {
                "requested_uuid": self.uuid,
                "device_uuid": self._string("nvmlDeviceGetUUID", device=True),
                "device_name": self._string("nvmlDeviceGetName", device=True),
                "driver_version": self._string("nvmlSystemGetDriverVersion"),
                "nvml_version": self._string("nvmlSystemGetNVMLVersion"),
                "persistence_mode": self._uint("nvmlDeviceGetPersistenceMode"),
                "compute_mode": self._uint("nvmlDeviceGetComputeMode"),
                "virtualization_mode": self._uint("nvmlDeviceGetVirtualizationMode"),
                "supported_clock_event_reasons": self._event_reasons(supported=True),
                "api_restrictions": {
                    "set_application_clocks": self._uint("nvmlDeviceGetAPIRestriction", 0),
                    "set_auto_boosted_clocks": self._uint("nvmlDeviceGetAPIRestriction", 1),
                },
                "clock_readbacks": {},
            }
            for name in ["nvmlSystemGetCudaDriverVersion_v2", "nvmlSystemGetCudaDriverVersion"]:
                version = ctypes.c_int()
                reading = self._query(name, [ctypes.byref(version)],
                                      [ctypes.POINTER(ctypes.c_int)], lambda: int(version.value))
                result["cuda_driver_api_version"] = reading
                if reading["rc"] == _SUCCESS:
                    break
            result["cuda_version_note"] = (
                "NVML reports the CUDA version supported by the driver; "
                "this is not a measurement of the installed CUDA runtime version."
            )
            for clock_name, clock_type in [("graphics", 0), ("sm", 1), ("memory", 2)]:
                result["clock_readbacks"][clock_name] = {
                    "current": self._uint("nvmlDeviceGetClockInfo", clock_type, unit="MHz"),
                    "maximum": self._uint("nvmlDeviceGetMaxClockInfo", clock_type, unit="MHz"),
                    "applications": self._uint("nvmlDeviceGetApplicationsClock", clock_type, unit="MHz"),
                    "default_applications": self._uint("nvmlDeviceGetDefaultApplicationsClock", clock_type, unit="MHz"),
                }
            minimum, maximum = _UINT(), _UINT()
            result["power_limit_constraints_mw"] = self._query(
                "nvmlDeviceGetPowerManagementLimitConstraints",
                [self._handle, ctypes.byref(minimum), ctypes.byref(maximum)],
                [_HANDLE, ctypes.POINTER(_UINT), ctypes.POINTER(_UINT)],
                lambda: {"minimum": int(minimum.value), "maximum": int(maximum.value)}, "mW",
            )
            result["power_limit_default_mw"] = self._uint(
                "nvmlDeviceGetPowerManagementDefaultLimit", unit="mW"
            )
            memory = self._clock_list("nvmlDeviceGetSupportedMemoryClocks")
            result["supported_memory_clocks_mhz"] = memory
            result["supported_graphics_clocks_by_memory_mhz"] = [
                {"memory_clock_mhz": clock,
                 "graphics_clocks_mhz": self._clock_list("nvmlDeviceGetSupportedGraphicsClocks", clock)}
                for clock in (memory["value"] or [])
            ]
            return result

    def close(self) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                return {"api": "nvmlShutdown", "rc": None, "value": None,
                        "note": "already closed; no additional shutdown call"}
            if self._initialized:
                result = self._query("nvmlShutdown", [], [], lambda: None)
            else:
                result = {"api": "nvmlShutdown", "rc": None, "value": None,
                          "note": "reader was not initialized"}
            self._closed = True
            self._initialized = False
            return result

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class Sampler:
    """A daemon sampler with explicit phase/visit/case labels and no disk I/O."""

    def __init__(self, nvml: NVML, period_s: float = 0.010):
        if not 0 < float(period_s) < float("inf"):
            raise ValueError("period_s must be positive and finite")
        self.nvml = nvml
        self.period_s = float(period_s)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._context: dict[str, Any] = {}
        self._context_sequence = 0
        self._rows: list[dict[str, Any]] = []

    def set_context(self, **labels: Any) -> None:
        """Merge labels; rows snapshot them immediately before each NVML query."""
        with self._lock:
            self._context.update(copy.deepcopy(labels))
            self._context_sequence += 1

    def _run(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                context = copy.deepcopy(self._context)
                context_sequence = self._context_sequence
            try:
                row = self.nvml.sample()
            except Exception as error:
                # Preserve an unexpected Python-side failure instead of silently
                # losing the background thread. NVML field errors stay per-field.
                row = {"monotonic_ns": time.monotonic_ns(),
                       "sampler_error": f"{type(error).__name__}: {error}"}
            with self._lock:
                row.update(sequence=len(self._rows), context_sequence=context_sequence,
                           context=context)
                self._rows.append(row)
            deadline += self.period_s
            now = time.monotonic()
            if deadline < now:
                # Skip missed deadlines rather than making a burst of stale reads.
                deadline += (int((now - deadline) / self.period_s) + 1) * self.period_s
            self._stop.wait(max(0.0, deadline - time.monotonic()))

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("Sampler instances can only be started once")
            self._thread = threading.Thread(target=self._run, name="nvml-readonly-sampler", daemon=True)
            self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join()
        with self._lock:
            return copy.deepcopy(self._rows)


def _read_file(path: str) -> dict[str, Any]:
    try:
        return {"path": path, "ok": True,
                "text": Path(path).read_text(encoding="utf-8", errors="replace")}
    except OSError as error:
        return {"path": path, "ok": False, "errno": error.errno,
                "error": f"{type(error).__name__}: {error}"}


def collect_env_readonly(nvml: NVML | None = None) -> dict[str, Any]:
    """Capture process/namespace/driver readbacks; never inspect credentials.

    Pass an already UUID-bound reader to include NVML capability readbacks. With
    no reader this function does not load NVML, select a GPU, or launch work.
    CAP_SYS_ADMIN is a namespace-local observation, not proof of host permission.
    """
    paths = ["/proc/self/status", "/proc/self/uid_map", "/proc/self/gid_map",
             "/proc/driver/nvidia/params", "/proc/driver/nvidia/version",
             "/sys/module/nvidia/version"]
    files = {path: _read_file(path) for path in paths}
    status: dict[str, str] = {}
    for line in files["/proc/self/status"].get("text", "").splitlines():
        key, separator, value = line.partition(":")
        if separator:
            status[key] = value.strip()
    cap_eff = None
    try:
        cap_eff = int(status["CapEff"], 16)
    except (KeyError, ValueError):
        pass
    result: dict[str, Any] = {
        "captured_monotonic_ns": time.monotonic_ns(),
        "process": {"pid": os.getpid(), "ppid": os.getppid(),
                    "uid": os.getuid(), "euid": os.geteuid(),
                    "gid": os.getgid(), "egid": os.getegid(),
                    "groups": os.getgroups(), "python_version": platform.python_version(),
                    "uname": list(platform.uname())},
        "files": files,
        "status_identity_and_security": {key: status.get(key) for key in [
            "Name", "Pid", "PPid", "Uid", "Gid", "NSpid", "CapInh", "CapPrm",
            "CapEff", "CapBnd", "CapAmb", "NoNewPrivs", "Seccomp", "Seccomp_filters"]},
        "effective_capability_mask": cap_eff,
        "cap_sys_admin_bit": _CAP_SYS_ADMIN,
        "cap_sys_admin_effective": None if cap_eff is None else bool(cap_eff & (1 << _CAP_SYS_ADMIN)),
        "capability_note": "Effective capabilities are relative to the process user namespace; host driver permissions may differ.",
        "read_only": True,
    }
    if nvml is not None:
        try:
            result["nvml"] = nvml.environment()
        except Exception as error:
            result["nvml"] = {"error": f"{type(error).__name__}: {error}"}
    return result
