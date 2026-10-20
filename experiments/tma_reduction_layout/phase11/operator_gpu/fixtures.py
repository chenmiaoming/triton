"""GPU checks of the existing semantic fixtures; not production benchmarks."""
from pathlib import Path
import sys
import traceback
import numpy as np
from experiments.tma_reduction_layout.phase11.operator_survey import oracles
from experiments.tma_reduction_layout.phase11.operator_survey.cases import launch_grid
from . import common as c
from .driver import Device, Loaded


def pack(array, dtype):
    array = np.asarray(array)
    if dtype == '*bf16':
        return (oracles.bf16_round(array).view(np.uint32) >> 16).astype('<u2').tobytes()
    if dtype == '*fp8e4nv':
        return array.astype(np.uint8).tobytes()
    return array.astype('<f4').tobytes()


def unpack(blob, dtype, shape):
    if dtype == '*bf16':
        return (np.frombuffer(blob, '<u2').astype(np.uint32) << 16).view(np.float32).reshape(shape)
    return np.frombuffer(blob, np.uint8 if dtype == '*fp8e4nv' else '<f4').reshape(shape)


class Buffers:
    def __init__(self, device, case, pattern):
        self.device, self.case = device, case
        self.data = oracles.inputs(case, pattern)
        self.answer = oracles.reference(case, self.data)
        self.rows = {}
        for name, dtype in case['signature'].items():
            array = self.data.get(name, self.answer.get(name, np.zeros(1, np.float32)))
            output = name in self.answer
            blob = pack(array, dtype)
            prefix, suffix = bytes([0xa5]) * 256, bytes([0x5a]) * 256
            content = bytes([0xcd]) * len(blob) if output else blob
            base = device.allocate(len(blob) + 512)
            raw = prefix + content + suffix
            device.upload(base, raw)
            self.rows[name] = {'base': base, 'pointer': base + 256, 'bytes': len(blob),
                               'shape': array.shape, 'dtype': dtype, 'output': output, 'initial': raw}
        self.scratch = device.allocate(256)

    def reset(self):
        for row in self.rows.values():
            if row['output']:
                self.device.upload(row['base'], row['initial'])

    def values(self):
        # JSON object order is not a function ABI. Names are extracted from
        # the original function signature and frozen explicitly as a list.
        names = self.case['runtime_argument_names']
        assert set(names) == set(self.case['signature']) and len(names) == len(self.case['signature'])
        return [self.rows[name]['pointer'] for name in names] + [self.scratch, 0]

    def check(self):
        checks = []
        for name, row in self.rows.items():
            raw = self.device.download(row['base'], row['bytes'] + 512)
            canary = raw[:256] == row['initial'][:256] and raw[-256:] == row['initial'][-256:]
            if not row['output']:
                checks.append({'name': name, 'input_unchanged': raw == row['initial'], 'canary_pass': canary,
                               'pass': raw == row['initial'] and canary})
                continue
            actual = unpack(raw[256:-256], row['dtype'], row['shape'])
            expected = self.answer[name]
            if row['dtype'] == '*bf16':
                expected = oracles.bf16_round(expected)
                rtol, atol = .008, 1.e-4
            else:
                rtol, atol = 1.e-4, 2.e-5
            exact = row['dtype'] == '*fp8e4nv'
            okay = np.array_equal(actual, expected) if exact else np.allclose(actual, expected, rtol=rtol, atol=atol)
            finite = np.isfinite(actual) & np.isfinite(expected)
            error = float(np.max(np.abs(actual[finite].astype(np.float64) - expected[finite]))) if finite.any() else 0.
            checks.append({'name': name, 'pass': bool(okay and canary), 'canary_pass': canary,
                           'complete_elements': actual.size, 'exact_required': exact,
                           'rtol': 0. if exact else rtol, 'atol': 0. if exact else atol,
                           'max_abs_error': error, 'actual_SHA256': c.sha(actual.tobytes()),
                           'reference_SHA256': c.sha(np.asarray(expected).tobytes())})
        return {'pass': all(row['pass'] for row in checks), 'outputs_and_inputs': checks}


def execute(protocol, payload, call_id):
    root = Path('/tmp') / ('operator-fixtures-' + call_id)
    c.extract(payload, root)
    result = {'status': 'FAILED', 'call_id': call_id, 'GPU_allocated': True, 'formal_timing': False,
              'scope': 'semantic fixtures only', 'checks': [], 'protocol_SHA256': c.sha(c.encode(protocol)),
              'payload_SHA256': c.sha(payload)}
    device = None
    try:
        device = Device()
        result['environment'] = device.environment
        metadata = c.read(root / 'result.json')['records']
        cases = protocol['cases']
        bytag = {row['tag']: row for row in metadata}
        for case in cases:
            selected = [row for row in metadata if row['case'] == case['id']]
            kernels = {}
            for row in selected:
                assert row['global_scratch_size'] == 0
                row = {**row, 'num_warps': case['W']}
                kernels[row['tag']] = Loaded(device, root, row)
            for pattern in protocol['correctness']['patterns']:
                buffers = Buffers(device, case, pattern)
                for row in selected:
                    tag = row['tag']
                    kernel = kernels[tag]
                    buffers.reset()
                    owners, params = kernel.parameters(buffers.values())
                    kernel.guard()
                    kernel.launch(launch_grid(case, case['groups_for_correctness']), params, device.stream)
                    device.synchronize()
                    check = buffers.check()
                    result['checks'].append({'tag': tag, 'pattern': pattern, **check})
                device.free_buffers()
                print('GPU fixtures', case['id'], pattern, 'checks', len(result['checks']), flush=True)
        result['bindings'] = [kernel.binding() for kernel in device.kernels]
        result['driver_return_codes'] = sorted(device.driver.codes)
        result['Triton_imported'] = any(x == 'triton' or x.startswith('triton.') for x in sys.modules)
        assert not result['Triton_imported'] and result['driver_return_codes'] == [0]
        result['status'] = 'OK' if all(row['pass'] for row in result['checks']) else 'CORRECTNESS_FAILED'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    finally:
        if device:
            try:
                device.close()
            except Exception:
                # A sticky CUDA error must not hide the first failure or
                # prevent the immutable result from reaching the client.
                result['cleanup_error'] = traceback.format_exc()
    return result
