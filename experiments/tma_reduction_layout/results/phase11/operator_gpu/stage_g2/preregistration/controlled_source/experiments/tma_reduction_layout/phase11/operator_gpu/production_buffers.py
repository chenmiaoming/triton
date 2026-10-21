"""Named pointer/scalar ABI, complete outputs and untouched padding for source reproductions."""
import numpy as np
from experiments.tma_reduction_layout.phase11.operator_survey.oracles import bf16_round
from . import common as c
from .production_cases import configuration


def pack(array, dtype):
    if dtype == '*bf16':
        return (bf16_round(array).view(np.uint32) >> 16).astype('<u2').tobytes()
    return np.asarray(array, dtype='<i4' if dtype == '*i32' else '<f4').tobytes()


def unpack(blob, dtype, shape):
    if dtype == '*bf16':
        return (np.frombuffer(blob, '<u2').astype(np.uint32) << 16).view(np.float32).reshape(shape)
    return np.frombuffer(blob, '<i4' if dtype == '*i32' else '<f4').reshape(shape)


class Buffers:
    def __init__(self, device, case, data):
        self.device, self.case, self.data = device, case, data
        self.rows = {}
        _, types2, values2, _ = configuration(case, stage2=True) if case['family'] == 'C02' else ({}, {}, {}, {})
        self.scalars = {**values2, **data['scalars']}
        types = {**types2, **data['types']}
        for name in list(data['arrays']) + list(data['output_shapes']):
            output = name in data['output_shapes']
            dtype = types[name]
            shape = data['output_shapes'][name] if output else data['arrays'][name].shape
            size = int(np.prod(shape)) * (2 if dtype == '*bf16' else 4)
            content = bytes([0xcd]) * size if output else pack(data['arrays'][name], dtype)
            raw = bytes([0xa5])*256 + content + bytes([0x5a])*256
            base = device.allocate(len(raw))
            device.upload(base, raw)
            self.rows[name] = {'base': base, 'pointer': base+256, 'size': size,
                               'dtype': dtype, 'shape': shape, 'output': output, 'initial': raw}
        self.scratch = device.allocate(256)

    def arguments(self, meta):
        result = []
        for name in meta['runtime_argument_names']:
            bound = {'Mid_O': 'Att_Out', 'Mid_O_1': 'Att_Lse'}.get(name, name)
            result.append(self.rows[bound]['pointer'] if bound in self.rows else self.scalars[name])
        for name, value in meta.get('specialized_runtime_values', {}).items():
            if name in self.scalars:
                assert self.scalars[name] == value, (name, self.scalars[name], value)
        return result + [self.scratch, 0]

    def reset(self):
        for row in self.rows.values():
            if row['output']:
                self.device.upload(row['base'], row['initial'])

    def check(self, stage2=True, include_inputs=True):
        checks = []
        for name, row in self.rows.items():
            if not row['output'] and not include_inputs:
                continue
            if name == 'O' and not stage2:
                continue
            raw = self.device.download(row['base'], row['size']+512)
            canary = raw[:256] == row['initial'][:256] and raw[-256:] == row['initial'][-256:]
            if not row['output']:
                checks.append({'name': name, 'pass': raw == row['initial'] and canary,
                               'input_unchanged': raw == row['initial'], 'canary_pass': canary})
                continue
            actual = unpack(raw[256:-256], row['dtype'], row['shape'])
            expected = self.data['expected'][name]
            mask = self.data['masks'][name]
            width = 2 if row['dtype'] == '*bf16' else 4
            byte_mask = np.repeat(mask.reshape(-1), width)
            after = np.frombuffer(raw[256:-256], np.uint8)
            before = np.frombuffer(row['initial'][256:-256], np.uint8)
            inactive = bool(np.array_equal(after[~byte_mask], before[~byte_mask]))
            if row['dtype'] == '*bf16':
                expected = bf16_round(expected)
                rtol, atol = .008, 1.e-4
            else:
                rtol, atol = 2.e-4, 2.e-5
            okay = np.allclose(actual[mask], expected[mask], rtol=rtol, atol=atol)
            error = float(np.max(np.abs(actual[mask].astype(np.float64)-expected[mask]))) if mask.any() else 0.
            checks.append({'name': name, 'pass': bool(okay and canary and inactive),
                           'canary_pass': canary, 'inactive_output_untouched': inactive,
                           'complete_elements': actual.size, 'active_elements': int(mask.sum()),
                           'rtol': rtol, 'atol': atol, 'max_abs_error': error,
                           'actual_SHA256': c.sha(actual.tobytes()),
                           'reference_SHA256': c.sha(np.asarray(expected).tobytes())})
        return {'pass': all(row['pass'] for row in checks), 'details': checks}
