"""Driver-only execution and CPU oracles; timing cannot import a compiler."""
import ctypes as C
from pathlib import Path
import random
import sys
import traceback
import numpy as np
from experiments.tma_reduction_layout.phase11.c2c import common as c
from experiments.tma_reduction_layout.phase11.c2 import data
from experiments.tma_reduction_layout.phase11.c2.driver import Driver, Loaded, Pointer, allocate, upload, Graph


def reference(meta, k, pids, pattern):
    if meta['kind'] == 'local':
        p, columns, r = meta['p'], meta['columns'], meta['R']
        linear = (np.asarray(pids) % 1024)[:, None, None, None]*128*r + np.arange(128*r).reshape(1, 128, p, columns)
        state = data.values(linear, pattern)
        offsets = np.arange(128*r).reshape(128, p, columns)
        step = data.bf16(((offsets*13 % 7)+1).astype(np.float32)*.25)
        acc = np.zeros((len(pids), 128), np.float32)
        for _ in range(k):
            state = np.where(state >= 8, data.bf16(state-16), data.bf16(state+step))
            partial = state[:, :, 0, :] if meta['variant'].startswith('fixture') else state.max(axis=2)
            acc += partial.sum(axis=2, dtype=np.float32)
        return np.concatenate((acc, state.reshape(len(pids), -1)), axis=1)
    m, n = meta['M'], meta['N']
    pids = np.asarray(pids)
    pm, pn = pids//(8192//n), pids % (8192//n)
    rows = np.arange(m)[None, :, None]
    cols = pn[:, None, None]*n + np.arange(n)[None, None, :]
    output = []
    for i in range(k):
        if meta['family'] in ('MC', 'ROW'):
            block = (pm+(i & 1)*((8192//m)//2)) % (8192//m)
        else:
            block = (pm+i) % (8192//m)
        values = data.values((block[:, None, None]*m+rows)*8192+cols, pattern)
        output.append(values.max(axis=1) if meta['family'] in ('MC', 'FULL', 'ORIGINAL_FULL', 'ONE_FULL') else values[:, 0, :])
    return np.stack(output, axis=1).reshape(len(pids), -1)


class Session:
    def __init__(self, root, metadata, maxgrid=2, maxk=256):
        self.driver = Driver()
        d = self.driver
        self.context = C.c_void_p()
        d.call('cuCtxCreate_v2', [C.POINTER(C.c_void_p), C.c_uint, C.c_int], C.byref(self.context), 0, d.device)
        self.cc = [d.attr(75), d.attr(76)]
        self.sm_count = d.attr(16)
        name = C.create_string_buffer(256)
        d.call('cuDeviceGetName', [C.c_void_p, C.c_int, C.c_int], C.cast(name, C.c_void_p), 256, d.device)
        self.device_name = name.value.decode()
        assert self.cc == [9, 0] and self.sm_count == 132 and 'H100' in self.device_name
        self.stream = C.c_void_p()
        d.call('cuStreamCreate', [C.POINTER(C.c_void_p), C.c_uint], C.byref(self.stream), 0)
        self.metadata = metadata
        self.kernels = {tag: Loaded(d, Path(root), meta) for tag, meta in metadata.items()}
        outsize = max(maxgrid*c.stride(meta, maxk)*4 for meta in metadata.values())
        scratch = max(1, max(meta['global_scratch_size']*maxgrid for meta in metadata.values()))
        self.pointers = [allocate(d, size) for size in (8192*8192*2, outsize, scratch)]
        self.allocated_bytes = 8192*8192*2+outsize+scratch
        assert self.allocated_bytes < 12*1024**3, self.allocated_bytes
        self.pattern = None
        self.set_pattern('quarter')

    def environment(self):
        return {**self.driver.environment(), 'cc': self.cc, 'SM_count': self.sm_count, 'name': self.device_name}

    def set_pattern(self, pattern):
        if self.pattern != pattern:
            upload(self.driver, self.pointers[0], data.input_bytes(pattern))
            self.pattern = pattern

    def params(self, tag, k):
        return self.kernels[tag].parameters(Pointer(self.pointers[0].value), Pointer(self.pointers[1].value),
                                             Pointer(self.pointers[2].value), k)

    def check(self, tag, k, pids):
        meta = self.metadata[tag]
        stride = c.stride(meta, k)
        actual = np.empty((len(pids), stride), np.float32)
        for i, pid in enumerate(pids):
            row = actual[i]
            self.driver.call('cuMemcpyDtoH_v2', [C.c_void_p, C.c_uint64, C.c_size_t],
                C.c_void_p(row.ctypes.data), self.pointers[1].value+pid*stride*4, row.nbytes)
        expected = reference(meta, k, pids, self.pattern)
        exact = np.array_equal(actual, expected)
        # Local FP32 summation can use a different valid tree than NumPy.
        tolerance = 2e-5*k if meta['kind'] == 'local' and self.pattern == 'random' else 0.
        okay = exact or (tolerance > 0 and np.allclose(actual, expected, rtol=2e-6, atol=tolerance))
        result = {'tag': tag, 'K': k, 'pids': pids, 'pattern': self.pattern,
                  'pass': bool(okay), 'bit_exact': bool(exact), 'absolute_tolerance': tolerance,
                  'max_abs_error': float(np.max(np.abs(actual-expected))), 'elements': int(actual.size),
                  'actual_SHA256': c.sha(actual.tobytes()), 'reference_SHA256': c.sha(expected.tobytes())}
        if not okay:
            bad = np.flatnonzero(actual.ravel() != expected.ravel())[:12]
            result.update(first_indices=bad.tolist(), actual=actual.ravel()[bad].tolist(),
                          expected=expected.ravel()[bad].tolist())
        return result

    def close(self):
        d = self.driver
        for kernel in self.kernels.values():
            d.call('cuModuleUnload', [C.c_void_p], kernel.module)
        for pointer in self.pointers:
            d.call('cuMemFree_v2', [C.c_uint64], pointer)
        d.call('cuStreamDestroy_v2', [C.c_void_p], self.stream)
        d.call('cuCtxDestroy_v2', [C.c_void_p], self.context)


def eligibility(protocol, metadata, root):
    result = {'status': 'FAILED', 'checks': [], 'resources': {}, 'formal_timing': False}
    session = None
    try:
        session = Session(root, metadata)
        result['environment'] = session.environment()
        for pattern in protocol['correctness']['patterns']:
            session.set_pattern(pattern)
            for tag, meta in metadata.items():
                ks = [1] if meta.get('resource_only') or meta.get('original_once') else [1, 16, 64]
                for k in ks:
                    owner, params, _ = session.params(tag, k)
                    kernel = session.kernels[tag]
                    kernel.guard()
                    kernel.launch(2, params, session.stream)
                    session.driver.call('cuStreamSynchronize', [C.c_void_p], session.stream)
                    result['checks'].append(session.check(tag, k, [0, 1]))
            print('C2c correctness', pattern, 'checks', len(result['checks']),
                  'failed', sum(not row['pass'] for row in result['checks']), flush=True)
        result['resources'] = {tag: kernel.binding() for tag, kernel in session.kernels.items()}
        result['driver_return_codes'] = sorted(session.driver.codes)
        result['Triton_imported'] = any(name == 'triton' or name.startswith('triton.') for name in sys.modules)
        assert not result['Triton_imported']
        result['status'] = 'OK' if all(row['pass'] for row in result['checks']) else 'CORRECTNESS_FAILED'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    finally:
        if session:
            session.close()
    return result


def cells_for(protocol, gate):
    cells = []
    seen = {}
    for tag, meta in gate['metadata'].items():
        if meta.get('resource_only') or gate['resources'][tag]['attributes']['local_bytes']:
            continue
        qs = protocol['timing']['local_q' if meta['kind'] == 'local' else 'composition_q']
        variants = []
        for q in qs:
            reservation = gate['resources'][tag]['reservations'][q]
            if q != 'natural' and not reservation.get('feasible') and not reservation.get('alias_of'):
                continue
            for multiplier in protocol['timing']['grid_SM_multipliers']:
                for k in ([1] if meta.get('original_once') else protocol['fit_K']):
                    variants.append((q, multiplier*132, k))
            if q == 'natural':
                variants += [(q, multiplier*132, k) for multiplier in (32, 64, 96) for k in ([1] if meta.get('original_once') else (1, 2, 4))]
                variants += [(q, 1, k) for k in ([1] if meta.get('original_once') else (1, 16, 64))]
        if meta['kind'] == 'composition':
            full = (8192//meta['M'])*(8192//meta['N'])
            variants += [('natural', full//divisor, 1) for divisor in (1, 2, 4)]
            if gate['resources'][tag]['reservations']['4'].get('feasible'):
                variants += [('4', full//divisor, 1) for divisor in (1, 2, 4)]
        for q, grid, k in variants:
            actual_q = gate['resources'][tag]['reservations'][q].get('alias_of', q)
            key = (meta['cubin_SHA256'], gate['resources'][tag]['reservations'][actual_q]['dynamic_bytes'], grid, k)
            cell = {'tag': tag, 'q': actual_q, 'grid': grid, 'K': k,
                    'cell_id': f'{tag}|q{q}|G{grid}|K{k}'}
            if key in seen:
                seen[key]['aliases'].append(cell['cell_id'])
                continue
            cell['aliases'] = []
            seen[key] = cell
            cells.append(cell)
    return cells


def execute(invocation, protocol, gate, payload, call_id):
    root = Path('/tmp') / ('c2c-timing-'+call_id)
    c.extract(payload, root)
    result = {'status': 'FAILED', 'invocation': invocation, 'call_id': call_id,
              'formal_timing': True, 'profiler_attached': False, 'cells': [],
              'protocol_SHA256': c.sha(c.encode(protocol)), 'payload_SHA256': c.sha(payload)}
    session = None
    try:
        cells = cells_for(protocol, gate)
        # Allocation is based on actually launched cells, not a fictitious
        # Cartesian product of full grids and maximum K.
        max_output = max(row['grid']*c.stride(gate['metadata'][row['tag']], row['K']) for row in cells)
        metadata = {tag: meta for tag, meta in gate['metadata'].items() if not meta.get('resource_only')}
        session = Session(root, metadata, maxgrid=1, maxk=1)
        session.driver.call('cuMemFree_v2', [C.c_uint64], session.pointers[1])
        session.pointers[1] = allocate(session.driver, max_output*4)
        session.driver.calls.clear()  # Aggregate counts/status remain recorded.
        result['environment'] = session.environment()
        result['output_allocation_bytes'] = max_output*4
        rng = random.Random(protocol['timing']['order_seed']+invocation)
        rows = [{**cell, 'samples_us': [], 'visits': []} for cell in cells]
        result['cells'] = rows  # Preserve completed samples even on a later error.
        order = list(range(len(cells)))
        for visit in range(protocol['timing']['visits']):
            rng.shuffle(order)
            for index in order:
                cell = cells[index]
                kernel = session.kernels[cell['tag']]
                owners, params, serialized = session.params(cell['tag'], cell['K'])
                graph = Graph(kernel, cell['grid'], params, session.stream, cell['q'])
                for _ in range(protocol['timing']['warmups']):
                    graph.run()
                samples = [graph.run() for _ in range(protocol['timing']['samples_per_visit'])]
                rows[index]['samples_us'] += samples
                rows[index]['visits'].append({'visit': visit, 'samples_us': samples, 'node_types': graph.node_types})
                graph.close()
                if visit == 0:
                    pids = sorted({0, cell['grid']//2, cell['grid']-1})
                    check = session.check(cell['tag'], cell['K'], pids)
                    assert check['pass'], check
                    rows[index]['correctness'] = check
                    rows[index]['parameters'] = serialized
                session.driver.calls.clear()
            print('C2c timing', invocation, 'visit', visit+1, 'cells', len(cells), flush=True)
            checkpoint = Path('/evidence/stage_c2c/timing')/call_id/f'checkpoint_{visit+1}.json'
            c.write_once(checkpoint, {'invocation': invocation, 'completed_visits': visit+1,
                                     'protocol_SHA256': result['protocol_SHA256'], 'cells': rows})
        result['cells'] = rows
        result['bindings'] = {tag: kernel.binding() for tag, kernel in session.kernels.items()}
        result['driver_return_codes'] = sorted(session.driver.codes)
        result['driver_call_counts'] = dict(session.driver.counts)
        result['Triton_imported'] = any(name == 'triton' or name.startswith('triton.') for name in sys.modules)
        assert not result['Triton_imported'] and result['driver_return_codes'] == [0]
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    finally:
        if session:
            session.close()
    return result
