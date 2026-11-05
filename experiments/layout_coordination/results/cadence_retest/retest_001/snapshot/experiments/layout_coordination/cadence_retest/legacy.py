"""Remeasure every accepted M3 timing condition with the common cadence engine.

These are explicit-layout Gluon programs. N-first is a paired reference, not a
source-aligned Triton U/G0 baseline. Original payloads and reports are read-only.
"""
import ctypes as C
from collections import defaultdict
from pathlib import Path
import traceback

import numpy as np

from experiments.tma_reduction_layout.phase11.operator_gpu.driver import Device
from experiments.tma_reduction_layout.phase11.operator_gpu import common as c
from experiments.tma_reduction_layout.phase11.d1.driver import Loaded
from experiments.tma_reduction_layout.phase11.m3.runtime import (
    Session as OriginalSession, cells_for, reference,
)


def groups(suite):
    """405 immutable physical conditions, paired without inventing U/G0."""
    cells = cells_for(suite['parent_protocol'], suite['gate'])
    if 'cells' in suite:
        assert c.encode(cells) == c.encode(suite['cells'])
    assert len(cells) == 405
    metadata = suite['gate']['metadata']
    result = defaultdict(list)
    for cell in cells:
        meta = metadata[cell['tag']]
        parts = cell['tag'].split(':')
        vector = parts[1] if parts[1].startswith('vec') else 'control'
        consumer = meta.get('output_policy', 'control')
        key = (meta['M'], meta['N'], meta['num_warps'], vector,
               consumer, cell['q'], cell['grid'])
        result[key].append(cell)
    assert len(result) == 216
    pairs = 0
    for key, entries in result.items():
        if key[3] == 'control':
            assert len(entries) == 1
        else:
            assert len(entries) == 2
            assert {row['tag'].split(':')[2] for row in entries} == {'n_first', 'within'}
            pairs += 1
    assert pairs == 189
    return sorted(result.items())


class OracleSession(OriginalSession):
    """Borrow the shared engine's context and reuse the original M3 oracle."""
    def __init__(self, device, artifact, metadata, cells):
        self.device = device
        self.driver, self.context, self.stream = device.driver, device.context, device.stream
        self.metadata = metadata
        self.kernels = {}
        for tag, meta in metadata.items():
            kernel = Loaded(self.driver, artifact, meta)
            self.kernels[tag] = kernel
            device.kernels.append(kernel)
        self.input_rows = max(meta.get('global_rows', 8192) for meta in metadata.values())
        self.input_bytes = self.input_rows*8192*2
        self.output_elements = max(row['grid']*metadata[row['tag']]['N'] for row in cells)+16
        maxgrid = max(row['grid'] for row in cells)
        scratch = max(1, max(meta['global_scratch_size']*maxgrid for meta in metadata.values()))
        sizes = (self.input_bytes, self.output_elements*4, scratch)
        assert sum(sizes) < 2*1024**3
        self.pointers = [C.c_uint64(device.allocate(size)) for size in sizes]
        self.pattern = None
        self.set_pattern('quarter')
        self.oracle_tables = {}

    def full_expected(self, meta, grid):
        # The original input has a documented finite period. Invoke the old CPU
        # oracle for a representative PID of every phase, then expand by exact
        # equality of input tiles. This avoids constructing gigabyte CPU tiles.
        assert not meta.get('profile_only') and meta.get('original_once')
        period = {'quarter': 97, 'random': 65536, 'boundary': 8}[self.pattern]
        pids = np.arange(grid, dtype=np.int64)
        pm, pn = pids//(8192//meta['N']), pids % (8192//meta['N'])
        phase = (pm*meta['M']*8192+pn*meta['N']) % period
        _, representatives, inverse = np.unique(phase, return_index=True, return_inverse=True)
        table = reference(meta, 1, representatives, self.pattern)
        return table[inverse]

    def full_check(self, tag, grid, q, params):
        kernel = self.kernels[tag]
        n = grid*self.metadata[tag]['N']
        self.driver.call('cuMemsetD32_v2', [C.c_uint64, C.c_uint, C.c_size_t],
                         self.pointers[1], 0x7fc00000, n+16)
        kernel.guard()
        kernel.launch(grid, params, self.stream, q)
        self.device.synchronize()
        return self.read_check(tag, grid, q)

    def read_check(self, tag, grid, q):
        """Inspect the current captured-graph output without another launch."""
        meta = self.metadata[tag]
        width, n = meta['N'], grid*meta['N']
        actual = np.frombuffer(self.device.download(self.pointers[1].value, n*4), dtype='<f4').reshape(grid, width)
        expected = self.full_expected(meta, grid)
        tail = self.device.download(self.pointers[1].value+n*4, 64)
        canary = tail == np.full(16, 0x7fc00000, dtype='<u4').tobytes()
        okay = np.array_equal(actual, expected) and canary
        maximum = float(np.max(np.abs(actual-expected)))
        return dict(pass_=bool(okay), **{'pass': bool(okay)}, bit_exact=actual.tobytes()==expected.tobytes(),
            complete_output=True, elements=n, grid=grid, K=1, q=q, pattern=self.pattern,
            max_abs_error=maximum if np.isfinite(maximum) else None,
            nonfinite_actual_elements=int(np.sum(~np.isfinite(actual))), tail_canary_pass=canary,
            actual_SHA256=c.sha(actual.tobytes()), expected_SHA256=c.sha(expected.tobytes()),
            oracle='Original M3 CPU reference expanded by exact documented input-period equivalence.')

    def eligibility(self, patterns):
        checks = []
        for pattern in patterns:
            self.set_pattern(pattern)
            for tag, meta in self.metadata.items():
                grid = 2*1024**2//(2*meta['M']*meta['N'])
                n = grid*(meta['M']*meta['N'] if meta.get('profile_only') else meta['N'])
                self.driver.call('cuMemsetD32_v2', [C.c_uint64, C.c_uint, C.c_size_t],
                                 self.pointers[1], 0x7fc00000, n+16)
                owners, params, _ = self.params(tag, 1)
                kernel = self.kernels[tag]
                kernel.guard()
                kernel.launch(grid, params, self.stream)
                self.device.synchronize()
                checked = self.check(tag, 1, list(range(grid)))
                checked['tail_canary_pass'] = self.device.download(self.pointers[1].value+n*4, 64) == np.full(16, 0x7fc00000, dtype='<u4').tobytes()
                checked['pass'] = checked['pass'] and checked['tail_canary_pass']
                checked['complete_output_footprint_bytes'] = 2*1024**2
                assert checked['pass'], checked
                checks.append(checked)
        self.set_pattern('quarter')
        return checks

    def readonly_input_check(self):
        from experiments.tma_reduction_layout.phase11.c2.data import values
        import hashlib
        pattern = self.pattern
        period = {'quarter': 97, 'random': 65536, 'boundary': 8}[pattern]
        raw = (values(np.arange(period), pattern).view(np.uint32)>>16).astype('<u2').tobytes()
        chunk = raw*16384
        expected, actual = hashlib.sha256(), hashlib.sha256()
        for offset in range(0, self.input_bytes, len(chunk)):
            size = min(len(chunk), self.input_bytes-offset)
            expected.update(chunk[:size])
            actual.update(self.device.download(self.pointers[0].value+offset, size))
        return dict(pass_=expected.digest()==actual.digest(), **{'pass': expected.digest()==actual.digest()},
            pattern=pattern, bytes=self.input_bytes, readonly_input_unchanged=expected.digest()==actual.digest(),
            expected_SHA256=expected.hexdigest(), actual_SHA256=actual.hexdigest())


def run(protocol, payload, root, invocation):
    from .engine import Session as Measurement
    root = Path(root)
    suite = protocol['suites']['legacy']
    assert c.sha(payload) == suite['payload_SHA256']
    artifact = root/'program_files'
    c.extract(payload, artifact)
    grouped = groups(suite)
    canonical_cells = [row for _, rows in grouped for row in rows]
    device, measurement, oracle = None, None, None
    error = None
    try:
        device = Device()
        measurement = Measurement(device, protocol, root, invocation, 'legacy')
        oracle = OracleSession(device, artifact, suite['gate']['metadata'], canonical_cells)
        for tag, kernel in oracle.kernels.items():
            expected = suite['gate']['resources'][tag]
            binding = kernel.binding()
            assert binding['ABI'] == expected['ABI']
            assert binding['attributes'] == expected['attributes'], tag
            assert binding['reservations'] == expected['reservations'], tag
        measurement.result['source_control_scope'] = 'Explicit Gluon N-first/within references; no Triton U/G0 pair is claimed.'
        measurement.result['canonical_physical_conditions'] = len(canonical_cells)
        measurement.result['execution_bindings'] = {}
        if invocation == 0:
            measurement.result['complete_eligibility'] = oracle.eligibility(
                suite['parent_protocol']['correctness']['patterns'])
        for (m,n,w,vector,consumer,q,grid), rows in grouped:
            context = dict(id=f'M3|M{m}|N{n}|W{w}|{vector}|{consumer}|q{q}|G{grid}',
                parent_stage='M3', M=m, N=n, W=w, vector=vector, consumer=consumer,
                q=q, grid=grid, K=1,
                reference_scope='N-first at the same vec/consumer/q/grid; never a source-aligned U/G0 alias.')
            items = []
            for cell in sorted(rows, key=lambda row: row['tag']):
                tag = cell['tag']
                kernel = oracle.kernels[tag]
                owners, params, serialized = oracle.params(tag, 1)
                role = 'default' if ':n_first:' in tag else 'candidate'
                item = dict(tag=cell['cell_id'], aliases=cell['aliases'], role=role,
                    kernels=[kernel], prepare=None,
                    expected_nodes=dict(kernels=1,memsets=0,copies=0),
                    launch=lambda kernel=kernel,params=params,q=q,grid=grid:kernel.launch(grid,params,device.stream,q),
                    check=lambda tag=tag,grid=grid,q=q,params=params:oracle.full_check(tag,grid,q,params),
                    check_graph=lambda tag=tag,grid=grid,q=q:oracle.read_check(tag,grid,q),
                    parameter_owners=owners, parameters=params, serialized_parameters=serialized,
                    original_tag=tag, q=q, grid=grid, K=1)
                items.append(item)
                measurement.result['execution_bindings'][cell['cell_id']] = dict(
                    original_tag=tag, aliases=cell['aliases'],
                    function_handle=kernel.function.value,
                    serialized_parameters=serialized,
                    q=q, grid=grid, K=1,
                    reservation=kernel.reservations[q],
                    tensordesc_meta=kernel.metadata['tensordesc_meta'],
                    target_CUBIN_SHA256=kernel.digest)
            measurement.measure(context, items)
        check = oracle.readonly_input_check()
        measurement.result['readonly_input_check'] = check
        assert check['pass'], check
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
    if measurement:
        bindings = {tag: kernel.binding() for tag,kernel in oracle.kernels.items()} if oracle else {}
        return measurement.finish(bindings, error=error)
    if device:
        device.close()
    result = dict(status='FAILED',suite='legacy',invocation=invocation,error=error)
    c.write_once(root/'result.json',result)
    return result
