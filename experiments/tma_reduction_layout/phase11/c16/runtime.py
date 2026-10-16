"""Driver-only timing and correctness; no Triton imports or compilation."""
import ctypes as C
import math
from pathlib import Path
import random
import sys
import traceback

from experiments.tma_reduction_layout.phase11.c16 import common as c
from experiments.tma_reduction_layout.phase11.launch import Loaded as DescriptorLoaded
from experiments.tma_reduction_layout.phase11.c1.driver import Loaded as MicroLoaded
from experiments.tma_reduction_layout.phase8.launch import Driver
from experiments.tma_reduction_layout.phase11.c15.data import input_tensor, reference as micro_reference, compare, tiles


class FlexibleGrid:
    def launch(self, grid, params, stream):
        assert 0 < grid <= (8192 // self.metadata['M']) * (8192 // self.metadata['N'])
        self.driver.call('cuLaunchKernel', [C.c_void_p] + [C.c_uint] * 7 +
                         [C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p], self.function,
                         grid, 1, 1, self.metadata['num_warps'] * 32, 1, 1,
                         self.metadata['dynamic_smem_bytes'], stream, params, None)
        self.launches += 1


class Full(FlexibleGrid, DescriptorLoaded):
    pass


class Micro(FlexibleGrid, MicroLoaded):
    pass


def load(driver, root, meta):
    return (Micro if meta['family'] in ('ML', 'MR') else Full)(driver, root, meta)


def expected(meta, k, pids, pattern='quarter'):
    if meta['family'] in ('ML', 'MR'):
        return micro_reference(meta, k, pids, pattern)
    value = tiles(meta['M'], meta['N'], pids, pattern=pattern)
    return value[:, 0, :] if meta['family'] == 'TMA' else value.max(axis=1)


def prepare(torch, driver, root, metadata, protocol):
    x = input_tensor(torch)
    out = torch.empty(8448 * 32 * 128, device='cuda', dtype=torch.float32)
    scratch_size = max(1, max(m['global_scratch_size'] * m['full_grid'] for m in metadata.values()))
    scratch = torch.empty(scratch_size, device='cuda', dtype=torch.uint8)
    eviction = torch.empty(protocol['cache_eviction_bytes'], device='cuda', dtype=torch.uint8)
    stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
    kernels, params = {}, {}
    for tag, meta in metadata.items():
        kernel = kernels[tag] = load(driver, root, meta)
        ks = protocol['K'] if meta['family'] in ('ML', 'MR') else [0]
        for k in ks:
            params[tag, k] = (kernel.parameters(x, out, scratch, k) if k else kernel.parameters(x, out, scratch))
    return x, out, scratch, eviction, stream, kernels, params


def check(torch, driver, kernel, grid, k, params, out, stream):
    meta = kernel.metadata
    stride = meta['M'] * meta['N'] if meta['family'] == 'ML' else meta['N']
    out[:grid * stride].fill_(float('nan'))
    for _ in range(3):
        kernel.guard()
        kernel.launch(grid, params, stream)
    driver.call('cuStreamSynchronize', [C.c_void_p], stream)
    pids = sorted(set([0, 1, grid // 2, grid - 1]))
    shape = (meta['M'], meta['N']) if meta['family'] == 'ML' else (meta['N'],)
    actual = torch.stack([out[p * stride:(p + 1) * stride].reshape(shape) for p in pids]).cpu().numpy()
    result = compare(actual, expected(meta, k, pids))
    assert result['pass'], (meta['tag'], grid, k, result)
    return {'tag': meta['tag'], 'grid': grid, 'K': k, 'pids': pids, **result}


def execute(invocation, protocol, gate, payload, call_id):
    import torch
    c.verify_sources(protocol)
    raw = {'status': 'FAILED', 'invocation': invocation, 'call_id': call_id,
           'protocol_SHA256': c.sha(c.encode(protocol)), 'payload_SHA256': c.sha(payload),
           'samples': [], 'checks': [], 'compilation': {'Triton_imported': False, 'subprocess_calls': 0}}
    def audit(event, args):
        if event == 'subprocess.Popen':
            raw['compilation']['subprocess_calls'] += 1
            raise RuntimeError('Subprocess/JIT forbidden in C1.6 event timing')
    sys.addaudithook(audit)
    driver, kernels = None, {}
    try:
        root = Path('/tmp/c16-frozen')
        c.extract(payload, root)
        torch.cuda.init()
        assert 'H100' in torch.cuda.get_device_name(0) and torch.cuda.get_device_capability(0) == (9, 0)
        driver = Driver()
        assert driver.attr(16) == 132, 'Frozen micro grid assumes 132 SM H100'
        raw['environment'] = {**driver.environment(), 'name': torch.cuda.get_device_name(0),
                              'SM_count': driver.attr(16), 'cc': [9, 0]}
        x, out, scratch, eviction, stream, kernels, params = prepare(torch, driver, root, gate['metadata'], protocol)
        cells = []
        for tag, kernel in kernels.items():
            meta = kernel.metadata
            if meta['family'] in ('ML', 'MR'):
                grids, ks, modes = [8448], protocol['K'], ['warm']
            else:
                grids, ks, modes = [meta['full_grid'] // d for d in (4, 2, 1)], [0], ['warm', 'cold']
            for grid in grids:
                for k in ks:
                    raw['checks'].append(check(torch, driver, kernel, grid, k, params[tag, k][1], out, stream))
                    cells.extend((tag, grid, k, mode) for mode in modes)
        assert len(cells) == 108
        first, last = driver.event(), driver.event()
        for visit in range(4):
            order = cells[:]
            random.Random(2026101600 + invocation * 100 + visit).shuffle(order)
            for tag, grid, k, mode in order:
                kernel = kernels[tag]
                owner, values, serialized = params[tag, k]
                # Re-establish expected cache state at every shuffled visit.
                if mode == 'warm':
                    for _ in range(3):
                        kernel.guard()
                        kernel.launch(grid, values, stream)
                row = {'tag': tag, 'grid': grid, 'K': k, 'cache': mode, 'visit': visit,
                       'CUBIN_SHA256': kernel.digest, 'parameters': serialized, 'samples_us': []}
                raw['samples'].append(row)
                for _ in range(5):
                    if mode == 'cold':
                        eviction.zero_()  # outside timing events, same stream
                    kernel.guard()
                    driver.record(first, stream)
                    kernel.launch(grid, values, stream)
                    driver.record(last, stream)
                    driver.synchronize_event(last)
                    value = driver.elapsed_us(first, last)
                    assert math.isfinite(value) and value > 0
                    row['samples_us'].append(value)
            print('C1.6 timing', invocation, 'visit', visit + 1, '/4', flush=True)
        for event in (first, last):
            driver.call('cuEventDestroy_v2', [C.c_void_p], event)
        raw['environment_after'] = driver.environment()
        raw['compilation']['Triton_imported'] = any(m == 'triton' or m.startswith('triton.') for m in sys.modules)
        assert not raw['compilation']['Triton_imported']
        raw['status'] = 'OK'
    except Exception:
        raw['error'] = traceback.format_exc()
        print(raw['error'], flush=True)
    finally:
        raw['binaries'] = {tag: kernel.binding() for tag, kernel in kernels.items()}
        if driver:
            for kernel in kernels.values():
                driver.call('cuModuleUnload', [C.c_void_p], kernel.module)
            raw['CUDA_call_counts'] = dict(driver.counts)
            raw['CUDA_return_codes'] = sorted(driver.codes)
    return raw
