"""Three invocation microbench timing; exact CUBIN, runtime K, no JIT."""
import ctypes as C
import math
from pathlib import Path
import random
import sys
import traceback

from experiments.tma_reduction_layout.phase11.c15 import common as c


def execute(invocation, protocol, gate, payload, call_id):
    import torch
    from experiments.tma_reduction_layout.phase11.c15.driver import Driver, Loaded
    from experiments.tma_reduction_layout.phase11.c15.data import input_tensor, reference, compare
    c.verify_sources(protocol)
    raw = {'status': 'FAILED', 'invocation': invocation, 'call_id': call_id, 'samples': [], 'checks': [],
           'protocol_SHA256': c.sha(c.encode(protocol)), 'payload_SHA256': c.sha(payload),
           'compilation': {'Triton_imported': False, 'subprocess_calls': 0, 'compile_calls': 0}}

    def audit(event, args):
        if event == 'subprocess.Popen':
            raw['compilation']['subprocess_calls'] += 1
            raise RuntimeError('No subprocess/JIT/compilation in C1.5 timing')
    sys.addaudithook(audit)
    driver, loaded = None, {}
    try:
        root = Path('/tmp/c1-timing-frozen')
        c.extract(payload, root)
        torch.cuda.init()
        assert 'H100' in torch.cuda.get_device_name(0) and torch.cuda.get_device_capability(0) == (9, 0)
        driver = Driver()
        sm_count = driver.attr(16)
        grids = [1, 64 * sm_count]
        raw['environment'] = {**driver.environment(), 'name': torch.cuda.get_device_name(0),
                              'cc': [9, 0], 'SM_count': sm_count}
        x = input_tensor(torch)
        max_output = max(grids) * max(m['M'] * m['N'] for m in gate['metadata'].values())
        out = torch.empty(max_output, device='cuda', dtype=torch.float32)
        scratch = torch.empty(max(1, max(m['global_scratch_size'] * max(grids)
                                      for m in gate['metadata'].values())), device='cuda', dtype=torch.uint8)
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        params = {}
        for tag, meta in gate['metadata'].items():
            kernel = loaded[tag] = Loaded(driver, root, meta)
            stride = meta['M'] * meta['N'] if meta['family'] == 'ML' else meta['N']
            for grid in grids:
                for k in protocol['K']:
                    key = (tag, grid, k)
                    params[key] = kernel.parameters(x, out, scratch, k)
                    owners, values, serialized = params[key]
                    out[:grid * stride].fill_(float('nan'))
                    for _ in range(protocol['timing']['warmups_per_cell']):
                        kernel.guard()
                        kernel.launch(grid, values, stream)
                    driver.call('cuStreamSynchronize', [C.c_void_p], stream)
                    pids = [0] if grid == 1 else [0, 1, grid - 1]
                    shape = (meta['M'], meta['N']) if meta['family'] == 'ML' else (meta['N'],)
                    actual = torch.stack([out[p * stride:(p + 1) * stride].reshape(shape) for p in pids]).cpu().numpy()
                    checked = compare(actual, reference(meta, k, pids))
                    raw['checks'].append({'tag': tag, 'grid': grid, 'K': k, 'pids': pids,
                                          'correctness': checked, 'parameters': serialized})
                    assert checked['pass'], (key, checked)
        cells = list(params)
        first, last = driver.event(), driver.event()
        for visit in range(4):
            order = cells[:]
            random.Random(2026101500 + invocation * 100 + visit).shuffle(order)
            for tag, grid, k in order:
                kernel = loaded[tag]
                owners, values, serialized = params[tag, grid, k]
                record = {'tag': tag, 'grid': grid, 'K': k, 'visit': visit,
                          'samples_us': [], 'CUBIN_SHA256': kernel.digest, 'parameters': serialized}
                raw['samples'].append(record)
                for _ in range(5):
                    kernel.guard()
                    driver.record(first, stream)
                    kernel.launch(grid, values, stream)
                    driver.record(last, stream)
                    driver.synchronize_event(last)
                    elapsed = driver.elapsed_us(first, last)
                    assert math.isfinite(elapsed) and elapsed > 0
                    record['samples_us'].append(elapsed)
            print('C1.5 timing', invocation, 'visit', visit + 1, 'of 4', flush=True)
        driver.call('cuEventDestroy_v2', [C.c_void_p], first)
        driver.call('cuEventDestroy_v2', [C.c_void_p], last)
        raw['environment_after'] = driver.environment()
        raw['compilation']['Triton_imported'] = any(m == 'triton' or m.startswith('triton.') for m in sys.modules)
        assert not raw['compilation']['Triton_imported']
        raw['status'] = 'OK'
    except Exception:
        raw['error'] = traceback.format_exc()
        print(raw['error'], flush=True)
    finally:
        raw['binaries'] = {tag: kernel.binding() for tag, kernel in loaded.items()}
        if driver:
            for kernel in loaded.values():
                driver.call('cuModuleUnload', [C.c_void_p], kernel.module)
            raw['CUDA_call_counts'] = dict(driver.counts)
            raw['CUDA_return_codes'] = sorted(driver.codes)
    return raw
