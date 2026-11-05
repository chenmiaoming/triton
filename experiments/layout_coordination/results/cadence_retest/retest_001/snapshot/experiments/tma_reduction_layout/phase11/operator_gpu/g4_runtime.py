"""Frozen exact-binary eligibility and paired timings; no compiler imports."""
from pathlib import Path
import random
import sys
import traceback
from . import common as c
from .driver import Device, Loaded
from .production_buffers import Buffers
from .production_data import make_data
from .production_runtime import PipelineGraph


def identity(meta):
    return (meta['cubin_SHA256'], tuple(meta['runtime_argument_names']),
            c.encode(meta.get('specialized_runtime_values', {})), meta['num_warps'], meta['dynamic_smem_bytes'])


def stages(kernels, meta, buffers, mode):
    kernel = kernels[meta['tag']]
    owner, params = kernel.parameters(buffers.arguments(meta))
    rows, owners = [(kernel, buffers.data['grid'], params)], [owner]
    if mode == 'decode':
        second = kernels[meta['stage2_tag']]
        owner, params = second.parameters(buffers.arguments(second.meta))
        rows.append((second, (buffers.data['batch'], buffers.case['H'], 1), params)); owners.append(owner)
    return rows, owners


def run(protocol, compiled, payload, call_id, invocation=0):
    import numpy as np
    c.verify_sources(protocol)
    root = Path('/tmp') / ('operator-' + protocol['stage'] + '-' + call_id)
    c.extract(payload, root)
    result = {'status': 'FAILED', 'protocol_SHA256': c.sha(c.encode(protocol)), 'payload_SHA256': c.sha(payload),
              'call_id': call_id, 'invocation': invocation, 'cells': [], 'checks': [], 'compile_calls': 0}
    device = None
    try:
        device = Device(); result['environment'] = device.environment
        kernels = {x['tag']: Loaded(device, root, x) for x in compiled['records']}
        rng = random.Random(protocol['timing']['seed'] + invocation)
        for group in protocol['groups']:
            case = group['case']; metas = [x for x in compiled['records'] if x['group'] == group['id'] and not x['stage2']]
            for work in group['workloads']:
                # Every workload receives complete independent output and canary checks.
                patterns = ['finite'] if invocation else protocol['correctness']['patterns']
                for pattern in patterns:
                    data = make_data(case, batch=work['batch'], length=work['length'], pattern=pattern)
                    buffers = Buffers(device, case, data)
                    for meta in metas:
                        buffers.reset(); chain, owners = stages(kernels, meta, buffers, 'decode' if case['family'] == 'C02' else 'merge')
                        for kernel, grid, params in chain: kernel.guard(); kernel.launch(grid, params, device.stream)
                        device.synchronize(); check = buffers.check(include_inputs=True)
                        result['checks'].append({'group': group['id'], 'tag': meta['tag'], **work, 'pattern': pattern, **check})
                        assert check['pass'], result['checks'][-1]
                    device.free_buffers()
                if not invocation: continue
                data = make_data(case, batch=work['batch'], length=work['length'])
                buffers = Buffers(device, case, data)
                modes = ['stage1', 'decode'] if case['family'] == 'C02' else ['merge']
                for mode in modes:
                    unique, seen = [], {}
                    for meta in metas:
                        key = (identity(meta), identity(kernels[meta['stage2_tag']].meta) if mode == 'decode' else None)
                        if key in seen: seen[key]['aliases'].append(meta['tag']); continue
                        cell = {'group': group['id'], 'geometry': case['id'], **work, 'mode': mode,
                                'tag': meta['tag'], 'aliases': [], 'visits': []}
                        unique.append(cell); seen[key] = cell
                    order = list(range(len(unique)))
                    for visit in range(protocol['timing']['visits']):
                        rng.shuffle(order)
                        for i in order:
                            cell = unique[i]; meta = kernels[cell['tag']].meta
                            buffers.reset(); chain, owners = stages(kernels, meta, buffers, mode)
                            graph = PipelineGraph(chain, device.stream)
                            for _ in range(protocol['timing']['warmups']): graph.run()
                            samples = [graph.run() for _ in range(protocol['timing']['samples'])]
                            assert all(np.isfinite(x) and x > 0 for x in samples)
                            cell['visits'].append({'visit': visit, 'samples_us': samples, 'node_types': graph.node_types})
                            graph.close()
                            if visit == 0:
                                check = buffers.check(stage2=mode != 'stage1', include_inputs=False)
                                assert check['pass']; cell['complete_output_check'] = check
                            device.driver.calls.clear()
                    result['cells'] += unique
                device.free_buffers()
                print('G4 worker', invocation, group['id'], work, 'cells', len(result['cells']), flush=True)
        result['bindings'] = {tag: k.binding() for tag, k in kernels.items()}
        result['driver_return_codes'] = sorted(device.driver.codes)
        result['Triton_imported'] = any(n == 'triton' or n.startswith('triton.') for n in sys.modules)
        assert not result['Triton_imported'] and result['driver_return_codes'] == [0]
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc(); print(result['error'], flush=True)
    finally:
        if device: device.close()
    return result
