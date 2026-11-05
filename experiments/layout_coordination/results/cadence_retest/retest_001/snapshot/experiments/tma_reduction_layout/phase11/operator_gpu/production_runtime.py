"""Compiler-free correctness and paired single-call graph timing of exported binaries."""
import ctypes as C
from pathlib import Path
import random
import sys
import traceback
from . import common as c
from .driver import Device, Loaded, Graph
from .production_data import make_data
from .production_buffers import Buffers


class PipelineGraph(Graph):
    def __init__(self, stages, stream):
        self.kernel, self.driver, self.stream = stages[0][0], stages[0][0].driver, stream
        self.stages = stages
        d = self.driver
        self.first, self.last = d.event(), d.event()
        self.graph, self.executable = C.c_void_p(), C.c_void_p()
        d.call('cuStreamBeginCapture', [C.c_void_p, C.c_int], stream, 0)
        d.call('cuEventRecordWithFlags', [C.c_void_p, C.c_void_p, C.c_uint], self.first, stream, 1)
        for kernel, grid, params in stages:
            kernel.guard()
            kernel.launch(grid, params, stream)
        d.call('cuEventRecordWithFlags', [C.c_void_p, C.c_void_p, C.c_uint], self.last, stream, 1)
        d.call('cuStreamEndCapture', [C.c_void_p, C.POINTER(C.c_void_p)], stream, C.byref(self.graph))
        d.call('cuGraphInstantiateWithFlags', [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint64], C.byref(self.executable), self.graph, 0)
        count = C.c_size_t()
        d.call('cuGraphGetNodes', [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)], self.graph, None, C.byref(count))
        nodes = (C.c_void_p * count.value)()
        d.call('cuGraphGetNodes', [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)], self.graph, nodes, C.byref(count))
        self.node_types = []
        for node in nodes:
            value = C.c_int()
            d.call('cuGraphNodeGetType', [C.c_void_p, C.POINTER(C.c_int)], node, C.byref(value))
            self.node_types.append(value.value)
        assert sorted(self.node_types) == [0]*len(stages)+[7, 7]

    def run(self):
        for kernel, _, _ in self.stages:
            kernel.guard()
        self.driver.call('cuGraphLaunch', [C.c_void_p, C.c_void_p], self.executable, self.stream)
        for kernel, _, _ in self.stages:
            kernel.graph_launches += 1
        self.driver.call('cuStreamSynchronize', [C.c_void_p], self.stream)
        return self.driver.elapsed_us(self.first, self.last)


def load(device, root, records):
    return {row['tag']: Loaded(device, root, row) for row in records}


def stages_for(kernels, meta, case, buffers, include_stage2=True, grid=None):
    first = kernels[meta['tag']]
    owners, params = first.parameters(buffers.arguments(meta))
    stages = [(first, grid or buffers.data['grid'], params)]
    keepalive = [owners]
    if include_stage2 and case['family'] == 'C02':
        second = kernels[case['id'] + ':stage2']
        values = buffers.arguments(second.meta)
        owner, params2 = second.parameters(values)
        keepalive.append(owner)
        stages.append((second, (buffers.data['batch'], case['H'], 1), params2))
    return stages, keepalive


def eligibility(protocol, compiled, payload, call_id):
    root = Path('/tmp') / ('operator-production-check-' + call_id)
    c.extract(payload, root)
    result = {'status': 'FAILED', 'formal_timing': False, 'checks': [], 'call_id': call_id,
              'protocol_SHA256': c.sha(c.encode(protocol)), 'payload_SHA256': c.sha(payload)}
    device = None
    try:
        device = Device()
        kernels = load(device, root, compiled['records'])
        result['environment'] = device.environment
        for case in protocol['geometries']:
            rows = [row for row in compiled['records'] if row['geometry'] == case['id'] and not row['stage2']]
            for pattern in protocol['correctness']['patterns']:
                data = make_data(case, batch=4, length=129, pattern=pattern, padded=case['family']=='C06')
                buffers = Buffers(device, case, data)
                for meta in rows:
                    buffers.reset()
                    stages, owners = stages_for(kernels, meta, case, buffers)
                    for kernel, grid, params in stages:
                        kernel.guard()
                        kernel.launch(grid, params, device.stream)
                    device.synchronize()
                    check = buffers.check()
                    result['checks'].append({'tag': meta['tag'], 'pattern': pattern, **check})
                device.free_buffers()
                print('Production correctness', case['id'], pattern, len(result['checks']), flush=True)
        result['bindings'] = {tag: kernel.binding() for tag, kernel in kernels.items()}
        result['driver_return_codes'] = sorted(device.driver.codes)
        result['Triton_imported'] = any(name == 'triton' or name.startswith('triton.') for name in sys.modules)
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
                result['cleanup_error'] = traceback.format_exc()
    return result


def workloads(case):
    if case['family'] == 'C06':
        return [{'batch': batch, 'length': 8192, 'mode': 'merge'} for batch in (1, 8, 64)]
    return [{'batch': batch, 'length': length, 'mode': mode} for batch in (1, 8)
            for length in (256, 2048) for mode in ('stage1', 'decode')]


def timing(protocol, compiled, payload, call_id, invocation):
    root = Path('/tmp') / ('operator-production-timing-' + call_id)
    c.extract(payload, root)
    result = {'status': 'FAILED', 'formal_timing': True, 'profiler_attached': False,
              'call_id': call_id, 'invocation': invocation, 'cells': [],
              'protocol_SHA256': c.sha(c.encode(protocol)), 'payload_SHA256': c.sha(payload)}
    device = None
    try:
        device = Device()
        kernels = load(device, root, compiled['records'])
        result['environment'] = device.environment
        rng = random.Random(protocol['timing']['order_seed'] + invocation)
        for case in protocol['geometries']:
            rows = [row for row in compiled['records'] if row['geometry'] == case['id'] and not row['stage2']]
            for work in workloads(case):
                data = make_data(case, batch=work['batch'], length=work['length'])
                buffers = Buffers(device, case, data)
                unique, seen = [], {}
                for meta in rows:
                    key = meta['cubin_SHA256']
                    if key in seen:
                        seen[key]['aliases'].append(meta['tag'])
                        continue
                    cell = {'tag': meta['tag'], 'geometry': case['id'], **work, 'aliases': [], 'visits': [], 'samples_us': []}
                    seen[key] = cell
                    unique.append(cell)
                order = list(range(len(unique)))
                for visit in range(protocol['timing']['visits']):
                    rng.shuffle(order)
                    for index in order:
                        cell = unique[index]
                        meta = kernels[cell['tag']].meta
                        buffers.reset()
                        stages, owners = stages_for(kernels, meta, case, buffers, work['mode'] != 'stage1')
                        graph = PipelineGraph(stages, device.stream)
                        for _ in range(protocol['timing']['warmups']):
                            graph.run()
                        samples = [graph.run() for _ in range(protocol['timing']['samples_per_visit'])]
                        cell['samples_us'] += samples
                        cell['visits'].append({'visit': visit, 'samples_us': samples, 'node_types': graph.node_types})
                        graph.close()
                        if visit == 0:
                            check = buffers.check(stage2=work['mode']!='stage1', include_inputs=False)
                            assert check['pass'], check
                            cell['complete_output_check'] = check
                        device.driver.calls.clear()
                result['cells'] += unique
                device.free_buffers()
                print('Timing', invocation, case['id'], work, len(unique), 'cells', flush=True)
                c.write_once(Path('/evidence/operator_gpu/timing') / call_id / f'checkpoint_{len(result["cells"])}.json',
                             {'completed_cells': len(result['cells']), 'invocation': invocation, 'cells': result['cells']})
        result['bindings'] = {tag: kernel.binding() for tag, kernel in kernels.items()}
        result['driver_return_codes'] = sorted(device.driver.codes)
        result['Triton_imported'] = any(name=='triton' or name.startswith('triton.') for name in sys.modules)
        assert not result['Triton_imported'] and result['driver_return_codes'] == [0]
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    finally:
        if device:
            try:
                device.close()
            except Exception:
                result['cleanup_error'] = traceback.format_exc()
    return result
