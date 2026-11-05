"""Same loaded CUBIN, buffers, shape and graph; diagnose timing conditions."""
import ctypes as C
import gc
from pathlib import Path
import random
import sys
import time
import traceback

from experiments.tma_reduction_layout.phase11.nonreduction_survey import common as c
from experiments.tma_reduction_layout.phase11.nonreduction_survey.fixtures import DriverFixture
from experiments.tma_reduction_layout.phase11.operator_gpu.driver import Device, Loaded, Graph
from .telemetry import NVML, Sampler, collect_env_readonly


class Preload:
    """Untimed repeated launches; target graph remains unchanged in all arms."""
    def __init__(self, kernel, grid, params, stream, repeats):
        self.driver, self.stream = kernel.driver, stream
        self.graph, self.executable = C.c_void_p(), C.c_void_p()
        d = self.driver
        d.call('cuStreamBeginCapture', [C.c_void_p, C.c_int], stream, 0)
        for _ in range(repeats):
            kernel.launch(grid, params, stream)
        d.call('cuStreamEndCapture', [C.c_void_p, C.POINTER(C.c_void_p)], stream, C.byref(self.graph))
        d.call('cuGraphInstantiateWithFlags', [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint64],
               C.byref(self.executable), self.graph, 0)
        count = C.c_size_t()
        d.call('cuGraphGetNodes', [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)],
               self.graph, None, C.byref(count))
        nodes = (C.c_void_p * count.value)()
        d.call('cuGraphGetNodes', [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)],
               self.graph, nodes, C.byref(count))
        self.node_types = []
        for node in nodes:
            kind = C.c_int()
            d.call('cuGraphNodeGetType', [C.c_void_p, C.POINTER(C.c_int)], node, C.byref(kind))
            assert kind.value == 0
            self.node_types.append(kind.value)
        assert count.value == repeats
        self.repeats = repeats
        self.runs = 0

    def run(self):
        self.driver.call('cuGraphLaunch', [C.c_void_p, C.c_void_p], self.executable, self.stream)
        self.driver.call('cuStreamSynchronize', [C.c_void_p], self.stream)
        self.runs += 1

    def close(self):
        self.driver.call('cuGraphExecDestroy', [C.c_void_p], self.executable)
        self.driver.call('cuGraphDestroy', [C.c_void_p], self.graph)


def run(protocol, payload, root, invocation):
    root = Path(root)
    artifact = root/'program_files'
    c.extract(payload, artifact)
    result = dict(status='FAILED', invocation=invocation,
        protocol_SHA256=c.sha(c.encode(protocol)), payload_SHA256=c.sha(payload),
        clock_control='unlocked', profiler_attached=False, compile_calls=0,
        cells=[], checks=[], bindings={}, orders=[], telemetry=[])
    device = reader = sampler = graph = preload = None
    try:
        assert c.sha(payload) == protocol['payload_SHA256']
        for name, digest in protocol['source_SHA256'].items():
            assert c.sha((c.ROOT/name).read_bytes()) == digest, name
        case, meta = protocol['cases'][0], protocol['programs'][0]
        assert len(protocol['cases']) == len(protocol['programs']) == 1
        device = Device()
        assert device.environment['SM_count'] == 132
        result['environment'] = device.environment
        reader = NVML(device.environment['gpu_uuid'])
        result['environment']['readonly'] = collect_env_readonly(reader)
        kernel = Loaded(device, artifact, meta)
        # Exactly one module/function is loaded. Labels A/B never load a copy.
        assert len(device.kernels) == 1
        result['identity'] = dict(case=case['id'], cubin_SHA256=kernel.digest,
            same_loaded_function_for_all_observations=True,
            same_target_graph_for_all_observations=True,
            independent_observation_labels=['A','B'], allocation_count=0)
        for pattern in protocol['correctness']['patterns']:
            spec = c.adapter(case).fixture(case, pattern)
            buffers = DriverFixture(spec, device)
            assert buffers.inplace == []
            owner, params = kernel.parameters(buffers.arguments(meta))
            buffers.reset()
            kernel.guard()
            kernel.launch(buffers.grid, params, device.stream)
            device.synchronize()
            check = buffers.check()
            result['checks'].append(dict(phase='before', pattern=pattern, **check))
            assert check['pass'], check
            device.free_buffers()
            del spec, buffers, owner, params
            gc.collect()
        spec = c.adapter(case).fixture(case, 'finite')
        buffers = DriverFixture(spec, device)
        assert buffers.inplace == []
        owner, params = kernel.parameters(buffers.arguments(meta))
        result['fixture'] = buffers.metadata()
        assert c.encode(result['fixture']) == c.encode(meta['fixture'])
        result['identity'].update(grid=list(buffers.grid),
            runtime_argument_values=buffers.arguments(meta),
            function_handle=kernel.function.value,
            context_handle=device.context.value, stream_handle=device.stream.value,
            parameter_vector_address=C.addressof(params),
            allocation_count=len(device.allocations))
        graph = Graph(kernel, buffers.grid, params, device.stream)
        preload = Preload(kernel, buffers.grid, params, device.stream,
                          protocol['timing']['preload_repeats'])
        result['identity']['target_graph_handle'] = graph.executable.value
        result['identity']['target_graph_node_types'] = graph.node_types
        result['identity']['preload_graph_node_types'] = preload.node_types
        result['identity']['preload_repeats'] = preload.repeats
        sampler = Sampler(reader, period_s=protocol['timing']['telemetry_period_s'])
        sampler.set_context(phase='prewarm')
        sampler.start()
        until = time.monotonic() + protocol['timing']['prewarm_seconds']
        while time.monotonic() < until:
            preload.run()
            device.driver.calls.clear()
        previous_completion = time.monotonic_ns()

        def observation(condition):
            nonlocal previous_completion
            guard_start = time.monotonic_ns()
            if condition == 'legacy_gap':
                kernel.guard()
            guard_end = time.monotonic_ns()
            sleep_start = time.monotonic_ns()
            if condition == 'idle_20ms':
                time.sleep(protocol['timing']['idle_seconds'])
            sleep_end = time.monotonic_ns()
            preload_start = time.monotonic_ns()
            if condition == 'gpu_preload':
                preload.run()
            preload_end = time.monotonic_ns()
            launch = time.monotonic_ns()
            device.driver.call('cuGraphLaunch', [C.c_void_p, C.c_void_p], graph.executable, device.stream)
            kernel.graph_launches += 1
            device.synchronize()
            elapsed = device.driver.elapsed_us(graph.first, graph.last)
            completed = time.monotonic_ns()
            row = (elapsed, launch-previous_completion,
                   guard_end-guard_start, launch, completed,
                   sleep_end-sleep_start, preload_end-preload_start)
            previous_completion = completed
            return row

        cells = {(condition, variant): dict(case=case['id'], condition=condition,
            variant=variant, visits=[]) for condition in protocol['conditions']
            for variant in protocol['variants']}
        rng = random.Random(protocol['timing']['seed'] + invocation)
        for visit in range(protocol['timing']['visits']):
            # Rotate conditions then reverse on alternating visits/workers.
            # Every condition occurs equally often in every serial position.
            conditions = list(protocol['conditions'])
            shift = (visit + invocation) % len(conditions)
            conditions = conditions[shift:] + conditions[:shift]
            if (visit//len(conditions) + invocation) % 2:
                conditions.reverse()
            for condition in conditions:
                kernel.guard()
                sampler.set_context(phase='warmup', condition=condition,
                                    case=case['id'], visit=visit)
                for _ in range(protocol['timing']['warmups']):
                    observation(condition)
                sampler.set_context(phase='timing', condition=condition,
                                    case=case['id'], visit=visit)
                start = time.monotonic_ns()
                observations = {v: [] for v in protocol['variants']}
                orders = []
                for _ in range(protocol['timing']['samples']):
                    order = list(protocol['variants'])
                    rng.shuffle(order)
                    orders.append(order)
                    for variant in order:
                        observations[variant].append(observation(condition))
                end = time.monotonic_ns()
                for variant, rows in observations.items():
                    assert all(row[0] > 0 for row in rows)
                    columns = list(zip(*rows))
                    cells[condition, variant]['visits'].append(dict(visit=visit,
                        samples_us=list(columns[0]), host_gaps_ns=list(columns[1]),
                        guard_duration_ns=list(columns[2]), launch_ns=list(columns[3]),
                        completion_ns=list(columns[4]), start_ns=start, end_ns=end,
                        sleep_duration_ns=list(columns[5]), preload_duration_ns=list(columns[6]),
                        node_types=graph.node_types))
                result['orders'].append(dict(visit=visit, condition=condition,
                    condition_order=conditions, round_orders=orders,
                    start_ns=start, end_ns=end))
                sampler.set_context(phase='block_guard', condition=condition, visit=visit)
                kernel.guard()
                device.driver.calls.clear()
            print('Same CUBIN/shape', invocation, 'visit', visit, 'complete', flush=True)
        sampler.set_context(phase='after')
        kernel.guard()
        buffers.reset()
        kernel.launch(buffers.grid, params, device.stream)
        device.synchronize()
        check = buffers.check()
        result['checks'].append(dict(phase='after', pattern='finite', **check))
        assert check['pass'], check
        result['cells'] = list(cells.values())
        result['identity']['preload_total_graph_launches'] = preload.runs
        result['identity']['arguments_unchanged_after_timing'] = (
            [value.value for value in owner] == buffers.arguments(meta))
        assert result['identity']['arguments_unchanged_after_timing']
        result['bindings'] = {'same_program': kernel.binding()}
        result['driver_return_codes'] = sorted(device.driver.codes)
        result['driver_call_counts'] = dict(device.driver.counts)
        result['Triton_imported'] = any(x == 'triton' or x.startswith('triton.') for x in sys.modules)
        assert result['driver_return_codes'] == [0] and not result['Triton_imported']
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    finally:
        result['cleanup_errors'] = []
        if sampler:
            try:
                result['telemetry'] = sampler.stop()
            except Exception:
                result['cleanup_errors'].append(traceback.format_exc())
        for resource in (preload, graph, device, reader):
            if resource:
                try:
                    resource.close()
                except Exception:
                    result['cleanup_errors'].append(traceback.format_exc())
        if device:
            result['driver_return_codes'] = sorted(device.driver.codes)
            result['driver_call_counts'] = dict(device.driver.counts)
        if result['cleanup_errors'] or result.get('driver_return_codes') != [0]:
            result['status'] = 'FAILED'
    c.write_once(root/'result.json', result)
    return result
