"""Frozen-program wake probe with one unchanged target and disjoint primer."""
import ctypes as C
import gc
import math
from pathlib import Path
import random
import sys
import time
import traceback

from experiments.tma_reduction_layout.phase11.nonreduction_survey import common as c
from experiments.tma_reduction_layout.phase11.nonreduction_survey.fixtures import DriverFixture
from experiments.tma_reduction_layout.phase11.operator_gpu.driver import Device, Loaded, Graph
from .telemetry import NVML, Sampler, collect_env_readonly


CONDITIONS = ('fast_gap', 'idle_20ms', 'idle_then_same_prime', 'idle_then_disjoint_prime')


def run(protocol, payload, root, invocation):
    root = Path(root)
    result = dict(status='FAILED', invocation=invocation,
        protocol_SHA256=c.sha(c.encode(protocol)), payload_SHA256=c.sha(payload),
        clock_control='unlocked', profiler_attached=False, compile_calls=0,
        cells=[], checks=[], bindings={}, orders=[], telemetry=[])
    device = reader = sampler = None
    graphs, kernels, buffers, owners, params = {}, {}, {}, {}, {}
    try:
        assert c.sha(payload) == protocol['payload_SHA256']
        for name, digest in protocol['source_SHA256'].items():
            assert c.sha((c.ROOT / name).read_bytes()) == digest, name
        assert len(protocol['cases']) == 1 and len(protocol['programs']) == 2
        assert tuple(protocol['conditions']) == CONDITIONS and protocol['variants'] == ['A', 'B']
        case = protocol['cases'][0]
        assert case['id'] == 'N02_k1024_p128_g128'
        metas = dict(zip(('target', 'primer'), protocol['programs']))
        assert metas['target']['variant'] == 'U'
        assert metas['primer']['variant'] == 'GL_packed_compute_vec1'
        assert c.encode(protocol['expected_fixture']) == c.encode(metas['target']['fixture'])
        artifact = root / 'program_files'
        c.extract(payload, artifact)
        device = Device()
        assert device.environment['SM_count'] == 132
        result['environment'] = device.environment
        reader = NVML(device.environment['gpu_uuid'])
        result['environment']['readonly'] = collect_env_readonly(reader)
        kernels = {role: Loaded(device, artifact, meta) for role, meta in metas.items()}
        assert len(device.kernels) == 2

        # These checks allocate temporary fixtures, all freed before timing.
        for role, kernel in kernels.items():
            for pattern in protocol['correctness']['patterns']:
                spec = c.adapter(case).fixture(case, pattern)
                fixture = DriverFixture(spec, device)
                assert fixture.inplace == []
                owner, vector = kernel.parameters(fixture.arguments(metas[role]))
                fixture.reset()
                kernel.guard()
                kernel.launch(fixture.grid, vector, device.stream)
                device.synchronize()
                check = fixture.check()
                result['checks'].append(dict(role=role, phase='before', pattern=pattern, **check))
                assert check['pass'], (role, pattern, check)
                device.free_buffers()
                del spec, fixture, owner, vector
                gc.collect()

        # Both finite fixtures remain allocated throughout all timing arms.
        identity = dict(case=case['id'],
            target_CUBIN_SHA256=kernels['target'].digest,
            primer_CUBIN_SHA256=kernels['primer'].digest,
            same_loaded_target_function_for_all_observations=True,
            same_target_graph_for_all_observations=True,
            independent_observation_labels=['A', 'B'],
            context_handle=device.context.value, stream_handle=device.stream.value)
        result['identity'] = identity
        result['fixture'] = {}
        for role, kernel in kernels.items():
            buffers[role] = DriverFixture(c.adapter(case).fixture(case, 'finite'), device)
            fixture = buffers[role]
            assert fixture.inplace == []
            owners[role], params[role] = kernel.parameters(fixture.arguments(metas[role]))
            result['fixture'][role] = fixture.metadata()
            assert c.encode(result['fixture'][role]) == c.encode(protocol['expected_fixture'])
            graphs[role] = Graph(kernel, fixture.grid, params[role], device.stream)
            values = fixture.arguments(metas[role])
            logical = {name: values[index] for index, name in enumerate(metas[role]['runtime_argument_names'])
                       if name in fixture.views}
            ranges = [dict(start=base, bytes=len(group['raw']) + 512)
                      for base, group in zip(fixture.bases, fixture.groups)]
            ranges.append(dict(start=fixture.scratch, bytes=256))
            identity[role] = dict(function_handle=kernel.function.value,
                module_handle=kernel.module.value,
                parameter_vector_address=C.addressof(params[role]),
                runtime_argument_values=values, grid=list(fixture.grid),
                graph_handle=graphs[role].executable.value,
                graph_node_types=graphs[role].node_types,
                logical_pointers=logical, scratch=fixture.scratch, storage_ranges=ranges)
        ranges = sorted((row['start'], row['start'] + row['bytes'])
                        for role in ('target', 'primer') for row in identity[role]['storage_ranges'])
        assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
        assert set(identity['target']['logical_pointers'].values()).isdisjoint(
            identity['primer']['logical_pointers'].values())
        assert kernels['target'].function.value != kernels['primer'].function.value
        identity.update(storage_disjoint=True, allocation_count=len(device.allocations))

        def launch(role, timed=False):
            """No guard, hash, copy or file I/O on any observation path."""
            graph, kernel = graphs[role], kernels[role]
            device.driver.call('cuGraphLaunch', [C.c_void_p, C.c_void_p],
                               graph.executable, device.stream)
            kernel.graph_launches += 1
            device.synchronize()
            if timed:
                return device.driver.elapsed_us(graph.first, graph.last)

        sampler = Sampler(reader, period_s=protocol['timing']['telemetry_period_s'])
        sampler.set_context(phase='prewarm', case=case['id'])
        sampler.start()
        until = time.monotonic() + protocol['timing']['prewarm_seconds']
        prewarm = dict(target=0, primer=0)
        while time.monotonic() < until:
            for role in ('target', 'primer'):
                launch(role)
                prewarm[role] += 1
            device.driver.calls.clear()
        identity['prewarm_graph_launches'] = prewarm
        activity = {condition: dict(observations=0, target_graph_launches=0,
            primer_graph_launches=0, prime_graph_launches=0) for condition in CONDITIONS}
        identity['observation_activity_by_condition'] = activity
        previous_completion = time.monotonic_ns()

        def observation(condition):
            nonlocal previous_completion
            sleep_start = time.monotonic_ns()
            if condition != 'fast_gap':
                time.sleep(protocol['timing']['idle_seconds'])
            sleep_end = time.monotonic_ns()
            prime_start = time.monotonic_ns()
            if condition == 'idle_then_same_prime':
                launch('target')
                activity[condition]['target_graph_launches'] += 1
                activity[condition]['prime_graph_launches'] += 1
            elif condition == 'idle_then_disjoint_prime':
                launch('primer')
                activity[condition]['primer_graph_launches'] += 1
                activity[condition]['prime_graph_launches'] += 1
            prime_end = time.monotonic_ns()
            started = time.monotonic_ns()
            elapsed = launch('target', timed=True)
            activity[condition]['target_graph_launches'] += 1
            activity[condition]['observations'] += 1
            completed = time.monotonic_ns()
            values = (elapsed, started - previous_completion, sleep_end - sleep_start,
                      prime_end - prime_start, started, completed)
            previous_completion = completed
            return values

        cells = {(condition, variant): dict(case=case['id'], condition=condition,
            variant=variant, visits=[]) for condition in CONDITIONS for variant in ('A', 'B')}
        result['cells'] = list(cells.values())
        rng = random.Random(protocol['timing']['seed'] + invocation)
        for visit in range(protocol['timing']['visits']):
            conditions = list(CONDITIONS)
            shift = (visit + invocation) % len(conditions)
            conditions = conditions[shift:] + conditions[:shift]
            if (visit // len(conditions) + invocation) % 2:
                conditions.reverse()
            for condition in conditions:
                for kernel in kernels.values():
                    kernel.guard()
                sampler.set_context(phase='warmup', condition=condition, visit=visit)
                for _ in range(protocol['timing']['warmups']):
                    observation(condition)
                sampler.set_context(phase='timing', condition=condition, visit=visit)
                start = time.monotonic_ns()
                observations, orders = dict(A=[], B=[]), []
                for _ in range(protocol['timing']['samples']):
                    order = ['A', 'B']
                    rng.shuffle(order)
                    orders.append(order)
                    for variant in order:
                        observations[variant].append(observation(condition))
                end = time.monotonic_ns()
                for variant, rows in observations.items():
                    assert all(math.isfinite(row[0]) and row[0] > 0 for row in rows)
                    columns = list(zip(*rows))
                    cells[condition, variant]['visits'].append(dict(visit=visit,
                        samples_us=list(columns[0]), host_gaps_ns=list(columns[1]),
                        sleep_duration_ns=list(columns[2]), prime_duration_ns=list(columns[3]),
                        guard_duration_ns=[0] * len(rows), launch_ns=list(columns[4]),
                        completion_ns=list(columns[5]), start_ns=start, end_ns=end,
                        node_types=graphs['target'].node_types))
                result['orders'].append(dict(visit=visit, condition=condition,
                    condition_order=conditions, round_orders=orders, start_ns=start, end_ns=end))
                sampler.set_context(phase='block_guard', condition=condition, visit=visit)
                for kernel in kernels.values():
                    kernel.guard()
                device.driver.calls.clear()
            print('Same target/disjoint primer', invocation, 'visit', visit, 'complete', flush=True)
        result['cells'] = list(cells.values())
        sampler.set_context(phase='after')
        for role, kernel in kernels.items():
            kernel.guard()
            fixture = buffers[role]
            fixture.reset()
            kernel.launch(fixture.grid, params[role], device.stream)
            device.synchronize()
            check = fixture.check()
            result['checks'].append(dict(role=role, phase='after', pattern='finite', **check))
            assert check['pass'], (role, check)
        identity['arguments_unchanged_after_timing'] = {
            role: [value.value for value in owners[role]] == buffers[role].arguments(metas[role])
            for role in kernels}
        assert all(identity['arguments_unchanged_after_timing'].values())
        result['bindings'] = {role: kernel.binding() for role, kernel in kernels.items()}
        result['driver_return_codes'] = sorted(device.driver.codes)
        result['driver_call_counts'] = dict(device.driver.counts)
        result['Triton_imported'] = any(name == 'triton' or name.startswith('triton.') for name in sys.modules)
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
        for resource in [*reversed(list(graphs.values())), device, reader]:
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
    c.write_once(root / 'result.json', result)
    return result
