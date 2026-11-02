"""Compiler-free paired validation of source-aligned LayerNorm merge binaries."""
from pathlib import Path
import random
import sys
import traceback
import numpy as np
from . import common as c
from .driver import Device, Loaded
from .g3b_runtime import Buffers, TrainingGraph, stages_for


def output_bytes(device, buffers, names=None):
    """Hash stored bytes rather than an unpacked or tolerance-rounded value."""
    names = names or [name for name in buffers.expected]
    return {name: c.sha(device.download(buffers.rows[name]['ptr'], buffers.rows[name]['size']))
            for name in names}


def compare_outputs(case, mode, m, pattern, snapshots, *, required):
    assert set(snapshots) == {'U', 'G0'}
    assert set(snapshots['U']) == set(snapshots['G0'])
    matches = {name: snapshots['U'][name] == snapshots['G0'][name] for name in snapshots['U']}
    value = {'geometry': case['id'], 'mode': mode, 'M': m,
             'kernel_M': case['P'] if mode == 'merge' else min(case['P'], m), 'pattern': pattern,
             'required_bitwise_equal': required, 'outputs_equal': matches,
             'bitwise_equal': all(matches.values()), 'output_SHA256': snapshots}
    if required:
        assert value['bitwise_equal'], value
    return value


def fixed_partial_compare(kernels, case, buffers, m, pattern, metas):
    """One produced partial state, followed by U/G0 merges on that same state.

    Independent backward executions may sum contended partials in a different
    order.  This diagnostic does not rerun dX and does not reset the partials.
    Only the two final output buffers are reset between the two merge launches.
    """
    device = buffers.device
    before = output_bytes(device, buffers, ['DW', 'DB', 'Lock'])
    snapshots = {}; checks = []
    for meta in metas:
        for name in ['FINAL_DW', 'FINAL_DB']:
            row = buffers.rows[name]
            device.upload(row['base'], row['initial'])
        kernel = kernels[meta['tag']]
        owners, params = kernel.parameters(buffers.args(meta))
        kernel.guard(); kernel.launch(((case['N'] + 127)//128, 1, 1), params, device.stream)
        device.synchronize()
        check = buffers.check(include_inputs=True)
        assert check['pass'], (meta['tag'], 'same_partials', check)
        role = meta['tag'].rsplit(':', 1)[1]
        snapshots[role] = output_bytes(device, buffers, ['FINAL_DW', 'FINAL_DB'])
        checks.append({'tag': meta['tag'], **check})
        assert output_bytes(device, buffers, ['DW', 'DB', 'Lock']) == before
    comparison = compare_outputs(case, 'same_partials_merge', m, pattern, snapshots, required=True)
    comparison.update(partials_and_lock_unchanged=True, partial_state_SHA256=before,
                      checks=checks, dX_executions=0)
    return comparison


def run(protocol, compiled, payload, root, invocation=0):
    """Validate all outputs or measure two separately loaded U/G0 variants.

    `compiled.records` may also contain inference metadata; only C01 merge/dX
    records are loaded here.  Equal program identities are never deduplicated.
    """
    c.verify_sources(protocol)
    root = Path(root)
    folder = Path('/tmp')/('g7-training-' + root.name)
    c.extract(payload, folder)
    result = {'status': 'FAILED', 'invocation': invocation,
              'formal_timing': invocation > 0, 'profiler_attached': False,
              'protocol_SHA256': c.sha(c.encode(protocol)), 'payload_SHA256': c.sha(payload),
              'compile_calls': 0, 'deduplicated_variants': 0,
              'cells': [], 'checks': [], 'bitwise_comparisons': [],
              'same_partial_comparisons': [], 'paired_visit_orders': []}
    device = None
    try:
        device = Device(); result['environment'] = device.environment
        records = [r for r in compiled['records'] if r.get('family') == 'C01']
        assert len({r['tag'] for r in records}) == len(records)
        kernels = {r['tag']: Loaded(device, folder, r) for r in records}
        rng = random.Random(protocol['timing']['seed'] + invocation)
        for case in protocol['training_cases']:
            metas = [r for r in records if r['geometry'] == case['id'] and r['role'] == 'merge']
            assert {r['tag'] for r in metas} == {case['id']+':U', case['id']+':G0'}
            assert case['id']+':dx' in kernels
            assert kernels[case['id']+':U'].module.value != kernels[case['id']+':G0'].module.value
            conditions = [('merge', 128)] + [('pipeline', m) for m in case['pipeline_M']]
            for mode, m in conditions:
                patterns = ['finite'] if invocation else protocol['training_correctness']['patterns']
                for pattern in patterns:
                    buffers = Buffers(device, case, mode, m, pattern)
                    cells = [{"tag": meta['tag'], 'geometry': case['id'], 'mode': mode,
                              'M': m, 'kernel_M': buffers.scalars['M'], 'pattern': pattern,
                              'role': meta['tag'].rsplit(':', 1)[1],
                              'aliases': [], 'visits': []} for meta in metas]
                    snapshots = {}
                    for visit in range(protocol['timing']['visits'] if invocation else 1):
                        order = list(range(2)); rng.shuffle(order)
                        result['paired_visit_orders'].append({'geometry': case['id'], 'mode': mode,
                            'M': m, 'pattern': pattern, 'visit': visit,
                            'tags': [cells[index]['tag'] for index in order]})
                        for index in order:
                            cell = cells[index]; meta = kernels[cell['tag']].meta
                            buffers.reset()
                            stages, owners = stages_for(kernels, meta, case, buffers, m)
                            if invocation:
                                graph = TrainingGraph(stages, buffers)
                                try:
                                    for _ in range(protocol['timing']['warmups']): graph.run()
                                    samples = [graph.run() for _ in range(protocol['timing']['samples'])]
                                    assert all(np.isfinite(x) and x > 0 for x in samples)
                                    cell['visits'].append({'visit': visit, 'samples_us': samples,
                                                           'node_types': graph.node_types})
                                finally:
                                    graph.close()
                            else:
                                buffers.zero_for_pipeline()
                                for kernel, grid, params in stages:
                                    kernel.guard(); kernel.launch(grid, params, device.stream)
                                device.synchronize()
                            if visit == 0:
                                check = buffers.check(include_inputs=True)
                                assert check['pass'], (cell['tag'], mode, check)
                                cell['complete_output_check'] = check
                                result['checks'].append({'tag': cell['tag'], 'geometry': case['id'],
                                    'mode': mode, 'M': m, 'kernel_M': buffers.scalars['M'],
                                    'pattern': pattern, **check})
                                snapshots[cell['role']] = output_bytes(device, buffers)
                            device.driver.calls.clear()
                    result['bitwise_comparisons'].append(compare_outputs(
                        case, mode, m, pattern, snapshots, required=mode == 'merge'))
                    if mode == 'pipeline' and invocation == 0:
                        result['same_partial_comparisons'].append(
                            fixed_partial_compare(kernels, case, buffers, m, pattern, metas))
                    result['cells'].extend(cells)
                    device.free_buffers()
                    print('G7 training', invocation, case['id'], mode, m, pattern, 'PASS', flush=True)
                    c.write_once(root/f'checkpoint_{len(result["cells"])}.json',
                                 {'cells': result['cells'], 'checks': result['checks'],
                                  'bitwise_comparisons': result['bitwise_comparisons'],
                                  'same_partial_comparisons': result['same_partial_comparisons']})
        result['bindings'] = {tag: kernel.binding() for tag, kernel in kernels.items()}
        result['driver_return_codes'] = sorted(device.driver.codes)
        result['Triton_imported'] = any(name == 'triton' or name.startswith('triton.') for name in sys.modules)
        assert not result['Triton_imported'] and result['driver_return_codes'] == [0]
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc(); print(result['error'], flush=True)
    finally:
        if device: device.close()
    c.write_once(root/'result.json', result)
    return result
