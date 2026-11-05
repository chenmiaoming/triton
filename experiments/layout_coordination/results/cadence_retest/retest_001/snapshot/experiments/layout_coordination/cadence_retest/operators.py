"""Replay the complete corrected G8 production matrix through the shared engine.

Only host orchestration changes. Every target iteration retains the original
frozen kernels, runtime arguments, output semantics, grids and pipeline zeroes.
"""
from pathlib import Path
import traceback

from experiments.tma_reduction_layout.phase11.operator_gpu import common as c
from experiments.tma_reduction_layout.phase11.operator_gpu.driver import Device, Loaded
from experiments.tma_reduction_layout.phase11.operator_gpu.production_buffers import Buffers as InferenceBuffers
from experiments.tma_reduction_layout.phase11.operator_gpu.production_data import make_data
from experiments.tma_reduction_layout.phase11.operator_gpu.g4_runtime import stages as inference_stages
from experiments.tma_reduction_layout.phase11.operator_gpu.g3b_runtime import Buffers as TrainingBuffers
from experiments.tma_reduction_layout.phase11.operator_gpu.g3b_runtime import stages_for as training_stages


def role(meta):
    name = meta['variant']['name'].split(':')[-1]
    return 'U' if name == 'upstream' else 'G0' if name == 'gluon_anchor' else 'GL'


def identity(meta):
    """Executable and launch identity; input buffers remain context-specific."""
    return (meta['cubin_SHA256'], tuple(meta['runtime_argument_names']),
        c.encode(meta.get('specialized_runtime_values', {})), meta['num_warps'],
        meta['dynamic_smem_bytes'], meta.get('global_scratch_size', 0))


def representatives(metas, kernels, mode):
    """Preserve exact GL aliases while always observing U and G0 separately."""
    result, seen = [], {}
    for meta in metas:
        key = (identity(meta),
            identity(kernels[meta['stage2_tag']].meta) if mode == 'decode' else None,
            'U' if role(meta) == 'U' else 'Gluon')
        if key in seen:
            seen[key]['aliases'].append(meta['tag'])
            continue
        row = dict(meta=meta, tag=meta['tag'], aliases=[], role=role(meta))
        result.append(row)
        seen[key] = row
    return result


def launcher(chain, device, owners, training_buffers=None):
    """One semantic iteration, legal both in graph capture and direct checks."""
    def launch(_owners=owners):
        # Keep the ctypes scalar owners alive for every captured parameter vector.
        if training_buffers is not None:
            # This is a no-op for standalone merge. For full backward it emits
            # the original DW/DB/Lock zero nodes INSIDE each semantic iteration.
            training_buffers.zero_for_pipeline()
        for kernel, grid, params in chain:
            kernel.launch(grid, params, device.stream)
    return launch


def inference_check(buffers, launch, device, mode):
    def check():
        buffers.reset()
        launch()
        device.synchronize()
        return buffers.check(stage2=mode != 'stage1', include_inputs=True)
    return check


def training_check(buffers, launch, device):
    def check():
        buffers.reset()
        launch()
        device.synchronize()
        return buffers.check(include_inputs=True)
    return check


def inference_graph_check(buffers, mode):
    def check():
        # Read the result of the graph just replayed by Session. A reset or
        # another launch here would hide a faulty priming/capture sequence.
        return buffers.check(stage2=mode != 'stage1', include_inputs=True)
    return check


def training_graph_check(buffers):
    def check():
        return buffers.check(include_inputs=True)
    return check


def inference_contexts(parent):
    for group in parent['inference_groups']:
        case = group['case']
        modes = ('stage1', 'decode') if case['family'] == 'C02' else ('merge',)
        for work in group['workloads']:
            for mode in modes:
                yield group, work, mode, dict(
                    id=f"op__{group['id']}__b{work['batch']}__l{work['length']}__{mode}",
                    domain='inference', panel=group['panel'], group=group['id'],
                    geometry=case['id'], batch=work['batch'], length=work['length'],
                    mode=mode, M=None, kernel_M=None,
                    expectedLogicalAliases=[group['id']+':'+v['name'] for v in group['variants']])


def training_contexts(parent):
    for case in parent['training_cases']:
        for mode, m in [('merge', 128)] + [('pipeline', m) for m in case['pipeline_M']]:
            yield case, mode, m, dict(
                id=f"op__{case['id']}__{mode}__M{m}",
                domain='training', panel='g3b', group=case['id'], geometry=case['id'],
                batch=None, length=None, mode=mode, M=m,
                kernel_M=case['P'] if mode == 'merge' else min(case['P'], m),
                expectedLogicalAliases=[case['id']+':'+v['name'] for v in case['variants']])


def run(protocol, payload, root, invocation):
    """Run all current production conditions with same-round U/G0 controls."""
    from .engine import Session
    root = Path(root)
    device = session = None
    kernels = {}
    error = None
    bindings = {}
    try:
        for name, digest in protocol['source_SHA256'].items():
            assert c.sha((c.ROOT/name).read_bytes()) == digest, name
        suite = protocol['suites']['operators']
        parent, compiled = suite['parent_protocol'], suite['compiled']
        assert compiled['status'] == 'OK'
        assert compiled['protocol_SHA256'] == c.sha(c.encode(parent))
        assert c.sha(payload) == suite['payload_SHA256'] == compiled['payload_SHA256']
        assert len(compiled['records']) == 168
        artifact = root/'program_files'
        c.extract(payload, artifact)
        for meta in compiled['records']:
            assert c.sha((artifact/meta['artifact_directory']/'kernel.cubin').read_bytes()) == meta['cubin_SHA256']
        device = Device()
        session = Session(device, protocol, root, invocation, 'operators')
        kernels = {meta['tag']: Loaded(device, artifact, meta) for meta in compiled['records']}
        assert len(kernels) == len(compiled['records'])
        physical_count = logical_count = context_count = 0

        for group, work, mode, context in inference_contexts(parent):
            case = group['case']
            buffers = InferenceBuffers(device, case,
                make_data(case, batch=work['batch'], length=work['length'], pattern='finite'))
            metas = [r for r in compiled['records'] if r['group'] == group['id'] and not r['stage2']]
            items = []
            for row in representatives(metas, kernels, mode):
                chain, owners = inference_stages(kernels, row['meta'], buffers, mode)
                launch = launcher(chain, device, owners)
                items.append(dict(tag=row['tag'], aliases=row['aliases'], role=row['role'],
                    kernels=[kernel for kernel, _, _ in chain], launch=launch, prepare=None,
                    expected_nodes=dict(kernels=len(chain), memsets=0, copies=0),
                    check=inference_check(buffers, launch, device, mode),
                    check_graph=inference_graph_check(buffers, mode)))
            assert {tag for item in items for tag in [item['tag'], *item['aliases']]} == set(context['expectedLogicalAliases'])
            assert any(item['role'] == 'U' for item in items) and any(item['role'] == 'G0' for item in items)
            session.measure(context, items)
            physical_count += len(items)
            logical_count += sum(1+len(item['aliases']) for item in items)
            context_count += 1
            device.free_buffers()
            print('Cadence operators', invocation, context['id'], 'complete', flush=True)

        for case, mode, m, context in training_contexts(parent):
            buffers = TrainingBuffers(device, case, mode, m, pattern='finite')
            metas = [r for r in compiled['records'] if r['geometry'] == case['id'] and r['role'] == 'merge']
            items = []
            for row in representatives(metas, kernels, mode):
                chain, owners = training_stages(kernels, row['meta'], case, buffers, m)
                launch = launcher(chain, device, owners, buffers)
                items.append(dict(tag=row['tag'], aliases=row['aliases'], role=row['role'],
                    kernels=[kernel for kernel, _, _ in chain], launch=launch, prepare=None,
                    expected_nodes=dict(kernels=len(chain), memsets=3 if mode == 'pipeline' else 0, copies=0),
                    check=training_check(buffers, launch, device),
                    check_graph=training_graph_check(buffers)))
            assert {tag for item in items for tag in [item['tag'], *item['aliases']]} == set(context['expectedLogicalAliases'])
            assert any(item['role'] == 'U' for item in items) and any(item['role'] == 'G0' for item in items)
            session.measure(context, items)
            physical_count += len(items)
            logical_count += sum(1+len(item['aliases']) for item in items)
            context_count += 1
            device.free_buffers()
            print('Cadence operators', invocation, context['id'], 'complete', flush=True)
        assert (context_count, physical_count, logical_count) == (64, 439, 517)
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
    if device is not None:
        try:
            bindings = {kernel.meta['tag']: kernel.binding() for kernel in device.kernels}
        except Exception:
            error = (error or '') + traceback.format_exc()
    if session is not None:
        return session.finish(bindings, error=error)
    if device is not None:
        try:
            device.close()
        except Exception:
            error = (error or '') + traceback.format_exc()
    result = dict(status='FAILED', suite='operators', invocation=invocation,
        error=error, protocol_SHA256=c.sha(c.encode(protocol)), payload_SHA256=c.sha(payload),
        compile_calls=0, cells=[], checks=[], bindings=bindings)
    c.write_once(root/'result.json', result)
    return result
