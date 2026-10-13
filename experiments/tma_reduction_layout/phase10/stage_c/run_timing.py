"""Frozen-binary same-worker pairs, three grids, alternating variant order."""
from pathlib import Path
import io
import os
import zipfile
import modal
from experiments.tma_reduction_layout.phase10.stage_c import common as c

app = modal.App('triton-phase10-stage-c-frozen-timing')
volume = modal.Volume.from_name('triton-phase10-evidence')
snapshot = Path(os.environ.get('TMA_STAGE_C_SNAPSHOT', '/tmp/tma-phase10-stage-c-snapshot'))
image = (modal.Image.from_id('im-qzTUZumWu2JXjWic8K82cZ').env({'PYTHONPATH': '/opt/triton-src'}).add_local_dir(
    snapshot / 'experiments', '/opt/triton-src/experiments', copy=True,
    ignore=['**/__pycache__/**', '**/*.pyc', '**/results/**']))


def order(case_ids, invocation):
    pairs = sorted(case + ':g' + str(f) for case in case_ids for f in (1, 4, 16))
    rounds = []
    for r, rotated in enumerate(c.schedule(pairs, invocation)):
        variants = ['baseline', 'patch'] if (invocation + r) % 2 == 0 else ['patch', 'baseline']
        rounds.append([(pair.rsplit(':g', 1)[0], int(pair.rsplit(':g', 1)[1]), variant)
                       for pair in rotated
                       for variant in variants])
    return rounds


def time_impl(target, invocation, protocol, metadata, payload):
    import ctypes as C
    import math
    import sys
    import traceback
    import torch
    from experiments.tma_reduction_layout.phase10.stage_c.launch import Driver, Loaded
    from experiments.tma_reduction_layout.phase10.stage_c.data import input_tensor, output_reference
    raw = {
        'target': target, 'invocation': invocation, 'call_id': modal.current_function_call_id(), 'status': 'FAILED',
        'protocol_SHA256': c.sha(c.encode(protocol)), 'payload_SHA256': c.sha(payload), 'warmups': [], 'visits': [],
        'compilation': {'triton_imported': False, 'subprocess_calls': 0, 'compiler_calls':
                        0}, 'worker_source_SHA256': protocol['worker_source_SHA256']
    }

    def audit(event, args):
        if event == 'subprocess.Popen':
            raw['compilation']['subprocess_calls'] += 1
            raise RuntimeError('No subprocess/compiler in timing worker')

    sys.addaudithook(audit)
    driver = None
    loaded = {}
    try:
        for name, digest in protocol['worker_source_SHA256'].items():
            assert c.sha((c.SOURCE / name).read_bytes()) == digest, name
        root = Path('/tmp/stage_c_frozen')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
            z.extractall(root)
        torch.cuda.init()
        spec = protocol['targets'][target]
        name, cc = torch.cuda.get_device_name(0), list(torch.cuda.get_device_capability(0))
        assert spec['name'] in name and cc == spec['cc']
        driver = Driver()
        raw['environment'] = {
            **driver.environment(), 'gpu_name': name, 'cc': cc, 'image_id': os.environ.get('MODAL_IMAGE_ID'), 'torch':
            torch.__version__, 'torch_cuda': torch.version.cuda
        }
        raw['allocations'] = {}
        inputs, outputs = {}, {}
        scratch = torch.empty(max(1, max(v['global_scratch_size'] * v['grid'] for v in metadata.values())),
                              dtype=torch.uint8, device='cuda')
        stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
        parameters = {}
        references = {}
        for tag, meta in metadata.items():
            if meta['dtype'] not in inputs:
                inputs[meta['dtype']] = input_tensor(meta['dtype'])
            x = inputs[meta['dtype']]
            key = (meta['M'], meta['N'], meta['dtype'], meta['op'], meta['mode'])
            if key not in outputs:
                outputs[key] = output_reference(x, meta)
            out, reference = outputs[key]
            references[meta['case_id']] = (out, reference)
            k = Loaded(driver, root, tag, meta)
            loaded[tag] = k
            parameters[tag] = k.parameters(x, out, scratch)
            raw['allocations'][tag] = {
                'input_pointer': x.data_ptr(), 'output_pointer': out.data_ptr(), 'scratch_pointer': scratch.data_ptr(),
                'scratch_bytes': scratch.numel(), 'output_elements': out.numel(), 'input_element_bytes':
                x.element_size()
            }
        cases = sorted(references)
        rounds = order(cases, invocation)
        for case_id, fraction, variant in rounds[0]:
            tag = case_id + ':' + variant
            k = loaded[tag]
            owner, params, values = parameters[tag]
            out, reference = references[case_id]
            grid = k.metadata['grid'] // fraction
            count = grid * c.output_per_cta(k.metadata)
            out.fill_(float('nan'))
            for _ in range(3):
                k.guard()
                k.launch(params, stream, grid)
            driver.call('cuStreamSynchronize', [C.c_void_p], stream)
            error = (out[:count].float() - reference[:count].float()).abs().max().item()
            raw['warmups'].append({
                'tag': tag, 'fraction': fraction, 'grid': grid, 'count': 3, 'checked_output_elements': count,
                'correctness_max_abs_diff': error, 'actual_parameters': values, 'binary': k.record(), 'gpu_uuid':
                raw['environment']['gpu_uuid']
            })
            assert error == 0, (tag, fraction, error)
        first, last = driver.event(), driver.event()
        for r, entries in enumerate(rounds, 1):
            for case_id, fraction, variant in entries:
                tag = case_id + ':' + variant
                k = loaded[tag]
                owner, params, values = parameters[tag]
                grid = k.metadata['grid'] // fraction
                visit = {
                    'round': r, 'tag': tag, 'fraction': fraction, 'grid': grid, 'samples_us': [], 'launch_SHA256': [],
                    'actual_parameters': values, 'binary': k.record(), 'gpu_uuid': raw['environment']['gpu_uuid']
                }
                raw['visits'].append(visit)
                for _ in range(5):
                    digest = k.guard()
                    driver.record(first, stream)
                    k.launch(params, stream, grid)
                    driver.record(last, stream)
                    driver.synchronize_event(last)
                    value = driver.elapsed_us(first, last)
                    assert math.isfinite(value) and value > 0
                    visit['samples_us'].append(value)
                    visit['launch_SHA256'].append(digest)
            print(target, invocation, 'round', r, '/10', flush=True)
        driver.call('cuEventDestroy_v2', [C.c_void_p], first)
        driver.call('cuEventDestroy_v2', [C.c_void_p], last)
        raw['compilation']['triton_imported'] = any(m == 'triton' or m.startswith('triton.') for m in sys.modules)
        assert not raw['compilation']['triton_imported']
        raw['status'] = 'OK'
    except Exception:
        raw['error'] = traceback.format_exc()
        print(raw['error'], flush=True)
    finally:
        raw['binaries'] = {tag: k.record() for tag, k in loaded.items()}
        if driver:
            for k in loaded.values():
                driver.call('cuModuleUnload', [C.c_void_p], k.module)
            raw['cuda_call_counts'] = dict(driver.counts)
            raw['cuda_return_codes'] = sorted(driver.codes)
    destination = Path('/evidence/stage_c/timing') / target / (raw['call_id'] + '.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(c.encode(raw))
    volume.commit()
    return raw


@app.function(image=image, gpu='H100!:1', timeout=3600, volumes={'/evidence': volume}, single_use_containers=True,
              include_source=False)
def timing_sm90(*args):
    return time_impl('sm90', *args)


@app.function(image=image, gpu='B200:1', timeout=3600, volumes={'/evidence': volume}, single_use_containers=True,
              include_source=False)
def timing_sm100(*args):
    return time_impl('sm100', *args)


@app.function(image=image, gpu='RTX-PRO-6000:1', timeout=3600, volumes={'/evidence': volume},
              single_use_containers=True, include_source=False)
def timing_sm120(*args):
    return time_impl('sm120', *args)


@app.local_entrypoint()
def main():
    protocol = c.protocol()
    gates = c.read(c.OUT / 'artifact_gate.json')
    for target, gate in gates.items():
        metadata = {}
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as z:
            for case_id in gate['eligible']:
                for variant in protocol['variants']:
                    tag = case_id + ':' + variant
                    root = c.OUT / 'artifacts' / target / variant / 'export' / tag.replace(':', '/')
                    metadata[tag] = c.read(root / 'metadata.json')
                    for ext in ('cubin', 'ptx'):
                        z.write(root / ('kernel.' + ext), tag.replace(':', '/') + '/kernel.' + ext)
        payload = stream.getvalue()
        root = c.OUT / 'timing' / target
        if not (root / 'payload.zip').exists():
            c.write_once(root / 'payload.zip', payload)
            c.write_once(root / 'metadata.json', metadata)
        for invocation in (1, 2, 3):
            if (root / f'raw_invocation_{invocation}.json').exists():
                assert c.read(root / f'raw_invocation_{invocation}.json')['status'] == 'OK'
                continue
            call = globals()['timing_' + target].spawn(invocation, protocol, metadata, payload)
            c.write_once(
                root / ('dispatch_' + call.object_id + '.json'),
                {'call_id': call.object_id, 'invocation': invocation, 'profile': os.environ.get('MODAL_PROFILE')})
            raw = call.get()
            c.write_once(root / ('attempt_' + raw['call_id'] + '.json'), raw)
            assert raw['status'] == 'OK', raw.get('error')
            c.write_once(root / f'raw_invocation_{invocation}.json', raw)
            print('Accepted', target, invocation, flush=True)
