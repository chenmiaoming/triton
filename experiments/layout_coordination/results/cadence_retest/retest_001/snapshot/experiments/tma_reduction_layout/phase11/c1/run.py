"""Bounded H100 C1 runner, reusing the established native/tools images."""
import os
from pathlib import Path
import modal

from experiments.tma_reduction_layout.phase11.c1 import common as c

app = modal.App('triton-phase11-c1-microbench')
SNAPSHOT = Path('/tmp/tma-phase11-c1-snapshot')
BASE_IMAGE = 'im-qzTUZumWu2JXjWic8K82cZ'
TOOLS_IMAGE = 'im-gCqknrtnwsoW0fVOiG5XF9'


def worker_image(identity):
    environment = {'PYTHONPATH': '/opt/triton-src',
                   'TRITON_CACHE_DIR': '/cache/phase11-upstream20d9-kernel-cache'}
    if identity == BASE_IMAGE:
        environment['PATH'] = '/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'
    return (modal.Image.from_id(identity)
        .env(environment)
        .add_local_dir(SNAPSHOT / 'experiments', '/opt/triton-src/experiments', copy=True,
                       ignore=['**/__pycache__/**', '**/*.pyc']))


image = worker_image(BASE_IMAGE)
tools_image = worker_image(TOOLS_IMAGE).pip_install('numpy==2.2.6')
cache = modal.Volume.from_name('triton-build-cache')
volume = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image, gpu='H100!:1', timeout=1800, volumes={'/cache': cache, '/evidence': volume},
              single_use_containers=True, include_source=False)
def collect(protocol):
    import subprocess
    import traceback
    import torch
    import triton
    import triton._C.libtriton as native
    from triton.backends.nvidia.compiler import get_ptxas
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
    from experiments.tma_reduction_layout.phase11.c1 import kernels, data
    from experiments.tma_reduction_layout.phase4.exact_cuda import ExactCUDA
    c.verify_sources(protocol)
    root = Path('/evidence/stage_c1/artifacts') / modal.current_function_call_id()
    root.mkdir(parents=True, exist_ok=False)
    result = {'status': 'FAILED', 'call_id': modal.current_function_call_id(), 'records': [],
              'protocol_SHA256': c.sha(c.encode(protocol)), 'GPU_timing': False}

    def command(*args):
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)
    try:
        torch.cuda.init()
        assert 'H100' in torch.cuda.get_device_name(0) and torch.cuda.get_device_capability(0) == (9, 0)
        assert c.sha(Path(native.__file__).read_bytes()) == protocol['baseline']['native_SHA256']
        for name, digest in protocol['baseline']['source'].items():
            assert c.sha((Path('/opt/triton-src') / name).read_bytes()) == digest, name
        cuda = ExactCUDA()
        ptxas = Path(get_ptxas(90).path)
        result['environment'] = {**cuda.environment(), 'name': torch.cuda.get_device_name(0), 'cc': [9, 0],
            'native_SHA256': c.sha(Path(native.__file__).read_bytes()), 'triton': triton.__version__,
            'torch': torch.__version__, 'image_id': os.environ.get('MODAL_IMAGE_ID'),
            'base_image': BASE_IMAGE, 'ptxas_SHA256': c.sha(ptxas.read_bytes()),
            'ptxas_version': command(str(ptxas), '--version'),
            'ccache_before': command('ccache', '--print-stats'), 'native_rebuild': False}
        triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device='cuda'))
        inputs = {p: data.input_tensor(torch, p) for p in protocol['correctness']['patterns']}
        result['input_SHA256'] = {p: c.sha(x.view(torch.uint8).cpu().numpy().tobytes()) for p, x in inputs.items()}
        for case in protocol['cases']:
            folder = root / case['tag'].replace(':', '/')
            folder.mkdir(parents=True, exist_ok=False)
            record = {**case, 'status': 'FAILED', 'correctness': []}
            try:
                m, n, w = case['M'], case['N'], case['num_warps']
                layout = gl.BlockedLayout(**case['layout'])
                shared = gl.NVMMASharedLayout(**case['shared_layout'])
                stride = m * n if case['family'] == 'ML' else n
                out = torch.empty(2 * stride, dtype=torch.float32, device='cuda')
                kwargs = {'BM': m, 'BN': n, 'W': w, 'FAMILY': case['family'],
                          'register_layout': layout, 'shared_layout': shared}
                events = []
                triton.knobs.runtime.jit_post_compile_hook = None
                triton.knobs.compilation.listener = lambda **event: events.append({
                    'cache_hit': event['cache_hit'], 'compile_hash': event['metadata']['hash']})
                desc = TensorDescriptor.from_tensor(inputs['quarter'], [1, m, n], shared)
                compiled = kernels.micro.warmup(desc, out, 16, **kwargs, num_warps=w, grid=(1,))
                record['compiler_cache_events'] = events
                for ext in ('glir', 'ttgir', 'llir', 'ptx', 'cubin'):
                    value = compiled.asm[ext]
                    c.write_once(folder / ('kernel.' + ext), value if isinstance(value, bytes) else value.encode())
                for ext, flag in (('sass', '-sass'), ('resource.txt', '-res-usage')):
                    c.write_once(folder / ('kernel.' + ext), command('cuobjdump', flag, str(folder / 'kernel.cubin')).encode())
                meta = {**case, 'function_name': compiled.name, 'dynamic_smem_bytes': compiled.metadata.shared,
                    'global_scratch_size': compiled.metadata.global_scratch_size,
                    'global_scratch_align': compiled.metadata.global_scratch_align,
                    'tensordesc_meta': compiled.metadata.tensordesc_meta,
                    'cubin_SHA256': c.sha(compiled.asm['cubin'])}
                c.write_once(folder / 'metadata.json', meta)
                occupancy = cuda.occupancy(folder / 'kernel.cubin', compiled.name, compiled.metadata.shared, w,
                                          c.sha((folder / 'kernel.resource.txt').read_bytes()))
                c.write_once(folder / 'occupancy.json', occupancy)
                compiled._init_handles()
                for pattern in protocol['correctness']['patterns']:
                    arg = TensorDescriptor.from_tensor(inputs[pattern], [1, m, n], shared)
                    for k in protocol['K']:
                        out.fill_(float('nan'))
                        compiled[(2, 1, 1)](arg, out, k, *kwargs.values())
                        torch.cuda.synchronize()
                        shape = (2, m, n) if case['family'] == 'ML' else (2, n)
                        checked = data.compare(out.cpu().numpy().reshape(shape), data.reference(case, k, [0, 1], pattern))
                        entry = {'pattern': pattern, 'K': k, 'grid': 2, **checked}
                        record['correctness'].append(entry)
                        assert checked['pass'], (case['tag'], entry)
                record['status'] = 'OK'
            except Exception:
                record['error'] = traceback.format_exc()
                print(record['error'], flush=True)
                c.write_once(folder / 'error.txt', record['error'].encode())
            c.write_once(folder / 'attempt.json', record)
            result['records'].append(record)
            print(case['tag'], record['status'], flush=True)
            if record['status'] != 'OK':
                break
        result['environment']['ccache_after'] = command('ccache', '--print-stats')
        assert result['environment']['ccache_before'] == result['environment']['ccache_after']
        if len(result['records']) == 16 and all(r['status'] == 'OK' for r in result['records']):
            result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    c.write_once(root / 'result.json', result)
    cache.commit()
    volume.commit()
    return c.archive(root)


@app.function(image=image, gpu='H100!:1', timeout=1800, volumes={'/evidence': volume},
              single_use_containers=True, include_source=False)
def timing(invocation, protocol, gate, payload):
    from experiments.tma_reduction_layout.phase11.c1.timing import execute
    raw = execute(invocation, protocol, gate, payload, modal.current_function_call_id())
    c.write_once(Path('/evidence/stage_c1/timing') / raw['call_id'] / 'raw.json', raw)
    volume.commit()
    return raw


@app.function(image=tools_image, gpu='H100!:1', timeout=3600, volumes={'/evidence': volume},
              single_use_containers=True, include_source=False)
def profiles(protocol, gate, payload):
    import subprocess
    import sys
    import traceback
    from experiments.tma_reduction_layout.phase8.launch import Driver
    c.verify_sources(protocol)
    root = Path('/evidence/stage_c1/profiles') / modal.current_function_call_id()
    root.mkdir(parents=True, exist_ok=False)
    records = []
    result = {'status': 'FAILED', 'records': records, 'call_id': modal.current_function_call_id(),
              'payload_SHA256': c.sha(payload), 'protocol_SHA256': c.sha(c.encode(protocol)),
              'formal_timing_sample': False}

    def run(path, argv):
        try:
            p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
            value = {'argv': argv, 'returncode': p.returncode, 'output': p.stdout}
        except (OSError, subprocess.TimeoutExpired) as e:
            value = {'argv': argv, 'returncode': 'EXECUTION_FAILURE', 'output': str(e)}
        c.write_once(path, value)
        return value
    try:
        driver = Driver()
        result['environment'] = driver.environment()
        assert [driver.attr(75), driver.attr(76)] == [9, 0]
        archives = root / 'archives'
        c.extract(payload, archives)
        c.write_once(root / 'metadata.json', gate['metadata'])
        result['NCU_version'] = run(root / 'ncu_version.json', ['ncu', '--version'])['output']
        for tag in sorted(gate['metadata']):
            for k in protocol['profiles']['K']:
                for group in ('software', 'hardware'):
                    folder = root / tag.replace(':', '/') / f'K{k}' / group
                    folder.mkdir(parents=True, exist_ok=False)
                    flags = (['--section', 'InstructionStats'] if group == 'software' else
                             ['--metrics', ','.join(protocol['profiles']['metrics'])])
                    exe = [sys.executable, '-m', 'experiments.tma_reduction_layout.phase11.c1.profile_one',
                           '--root', str(archives), '--metadata', str(root / 'metadata.json'),
                           '--tag', tag, '--k', str(k)]
                    collected = run(folder / 'collect.json', ['ncu', '--replay-mode', 'kernel',
                        '--cache-control', 'none', '--clock-control', 'none', '--profile-from-start', 'off',
                        '--launch-count', '1', *flags, '--export', str(folder / 'report'), *exe])
                    reports = list(folder.glob('report.ncu-rep*'))
                    success = collected['returncode'] == 0 and len(reports) == 1
                    if success:
                        raw = run(folder / 'import.json', ['ncu', '--import', str(reports[0]),
                            '--page', 'raw', '--csv', '--print-units', 'base', '--print-fp'])
                        source = run(folder / 'source.json', ['ncu', '--import', str(reports[0]),
                            '--page', 'source', '--print-source', 'sass', '--csv'])
                        success = raw['returncode'] == 0 and source['returncode'] == 0
                    records.append({'tag': tag, 'K': k, 'group': group, 'pass': success})
                    print('C1 profile', tag, k, group, 'OK' if success else 'FAILED', flush=True)
                    if not success:
                        raise RuntimeError('NCU technical failure; preserve and stop: ' + str(records[-1]))
        assert len(records) == protocol['profiles']['report_limit']
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    c.write_once(root / 'result.json', result)
    volume.commit()
    return c.archive(root)


@app.local_entrypoint()
def main(action: str = 'collect', attempt: int = 1, invocation: int = 1):
    protocol = c.load_protocol()
    c.protect_prior()
    if action == 'collect':
        folder = c.OUT / 'artifacts' / f'attempt_{attempt}'
        function, args = collect, (protocol,)
    else:
        assert action in ('profiles', 'timing')
        gate = c.read(c.OUT / 'artifact_gate.json')
        assert gate['status'] == 'OK'
        payload = (c.OUT / 'frozen_payload.zip').read_bytes()
        if action == 'profiles':
            folder, function, args = c.OUT / 'profiles' / f'attempt_{attempt}', profiles, (protocol, gate, payload)
        else:
            assert c.read(c.OUT / 'counter_gate.json')['status'] == 'OK', 'NCU gate required before timing'
            folder, function, args = c.OUT / 'timing' / f'invocation_{invocation}', timing, (invocation, protocol, gate, payload)
    assert not folder.exists(), folder
    call = function.spawn(*args)
    c.write_once(folder / 'dispatch.json', {'call_id': call.object_id, 'profile': os.environ.get('MODAL_PROFILE'),
                                          'base_image': TOOLS_IMAGE if action == 'profiles' else BASE_IMAGE,
                                          'action': action})
    try:
        value = call.get()
    except Exception as e:
        c.write_once(folder / 'retrieval_failure.json', {'error': repr(e), 'call_id': call.object_id})
        raise
    if isinstance(value, bytes):
        c.write_once(folder / 'raw_export.zip', value)
        c.extract(value, folder / 'export')
        status = c.read(folder / 'export/result.json')['status']
    else:
        c.write_once(folder / 'raw.json', value)
        status = value['status']
    print('C1', action, 'returned', status, flush=True)
