"""Bounded Modal H100 jobs, persistent compiler cache, write-once exports."""
import os
from pathlib import Path
import modal
from experiments.tma_reduction_layout.phase11.c16 import common as c

app = modal.App('triton-phase11-c16-overlap-audit')
BASE_IMAGE = 'im-qzTUZumWu2JXjWic8K82cZ'
TOOLS_IMAGE = 'im-gCqknrtnwsoW0fVOiG5XF9'


def image_for(identity):
    env = {'PYTHONPATH': '/opt/triton-src', 'TRITON_CACHE_DIR': '/cache/phase11-upstream20d9-kernel-cache'}
    if identity == BASE_IMAGE:
        env['PATH'] = '/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'
    return (modal.Image.from_id(identity).env(env)
            .add_local_dir(c.SNAPSHOT / 'experiments', '/opt/triton-src/experiments', copy=True,
                           ignore=['**/__pycache__/**', '**/*.pyc']))


image = image_for(BASE_IMAGE)
tools_image = image_for(TOOLS_IMAGE).pip_install('numpy==2.2.6')
cache = modal.Volume.from_name('triton-build-cache')
volume = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image, timeout=600, volumes={'/cache': cache, '/evidence': volume}, include_source=False)
def preflight(protocol):
    import traceback
    import triton
    import triton._C.libtriton as native
    from triton.backends.compiler import GPUTarget
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from experiments.tma_reduction_layout.phase11.c16.kernels import transfer
    c.verify_sources(protocol)
    assert c.sha(Path(native.__file__).read_bytes()) == protocol['baseline']['native_SHA256']
    root = Path('/evidence/stage_c16/cpu_preflight') / modal.current_function_call_id()
    root.mkdir(parents=True, exist_ok=False)
    result = {'status': 'FAILED', 'GPU_allocated': False, 'call_id': modal.current_function_call_id(),
              'protocol_SHA256': c.sha(c.encode(protocol)), 'programs': []}
    try:
        for case in protocol['new_cases']:
            shared = gl.NVMMASharedLayout(**case['shared_layout'])
            signature = {'desc': f'tensordesc<bf16[1,32,{case["N"]}],{shared}>', 'out': '*fp32',
                         'BM': 'constexpr', 'BN': 'constexpr', 'W': 'constexpr', 'shared_layout': 'constexpr'}
            source = gluon.GluonASTSource(transfer, signature,
                {'BM': 32, 'BN': case['N'], 'W': case['num_warps'], 'shared_layout': shared},
                {(1,): [['tt.divisibility', 16]]})
            compiled = triton.compile(source, target=GPUTarget('cuda', 90, 32), options={'num_warps': case['num_warps']})
            folder = root / case['tag'].replace(':', '/')
            for ext in ('glir', 'ttgir', 'llir', 'ptx', 'cubin'):
                value = compiled.asm[ext]
                c.write_once(folder / ('kernel.' + ext), value if isinstance(value, bytes) else value.encode())
            result['programs'].append({'tag': case['tag'], 'CUBIN_SHA256': c.sha(compiled.asm['cubin'])})
            print('CPU-only target sm90 compile', case['tag'], 'PASS', flush=True)
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    c.write_once(root / 'result.json', result)
    cache.commit()
    volume.commit()
    return c.archive(root)


@app.function(image=image, gpu='H100!:1', timeout=1200, volumes={'/cache': cache, '/evidence': volume},
              single_use_containers=True, include_source=False)
def collect(protocol, reused):
    import subprocess
    import traceback
    import torch
    import triton
    import triton._C.libtriton as native
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
    from experiments.tma_reduction_layout.phase11.c16.kernels import transfer
    from experiments.tma_reduction_layout.phase11.c16.runtime import expected
    from experiments.tma_reduction_layout.phase11.c15.data import input_tensor, compare
    c.verify_sources(protocol)
    root = Path('/evidence/stage_c16/artifacts') / modal.current_function_call_id()
    root.mkdir(parents=True, exist_ok=False)
    c.extract(reused, root)
    result = {'status': 'FAILED', 'call_id': modal.current_function_call_id(), 'new': [],
              'protocol_SHA256': c.sha(c.encode(protocol)), 'reuse_SHA256': c.sha(reused), 'GPU_timing': False}
    def command(*argv):
        return subprocess.check_output(argv, text=True, stderr=subprocess.STDOUT)
    try:
        torch.cuda.init()
        assert torch.cuda.get_device_capability(0) == (9, 0) and 'H100' in torch.cuda.get_device_name(0)
        assert c.sha(Path(native.__file__).read_bytes()) == protocol['baseline']['native_SHA256']
        for name, digest in protocol['baseline']['source'].items():
            assert c.sha((Path('/opt/triton-src') / name).read_bytes()) == digest, name
        result['environment'] = {'name': torch.cuda.get_device_name(0), 'cc': [9, 0],
            'native_SHA256': c.sha(Path(native.__file__).read_bytes()),
            'ccache_before': command('ccache', '--print-stats'), 'native_rebuild': False}
        metadata = dict(protocol['reused_metadata'])
        for tag, meta in metadata.items():
            for name, digest in protocol['reused_bindings'][tag]['files'].items():
                assert c.sha((root / tag.replace(':', '/') / name).read_bytes()) == digest
        inputs = {pattern: input_tensor(torch, pattern) for pattern in ('quarter', 'random')}
        for case in protocol['new_cases']:
            folder = root / case['tag'].replace(':', '/')
            folder.mkdir(parents=True, exist_ok=False)
            shared = gl.NVMMASharedLayout(**case['shared_layout'])
            kwargs = {'BM': 32, 'BN': case['N'], 'W': case['num_warps'], 'shared_layout': shared}
            out = torch.empty(2 * case['N'], device='cuda', dtype=torch.float32)
            events = []
            triton.knobs.runtime.jit_post_compile_hook = None
            triton.knobs.compilation.listener = lambda **event: events.append({
                'cache_hit': event['cache_hit'], 'compile_hash': event['metadata']['hash']})
            desc = TensorDescriptor.from_tensor(inputs['quarter'], [1, 32, case['N']], shared)
            compiled = transfer.warmup(desc, out, **kwargs, num_warps=case['num_warps'], grid=(1,))
            for ext in ('glir', 'ttgir', 'llir', 'ptx', 'cubin'):
                value = compiled.asm[ext]
                c.write_once(folder / ('kernel.' + ext), value if isinstance(value, bytes) else value.encode())
            for ext, flag in (('sass', '-sass'), ('resource.txt', '-res-usage')):
                c.write_once(folder / ('kernel.' + ext), command('cuobjdump', flag, str(folder / 'kernel.cubin')).encode())
            meta = {**case, 'function_name': compiled.name, 'dynamic_smem_bytes': compiled.metadata.shared,
                'global_scratch_size': compiled.metadata.global_scratch_size,
                'tensordesc_meta': compiled.metadata.tensordesc_meta, 'cubin_SHA256': c.sha(compiled.asm['cubin'])}
            c.write_once(folder / 'metadata.json', meta)
            metadata[case['tag']] = meta
            checks = []
            compiled._init_handles()
            for pattern, x in inputs.items():
                arg = TensorDescriptor.from_tensor(x, [1, 32, case['N']], shared)
                out.fill_(float('nan'))
                compiled[(2, 1, 1)](arg, out, *kwargs.values())
                torch.cuda.synchronize()
                checked = compare(out.cpu().numpy().reshape(2, case['N']), expected(meta, 0, [0, 1], pattern))
                checks.append({'pattern': pattern, **checked})
                assert checked['pass'], checked
            record = {'tag': case['tag'], 'checks': checks, 'compiler_cache_events': events}
            result['new'].append(record)
            c.write_once(folder / 'checks.json', record)
            print('C1.6 control', case['tag'], 'PASS', flush=True)
        result['metadata'] = metadata
        result['environment']['ccache_after'] = command('ccache', '--print-stats')
        assert result['environment']['ccache_before'] == result['environment']['ccache_after']
        result['status'] = 'OK'
    except Exception:
        result['error'] = traceback.format_exc()
        print(result['error'], flush=True)
    c.write_once(root / 'result.json', result)
    cache.commit()
    volume.commit()
    return c.archive(root)


@app.function(image=image, gpu='H100!:1', timeout=1200, volumes={'/evidence': volume},
              single_use_containers=True, include_source=False)
def timing(invocation, protocol, gate, payload):
    from experiments.tma_reduction_layout.phase11.c16.runtime import execute
    raw = execute(invocation, protocol, gate, payload, modal.current_function_call_id())
    c.write_once(Path('/evidence/stage_c16/timing') / raw['call_id'] / 'raw.json', raw)
    volume.commit()
    return raw


@app.function(image=tools_image, gpu='H100!:1', timeout=2400, volumes={'/evidence': volume},
              single_use_containers=True, include_source=False)
def profiles(protocol, gate, payload):
    import subprocess
    import sys
    import traceback
    c.verify_sources(protocol)
    root = Path('/evidence/stage_c16/profiles') / modal.current_function_call_id()
    root.mkdir(parents=True, exist_ok=False)
    archives = root / 'archives'
    c.extract(payload, archives)
    c.write_once(root / 'metadata.json', gate['metadata'])
    c.write_once(root / 'protocol.json', protocol)
    result = {'status': 'FAILED', 'call_id': modal.current_function_call_id(), 'records': [],
              'protocol_SHA256': c.sha(c.encode(protocol)), 'payload_SHA256': c.sha(payload),
              'formal_timing_sample': False}
    def run(path, argv):
        try:
            p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
            value = {'argv': argv, 'returncode': p.returncode, 'output': p.stdout}
        except Exception as error:
            value = {'argv': argv, 'returncode': 'EXECUTION_FAILURE', 'output': str(error)}
        c.write_once(path, value)
        return value
    try:
        run(root / 'ncu_version.json', ['ncu', '--version'])
        for tag, meta in gate['metadata'].items():
            if meta['family'] not in ('FULL', 'TMA'):
                continue
            for mode in ('warm', 'cold'):
                folder = root / tag.replace(':', '/') / mode
                folder.mkdir(parents=True, exist_ok=False)
                exe = [sys.executable, '-m', 'experiments.tma_reduction_layout.phase11.c16.profile_one',
                       '--root', str(archives), '--metadata', str(root / 'metadata.json'),
                       '--protocol', str(root / 'protocol.json'), '--tag', tag, '--cache', mode]
                collected = run(folder / 'collect.json', ['ncu', '--replay-mode', 'application',
                    '--cache-control', 'none', '--clock-control', 'none', '--profile-from-start', 'off',
                    '--launch-count', '1', '--metrics', ','.join(protocol['profiles']['metrics']),
                    '--export', str(folder / 'report'), *exe])
                assert collected['returncode'] == 0, collected['output'][-5000:]
                report = folder / 'report.ncu-rep'
                assert report.exists()
                imported = run(folder / 'import.json', ['ncu', '--import', str(report), '--page', 'raw',
                    '--csv', '--print-units', 'base', '--print-fp'])
                assert imported['returncode'] == 0, imported['output']
                result['records'].append({'tag': tag, 'cache': mode, 'pass': True})
                print('C1.6 DRAM profile', tag, mode, 'PASS', flush=True)
        assert len(result['records']) == protocol['profiles']['report_limit']
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
    if action == 'preflight':
        folder, function, args = c.OUT / 'cpu_preflight' / f'attempt_{attempt}', preflight, (protocol,)
    elif action == 'collect':
        assert c.read(c.OUT / 'cpu_preflight' / f'attempt_{attempt}' / 'export/result.json')['status'] == 'OK'
        folder, function = c.OUT / 'artifacts' / f'attempt_{attempt}', collect
        args = (protocol, (c.OUT / 'preregistration/reuse.zip').read_bytes())
    else:
        gate = c.read(c.OUT / 'artifact_gate.json')
        assert gate['status'] == 'OK'
        payload = (c.OUT / 'frozen_payload.zip').read_bytes()
        if action == 'profiles':
            folder, function, args = c.OUT / 'profiles' / f'attempt_{attempt}', profiles, (protocol, gate, payload)
        else:
            assert action == 'timing' and c.read(c.OUT / 'counter_gate.json')['status'] == 'OK'
            folder, function, args = c.OUT / 'timing' / f'invocation_{invocation}', timing, (invocation, protocol, gate, payload)
    assert not folder.exists(), folder
    call = function.spawn(*args)
    c.write_once(folder / 'dispatch.json', {'call_id': call.object_id, 'profile': os.environ.get('MODAL_PROFILE'),
                 'protocol_SHA256': c.sha(c.encode(protocol)), 'action': action})
    try:
        value = call.get()
    except Exception as error:
        c.write_once(folder / 'retrieval_failure.json', {'error': repr(error), 'call_id': call.object_id})
        raise
    if isinstance(value, bytes):
        c.write_once(folder / 'raw_export.zip', value)
        c.extract(value, folder / 'export')
        status = c.read(folder / 'export/result.json')['status']
    else:
        c.write_once(folder / 'raw.json', value)
        status = value['status']
    print('C1.6', action, status, flush=True)
