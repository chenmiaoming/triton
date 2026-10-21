"""Bounded CPU preflight, actual JIT verification and compiler-free H100 jobs."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import modal
from . import common as c
from .g4_prepare import out, snapshot, protocol_path, gate_path, payload_path

app = modal.App('triton-operator-layout-g4')
# A shared snapshot is prepared before Modal imports this module.
SNAPSHOT = Path('/tmp/operator-gpu-g4-worker-snapshot-v1')
base = (modal.Image.from_id(c.BASE_IMAGE)
        .env({'PYTHONPATH': '/opt/triton-src', 'TRITON_CACHE_DIR': '/cache/phase11-upstream20d9-kernel-cache',
              'PATH': '/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
        .add_local_dir(SNAPSHOT / 'experiments', '/opt/triton-src/experiments', copy=True,
                       ignore=['**/__pycache__/**', '**/*.pyc']))
cache = modal.Volume.from_name('triton-build-cache')
evidence = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=base, cpu=8, memory=32768, timeout=1800, include_source=False,
              volumes={'/cache': cache, '/evidence': evidence})
def cpu(protocol, old_gate):
    from experiments.tma_reduction_layout.phase11.operator_gpu.g4_compile import cpu as execute
    root = Path('/evidence/operator_gpu') / protocol['stage'] / ('cpu-' + modal.current_function_call_id())
    execute(protocol, old_gate, root); cache.commit(); evidence.commit()
    return c.archive(root)


@app.function(image=base, gpu='H100!:1', cpu=8, memory=32768, timeout=2400,
              single_use_containers=True, include_source=False, volumes={'/cache': cache, '/evidence': evidence})
def bind(protocol, compiled):
    from experiments.tma_reduction_layout.phase11.operator_gpu.g4_compile import bind as execute
    result = execute(protocol, compiled)
    result['call_id'] = modal.current_function_call_id()
    c.write_once(Path('/evidence/operator_gpu') / protocol['stage'] / ('bind-' + result['call_id']) / 'raw.json', result)
    cache.commit(); evidence.commit()
    return result


@app.function(image=base, gpu='H100!:1', cpu=8, memory=32768, timeout=3600,
              single_use_containers=True, include_source=False, volumes={'/evidence': evidence})
def measure(protocol, compiled, payload, invocation):
    from experiments.tma_reduction_layout.phase11.operator_gpu.g4_runtime import run
    call = modal.current_function_call_id()
    result = run(protocol, compiled, payload, call, invocation)
    c.write_once(Path('/evidence/operator_gpu') / protocol['stage'] / ('runtime-' + call) / 'raw.json', result)
    evidence.commit()
    return result


@app.local_entrypoint()
def main(stage: str = 'g4a', action: str = 'cpu', attempt: int = 1):
    import os
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    target = out(stage); protocol = c.read(protocol_path(stage)); c.verify_sources(protocol)
    if action == 'cpu':
        jobs = [(target / 'cpu' / f'attempt_{attempt:03}', cpu, (protocol, c.read(c.OUT / 'stage_g2/compile_gate.json')))]
    else:
        gate = c.read(gate_path(stage)); assert gate['status'] == 'OK'
        if action == 'bind': jobs = [(target / 'bind' / f'attempt_{attempt:03}', bind, (protocol, gate))]
        else:
            bindings = [c.read(path) for path in (target / 'bind').glob('attempt_*/raw.json')]
            assert any(x['status'] == 'OK' and x['protocol_SHA256'] == c.sha(c.encode(protocol)) for x in bindings)
            payload = payload_path(stage).read_bytes()
            if action == 'collect': jobs = [(target / 'correctness' / f'attempt_{attempt:03}', measure, (protocol, gate, payload, 0))]
            else:
                assert action == 'timing_all'
                assert c.read(target / 'correctness/attempt_001/raw.json')['status'] == 'OK'
                jobs = [(target / 'timing' / f'invocation_{i}', measure, (protocol, gate, payload, i))
                        for i in range(1, protocol['timing']['workers'] + 1)]
    calls = []
    for folder, fn, arguments in jobs:
        assert not folder.exists(), folder
        call = fn.spawn(*arguments)
        c.write_once(folder / 'dispatch.json', {'call_id': call.object_id, 'action': action,
                    'protocol_SHA256': c.sha(c.encode(protocol)), 'profile': 'chenmiaoming'})
        calls.append((folder, call))
    def save(item):
        folder, call = item; result = call.get()
        if isinstance(result, bytes):
            c.write_once(folder / 'raw_export.zip', result); c.extract(result, folder / 'export')
            result = c.read(folder / 'export/result.json')
        else: c.write_once(folder / 'raw.json', result)
        print(stage, action, folder.name, result['status'], flush=True)
        return result['status']
    with ThreadPoolExecutor(max_workers=len(calls)) as pool: statuses = list(pool.map(save, calls))
    assert all(x == 'OK' for x in statuses), statuses
