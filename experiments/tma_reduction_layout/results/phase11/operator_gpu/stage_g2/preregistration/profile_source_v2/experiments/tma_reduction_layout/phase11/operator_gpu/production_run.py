"""Real-JIT binding, correctness gate, then compiler-free paired H100 timing."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import modal
from . import common as c

SNAPSHOT = Path('/tmp/operator-gpu-production-snapshot-v1')
app = modal.App('triton-operator-production-layout-experiment')
image = (modal.Image.from_id(c.BASE_IMAGE)
         .env({'PYTHONPATH': '/opt/triton-src', 'TRITON_CACHE_DIR': '/cache/phase11-upstream20d9-kernel-cache',
               'PATH': '/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
         .add_local_dir(SNAPSHOT / 'experiments', '/opt/triton-src/experiments', copy=True,
                        ignore=['**/__pycache__/**', '**/*.pyc']))
cache = modal.Volume.from_name('triton-build-cache')
evidence = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image, gpu='H100!:1', cpu=8, memory=32768, timeout=2400,
              single_use_containers=True, include_source=False, volumes={'/cache': cache, '/evidence': evidence})
def bind(protocol):
    import traceback
    from experiments.tma_reduction_layout.phase11.operator_gpu.production_compile import run_gpu
    root = Path('/evidence/operator_gpu/production_bind') / modal.current_function_call_id()
    root.mkdir(parents=True, exist_ok=False)
    try:
        run_gpu(protocol, root)
    except Exception:
        error = traceback.format_exc()
        c.write_once(root / 'result.json', {'status': 'FAILED', 'error': error})
        print(error, flush=True)
    cache.commit()
    evidence.commit()
    return c.archive(root)


@app.function(image=image, gpu='H100!:1', cpu=4, memory=16384, timeout=2400,
              single_use_containers=True, include_source=False, volumes={'/evidence': evidence})
def collect(protocol, compiled, payload):
    from experiments.tma_reduction_layout.phase11.operator_gpu.production_runtime import eligibility
    c.verify_sources(protocol)
    call_id = modal.current_function_call_id()
    result = eligibility(protocol, compiled, payload, call_id)
    c.write_once(Path('/evidence/operator_gpu/production_correctness') / call_id / 'raw.json', result)
    evidence.commit()
    return result


@app.function(image=image, gpu='H100!:1', cpu=8, memory=32768, timeout=5400,
              single_use_containers=True, include_source=False, volumes={'/evidence': evidence})
def measure(protocol, compiled, payload, invocation):
    from experiments.tma_reduction_layout.phase11.operator_gpu.production_runtime import timing
    c.verify_sources(protocol)
    call_id = modal.current_function_call_id()
    result = timing(protocol, compiled, payload, call_id, invocation)
    c.write_once(Path('/evidence/operator_gpu/production_timing') / call_id / 'raw.json', result)
    evidence.commit()
    return result


def save(folder, call):
    import traceback
    try:
        result = call.get()
    except Exception:
        c.write_once(folder / 'retrieval_failure.json', {'error': traceback.format_exc(), 'call_id': call.object_id})
        raise
    if isinstance(result, bytes):
        c.write_once(folder / 'raw_export.zip', result)
        c.extract(result, folder / 'export')
        result = c.read(folder / 'export/result.json')
    else:
        c.write_once(folder / 'raw.json', result)
    print('Production return', folder.name, result['status'], flush=True)
    return result['status']


@app.local_entrypoint()
def main(action: str = 'bind', attempt: int = 1):
    import os
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    protocol = c.read(c.OUT / 'stage_g2/preregistration/protocol.json')
    c.verify_sources(protocol)
    if action == 'bind':
        jobs = [(c.OUT/'stage_g2/bind'/f'attempt_{attempt:03}', bind, (protocol,))]
    else:
        gate = c.read(c.OUT / 'stage_g2/compile_gate.json')
        assert gate['status'] == 'OK'
        payload = (c.OUT / 'stage_g2/compiled_payload.zip').read_bytes()
        if action == 'collect':
            jobs = [(c.OUT/'stage_g2/correctness'/f'attempt_{attempt:03}', collect, (protocol, gate, payload))]
        else:
            assert action == 'timing_all'
            assert c.read(c.OUT/'stage_g2/artifact_gate.json')['status'] == 'OK'
            jobs = [(c.OUT/'stage_g2/timing'/f'invocation_{i}', measure, (protocol, gate, payload, i))
                    for i in range(1, protocol['timing']['workers']+1)]
    calls = []
    for folder, function, arguments in jobs:
        assert not folder.exists(), folder
        call = function.spawn(*arguments)
        c.write_once(folder/'dispatch.json', {'call_id': call.object_id, 'profile': 'chenmiaoming',
                                             'protocol_SHA256': c.sha(c.encode(protocol)), 'action': action})
        calls.append((folder, call))
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        statuses = list(pool.map(lambda pair: save(*pair), calls))
    assert all(status == 'OK' for status in statuses), statuses
