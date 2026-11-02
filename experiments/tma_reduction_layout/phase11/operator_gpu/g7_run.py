"""Actual tensor JIT binding, full correctness, then frozen U/G0 paired timing."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import modal
from . import common as c
from .g7_prepare import OUT, SNAPSHOT

app = modal.App('triton-g7-source-baseline-GPU-verification')
image = (modal.Image.from_id(c.BASE_IMAGE)
    .env({'PYTHONPATH':'/opt/triton-src', 'TRITON_CACHE_DIR':'/cache/phase11-upstream20d9-kernel-cache-g6-source-only',
          'PATH':'/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin'})
    .add_local_dir(SNAPSHOT/'experiments','/opt/triton-src/experiments',copy=True,
                   ignore=['**/__pycache__/**','**/*.pyc']))
cache = modal.Volume.from_name('triton-build-cache')
evidence = modal.Volume.from_name('triton-phase11-evidence')


def current_protocol():
    path = OUT/'preregistration/protocol_v2.json'
    return path if path.exists() else OUT/'preregistration/protocol.json'


@app.function(image=image, gpu='H100!:1', cpu=8, memory=32768, timeout=2400,
              single_use_containers=True, include_source=False, volumes={'/cache':cache,'/evidence':evidence})
def binding(p, gate, payload):
    from .g7_bind import run
    root = Path('/evidence/operator_gpu/g7/binding')/modal.current_function_call_id()
    result = run(p, gate, payload, root)
    result['call_id'] = modal.current_function_call_id()
    c.write_once(root/'returned_result.json', result); cache.commit(); evidence.commit()
    return c.archive(root)


@app.function(image=image, gpu='H100!:1', cpu=8, memory=32768, timeout=3600,
              single_use_containers=True, include_source=False, volumes={'/evidence':evidence})
def measure(p, gate, payload, invocation):
    from .g7_inference import run as inference
    from .g7_training import run as training
    call_id = modal.current_function_call_id()
    root = Path('/evidence/operator_gpu/g7/runtime')/call_id
    first = inference(p, gate, payload, call_id, invocation)
    c.write_once(root/'inference.json', first); evidence.commit()
    second = training(p, gate, payload, root/'training', invocation)
    result = dict(status='OK' if first['status'] == second['status'] == 'OK' else 'FAILED',
                  invocation=invocation, call_id=call_id, inference=first, training=second,
                  protocol_SHA256=c.sha(c.encode(p)), payload_SHA256=c.sha(payload))
    c.write_once(root/'result.json', result); evidence.commit()
    return result


@app.local_entrypoint()
def main(action: str='bind', attempt: int=1):
    import os
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    p = c.read(current_protocol()); c.verify_sources(p)
    gate = c.read(OUT/('compile_gate_v2.json' if current_protocol().stem.endswith('_v2') else 'compile_gate.json'))
    payload = (OUT/'compiled_payload.zip').read_bytes()
    assert gate['protocol_SHA256'] == c.sha(c.encode(p)) and gate['payload_SHA256'] == c.sha(payload)
    if action == 'bind':
        jobs = [(OUT/'binding'/f'attempt_{attempt:03}', binding, (p,gate,payload))]
    else:
        binding_gate = c.read(OUT/'binding_gate.json')
        assert binding_gate['status'] == 'OK' and binding_gate['protocol_SHA256'] == c.sha(c.encode(p))
        if action == 'check':
            jobs = [(OUT/'correctness'/f'attempt_{attempt:03}', measure, (p,gate,payload,0))]
        else:
            assert action == 'timing'
            assert c.read(OUT/'correctness/attempt_001/raw.json')['status'] == 'OK'
            jobs = [(OUT/'timing'/f'invocation_{i}', measure, (p,gate,payload,i)) for i in range(1,p['timing']['workers']+1)]
    calls = []
    for folder, fn, args in jobs:
        assert not folder.exists(), folder
        call = fn.spawn(*args)
        c.write_once(folder/'dispatch.json', dict(call_id=call.object_id, profile='chenmiaoming',
                     action=action, protocol_SHA256=c.sha(c.encode(p))))
        calls.append((folder, call))
    def save(job):
        folder, call = job; result = call.get()
        if isinstance(result, bytes):
            c.write_once(folder/'raw_export.zip', result); c.extract(result, folder/'export')
            result = c.read(folder/'export/returned_result.json')
        else:
            c.write_once(folder/'raw.json', result)
        print('G7', action, folder.name, result['status'], flush=True)
        return result['status']
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        states = list(pool.map(save, calls))
    assert all(s == 'OK' for s in states), states
