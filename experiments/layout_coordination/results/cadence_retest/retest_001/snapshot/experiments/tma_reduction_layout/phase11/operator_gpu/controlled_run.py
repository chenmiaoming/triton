"""CPU gate for the controlled Gluon derivatives, before real call-site binding."""
from pathlib import Path
import modal
from . import common as c

SNAPSHOT = Path('/tmp/operator-gpu-controlled-snapshot-v1')
app = modal.App('triton-operator-controlled-cpu')
image = (modal.Image.from_id(c.BASE_IMAGE)
         .env({'PYTHONPATH': '/opt/triton-src', 'TRITON_CACHE_DIR': '/cache/phase11-upstream20d9-kernel-cache',
               'PATH': '/opt/phase9-cuda-tools/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin',
               'CUDA_VISIBLE_DEVICES': ''})
         .add_local_dir(SNAPSHOT / 'experiments', '/opt/triton-src/experiments', copy=True,
                        ignore=['**/__pycache__/**', '**/*.pyc']))
cache = modal.Volume.from_name('triton-build-cache')
evidence = modal.Volume.from_name('triton-phase11-evidence')


@app.function(image=image, cpu=8, memory=16384, timeout=2400, include_source=False,
              volumes={'/cache': cache, '/evidence': evidence})
def preflight(protocol):
    from experiments.tma_reduction_layout.phase11.operator_gpu.production_compile import run_cpu
    root = Path('/evidence/operator_gpu/controlled_cpu') / modal.current_function_call_id()
    run_cpu(protocol, root)
    cache.commit()
    evidence.commit()
    return c.archive(root)


@app.local_entrypoint()
def main(attempt: int = 1):
    import os
    assert os.environ.get('MODAL_PROFILE') == 'chenmiaoming'
    root = c.OUT / 'stage_g2/controlled_cpu' / f'attempt_{attempt:03}'
    assert not root.exists()
    protocol = c.read(c.OUT / 'stage_g2/preregistration/controlled_protocol.json')
    c.verify_sources(protocol)
    call = preflight.spawn(protocol)
    c.write_once(root / 'dispatch.json', {'call_id': call.object_id, 'profile': 'chenmiaoming'})
    payload = call.get()
    c.write_once(root / 'raw_export.zip', payload)
    c.extract(payload, root / 'export')
    result = c.read(root / 'export/result.json')
    print(result['status'], len(result['records']), 'controlled CPU compiles')
    assert result['status'] == 'OK'
